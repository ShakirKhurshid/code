from xmlrpc.client import Boolean
import numpy as np
import torch
import os
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import matplotlib.pyplot as plt


def process_arguments():
    '''Collect the input argument's according to the syntax
	   Return a parser with the arguments
    '''
    parser = argparse.ArgumentParser(description = 'Train the model on a the dataset and save the model')

    parser.add_argument('-d',
                        '--data_directory',
		                type = str,
		                required = True,
		                default = 'data/train',
		                help = 'Input directory for training data')

    parser.add_argument('-o',
                        '--output_dir',
		                type = str,
		                dest = 'save_directory',
		                default = 'checkpoint_dir',
		                help = 'Directory where the checkpoint file is saved'
                        )


    parser.add_argument('-e',
                        '--epochs',
                   		dest = 'epochs', 
                   		type = int, 
                   		default = 5,
                   		help = 'Number of Epochs for the training'
                        )
    
    
    parser.add_argument('-t0', 
						'--t-zero', 
						dest='t_zero', 
						type=int,
                    	default=5,
                    	help='The initial number of epochs for the first warm restart.')

    
    parser.add_argument('-tm', 
						'--t-mult', 
						dest='t_mult', 
						type=int,
                    	default=1,
                    	help='The multiplicative factor for the number of epochs for the warm restart')
    
    parser.add_argument('-pt', 
	                    '--pre-train', 
	                    dest='pretrain', 
	                    type=Boolean,
	                    default=True,
	                    help='Use a pretrained model'
	                   )

    return parser.parse_args()


def plot_graph(train_loss , val_loss):
    '''Plots Training vs Validaton loss 
    '''
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_loss,label="val")
    plt.plot(train_loss,label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# Get the input parameters and train the specific network
def main():
    input_arguments = process_arguments()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO]: Computation device: {device}")
    

    # Specify transforms using torchvision.transforms as transforms library
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    #load Training and validation data
    train_set = datasets.ImageFolder(input_arguments.data_directory+ '/train', transform = transformations)
    val_set = datasets.ImageFolder(input_arguments.data_directory+ '/val', transform = transformations)
    class_names  = train_set.classes


    # Creating Data_Loader with a batch size of 32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)


    # Load a Pretrained Model Resnet-50
    # Set pretrained = False if you want to train the completetly on your own dataset
    model = models.resnet50(pretrained= input_arguments.pretrain) 


    #Set True to train the whole network
    for param in model.parameters():
        param.requires_grad = True 

    # Creating final fully connected Layer that accorting to the no of classes we require
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512,len(class_names)),
                            nn.LogSoftmax(dim=1))
    model.to(device)


    # Loss and optimizer
    criterion = nn.NLLLoss()

    # In order to apply layer-wise learning rate i.e differnt 
    # Use of different LRs on different part of the network with
    # lower lR in the initial layers and then increasing it in latter layers
    # Earlier layers learn generic features so they need not to be changed that much
    # Deeper layers learn data specific patterns ,Therefore we need to modify them
    optimizer = optim.SGD([{'params': model.conv1.parameters(), 'lr':1e-4}, 
                           {'params': model.layer1.parameters(), 'lr':1e-4},
                           {'params': model.layer2.parameters(),'lr':1e-4},
                           {'params': model.layer3.parameters(),'lr':1e-3},
                           {'params': model.layer4.parameters() ,'lr':1e-3},
                           {'params': model.fc.parameters(), 'lr': 1e-2}   
                           ], 
                           lr=0.001, weight_decay=0.0005
                         )


    # Restarts the learning rate after every t-0 epoch
    # SGDR uses cosine annealing, which decreases learning rate in the form of half a cosine curve
    print('[INFO]: Initializing Cosine Annealing with Warm Restart Scheduler')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                    optimizer, 
                                                                    T_0= input_arguments.t_zero, # Number of iterations for the first restart.
                                                                    T_mult= input_arguments.t_mult, #  A factor increases Ti after the restart
                                                                    )
    print(f"[INFO]: Number of epochs for first restart: {input_arguments.t_zero}")
    print(f"[INFO]: Multiplicative factor: {input_arguments.t_mult}")

                                                    
    epochs = input_arguments.epochs
    best_acc = 0.0
    iters = len(train_loader)
    train_loss_list, val_loss_list = [], []
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0


        # Traininng the model
        print('\nTraining....')
        model.train()
        counter = 0
        for i, sample in enumerate(train_loader):
            inputs, labels = sample

            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear Optimizers
            optimizer.zero_grad()

            # Forward Pass
            logps = model.forward(inputs)

            # Loss
            loss = criterion (logps, labels)

            # Backprop (Calculate Gradients)
            loss.backward()

            # Adjust parameters based on gradients
            optimizer.step()

            # Reduce the LR with Cosine Annealing
            scheduler.step(epoch + i/iters)

            # Add the loss to the trainining set's running loss
            train_loss += loss.item() * inputs.size(0)

            # Print the progress of our training
            counter += 1
            print (counter, "/", len(train_loader))


        # Evaluating the model
        print('\nEvaluating....')     
        model.eval ()
        counter = 0

        # Tell torch not to calculate gradients
        with torch.no_grad ():
            for inputs, labels in val_loader:
                
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                output = model.forward(inputs)
                
                # Calculate Loss
                loss = criterion(output, labels)
    
                # Add loss to the validation set's running loss
                valid_loss += loss.item() * inputs.size(0)
                
                # Since our model outputs a LogSoftmax, find the real
                # percentages by reversing the log function
                output = torch.exp(output)
                
                # Get the top class of the output
                top_p, top_class = output.topk (1, dim=1)

                # See how many of the classes were correct?
                equals = top_class == labels.view (*top_class.shape)

                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
                accuracy += torch.mean (equals.type (torch.FloatTensor)).item ()

                # Print the progress of our evaluation
                counter += 1
                print (counter, "/", len(val_loader))

            # Save_the_best _accuracy_model
            if (accuracy / len (val_loader)) > best_acc:
                best_acc = accuracy / len (val_loader)

                #Create a file path using the specified save_directory
                #to save the file as checkpoint.pth under that directory
                if not os.path.exists(input_arguments.save_directory):
                    os.makedirs(input_arguments.save_directory)
                checkpoint_file_path = os.path.join(input_arguments.save_directory, 'RESNET-50'+"_"+str(input_arguments.epochs)+".pth")
                torch.save ( model.state_dict(), checkpoint_file_path)

        # Get the average loss for the entire epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(val_loader.dataset)

        # Print out the information
        print ('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format (epoch, train_loss, valid_loss))
        print ('Accuracy: ', accuracy / len(val_loader))

        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)

    print ('\nBest Accuracy', best_acc)

 
    plot_graph(train_loss_list, val_loss_list)
    pass


if __name__ == '__main__':
    main()


