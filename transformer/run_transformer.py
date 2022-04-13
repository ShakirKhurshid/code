

from transform  import four_point_transform
import numpy as np
import cv2
import imutils
import argparse

def process_arguments():
    '''Collect the input argument's according to the syntax
	   Return a parser with the arguments
    '''
    parser = argparse.ArgumentParser(description = 'Train the model on a the dataset and save the model')

    parser.add_argument('-i',
                        '--input',
		                action='store',
		                type = str,
		                required = True,
		                default = '/sample_pictures/4.jpg',
		                help = 'Input directory of image')
    
    parser.add_argument('-o',
                    	'--output',
                    	action='store',
                    	type = str,
                    	default = 'output/',
                    	help = 'Output directory')
    
    parser.add_argument('-u',
                		'--upper',
               			type = int,
               			dest='upper',
                		default = 255,
                		help = 'Thresholding upperbound')
    
    parser.add_argument('-l',
                		'--lower',
                		type = int,
                		dest='lower',
                		default = 165,
                		help = 'Thresholding lowerbound')
    
    return parser.parse_args()


def main():
    input_arguments = process_arguments()

    
    if input_arguments.lower >= input_arguments.upper:
    	print('[INPUT_ERROR!]: Upper threshold cannot be less than the lower threshold')
    	return None

    # Load input image
    image = cv2.imread(input_arguments.input)

    # Resizing the image
    image = imutils.resize(image, height = 300)

    # Graying out the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Clipped Local Adaptive Histogram Equalization (CLAHE) is applied to get better contrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    #cv2.imshow('c1',cl1)

    # Thresholding applied to get the binary image 
    # lower = 165 and upper = 255 works the best
    thresh = cv2.threshold(cl1, input_arguments.lower, input_arguments.upper,cv2.THRESH_BINARY)[1]

    # Finds contours
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]

  	# Draws contours
    for c in cnts:
        if cv2.contourArea(c)  < 5000:
            continue
        print("area",cv2.contourArea(c))

        ## Draw rotated rectangle
        print("test")
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,255,0),2)
        break  


    warped = four_point_transform(image, box)
    cv2.imwrite('output.jpg', warped)
    
    # show the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)

    while True:
        # it waits till we press a key
        print('Press Esc to exit all windows')
        key = cv2.waitKey(0)

        # if we press esc
        if key == 27:
            print('esc is pressed closing all windows')
            cv2.destroyAllWindows()
            break
    
    pass


if __name__ == '__main__':
    main()

