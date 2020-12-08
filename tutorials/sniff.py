#! /usr/bin/env python
# Auth: Jennifer Chang
# Date: 2020/12/08
# Desc: Modified from the following tutorial and will print out the dimension of channels and display image
#       https://geospatial.101workbook.org/Workshops/Tutorial1_Image_Processing_Essentials_Boucheron.html
# USAGE: python sniff.py <color or grayscale image>

import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.color
import skimage.transform
import scipy.ndimage as ndimage

import sys

def main():
    INFILE = sys.argv[1]
    print("Image = {}".format(INFILE))
    I_input = np.asarray(imageio.imread(INFILE))

    # Print basic information of Image
    print("  dim:{}".format(I_input.shape))
    if (len(I_input.shape) == 2 ):
        print("  Grayscale image")
        print("  The range of values of Image:\n    min:{}\n    max:{}".format(str(I_input.min()), str(I_input.max())))
    else:
        print("  Color image")
        print("  Red Channel:\n    min:{}\n    max:{}".format(str(I_input[:,:,0].min()), str(I_input[:,:,0].max())))
        print("  Green Channel:\n    min:{}\n    max:{}".format(str(I_input[:,:,1].min()), str(I_input[:,:,1].max())))
        print("  Blue Channel:\n    min:{}\n    max:{}".format(str(I_input[:,:,2].min()), str(I_input[:,:,2].max())))

    # Display Image
    plt.figure()
    
    if ( len(I_input.shape) == 2 ):          # Grayscale 2 dimensions
        plt.imshow(I_input, cmap='gray')
    else:
        plt.imshow(I_input)                  # Color

    #plt.axis('off')
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title('Title')
    plt.show()        

if __name__ == '__main__':
    main()
