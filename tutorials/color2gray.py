#! /usr/bin/env python
# Auth: Jennifer Chang
# Date: 2020/12/08
# Desc: Modified from the following tutorial and converts a color image to grayscale
#       https://geospatial.101workbook.org/Workshops/Tutorial1_Image_Processing_Essentials_Boucheron.html
# USAGE: python color2gray.py <color image.png>

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

    # Color to gray
    I_input_gray = skimage.color.rgb2gray(I_input)
    
    # Plot Side-by-Side Comparison
    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.imshow(I_input)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1,2,2)
    plt.imshow(I_input_gray, cmap='gray')
    plt.axis('off')
    plt.title('Grayscale')

    plt.show()

#    # Save Grayscale
#    plt.figure()
#    plt.imshow(I_input_gray, cmap="gray")
#    plt.axis('off')
#    plt.margins(0,0)
#    plt.savefig("grayscale.png")

if __name__ == '__main__':
    main()
