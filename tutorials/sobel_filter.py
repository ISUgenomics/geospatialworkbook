#! /usr/bin/env python
# Auth: Jennifer Chang
# Date: 2020/12/08
# Desc: Modified from the following tutorial and performs vertical and horizontal first derivative filtering
#       https://geospatial.101workbook.org/Workshops/Tutorial1_Image_Processing_Essentials_Boucheron.html
# USAGE: python sobel_filter.py <grayscaleimage.png>

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
    I_camera = np.asarray(imageio.imread(INFILE))
    
    # Sobel Filter
    v_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]         # vertical first derivative
    h_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]         # horizontal first derivative

    I_camera_v = ndimage.filters.convolve(I_camera.astype(float), v_filter)
    I_camera_h = ndimage.filters.convolve(I_camera.astype(float), h_filter)

    print("Sobel filtered:")
    print("  original:\n    min:{}\n    max:{}".format(str(I_camera.min()), str(I_camera.max())))
    print("  vertical filtered:\n    min:{}\n    max:{}".format(str(I_camera_v.min()), str(I_camera_v.max())))
    print("  horizontal filtered:\n    min:{}\n    max:{}".format(str(I_camera_h.min()), str(I_camera_h.max())))

    # Plot Side-by-Side Comparison
    plt.figure(figsize=(8,3))
    
    plt.subplot(1,3,1)
    plt.imshow(I_camera, cmap='gray')
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1,3,2)
    I_camera_v[I_camera_v < 0] = 0
    plt.imshow(I_camera_v, cmap='gray')
    plt.axis('off')
    plt.title('vertical_filter')

    plt.subplot(1,3,3)
    I_camera_h[I_camera_h < 0] = 0
    plt.imshow(I_camera_h, cmap='gray')
    plt.axis('off')
    plt.title('horizontal_filter')

    plt.show()
    

if __name__ == '__main__':
    main()
