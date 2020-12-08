#! /usr/bin/env python
# Auth: Jennifer Chang
# Date: 2020/12/08
# Desc: Modified from the following tutorial and trains the model
#       https://geospatial.101workbook.org/Workshops/Tutorial1_Image_Processing_Essentials_Boucheron.html
# USAGE: python classify.py "101_ObjectCategories/*" "Annotations/*"

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import imageio
import skimage.measure
import skimage.feature
import scipy.io as spio
import sklearn
import sklearn.metrics

import sys

def main():
    INFILE = sys.argv[1]                      # This is expecting "101_ObjectCategories/*"
    categories = sorted(glob.glob(INFILE))
#    print("Categories = {}".format(categories))
    an_categories = sorted(glob.glob(sys.argv[2]))    # Expecting "Annotations/*"

    # Diagnostic plot, show one example image per category
    plt.figure(figsize = (8,8))
    plt.rcParams.update({'font.size':5})
    for k, category in enumerate(categories):
        I = np.asarray(imageio.imread(category+'/image_0001.jpg'))    # Get first image in the category "image_0001.jpg"
        if len(I.shape)==2:
            plt.set_cmap('gray')
        plt.subplot(11, 10, k+1)
        plt.imshow(I)
        plt.axis('off')
        plt.title(os.path.basename(category))

    plt.show()

    # Load in annotations (this was pre-computed?)
    I = np.asarray(imageio.imread('101_ObjectCategories/emu/image_0001.jpg'))

    ann = spio.loadmat('Annotations/emu/annotation_0001.mat')
    print("box_coord:\n  {}".format(ann['box_coord']))
    print("obj_contour:\n  {}".format(ann['obj_contour']))

    # Diagnostic plot (make sure coordinates are not flipped in annotation
    plt.figure()
    plt.imshow(I)
    
    # plot expects the x-axis first and the y-axis second (col first, row second)
    plt.plot(ann['obj_contour'][0,:]+ann['box_coord'][0,2]-1,
             ann['obj_contour'][1,:]+ann['box_coord'][0,0]-1,'w')
    
    plt.axis('off')
    plt.title('Annotated Emu')
    plt.show()

    # Compute binary object mask
    r,c = skimage.draw.polygon(ann['obj_contour'][1,:] + ann['box_coord'][0,0]-1,
                               ann['obj_contour'][0,:] + ann['box_coord'][0,2]-1, I.shape)
    binary_mask = np.zeros(I.shape)
    binary_mask[r,c] = 1

    plt.figure()
    
    plt.subplot(1,2,1)
    plt.imshow(I)
    plt.axis('off')
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.title("Binary Mask")

    plt.show()

    # Diagnostic plot, show the masks per image
    plt.figure(figsize = (8,8))
    plt.rcParams.update({'font.size':5})
    for k, category in enumerate(categories):
        an_catagory = an_categories[k]
        ann = spio.loadmat(an_catagory + '/annotation_0001.mat')
        I = np.asarray(imageio.imread(category+'/image_0001.jpg'))    # Get first image in the category "image_0001.jpg"
        r,c = skimage.draw.polygon(ann['obj_contour'][1,:] + ann['box_coord'][0,0]-1,
                                   ann['obj_contour'][0,:] + ann['box_coord'][0,2]-1, I.shape)
        binary_mask = np.zeros(I.shape)
        binary_mask[r,c] = 1
        plt.subplot(11, 10, k+1)
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        plt.title(os.path.basename(category))

    plt.show()
                 
    
if __name__ == '__main__':
    main()
