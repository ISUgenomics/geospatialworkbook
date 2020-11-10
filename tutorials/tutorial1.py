#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 
#%matplotlib inline
import imageio
import skimage.color
import skimage.transform
import scipy.ndimage as ndimage

I_camera = np.asarray(imageio.imread('data/cameraman.png'))
plt.figure()                      # open a new figure window
plt.imshow(I_camera, cmap='gray') # visualize the I_camera image with a grayscale colormap
# plt.show()                        # show the plot
plt.savefig('cameraman_grayscale.png')
