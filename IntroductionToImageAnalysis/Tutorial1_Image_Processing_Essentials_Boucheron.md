---
title: Image Processing Fundamentals
layout: single
author: Laura Boucheron
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

# Section 0: Preliminaries 

## Section 0.1 A reminder on Jupyter Notebooks

There are two main types of cells in this notebook: code and markdown (text).  You can add a new cell with the plus sign in the menu bar above and you can change the type of cell with the dropdown menu in the menu bar above.  As you complete this tutorial, you may wish to add additional code cells to try out your own code and markdown cells to add your own comments or notes. 

Markdown cells can be augmented with a number of text formatting features, including
  - bulleted
  - lists

embedded $\LaTeX$, monotype specification of `code syntax`, **bold font**, and *italic font*.  There are many other features of markdown cells--see the jupyter documentation for more information.

You can edit a cell by double clicking on it.  If you double click on this cell, you can see how to implement the various formatting referenced above.  Code cells can be run and markdown cells can be formatted using Shift+Enter or by selecting the Run button in the toolbar above.

Once you have completed (all or part) of this notebook, you can share your results with colleagues by sending them the `.ipynb` file.  Your colleagues can then open the file and will see your markdown and code cells as well as any results that were printed or displayed at the time you saved the notebook.  If you prefer to send a notebook without results displayed (like this notebook appeared when you downloaded it), you can select ("Restart & Clear Output") from the Kernel menu above.  You can also export this notebook in a non-executable form, e.g., `.pdf` through the File, Download As or File, Export Notebook as menu.

**Last Update**: Jan 2024 by Noa Mills <br />
**Jupyter Notebook**: [Tutorial1_Image_Processing_Essentials_Boucheron.ipynb](https://geospatial.101workbook.org/tutorials/Tutorial1_Image_Processing_Essentials_Boucheron.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ISUgenomics/geospatialworkbook/HEAD?filepath=tutorials)

# Image Processing Fundamentals

This tutorial was modified from the 2020 AI Workshop which itself was modifed from tutorials given by Laura E. Boucheron, Electrical & Computer Engineering, NMSU

> Copyright (C) 2020  Laura E. Boucheron
> 
> This information is free; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
> 
> This work is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
> 
> You should have received a copy of the GNU General Public License along with this work in a file `COPYING.TXT`; if not, see <https://www.gnu.org/licenses/>.
> 

## Overview

In this tutorial, we present a brief overview of image processing concepts necessary to understand machine learning and deep learning.  Completion of this tutorial should give participants the basic background and terminology necessary for an understanding of the basics of image processing and the common manipulations of images used for machine learning and deep learning.

A Jupyter notebook is made available so you can interactively work through the tutorial. You can learn more about Jupyter from the tutorials in the data science workbook [here](https://datascience.101workbook.org/04-DevelopmentEnvironment/01B-jupyter-basics). Below, you will find instructions to download the notebook for this tutorial to scinet, as well as how to set up the computing environment.

This tutorial contains 5 sections:
  - **Section 0: Preliminaries**: some notes on using this notebook and how to download the two images that we will use for this tutorial
  - **Section 1: Working with Grayscale Images**: how to read, query characteristics, intepret, and display grayscale images
  - **Section 2: Working with Color Images**: how to read, query characteristics, interpret, and display color images
  - **Section 3: Transforming Images**: how to convert between grayscale and color images, how to rescale the spatial dimensions of an image through cropping and resizing, and other geometric transformations
  - **Section 4: Image Convolution**: the basics of filtering images through convolution with a filter kernel
  
There are subsections with the heading **<span style='color:Green'> Your turn: </span>** throughout this tutorial in which you will be asked to apply what you have learned. **We encourage you to reveal the sample answers provided after you attempt to answer the questions yourself.** 

## Section 0.2a Set Up your Computing Environment

This tutorial can either be run on a local machine, or on Atlas. We recommend that you pull up the ["view-able" version of this notebook](https://geospatial.101workbook.org/IntroductionToImageAnalysis/Tutorial1_Image_Processing_Essentials_Boucheron.html) in your browser, and follow the instructions to set up the conda environment and your project directory before opening up this tutorial.

##### If Running on your local machine:

Ensure that you have either miniconda or anaconda installed. You can follow the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/) to install miniconda. Open up the anaconda or miniconda terminal.

##### If Running on Atlas:

Connect to Atlas through the [Open OnDemand interface](https://atlas-ood.hpc.msstate.edu/). Open a terminal from inside JupyterLab (File > New > Terminal). Run the following command to load miniconda.

```bash
module load miniconda
```

### Create your Project Directory and Conda Environment

Navigate to the directory you wish to use for the image processing tutorial series. This could be in your home directory, a project directory, or 90daydata. Run the following commands to create the directory tree for this tutorial series and the conda environment. Note that the creation of the conda environment may take 5-10 minutes.

```bash
mkdir ImageProcessingSeries
cd ImageProcessingSeries
mkdir Tutorial1 Tutorial2 Tutorial3 Tutorial4 Tutorial5
wget https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/aiworkshop.yml
source activate
conda env create aiworkshop -f aiworkshop.yml
conda activate aiworkshop
python -m ipykernel install --user --name=aiworkshop
wget https://geospatial.101workbook.org/tutorials/Tutorial1_Image_Processing_Essentials_Boucheron.ipynb -P Tutorial1
```
</br>

Now, you can open up this tutorial from `ImageProcessingSeries/Tutorial1/Tutorial1_Image_Processing_Essentials_Boucheron.ipynb`. If you are running this tutorial on your local machine, run the command `jupyter notebook` in your terminal and navigate to the file. If you are running this tutorial on Atlas and you created the `ImageProcessingSeries` in your home directory, you can navigate to the tutorial file within the Jupyter Labs navigation pane. If you are running this tutorial on Atlas and you created the `ImageProcessingSeries` directory elsewhere, you can create a symbolic link between your project directory and your home directory as follows, then you can find the tutorial within the navigation pane:

```bash
ln -s /path/to/ImageProcessingSeries ~
```

When the environment finishes building, open up the tutorial. Next, you can set your kernel by selecting the `Kernel` tab in the top left of the screen, selecting `Change Kernel`, then selecting `aiworkshop`. Now you should see the label `aiworkshop` in the top right of the screen.

You're all set!

##### Troubleshooting recommendations can be found [here](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/setup/).

## Section 0.2b Downloading Images
Next, we need to download images to work with in this tutorial.  In your terminal, navigate to `/path/to/ImageProcessingSeries/Tutorial1` and run the following commands. 

```bash
wget https://geospatial.101workbook.org/tutorials/data/cameraman.png
wget https://geospatial.101workbook.org/tutorials/data/peppers.png
```

Both of these images are common example images used in image processing and are often included as part of the distribution of image processing toolboxes.

**cameraman.png** 

<img src="https://geospatial.101workbook.org/tutorials/data/cameraman.png">

**peppers.png**

<img src="https://geospatial.101workbook.org/tutorials/data/peppers.png">

Notice how the first image `cameraman.png` is in grayscale while the `peppers.png` image is in color. This means the amount of information (bands) per pixel is going to be different. Sometimes you want to focus on certain features which is better rendered/analyzed in grayscale or in color. You may be reducing the data size by converting the color image to grayscale, or mapping an important band to a color (more on that later).

## Section 0.3 Import Necessary Libraries

First, we import necessary libraries:
  - We `import numpy as np` so we can store our image data as numpy arrays and apply numpy functions to our images.
  - We import the `matplotlib` plotting library which provides many common plotting routines (including image visualization).  There are other plotting libraries, but `matplotlib` was designed to mimic much of the functionality of Matlab plotting and is thus very nice for those of us who transitioned from Matlab to python.
  - We use `%matplotlib inline` to specify that plots should occur inline in the notebook (rather than in external figure windows).  This is very convenient if you want a single document of your code and results.
  - We import the `imageio` library which provides functions to read common image formats.  We use imageio here since it returns images as an array.  We note that there are other powerful image libraries, including `PIL` / `pillow` which is used by the very nice `pandas` library for data manipulation.  We work with images as simple `nparrays` here since that best illustrates the basic image processing concepts.
  - We import two packages from scikit-image (`skimage`) which provides image manipulation functions.
  - We import the `ndimage` package from `scipy` which provides image filtering functions.


```python
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
import imageio
import skimage.color
import skimage.transform
import scipy.ndimage
```

# Section 1: Working with Grayscale Images

## 1.1 Reading in the image
We can read in the images using the `imageio.imread` command.  We explicitly cast the image as an `nparray` as this will give us access to some helpful characteristics of the image.  We begin with the grayscale `cameraman.png` image.


```python
INFILE='data/cameraman.png'                    # path to image
I_camera = np.asarray(imageio.v2.imread(INFILE))  # load image and convert to array
```

## 1.2 Displaying the image
Let's display this image.  We use the `matplotlib` `imshow` command.


```python
plt.figure()                      # open a new figure window
plt.imshow(I_camera, cmap='gray') # visualize the I_camera image with a grayscale colormap
plt.show()                        # show the plot
```


    
![png](output_10_0.png)
    


### A note about coordinate conventions

By default, axis labels are included which demarcate pixel counts.  You may notice that the origin of an image is interpreted as the **upper left** corner and not the lower left corner as you might have expected.  This is a consequence of the fact that we use standard linear algebra style indexing for images where pixel $(n,m)$ is indexed in row, column order.  For those of you who might be particularly concerned, this coordinate system still describes a right-handed system. 

This coordinate system can cause issues later on if you accidentally swap indices.  You might think you are looking in the upper right but are actually looking in the lower left.  You might think you are traversing left to right and are actually traversing up to down.

## 1.3 Changing display parameters
There are various choices in display that you can make, including:
  - scaling the figure window using `figsize=(x,y)` within the `plt.figure()` command.  In this, `x` and `y` are in units of inches by default. The concept of measuring the image in inches comes from printing standards, and doesn't apply very intuitively to computer graphics. An image that is defined as a given size in inches may render as larger on one screen and smaller on another since different screens have different resolutions. When the user specifies the dimensions of an image in inches, these dimensions are converted from inches to pixels. By default, matplotlib uses 72 pixels per square inch, though this value can be modified by the user. A reasonable choice for these values will depend on your computer's resolution and screen size. More information on figure size units can be found [here](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html).
  - scaling the size of the text labels with the command `plt.rcParams.update({'font.size': f})` where `f` is the font size you desire in units of pt, e.g., 20.  You need to run this only once to update the font size parameters, after which all subsequent figure windows will use this new font size. The "rc" in "rcParams" stands for runtime configuration, and the rcParams variable stores configuration variables in a dictionary-like datatype. You can find more information about runtime configurations for matplotlib [here](https://matplotlib.org/stable/users/explain/customizing.html#customizing-with-dynamic-rc-settings).
  - removing all axis decorators (including ticks, tick labels, axis labels, etc.) with the command `plt.axis('off')`
  - adding axis labels or a title to your plot, e.g., `plt.xlabel('flamingos')`, `plt.ylabel('emus')`, `plt.title('Emus versus flamingos')`.  Note that if you have turned the axes off, your titles will not show up. You can verify this by commenting out the line `plt.axis('off')` and observing how it affects the output.


```python
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,10))              # open a new figure window of size 10x10 (artbitrary units)
plt.imshow(I_camera, cmap='gray')        # visualize the I_camera image with a grayscale colormap
plt.axis('off')                          # turn off the axis decorators
plt.xlabel('flamingos')                  # provide a label for the x axis
plt.ylabel('emus')                       # provide a label for the y axis
plt.title('Emus versus flamingos')       # provide a title for the plot
plt.show()                               # show the plot
```


    
![png](output_13_0.png)
    


## <span style='color:Green'> Your turn:  </span>
Choose a figure size so that the image fills the width of your notebook and provide a descriptive title to your image.  You may also choose to label your axes or not, per your preference.  For what it's worth, image processing people don't tend to display axis labels. 


```python
# Type and run your answer here
```

<details>

<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
plt.figure(figsize=(10,10)) # open a new figure window of size 10x10 inches
plt.imshow(I_camera, cmap='gray') # visualize the I_camera image with a grayscale colormap
plt.axis('off') # turn off the axis decorators
plt.title('Cameraman, grayscale') # provide a title for the plot
plt.show() # show the plot
```
</details>

## 1.4 Printing Image Characteristics
We can check on important characteristics of `I_camera` by using built-in numpy attributes and functions. The following code prints out the data's shape, the number of dimensions, the type of data each index in the numpy array holds, and the maximum and minimum values.

### 1.4.1 Accessing numpy array attributes and functions


```python
print("Array shape: ", I_camera.shape)
print("Number of dimensions: ", I_camera.ndim)
print("Data type: ", I_camera.dtype)
print("Maximum value: ", I_camera.max())
print("Minimum value: ", I_camera.min())
```

    Array shape:  (256, 256)
    Number of dimensions:  2
    Data type:  uint8
    Maximum value:  253
    Minimum value:  7
    

### A note on common image variable types
We see that `I_camera` is an `ndarray` of size $256\times256$ pixels and of variable type `uint8` (unsigned 8-bit integer).  Remember that computers store data natively in binary (base-2) format.  The `uint8` variable type means we have 8 bit, unsigned (positive) integers. It is very common that image pixels are represented as `uint8` variables, which indicates that the pixel values are within the range $[0,255]$ since there are 256 total different values you can represent with 8 bits.  

Since there is only one color channel, i.e., `I_camera` is a 2D array $\in\mathbb{R}^{N\times M}$ rather than a 3D array $\in\mathbb{R}^{N\times M\times C}$, we also know that this is a grayscale image. As we shall see in section 2, color images have an additional dimension because each pixel holds multiple values that together represent the color of the pixel. Greyscale images, on the other hand, only require one value per pixel.

### A note on image intensity conventions
We note that this ```I_camera``` image spans the range $[7,253]$.  In grayscale images, it is common interpretation that **darker pixels have smaller intensity values and lighter pixels have larger intensity values**.  

### 1.4.1 Printing a portion of the image
It is also important to remember that the computer "sees" only an array of values.  To reinforce this, we can look at what the computer "sees" in a portion of the image.


```python
print(I_camera[100:110, 100:110])   # subregion that is 10 x 10 pixels          
I_camera_portion = I_camera[100:110, 100:110]
```

    [[  9  11  13  11  11  11  16 106 178  68]
     [ 12  12  12  11  12  11  69 181  62  15]
     [ 13  12  12  11  12  82 168  60  14  13]
     [ 11  10   9  10  69 182  67  14  12  14]
     [ 10  10  10  71 200  81  15  12  14  14]
     [ 12  12  58 204  91  17  12  14  14  17]
     [ 11  46 201 106  18  14  16  15  16  16]
     [ 34 185 122  23  10  14  17  16  13  13]
     [186 135  30  11   9   9  10  10   9  10]
     [154  33  11  13  12   9   9   9   9  11]]
    

## <span style='color:Green'> Your turn:  </span>
What does this printout tell us about that part of the image?  

Write your answer here in plain text.

<details>

<summary>Click to reveal sample answer</summary>

There is a "stripe" of light-valued pixels (large intensity values) oriented at approximately 45 degrees through this portion of the image. On either side of that bright stripe, the image is very dark.

</details>

### 1.4.2 Visualizing a portion of an image
We could use `plt.imshow` to display that small portion of the image.


```python
plt.figure()
plt.imshow(I_camera[100:110, 100:110], 
           cmap='gray')
plt.axis('off')
plt.title('Cameraman portion, grayscale') 
plt.show() 
```


    
![png](output_28_0.png)
    


## <span style='color:Green'> Your turn:  </span>
Does this display of the image verify your interpretation from the printout of the pixel values?

Write your answer here in plain text.

<details>

<summary> Click here to reveal sample answer </summary>

Yes, there is indeed a white stripe across the image from the lower left to the upper right. This makes sense, because the intensity values of those pixels are high, and the intensity values of the surrounding pixels are low.

</details>

### 1.4.3 Another visualization of a portion of an image
Here, we maintain the display of the whole image, and plot a yellow box around the area that we've been discussing.  This can be a helpful visualization since it maintains the context of the box.


```python
plt.figure(figsize=(10,10)) 
plt.imshow(I_camera, cmap='gray') 
plt.axis('off') 
plt.title('Cameraman, grayscale') 
plt.plot([100,100], [100,110], 'y-',linewidth=3)       # Draws the yellow rectangle
plt.plot([110,110], [100,110], 'y-',linewidth=3)
plt.plot([100,110], [100,100], 'y-',linewidth=3)
plt.plot([100,110], [110,110], 'y-',linewidth=3)
plt.show() 
```


    
![png](output_33_0.png)
    


## <span style='color:Green'> Your turn:  </span>
What happens if you plot the image using `imshow` but "forget" to specify the colormap as `gray`?


```python
# Write your answer here

```

<details>

<summary> Click here to reveal sample answer </summary>

```python    
plt.figure(figsize=(5,5)) # open a new figure window of size 5x5 inches
plt.imshow(I_camera) # visualize the I_camera image with a grayscale colormap
plt.axis('off') # turn off the axis labels
plt.title('Cameraman, default colormap') # provide a title for the plot
plt.show() # show the plot
```
</details>


### A note on colormaps
You should have found that the grayscale image now appears colored.  How can that be if the image is a single channel, i.e., grayscale image?  In this case, python is applying the default colormap to the intensities.  In this default colormap, pixels with values closer to 0 appear dark blue, pixels with values in the middle appear green, and pixels with values closer to 255 appear yellow. (Your computer may use a different default colormap in which case the colors noted above may not be correct). 

See [here](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html) for a comprehensive list of available colormaps and examples.

There are also many other options for `plt.imshow`, see `help(plt.imshow)` for more details.

# Section 2: Working with Color Images

## 2.1 Reading in and displaying the image
Now, we turn to the color `peppers.png` image.  We use the same command to read in the image and the same basic commands to visualize the image.  The only difference here is that we allow python to choose a default colormap for this color image.


```python
I_pepper = np.asarray(imageio.v2.imread('data/peppers.png'))
plt.figure(figsize=(20,20))            # open a new figure window of size 5x5
plt.imshow(I_pepper)                 # visualize the I_pepper image with a default colormap
plt.axis('off')                      # turn off the axis labels and other decorators
plt.title('Peppers, RGB')            # provide a title for the plot
plt.show()                           # show the plot
```


    
![png](output_40_0.png)
    


## 2.2 Printing image characteristics
We can check on important characteristics of `I_pepper` with the `.shape` , `.ndim` and `.dtype` attributes.

### 2.2.1 Image characteristics


```python
print("Image shape: ", I_pepper.shape)
print("Number of dimensions: ", I_pepper.ndim)
print("Image dtype: ", I_pepper.dtype)
```

    Image shape:  (384, 512, 3)
    Number of dimensions:  3
    Image dtype:  uint8
    

### A note on color channel conventions
We see that `I_pepper` is an `ndarray` of size $384\times512\times 3$ pixels and of variable type `uint8` (unsigned 8-bit integer).  We thus have a 3-channel image where the three channels are assumed to be a red (R), green (G), and blue (B) channel, in that order. These images are commonly called RGB images.

Again, we note that image pixels are represented as `uint8` variables.  In this case, however, each pixel is associated with 3 `uint8` values, one for each channel. This results in $2^8 2^8 2^8=2^{24}=16,777,216$ unique colors.  **Colors which have equal values of R, G, and B are grayscale.**

### 2.2.2 Max and min values
We can check for the actual maximum and minimum values of the image or of the R, G, or B channels.


```python
print('Max and min values of the image:')          # Min and max across all channels
print('    Min: '+str(I_pepper.min()))
print('    Max: '+str(I_pepper.max()))
print('Max and min values of the red channel:')    # The zero [:,:,0] indices is R (red) channel
print('    Min: '+str(I_pepper[:,:,0].min()))
print('    Max: '+str(I_pepper[:,:,0].max()))
print('Max and min values of the green channel:')  # The one [:,:,1] indices is G (green) channel
print('    Min: '+str(I_pepper[:,:,1].min()))
print('    Max: '+str(I_pepper[:,:,1].max()))
print('Max and min values of the blue channel:')   # The two [:,:,2] indices is B (blue) channel
print('    Min: '+str(I_pepper[:,:,2].min()))
print('    Max: '+str(I_pepper[:,:,2].max()))
```

    Max and min values of the image:
        Min: 0
        Max: 255
    Max and min values of the red channel:
        Min: 5
        Max: 255
    Max and min values of the green channel:
        Min: 1
        Max: 255
    Max and min values of the blue channel:
        Min: 0
        Max: 255
    

### A note on intensity conventions in color images
We note that this ```I_pepper``` image spans the range $[5,255]$ in R, $[1,255]$ in G, and $[0,255]$ in B.  We also note that when we didn't specify a color channel, python returned the max and min across the three color channels.

Extending the interpretation of a single channel image in which darker pixels have smaller intensity values and lighter pixels have larger intensity values, a color is defined as the contribution of R, G, and B, where larger intensities in those channels correspond to larger contribution from those colors. For example, if the RGB values of a pixel are (255, 0, 0), then the pixel is red. You can see more about the RGB color model [here](https://www.britannica.com/science/RGB-colour-model).

### 2.2.3 Visualizing a single color channel
We can visualize the contribution of each color channel across the image by plotting the image with a single color channel in grayscale. Compare the objects in the image with larger contributions from a single color channel to objects in the image with larger contributions from multiple color channels.


```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,12))      # open a new figure window with three subplots 
ax1.imshow(I_pepper[...,0], cmap="grey")                          # visualize the I_pepper image with only the red values
ax1.set_title('Red Channel')
ax1.axis('off')      
ax2.imshow(I_pepper[...,1], cmap="grey")                          # visualize the I_pepper image with only the green values
ax2.set_title('Green Channel')
ax2.axis('off')      
ax3.imshow(I_pepper[...,2], cmap="grey")                          # visualize the I_pepper image with only the blue values
ax3.set_title('Blue Channel')
ax3.axis('off')      
plt.show()                                                        # show the plot
```


    
![png](output_49_0.png)
    


Recall that a lighter shade indicates higher intensity contribution from that color channel. Objects that appear white appear to have more contribution from all color channels.

We can also visualize the contribution of each channel in color, by setting the all values from the other color channels to 0. Note the relatively large contributions of red and green to yellow objects, and the dark contrast of blue values corresponding to that channels low overall contribution to the image.


```python
# make copies of the image
I_pepper_red = np.copy(I_pepper)
I_pepper_green = np.copy(I_pepper)
I_pepper_blue = np.copy(I_pepper)

# set values of a single color channel to 0
I_pepper_red[...,1] *= 0 
I_pepper_red[...,2] *= 0
I_pepper_green[..., 0] *= 0
I_pepper_green[...,2] *= 0
I_pepper_blue[..., 0] *= 0
I_pepper_blue[...,1] *= 0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,12))      # open a new figure window with three subplots 
ax1.imshow(I_pepper_red)                                          # visualize the I_pepper image with only the red values
ax1.set_title('Red Channel')
ax1.axis('off')      
ax2.imshow(I_pepper_green)                                        # visualize the I_pepper image with only the green values
ax2.set_title('Green Channel')
ax2.axis('off')      
ax3.imshow(I_pepper_blue)                                         # visualize the I_pepper image with only the blue values
ax3.set_title('Blue Channel')
ax3.axis('off')      
plt.show()                                                        # show the plot
```


    
![png](output_51_0.png)
    


## <span style='color:Green'> Your turn:  </span>
Try plotting two color channels together for each channel combination.


```python
# Write your answer here
 
```

<details>

<summary> Click here to reveal sample answer </summary>

```python    
# make copies of the image
I_pepper_nored = np.copy(I_pepper)
I_pepper_nogreen = np.copy(I_pepper)
I_pepper_noblue = np.copy(I_pepper)

# set values of a single color channel to 0
I_pepper_nored[..., 0] *= 0
I_pepper_nogreen[..., 1] *= 0
I_pepper_noblue[..., 2] *= 0
    
# plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,12))      # open a new figure window with three subplots 
ax1.imshow(I_pepper_noblue)                        # visualize the I_pepper image with red-green values
ax1.set_title('Red/Green Channels')
ax1.axis('off')      
ax2.imshow(I_pepper_nored)                           # visualize the I_pepper image with green-blue values
ax2.set_title('Green/Blue Channels')
ax2.axis('off')      
ax3.imshow(I_pepper_nogreen)                         # visualize the I_pepper image with red-blue values
ax3.set_title('Red/Blue Channel')
ax3.axis('off')      
plt.show()    
```
</details>


### 2.2.4 Printing a portion of the image
Let's explore a 10x10 pixel portion of our image to better understand how the color channels work. First, we will visualize the portion by printing out a matrix of intensity values for each channel, then we will visualize the portion by plotting a greyscale image of each channel.


```python
print('Red:')
print(I_pepper[100:110,100:110,0])
print('Green')
print(I_pepper[100:110,100:110,1])
print('Blue')
print(I_pepper[100:110,100:110,2])
```

    Red:
    [[ 62  60  60  61  61  61  60  57  61  67]
     [ 64  61  59  60  59  56  55  56  59  64]
     [ 62  62  61  59  56  52  62  73  85  92]
     [ 64  63  61  66  67  77  92 101 108 112]
     [ 63  60  59  71  86  96 102 109 110 114]
     [ 71  67  71  84  96 100 106 115 119 120]
     [107 104 100  99 104 109 112 123 126 127]
     [118 116 116 113 114 118 113 117 129 135]
     [120 119 117 110 105 108 110 107 114 127]
     [129 129 123 121 115 115 111 105 103 105]]
    Green
    [[ 34  35  36  36  33  32  32  34  34  38]
     [ 37  36  36  36  33  33  33  33  34  41]
     [ 35  36  37  39  38  37  44  58  72  85]
     [ 36  36  35  38  47  69  87  99 108 114]
     [ 39  37  39  57  82  97 103 107 113 117]
     [ 55  49  60  85  96 105 110 114 119 122]
     [104 100  99 101 103 108 116 119 121 124]
     [115 113 114 114 114 115 114 116 125 133]
     [116 112 110 109 105 106 107 108 111 123]
     [123 124 120 117 113 111 108 104 100  99]]
    Blue
    [[60 63 64 61 58 56 55 55 61 66]
     [66 63 61 60 60 59 56 57 55 55]
     [60 63 67 62 58 56 50 46 40 34]
     [61 61 64 60 50 44 29 21 16 11]
     [66 60 55 43 36 28 16 10 15 21]
     [49 48 37 26 22 19 13  6 10 16]
     [23 23 21 20 19 18 14  9 10  6]
     [24 21 23 24 24 30 23 17 19 22]
     [32 32 31 28 23 26 27 22 15 22]
     [32 26 15 11 12  7  8 15 12 14]]
    


```python
plt.figure(figsize=(5,5)) # open a new figure window of size 5x5
plt.imshow(I_pepper[100:110,100:110,0], cmap="gray") # visualize the red channel of a section of the image
plt.axis('off') # turn off the axis labels and other decorators
plt.title('Red Values') # provide a title for the plot
plt.show() # show the plot
```


    
![png](output_57_0.png)
    



```python
plt.figure(figsize=(5,5)) # open a new figure window of size 5x5
plt.imshow(I_pepper[100:110,100:110,1], cmap="gray") # visualize the green channel of a section of the image
plt.axis('off') # turn off the axis labels and other decorators
plt.title('Green Values') # provide a title for the plot
plt.show() # show the plot
```


    
![png](output_58_0.png)
    



```python
plt.figure(figsize=(5,5)) # open a new figure window of size 5x5
plt.imshow(I_pepper[100:110,100:110,2], cmap="gray") # visualize the green channel of a section of the image
plt.axis('off') # turn off the axis labels and other decorators
plt.title('Blue Values') # provide a title for the plot
plt.show() # show the plot
```


    
![png](output_59_0.png)
    


What does this tell us about what this portion of the image looks like?  It can be a bit harder to interpret this sort of visualization for a color image since we must keep track of multiple color channels simultaneously.

There appear to be two basic regions of different characteristics: one in the upper left triangle of the window and one in the lower right.  This is most obvious in the R and G channels, where we see a transition from small values in the upper left transitioning to larger values in the lower right.  We also see a smaller effect in the B channel transitioning from larger values in the upper left to smaller values in the lower right.

In the upper left triangle, it appears that the image is a dark grayish purple since:
 - the R and B contributions are approximately equal
 - there is less G than R or B
 - all contributions are fairly small
 - since R+B yields magenta, I expect the region to be a dark magenta
 - since there is also some contribution from G, I expect the color to be a bit "muddied"

In the lower right triangle, it appears that the image is greenish since:
 - the R and G contributions are approximately equal
 - there is a very small contribution from B
 - since R+G yields yellow, I expect the region to be green
 - since the intensities of R and G are mid-range and the contribution from B is very small, I expect this to be a mid-green

Here we visualize the section of our image in RGB.


```python
plt.figure(figsize=(5,5))
plt.imshow(I_pepper[100:110,100:110]) # visualize the I_pepper image
plt.axis('off') # turn off the axis labels
plt.title('Peppers Section, RGB') # provide a title for the plot
plt.show() # show the plot
```


    
![png](output_63_0.png)
    


## <span style='color:Green'> Your turn:  </span>
Visualize where in the image we are looking by overlaying a box on the image visualization.


```python
# Write your answer here

```

<details>
<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
plt.figure(figsize=(20,20))
plt.imshow(I_pepper) # visualize the I_pepper image
plt.axis('off') # turn off the axis labels
plt.title('Peppers, RGB') # provide a title for the plot
plt.plot([100,100],[100,110],'w-',linewidth=2)
plt.plot([110,110],[100,110],'w-',linewidth=2)
plt.plot([100,110],[100,100],'w-',linewidth=2)
plt.plot([100,110],[110,110],'w-',linewidth=2)
plt.show() # show the plot
```
</details>

We find that our conclusions regarding the appearance of the image in the window are validated: we have a dark purple region in the upper left corner, transitioning to the green of the pepper in the lower right.

# Section 3: Transforming Images
We will find that many deep learning methods are very particular about the size of input images.  This particularity about size extends across all three dimensions--the two spatial dimensions and the color dimension. For example, some models may only accept greyscale images that are 28x28 pixels, whereas others only accept RGB images that are 250x250 pixels. As such, it is useful to learn a couple of common methods to rescale images in all three dimensions. Here, we will learn how to **convert between RGB and grayscale**, how to **convert between RGB and other color spaces**, how to **crop** images, how to **resize** images.  

## 3.1 Color to Grayscale
We can convert a color image to a grayscale image using a standard command included in Scikit-Image.  We can use the `skimage.color.rgb2gray` function to convert the RGB image `I_pepper` to a grayscale image.  The `skimage.color.rgb2gray` function applies a weighted averaging of the three color channels to yield a grayscale image.  As a note, there is no single accepted weighting to convert between a color and grayscale image, so your results using `skimage` may differ from results using other libraries or programming languages. You can find more information about the weighting used in the Scikit-Image library [here](https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html).


```python
I_pepper_gray = skimage.color.rgb2gray(I_pepper)
```

## <span style='color:Green'> Your turn:  </span>
What are the dimensions of `I_pepper_gray`?  How many channels does it have?  What is the variable type?  What are the max and min values? Write a few lines of code to display these values.


```python
# Write your code here
```

<details>
<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
print("Image shape: ", I_pepper_gray.shape)
print("Image dtype: ", I_pepper_gray.dtype)
print('Min: ', I_pepper_gray.min())
print('Max: ', I_pepper_gray.max())
```
</details>

The image `I_pepper_gray` is $385\times512$ pixels (the same spatial dimensions as `I_pepper`) and has one color channel.  It is of variable type `float64`.

Here we visualize the grayscale version of the peppers image. Note that matplotlib's `imshow` function expects pixel values to either be integers in the range [0,255] or floats in the range [0,1]. When we called the `skimage.color.rgb2gray(I_pepper)` function, the resulting `I_pepper_gray` object has pixel values that are floats between 0 and 1, whereas the original `I_pepper` object has pixels that are integers from 0 to 255.


```python
plt.figure(figsize=(20,20)) 
plt.imshow(I_pepper_gray,cmap='gray') 
plt.axis('off') 
plt.title('Peppers, rgb2gray')
plt.show()
```


    
![png](output_76_0.png)
    


Here's an example of the importance of variable types and the implied ranges. Here, we take the I_pepper_gray image, which has float values bewteen 0 and 1, and we cast it to `uint8`. This means that all float values that were in the range [0,1) are cast to the integer value 0, and only float values that are exactly 1 are cast to the integer value 1.


```python
I_pepper_gray_uint8 = I_pepper_gray.astype(np.uint8)

plt.figure(figsize=(10,10))
plt.imshow(I_pepper_gray_uint8,cmap='gray')
plt.show()
```


    
![png](output_78_0.png)
    


### A note about float-valued images
Some functions, e.g., functions that write out to standard image formats, may expect `uint8` variables.  You can always cast back to `uint8` as needed, e.g., `I_pepper_gray_uint8=(I_pepper_gray*255).astype(np.uint8)`. Here's how we can correctly cast our float-valued `I_pepper_gray` object to integers in the range [0,255]:


```python
I_pepper_gray_uint8=(I_pepper_gray*255).astype(np.uint8)

plt.figure(figsize=(5,5))
plt.imshow(I_pepper_gray_uint8, cmap="gray")
plt.show()
```


    
![png](output_80_0.png)
    


A common issue in image processing is a mismatch between the expected and actual variable type and/or intensity range.  If a function is expecting a `float` in the range $[0,1]$ and gets instead a `uint8` in the range $[0,255]$, unexpected things can happen.  A non-exhaustive list of some of the issues you might encounter:
  - The code will throw an error.  
  - The code will intelligently convert between the variable types (but this might mean you receive a different intensity range back from the code).
  - The code will unintelligently convert between the variable types.  
  - You accidentally end up performing integer arithmetic instead of floating-point arithmentic.  This is a particularly fun one to track down.

## 3.2 Grayscale to Color
We can similarly convert a grayscale image to a color image using a standard command included in Scikit-Image.  It is important to note that this conversion is really just creation of an image with a third dimension.  Each of the color channels will be identical since we cannot infer color from solely a grayscale image.


```python
I_camera_rgb = skimage.color.gray2rgb(I_camera)
```

## <span style='color:Green'> Your turn:  </span>
What are the dimensions of `I_camera_rgb`?  How many channels does it have?  What is the variable type?  What are the max and min values of each channel? Write a few lines of code to display these values.


```python
# Write your answer here
```

<details>
<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
print("I_camera_rgb shape: ", I_camera_rgb.shape)
print("I_camera_rgb dtype: ", I_camera_rgb.dtype)
print("I_camera_rgb min: ", I_camera_rgb.min())
print("I_camera_rgb max: ", I_camera_rgb.max())
```
</details>

The image `I_camera_rgb` has the dimension $256\times256\times3$. It has the same spatial dimensions as `I_camera`) and has three color channels.  It is of variable type `uint8`.


```python
print('Max and min values of the image:')
print('    Min: '+str(I_camera_rgb.min()))
print('    Max: '+str(I_camera_rgb.max()))
print('Max and min values of the red channel:')
print('    Min: '+str(I_camera_rgb[:,:,0].min()))
print('    Max: '+str(I_camera_rgb[:,:,0].max()))
print('Max and min values of the green channel:')
print('    Min: '+str(I_camera_rgb[:,:,1].min()))
print('    Max: '+str(I_camera_rgb[:,:,1].max()))
print('Max and min values of the blue channel:')
print('    Min: '+str(I_camera_rgb[:,:,2].min()))
print('    Max: '+str(I_camera_rgb[:,:,2].max()))
```

    Max and min values of the image:
        Min: 7
        Max: 253
    Max and min values of the red channel:
        Min: 7
        Max: 253
    Max and min values of the green channel:
        Min: 7
        Max: 253
    Max and min values of the blue channel:
        Min: 7
        Max: 253
    

## <span style='color:Green'> Your turn:  </span>
We expect that the three color channels in this `I_camera_rgb` image are identical. Write a few lines of code to confirm that they are indeed identical.


```python
# Write your answer here
```

<details>
<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
print((I_camera_rgb[:,:,0] == I_camera_rgb[:,:,1]).all() and 
     (I_camera_rgb[:,:,1] == I_camera_rgb[:,:,2]).all())
```
</details>

## <span style='color:Green'> Your turn:  </span>
Display this new RGB image `I_camera_rgb`.


```python
# Write your answer here
```


<details>

<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
plt.figure(figsize=(10,10)) 
plt.imshow(I_camera_rgb) 
plt.axis('off') 
plt.title('Cameraman, gray2rgb')
plt.show()
```
</details>

### A note about why we might convert a grayscale image to a "color" image
We note, unsurprisingly, that the `I_camera_rgb` still appears as a grayscale image.  It just happens to have 3 identical color channels.  In the meantime, we may be using three times the space to represent this image, but the fact that it now has 3 color channels instead of 1 will allow us to use this image in neural network models that expect 3 channels.

## 3.3 Converting between color spaces
RGB is just one representation of color images, albeit the most common. Images may also be represented in [HSV](https://learn.leighcotnoir.com/artspeak/elements-color/hue-value-saturation/) (Hue, Saturation, Value) space, which is particularly useful when differentiating different areas of an image for tasks like segmentation. Hue refers to the visible color spectrum; saturation refers to color intensity - the lower the intensity, the closer the color is to grayscale; value refers to the relative luminescance (lightness or darkness) of a color.

The Scikit-Image library makes it easy to convert an image from one color space to another.


```python
I_pepper_hsv = skimage.color.rgb2hsv(I_pepper)                    # convert RGB image to HSV

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (15,12)) # open a new figure window with three subplots 
ax1.imshow(I_pepper)                                              # visualize RGB image
ax1.set_title('RGB')
ax1.axis('off')      
ax2.imshow(I_pepper_hsv[...,0])                                   # visualize hue channel
ax2.set_title('Hue Channel')
ax2.axis('off')      
ax3.imshow(I_pepper_hsv[...,1])                                   # visualize the saturation channel
ax3.set_title('Saturation Channel')
ax3.axis('off')   
ax4.imshow(I_pepper_hsv[...,2])                                   # visualize the value channel
ax4.set_title('Value Channel')
ax4.axis('off') 
plt.show()     
```


    
![png](output_97_0.png)
    


## 3.4 Cropping
Suppose that we have a network that expects a $256\times256$ greyscale image as input, i.e., the same **dimensionality** as the `cameraman.png` image.  If we want to input `peppers.png` we have two problems: it has three color channels and it is of spatial dimension $384\times512$.  We know that we can convert the RGB image to a grayscale image.  Now we have to figure out how to rescale the spatial dimensions.

If we crop the image, we choose some $256\times256$ pixels to retain.  For example if we kept the upper left corner of the image, we would have an image such as follows.


```python
I_pepper_gray_crop = I_pepper_gray[0:256,0:256]
plt.figure(figsize=(5,5)) 
plt.imshow(I_pepper_gray_crop,cmap='gray') 
plt.axis('off') 
plt.title('Peppers, gray, cropped')
plt.show()
```


    
![png](output_99_0.png)
    


### Cropping removes parts of the image
We note, unsurprisingly, that we have completely removed parts of the pepper image.

## <span style='color:Green'> Your turn:  </span>
We typically want to crop an image to isolate a particular object or feature. Try cropping the `peppers.png` image to just the garlic bulb.


```python
# Write your answer here

```


<details>

<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

We can review the image plot with axes labels to estimate the coordinates of the upper left and lower right corners of a box containing the garlic bulb.

```python
plt.figure(figsize=(5,5))                             # open figure window of size (5,5)
plt.imshow(I_pepper[230:320,410:512,:])               # refer to section 3.1 to estimate coordinates for garlic
plt.axis('off') 
plt.title('Garlic crop')
plt.show()

```
</details>

## 3.5 Resizing
What if the `peppers.png` image had fewer than 256 pixels?  What if we are unhappy with the loss of information associated with cropping?  Here we can use the `skimage.transform.resize` function to resize the image.  In the following syntax, we are asking the function to resize `I_pepper_gray` to a size $256\times256$ pixels.

We note that there are many options to the resize command, including specification of what form of interpolation to use, whether to anti-alias filter, and different means of specifying the scale of the output.  See `help(skimage.transform.resize)` for more information.  The syntax used here assumes defaults for all parameters (a good starting point) and specifies the desired output size as a tuple.


```python
I_pepper_gray_resize = skimage.transform.resize(I_pepper_gray,\
                                                (256,256))
plt.figure(figsize=(5,5)) 
plt.imshow(I_pepper_gray_resize,cmap='gray') 
plt.axis('off') 
plt.title('Peppers, gray, resized')
plt.show()
```


    
![png](output_105_0.png)
    


### Resizing can distort the aspect ratio
Here we note that we have distorted the aspect ratio of the original ```peppers.png``` image.  In some applications this may not matter and in others it might matter a great deal.  In general, depending on the application, you may want to consider a combination of resizing and cropping.  

## 3.6 Combining Cropping and Resizing

## <span style='color:Green'> Your turn:  </span>
Combine cropping and resizing to yield a $256\times256$ pixel grayscale peppers image that you think retains the majority of the original "intent" of the image.  Note--there is no "right" answer here.


```python
# Write your answer here
```

<details>

<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
# First, use cropping to create square peppers image 
# Then, resize to 256x256
# Peppers is 384x512, so take center 384x384 section
# (512-384)/2=64
I_pepper_gray_myresize = skimage.transform.resize(I_pepper_gray[:,65:65+384],(256,256))
plt.figure(figsize=(5,5)) 
plt.imshow(I_pepper_gray_myresize,cmap='gray') 
plt.axis('off') 
plt.title('Peppers, gray, cropped and resized')
plt.show()
```
</details>

## <span style='color:Green'> Your turn:  </span>
How would you reconfigure the `cameraman` image to be the $384\times512\times3$ size of `peppers`?  Would you find this an easier conversion to make or a more difficult one?  Note--there is no "right" answer here either.


```python
# Write your answer here
```

<details>

<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
# First, resize camera image to a square with largest dimension of peppers.
# Square to avoid distorting aspect ratio and largest dimension because we'll crop down to the shape of the peppers image.
I_camera_myresize = skimage.transform.resize(I_camera,(512,512))
# Next, crop vertical dimension to 384, keeping center of image
# (512-384)/2 = 64
I_camera_myresize = I_camera_myresize[65:65+384,:]
# finally, convert to color
I_camera_myresize = skimage.color.gray2rgb(I_camera_myresize)
plt.figure(figsize=(5,5)) 
plt.imshow(I_camera_myresize,cmap='gray') 
plt.axis('off') 
plt.title('Cameraman, color, resized and cropped')
plt.show()
```
</details>

# Section 4: Image Convolution
We will find that a key element of [convolutional neural networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) are **convolutional layers**.  It is thus critical that we understand the basics of **image convolution** and how to interpret those results. We will describe how convolution is used in CNNs in Tutorial 3.

Convolution is a mathematical function used to extract information from images, and is useful in CNNs to identify various features of an image. The inputs to this function are our image and a **filter kernel**, which is a small matrix of coefficients, often 3x3 or 5x5. The filter kernel dimensions are almost always square, and have an odd number of pixels in each dimension, so that the filter kernel has a well-defined center pixel. The output of the convolution function is a matrix, often of a similar size as the input.

The convolution process often follows these steps to construct an output matrix:
  1. Align the center of the filter kernel at pixel position (m, n) in the input image in order to calculate the output pixel at position (m,n)
  2. Multiply the corresponding elements of the filter kernel with the input image element-wise, 
  3. Add these values together to produce the desired output value
  4. Slide the filter kernel (typically 1-2 units) and repeat steps 1-3 to create a smoothed, moving average of the input images pixels processed through the filter kernel.

Parameters of the convolution function may vary in context of a particular problem.


### Convolutions on the edges of images

This process is not inherently well-defined for the edges of images, and there are multiple different ways that users can handle the edge cases of convolution. One approach is to only create output pixels in positions where the filter kernel fully overlaps with the input image. This results in an output matrix that is smaller than the input matrix. For example, if our input image is 4x4 pixels large, and our filter kernel is 3x3 pixels large, then our output kernel would be 2x2 pixels. 

Another approach is to "pad" the outside of the input image to ensure that the input and output matricies are the same size. Padding may be easily done with a constant value such as 0, but this approach introduces edge effects. For example, we may notice that the edges of the output matrix are darker than would be expected because they assuming that everything outside of the image is black. The approach used in the graphic shown below involves setting the pixel values outside of the image to that of the nearest pixel. This way, the pixel values are "extended" over the edge of the image, and the input image and output image are the same size.  

![convolution_gif](data/2D_Convolution_Animation.gif)

Note that the function we will use to produce image convolutions, `scipy.ndimage.convolve`, automatically produces an output matrix the same size as the input matrix; how that padding occurs can be controlled with the [mode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html#scipy.ndimage.convolve) argument; for simplicity, we leave the default option in this tutorial.

### Resources for Understanding Image Convolution

We recommend that readers who do not have a strong grasp of image convolution refer to these resources before proceeding:
  * [Video by Computerphile](https://www.youtube.com/watch?v=C_zFhWdM4ic), 
  * [Article by Hypermedia Image Processing](https://homepages.inf.ed.ac.uk/rbf/HIPR2/convolve.htm)

Interested readers may also wish to refer to [this more mathematically rigorous resource](https://www.youtube.com/watch?v=KuXjwB4LzSA) by 3blue1brown, which provides very useful visualizations of convolution for image processing starting at 8:32.

## 4.1 Low Pass (Smoothing) Filters
The values within the filter kernel affect the behavior of the convolution operation. Many commonly used image filters are defined in `scipy.ndimage`.  Here, we explore how to explicity define a filter kernel and convolve that kernel over an image with the `scipy.ndimage.convolve` function. This will prepare us to better interpret the convolutional layers in CNNs.

### 4.1.1 Define the filter kernels
We define two filters `h1` and `h2`.  These are very simple low pass (smoothing) filters where all the coefficients are equal in value and are normalized such that their sum is 1.


```python
h1 = 1/9.*np.ones((3,3))
h2 = 1/25.*np.ones((5,5))
print("Filter kernel h1:")
print(h1)
print(" ")
print("Filter kernel h2:")
print(h2)
```

    Filter kernel h1:
    [[0.11111111 0.11111111 0.11111111]
     [0.11111111 0.11111111 0.11111111]
     [0.11111111 0.11111111 0.11111111]]
     
    Filter kernel h2:
    [[0.04 0.04 0.04 0.04 0.04]
     [0.04 0.04 0.04 0.04 0.04]
     [0.04 0.04 0.04 0.04 0.04]
     [0.04 0.04 0.04 0.04 0.04]
     [0.04 0.04 0.04 0.04 0.04]]
    

### 4.1.2 Convolving the filter kernels with a grayscale image
We compute the filtered output by convolving the image `I_camera` with each of the filter kernels using `scipy.ndimage.convolve`.  We then visualize the filtered images.  We cast the image `I_camera` as a `float` to avoid integer arithmetic in the convolution operations.


```python
I_camera_h1 = scipy.ndimage.convolve(I_camera.astype(float),h1)
I_camera_h2 = scipy.ndimage.convolve(I_camera.astype(float),h2)
```


```python
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(I_camera,cmap='gray')
plt.axis('off')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(I_camera_h1,cmap='gray')
plt.axis('off')
plt.title('h1')
plt.subplot(1,3,3)
plt.imshow(I_camera_h2,cmap='gray')
plt.axis('off')
plt.title('h2')
plt.show()
```


    
![png](output_124_0.png)
    


## <span style='color:Green'> Your turn:  </span>
What effect has each of the filters `h1` and `h2` had on the image?

Write your answer here in plain text.

<details>

<summary>Click to reveal sample answer</summary> </br>

Both filters have blurred the image, with `h2` having a more pronounced effect (larger blurring) than `h1`. This is because each pixel in the output has an intensity value that is the average of the values of a 3x3 or 5x5 area of pixels in the original image.

</details>

### 4.1.3 Convolving the filter kernels with a color image

If we try to apply filters `h1` and `h2` to our color `I_pepper` image, we encounter a dimensionality error. We achieve the same blurring convolution to this image by redefining the filter kernels as three-dimensional.


```python
h1a = 1/9.*np.ones((3,3,1))   # duplicate our 3x3 matrix across three dimensions
h2a = 1/25.*np.ones((5,5,1))  

print("Filter kernel h1a:")
print(h1a)
print(" ")
print("Filter kernel h2a:")
print(h2a)
```

    Filter kernel h1a:
    [[[0.11111111]
      [0.11111111]
      [0.11111111]]
    
     [[0.11111111]
      [0.11111111]
      [0.11111111]]
    
     [[0.11111111]
      [0.11111111]
      [0.11111111]]]
     
    Filter kernel h2a:
    [[[0.04]
      [0.04]
      [0.04]
      [0.04]
      [0.04]]
    
     [[0.04]
      [0.04]
      [0.04]
      [0.04]
      [0.04]]
    
     [[0.04]
      [0.04]
      [0.04]
      [0.04]
      [0.04]]
    
     [[0.04]
      [0.04]
      [0.04]
      [0.04]
      [0.04]]
    
     [[0.04]
      [0.04]
      [0.04]
      [0.04]
      [0.04]]]
    


```python
I_pepper_h1a = scipy.ndimage.convolve(I_pepper.astype(float)/255,h1a)
I_pepper_h2a = scipy.ndimage.convolve(I_pepper.astype(float)/255,h2a)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(I_pepper)
plt.axis('off')
plt.title('Peppers')
plt.subplot(1,3,2)
plt.imshow(I_pepper_h1a)
plt.axis('off')
plt.title('h1a')
plt.subplot(1,3,3)
plt.imshow(I_pepper_h2a)
plt.axis('off')
plt.title('h2a')
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](output_131_1.png)
    


Notice that the filter kernel matrix has been "flipped" so it applies across all three color channels. 

## <span style='color:Green'> Your turn:  </span>
What would happen if we applied a cubed three-dimensional filter kernel to the image? Create new simple filters `c1` and `c2` and use them to convolute the image. How is the effect of these filter kernels different from the first set of kernels?
 


```python
# Write your answer here
```

<details>

<summary>Click to reveal sample answer</summary> </br>
Note: This code is not runnable since it is in a markdown cell. To run this snippet of code, copy and paste it into a code block. </br> </br>

```python
# First, create and print the filter kernels
c1 = 1/9.*np.ones((3,3,3))   # duplicate our 3x3 matrix across three dimensions
c2 = 1/25.*np.ones((3,5,5))  # note that we must place our dimensionality first in the numpy array
print("Filter kernel c1:")
print(c1)
print(" ")
print("Filter kernel c2:")
print(c2)

# Then, convolute and visualize the images
I_pepper_c1 = scipy.ndimage.convolve(I_pepper.astype(float)/255,c1)
I_pepper_c2 = scipy.ndimage.convolve(I_pepper.astype(float)/255,c2)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(I_pepper)
plt.axis('off')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(I_pepper_c1)
plt.axis('off')
plt.title('c1')
plt.subplot(1,3,3)
plt.imshow(I_pepper_c2)
plt.axis('off')
plt.title('c2')
plt.show()

Filters c1, c2 reduce the color contributions across the image.
```
</details>

## 4.2 Gaussian (Normalizing) Filters

### 4.2.1 Gaussian filter definition

### 4.2.2 Gaussian filters on a grayscale image

### 4.2.3 Gaussian filters on a color image

## 4.3 High Pass (Edge Enhancing) Filters

### 4.3.1 Define the filter kernels
Edge enhancing filters detect variations in pixel values as the filter moves, highlighting edges of objects in an image. Here we define two filters `h3` and `h4`.  These are very simple highpass (edge enhancing) filters called the [Sobel filters](https://en.wikipedia.org/wiki/Sobel_operator).


```python
h3 = [[-1,-2,-1],[0,0,0],[1,2,1]]
h4 = [[-1,0,1],[-2,0,2],[-1,0,1]]
```

We can visualize our kernels as follows, where white represents 0, dark blue represents -2, and dark red represents 2.


```python
plt.figure()
plt.subplot(1,2,1)
plt.imshow(h3,cmap="seismic")
plt.subplot(1,2,2)
plt.imshow(h4,cmap="seismic")
```




    <matplotlib.image.AxesImage at 0x16080fe4650>




    
![png](output_144_1.png)
    


### 4.3.2 Convolving the filter kernels with a grayscale image
We compute the filtered outputs by convolving the image `I_camera` with each of the filter kernels.  We again cast the image `I_camera` as a `float` to avoid integer arithmetic in the convolution operations.


```python
I_camera_h3 = scipy.ndimage.convolve(I_camera.astype(float),h3)
I_camera_h4 = scipy.ndimage.convolve(I_camera.astype(float),h4)
```

### A note on filtered images that have negative values
It is common that filtered images may end up with intensity values outside of the original range.  In this case, the image `I_camera` was in the range $[0,255]$.  If we look at the range of the filtered images, we find that the filtered images now span a much larger range:


```python
print('Max and min values of the h3 filtered image:')
print('    Min: '+str(I_camera_h3.min()))
print('    Max: '+str(I_camera_h3.max()))
print('Max and min values of the h4 filtered image:')
print('    Min: '+str(I_camera_h4.min()))
print('    Max: '+str(I_camera_h4.max()))
```

    Max and min values of the h3 filtered image:
        Min: -861.0
        Max: 893.0
    Max and min values of the h4 filtered image:
        Min: -900.0
        Max: 882.0
    

The Sobel filters are designed to approximate the first derivative of the image. When the filter is passing over a 3x3 section of the image that contains all the same pixel values, then the output at that location is zero. The filter `h3` will produce a large positive value when the pixel intensitives of the image are increasing from left to right, and will have a large negative value when the pixel intensities in the image are decreasing from left to right. Likewise, the filter `h4` will produce a large positive value when the pixel intensities in the image are increasing from top to bottom, and a large negative value when the pixel intensities are decreasing from top to bottom. If the change in intensity values is large enough, we can end up with values in our output array that are not in the range $[0,255]$.

**Note that the `plt.imshow()` function will automatically transform our data to fit the range $[0,1]$.** So, we don't need to worry about the range of our data when plotting.

We can get a better sense of the edge enhancement capabilities of `h3` and `h4` if we look only at the positive values. Recall that the positive values in the output image are associated with regions in the input image where the pixel intensity is increasing from left to right (for `h3`) or top to bottom (for `h4`).  Looking only at the positive values rather than the absolute value will be more consistent with the activation function we will use in convolutional neural networks.  We first clip all negative values in the images to zero and then visualize the filtered output.


```python
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(I_camera,cmap='gray')
plt.axis('off')
plt.title('Original')
plt.subplot(1,3,2)
I_camera_h3[I_camera_h3<0] = 0
plt.imshow(I_camera_h3,cmap='gray')
plt.axis('off')
plt.title('h3')
plt.subplot(1,3,3)
I_camera_h4[I_camera_h4<0] = 0
plt.imshow(I_camera_h4,cmap='gray')
plt.axis('off')
plt.title('h4')
plt.show()
```


    
![png](output_151_0.png)
    


Here we view the output images including both positive and negative. Notice that zero intensity manifests as medium gray now. Large positive are white. Large negative are black.


```python
I_camera_h3 = scipy.ndimage.convolve(I_camera.astype(float),h3)
I_camera_h4 = scipy.ndimage.convolve(I_camera.astype(float),h4)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(I_camera,cmap='gray')
plt.axis('off')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(I_camera_h3,cmap='gray')
plt.axis('off')
plt.title('h3')
plt.subplot(1,3,3)
plt.imshow(I_camera_h4,cmap='gray')
plt.axis('off')
plt.title('h4')
plt.show()
```


    
![png](output_153_0.png)
    


Here, we look at the absolute value of the filtered image. In these images, the white pixels represent regions in the original image where the pixel intensity _either_ increases or decreases dramatically.


```python
I_camera_h3 = scipy.ndimage.convolve(I_camera.astype(float),h3)
I_camera_h4 = scipy.ndimage.convolve(I_camera.astype(float),h4)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(I_camera,cmap='gray')
plt.axis('off')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(abs(I_camera_h3),cmap='gray')
plt.axis('off')
plt.title('h3')
plt.subplot(1,3,3)
plt.imshow(abs(I_camera_h4),cmap='gray')
plt.axis('off')
plt.title('h4')
plt.show()
```


    
![png](output_155_0.png)
    


When we focus only on the positive values of the filtered output, we see that the majority of the filtered image is now close to a value of 0 (i.e., black), and it is only at the edges of the image objects that we see lighter values. We see that `h3` has enhanced edges oriented in a horizontal direction and `h4` has enhanced edges oriented in a vertical direction.

### 4.3.3 Convolving the filter kernels with a color image

## 4.4 Fourier Filters

## 4.5 Other filters and transformations?
