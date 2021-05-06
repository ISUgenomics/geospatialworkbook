---
title: Premeeting
layout: single
author: Laura Boucheron
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

# Tutorial 4: Visualizing and Modifying DL Networks
## Laura E. Boucheron, Electrical & Computer Engineering, NMSU
### October 2020
Copyright (C) 2020  Laura E. Boucheron

This information is free; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This work is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this work; if not, If not, see <https://www.gnu.org/licenses/>.

## Overview
In this tutorial, we pick up with the trained MNIST Network from Tutorial 2 and explore some ways of probing the characteristics of the trained network to help us debug common pitfalls in adapting network architectures. 

This tutorial contains 5 sections:
  - **Section 0: Preliminaries**: some notes on using this notebook, how to download the image dataset that we will use for this tutorial, and import commands for the libraries necessary for this tutorial
  - **Section 1: Printing Characteristics of the CNN** how to print textual summaries of the CNN architecture
  - **Section 2: Visualizing Activations** how to filter an example image through thhe MNIST network and visualize the activations
  - **Section 3: Inputting New and Different Data to the Network** how to process new data to be compatible with the MNIST network and the effects of showing a non-digit image to the network
  - **Section 4: The VGG Network** an exploration of the VGG16 network.

There are a few subsections with the heading "**<span style='color:Green'> Your turn: </span>**" throughout this tutorial in which you will be asked to apply what you have learned.  

Portions of this tutorial have been taken or adapted from https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/ and the documentation at https://keras.io.

# Section 0: Preliminaries
## A Note on Jupyter Notebooks

There are two main types of cells in this notebook: code and markdown (text).  You can add a new cell with the plus sign in the menu bar above and you can change the type of cell with the dropdown menu in the menu bar above.  As you complete this tutorial, you may wish to add additional code cells to try out your own code and markdown cells to add your own comments or notes.

Markdown cells can be augmented with a number of text formatting features, including
  - bulleted
  - lists

embedded $\LaTeX$, monotype specification of `code syntax`, **bold font**, and *italic font*.  There are many other features of markdown cells--see the jupyter documentation for more information.

You can edit a cell by double clicking on it.  If you double click on this cell, you can see how to implement the various formatting referenced above.  Code cells can be run and markdown cells can be formatted using Shift+Enter or by selecting the Run button in the toolbar above.

Once you have completed (all or part) of this notebook, you can share your results with colleagues by sending them the `.ipynb` file.  Your colleagues can then open the file and will see your markdown and code cells as well as any results that were printed or displayed at the time you saved the notebook.  If you prefer to send a notebook without results displayed (like this notebook appeared when you downloaded it), you can select ("Restart & Clear Output") from the Kernel menu above.  You can also export this notebook in a non-executable form, e.g., `.pdf` through the File, Save As menu.

## Section 0.1 Downloading Images
Download the `my_digits1_compressed.jpg` and `latest_256_0193.jpg` files available on the workshop webpage.  We will use those images in Sections 3 and 4 of this tutorial.  

We will also use the `cameraman.png` and `peppers.png` files that we used in Tutorial 1 and the CalTech101 dataset that we used in Tutorial 2.

## Section 0.2a Import Necessary Libraries (For users using a local machine)
Here, at the top of the code, we import all the libraries necessary for this tutorial.  We will introduce the functionality of any new libraries throughout the tutorial, but include all import statements here as standard coding practice.  We include a brief comment after each library here to indicate its main purpose within this tutorial.

It would be best to run this next cell before the workshop starts to make sure you have all the necessary packages installed on your machine.

A few other notes:
 - After the first import of keras packages, you may get a printout in a pink box that states
```
Using Theano backend
```
or
```
Using TensorFlow backend
```
 - You may get one or more warnings complaining about various configs.  As long as you don't get any errors, you should be good to go.  You can, if you wish, fix whatever is causing a warning at a later point in time.  I find it best to copy and paste the error warning itself into a Google search and tack on the OS in which you encountered the error.  Seldom have I encountered an error that someone else hasn't encountered in my same OS.
 - The third to the last line in the following code cell imports the MNIST dataset.
 - The last two lines load the VGG16 network and the weights for that network trained on the ImageNet dataset.  The code below will load the VGG16 network, trained on ImageNet.  The first time this code is run, the trained network will be downloaded.  Subsequent times, the trained network will be loaded from the local disk.  This network is very large (528 MB) as we will see shortly, so it may take some time to download.  Generally, we would include the last line below as part of our code rather than imports, but we include it here to allow that download to complete before the workshop.


```python
import numpy as np # mathematical and scientific functions
import imageio # image reading capabilities
import skimage.color # functions for manipulating color images
import skimage.transform # functions for transforms on images
import matplotlib.pyplot as plt # visualization

# format matplotlib options
%matplotlib inline
plt.rcParams.update({'font.size': 16})

import keras.backend # information on the backend that keras is using
from keras.models import Model # a generic keras model class used to modify architectures
from keras.utils import np_utils # functions to wrangle label vectors
from keras.models import Sequential # the basic deep learning model
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D # important CNN layers
from keras.models import load_model # to load a pre-saved model (may require hdf libraries installed)
from keras.preprocessing.image import load_img # keras method to read in images
from keras.preprocessing.image import img_to_array # keras method to convert images to numpy array
from keras.applications.vgg16 import preprocess_input # keras method to transform images to VGG16 expected characteristics
from keras.applications.vgg16 import decode_predictions # keras method to present highest ranked categories
from keras.preprocessing.image import ImageDataGenerator # framework to input batches of images into keras

from keras.datasets import mnist # the MNIST dataset
from keras.applications import vgg16 # the VGG network
model_vgg16 = vgg16.VGG16(include_top=True,weights='imagenet') # download the ImageNet weights for VGG16
```

## Section 0.2b Build the Conda Environment (For users using the ARS HPC Ceres with JupyterLab)
Open a terminal from inside JupyterLab (File > New > Terminal) and type the following commands
```
source activate
wget https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/aiworkshop.yml
conda env create --prefix /project/your_project_name/envs/aiworkshop -f aiworkshop.yml
```
This will build the environment in one of your project directories. It may take 5 minutes to build the Conda environment.

See https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/setup/ for more information.

When the environment finishes building, select this environment as your kernel in your Jupyter Notebook (click top right corner where you see Python 3, select your new kernel from the dropdown menu, click select)

You will want to do this BEFORE the workshop starts.

## Section 0.4 Load Your Trained MNIST Model
At the end of Tutorial 3 we saved the trained MNIST model `model1` in `model1.h5`.  Here will load that model and we can pick up right where we left off.

If you were not able to save the model at the end of Tutorial 3, you can re-run the training of the MNIST model here before we start the rest of the tutorial.  For your convenience, below is the complete code that will load and preprocess the MNIST data and define and train the model.  You can cut and paste the code here into a code cell in this notebook and run it.
```
from keras.datasets import mnist
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
model1 = Sequential()
model1.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model1.add(Convolution2D(32, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(10, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model1.fit(X_train, Y_train, batch_size=64, epochs=1, verbose=1)
```


```python
model1 = load_model('model1.h5')
```

We have now loaded the trained MNIST model from Tutorial 3.  Since this is a new notebook, however, we do not have the actual MNIST data loaded. We copy the code for loading and preprocessing the MNIST data from the Tutorial 3 notebook.


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
```

# Section 1: Printing Characteristics of the CNN

## 1.1 The `summary` method
The `summary` method of a keras model will display a basic text summary of the CNN architecture with layer name, layer type, output shape, and number of parameters.


```python
model1.summary()
```

### A note about the `None` value in shape
The `None` value in the output shapes is used as a placehoder before the network knows how many samples it will be processing.

### Using tab-compete to explore attributes and methods
The tab-complete feature of `ipython` can be very helpful to explore the available attributes and methods for a variable.  There are useful attributes and methods for the model `model1` and for the layers in the model, accessed with the `layers` attribute of the model.

## **<span style='color:Green'> Your turn: </span>**
Explore the attributes and methods of the model variable `model1` by placing your cursor after the `.` and pressing tab.


```python
help(model1.summary)
```

## **<span style='color:Green'> Your turn: </span>**
Explore the attributes and methods of the layers in `model1`.  Change the index into `model1.layers` in the first code cell and run that cell to access the different CNN layers.  Then, place your cursor after the `.` and press tab.


```python
layer = model1.layers[0]
```


```python
help(layer.activation)
```

## 1.2 A layer-wise summary of input and output shapes
While the `summary` method of the model prints some useful information, there are additional pieces of information that can be very useful.  Below is a function definition  which will print a layer-wise summary of the input and output shapes.  This information can be very helpful in helping to understand and debug the workings (or non-workings) of the model.  This code loops over each layer in the model using the `layers` attribute of the model.


```python
def print_shapes(model):
    print('Layer Name\t\tType\t\tInput Shape\t\tOutput Shape\tTrainable')# print column headings
    for layer in model.layers:  # loop over layers
        lname = layer.name # grab layer name
        ltype = type(layer).__name__ # grab layer type
        ltype[ltype.find('/'):] # parse for only the last part of the string
        if ltype=='Conv2D': # print for convolutional layers
            print(lname+'\t\t'+ltype+'\t\t'+str(layer.input_shape)+'\t'+\
                  str(layer.output_shape)+'\t'+str(layer.trainable))
        elif ltype=='MaxPooling2D': # print for maxpool layers
            print(lname+'\t\t'+ltype+'\t'+str(layer.input_shape)+'\t'+\
                  str(layer.output_shape))
        elif ltype=='Flatten': # print for flatten layers
            print(lname+'\t\t'+ltype+'\t\t'+str(layer.input_shape)+'\t'+\
                  str(layer.output_shape))
        elif ltype=='Dense': # print for dense layers
            print(lname+'\t\t\t'+ltype+'\t\t'+str(layer.input_shape)+'\t\t'+\
                  str(layer.output_shape)+'\t'+str(layer.trainable))
```

We can print a summary of the input and output shapes by passing `model1` to the `print_shapes` function.


```python
print_shapes(model1)
```

## **<span style='color:Green'> Your turn: </span>**
Does this summary reconcile with the discussion in Tutorial 3 about the architecture of the MNIST model?  You might find it helpful to refer to the Tutorial 3 slides with the visualization of the MNIST network.

These shapes are consistent with the discussion in the Tutorial 2 slides.

## 1.3 A layer-wise summary of filter shape and parameters
Below is a function definition which will print a layer-wise summary of the filters and parameters.  This information can also be very helpful in helping to understand and debug the workings (or non-workings) of the model.


```python
def print_params(model):
    total_params = 0 # initialize counter for total params
    trainable_params = 0 # initialize counter for trainable params
    print('Layer Name\t\tType\t\tFilter shape\t\t# Parameters\tTrainable') # print column headings
    for layer in model.layers: # loop over layers
        lname = layer.name # grab layer name
        ltype = type(layer).__name__ # grab layer type
        ltype[ltype.find('/'):] # parse for only the last part of the string
        if ltype=='Conv2D': # print for convolutional layers
            weights = layer.get_weights()
            print(lname+'\t\t'+ltype+'\t\t'+str(weights[0].shape)+'\t\t'+\
                  str(layer.count_params())+'\t'+str(layer.trainable))
            if layer.trainable:
                trainable_params += layer.count_params()
            total_params += layer.count_params() # update number of params
        elif ltype=='MaxPooling2D': # print for max pool layers
            weights = layer.get_weights()
            print(lname+'\t\t'+ltype+'\t---------------\t\t---')
        elif ltype=='Flatten': # print for flatten layers
            print(lname+'\t\t'+ltype+'\t\t---------------\t\t---')
        elif ltype=='Dense': # print for dense layers
            weights = layer.get_weights()
            print(lname+'\t\t\t'+ltype+'\t\t'+str(weights[0].shape)+'\t\t'+\
                  str(layer.count_params())+'\t'+str(layer.trainable))
            if layer.trainable:
                trainable_params += layer.count_params()
            total_params += layer.count_params() # update number of params
    print('---------------')
    print('Total trainable parameters: '+str(trainable_params)) # print total params
    print('Total untrainable parameters: '+str(total_params-trainable_params))
    print('Total parameters: '+str(total_params))
```

We can print a summary of the input and output shapes by passing `model1` to the `print_shapes` function.


```python
print_params(model1)
```

### Check out the total number of parameters!!!
The MNIST model that we trained yesterday has **more than 600,000 parameters** that it learned during training!  We will see later in this tutorial that is is actually a very small network.

We note a few things about the number of parameters per layer:
  - The second conv layer has a lot more parameters than the first.  That is due to the fact that the second conv layer filters across all 32 channels of the activations from the first conv layer.  
  - The max pool and flatten layers don't have any parameters.
  - The fully connected (dense) layers are the source of a large proportion of the total parameters in this network.

## **<span style='color:Green'> Your turn: </span>**
What implications does the number of trainable parameters per layer have on transfer learning and decisions about which layers to freeze?  There is a code cell below prepopluated with the code to clone `model1` and to freeze all but the last layer for you to modify and explore using the `print_params` and `print_shapes` functions defined above.


```python
model1_clone = keras.models.clone_model(model1)
model1_clone.set_weights(model1.get_weights())

for layer in model1_clone.layers[:-1]:
    layer.trainable=False

print_params(model1_clone)
print('')
print_shapes(model1_clone)
```

### A note on number of parameters per layer
This cell describes a few more details of how you can reconcile the filter shape and the total number of parameters.  The uninterested reader can skip this section without impeding their ability to complete the rest of the tutorial.

Recall that a basic neuron has a set of weights and a bias.  The parameters that must be learned in deep learning layers include both the weights and biases.  We break down the computation for convolutional layers and for fully connected (dense) layers.  We will use the same notation as from the Tutorial 3 slides.

#### Convolutional layers:
Each filter in a convolutional layer has $K\cdot K\cdot C$ weights and one bias, where $K$ is the kernel size and $C$ is the number of channels.  Thus we have a total of $M_{conv}\cdot K\cdot K\cdot C$ weights and $M_{conv}$ biases, where $M_{conv}$ is the number of filters in the layer.  This means a total number of trainable parameters of $M_{conv}(K^2C+1)$.

#### Fully connected layers:
Each node in a fully connected layer is connected to every node in the previous layer and we thus have $M_{FC}^{(i-1)}$ weights and one bias per node, where $M_{FC}^{(i-1)}$ is the number of weights in the previous fully connected (or flattened) layer.  Thus we have a total of $M_{FC}^{(i)}\cdot M_{FC}^{(i-1)}$ weights and $M_{FC}^{(i)}$ biases, where $M_{FC}^{(i)}$ is the number of nodes in the current fully connected layer.  This means we have a total number of trainable parameters of $M_{FC}^{(i)}(M_{FC}^{(i-1)}+1)$.

# Section 2 Visualizing Activations
In this section we will explore means to visualize the activations in different layers throughout the network.

## Section 2.1 Wrangling the example input image dimensions
The responses (activations) for each filter in a layer can be computed by sending an example image through the network and requesting that the network report the output at the layer of interest (rather than at the output layer).

First, we need to choose an image to filter through the network.  It is this image for which the activations will be computed.  We begin here with the first test image.  Recall that the network expects a tensor in the form samples$\times28\times28\times1$.  In this case, we'll be providing only one sample, so we need our input to be $1\times28\times28\times1$.

The following code reshapes the zeroth test image with is shape $28\times28\times1$ into a tensor of shape $1\times28\times28\times1$ where the leading dimension of 1 is just wrangling the dimensionality to the samples$\times28\times28\times1$ format expected of an input tensor.


```python
X_example = X_test[0].reshape(1,28,28,1)
```

We can plot the image to give us an idea of the appearance of the original image.  This will allow us to better analyze the filtered images that we will see when we plot the activations.  Since we have added some extra dimensions to this image, we use the `np.squeeze` function to remove those dimensions with only one entry before sending it to `plt.imshow`.  In this case, the `np.squeeze` function returns a $28\times28$ `numpy` array.


```python
plt.figure()
plt.imshow(np.squeeze(X_example),cmap='gray')
plt.axis('off')
plt.show()
```

## **<span style='color:Green'> Your turn: </span>**
Explore the dimensionalities of `X_example` relative to `X_test[0]` and `np.squeeze(X_example)`.


```python
print('The dimensions of X_example are '+str(X_example.shape))
print('The dimensions of X_test[0] are '+str(X_test[0].shape))
print('The dimensions of np.squeeze(X_example) are '+str(np.squeeze(X_example).shape))
```

## Section 2.2 Modify `model` to output activations after the first conv layer
Now, we modify our `model1` to output the activations after the first convolutional layer.  We use the generic `Model` class from `keras` and specify the same inputs as `model1`, but specify the output to be after the zeroth layer.  This is where the `print_shapes` and `print_params` functions can be very helpful to determine which layer you actually want to specify as output. We call this new model `model1_layer0` to designate that it is the same as `model`, but outputing information after layer 0, i.e., the first convolutional layer.


```python
model1_layer0 = Model(inputs=model1.inputs, outputs=model1.layers[0].output)
```

Now if we ask for the prediction of the model for `X_example`, the model will output the activations at the first convolutional layer instead of the activations at the final softmax layer.


```python
layer0_activations = model1_layer0.predict(X_example)
```

## **<span style='color:Green'> Your turn: </span>**
Based on our textual summaries of this network, we expect that the output should be of shape $1\times26\times26\times32$.  Check the dimensionality and variable type of `layer0_activations`, as well as the intensity range.


```python
print('The shape is: '+str(layer0_activations.shape))
print('The variable type is: '+str(layer0_activations.dtype))
print('The intensity range is ['+str(layer0_activations.min())+','+\
      str(layer0_activations.max())+']')
```

## Section 2.3 Visualizing the 32 activations of the first conv layer
We can loop over the 32 activations and plot each.  `plt.imshow` will, by default, choose an intensity range to match that of the input image.  This can make it difficult to compare between activations: a bright pixel in one image might actually be more activated than in another.  We can force the plots to be on the same scale of intensities by passing the minimum and maximum intensities to `plt.imshow` using the `vmin` and `vmax` options.


```python
plt.figure(figsize=(7,7))
min_int = layer0_activations.min() # find min intensity for all activations
max_int = layer0_activations.max() # find max intensity for all activations
subplot_rows = np.ceil(np.sqrt(layer0_activations.shape[-1])).astype(int) # determine subplots rows
for f in range(0,layer0_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer0_activations[:,:,:,f]),cmap='gray',\
               vmin=min_int,vmax=max_int) # plot activations
    plt.axis('off')
```


```python
plt.figure(figsize=(7,7))
min_int = layer0_activations.min() # find min intensity for all activations
max_int = layer0_activations.max() # find max intensity for all activations
subplot_rows = np.ceil(np.sqrt(layer0_activations.shape[-1])).astype(int) # determine subplots rows
for f in range(0,layer0_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer0_activations[:,:,:,f]),cmap='gray',\
               vmin=min_int,vmax=max_int) # plot activations
    plt.axis('off')
```

We see that there are some filters that respond to the entire digit, some that respond only the the horizontal stroke of the digit, some that respond only to the vertical stroke, and we may even see some that don't respond at all.  These filters that don't respond may be tuned for shapes (e.g., curves) that don't appear in the digit 7.

Here is how you can get at the actual filter weights in any given layer.  This example is for the first convolutional layer.  The zeroth element in the weights attribute are the filter weights an the first element are the biases.  Note that the weights come out as a tensorflow tensor `tf.Tensor`.  This variable type could be formatted into a differen variable type if desired.


```python
layer0_weights = model1.layers[0].weights
filters = layer0_weights[0]
for i in range(32):
    print(filters[:,:,:,i])
```

## **<span style='color:Green'> Your turn: </span>**
Explore the change in visualization if you do not use the `vmin` and `vmax` options above.  For your convenience, the code from the cell above is copied below for you to modify.


```python
plt.figure(figsize=(7,7))
subplot_rows = np.ceil(np.sqrt(layer0_activations.shape[-1])) # determine subplots rows
for f in range(0,layer0_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer0_activations[:,:,:,f]),cmap='gray')
    plt.axis('off')
```

## Section 2.4 Modify model to visualize activations after the second conv layer
We can look at the activations of the second convolutional layer with a simple modification of the code above.


```python
model1_layer1 = Model(inputs=model1.inputs, outputs=model1.layers[1].output)
layer1_activations = model1_layer1.predict(X_example)
plt.figure(figsize=(7,7))
min_int = layer1_activations.min() # find min intensity for all activations
max_int = layer1_activations.max() # find max intensity for all activations
subplot_rows = np.ceil(np.sqrt(layer1_activations.shape[-1])) # determine subplots rows
for f in range(0,layer1_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer1_activations[:,:,:,f]),cmap='gray',\
               vmin=min_int,vmax=max_int) # plot activations
    plt.axis('off')
```

We see that the second layer filters have gotten more specific in the structures to which they are responding.  This is consistent with what we know about the hierarchical nature of feature learning in CNNs.

## Section 2.5 Modify model to visualize activations after the max pool layer
## **<span style='color:Green'> Your turn: </span>**
Modify the code above to visualize the output after the max pool layer.  For your convenience, the code from the cell above is copied below for you to modify.  Note--you probably want to define a new variable for this model to avoid overwriting the other models from above.


```python
model1_layer2 = Model(inputs=model1.inputs, outputs=model1.layers[2].output)
layer2_activations = model1_layer2.predict(X_example)
plt.figure(figsize=(7,7))
min_int = layer2_activations.min() # find min intensity for all activations
max_int = layer2_activations.max() # find max intensity for all activations
subplot_rows = np.ceil(np.sqrt(layer2_activations.shape[-1])) # determine subplots rows
for f in range(0,layer2_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer2_activations[:,:,:,f]),cmap='gray',\
               vmin=min_int,vmax=max_int) # plot filter coeffs
    plt.axis('off')
```

We note that the activations of the max pool layer are simply lower resolution representations of the second convolutional layer activations.

## Section 2.6 Modify model to visualize activations of the fully connected layers
While the output of the flattened and fully connected layers are not images, we can visualize the activations by treating them like a one-row image.  This can give us some insight into which neurons are responding the most to the digit 7.
### The flattened layer
Since the dimensions of the flattened layer are $1\times4608$, we need to "stretch" out the pixels to actually be able to see them.  We use the `aspect` parameter in `plt.imshow` to do this.


```python
model1_layer3 = Model(inputs=model1.inputs, outputs=model1.layers[3].output)
layer3_activations = model1_layer3.predict(X_example)
plt.figure(figsize=(20,20))
plt.imshow(layer3_activations,cmap='gray',aspect=50) # plot filter coeffs
plt.axis('off')
plt.show()
```

This visualization may not be particularly elucidating, but we include it here for the sake of completeness.  Note that this $1\times4608$ vector of activations is just a reshaping of the $32\times12\times12=4608$ pixels in the max pool activations.

### The first fully connected layer
Since the dimensions of the first fully connected layer is only $1\times128$, we don't need to mess with the aspect ratio of the `plt.imshow` visualization.


```python
model1_layer4 = Model(inputs=model1.inputs, outputs=model1.layers[4].output)
layer4_activations = model1_layer4.predict(X_example)
plt.figure(figsize=(20,20))
plt.imshow(layer4_activations,cmap='gray') # plot filter coeffs
plt.axis('off')
plt.show()
```

This visualization can be interpreted in some sense as some aggregate of features that this layer is cueing on from the max pool layer.  We expect that different of these neurons will activate for different digits.

### The second fully connected layer
Note that the second fully connected layer is also the softmax output layer.  We leave the axis labels on here for more easy determination of which digit(s) the network is claiming probability for.  We also put grid lines on the image to even better delineate the different digits.


```python
model1_layer5 = Model(inputs=model1.inputs, outputs=model1.layers[5].output)
layer5_activations = model1_layer5.predict(X_example)
plt.figure(figsize=(20,20))
plt.imshow(layer5_activations,cmap='gray') # plot filter coeffs
#plt.axis('off')
plt.grid('on')
plt.show()
```

We notice that the output here is a very high confidence in the digit 7 and very little in other digits.  This is consistent with the interpretation of the softmax output layer if we were to look at the actual probability values.


```python
print(layer5_activations)
```

## **<span style='color:Green'> Your turn: </span>**
Explore the activations of the network for other input images.  For your convenience, code cells from above have been copied here for you to modify.

### Defining a specific input image


```python
X_example = X_test[3].reshape(1,28,28,1)
print('Original image')
plt.figure()
plt.imshow(np.squeeze(X_example),cmap='gray')
plt.axis('off')
plt.show()
```

### Output of the first convolutional layer


```python
print('First convolutional layer')
model1_layer0 = Model(inputs=model1.inputs, outputs=model1.layers[0].output)
layer0_activations = model1_layer0.predict(X_example)
plt.figure(figsize=(7,7))
min_int = layer0_activations.min() # find min intensity for all activations
max_int = layer0_activations.max() # find max intensity for all activations
subplot_rows = np.ceil(np.sqrt(layer0_activations.shape[-1])) # determine subplots rows
for f in range(0,layer0_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer0_activations[:,:,:,f]),cmap='gray',\
               vmin=min_int,vmax=max_int) # plot filter coeffs
    plt.axis('off')
```

### Second convolutional layer


```python
print('Second convolutional layer')
model1_layer1 = Model(inputs=model1.inputs, outputs=model1.layers[1].output)
layer1_activations = model1_layer1.predict(X_example)
plt.figure(figsize=(7,7))
min_int = layer1_activations.min() # find min intensity for all activations
max_int = layer1_activations.max() # find max intensity for all activations
subplot_rows = np.ceil(np.sqrt(layer1_activations.shape[-1])) # determine subplots rows
for f in range(0,layer1_activations.shape[-1]): # loop over filters
    plt.subplot(subplot_rows,subplot_rows,f+1) # choose current subplot
    plt.imshow(np.squeeze(layer1_activations[:,:,:,f]),cmap='gray',\
               vmin=min_int,vmax=max_int) # plot filter coeffs
    plt.axis('off')
```

### The flattened layer


```python
print('The flattened layer')
model1_layer3 = Model(inputs=model1.inputs, outputs=model1.layers[3].output)
layer3_activations = model1_layer3.predict(X_example)
plt.figure(figsize=(20,20))
plt.imshow(layer3_activations,cmap='gray',aspect=50) # plot filter coeffs
plt.axis('off')
plt.show()
```

### The first fully connected layer


```python
print('First fully connected layer')
model1_layer4 = Model(inputs=model1.inputs, outputs=model1.layers[4].output)
layer4_activations = model1_layer4.predict(X_example)
plt.figure(figsize=(20,20))
plt.imshow(layer4_activations,cmap='gray') # plot filter coeffs
plt.axis('off')
plt.show()
```

### The second fully connected layer (the output layer)


```python
print('Second fully connected layer (output layer)')
model1_layer5 = Model(inputs=model1.inputs, outputs=model1.layers[5].output)
layer5_activations = model1_layer5.predict(X_example)
plt.figure(figsize=(20,20))
plt.imshow(layer5_activations,cmap='gray') # plot filter coeffs
#plt.axis('off')
plt.grid('on')
plt.show()
```

# Section 3 Inputting New and Different Data to the Trained Network
In this section, we'll explore the use of this trained network to operate on new data.  We will use the `my_digits1_compressed.jpg` image provided as part of this tutorial.


```python
I = imageio.imread('my_digits1_compressed.jpg')

plt.figure(figsize=(7,7))
plt.imshow(I,cmap='gray')
plt.show()
```

Note that image is an RGB image of 10 handwritten digits.  You will wrangle this image into a format suitable for input to the MNIST network in this section.

## **<span style='color:Green'> Your turn: </span>**

You should have noticed that image is an RGB image of 10 handwritten digits.  Use what you have learned in Tutorials 1, 2 and 3 to extract each of those 10 digits from the image and get it in the correct form to input to the MNIST network.

As a reminder you probably want to pay attention to:
  - RGB versus gray
  - Variable type
  - Intensity range (Hint--you can invert the intensities and have light digits on a dark background by subtracting the image from the maximum intensity.)
  - Cropping indices (Hint--rows 295 through 445 and columns 1160 through 1310 will crop the digit 0)
  - Resizing
  - Correct tensor dimensions: recall that the network expects a tensor in the form samples$\times28\times28\times1$.  In this case, you'll be providing only one sample, so you will need your input to be $1\times28\times28\times1$.

Use your extracted digits as input to the MNIST network `model1`.  Does the network predict the correct label for the digit?  What does the predicted softmax label tell you about the confidence in the prediction for this new image?


```python
I_gray = skimage.color.rgb2gray(I) # convert to grayscale
I_gray = 1-I_gray # invert colors
I0 = I_gray[295:445,1160:1310] # crop out the digit 0
I0 = skimage.transform.resize(I0,(28,28)) # resize to 28x28

I1 = I_gray[355:505,2035:2190]
I1 = skimage.transform.resize(I1,(28,28))

I2 = I_gray[425:625,2900:3100]
I2 = skimage.transform.resize(I2,(28,28))

I3 = I_gray[465:665,3775:3975]
I3 = skimage.transform.resize(I3,(28,28))

I4 = I_gray[1250:1400,1140:1290]
I4 = skimage.transform.resize(I4,(28,28))

I5 = I_gray[1270:1460,1950:2140]
I5 = skimage.transform.resize(I5,(28,28))

I6 = I_gray[1365:1515,2855:2995]
I6 = skimage.transform.resize(I6,(28,28))

I7 = I_gray[1375:1565,3705:3895]
I7 = skimage.transform.resize(I7,(28,28))

I8 = I_gray[1890:2090,1100:1300]
I8 = skimage.transform.resize(I8,(28,28))

I9 = I_gray[1915:2100,1890:2075]
I9 = skimage.transform.resize(I9,(28,28))

plt.figure()
plt.imshow(I9,cmap='gray')
plt.show()
```


```python
print('Actual 0')
digit = I0.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 1')
digit = I1.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 2')
digit = I2.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 3')
digit = I3.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 4')
digit = I4.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 5')
digit = I5.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 6')
digit = I6.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)

print('Actual 7')
digit = I7.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 8')
digit = I8.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 9')
digit = I9.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
```

We see that the network is a bit less certain about this digit than the digits we looked at in Tutorial 2.  This is not surprising considering that this data came from a completely different source.

The following code is just demonstration that you can combine the resize and reshape into one command if you so desire.


```python
test = I_gray[295:445,1160:1310] # crop out the digit 0
test = skimage.transform.resize(test,(1,28,28,1)) # resize to 28x28
print(test.shape)
```

This is a small demonstration of the ability for this network to correctly classify data from an entirely new source.  We note, however, that the preparation of the data is critical for this success.  If we were to pre-process the data in a manner not designed for the MNIST network, we might get very different results.  

## **<span style='color:Green'> Your turn: </span>**
Repeat the above analysis, but keep the digit as dark on a light background.


```python
I0 = 1 - I0
I1 = 1 - I1
I2 = 1 - I2
I3 = 1 - I3
I4 = 1 - I4
I5 = 1 - I5
I6 = 1 - I6
I7 = 1 - I7
I8 = 1 - I8
I9 = 1 - I9

plt.figure()
plt.imshow(I9,cmap='gray')
plt.show()
```


```python
print('Actual 0')
digit = I0.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 1')
digit = I1.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 2')
digit = I2.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 3')
digit = I3.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 4')
digit = I4.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 5')
digit = I5.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 6')
digit = I6.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)

print('Actual 7')
digit = I7.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 8')
digit = I8.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
print('')

print('Actual 9')
digit = I9.reshape(1, 28, 28, 1) # reshape to 1x28x28x1
Y = model1.predict(digit,verbose=1) # predict label
print(Y)
y = np.argmax(Y)
print(y)
```

We see that the network has now incorrectly classified each digit as a 3 or 8 (your mileage may vary depending on exactly how your network converged).

We note that the network is approximately equally certain that this digit is a 0, 3, 6, or 8.  This uncertainty is common in situations where the network is exposed to data of a form it has never seen before.

By keeping the background light and the digit dark, the network interprets the background as the digit.  It is not actually processing the information in the same way we interpret this image as the digit 0.  

# Section 4: The VGG16 Network
In this section, we will use what we have learned about deep learning and image processing to explore a common CNN network called VGG16.  The VGG network is described in this paper: https://arxiv.org/abs/1409.1556 and is a common architecture for image classification.  This network was trained to classify 1000 categories (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  

Note--VGG16 is some 528 MB to download.  This is a typical size for state-of-the-art CNNs.  We actually downloaded these weights when we imported libraries (local machines) or activated the conda environment (HPC).  

## Section 4.1 Loading the VGG Network Trained on ImageNet
`keras` includes functions to load a VGG16 network with additional options to download a "pretrained" network--i.e., one that has been trained on ImageNet.  ImageNet is a database of millions of images (see http://www.image-net.org/) spanning thousands of categories.

The code below will load the VGG16 network, trained on ImageNet.  The first time this code is run, the trained network will be downloaded.  Subsequent times, the trained network will be loaded from the local disk.  This network is very large as we will see shortly, so it may take some time to download.

Similar to how we saved our MNIST model at the end of Tutorial 3 and loaded it at the beginning of this tutorial, we are loading a VGG16 model that has already been trained on the millions of images and 1000 categories of ImageNet.  It is no trivial task to train a network the size of VGG16 (weeks on a multiple GPUs), so we want to leverage the work that has already been done.


```python
model_vgg = vgg16.VGG16(include_top=True,weights='imagenet')
```

We can use `print_shapes` and `print_params` to explore the structure of the VGG16 model.


```python
print_shapes(model_vgg)
```


```python
print_params(model_vgg)
```

There are over 138 million parameters in this network!!!  It has many more layers than the simple MNIST network that we have been working with.

## Section 4.2: Classification Capabilities of the VGG16 Network
So, what happens if we show a new image to this network?  Let's see what happens if we show it the `cameraman.png` image.  
This example is adapted from https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e.

In the code below, we are leveraging many built-in `keras` functions, including
  - `load_img` from `keras` to read in an image while resizing to the expected input size of $224\times224$
  - `preprocess_input` specific to the VGG16 model in `keras` to scale the intensities to the expected range.  I'm unsure of the details of this, but I do know that it's a process more complicated than a simple intensity scaling (since it results in negative intensities).  Further details are likely in the VGG paper (https://arxiv.org/abs/1409.1556), but can be difficult to find sometimes.
  - `decode_predictions` specific to the VGG16 model in `keras` to find the top three confidences and map those to the class labels


```python
# Example adapted from https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

# load an image from file
image = load_img('cameraman.png', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model_vgg.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
for k in range(0,3):
    labelk = label[0][k]
    # print the classification
    print('%s (%.2f%%)' % (labelk[1], labelk[2]*100))
```

We notice that the network's performance on this image is pretty decent, specifying that it believes the image is one of a "tripod."  There is not a "cameraman" category in ImageNet (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), so the network chose the most likely category from the available ones.

Here, we repeat the above for the `peppers.png` image.  


```python
# Example adapted from https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

# load an image from file
image = load_img('peppers.png', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model_vgg.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
for k in range(0,3):
    labelk = label[0][k]
    # print the classification
    print('%s (%.2f%%)' % (labelk[1], labelk[2]*100))
```

We find that the network's performance on this model is also very good, specifying that the image is of a "bell pepper".  

Both the cameraman and peppers image contain objects similar to those that the VGG16 network encountered in the ImageNet database.  As such, it does a remarkably good job of classifying those images.  

What happens if the network encounters something really different than what it's seen before?  What if you show it an image of something not included in the 1000 classes of objects?  The image `latest_256_0193.jpg` image is an image of the Sun at a wavelength of 193 angsgtroms from NASA's Solar Dynamics Observatory satellite (https://sdo.gsfc.nasa.gov/data/)


```python
# Example adapted from https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

# load an image from file
image = load_img('latest_256_0193.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model_vgg.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
for k in range(0,3):
    labelk = label[0][k]
    # print the classification
    print('%s (%.2f%%)' % (labelk[1], labelk[2]*100))

I = imageio.imread('latest_256_0193.jpg')
plt.figure(figsize=(5,5))
plt.imshow(I)
plt.title('latest_256_0193.jpg')
plt.show()
```

The network is torn between classifying this image of the sun as a "tick", "French loaf", or a "nail".  Hmmm.... Things aren't looking so good anymore.  But we must remember that we never showed the network ground truth of the sun in 193 angstroms during training.  We can't really expect that it can jump to that conclusion.

## **<span style='color:Green'> Your turn: </span>**
What class does the VGG16 network think your data belong to?  If you don't have data with you, you can peruse the internet for images of something that you want to try classifying with the network.  Just save the image to the same directory as this notebook and use the code above to classify it.

## **<span style='color:Green'> Your turn: </span>**
Using an image of your choice, use the methods we learned above to explore the workings of the VGG16 model.

## 4.3 Transfer Learning on the VGG16 Architecture

Here we show an example of how to perform transfer learning on the VGG16 architecture using the CalTech101 dataset as the new input data.

Similar to how we modified the model to output activations at certain layers bove, we can truncate the VGG16 model to any desired layer and then add on additional layers at our discretion.  In this case, since we noted relatively good performance of the basic VGG16 architecture on images of similar appearance to the object categories in ImageNet, we expect that we won't need to change too much of the VGG16 architecture.  In this example, we choose to modify only the final prediction layer.

It is also common practice to retrain on all the fully connected layers.  Generally speaking, the further in appearance your data are from the ImageNet images, the further back in the architecture you probably want to retrain.  

In the following code, we keep the entire VGG16 architecture up until the last fully connected layer (which also happens to be the output layer) and define a new model `model2_vgg` which consists only of those layers we want to keep.


```python
model2_vgg = Model(inputs=model_vgg.input,outputs=model_vgg.layers[-2].output)
```

Now we need to add in at least a new prediction layer as the final layer.  We could also add additional layers within the network if we thought they were needed.  Since the CalTech101 dataset has 101 classes, we need the final fully connected layer to have 101 nodes and a softmax activation.


```python
new_output = model2_vgg.output # take the output as currently defined
new_output = Dense(101,activation='softmax')(new_output) # operate on that output with another dense layer
model2_vgg = Model(inputs=model2_vgg.input,outputs=new_output) # define a new model with the new output
```

Up to this point we have defined a new architecture, where we amputated the final fully connected layer and stitched back on a new one.  If we look at the layers of this new model using the modified `print_params` function,


```python
print_params(model2_vgg)
```

we see that all the layers have attribute `trainable` set to `True`.  This means that if we were to train this new model, we will train all 138 million parameters.  We don't want to do this.

We want to "freeze" the parameters for all layers except that new one that we added.  To do this, we set the `trainable` attribute of all layers we wish to freeze to `False`.  


```python
for layer in model2_vgg.layers[:-1]:
    layer.trainable=False
```

Now if we check the trainability of the layers, we see that all layers except the final layer are frozen, and we will be training (learning) only some 413,797 parameters.


```python
print_params(model2_vgg)
```

Now we need to point out new architecture at the CalTech101 data in order to train that final layer.  We again use built-in functions in `keras` to handle the flow of data through the network.  With MNIST we could fit all the training images in one array in memory and grab batches from there.  With a larger dataset like CalTech101, we begin to lose our ability to fit everything in memory and instead will read image from the specified directory into batches at training time.

The code below uses an `ImageDataGenerator` class from keras and defines the preprocessing applied to that image as the `preprocess_input` function we already used above.  Recall that this is a preprocessing function defined specifically for the VGG16 network.

Next, we use the `flow_from_directory` function of the `ImageDataGenerator` to define a flow of images from a specified directory. It is assumed that the specified directory contains a subdirectory for each class.  Other options specified for this function are
  - `target_size` which specifies the spatial dimensions to which all input data will be resized
  - `color_mode` which specifies that these images should be treated as RGB images.  These built-in functions are nice to use since they (often, not always) have niceties associated with taking care of things like converting grayscale images to the correct dimensionality.
  - `batch_size` the number of images to process per batch in the training
  - `class_mode` which specifies that this is a multi-class classification problem
  - `shuffle` which specifies that the batches will be selected randomly rather than in alphanumerical order

There are other options available, see `help(train_datagen.flow_from_directory)`.


```python
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('101_ObjectCategories',\
                                                    target_size=(224,224), color_mode='rgb',\
                                                    batch_size=32, class_mode='categorical',\
                                                    shuffle=True)
```

Now we compile the model and specify the same options as we used for the MNIST network.


```python
model2_vgg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Now we can train the model.  Since we don't have the entire training data to feed to the training, we instead invoke the `fit_generator` method which can utilize the `train_generator` we define above.  Additionally, the `fit_generator` takes a `steps_per_epoch` option rather than a `batch_size` option.  We define the `steps_per_epoch` as the number of images over which we will train divided by the batch size.

This code will take a while.  It took about 30 minutes per epoch on 6 4.8 GHz Intel i9 processors.  In most applications, we would train this over multiple epochs to boost the accuracy even higher.  Here we train for one epoch to limit the computational time.  After one epoch, the network reported accuracy above 80%.


```python
step_size_train = train_generator.n//train_generator.batch_size # the // does a floor after division
model2_vgg.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=2, verbose=1)
```

## **<span style='color:Green'> Your turn: </span>**
Using an image of your choice from CalTech101, use the methods we learned above to explore the workings of the new transfer-learned VGG16 model.

Since the `decode_predictions` method is specific to VGG16, it will complain if asked to operate on the outputs of this modified architecture.  Instead, we will determine the class label by using the argmax of the output `yhat` and by asking the model for a dictionary of the labels with `train_generator.class_indices`.


```python
# Example adapted from https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

# load an image from file
image = load_img('101_ObjectCategories/emu/image_0001.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model2_vgg.predict(image)
# convert the probabilities to class labels
label_list_str_to_num = train_generator.class_indices
label_list_num_to_str = {v: k for k, v in label_list_str_to_num.items()}
ksorted = np.argsort(yhat[0])
# retrieve the most likely result, e.g. highest probability
for k in ksorted[-1:-3:-1]:
    # print the classification
    print('%s (%.2f%%)' % (label_list_num_to_str[k], yhat[0][k]))
```


```python

```
