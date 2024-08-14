---
title: "Raster Vision Tutorial Series Part 2: Overview of Deep Learning for Imagery and the Raster Vision Pipeline"
layout: single
author: Noa Mills
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

# Semantic Segmentation of Aerial Imagery with Raster Vision 
## Part 2: Overview of Deep Learning for Imagery and the Raster Vision Pipeline

This tutorial series walks through an example of using [Raster Vision](https://rastervision.io/) to train a deep learning model to identify buildings in satellite imagery.<br>

*Primary Libraries and Tools*:

|Name|Description|Link|
|-|-|-|
| `Raster Vision ` | Library and framework for geospatial semantic segmentation, object detection, and chip classification in python| https://rastervision.io/ |
| `Apptainer` | Containerization software that allows for transportable and reproducible software | https://apptainer.org/ |
| `pandas` | Python library supporting dataframes and other datatypes for data analysis and manipulation | https://pandas.pydata.org/ |
| `geopandas` | Python library that extends pandas to support geospatial vector data and spatial operations | https://geopandas.org/en/stable/ |
| `rioxarray` | Python library supporting data structures and operations for geospatial raster data | https://github.com/corteva/rioxarray |
| `pathlib` | A Python library for handling files and paths in the filesystem | https://docs.python.org/3/library/pathlib.html |

*Prerequisites*:
  * Basic understanding of navigating the Linux command line, including navigating among directories and editing text files
  * Basic python skills, including an understanding of object-oriented programming, function calls, and basic data types
  * Basic understanding of shell scripts and job scheduling with SLURM for running code on Atlas
  * A SCINet account for running this tutorial on Atlas
  * **Completion of tutorial part 1 of this series**

*Tutorials in this Series*:
  * 1\. **Tutorial Setup on SCINet**
  * 2\. **Overview of Deep Learning for Imagery and the Raster Vision Pipeline <span style="color: red;">_(You are here)_</span>**
  * 3\. **Constructing and Exploring the Apptainer Image**
  * 4\. **Exploring the Dataset and Problem Space**
  * 5\. **Overview of Raster Vision Model Configuration and Setup**
  * 6\. **Breakdown of Raster Vision Code Version 1**
  * 7\. **Evaluating Training Performance and Visualizing Predictions**
  * 8\. **Modifying Model Configuration - Hyperparameter Tuning**

# 1. Overview of Deep Learning for Imagery Concepts

#### What is a Neural Network
A neural network is essentially a complicated mathematical function that receives inputs, such as images, and outputs predictions, such as image classification. A neural network has very many, often millions of parameters that control its functionality. You can think of each parameter as a dial, and the process of training a model involves iteratively adjusting the dials to improve the model's performance. Each iteration of the model training process involves passing data through the model, observing the model's accuracy, applying slight adjustments to the parameters to improve model performance, and repeating. If you are interested in learning more about the inner workings of neural networks, you can find more information [here](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). For this tutorial, we do not need an in depth understanding of the inner workings of a neural network, since we are not building and training a neural network from scratch. Raster Vision allows us to use a pre-defined model structure, which allows us to benefit from transfer learning.<br>

#### Process of training a neural network:
- Acquire a fully-labeled dataset.
- Split dataset into training, validating, and testing sets. (Learn more about training, validation, and testing sets [here](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7))
- Define model structure (or select pre-trained model if using transfer learning).
- Training loop:
    - Split the training dataset into batches.
    - For each batch of data:
         - Pass the batch of data through the model.
         - Observe the model accuracy.
         - Update the model parameters to improve model performance on that batch.
- Iterate through the training loop several times.
- Run the validation data through the model and observe performance. This allows you to gauge how well the model performs on data it has not been trained on. Modify training procedure as desired, and train again.
- Once you have a model that you are happy with, then run the model on the test data. This will gauge how well the model performs on data is has not been trained or validated on.
- Deploy model for use.

#### What is Transfer Learning

Training a neural network from scratch requires a lot of time and computational resources because there are so many parameters in our model to tune. Transfer Learning is a very common approach used to decrease the time it takes to train a model. With transfer learning, we first find a model that has already been trained to perform a certain task. Then, we use that model as a starting point, and further train it to perform new task. For example, say we wish to build a model that can identify trucks in images. If we already have a model that is trained to identify cars in images, then we can use that model as the starting point of our training procedure, and further train our pre-trained model using a dataset of truck images. This will work a lot faster than building a new model from scratch. <br>

For this tutorial, we will use the [ResNet50 model](https://arxiv.org/abs/1512.03385), which is pre-trained on the [ImageNet dataset](https://www.image-net.org/index.php). The ImageNet dataset contains over a million labeled images of objects in 1000 different classes, such as "canoe", "isopod", "acorn", and "miniature schnauzer". Since the ImageNet dataset contains a large breadth of image classes, the ResNet50 model can extract various image features and can thus be applied to diverse use cases.

#### Hyperparameters

<b>Parameters</b> are the "dials" within the model that are adjusted to improve the training accuracy. Parameter values are not directly set or updated by the analyst. Rather, they are initialized and updated through the model training process. <b>Hyperparameters</b>, on the other hand, are variables that control the process of training. Hyperparameters are set manually by the analyst, and analysts will often try a variety of different hyperparameter values to see which yields the best model. <br><br>
Examples of hyperparameters include:
- <b>Number of epochs</b>: the number of times we pass the entire training set through the model during model training.
- <b>Batch size</b>: the number of individual samples (ie labeled image chips) we pass through the model before updating the model parameters. Through the training process, we pass a batch of data through the model, observe the model performance, update the model parameters, and repeat. Once we have passed all of the training data through the dataset, we have completed one epoch.
- <b>Learning rate</b>: a scaling factor for the magnitude of adjustments to parameters. If we have too small of a learning rate, we will take very small "steps", and training will be slow. If we have too large of a learning rate, we won't have very fine-tune control of our parameters and we may "overstep" the optimal parameter values.

#### Image Chipping

Each neural network expects a specific input data size. For image datasets, this input data size refers to the pixel dimensions of the image, and the number of channels (most commonly, red, green, and blue). In geospatial data science, we often have very large images from satellite or drone imagery datasets. Neural networks generally operate on much smaller input sizes, so instead of passing an entire satellite image through a neural network, we break up our large imagery into smaller, bite-sized pieces of consistent dimensions called "chips". Chips can be sampled from an image dataset either in a grid-like fashion, or by random sampling. The chip size is another hyperparameter chosen by the analyst to fit the problem context, and various chip sizes can be tried. <br>

###### Note: Some resources use the term "tile" instead of "chip". These terms mean the same thing.

#### Image Classification

There are many different deep learning tasks we may wish to perform. Image Classification is the most basic deep learning task for image-based data. The goal of Image Classification is to input an image to a model and have the model output the image's class. For example, an Image Classification model could be trained to classify pictures as either "cats" or "dogs". Note that Image Classification models have a pre-defined set of classes to choose from, so if you have a model that can only choose between "cats" and "dogs", and you give that model a picture of a pig, the model will still return a prediction of either "cats" or "dogs".

For geospatial applications, we can build a model to classify chips of our dataset, instead of entire images. Hence, the Raster Vision documentation refers to this task as "Chip Classification" instead of "Image Classification" for clarity.

#### Object Detection

Object Detection allows us to find objects of interest within images. Image Classification can tell us, for example, that a picture is of a cat. Image Classification cannot tell us where in the image the cat is, or how many cats are in the image. An Object Detection model will output bounding boxes around objects of interest. <br>
![IC vs OD](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522766480/1_6j34dAOTijqP6HDFnjxPFA_udggex.png)
###### Image source: [DataCamp](https://www.datacamp.com/tutorial/object-detection-guide)<br>
Geospatial example: Object Detection could be used to analyze traffic conditions by detecting and counting cars on roads.

#### Semantic Segmentation

Semantic Segmentation models provide classification for every pixel within an image. While semantic segmentation doesn't allow us to count individual instances of objects, it does provide us with more detailed outlines of where one class ends and the next begins.<br>

![SS ex](https://assets-global.website-files.com/614c82ed388d53640613982e/63f498f8d4fe7da3b3a60cc2_semantic%20segmentation%20vs%20instance%20segmentation.jpg) 
###### Semantic Segmentation Image from [Li, Johnson, and Yeung](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)<br>
Geospatial example: Semantic Segmentation could be used to identify buildings in satellite images.

## 2. The Raster Vision Pipeline

##### "Raster Vision is an open source library and framework for Python developers building computer vision models on satellite, aerial, and other large imagery sets (including oblique drone imagery). There is built-in support for chip classification, object detection, and semantic segmentation using PyTorch." [(rastervision.io)](https://rastervision.io/) <br>
Raster Vision is a geospatial software tool produced by the company [Azavea](https://www.azavea.com/) that can be used as either a framework or as a library. The Raster Vision framework abstracts away many technical details of geospatial deep learning, and allows users to customize and run a deep learning pipeline. Advanced python programmers can use the Raster Vision library to use pieces of Raster Vision code in their own projects. We will focus solely on how to use the Raster Vision framework in this tutorial. <br><br>
Raster Vision is built on pytorch, which is a popular python library used for building and training neural networks. The Raster Vision framework utilizes a pipeline of execution that performs a series of steps to prepare the data, train the model, use the model to predict on the validation set, calculate evaluation metrics, and bundle the model for deployment. <br>

![RV pipeline](https://docs.rastervision.io/en/0.30/_images/rv-pipeline-overview.png) 
###### Image Source: [Raster Vision](https://docs.rastervision.io/en/0.30/framework/pipelines.html)<br>

Raster Vision is a low-code platform. Users will still need to write python code to specify how they want to build their model, however they will need to write much less code than if they were building the same model from scratch in pytorch. For example, users will not have to write code to chip the data or perform the training loop, but they will need to specify the chip size, the method for constructing chips, what model architecture to use, and which of the three supported Deep Learning tasks to perform (chip classification, object detection, or semantic segmentation). <br><br>

Raster Vision is ideal for ARS researchers who:
* Have large, fully labelled geospatial datasets they wish to expand to cover additional sites
    * Ex: satellite imagery, and associated vector data outlining objects of interest for Object Detection
    * Ex: aerial drone imagery, and associated raster data representing segmentation masks for Semantic Segmentation
* Can run their code on Atlas to take advantage of GPU acceleration
* Have python experience

##### Note: Raster Vision is not backwards compatible. When reading through documentation, ensure you are looking at the right version of Raster Vision. This tutorial is based on version 0.30.
The most up-to-date documentation can be found at [rastervision.io](https://rastervision.io/).

#### Conclusion
You now have an understanding of what Deep Learning is, what the Raster Vision pipeline does, and what kinds of problems it can help you solve. In the next tutorial, you will explore the apptainer container we will use to run Raster Vision.
