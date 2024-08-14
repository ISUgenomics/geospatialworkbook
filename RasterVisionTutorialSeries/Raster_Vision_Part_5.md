---
title: "Raster Vision Tutorial Series Part 5: Overview of Raster Vision Model Configuration and Setup"
layout: single
author: Noa Mills
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---


# Semantic Segmentation of Aerial Imagery with Raster Vision 
## Part 5: Overview of Raster Vision Model Configuration and Setup

This tutorial series walks through an example of using [Raster Vision](https://rastervision.io/) to train a deep learning model to identify buildings in satellite imagery.</br>

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
  * **Completion of tutorial parts 1-4 of this series**

*Tutorials in this Series*:
  * 1\. **Tutorial Setup on SCINet**
  * 2\. **Overview of Deep Learning for Imagery and the Raster Vision Pipeline**
  * 3\. **Constructing and Exploring the Apptainer Image**
  * 4\. **Exploring the Dataset and Problem Space**
  * 5\. **Overview of Raster Vision Model Configuration and Setup <span style="color: red;">_(You are here)_</span>**
  * 6\. **Breakdown of Raster Vision Code Version 1**
  * 7\. **Evaluating Training Performance and Visualizing Predictions**
  * 8\. **Modifying Model Configuration - Hyperparameter Tuning**

## Overview of Raster Vision Model Configuration and Setup

Raster Vision provides a plethora of classes used for various aspects of model configuration. Raster Vision relies heavily on Abstract Base Classes (ABC's) and pydantic models. If you are not familiar with ABC's in python, you can learn more about them [here](https://docs.python.org/3/library/abc.html#abc.ABC), and if you are not familiar with pydantic models, you can find a brief introduction [here](https://docs.pydantic.dev/latest/) and a thorough description of how to use them [here](https://docs.pydantic.dev/latest/concepts/models/).

One of the biggest hurdles to understanding Raster Vision code is understanding all of the different classes that Raster Vision defines. Many classes in Raster Vision are subclasses of other classes in Raster Vision, or have other class objects as attributes. This can make the documentation confusing for a newcomer, as further research into one class will only yield several more unfamiliar classes. Here, we provide an overview of what classes and functions are used to configure a basic model.

###### Note: In this tutorial, all Raster Vision class names will be hyperlinks to documentation, although they will be in code format so they won't appear blue or underlined.

### 1. Config Objects and the get_config() Function

Raster Vision users configure a model pipeline by writing a python script that defines a function called `get_config()`. This function builds and returns an instance of [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html). The class [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html) is an Abstract Base Class (ABC), and users must build an instance of one of [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html)'s three concrete subclasses: 
- [`ChipClassificationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.chip_classification_config.ChipClassificationConfig.html#chipclassificationconfig)
- [`ObjectDetectionConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.object_detection_config.ObjectDetectionConfig.html)
- [`SemanticSegmentationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig.html) </br>

The [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html) object encapsulates all the information that the Raster Vision pipeline needs to build the model, including what deep learning task to perform, where the data is stored, what model architecture to build, and various hyperparameter values. The Raster Vision pipeline calls the `get_config()` function defined by the user to produce a [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html) object, uses that [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html) object as a blueprint for how to build the desired model, and follows the steps of the pipeline as described in tutorial 2.

When reading through the Raster Vision documentation and code, you will see many classes defined by Raster Vision with names that end with [`Config`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pipeline.config.Config.html), such as [`RVPipelineConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig.html), [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html), and [`DatasetConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.dataset_config.DatasetConfig.html). All of these objects are subclasses of Raster Visions [`Config`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pipeline.config.Config.html) class, which is itself a pydantic model. Config objects are created to take advantage of pydantic's validation features, so behind the scenes, Raster Vision can validate the user's input to ensure that all of the parameters are valid. Many [`Config`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pipeline.config.Config.html) objects have associated objects - for example, [`DatasetConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.dataset_config.DatasetConfig.html) objects are blueprints for pytorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) objects and [`SemanticSegmentationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig.html) objects are blueprints for [`SemanticSegmentation`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.semantic_segmentation.SemanticSegmentation.html#rastervision.core.rv_pipeline.semantic_segmentation.SemanticSegmentation) objects. This allows Raster Vision to validate the user's input before creating and using an object.</br></br>

### 2. Directory Tree
There are many different ways a user can set up a directory tree to store their singularity file, code scripts, input data, and output files. Here's a reminder of what your project directory tree looks like.

|-- model/ </br>
|-- |-- local/ </br>
|-- |-- src/ </br>
|-- |-- run_model1.sh </br>
|-- |-- run_model2.sh </br>
|-- |-- make_apptainer_img.sh </br>
|-- |-- raster-vision_pytorch-0.30.sif </br>
|-- tutorial_notebooks/ </br>
|-- |-- imgs/ </br>
|-- |-- Raster_Vision_Part_1.ipynb </br>
|-- |-- Raster_Vision_Part_2.ipynb </br>
... </br>
|-- |-- Raster_Vision_Part_10.ipynb </br>

The `model/` directory is where we will run the Raster Vision pipeline - this is where our code is, and where our output data will go. Here we describe the contents of this folder more thoroughly: 
- The `model/src/` directory contains python scripts that define different versions of the `get_config()` function. The first script, `tiny_spacenet1.py`, is practically identical to the quickstart code produced by the Raster Vision team. The script `tiny_spacenet2.py` includes updates that we will apply in the last tutorial.
- The files `run_model1.sh` and `run_model2.sh` are a shell script we use to execute the pipelines defined by `tiny_spacenet1.py` and `tiny_spacenet2.py`, respectively. These scripts build the apptainer image with the needed path bindings and invoke the Raster Vision pipeline.
- The `model/local/` directory is included to provide scratch space for apptainer. We don't need to put any files in this directory, but apptainer will use this directory when we build our container, and will throw errors if it does not exist.

Each time we run the pipeline in this tutorial series, we specify the name of an output directory to store all of our output files in. The pipeline will create this folder in `model/` if it does not yet exist. The Raster Vision pipeline will populate the output directory with many files and subdirectories, only a few of which we will need to reference in this tutorial series. These include the `eval/` directory, which will contain our evaluation metrics, the `predict/` directory which will contain prediction rasters associated with the validation and test sets, the `train/` directory which contains metrics collected during the training process, and the `bundle/` directory which contains a bundle of the model for deployment.

Lastly, let's take a look at the directory tree of the `/reference/workshops/rastervision/` directory. </br>

/reference/workshops/rastervision/ </br>
|-- input/ </br>
|-- |-- train/ </br>
|-- |-- test/ </br>
|-- |-- val/ </br>
|-- rastervision_env/ </br>
|-- model/ <b># Copied to your project directory</b> </br>
|-- tutorial_notebooks/ <b># Copied to your project directory </b></br>
|-- requirements.txt

You have already copied the `model/` and `tutorial_notebooks/` directories to your project directory. You'll also see the `rastervision_env/` directory, which you used to build the jupyter kernel. Lastly, you'll see the `input/` directory. This contains all of the data we will use for model training, validation, and testing, split into three subdirectories. Instead of copying all of this data over to your project directory, our code will refer to the input data in-place to save space.

#### Conclusion
You now know the following:
- To build a Raster Vision model, you must write a script that defines the `get_config()` function.
- Where our input data is
- Where our python and shell scripts are
- Where our output data goes </br>

In the next tutorial, we'll take a look at what goes into the `get_config()` function, and run our first version of the code!
