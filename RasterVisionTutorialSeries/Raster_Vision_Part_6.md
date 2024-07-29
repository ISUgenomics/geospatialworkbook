# Semantic Segmentation of Aerial Imagery with Raster Vision 
## Part 6: Breakdown of Raster Vision Code Version 1

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
  * **Completion of tutorial parts 1-5 of this series**

*Tutorials in this Series*:
  * 1\. **Tutorial Setup on SCINet**
  * 2\. **Overview of Deep Learning for Imagery and the Raster Vision Pipeline**
  * 3\. **Constructing and Exploring the Apptainer Image**
  * 4\. **Exploring the Dataset and Problem Space**
  * 5\. **Overview of Raster Vision Model Configuration and Setup**
  * 6\. **Breakdown of Raster Vision Code Version 1 <span style="color: red;">_(You are here)_</span>**
  * 7\. **Evaluating Training Performance and Visualizing Predictions**
  * 8\. **Modifying Model Configuration - Hyperparameter Tuning**

## Breakdown of Raster Vision Code 
Here we will present the basic structure of the `get_config()` function, and a helper function we use within `get_config()` called `make_scene()`. Then, we will convert our pseudocode to actual code bit by bit.

Finally, we will invoke the Raster Vision pipeline on Atlas through SLURM to train our first model!

### 1. Pseudocode

This tutorial series uses scripts that are based on the quickstart code that [Azavea](https://www.azavea.com/) provides. Script `tiny_spacenet1.py` is mostly identical to the [quickstart](https://docs.rastervision.io/en/0.30/framework/quickstart.html) code. Here are the few differences between the original quickstart code and our code: 
- The original Raster Vision quickstart code uses only 2 total images, whereas we will use 1000 images for training, 50 for validation, and 10 for testing. Both of our scripts, `tiny_spacenet1.py` and `tiny_spacenet2.py` refer to a set of data stored in `/reference/workshops/rastervision/input/`. Raster Vision's quickstart code hard-codes the names of the input data files, which are stored in AWS storage. Since we are using a much larger dataset, our code identifies all files that match the data file naming conventions in the `train/`, `val/`, and `test/` directories respectively, instead of hard-coding each name individually.
- Our scripts allows the user to specify the output directory at runtime, whereas the original quickstart code hardcodes the output directory name. We do this so the user (you) can invoke the pipeline multiple times without overwriting the output directory.
- Our `tiny_spacenet1.py` script trains for 3 epochs, instead of 1. This way, we can visualize how the model performance metrics change over the course of the 3 epochs. If we just run for one epoch, then we can only evaluate the model performance for that one epoch and can't see any trends in the training process.
- Our `tiny_spacenet1.py` script sets the variable `max_window` to 5 instead of 10. This means that for each 650x650 pixel training image, we randomly select 5 300x300 training chips. This decreases our total dataset size, but also reduces redundancy in the training data, and greatly decreases run time.

Here is the pseudocode for `tiny_spacenet1.py`.

```python
def get_config(runner, user_configured_arguments) -> SemanticSegmentationConfig:
    '''
    1. Define the uri's for input and output data
    2. Define the ClassConfig object to specify the classes that the model will predict (building and background)
    3. Define the uri's of the training, validation, and test data files
    4. Create SceneConfig objects for the training, validation, and test data by calling the make_scene() helper function
    5. Create a DatasetConfig object by referencing the training, validation, and test SceneConfig objects, and the ClassConfig object
    6. Configure the model backend:
        a. Specify the data for the model, which is based on the DatasetConfig object, and methods for constructing chips from raster images within that DatasetConfig object
        b. Specify the model architecture to use (we choose ResNet50)
        c. Configure the solver, specifying model hyperparameters
    7. Return the SemanticSegmentationConfig object, which refers to the output uri, the DatasetConfig object, the backend, and the chip sizes
    '''
def make_scene(scene_id: str, image_uri: str, label_uri: str,
               class_config: ClassConfig):
    '''
   1.  Configure RasterioSourceConfig object to read in a raster from a data file
   2.  Configure GeoJSONVectorSourceConfig object to read in vector data from a data file
   3.  Create SemanticSegmentationLabelSourceConfig object by rasterizing the vector source and specifying the class values
    ''' 
```

### 2. Analyzing Code: tiny_spacenet1.py

In your terminal, navigate from your project directory to `model/src/` and open up `tiny_spacenet1.py` in your favorite text editor (ie `nano tiny_spacenet1.py`). Now, we will go through each step listed in the pseudocode above and convert it to the code you see in `tiny_spacenet1.py`.

<b> We highly recommend reading through the `tiny_spacenet1.py` script alongside section 2.1 of this tutorial to understand how this code works. </b>

##### A note about the output directory:
We encourage users specify a different output directory from the command line each time they train a model. This way, data from previous runs is not overwritten. Also, Raster Vision is equipped to check the output directory for any pre-built model configurations, and may load the existing model bundle instead of re-training the model from scratch.

### 2.1 The get_config() 

The following 7 steps represent the code within the `get_config()` function definition.

##### Step 1: Define the uri's for input and output data

The input data uri is easy. We assume that the input data will stay in the same place each time we run our code, so we will specify the input directories as `Path` objects from the `pathlib` package. The output directory uri is more difficult. Each time we run our code, we want the output to go to a new directory, otherwise our outputs from previous runs will be overwritten. Raster Vision allows us to configure user-specified command line arguments so we can modify the behavior of the pipeline at run time. We will create a command line argument called `output_uri` so the user can specify the output directory as they invoke the pipeline. This takes two steps:
1. We must list the user-specified arguments as inputs to our `get_config()` function. This tells the `get_config()` function what command line arguments to expect. Here, we include `output_uri` as an input to the `get_config()` function.
2. When we invoke the Raster Vision pipeline, we must specify our user-specified arguments as key value pairs. We will explain the specifics of this step later in section 3.2 when we analyze the script we will use to invoke the pipeline.

Here's what the header of the `get_config()` function looks like, including the CLI argument, `output_uri`.

```python
def get_config(runner, output_uri) -> SemanticSegmentationConfig:
```
The `runner` object allows us to run the steps in our pipeline. Every `get_config()` function takes a runner object as an input. We specify the value of the runner when we invoke the Raster Vision pipeline. We will discuss this more in section 3.3 when we describe the script we use to invoke the pipeline. </br>

We accept the `output_uri` variable as an input to the `get_config()`, but won't need to refer to it until the very end of our code in step 7.

We use the [pathlib](https://docs.python.org/3/library/pathlib.html) library to define the paths of our training, validation, and test datasets. Here's what this looks like: 

```python
# Specify directory for input files - training, validation, and testing
input_uri = Path("/opt/data/input")
train_uri = Path(input_uri / "train")
val_uri = Path(input_uri / "val")
test_uri = Path(input_uri / "test")
```
You may recall that we have all of our input data stored at `/reference/workshops/rastervision/input/`, but here we see the the input data stored at `/opt/data/input/`. This is because when we build our apptainer image, we bind the `/reference/workshops/rastervision/input/` directory from the host file system to the directory `/opt/data/input/` within the container. This allows our input data to be accessed in the container at `/opt/data/input/`. We will describe how we bind these directories in section 3.3. For now, all you need to know if that all of the contents in `/reference/workshops/rastervision/input/` on the host system are available at `/opt/data/input/` in the container.

##### Step 2: Define the [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object to specify the classes that the model predicts

[`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) objects list the class values we want our model to differentiate between. For this problem, since we are building a semantic segmentation model to identify buildings, we will define two classes: building and background. Here's what the code for step 2 looks like:

```python
class_config = ClassConfig(names=['building', 'background'])
```
For this problem, we don't need to specify any other parameters for the [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object.

##### Step 3: Define the uri's of the training and validation data files

We have 1000 training images, 50 validation images, and 10 testing images. The original [quickstart](https://docs.rastervision.io/en/0.30/framework/quickstart.html) code explicitly writes out the paths to the two images used for training and validation. It would be inefficient to write out the paths for 1060 images and 1060 labels, so instead, we will use the [Path.glob()](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) function in the [pathlib](https://docs.python.org/3/library/pathlib.html) library to create lists of all the files that match our desired filename [regex](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions/Cheatsheet). Here's what the code for this step looks like:

```python
# Create lists of file paths
train_image_uris = train_uri.glob("RGB-PanSharpen_AOI_2_Vegas_img*.tif")
train_label_uris = train_uri.glob("buildings_AOI_2_Vegas_img*.geojson")
val_image_uris = val_uri.glob("RGB-PanSharpen_AOI_2_Vegas_img*.tif")
val_label_uris = val_uri.glob("buildings_AOI_2_Vegas_img*.geojson")
test_image_uris = test_uri.glob("RGB-PanSharpen_AOI_2_Vegas_img*.tif")
test_label_uris = test_uri.glob("buildings_AOI_2_Vegas_img*.geojson")
```

##### Step 4: Create [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) objects for the training, validation, and test data by calling the make_scene() helper function

Next, we need to create a list of [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) objects. [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) objects contain following information: the scene ID, the raster source, and the label source. We will use a helper function, `make_scene()` to create our SceneConfig objects. We will go through all of the code in the `make_scene()` function in section 2.2. For now, all we need to know about the `make_scene()` function is that it takes four inputs (an ID, a raster uri, a label uri that corresponds to the raster uri, and [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object), and returns a [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) object.

We will loop through the image files in the train, validation, and test data directories respectively, and construct lists of SceneConfig objects. To do this, we extract the scene ID from the image file name using the string `split()` function. Then, we use that ID to construct the filename of the corresponding vector data file. Lastly, we call the `make_scene()` function, and add the returned [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) object to our list. Here is the code for creating the list `train_scenes`. 

```python
train_scenes = []
for filename in train_image_uris:
    index = str(filename).split("RGB-PanSharpen_AOI_2_Vegas_img")[1].split(".tif")[0]
    label_filename = "buildings_AOI_2_Vegas_img" + index + ".geojson"
    if Path(train_uri / label_filename).is_file():
    train_scenes.append(make_scene(
            index, 
            str(Path(train_uri / filename)),
            str(Path(train_uri / label_filename)),
            class_config
            )
        )
    else:
        print("No train label file found for index) ", index)
```

We use equivalent code in `tiny_spacenet1.py` to create `validation_scenes` and `test_scenes` lists, the only difference being the names "train", "validation", and "test". We omit that code here for brevity.

Now, we have three lists, `train_scenes`, `validation_scenes` and `test_scenes`, each which contain [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) objects. Each [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) object refers to the uri of a .tif file, the associated .geojson file, the scene ID, and the [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object.

##### Step 5: Create a [`DatasetConfig`](https://docs.rastervision.io/en/0.30/search.html?q=datasetconfig&check_keywords=yes&area=default) object by referencing the training, validation, and test [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html#rastervision.core.data.scene_config.SceneConfig) objects and the [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object

Raster Vision's [`DatasetConfig`](https://docs.rastervision.io/en/0.30/search.html?q=datasetconfig&check_keywords=yes&area=default) objects contain the lists of training, validation, and testing scenes, plus the [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) information. Here is the code we use to create our [`DatasetConfig`](https://docs.rastervision.io/en/0.30/search.html?q=datasetconfig&check_keywords=yes&area=default) object.

```python
scene_dataset = DatasetConfig(
    class_config=class_config,
    train_scenes=train_scenes,
    validation_scenes=validation_scenes,
    test_scenes=test_scenes
)
```
This [`DatasetConfig`](https://docs.rastervision.io/en/0.30/search.html?q=datasetconfig&check_keywords=yes&area=default) object is one of the components we will need to build the [`SemanticSegmentationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig.html) object that the `get_config()` function returns.

##### Step 6: Configure the model backend

Now that we have our data, we will build our backend. The backend specifies what dataset we are using, how to pull chips from that dataset, what model backbone to use, and what hyperparameters to use when training. Currently, all backends in Raster Vision use pytorch, so we will build our backend object with the [`PytorchSemanticSegmentationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pytorch_backend.pytorch_semantic_segmentation_config.PyTorchSemanticSegmentationConfig.html#pytorchsemanticsegmentationconfig) class. The default loss function is `nn.CrossEntropyLoss`, and the optimizer is `optim.Adam`. You can learn more about Cross Entropy Loss [here](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) and about Adam optimization [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). 

Raster Vision is designed for problems involving large raster datasets, such as satellite images. These images are usually way too large to input into a neural network, so Raster Vision chips our data into smaller, consistently sized chips. We need to specify how large we want our chips to be, how to select chips from our raster images (using either a random or sliding window method), and if we select chips using the random method, we also need to specify the maximum number of chips to take from a single scene. 

We use the [`SemanticSegmentationGeoDataConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pytorch_learner.semantic_segmentation_learner_config.SemanticSegmentationGeoDataConfig.html) object to encapsulate the following information: 
- The [`DatasetConfig`](https://docs.rastervision.io/en/0.30/search.html?q=datasetconfig&check_keywords=yes&area=default) object we created above which encapsulates our training, validation, and test scenes.
- A [`GeoDataWindowConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pytorch_learner.learner_config.GeoDataWindowConfig.html) object which will specify how to select chips from our scenes.
- A [`SemanticSegmentationModelConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pytorch_learner.semantic_segmentation_learner_config.SemanticSegmentationModelConfig.html#semanticsegmentationmodelconfig) object which will specify our model backbone. For this tutorial, we will use ResNet50 as our backbone.
- A [`SolverConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.pytorch_learner.learner_config.SolverConfig.html#solverconfig) object which will specify our training hyperparameters such as learning rate and batch size.

Here's how we construct our backend object:

```python
chip_sz = 300
backend = PyTorchSemanticSegmentationConfig(
    data=SemanticSegmentationGeoDataConfig(
        scene_dataset=scene_dataset,
        sampling=WindowSamplingConfig(
                # randomly sample training chips from scene
                method=WindowSamplingMethod.random,
                # ... of size chip_sz x chip_sz
                size=chip_sz,
                # ... and at most 4 chips per scene
                max_windows=5)),
    model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
    solver=SolverConfig(lr=1e-4, num_epochs=3, batch_sz=2)
)
```

##### Step 7: Return [`SemanticSegmentationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig.html) Object

Lastly, we need to return a [`SemanticSegmentationConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig.html) object that encapsulates all of the information the Raster Vision Pipeline needs to build our model. Here's what this code looks like:

```python
return SemanticSegmentationConfig(
    root_uri=output_uri,
    dataset=scene_dataset,
    backend=backend,
    predict_options=SemanticSegmentationPredictOptions(chip_sz=chip_sz))
```

Recall that the `output_uri` variable is a user-specified command line argument that is input to the `get_config()` function.

### 2.2 The make_scene() Function

Now, we describe the `make_scene()` helper function we called in step 4 of section 2.1. Each "scene" corresponds to one raster file and the corresponding vector file. Our datasets are made of collections of scenes. The `make_scene()` function takes the following four inputs, and returns a [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html) object. 

- The scene ID, a string
- The URI of the raster file, a string
- The URI of the label file, a string
- A [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object

To build a [`SceneConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.scene_config.SceneConfig.html) object, we need the following objects:
- The scene ID, a string
- A [`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfig.html) object
- A [`LabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.label_source_config.LabelSourceConfig.html) object

So, our `make_scene()` object must create a [`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfig.html) object using the URI of the raster image, and must create a [`LabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.label_source_config.LabelSourceConfig.html) object from the URI of the label file and the [`ClassConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.class_config.ClassConfig.html) object. Both [`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfig.html) and [`LabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.label_source_config.LabelSourceConfig.html) are ABCs with subclasses that we will choose from  based the form of our data and the kind of model we wish to build.

[`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfigtm.html) objects simply represent the source of raster data for a scene. There are various subclasses of [`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfigtm.html) used for various raster data formats. Examples of subclasses of the [`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfig.html) include:

- [`RasterioSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.rasterio_source_config.RasterioSourceConfig.html) for raster files that can be opened by GDAL/Rasterio
- [`MultiRasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.multi_raster_source_config.MultiRasterSourceConfig.html#multirastersourceconfig) for concatenating multiple [`RasterSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.raster_source_config.RasterSourceConfig.html) objects along the channel dimension
- [`RasterizedSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.rasterized_source_config.RasterizedSourceConfig.html) for creating raster sources by rasterizing vector data

###### Note: The [`XarraySource`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.xarray_source.XarraySource.html#rastervision.core.data.raster_source.xarray_source.XarraySource) object used for creating RasterSource objects from Xarray data is still in beta, and does not yet have an associated config object.

Likewise, Raster Vision provides the [`VectorSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_source.vector_source_config.VectorSourceConfig.html) class to represent the vector data of a scene. The only subclass of [`VectorSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_source.vector_source_config.VectorSourceConfig.html) is [`GeoJSONVectorSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_source.geojson_vector_source.GeoJSONVectorSource.html) for geojson files. This means we must ensure our vector data is in geojson format. 

For this project, we only have two classes: building and background. Our vector data outlines each building, so we can assume whatever is inside a polygon is a building and whatever is outside a polygon is the background. If your semantic segmentation project involves more than two classes, you will need to provide a `class_id` label for each of your polygons. The [`GeoJSONVectorSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_source.geojson_vector_source.GeoJSONVectorSource.html) object includes the field `transformers` which can be used to apply the default class ID to each polygon, or to otherwise transform class IDs. In the code below, you will see how we use a [`ClassInferenceTransformerConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_transformer.class_inference_transformer_config.ClassInferenceTransformerConfig.html) object in the `transformers` field to apply the default class ID.

Our label data may be in either raster or vector format, and will vary based on the deep learning task we are performing. For example, for semantic segmentation, our label data must be in raster form, and for object detection, our label data must be in vector form. We use the [`LabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.label_source_config.LabelSourceConfig.html) class to store our label data. The three subclasses of [`LabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.label_source_config.LabelSourceConfig.html) are:
- [`ChipClassificationLabelSourceConfig`](https://docs.rastervision.io/en/0.30/search.html?q=chipclassificationlabelsourceconfig&check_keywords=yes&area=default)
- [`ObjectDetectionLabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.object_detection_label_source_config.ObjectDetectionLabelSourceConfig.html)
- [`SemanticSegmentationLabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.semantic_segmentation_label_source_config.SemanticSegmentationLabelSourceConfig.html)

We will use the [`SemanticSegmentationLabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.semantic_segmentation_label_source_config.SemanticSegmentationLabelSourceConfig.html) object for this project. Since we have label data in geojson format, and we need to provide label data for the [`SemanticSegmentationLabelSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.label_source.semantic_segmentation_label_source_config.SemanticSegmentationLabelSourceConfig.html) object in raster format, we will first read our data into a [`GeoJSONVectorSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_source.geojson_vector_source.GeoJSONVectorSource.html) object, then build a [`RasterizedSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.raster_source.rasterized_source_config.RasterizedSourceConfig.html) object from our [`GeoJSONVectorSourceConfig`](https://docs.rastervision.io/en/0.30/api_reference/_generated/rastervision.core.data.vector_source.geojson_vector_source.GeoJSONVectorSource.html) object.

Here's what our `make_scene()` function looks like:
```python
def make_scene(scene_id: str, image_uri: str, label_uri: str,
               class_config: ClassConfig) -> SceneConfig:
    """Define a Scene with images and labels from the given URIs."""
    raster_source = RasterioSourceConfig(
        uris=image_uri,
        # use only the first 3 bands
        channel_order=[0, 1, 2]
    )

    # configure GeoJSON reading
    vector_source = GeoJSONVectorSourceConfig(
        uris=label_uri,
        # The geoms in the label GeoJSON do not have a "class_id" 
        # property, so classes must be inferred. Since all geoms are for 
        # the building class, this is easy to do: we just assign the 
        # building class ID to all of them.
        transformers=[
            ClassInferenceTransformerConfig(
                default_class_id=class_config.get_class_id('building'))
        ])
    # configure transformation of vector data into semantic
    # segmentation labels
    label_source = SemanticSegmentationLabelSourceConfig(
        # semantic segmentation labels must be rasters, so rasterize
        # the geoms
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(
                # Mark pixels outsidas background.
                background_class_id = \
                    class_config.get_class_id('background'))))

    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
    )
```

### 3. Analysis of Shell Scripts to Run Raster Vision

Now that we have a better understanding of the code we use to specify how we want to build and train our model, we get to the fun part - actually running it! We will run our code in a batch script through SLURM. If you aren't familiar with using SLURM, check out the workbook [here](https://datascience.101workbook.org/06-IntroToHPC/05-JOB-QUEUE/01-SLURM/01-slurm-basics#gsc.tab=0).

From your project directory, navigate to the model directory and open up the `run_model1.sh` script in your favorite text editor (such as nano) as follows:

`cd $project_dir/model` </br>
`nano run_model1.sh` </br></br>
You will now see the shell script we will use to invoke the Raster Vision pipeline in the text editor. 

#### 3.1 SBATCH Header Lines
At the very beginning, you will see:

`#!/bin/bash -l` </br>
`#SBATCH -t 150` </br>
`#SBATCH -A geospatialworkshop` </br>
`#SBATCH --mem=256gb` </br>
`#SBATCH --partition=gpu-a100-mig7`</br>
`#SBATCH --gres=gpu:a100_1g.10gb:1` </br>
`#SBATCH -n 4` </br>
`#SBATCH --cpus-per-task 2` </br>

If you are not a part of the geospatialworkshop project group, go ahead and modify the line `#SBATCH -A geospsatialworkshop` to list a project group that you are a part of.

#### 3.2 Reading in User-Specified Arguments

In this script, we allow the user to specify the name of the output directory at runtime. We can do this by accepting one positional argument. Here, `$#` refers to the number of command line arguments provided, `$1` refers to the first argument. We first check that there is exactly one argument provided, and then set the value of that argument to the variable name `OUT_DIR`.

```bash
if [ ! $# -eq 1 ]
  then
    echo "Usage: sbatch run_model1.sh output_directory_name"
  exit
fi

OUT_DIR=$1
echo Output directory set as: $OUT_DIR
```


### 3.3 The Shell Script to Invoke the Raster Vision Pipeline

Lastly, we need to spin up our apptainer container and run Raster Vision! Before we run any apptainer commands, we need to first load the apptainer module. As of the time of writing, the default version of apptainer causes errors when running on the gpu nodes, so we will load a different version that does not cause errors:

`module load apptainer/1.1.9`

Next, we will describe how we use `apptainer exec` to build our container, and then we will describe the Raster Vision command we will use `apptainer exec` to run.

#### The `apptainer exec` command
As you may recall, we use `apptainer exec` as follows: <br> 
`apptainer exec [EXEC OPTIONS] CONTAINER COMMAND`. 

We will use the `--nv` option of `apptainer exec` to specify that we would like Nvidia support, since we are running our code on a gpu node. Then, we use the `--bind` option to bind our input data in `/reference/workshops/rastervision/input/` on the host machine to `/opt/data/input/` in the container so we can access our data. We also bind `` `pwd`/local `` on the host machine with `/local` in the container. This provides the necessary scratch space for apptainer. Recall that by default, apptainer binds the current working directory on the host machine to the container, so our `model/` directory will be available within the container. So far, our `apptainer exec` command looks like this:

```bash
apptainer exec --nv --bind \
/reference/workshops/rastervision/input/:/opt/data/input/ \
--bind `pwd`/local/:/local/ \ 
raster-vision_pytorch-0.30.sif \ 
COMMAND
```

#### The `rastervision run` command
The command we will use to invoke the Raster Vision pipeline is [`rastervision run`](https://docs.rastervision.io/en/0.20/framework/cli.html#run). The formula for using [`rastervision run`](https://docs.rastervision.io/en/0.20/framework/cli.html#run) is as follows: <br>
`rastervision run [OPTIONS] RUNNER CFG_MODULE [COMMANDS]...`

#### The `runner` argument
The `runner` argument is required for every call to `rastervision run`, and for every example in this tutorial, our `runner` will be set to `local`. When we set our runner to `local`, we are specifying that we want to run our code on the local machine, and we want to run splittable commands in parallel. Other options for the runner include `inprocess` which will run everything sequentially, and `batch` which is for submitting batch jobs to Amazon Web Services. 

#### The `--splits` option
The [`rastervision run`](https://docs.rastervision.io/en/0.20/framework/cli.html#run) command allows us to parallelize the execution of our code. This helps us speed up the chipping and predicting tasks in particular. After some trial and error, the authors have determined that this tutorial's code runs the fastest when split into 4 processes, so we set the number of splits to 4 like this: `--splits 4` or `-s 4`.

#### User-specified CLI arguments passed to get_config()

You may recall that our `get_config()` function, described in section 2.1, requires two arguments: `runner` and `output_uri`. The `runner` argument, as described above, we set to `local`. If you choose to include user-specified CLI arguments in your code, you can specify the values of those arguments as options to the `rastervision run` command. We specify the names of arguments and the values of arguments as follows: `-a KEY VALUE` or `--arg KEY VALUE`. Since our argument name is `output_uri`, and we have read in the name of the output directory into the variable `OUT_DIR` in step 3.2, our argument specification will look like this: `-a output_uri $OUT_DIR`.

#### The CFG_MODULE
The `CFG_MODULE` refers to the python script containing the `get_config()` function definition. In step 3.2, we read the python script name into the `SCRIPT` variable.

The code to load apptainer, build our container, and invoke the Raster Vision pipeline within the container is as follows:

```bash
module load apptainer/1.1.9
apptainer exec --nv --bind /reference/workshops/rastervision/input/:/opt/data/input/ \
--bind `pwd`/local/:/local/ raster-vision_pytorch-0.30.sif \
rastervision run -s 4 -a output_uri `pwd`/$OUT_DIR \
local `pwd`/src/$SCRIPT
```

### 4. Invoking the Raster Vision Pipeline
Now we're ready to run our code! Run the following commands:

```
cd $project_dir/model
sbatch run_model1.sh output1
```
This will create an output directory named `output1`, invoke the pipeline, and put all output files in `output1/`. Once you have sbatch-ed your script, you can use `squeue --me` to track your running jobs. Since you are currently running an interactive jupyter session, you will see a job named `sys/dash` which corresponds to your jupyter session. If you see a second job listed, then that means that your code is either queued or running. Once your job starts running, if you run `ls`, you will notice a slurm log file in the directory from which you sbatch-ed the job. You can run the following command to watch the output file as it is being created:

`watch -n 5 tail -n 20 slurm-...` (tab complete to fill in the rest of the slurm log file name)

#### Conclusion
You are training your first Raster Vision model! In the next tutorial, we will explore how to evaluate our model performance.
