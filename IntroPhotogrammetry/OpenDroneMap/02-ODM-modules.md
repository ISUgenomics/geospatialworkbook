---
title: "Command-line ODM modules"
layout: single
author: Aleksandra Badaczewska
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /IntroPhotogrammetry/assets/images/geospatial_workbook_banner.png
---

{% include toc %}


# Getting started with ODM

ODM stands for OpenDroneMap, an **open-source** photogrammetry software. OpenDroneMap project includes several modules which facilitate geospatial analysis in customized configurations of available computing power and for projects of different scales. The [OpenDroneMap/ODM](https://www.opendronemap.org/odm/) [[GitHub](https://github.com/OpenDroneMap/ODM)] module is the analytical core of the software. Typically it is employed by the higher-level layers of other OpenDroneMap products that provide a graphical user interface or manage analysis configuration and task scheduling *(learn more in the [Introduction to OpenDroneMap](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/00-IntroODM) section of the [Geospatial Workbook](https://geospatial.101workbook.org)*.

The point is that you can **directly use the ODM module on the command line** in the terminal for drone imagery mapping. This tutorial will take you **step-by-step** through creating an efficient file structure and setting up a script to send the job to the SLURM queue on an HPC cluster, which will allow you **to collect all the results** of the photogrammetric analysis with OpenDroneMap.

<span style="color: #ff3870;font-weight: 500;">
The complete workflow includes photo alignment and generation of the dense point cloud (DPC), elevation models (DSMs & DTMs), textured 3D meshes, and orthomosaics (georeferenced & orthorectified).
</span>
<b>Note that in this approach there is NO support to create web tiles</b>. So, the results can not be directly opened in the complementary WebODM graphical interface. But, the files can be still visualized in external software that supports the given format.

# Run ODM in the command line <br><i>using Atlas cluster of the SCINet HPC</i>

## **Create file structure**

The ODM module for the command-line usage requires the specific file structure to work without issues. Specifically, it requires that input imagery be in the `code/images` subdirectories structure of the working directory for a given project.

To facilitate proper path management, I suggest creating the core of ordered file structure for all future projects. Let's say it will be the **ODM** directory, serving as a working directory for all future ODM analyses. It should contain two main subdirectories: IMAGES and RESULTS. In the **IMAGES** directory, you will create a new folder with a unique name for each new project, where you will place photos in JPG format *(e.g., ~/ODM/IMAGES/PROJECT-1)*. In the **RESULTS** directory, when submitting an ODM task into the SLURM queue, a subdirectory with ODM outputs will be automatically created for each project. And there, also automatically, an `code/images` subdirectories will be created with soft links to photos from the relative project.

<div style="background: #cff4fc; padding: 15px;">
<b>ODM/</b> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<i>(storage directory for all future ODM projects)</i><br>
|― odm_latest.sif &emsp;&emsp;&emsp;&ensp;<i>(singularity image of the ODM tool)</i> <br>
|― run_odm_latest.sh &emsp;&ensp;<i>(SLURM job submission script)</i> <br>
|― <b>IMAGES/</b> <br>
 &emsp;&nbsp; |― <b>PROJECT-1/</b> &emsp;&emsp;&ensp;<i>(directory with images in JPG format)</i> <br>
 &emsp;&nbsp; |― <b>PROJECT-2/</b> <br>
|― <b>RESULTS/</b> &emsp;&emsp;&emsp;&emsp;&emsp;<i>(parent directory for ODM analysis outputs)</i> <br>
<span style="color: #8ca3a8;">
&emsp;&nbsp; |― PROJECT-1-tag/ &ensp; <i>(automatically created directory with ODM analysis outputs)</i> <br>
&emsp; &nbsp;&emsp;&nbsp; |― code/ &emsp;&emsp;&emsp;&emsp; <i>(automatically created dir for analysis outputs; <u>required!</u>)</i> <br>
&emsp; &emsp;&emsp; &emsp; |― images/ &emsp;&nbsp; <i>(automatically created dir with soft links to JPG images)</i> <br>
</span>
</div><br>

This way, if you want to perform several analyses with different parameters on the same set of images, you will not need to have a hard copy for each repetition. Instead, you will use soft links to the original photos stored in the IMAGES directory. **That will significantly save storage space and prevent you from accidentally deleting input imagery when resuming the analysis in the working directory.**

<br><span style="color: #ff3870;font-weight: 600; font-size:24px;">
To set up the file structure for ODM analysis follow the steps in the command line:
</span><br>

**0.** Open the terminal window on your local machine and login to the SCINet Atlas cluster *(or any HPC infrastructure)* using the `ssh` command and respective hostname:

```
ssh <user.name>@atlas-login.hpc.msstate.edu
```

Then, enter on the keyboard your 1) [multi-factor authenticator](https://scinet.usda.gov/guide/multifactor/) number (6 digits), followed by 2) password in a separate prompt.

<!-- ![Atlas login with authenticator](img) -->

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">
If you happen to forget the hostname for the Atlas cluster, you can save the login command to a text file in your home directory on your local machine: <br><br>
<span style="color: black;font-weight: 600;">
$ echo "ssh user.name@atlas-login.hpc.msstate.edu" > login_atlas <br><br>
</span>
Then every time you want to log in, just call the name of this file with a period and space preceding it: <br><br>
<span style="color: black;font-weight: 600;">
$ . login_atlas <br><br>
</span>
You can also review the HPC clusters available on SCINet at <a href="https://scinet.usda.gov/guide/quickstart#hpc-clusters-on-scinet" style="color: #3f5a8a;">https://scinet.usda.gov/guide/quickstart#hpc-clusters-on-scinet</a>.
</span>
</div><br>


**1.** Once logged in to Atlas HPC, go into your group folder in the `/project` location
```
cd /project/<your_account_folder>/

# e.g., cd /project/90daydata
```

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">
If you do not remember or do not know the name of your group directory in <b>/project</b> location, try the command: <br><br>
<span style="color: black;font-weight: 600;">
$ ls /project
</span><br><br>
That will display all available group directories. You can search for suitable ones by name or you can quickly filter out only the ones you have access to: <br><br>
<span style="color: black;font-weight: 600;">
$ ls /project/* 2> /dev/null
</span><br><br></span>
</div><br>

<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Note that you do NOT have access to all directories in the <b>/project</b> location. You also can NOT create the new directory there on your own. All users have access to <b>/project/90daydata</b>, but data is stored there only for 90 days and the folder is dedicated primarily to collaborative projects between groups. If you do NOT have access to your group's directory or need a directory for a new project <b><a href="https://scinet.usda.gov/support/request-storage#to-request-a-new-project-allocation" style="color: #3f5a8a;">request a new project allocation</a></b>.
</span>
</div><br>


**2.** Create a working directory (`mkdir`) for all future ODM projects and get into it (`cd`):

```
mkdir ODM
cd ODM
```

**3.** Create a directory for input images (IMAGES) and ODM analysis outputs (RESULTS):
```
mkdir IMAGES RESULTS
```

## **Download the ODM module**

Make sure you are in your ODM working directory at the **/project** path:

```
pwd
```

It should return a string with your current location, something like **/project**/project_account/user/**ODM**. If the basename of your current directory is different from "ODM" use the `cd` command to get into it. When you get to the right location in the file system follow the next instructions.


**Download the ODM docker image using the singularity module:**
```
module load singularity
singularity pull --disable-cache  docker://opendronemap/odm:latest
```

<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
<b>Do it only once (!)</b> when the first time you configure your work with the command-line ODM module. Once created, the singularity image of an ODM tool can be used any number of times.
</span>
</div><br>

Executing the code in the command line should create a new file named `odm_latest.sif`. This is an image of an ODM tool whose configuration ensures that it can be used efficiently on an HPC cluster. You can check the presence of the created file using the `ls` command, which will display the contents of the current directory.

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span><br>
You can display the contents of any directory while being in any other location in the file system. To do this, you need to know the relative or absolute path to the target location. <br><br>
The <b>absolute path</b> requires defining all intermediate directories starting from the root (shortcutted by a single <b>/</b>): <br>
$ <b>ls /project/90daydata</b> <br><br>
The <b>relative path</b> requires defining all intermediate directories relative to the current location. To indicate the parent directories use the <b>../</b> syntax for each higher level. To point to child directories you must name them directly. Remember, however, that pressing the tab key expands the available options, so you don't have to remember entire paths. <br>
$ <b>ls ../../inner_folder</b> <br><br>
The same principle applies to relocation in the file system using the <b>cd</b> command.
</div><br>


## **Copy input imagery on Atlas**

### *A. export from local machine*

In case the input images are on your local machine, you can transfer them to HPC Atlas via the command line using the `scp` command with syntax: <br> `scp <location on local machine> <user.name>@atlas-dtn.hpc.msstate.edu:<project location on Atlas cluster>`.

The complete command should look like that:
```
scp /local/mashine/JPGs/* alex.badacz@atlas-dtn.hpc.msstate.edu:/project/isu_gif/Alex/ODM/IMAGES/project-X
```

...and it has to be executed in the terminal window from the selected location in the file system on your local machine.


<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span><br>
Note that you HAVE to use <b>transfer nodes</b> every time you want to move data to a cluster. Transfer nodes <u>differ</u> from login nodes <u>by the hostname</u> while your user.name remains the same. <br>
For data transfer purposes, always use the transfer hostname:
user.name@<b>atlas-dtn.hpc.msstate.edu</b>
</span>
</div><br>

You can also transfer data from your local machine to the Atlas cluster using the web-based Globus approach. Learn more by following the tutorial in the DataScience workbook: [Copying Data using Globus](https://datascience.101workbook.org/07-DataParsing/01-FILE-ACCESS/02-2-tutorial-copy-globus).

### *B. import from other HPC*

You can use the approach from section A to <b>export data from any computing machine</b> (including another HPC, e.g., Ceres) to an Atlas cluster. You need to be physically logged into that other machine and follow the procedure described in step A.

If you want to make a transfer from another machine <u>while logged into Atlas</u> then you will <b>import the data</b>. You can also do this using the `scp` command, but you NEED to know the <u>hostname</u> for that other external machine. The idea of syntax is simple: `scp source_location destination_location`.

```
scp username@external-hostname:/path/JPGs/* /project/project_account/on/Atlas/ODM/IMAGES/project-X
```

Sometimes an external machine may require access from a specific port, in which case you must use the `-P` option, i.e., `scp -P port_number source_host:location destination_location_on_atlas`.

You can probably transfer data from other HPC infrastructure to the Atlas cluster using the web-based Globus approach. Learn more by following the tutorial in the DataScience workbook: [Copying Data using Globus](https://datascience.101workbook.org/07-DataParsing/01-FILE-ACCESS/02-2-tutorial-copy-globus).

### *C. move locally on Atlas*

To move data locally in the file system on a given machine (such as an Atlas cluster) use the `cp` command with the syntax: <br> `cp source_location destination_location`.

The complete command should look like that:
```
cp /project/90daydata/shared/project-X/JPGs/* /project/project_account/user/ODM/IMAGES/project-X
```

...and it has to be executed in the terminal window when logged into Atlas cluster.

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span><br>
Absolute paths work regardless of the current location in the file system. If you want to simplify the path syntax, first go to the <u>source</u> or <u>destination</u> location and replace them with <b>./*</b> or <b>./</b> respectively. An asterisk (*) means that all files from the source location will be transferred. <br><br>
<b>transfer while in the source location: </b> cp ./* /project/.../ODM/IMAGES/project-X <br><br>
<b>transfer while in the destination location: </b> cp /project/90daydata/project-X/JPGs/* ./
</span><br>
</div><br>


## **Setup SLURM script**

**Create an empty file for the SLURM script and open it with your favorite text editor:**

```
touch run_odm_latest.sh
nano run_odm_latest.sh           # nano, vim, mcedit are good text file editors

```

**Copy-paste the template code provided below:**

```
#!/bin/bash

# job standard output will go to the file slurm-%j.out (where %j is the job ID)
# DEFINE SLURM VARIABLES
#SBATCH --job-name="geo-odm"                   # custom SLURM job name visible in the queue
#SBATCH --partition=atlas                      # partition: atlas, gpu, bigmem, service
#SBATCH --nodes=1                              # number of nodes
#SBATCH --ntasks=48                            # 24 processor core(s) per node X 2 threads per core
#SBATCH --time=04:00:00                        # walltime limit (HH:MM:SS)
#SBATCH --account=<project_account>            # EDIT ACCOUNT, provide your SCINet project account

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load singularity                        # load container dependency
module load python/3.9.2                       # load phython 3.9 (default)

# DEFINE CODE VARIABLES
workdir=/project/<project_account>/.../ODM     # EDIT PATH, path to your ODM directory
project=PROJECT-1                              # EDIT PROJECT NAME, name of the directory with input JPG imagery
tag=`date +%Y%b%d-%T | tr ':' '.'`             # EDIT CUSTOM TAG, by default it is a date in format 2022Aug16-16.57.59
images_dir=$workdir/IMAGES/$project            # automatically generated path to input images when stored in ~/ODM/IMAGES; otherwise provide absolute path
output_dir=$workdir/RESULTS/$project-$tag      # automatically generated path to project outputs
mkdir -p $output_dir/code/images               # automatically generated images directory
ln -s $images_dir/* $output_dir/code/images/   # automatically linked input imagery
cp $BASH_SOURCE $output_dir/submit_odm.sh      # automatically copied the SLURM script into the outputs directory (e.g., for future reuse or reference of used options)

# DEFINE ODM COMMAND
singularity run --bind $images_dir:$output_dir/code/images, --writable-tmpfs odm_latest.sif  \
--orthophoto-png --mesh-octree-depth 12 --ignore-gsd --dtm \
--smrf-threshold 0.4 --smrf-window 24 --dsm --pc-csv --pc-las --orthophoto-kmz \
--ignore-gsd --matcher-type flann --feature-quality ultra --max-concurrency 16 \
--use-hybrid-bundle-adjustment --build-overviews --time --min-num-features 10000 \
--project-path $output_dir #--geo $output_dir/code/images/geo.txt
```

<br><span style="color: #ff3870; font-weight: 600; font-size:24px;">
Each time before submitting the script to the queue...
</span><br>

### *A. Adjust script variables and paths*

<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Follow the adjustment steps <u>each time</u> before submitting the job into the SLURM queue. Note that you SHOULD use the same script file every time you arrange an ODM analysis. For your convenience, when you submit a job to the queue, the script with all the settings is automatically copied to the corresponding folder of ODM analysis outputs (located directly in the RESULTS directory).
</span>
</div><br>

<span style="font-weight: 500; font-size:22px;"><i>^ <b>Adjust</b> the script lines marked with <b># EDIT</b> comment</i></span><br>

**0.** Select Atlas partition in section **# DEFINE SLURM VARIABLES** (optional)

<div style="background: #f0f0f0; padding: 15px;">
#SBATCH --partition=
<span style="font-weight:800;">atlas</span>
</div><br>

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span><br>
For most jobs, Atlas users should specify the <b>atlas</b> partition. The specification for all available Atlas partitions is provided in <b><a href="https://www.hpc.msstate.edu/computing/atlas/" style="color: #3f5a8a;">Atlas Documentation</a></b>, in section <i>Available Atlas Partitions</i>.
</div><br>

**1.** Enter your SCINet project account in section **# DEFINE SLURM VARIABLES** (obligatory)

<div style="background: #f0f0f0; padding: 15px;">
#SBATCH --account=
<span style="font-weight:800;">project_account</span>
</div><br>

For example, I use `isu_gif_vrsc` account: `#SBATCH --account=isu_gif_vrsc`

**2.** Edit path to your ODM directory in section **# DEFINE CODE VARIABLES** (obligatory)

<div style="background: #f0f0f0; padding: 15px;">
workdir=
<span style="font-weight:800;">/project/project_account/.../ODM</span>
</div><br>

For example, I use the following path: `workdir=/project/isu_gif_vrsc/Alex/geospatial/ODM`

**3.** Edit name of the directory with input imagery in section **# DEFINE CODE VARIABLES** (obligatory)

<span style="color: #ff3870;font-weight: 600;">IMPORTANT:</span>
This step determines which set of images will be used in the ODM analysis!

<div style="background: #f0f0f0; padding: 15px;">
project=
<span style="font-weight:800;">PROJECT-1</span>
</div><br>

<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
CASE 1: Note that the entered name should match the subdirectory existing directly in your ~/ODM/IMAGES/ where you store imagery for a particular analysis. Then you do NOT need to alter the <b>images_dir</b> variable (with the full path to the input photos) because it will be created automatically. <br>
<b>Keep images_dir default:</b> <br>
images_dir=$workdir/IMAGES/$project <br><br>

CASE 2: Otherwise, give a customized (any) name for the project outputs but remember <u>to provide</u> the absolute path (of any location in the HPC file system) to the input photos in the <b>images_dir</b> variable. <br>
<b>Provide the absolute path to imagery:</b> <br>
images_dir=/aboslute/path/to/input/imagery/in/any/location
</span>
</div><br>

**4.** Edit tag variable to customize project outputs directory **# DEFINE CODE VARIABLES** (optional)

By default, the `tag` variable is tagging the name of the directory with the ODM analysis outputs by adding the date and time (in the format: 2022Aug16-16:57:59) when the job is submitted. This prevents accidental overwriting of results for a project started from the same input images.

<div style="background: #f0f0f0; padding: 15px;">
tag=
<span style="font-weight:800;">`date +%Y%b%d-%T | tr ':' '.'`</span>
</div><br>

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span><br>
You can overwrite the value of the <b>tag</b> variable in any way that will distinguish the analysis variant and make the name of the new folder unique. <br>
Avoid overwriting the tag with manually typed words, and remember to always add an automatically generated randomization part in the variable to prevent overwriting the results in a previous project (for example, when you forget to update the tag).
</div><br>


### *B. Choose ODM options for analysis*

<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
The script template provided in this section has a <b>default configuration</b> of options available in the command-line ODM module. You may find that these settings are not optimal for your project. Follow the instructions in this section to learn more about the <b>available ODM options and their possible values</b>.
</span>
</div><br>


<span style="font-weight: 500; font-size:22px;"><i>^ <b>Adjust</b> flags in the <b># DEFINE ODM COMMAND</b> section in the script file</i></span><br>

```
# DEFINE ODM COMMAND
singularity run --bind $images_dir:$output_dir/code/images, --writable-tmpfs odm_latest.sif  \
--feature-quality ultra \                                           # feature
--pc-csv --pc-las \                                                 # point cloud
--mesh-octree-depth 12 \                                            # mesh
--gcp $output_dir/code/images/geo.txt \                             # georeferencing
--dsm --dtm --smrf-threshold 0.4 --smrf-window 24 \                 # 3D model
--orthophoto-png --orthophoto-kmz --build-overviews \               # orthophoto
--use-hybrid-bundle-adjustment --max-concurrency 16 --ignore-gsd \  # performance
--project-path $output_dir \                                        # project path
--time                                                              # runtime info
```

The syntax of the first line launches via the singularity container the odm image. All further --X flags/arguments define the set of options used in photogrammetry analysis. For clarity and readability, a long command line has been broken into multiple lines using the special character, backslash ` \ `. Thus, be careful when adding or removing options. <br>*The order of the options entered does not matter but they have been grouped by their impact on various outputs.*

You can find a complete <b>list of all available options</b> with a description in the official OpenDroneMap documentation: [v2.8.8](https://docs.opendronemap.org/arguments/).

**MANAGE WORKFLOW**

**A.** To <b>end processing at selected stage</b> use `--end-with` option followed by the keyword for respective stage:
* `dataset`
* `split`
* `merge`
* `opensfm`
* `openmvs`
* `odm_filterpoints`
* `odm_meshing`
* `mvs_texturing`
* `odm_georeferencing`
* `odm_dem`
* `odm_orthophoto`
* `odm_report`
* `odm_postprocess` [*default*]

**B.** There are several options to restart workflow:

* To <b>restart the selected stage</b> only and stop use `--rerun` option followed by the keyword for respective stage (see list in section A).
* To resume processing <b>from selected stage to the end</b>, use `--rerun-from` option followed by the keyword for respective stage (see list in section A).
* To permanently <b>delete all</b> previous results <b>and rerun</b> the processing pipeline use `--rerun-all` flag.

**C.** For <b>fast generation of orthophoto</b> skip dense reconstruction and 3D model generation using `--fast-orthophoto` flag. It creates the orthophoto directly from the sparse reconstruction which saves the time needed to build a full 3D model.

**D.** Skip individually other stages of the workflow:

* Skip generation of a full 3D model with `--skip-3dmodel` flag in case you only need 2D results such as orthophotos and DEMs. *Saves time!*
* Skip generation of the orthophoto with `--skip-orthophoto` flag n case you only need 3D results or DEMs. *Saves time!*
* Skip generation of PDF report with `--skip-report` flag in case you do not need it. *Saves time!*


**PHOTO ALIGNMENT options**

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|--feature-type|akaze / hahog / orb / sift|sift|algorithm for extracting keypoints and computing descriptors||
|**--min-num-features**|integer|10000|minimum number of features to extract per image|*More features ~ more matches between images. Improves reconstruction of areas with little overlap or insufficient features.* <br>***More features slow down processing.***|
|**--feature-quality**|ultra / high / medium / low / lowest|high|levels of feature extraction quality|*Higher quality generates better features, but requires more memory and takes longer.*|
|--resize-to|integer|2048|resizes images by the largest side for feature extraction purposes only|*Set to <b>-1</b> to disable or use <b>--feature-quality</b> instead. This does not affect the final orthophoto resolution quality and will not resize the original images.*|
|--matcher-neighbors|positive integer|0|performs image matching with the nearest N images based on GPS exif data|*Set to **0** to match by triangulation.*|
|--matcher-type|bow / bruteforce / flann|flann|image matcher algorithm|*FLANN is slower, but more stable. <br>BOW is faster, but can sometimes miss valid matches. <br>BRUTEFORCE is very slow but robust.*|


**SfM & DPC options**

Structure from Motion (SfM) algorithm estimates camera positions in time (motions) and generates a 3D Dense Point Cloud (DPC) of the object from multi-view stereo (MVS) photogrammetry on the set of images.

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|--sfm-algorithm|incremental / triangulation / planar |incremental|structure from motion algorithm (SFM)|*For aerial datasets, if camera GPS positions and angles are available, triangulation can generate better results. For planar scenes captured at fixed altitude with nadir-only images, planar can be much faster.*|
|--depthmap-resolution|positive float|640|controls the density of the point cloud by setting the resolution of the depthmap images|*Higher values take longer to compute but produce denser point clouds.* <br>***Overrides --pc-quality***|
|--pc-quality|ultra / high / medium / low / lowest|medium|the level of point cloud quality|*Higher quality generates better, denser point clouds, but requires more memory and takes longer (~4x/level).*|
|--pc-geometric||off|improves the accuracy of the point cloud by computing geometrically consistent depthmaps|*Increases processing time, but can improve results in urban scenes.*|
|--pc-rectify||off|performs ground rectification on the point cloud|*The wrongly classified ground points will be re-classified and gaps will be filled.* <br>***Useful for generating DTMs.***|
|--pc-filter|positive float|2.5|filters the point cloud by removing points that deviate more than N standard deviations from the local mean|**0** *- disables filtering*|
|--pc-sample|positive float|0.0|filters the point cloud by keeping only a single point around a radius N [meters]|*Useful to limit the output resolution of the point cloud and remove duplicate points.* <br>**0** *- disables sampling*|
|--pc-copc||off|exports the georeferenced point cloud|Cloud Optimized Point Cloud (COPC) format|
|--pc-csv||off|exports the georeferenced point cloud|CSV format|
|--pc-ept||off|exports the georeferenced point cloud|Entwine Point Tile (EPT) format|
|--pc-las||off|exports the georeferenced point cloud|LAS format|


**MESHING & TEXTURING options**

|flag&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|**--mesh-octree-depth**|integer: <br>1 <= x <= 14|11|octree depth used in the mesh reconstruction|
|--mesh-size|positive integer|200000|the maximum vertex count of the output mesh|
|--texturing-data-term|gmi / area|gmi|texturing feature|*When texturing the 3D mesh, for each triangle, choose to prioritize images with sharp features (gmi) or those that cover the largest area (area).*|
|--texturing-keep-unseen-faces||off|keeps faces in the mesh that are not seen in any camera||
|--texturing-outlier-removal-type|none / gauss_clamping / gauss_damping|gauss_clamping|type of photometric outlier removal method||
|--texturing-skip-global-seam-leveling||off|skips normalization of colors across all images|*Useful when processing radiometric data.*|
|--texturing-skip-local-seam-leveling||off|skips the blending of colors near seams||
|--texturing-tone-mapping|none | gamma|none|turns on gamma tone mapping or none for no tone mapping||


**GEOREFERENCING options**

|flag&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|--force-gps|| off|uses images’ GPS exif data for reconstruction, even if there are GCPs present|*Useful if you have high precision GPS measurements.* <br>***If there are no GCPs, it does nothing.***|
|--gcp|*PATH* string|none|path to the file containing the ground control points used for georeferencing||
|--use-exif||off|EXIF-based georeferencing|*Use this tag if you have a GCP File but want to use the EXIF information for georeferencing instead.*|
|--geo|*PATH* string|none|path to the image geolocation file containing the camera center coordinates used for georeferencing|*Note that omega/phi/kappa are currently not supported (you can set them to 0).*|
|--gps-accuracy|positive float|10|value in meters for the GPS Dilution of Precision (DOP) information for all images|*If you use high precision GPS (RTK), this value will be set automatically. You can manually set it in case the reconstruction fails. Lowering the value can help control bowling-effects over large areas.*|


**DSM - Digital Surface Model options**

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|**--dsm**||off|builds DSM (ground + objects) using a progressive morphological filter|*Use -–dem⁕ parameters for finer tuning.*|
|--dem-resolution|float|5.0|DSM/DTM resolution in cm / pixel|*The value is capped to 2x the ground sampling distance (GSD) estimate.* <br> *^ use* ***–-ignore-gsd*** *to remove the cap*|
|--dem-decimation|positive integer|1|decimates the points before generating the DEM <br>**1** is no decimation (full quality) <br>**100** decimates ~99% of the points|*Useful for speeding up generation of DEM results in very large datasets.*|
|--dem-euclidean-map||RowanGaffney|computes an euclidean raster map for each DEM|*Useful to isolate the areas that have been filled.*|
|--dem-gapfill-steps|positive integer|3|number of steps used to fill areas with gaps <br>**0** disables gap filling|*see details in the [docs](https://docs.opendronemap.org/arguments/dem-gapfill-steps/#dem-gapfill-steps)*|

**DTM - Digital Terrain Model options**

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|**--dtm**||off|builds DTM (ground only) using a simple morphological filter|*Use -–dem⁕ and -–smrf⁕ parameters for finer tuning.*|
|**--smrf-threshold**|positive float|0.5|Simple Morphological Filter elevation threshold parameter [meters]|
|**--smrf-window**|positive float|18.0|Simple Morphological Filter window radius parameter [meters]|
|--smrf-slope|positive float|0.15|Simple Morphological Filter slope parameter (rise over run)|
|--smrf-scalar|positive float|1.25|Simple Morphological Filter elevation scalar parameter|


**ORTHOPHOTO options**

| flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|--orthophoto-resolution|float > 0.0|5.0|orthophoto resolution in cm / pixel|
|--orthophoto-compression|JPEG, LZW, LZMA, DEFLATE, PACKBITS, NONE|DEFLATE|compression to use for orthophotos|
|--orthophoto-cutline||off|generates a polygon around the cropping area that cuts the orthophoto around the edges of features|*The polygon can be useful for stitching seamless mosaics with multiple overlapping orthophotos.*|
|**--orthophoto-png**||off|generates rendering of the orthophoto|*PNG format*|
|**--orthophoto-kmz**||off|generates rendering of the orthophoto|*Google Earth (KMZ) format*|
|--orthophoto-no-tiled||off|generates striped GeoTIFF|
|**--build-overviews**||off|builds orthophoto overviews|*Useful for faster display in programs such as QGIS.*|
|--use-3dmesh||off|uses a full 3D mesh to compute the orthophoto instead of a 2.5D mesh|*This option is a bit faster and provides similar results in planar areas.*|


**GENERAL QUALITY OPTIMIZTION options**

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|--auto-boundary||off|automatically set a boundary using camera shot locations to limit the area of the reconstruction|*Useful to remove far away background artifacts (sky, background landscapes, etc.).*|
|--boundary|JSON file|none|GeoJSON polygon limiting the area of the reconstruction|*Can be specified either as path to a GeoJSON file or as a JSON string representing the contents of a GeoJSON file.*|
|--camera-lens| auto / perspective / brown / fisheye / spherical / dual / equirectangular|auto|camera projection type|*Manually setting a value can help improve geometric undistortion.*|
|--cameras|JSON file|none|camera parameters computed from another dataset|*Use params from text file instead of calculating them.* <br>*Can be specified either as path to a cameras.json file or as a JSON string representing the contents of a cameras.json file.*|
|--use-fixed-camera-params||off|turns off camera parameter optimization during bundle adjustment|*This can be sometimes useful for improving results that exhibit doming/bowling or when images are taken with a rolling shutter camera.*|
|--cog||off|creates cloud-optimized GeoTIFFs instead of normal GeoTIFFs||


**PERFORMANCE OPTIMIZATION options**

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|**--use-hybrid-bundle-adjustment**||off|runs local bundle adjustment for every image added to the reconstruction and a global adjustment every 100 images|*Speeds up reconstruction for very large datasets.*|
|**--max-concurrency**|positive integer|4|maximum number of processes to use in various processes|*Peak memory requirement is ~1GB per thread and 2 megapixel image resolution.*|
|--no-gpu||off|does not use GPU acceleration, even if it’s available||
|--optimize-disk-space||off|deletes heavy intermediate files to optimize disk space usage|*This disables restarting the pipeline from an intermediate stage, but allows the analysis on machines that don’t have sufficient disk space available.*|


**INPUT / OUTPUT MANAGEMENT options**

|flag&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;| values | default | description |notes|
|-----|--------|---------|-------------|-----|
|--project-path|*PATH* string|none|path to the project folder|*Your project folder should contain subfolders for each dataset. Each dataset should have an “images” folder.*|
|--name|*NAME* string|code|name of dataset|*That is the ODM-required subfolder within project folder.*|
|--ignore-gsd||off|ignores Ground Sampling Distance (GSD)|*GSD caps the maximum resolution of image outputs and resizes images, resulting in faster processing and lower memory usage. Since GSD is an estimate, sometimes ignoring it can result in slightly better image output quality.*|
|--crop|positive float|3|crop image outputs by creating a smooth buffer around the dataset boundaries, shrunk by N meters|*Use* ***0*** *to disable cropping.*|
|--copy-to|*PATH*|none|copies output results to this folder after processing||

<br>

See description of other options directly in the OpenDroneMap documentation: <br>
* general usage: [help](https://docs.opendronemap.org/arguments/help/#help), [debug](https://docs.opendronemap.org/arguments/debug/#debug), <br>
* large datasets: [split](https://docs.opendronemap.org/arguments/split/#split), [split-image-groups](https://docs.opendronemap.org/arguments/split-image-groups/#split-image-groups), [split-overlap](https://docs.opendronemap.org/arguments/split-overlap/#split-overlap),
* multispectral datasets: [primary-band](https://docs.opendronemap.org/arguments/primary-band/#primary-band), [radiometric-calibration](https://docs.opendronemap.org/arguments/radiometric-calibration/#radiometric-calibration), [skip-band-alignment](https://docs.opendronemap.org/arguments/skip-band-alignment/#skip-band-alignment), <br>
* rolling-shutter camera: [rolling-shutter](https://docs.opendronemap.org/arguments/rolling-shutter/#rolling-shutter), [rolling-shutter-readout](https://docs.opendronemap.org/arguments/rolling-shutter-readout/#rolling-shutter-readout)


## **Submit ODM job into the SLURM queue**

The SLURM is a workload manager available on the Atlas cluster. It is a simple Linux utility for resource management and computing task scheduling. In simple terms, you HAVE to use it every time you want to outsource some computation on HPC infrastructure. To learn more about SLURM take a look at the tutorial [SLURM: Basics of Workload Manager](https://datascience.101workbook.org/06-IntroToHPC/05-JOB-QUEUE/01-SLURM/01-slurm-basics) available in the [DataScience Workbook](https://datascience.101workbook.org).

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span><br>
If you are working on an HPC infrastructure that uses the PBS workload manager, take a look at the tutorial <b><a href="https://datascience.101workbook.org/06-IntroToHPC/05-JOB-QUEUE/02-PBS/01-pbs-basics" style="color: #3f5a8a;">PBS: Portable Batch System</a></b> to learn more about the command that sends a task to the queue and the script configuration. <br>
[<i><a href="https://datascience.101workbook.org" style="color: #3f5a8a;">source: DataScience Workbook</a></i>]
</div><br>

Use the `sbatch` SLURM command to submit the computing job into the queue:

```
sbatch run_odm_latest.sh
```

<div style="background: mistyrose; padding: 15px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Tasks submitted into the queue are sent to <b>powerful compute nodes</b>, while all the commands you write in the terminal right after logging in are executed on the <b>capacity-limited login node</b>.<br><br>
<b>Never perform an aggravating computation on a logging node.</b><br>
A. If you want to optimize some computationally demanding procedure and need a live preview for debugging, start an <b><a href="https://datascience.101workbook.org/06-IntroToHPC/05-JOB-QUEUE/01-SLURM/01-slurm-introduction/interactive-session" style="color: #3f5a8a;">interactive session on the compute node</a></b>.<br>
B. If you want to migrate a large amount of data use transfer node: <b>@atlas-dtn.hpc.msstate.edu</b>.
</span>
</div><br>

## **Access ODM analysis results**

The figure shows the file structure of all outputs generated by the ODM command-line module. The original screenshot comes from the official [OpenDroneMap (v2.8.7) Documentation](https://docs.opendronemap.org/outputs/#list-of-all-outputs).

![OpenDroneMap outputs](../assets/images/odm_outputs.png)


# Get ODM on local machine

**A. Download docker image using singularity** <br>
 *^ suggested for usage on computing machines where the singularity is available*

 ```
 module load singularity
 singularity pull --disable-cache  docker://opendronemap/odm:latest
 ```

**B. Download docker image using Docker** <br>
*^ suggested for usage on computing machines where the Docker can be installed* <br>
*^ requires Docker installation*

 ```
 # Windows
 docker run -ti --rm -v c:/Users/youruser/datasets:/datasets opendronemap/odm --project-path /datasets project

 # Mac or Linux
 docker run -ti --rm -v /home/youruser/datasets:/datasets opendronemap/odm --project-path /datasets project
 ```

 **C. ODM at GitHub:** [https://github.com/OpenDroneMap/OpenDroneMap/](https://github.com/OpenDroneMap/OpenDroneMap/)

 ```
 git clone https://github.com/OpenDroneMap/ODM.git
 ```

**D. Download zipped source code:** [https://github.com/OpenDroneMap/OpenDroneMap/archive/master.zip](https://github.com/OpenDroneMap/OpenDroneMap/archive/master.zip)

 ```
 wget https://github.com/OpenDroneMap/OpenDroneMap/archive/master.zip
 ```



___
# Further Reading
* []()


___

[Homepage](../index.md){: .btn  .btn--primary}
[Section Index](../00-IntroPhotogrammetry-LandingPage){: .btn  .btn--primary}
[Previous](01-WebODM){: .btn  .btn--primary}
[Next](){: .btn  .btn--primary}
