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

## Create file structure

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

## Download the ODM module

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


## Copy input imagery on Atlas

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




## Setup SLURM script

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
--ignore-gsd  --matcher-type flann --feature-quality ultra --max-concurrency 16 \
--use-hybrid-bundle-adjustment --build-overviews --time --min-num-features 10000 \
--project-path $output_dir
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

<span style="color: #ff3870;font-weight: 600;">section in development</span>

## Submit ODM job into the SLURM queue

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

## Access ODM analysis results

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
