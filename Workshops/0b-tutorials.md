---
title: Tutorials
layout: single
author: Kerrie Geil
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/pattern.png
---

{% capture text %}
We will stop approving new registrations about an hour before each session. Please register in advance of that cutoff so you don't get left out!
{% endcapture %} 
{% include alert.md text=text color='warning' %}

All tutorials are hosted on Github at (repo location coming soon). 

Below is a brief description of each tutorial as well as an instructional on how to download and run them on the SCINet Ceres HPC system.

{% capture text %}
* [**How to Run Tutorials on Ceres**](#how-to-run-tutorials-on-ceres)
* [**Session 2 Tutorial: Intro to the Ceres HPC System Environment (SSH, JupyterHub, Basic Linux, SLURM batch script)**](#session-2-tutorial-intro-to-the-ceres-hpc-system-environment-ssh-jupyterhub-basic-linux-slurm-batch-script)
* [**Session 3 Tutorial: Intro to Distributed Computing on the Ceres HPC System Using Python and Dask**](#session-3-tutorial-intro-to-distributed-computing-on-the-ceres-hpc-system-using-python-and-dask) 
* [**Session 4 Tutorial: Computational Reproducibility Tools (Git/Github, Conda, Docker/Singularity containers**](#session-4-tutorial-computational-reproducibility-tools-git-github-conda-docker-singularity-containers) 
* [**Session 5 Tutorial: Distributed Machine Learning: Using Gradient Boosting to Predict NDVI Dynamics**](#session-5-tutorial-distributed-machine-learning-using-gradient-boosting-to-predict-ndvi-dynamics)
{% endcapture %}
{% include card.md header="Table of Contents" text=text %}
<br><br>

# How to Run Tutorials on Ceres
<br>
For Sessions 2 & 4 tutorials ("Intro to Ceres" and "Computational Reproducibility Tools") login to your SCINet account using SSH (or through JupyterHub and then open a terminal) and work through the html tutorials from the command line (links in the tutorial sections below).

For Sessions 3 & 5 tutorials ("Intro to Distributed Computing" and "Distributed Machine Learning") follow the instructions in this section to open the tutorial in a Jupyter Notebook:

1. Login to your SCINet/Ceres account through the JupyterHub web interface
   * Go to [https://jupyterhub.scinet.usda.gov](https://jupyterhub.scinet.usda.gov)
   * Login to the system with your SCINet credentials
   * Submit the Spawning Page with the following information (if not specified below, leave blank):<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Node Type: ```short```<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of Cores: ```4```<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Job Duration: ```02:00:00```<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to the container image: ```/lustre/project/geospatial_tutorials/wg_2020_ws/data_science_im_rs_vSCINetGeoWS_2020.sif```<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Container Exec Args: ```--bind /etc/munge --bind /var/log/munge --bind /var/run/munge --bind /usr/bin/squeue --bind /usr/bin/scancel --bind /usr/bin/sbatch --bind /usr/bin/scontrol --bind /usr/bin/sinfo --bind /system/slurm:/etc/slurm --bind /run/munge --bind /usr/lib64/libslurm.so --bind /usr/lib64/libmunge.so.2 --bind /usr/lib64/slurm --bind  /project --bind /lustre```
   
    After a few minutes, a JupyterLab instance, running on the Ceres HPC, should open in your browser. After several attempts, if the spawner fails and JupyterLab does not open correctly, please contact the SCINet Virtual Research Support Core (VRSC) for assistance at scinet_vrsc@usda.gov.

2. Download the tutorial material from the workshop GitHub repo
   * Open a terminal: File-->New-->terminal (alternatively, click the "+" icon (launcher) on the left and then choose the "terminal" icon on the launcher screen) 
   * Download the tutorials
      * instruction coming soon 
      <!-- ```bash
      git clone --single-branch https://github.com/kerriegeil/SCINET-GEOSPATIAL-RESEARCH-WG.git
      ```-->
3. Run a notebook:
   * instruction coming soon 
  
<!--
   * You should now see a folder (file system extension on the left hand side of JuputerLab) titled *SCINET-GEOSPATIAL-RESEARCH-WG*.
   * Navigate to ```/SCINET-GEOSPATIAL-RESEARCH-WG/tutorials/```
   * Open the desired tutorial
   * Select the py_geo kernel (upper right corner in the notebook)
   * Execute blocks of script by clicking the "play" icon in the notebook or typing Shift+Enter 
-->
<br><br>

# Session 2 Tutorial: Intro to the Ceres HPC System Environment (SSH, JupyterHub, Basic Linux, SLURM batch script)
<br>
**Link to HTML (static) Material:** coming soon <!--[Session 2 Tutorial](/link-to-tutorial)--><br>

 Login to your SCINet account using SSH (or through JupyterHub and then open a terminal) and work through the html tutorial from the command line.

**Learning Goals:**

- access the SCINet Ceres HPC system by using Secure Shell at the command line
- access the SCINet Ceres HPC system using the JupyterHub web interface
- access JupyterLab and RStudio on the Ceres HPC through the JupyterHub web interface
- basic linux commands
- how to write a SLURM batch script to submit a compute job on the Ceres HPC
<br><br>

# Session 3 Tutorial: Intro to Distributed Computing on the Ceres HPC System Using Python and Dask
<br>
**Link to Jupyter Notebook on Github:** coming soon <br>
**Link to HTML (static) Material:** coming soon <!--[Session 3 Tutorial](/link-to-tutorial)--><br>

To launch JupyterHub and download the workshop materials, see the [How to Run Tutorials on Ceres](#how-to-run-tutorials-on-ceres) section above.

Run the notebook titled *intro-to-python-dask-on-ceres.ipynb*

**Learning Goals:**

- data analysis technique for very large datasets (the tools in this tutorial are most appropriate for analysis of large earth-science-type datasets)
- set up SLURM cluster to compute "in parallel"
- scale clusters for heavy compute
- use adaptive clusters to dynamically scale up and down your computing
- view Dask diagnostics to visualize cluster computations in real time
<br><br>

# Session 4 Tutorial: Computational Reproducibility Tools (Git/Github, Conda, Docker/Singularity containers)
<br>
**Link to HTML (static) Material:** coming soon <!--[Session 4 Tutorial](/link-to-tutorial)--><br>

 Login to your SCINet account using SSH (or through JupyterHub and then open a terminal) and work through the html tutorial from the command line. **To follow along with the Git/Github portion of this tutorial you must [create a free Github account](https://github.com/join) for yourself ahead of time and remember your Github username and password.**

**Learning Goals:**

- versioning and archiving your codes with Git and Github
    - fork/copy an existing online repo to your SCINet/Ceres account
    - push your local fork/repo online to your own Github account
    - make an edit to your fork/repo
    - make a pull request to have your edits incorporated into the original repo
- use the Conda package/environment management system on the Ceres HPC system
    - access or install Conda on Ceres
    - use Conda to download software on Ceres
    - use Conda environments to document all the software you are using and eliminate dependency issues
    - save your Conda environment details to a specification file so that you can quickly recreate your complete software environment for any project
- how containers can allow your codes to run successfully on different operating systems
    - use (and create) a Docker image
    - use Singularity on the Ceres HPC to run a container from a Docker image
<br><br>

# Session 5 Tutorial: Distributed Machine Learning: Using Gradient Boosting to Predict NDVI Dynamics
<br>
**Link to Jupyter Notebook on Github:** coming soon <br>
**Link to HTML (static) Material:** coming soon <!--[Session 5 Tutorial](/link-to-tutorial)--><br>

To launch JupyterHub and download the workshop materials, see the [How to Run Tutorials on Ceres](#how-to-run-tutorials-on-ceres) section above.

Run the notebook titled *Machine_Learning_Tutorial.ipynb*

**Learning Goals:**

- use a gradient boosting machine learning model to predict NDVI
- set up a cluster on Ceres (Dask Distributed)
- read data and interpolate onto a consistent grid (Xarray, Dask Dataframe)
- merge/shuffle/split the data (Dask_ML, Scikit Learn)
- optimize the hyperparameters (Dask_ML, Scikit Learn, XGBoost)
- train a distributed XGBoost model (Scikit Learn, XGBoost, Dask Distributed, datashader)
- quantify the accuracy and visualize the results (Scikit Learn, SHAP)
