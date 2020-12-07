---
title: Python Setup
layout: single
author: Jennifer Chang
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

# Introduction

Python is a dynamically typed, object-oriented scripting language developed by Guido van Rossum and released in 1991.

* [Wikipedia - Python (programming language)](https://en.wikipedia.org/wiki/Python_(programming_language)) 
* [List of Python projects on GitHub](https://github.com/topics/python)

# Getting Started in Python

First you will need to install Python or have access to an Python environment. Due to incompatabilities between certain python libraries, it is highly recommended to manage python packages in virtual environments (`conda` , `venv`, or equivalent) , one environment per project. 

## Setup Python Locally

1. Install the latest version of Python from their website - [https://www.python.org/downloads/](https://www.python.org/downloads/) 
2. Pick an IDE (integrated development Environment)
   1. PyCharm
   2. Anaconda
   3. Atom
   4. ... there are other options
3. Pick a Python Virtual Envirnoment Manager
   1. conda
   2. venv

When you first open RStudio, you'll notice several panes. Note: To see the Code Editor, you may need to go to Rstudio's  `File/New File/R Script`.

1. **Code Editor** - The top left pane is where you will be editing an R or R markdown script. You can run commands line by line by placing cursor on a line and hitting `Ctrl+Enter` or `Cmd+Return`. Usually scripts are designed (and debugged) in RStudio but then run form the command line when you are processing multiple files.
2. **R Console** - R code is run in the bottom left pane. This will display any messages, warnings and error messages. This is useful for copying and pasting the error/warning into a search engine to debug a pipeline.
3. **Data and Variables** - This top right pane lists any loaded datasets and functions. If you double-click on `data | 10 obs. of 2 variables`, an excel-like view of the data is shown. Check the this pane to make sure data is loaded and processed correctly. If items have 0 obs, then the dataset has been completely filtered out.
4. **Graphics View** - Any plots will show up on the bottom right corner in the Plots pane.

Together these panes allow you to interactively design an R pipeline.

### Hands-On Exercise - using Jupyter Notebooks

Copy the following into the **Code Editor**, run it line by line, and see if you can recreate the graph in the **Graphics View**.  If you can, try modifying and running the script. Experiment to see what changes break the script.

```R
#! /usr/bin/env Rscript

install.packages("ggplot2")
library(ggplot2)

x <- c(1:10)
y <- c(2:11)
data <- data.frame(x = x, y = y)

ggplot(data, aes(x = x, y = y)) +
  geom_point() +
  labs(title = "Hello World")
```

## Installing Python Libraries

R functions are made available as libraries (also referred to as packages) which need to be installed from somewhere. R libraries can be indexed on CRAN, bioconductor and GitHub. What's the difference between installing from these locations?

* **pypy** python libraries 

  ```R
  cran_pkgs = c("ggplot2", "devtools")    # List one or multiple libraries
  install.packages(cran_pkgs)
  library(ggplot2)                        # Load the library
  ```

* **conda** python libraries are focused on bioinformatic analysis and may or may not be available on CRAN but can be the latest version of a tools. [Bioconductor Website](https://www.bioconductor.org/install/)

  ```R
  install.packages("BiocManager")
  biocond_pkgs = c ("wgcna", "deseq2")    # List one or multiple libraries
  BiocManager::install(biocond_pkgs)
  ```

* **GitHub python libraries** tend to be the most recently developed libraries and may not have been submitted to Bioconductor or CRAN yet.

  ```R
  install.packages("devtools")
  devtools::install_github("username/reponame")      # Github R Library name
  ```

In order of preference, first attempt to install from CRAN. If the library is not available on CRAN, check Bioconductor, then the GitHub repo.

![R Libraries](images/R_libraries.png)

**Warning:** Have a regular schedule (maybe once every 6 months) to keep your R libraries up to date. Some libraries depend on other libraries and will not install until you have the latest version.

### Hands-On Exercise - installing Python libraries

Write an R script to install one package each from CRAN, bioconductor, and GitHub. Use a search engine (like Google) to find a library. If one library doesn't install, copy the error message and paste into the search engine and try to figure out why. Don't get hung up on getting it to install, this is practice.

```bash
pip install python_library
conda install -c bioconda r-wgcna
```

## Using Python on SCINet

On the SCINet HPC resources (Ceres and Atlas), R should be available as a module.

```bash
module load miniconda
conda env create -f environment.yml
```

However since this will load base R without the R libraries, you may need to install your own R libraries which are installed to home directory by default.

## Using JupyterLab on SCINet

Recently, Ceres HPC was configured to run JupyterLab. This allows you to run an Jupyter Notebook like interface to the Ceres HPC. See full instructions on the SCINet website: [https://scinet.usda.gov/guide/jupyter/](https://scinet.usda.gov/guide/jupyter/)

