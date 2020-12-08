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
   1. [PyCharm](https://www.jetbrains.com/pycharm/)
   2. [Anaconda](https://www.anaconda.com/)
   3. [Atom](https://atom.io/)
   4. ... there are other options
   5. A discussion on Python IDE - [https://realpython.com/python-ides-code-editors-guide/](https://realpython.com/python-ides-code-editors-guide/)
   6. Skip to [JupyterLab Notebook](https://jupyter.org/install)
3. Pick a Python Virtual Envirnoment Manager
   1. [conda](https://docs.conda.io/projects/conda/en/latest/) / [miniconda](https://docs.conda.io/en/latest/miniconda.html)
   2. [venv](https://docs.python.org/3/library/venv.html)

Regardless of IDE, a python environment should have a way to edit code. Many of the geospatial workbook tutorials are written in a Jupyter Notebook. This is an interactive report where blocks of text are either code, or documentation

1. **Code Block**
2. **Text Block**

Together these panes allow you to interactively design an Python pipeline.

### Hands-On Exercise - using Jupyter Notebooks

Copy the following into the **Code Editor**, run it line by line, and see if you can recreate the graph in the **Graphics View**.  If you can, try modifying and running the script. Experiment to see what changes break the script.

```python
#! /usr/bin/env python

import numpy as np

print("Hello World")
```

## Installing Python Libraries

R functions are made available as libraries (also referred to as packages) which need to be installed from somewhere. R libraries can be indexed on CRAN, bioconductor and GitHub. What's the difference between installing from these locations?

* **pypy** python libraries

  ```bash
  pip install libraryname
  ```

* **conda** python libraries are focused on bioinformatic analysis and may or may not be available on CRAN but can be the latest version of a tools. [Bioconductor Website](https://www.bioconductor.org/install/)

  ```R
  conda install -c bioconda libraryname
  ```

* **GitHub python libraries** tend to be the most recently developed libraries and may not have been submitted to Bioconductor or CRAN yet.

  ```R
  install.packages("devtools")
  devtools::install_github("username/reponame")      # Github R Library name
  ```

**Warning:** Do not mix python environments. Pick a python environment manager and stick with it. Ideally create a `environment.yml` for each project.

### Hands-On Exercise - installing Python libraries

Create a `environment.yml` as your basic python envirnment install. This can serve as our basic outline for any other python environment setup.

```bash
pip install python_library
conda install -c bioconda r-wgcna
conda env create -f environment.yml
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
