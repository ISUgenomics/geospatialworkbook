---
title: "Metashape Python Scripting"
layout: single
header:
  overlay_color: "444444"
  overlay_image: /assets/images/nasa.png
---



## Introduction

Python scripting is supported only in Metashape Professional edition.

Metashape Professional uses Python 3.8.

### Metashape functionality available via Python

* Open/save/create Metashape projects.
* Add/remove chunks, cameras, markers.
* Add/modify camera calibrations, ground control data, assign geographic projections and coor- dinates.
* Performprocessingsteps(alignphotos,builddensecloud,buildmesh,texture,decimatemodel, etc...).
* Export processing results (models, textures, orthophotos, DEMs).
* Access data of generated models, point clouds, images.
* Start and control network processing tasks.

### Overview of Metashape module in Python

Documentation for [Metashape_python_api_1.7.3](https://www.agisoft.com/pdf/metashape_python_api_1_7_3.pdf) .

`import Metashape as ms` # select custom shortcut for the imported module

#### Global application attributes
`Metashape.app.<attribute>`

Metashape.Application class provides access to global attributes:

| attribute | DESCRIPTION | VALUE TYPE  |
|---------|---------|---------|
| *document*| main application document object | [document] |
| *enumGPUDevices* | enumerate installed GPU devices | [array] |
| *gpu_mask* | GPU device bit mask | [int] *1* - use device, *0* - do not use |
| *cpu_enable* | use cpu when GPU is active | [bool]<br>*False* - Disable CPU for GPU accelerated tasks <br>*True* - Enable CPU for GPU accelerated processing |
