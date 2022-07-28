---
title: "Introduction to Open Drone Map"
layout: single
author: Aleksandra Badaczewska
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /IntroPhotogrammetry/assets/images/geospatial_workbook_banner.png
---

{% include toc %}

# Introduction

[OpenDroneMap](https://www.opendronemap.org) is open-source software for processing aerial imagery. In simple terms, it enables digital 3D mapping of objects using overlapping photos showing the scene from different perspectives. Although there are many practical applications, here we will focus on the use of OpenDronMap for processing **drone-taken photos of agricultural land areas** for mapping crops and monitoring woodlands.

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">
<b>OpenDroneMap</b> is still an actively developing project. To stay up to date with recent releases and new features check out the blog: <a href="https://www.opendronemap.org/blog/" style="color: #3f5a8a;">https://www.opendronemap.org/blog/</a>.
</span>
</div><br>


OpenDroneMap project has gained scale that is accompanied by a growing community: <a href="https://community.opendronemap.org">https://community.opendronemap.org</a>.
<span style="color: #ff3870; font-weight: 600;">Do not hesitate to <u>actively join</u> both when you have questions or developed solutions that enrich the <u>user community</u>.</span>


# OpenDronMap Modules

**OpenDroneMap** is not a single software but rather a modular package that combines individual components which can be assembled in various configurations to meet the needs of a wide range of users and their computing setups.

![ODM modules](../assets/images/odm_modules.png)<br>
**Figure 1.** OpenDroneMap project includes several modules: **ODM** is the core of software that can be run directly from the command line; **NodeODM** is a communication channel between components; **ClusterODM** allows management of multiple ODM nodes; **WebODM** is a web-based graphical user interface; **CloudODM** faciliotates in-cloud computations; **pyODM** enables incorporation of ODM into the customized  applications.


## ODM

The **[ODM](https://www.opendronemap.org/odm/)** is analytical core of the OpenDroneMap software. It is an open source toolkit that actually process aerial images. The ODM module can be **run directly from the command line** on a local machine or remotely on HPC infrastructure. It generates a complete set of files organized in hierarchical directories (**point clouds, 3D textured models, georeferenced ortophotos and elevation models**). Unfortunately, in this approach there is **no support to create web tiles**. Thus, the analysis results cannot be directly opened in the complementary WebODM graphical interface. But the files can be still visualized in external software that supports the given format. The other modules of the package are higher-level layers that facilitate the use of the software, but they always call out functions from the built-in ODM core.

To learn how to run **ODM from the command line**, and specifically on HPC infrastructure such as SCINet clusters, go to the hands-on tutorial: [Command-line ODM modules](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/02-ODM-modules).

## WebODM

The **[WebODM](https://www.opendronemap.org/webodm/)** is a **web-based graphical user interface** (GUI) for the OpenDroneMap software. In the back-end, it employs by default a single NodeODM to facilitate communication with a built-in ODM core. It generates an identical set of outputs to ODM on the command line, but it also **automatically creates web tiles**, so it is possible to **visualize the results** directly in the graphical interface. Other features include project management and the ability to **reopen and/or restart tasks**. There is also a built-in tool for **creating grand control points**.<br>
In general, after start the GUI is available on localhost (port 8000) in a web browser directly on the machine on which WebODM is running. That's why it's a perfect solution **when working on a local machine**. WebODM can also be available **remotely on HPC infrastructure through the OpenOn Demand service**. Depending on the admins' configuration, it will probably have more processing nodes (multiple NodeODM instances) added, which will increase computing performance. *Adding more processing nodes enables to run of many jobs in parallel.*

To learn how to run **graphical interface of the WebODM** on your local machine or remotely using Open OnDemand service on HPC infrastructure (such as SCINet clusters), go to the hands-on tutorial: [WebODM: web-based graphical interface](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/01-WebODM).

<!--## NodeODM

## ClusterODM

## CloudODM

## pyODM


# Getting started with ODM

## Setup on local machine

## Remote access on HPC


# Prepare input imagery

## DNG to JPG conversion

## Keep EXIF/GEO metadata
-->

___
# Further Reading
* [WebODM](01-WebODM)
* [Command-line ODM modules](02-ODM-modules)


___

[Homepage](../index.md){: .btn  .btn--primary}
[Section Index](../00-IntroPhotogrammetry-LandingPage){: .btn  .btn--primary}
[Next](01-WebODM){: .btn  .btn--primary}
