---
title: "Image Machine Learning Workshop"
layout: single
header:
  overlay_color: "444444"
  overlay_image: /assets/images/joshua-sortino-LqKhnDzSF-8-unsplash.jpg
---

Materials modified from the [USDA ARS AI Workshop 2020](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/).

## Index

| Tutorial | Python | R |
|:--|:--|:--|
| 01 - Image Processing Essentials | [view](Tutorial1_Image_Processing_Essentials_Boucheron.md) |[view](R_Image_Processing.md) |
| 02 - Classical Machine Learning | [view](Tutorial2_Classical_Machine_Learning_Boucheron.md)| |
| 03 - Deep Learning for Images | [view](Tutorial3_Deep_Learning_for_Images_Boucheron.md) | |
| 04 - Visualizing and Modifying Deep Learning Networks | [view](Tutorial4_Visualizing_and_Modifying_DL_Networks_Boucheron.md) | |
| 05 - Advanced Deep Learning Networks | [view](Tutorial5_Advanced_DL_Networks.md)| |


<!--

<hr />
The original learn page is below:

## Learn

All tutorials available on this page were created by Dr. Laura Boucheron of the Klipsch School of Electrical and Computer Engineering at New Mexico State University.

This information is free; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This work is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

A copy of the GNU General Public License can be viewed/downloaded at [COPYING.txt](/COPYING.txt) or see <https://www.gnu.org/licenses/>.
<br><br>

# Workshop 1

## Day 1

Download the Jupyter Notebook and image data

**If working on a laptop** (right click the links to download)

[Tutorial1_Image_Processing_Essentials.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial1_Image_Processing_Essentials.ipynb)

[cameraman.png](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/cameraman.png)

[peppers.png](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/peppers.png)

save these items in your working directory

**If working on the ARS Ceres HPC**

navigate to your working directory, then:

```
curl -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial1_Image_Processing_Essentials.ipynb -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/cameraman.png -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/peppers.png
```

**To see a static notebook with all the outputs/answers:** 

[Tutorial1_Image_Processing_Essentials_complete.html](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial1_Image_Processing_Essentials_complete.html)

**To access the instructor's ipynb with the ad hoc cells added during instruction on 10/19/2020:**

[Tutorial1_Image_Processing_Essentials_Boucheron.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial1_Image_Processing_Essentials_Boucheron.ipynb) (right click to download)

**To access the Zoom recording:**

to be posted




## Day 2

Download the Jupyter Notebook and image data

**If working on a laptop** 

[Tutorial2_Classical_Machine_Learning.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial2_Classical_Machine_Learning.ipynb) (right click the link to download)

[CalTech101 dataset 101_ObjectCategories.tar.gz](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz) (126 MB; follow the link to download)

[CalTech101 dataset Annotations.tar](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar) (13 MB; follow the link to download)

Move the compressed image data folders to your working directory and unzip. Unzip using a terminal (e.g. Windows PowerShell) with ```tar -xvf filename```

**If working on the ARS Ceres HPC**

navigate to your working directory, then:

```
curl -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial2_Classical_Machine_Learning.ipynb
cp /project/shared_files/NMSU-AI-WORKSHOP/*.zip ./
unzip '*.zip'
```

Note: If you are following this tutorial after the workshop has ended and the NMSU-AI-WORKSHOP shared folder no longer exists, do the following:
- Download and untar the image data to your local machine with the laptop instructions above
- Zip both folders of data (on Windows: right click > Send to > Compressed (zipped) folder)
- Login to Ceres with JupyterHub and upload the zip files (the larger zip will take a few minutes). The upload button is on the JupyterLab navigation pane between the New Folder icon and the Refresh File List icon
- Move the files to your working directory on Ceres and ```unzip `*.zip'```

**Presentation slides:** 

[Day2_Rules_ML_DL.pdf](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/slides/Day2_Rules_ML_DL.pdf)

**To see a static notebook with all the outputs/answers:** 

[Tutorial2_Classical_Machine_Learning_complete.html](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial2_Classical_Machine_Learning_complete.html)

**To access the instructor's ipynb with the ad hoc cells added during instruction on 10/21/2020:**

[Tutorial2_Classical_Machine_Learning_Boucheron.html](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial2_Classical_Machine_Learning_Boucheron.ipynb)

**To access the Zoom recording:**

to be posted



## Day 3

Download the Jupyter Notebook.

Also make sure you have built the workshop Conda environment and then activate it (use the kernel) inside your Jupyter Notebook ([use the instructions on the setup page](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/setup/)). You will not be able to run the Notebook for Day 3 if you haven't built and activated the workshop Conda environment successfully.

**If working on a laptop** 

[Tutorial3_Deep_Learning_for_Images.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial3_Deep_Learning_for_Images.ipynb) 

save to your working directory

**If working on the ARS Ceres HPC**

navigate to your working directory, then:

```
curl -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial3_Deep_Learning_for_Images.ipynb
```

**Presentation slides:** 

[Day3_CNNs.pdf](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/slides/Day3_CNNs.pdf)

[Day3_CNN_Epic_Fails.pdf](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/slides/Day3_CNN_Epic_Fails.pdf)

**To see a static notebook with all the outputs/answers:** 

[Tutorial3_Deep_Learning_for_Images_complete.html](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial3_Deep_Learning_for_Images_complete.html)

**To access the instructor's ipynb with the ad hoc cells added during instruction on 10/23/2020:**

[Tutorial3_Deep_Learning_for_Images_Boucheron.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial3_Deep_Learning_for_Images_Boucheron.ipynb)

**To access the Zoom recording:**

to be posted


# Workshop 2

## Day 1

Download the Jupyter Notebook and image data

**If working on a laptop** (right click the links to download)

[Tutorial4_Visualizing_and_Modifying_DL_Networks.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial4_Visualizing_and_Modifying_DL_Networks.ipynb)

[my_digits1_compressed.jpg](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/my_digits1_compressed.jpg)

[latest_256_0193.jpg](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/latest_256_0193.jpg)

save these items in your working directory

**If working on the ARS Ceres HPC**

navigate to your working directory, then:

```
curl -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial4_Visualizing_and_Modifying_DL_Networks.ipynb -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/my_digits1_compressed.jpg -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/data_images/latest_256_0193.jpg
```

**To see a static notebook with all the outputs/answers:** 

[Tutorial4_Visualizing_and_Modifying_DL_Networks_complete.html](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial4_Visualizing_and_Modifying_DL_Networks_complete.html)

**To access the instructor's ipynb with the ad hoc cells added during instruction on 10/26/2020:**

[Tutorial4_Visualizing_and_Modifying_DL_Networks_Boucheron.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial4_Visualizing_and_Modifying_DL_Networks_Boucheron.ipynb)

**To access the Zoom recording:**

to be posted


## Day 2

Download the Jupyter Notebook and model weights

**If working on a laptop** 

[Tutorial5_Advanced_DL_Networks.ipynb](https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial5_Advanced_DL_Networks.ipynb)

[https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) (236 MB)

[https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/zebra.jpg](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/zebra.jpg)

save these items in your working directory

**If working on the ARS Ceres HPC**

navigate to your working directory, then:

```
curl -O https://kerriegeil.github.io/NMSU-USDA-ARS-AI-Workshops/tutorials/Tutorial5_Advanced_DL_Networks.ipynb -O https://pjreddie.com/media/files/yolov3.weights -O https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/zebra.jpg
```

**To see a static notebook with all the outputs/answers:** 

to be posted

**To access the instructor's ipynb with the ad hoc cells added during instruction on 10/28/2020:**

to be posted

**To access the Zoom recording:**

to be posted
-->

