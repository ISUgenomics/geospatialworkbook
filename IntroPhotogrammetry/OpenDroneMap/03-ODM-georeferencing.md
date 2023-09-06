---
title: "Geolocation data for the ODM workflow"
layout: single
author: Aleksandra Badaczewska
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /IntroPhotogrammetry/assets/images/geospatial_workbook_banner.png
---

{% include toc %}

# Georeferencing

Georeferencing process results in locating a piece of the landscape visible in the photo in the corresponding geographic destination on the real-world map.

![Georeferencing](../assets/images/georeferencing.png)

Most software designed for photogrammetric workflows has a built-in geolocation step. The same is true for <b>OpenDroneMap</b>.
In any case, to make geospatial localization of [aerial] photos, **GPS data** *(Global Positioning System)* is required. Most modern professional cameras record GPS coordinates automatically in the images, usually with an accuracy of 10-30 feet (3-9 meters) *[[source](https://www.blog.jimdoty.com/?p=14661)]*. While such a result is sufficient for most ordinary purposes, your research may be more precision demanding. For example, consider the case where the target area is even smaller than the geolocation threshold. The novel, more accurate geolocation systems, such as VPS *(Visual Positioning System)* or CPS *(Camera Positioning Standard)*, are now being developed *[[learn more](https://www.mosaic51.com/community/alternative-to-gps-how-to-get-better-accuracy/)]* and will probably supplant GPS technology in the future. By this time, though, the best patch for improving georeferencing process is still to use a **high-accuracy GPS point reference**, which will minimize the error.


## Georeferencing options in OpenDroneMap

The user has some level of control over the ODM settings for the photo georeferencing stage. By default, ODM tries to use the GPS information embedded in the images automatically while recording the mission. If this is the case, you don't need to add any additional option for geolocation to be performed.

**1. Force the use of geolocation from images' EXIF metadata**

```
--force-gps --use-exif \
```

Use `--force-gps` and `--use-exif` flags when you have a GCP data file in the project file structure but <u>want to force</u> the use of the GPS data stored in the image metadata.

That is especially useful when the original imagery is not geotagged, but you have GPS data in separate text files. In such a case, you can add this information to the image EXIF metadata using [ExifTool](https://exiftool.org) software. Follow the instructions in the tutorial ["Keep EXIF GEO metadata"](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/00-IntroODM#keep-exif-geo-metadata) *(section: add EXIF tags from a text file using exiftool)* to accomplish this step.

**2. Force the use of GPS data (e.g., RTK) from a text file**

```
--geo geo.txt \
```

Regardless of whether your imagery is geotagged, once you have alternative GPS information <u>stored in a text file</u>, you can force direct use of it with the option `--geo geo.txt`. This is especially useful when you have more **accurate geolocation data such as RTK** *(Real-Time Kinematic positioning)* that corrects some of the common errors in current satellite navigation (GNSS) systems.

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Keep in mind that the format of the text file containing the GPS data is strictly defined! <br><br>
If the <b>geo.txt</b> file is somewhere outside of your project's workdir or you have several GPS files, then provide the <b>absolute path</b> to the one you want.
</span>
</div>

<div style="background: #cff4fc; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">geo.txt</span> <i>(file content below)</i>
<span style="color:navy;"><i>[see details in the <a href="https://docs.opendronemap.org/geo/#image-geolocation-files" style="color: blue;">ODM Documentation: GPS data</a>]</i></span>
<br><br>
+proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs &emsp; # header <br>
DJI_0028.JPG &emsp; -91.9942096 &emsp; 46.8425252 &emsp; 198.609 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; # GPS data <br>
DJI_0032.JPG &emsp; -91.9938293 &emsp; 46.8424584 &emsp; 198.609 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; # GPS data <br>
</div>

* The first line should contain the name of the projection used for the geo-coordinates, in one of the following formats:

```
* PROJ string:   +proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs
*   EPSG code:   EPSG:4326
*   WGS84 UTM:   WGS84 UTM 16N
```

* Subsequent lines are the GPS information for a given image <u><i>(the first 3 columns are obligatory)</i></u>:

```
     col-1 col-2 col-3  col-4              col-5          col-6             col-7                    col-8                    col-9      col-10
     _____ _____ _____ ______ _________________  ______________  ________________  _______________________  _______________________  ___________
image_name geo_x geo_y [geo_z] [omega (degrees)] [phi (degrees)] [kappa (degrees)] [horz accuracy (meters)] [vert accuracy (meters)] [extras...]
```


**3. Force fixed value of GPS Dilution of Precision** *(use along with variant 1 or 2)*

```
--gps-accuracy 10.0 \
```

If you know the **estimated error of GPS** location determined by the camera in use, consider setting it as a value of the `--gps-accuracy value` option. The value is a positive float in meters and will be used as a GPS Dilution of Precision for all images. *The default is 10 meters.*

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
If you use high-precision GPS (RTK), this value will be set automatically. You can manually set it in case the reconstruction fails. Lowering the value can help control bowling effects over large areas.
</span>
</div>

**3. Force the use of GCP-based georeferencing**

**Ground Control Points** (GCPs) are clearly visible objects which can be easily identified in several images. Using the precise GPS position of these ground points is a good reference that improves significantly the accuracy of the project's geolocation. Ground control points can be any **steady structure** existing in the mission area, otherwise can be set using **targets placed on the ground**. *Learn more about recommended practices for GCPs in ODM workflow from the OpenDronMap Documentation: [Ground Control Points](https://docs.opendronemap.org/gcp/#ground-control-points).*

If you have a file with GCPs detected on the image collection, force georeferencing using it by option `--gcp gcp_list.txt`.

```
--gcp gcp_list.txt \
```



<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Keep in mind that the format of the text file containing the GCP data is strictly defined! <br><br>
If the <b>gcp_list.txt</b> file is somewhere outside of your project's workdir or you have several GCP files, then provide the <b>absolute path</b> to the one you want.
</span>
</div>

<br><span style="color: #ff3870;font-weight: 600; font-size:24px;">
Detect each ground control point in at least 5-10 photos!
</span><br>


<div style="background: #cff4fc; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">gcp_list.txt</span> <i>(file content below)</i>
<span style="color:navy;"><i>[see details in the <a href="https://docs.opendronemap.org/gcp/#ground-control-points" style="color: blue;">ODM Documentation: GCPs</a>]</i></span>
<br><br>
EPSG:4326 <br>
-116.74998 &emsp; 43.06477 &emsp; 2090.14 &emsp; 1559.41645 &emsp; 1372.84669 &emsp; DJI_0177.JPG &emsp; 100 <br>
-116.74998 &emsp; 43.06477 &emsp; 2090.14 &emsp; 1491.01638 &emsp; 2471.85207 &emsp; DJI_0355.JPG &emsp; 100 <br>
-116.74998 &emsp; 43.06477 &emsp; 2090.14 &emsp; 1524.14196 &emsp; 2214.43593 &emsp; DJI_0178.JPG &emsp; 100 <br>
-116.74998 &emsp; 43.06477 &emsp; 2090.14 &emsp; 1142.59915 &emsp; 1739.80028 &emsp; DJI_0152.JPG &emsp; 100 <br>
-116.74998 &emsp; 43.06477 &emsp; 2090.14 &emsp; 1207.88116 &emsp; 1863.28946 &emsp; DJI_0329.JPG &emsp; 100 <br>
-116.75048 &emsp; 43.06475 &emsp; 2088.22 &emsp; 1737.22646 &emsp; 1763.28507 &emsp; DJI_0172.JPG &emsp; 101 <br>
-116.75048 &emsp; 43.06475 &emsp; 2088.22 &emsp; 1660.24277 &emsp; 2912.50267 &emsp; DJI_0350.JPG &emsp; 101 <br>
-116.75048 &emsp; 43.06475 &emsp; 2088.22 &emsp; 1736.70576 &emsp; 1411.55810 &emsp; DJI_0171.JPG &emsp; 101 <br>
-116.75048 &emsp; 43.06475 &emsp; 2088.22 &emsp; 989.578526 &emsp; 1391.94185 &emsp; DJI_0157.JPG &emsp; 101 <br>
-116.75048 &emsp; 43.06475 &emsp; 2088.22 &emsp; 877.826253 &emsp; 2459.65369 &emsp; DJI_0335.JPG &emsp; 101 <br>
</div>

* The first line should contain the name of the projection used for the geo-coordinates, in one of the following formats:

```
* PROJ string:   +proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs
*   EPSG code:   EPSG:4326
*   WGS84 UTM:   WGS84 UTM 16N
```

* Subsequent lines are the GPS information for a given image <u><i>(the first 6 columns are obligatory)</i></u>:

```
col-1 col-2 col-3 col4 col5    col-6     col-7     col-8    col-9
_____ _____ _____ ____ ____ __________ __________ ________ ________
geo_x geo_y geo_z im_x im_y image_name [gcp_name] [extra1] [extra2]
```


# Software for manual detection of GCPs

<span style="color: #ff3870;font-weight: 600;">Section in development...</span>

Use any software for tagging GCPs, e.g., the [GCP Editor Pro](https://uav4geo.com/software/gcpeditorpro) is a good match for the ODM. Download source code from the [GitHub](https://github.com/uav4geo/GCPEditorPro).


# Automatic detection of ArUco targets

ArUco markers are a type of fiducial marker that are often used in computer vision applications. A fiducial marker is an object placed in the field of view of the camera which appears in the image with a known location, size, and appearance. <b>Aruco markers are square black-and-white patterns which can be easily detected</b>, identified, and used to calculate the camera's pose with respect to the marker.



**Features of Aruco**

| feature         | description |
|-----------------|-------------|
|Simple Structure | *Being just black and white squares, they are relatively easy and efficient to detect in an image.*|
|Identification   | *Each Aruco marker has a unique identifier assigned based on its pattern, which allows the system to distinguish between different markers.*|
|Flexibility      | *Aruco markers come in different dictionaries, or sets of markers. Each dictionary varies in the number of bits in the marker and the number of different markers. This allows for a balance between the total number of unique markers and the robustness against detection errors.*|
|Pose Estimation  | *By knowing the real size of the marker and its position in the image, one can determine the position and orientation (pose) of the camera with respect to the marker.*|
|Calibration      | *ArUco markers can be employed to calibrate cameras by capturing images of the markers from different orientations and positions.*|


<div style="background: #dff5b3; padding: 15px;">
<span style="font-weight:800;">NOTE:</span>
<br><span style="font-style:italic;">Aruco markers originated from academic research, and the term "aruco" is derived from the name of the library that introduced and popularized these markers. The <b>ArUco library</b> was initially a standalone project but later got integrated into the <b>OpenCV (Open Source Computer Vision Library)</b>. Therefore, when you're looking to utilize ArUco functionalities, you'd typically access it through OpenCV's ArUco module.</span>
</div><br>


## Working with ArUco in Land Surveying Tasks

The most official and widely recognized source for information and usage of Aruco markers is within the **OpenCV** (*Open Source Computer Vision Library*) library, which has integrated Aruco module functionalities. The OpenCV documentation provides detailed information on how to use Aruco markers, and you can access it via the [official OpenCV website](https://docs.opencv.org/master/d9/d6a/group__aruco.html).

**OpenCV is one of the most popular and comprehensive libraries for computer vision and machine learning tasks.** It was initially developed by Intel and released in 2000. Since then, it has become a standard tool for computer vision researchers and developers due to its rich set of functionalities and performance optimizations.
* **OpenCV is organized into modules**, each focused on a specific aspect of computer vision or image processing. This includes modules for:
  * image filtering with `imgproc` module
  * **pose estimation with** `aruco` **module**
  * feature detection with `features2d` module
  * object detection with `objdetect` module
  * machine learning with `ml` module
  * camera calibration with `calib3d` module
  * motion analysis with `video` module
  * and more.
* While OpenCV was originally written in `C++`, it now has bindings for `Python`, `Java`, and several other languages. This makes it accessible to a wide range of developers. **The Python bindings have become extremely popular** enabling developers to create custom computer vision applications, such as [Find-GCP](https://github.com/zsiki/Find-GCP) utility for finding ArUco markers in digital photos.

<span style="color: #ff3870;font-weight: 500;">Learn more about ArUco markers in OpenCV library: <a href="https://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html" target="_blank">https://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html</a></span>

### Steps in the land surveying

1. **Markers Generation:** Before the survey, ArUco markers are generated using specialized software or libraries, such as the `aruco_make.py` python utility *(using ArUco module in OpenCV library)*.

2. **Printing and Placement:** Once generated, these markers are printed on sturdy material to withstand outdoor conditions. They're then placed strategically at known positions within the area to be surveyed.

3. **Capturing Imagery:** Using drones, satellites, or handheld cameras, images of the area are taken. These images capture the terrain as well as the ArUco markers.

4. **Detection and Analysis:** During the post-processing phase, software detects the ArUco markers in the captured images. Given the known size and ID of each marker, as well as its location in the image, software can estimate the camera's pose and the 3D position of the marker.

5. **Georeferencing:** Knowing the real-world coordinates of each ArUco marker, the captured images can be georeferenced (assigned to a specific location in a spatial reference system). This ensures that the imagery aligns accurately with geographic coordinates.

## Find-GCP python utility: installation

**[Find-GCP](https://github.com/zsiki/Find-GCP)** is a Python tool leveraging the OpenCV library, designed to detect ArUco Ground Control Points in imagery and generate the corresponding GCP file required for photogrammetric programs such as Open Drone Map. The GitHub repo contains a few small utilities useful in Land Surveying Tasks:

* [aruco_make.py](https://github.com/zsiki/Find-GCP#aruco_makepy) - generates aruco marker images using different standard dictionaries
  * [dict_gen_3x3.py](https://github.com/zsiki/Find-GCP#dict_gen_3x3py) - generates 32 custom markers using specifically 3x3 ArUco dictionary


* [gcp_find.py](https://github.com/zsiki/Find-GCP#gcp_findpy) - identifies Ground Control Points (GCP) in imagery
* [gcp_check.py](https://github.com/zsiki/Find-GCP#gcp_checkpy) - helps the visual check of the found GCPs by `gcp_find.py`

**INSTALLATION:**

**A. On SCINet HPC (Atlas via ssh in terminal):**
  1.  check available **conda** modules and load selected one:
  ```
  module avail conda
  module load miniconda/4.12.0
  ```
  2. create python environment for geospatial analysis:
  ```
  conda create -n geospatial python=3.9
  ```
  3. activate this environment:
  ```
  source activate geospatial
  ```
  4. install required libraries:
  ```
  pip install opencv-python opencv-contrib-python Pillow numpy matplotlib
  ```
  5. clone the Find-GCP repo from GitHub:
  ```
  git clone https://github.com/zsiki/Find-GCP.git
  ```
  When you clone a repository from GitHub, it creates a new directory on your current path with the name of the repository. Inside this directory, you'll find the contents of the repository. Once you navigate into this directory, you should see 6 files with the `.py` extension. **These .py files are the Find-GCP python utilities for working with ArUco markers in Land Surveying Tasks.**
  ```
  cd Find-GCP
  ls
  ```

  ![find_gcp_repo](../assets/images/find_gcp_repo.png)

**B. On your local machine (alternatively):** <br>
* If you already have the Conda environment manager installed, skip step 1 and proceed with the instructions outlined above.
* For those unfamiliar with Conda, it's a valuable tool for computational tasks, and you can learn how to use it through the practical tutorials in the DataScience workbook: [Conda on Linux](https://datascience.101workbook.org/03-SetUpComputingMachine/03C-tutorial-installations-on-linux#conda), [Conda on macOS](https://datascience.101workbook.org/03-SetUpComputingMachine/03A-tutorial-installations-on-mac#-install-conda), [Python Setup on your computing machine](https://datascience.101workbook.org/04-DevelopmentEnvironment/02A-python-setup-locally#conda).
* If you choose not to use Conda, you can jump directly to step 4 in the guide, though <u>this is not recommended</u> because the necessary libraries will install system-wide, rather than in an isolated environment.

### Automatic generation of ArUco codes *(using Find-GCP)*

*ArUco markers provide known reference points in the imagery, enhancing the accuracy of photogrammetric analysis. This ensures that data derived from the imagery correctly corresponds to actual locations on the ground.*

#### Available ArUco dictionaries
ArUco markers come in various dictionaries, where **each dictionary defines a set of distinct markers**. The choice of dictionary affects the size and resilience of the markers, as well as how many unique markers the dictionary contains.<br>
In the naming convention like `DICT_4X4_100` or `DICT_6X6_250`: <br>
**-** The first part (4X4 or 6X6) represents the size of the marker grid. For example, a 4X4 marker has a 4x4 grid of black or white squares, while a 6X6 marker has a 6x6 grid. <br>
**-** The second part (100 or 250) indicates the number of unique markers available in that dictionary. So, DICT_4X4_100 has 100 unique 4x4 markers, while DICT_6X6_250 contains 250 unique 6x6 markers.

***When choosing a dictionary,*** one must consider the application.
* Smaller markers (like 4x4) can be detected at shorter distances and might be harder to distinguish at low resolutions or with noise.
*Larger markers (like 6x6 or 7x7) can be detected from a greater distance, are generally more resilient to noise, but they also take up more space in the image. The number of unique markers needed will also influence the choice of dictionary.


#### Generating markers using ready-made tools:

**A. ArUco marker images in PNG** <br>
To generate ArUco markers for your land surveying task, first get Find-GCP python utility installed ([see section above](#find-GCP-python-utility-installation)). In a directory of cloned **Find-GCP** repo, you will find the `aruco_make.py` and `dict_gen_3x3.py` python scripts. Use the first one to generate markers using the standard dictionaries (listed below) and use the second one to generate smaller markers of size 3 by 3 squares.
* While in the **Find-GCP** directory, use `pwd` command to print the path on the screen. *You will need this path to run python scripts from another location in the file system.*


* Navigate to the selected location in the file system and create the **markers** directory:
```
mkdir markers
cd markers
```


* Then use the `aruco_make.py` script like this: <br>
`python aruco_make.py -d <DICT> -s <START> -e <END> -v`, for example:
```
python aruco_make.py -d 1 -s 0 -e 9 -v
```

  ![aruco_make](../assets/images/aruco_make.png)

  <i>This command will create 10 markers, numbered from 0 to 9 (e.g., marker0.png), using the dictionary DICT_4x4_100.</i><br>
  * the number provided with the `-d` option determines the dictionary *(see the table below)*, default = 1
  |code  | dictionary|code  | dictionary |code  | dictionary |code  | dictionary  |
  |------|-----------|------|------------|------|------------|------|-------------|
  |**0** |DICT_4X4_50|**1** |DICT_4X4_100|**2** |DICT_4X4_250|**3** |DICT_4X4_1000|
  |**4** |DICT_5X5_50|**5** |DICT_5X5_100|**6** |DICT_5X5_250|**7** |DICT_5X5_1000|
  |**8** |DICT_6X6_50|**9** |DICT_6X6_100|**10**|DICT_6X6_250|**11**|DICT_6X6_1000|
  |**12**|DICT_7X7_50|**13**|DICT_7X7_100|**14**|DICT_7X7_250|**15**|DICT_7X7_1000|
  |**16**|DICT_ARUCO_ORIGINAL|
  |**17**|DICT_APRILTAG_16H5|**18**|DICT_APRILTAG_25H9|**19**|DICT_APRILTAG_36H10|**20**|DICT_APRILTAG_36H11|
  |**99**|DICT_3X3_32 custom|
  * the number provided with the `-s` option determines the index of the first marker, default = 0
  * the number provided with the `-e` option determines the index of the last marker, default = -1 *(only one marker is generated with index 0)*
  * the optional `-v` flag shows marker on monitor *(when applicable, e.g., when working on a local machine)*
  * the optional `-g` flag generates black/gray marker *(instead black/white to reduce the effect of white burnt in)*
    * the optional option `--value <VAL>` determines shade of background use with `-g`, default=95
  * the optional option `-p <PAD>` determines border width around marker in inches, default= 0.5
<br><br><br>
* The `dict_gen_3x3.py` has no built-in options and generates 32 custom 3x3 ArUco dictionary markers in dict_3x3 subdirectory:
```
python dict_gen_3x3.py
```

**B. ArUco marker images in SVG** <br>
There is another GitHub repo, [gcp_aruco_generator](https://github.com/qaptadrone/gcp_aruco_generator), providing a simple python tool for generating ArUco markers with a real sizing of the image saved in SVG format. It also has a few more options, including `--print-id` in the corner of the marker and adding a watermark on the four borders. Follow the [Setup and use](https://github.com/qaptadrone/gcp_aruco_generator#setup-and-use) guide to get started with this tool. The generated ArUco markers are compatible with the Find-GCP tool, so you can use it after the flight to find the markers in your pictures.

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">For good size recommendations, please see <a href="http://www.agt.bme.hu/on_line/gsd_calc/gsd_calc.html" target="_blank">http://www.agt.bme.hu/on_line/gsd_calc/gsd_calc.html</a>.</span>
</div><br>

**C. ArUco marker images <u>generated online</u> (SVG or PDF)** <br>
Finally, you can use the free online ArUco markers generator: [https://chev.me/arucogen/](https://chev.me/arucogen/) <br>
*See the corresponding GitHub repo:* [https://github.com/okalachev/arucogen](https://github.com/okalachev/arucogen)

### Automatic recognition of ArUco codes *(using Find-GCP)*






___
# Further Reading
<!-- * [Command-line ODM modules](02-ODM-modules) -->


___

[Homepage](../index.md){: .btn  .btn--primary}
[Section Index](../00-IntroPhotogrammetry-LandingPage){: .btn  .btn--primary}
[Previous](02-ODM-modules){: .btn  .btn--primary}
<!-- [Next](){: .btn  .btn--primary} -->
