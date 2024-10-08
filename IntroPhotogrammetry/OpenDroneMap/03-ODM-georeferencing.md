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

---

# Software for manual detection of (any) GCPs

<span style="color: #ff3870;font-weight: 600;">Section in development...</span>

Use any software for tagging GCPs, e.g., the [GCP Editor Pro](https://uav4geo.com/software/gcpeditorpro) is a good match for the ODM. Download source code from the [GitHub](https://github.com/uav4geo/GCPEditorPro).

---

# Automatic detection of ArUco targets

ArUco markers are a type of fiducial marker that are often used in computer vision applications. A fiducial marker is an object placed in the field of view of the camera which appears in the image with a known location, size, and appearance. <b>Aruco markers are square black-and-white patterns which can be easily detected</b>, identified, and used to calculate the camera's pose with respect to the marker.


| ArUco_feature         | description |
|-----------------|-------------|
|Simple Structure | *Being just black and white squares, they are relatively easy and efficient to detect in an image.*|
|Identification   | *Each Aruco marker has a unique identifier assigned based on its pattern, which allows the system to distinguish between different markers.*|
|Flexibility      | *Aruco markers come in different dictionaries, or sets of markers. Each dictionary varies in the number of bits in the marker and the number of different markers. This allows for a balance between the total number of unique markers and the robustness against detection errors.*|
|Pose Estimation  | *By knowing the real size of the marker and its position in the image, one can determine the position and orientation (pose) of the camera with respect to the marker.*|
|Calibration      | *ArUco markers can be employed to calibrate cameras by capturing images of the markers from different orientations and positions.*|


<div style="background: #dff5b3; padding: 15px;">
<span style="font-weight:800;">NOTE:</span>
<br><span style="font-style:italic;">ArUco markers originated from academic research, and the term "aruco" is derived from the name of the library that introduced and popularized these markers. The <b>ArUco library</b> was initially a standalone project but later got integrated into the <b>OpenCV (Open Source Computer Vision Library)</b>. Therefore, when you're looking to utilize ArUco functionalities, you'd typically access it through OpenCV's ArUco module.</span>
</div>


## Working with ArUco in Land Surveying Tasks

The most official and widely recognized source for information and usage of ArUco markers is within the **OpenCV** (*Open Source Computer Vision Library*) library, which has integrated ArUco module functionalities. The OpenCV documentation provides detailed information on how to use ArUco markers, and you can access it via the [official OpenCV website](https://docs.opencv.org/master/d9/d6a/group__aruco.html).

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

**1. Markers Generation:** Before the survey, ArUco markers are generated using specialized software or libraries, such as the `aruco_make.py` python utility *(using ArUco module in OpenCV library)*. <br>

<div style="background: #cff4fc; padding: 15px; margin-bottom: 30px;">
<span style="font-weight:800;">PRO TIP:</span><br>
You can generate GCP (Ground Control Point) markers ranging from 1 to up to 1000 using various ArUco dictionaries by leveraging OpenCV's built-in ArUco module. <span style="color: #ff3870;font-weight: 500;"> To learn more, see section <a href="#automatic-generation-of-aruco-codes ">Automatic generation of ArUco codes </a>.</span>
</div>

**2. Printing and Placement:** Once generated, these markers are printed on sturdy material to withstand outdoor conditions. They're then placed strategically at known positions within the area to be surveyed.

<div style="background: #cff4fc; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">PRO TIP:</span><br>
When recording the exact positions of the placed Ground Control Points (GCPs), <b>always note which GCP corresponds to which ArUco marker ID</b><i> (e.g., create GCP_reference.txt file)</i>; this association will be crucial when pinpointing them on the images during post-processing. <br>
Expected format of the <i>GCP_reference.txt</i> is 4-columns: <b>ArUco_ID x y z </b><br>
0 523287.368 4779588.335 1397.823 <br>
1 523305.976 4779572.588 1397.817 <br>
2 523347.074 4779571.424 1397.653 <br>
3 523364.648 4779587.932 1395.735 <br>
4 523394.376 4779529.525 1398.728 <br>
5 523363.938 4779530.027 1400.244 <br>
</div>

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">When using ArUco markers for Ground Control Points (GCPs), it's strongly recommended to <b>avoid using custom IDs</b>. If custom IDs are unavoidable, make sure to maintain an accurate reference that matches your custom IDs with the corresponding ArUco marker ID or type <i>(i.e., marker's pattern)</i>.</span>
</div>

<table style="background: mistyrose; padding: 15px; margin-bottom: 30px;">
<tr><th style="width: 500px;">WARNING:</th><th>digital marker: ID 9</th><th>printed marker</th></tr>
<tr><td><b>Always verify that the printed marker matches the digital version.</b> Printing errors can occasionally distort the marker, leading to recognition issues during post-processing of land surveying imagery. <br><br><i>In the given example, the printed marker is missing one black square. This omission can greatly hinder its recognition, making it challenging to correctly identify it as an ArUco marker with ID 9.</i></td>
<td><img src="../assets/images/marker9.png" style="width:200px;"></td>
<td><img src="../assets/images/marker9_printed.jpg" style="width:200px;"></td></tr>
</table>


**3. Capturing Imagery:** Using drones, satellites, or handheld cameras, images of the area are taken. These images capture the terrain as well as the ArUco markers.

**4. Detection and Analysis:** During the post-processing phase, software detects the ArUco markers in the captured images. Given the known size and ID of each marker, as well as its location in the image, software can estimate the camera's pose and the 3D position of the marker. <span style="color: #ff3870;font-weight: 500;"><br>To learn more, see section <a href="#automatic-recognition-of-aruco-codes" target="_blank">Automatic recognition of ArUco codes</a>.</span>

**5. Georeferencing:** Knowing the real-world coordinates of each ArUco marker, the captured images can be georeferenced (assigned to a specific location in a spatial reference system). This ensures that the imagery aligns accurately with geographic coordinates.

---

# Create env for geospatial analysis

Creating a Conda environment for geospatial analyses streamlines your workflow by isolating installations and dependencies specific to selected geospatial tools. This offers the advantage of consistency, compatibility and convenience, beneficial for both High-Performance Computing (HPC) and local machines.

Isolating installations and dependencies ensures that you can **effortlessly integrate ready-made packages and Python utilities**, such as those for marker detection in imagery for further analysis with ODM. With a **one-time setup of a unified "geospatial" environment**, you can seamlessly utilize a suite of tools from various GitHub repositories, significantly streamlining your daily tasks.

*Give it a shot! It's an investment that pays off!*

<span style="font-size: 24px;"><b>A. On SCINet HPC: Atlas</b></span> <br>
*(via ssh in CLI terminal or via OOD in JupyterLab Terminal app)*

**1.**  check available **conda** modules and load selected one:
```
module avail conda
module load miniconda/4.12.0
```
![load_miniconda.png](../assets/images/load_miniconda.png)

**2.** create python environment for geospatial analysis:
```
conda create -n geospatial python=3.9
```
![create_geospatial_env.png](../assets/images/create_geospatial_env.png)

**3.** activate this environment:
```
source activate geospatial
```
![activate_geospatial_env.png](../assets/images/activate_geospatial_env.png)

**4.** install required libraries:
```
pip install numpy==1.22.2 opencv-python==4.8.0.76 opencv-contrib-python==4.8.0.76 Pillow==10.0.0 pyproj==3.6.0 Shapely==1.8.1.post1 svgwrite==1.4.1 matplotlib pandas
```
<i>This command installs a foundational set of dependencies crucial for the Python utilities detailed below (sourced from GitHub repos). With these dependencies in place and the environment activated, these tools are set to operate immediately post-cloning, eliminating the need of further setup.</i><br>

<div style="background: #cff4fc; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">
In the future, if you seek to augment this environment with more packages, you can effortlessly do so at any point using either <b>conda install</b> or <b>pip install</b> commands in the activated environment.
</span>
</div>

**5.** deactivate this environment *(if you no longer intend to use it for this session)*:
```
conda deactivate
```

<div style="background: mistyrose; padding: 15px; margin-bottom: 25px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Before starting to use Conda on HPC cluster <i>(e.g. Atlas or Ceres)</i>, it’s advisable to change the default location (your home directory) where Conda will install your customized libraries. Installing a lot of Python libraries may contribute to the default 5G soft limit quota on your home directory being surpassed. To overcome this issue you can <b>move .conda directory from your home directory to your project directory and create a symbolic link to the new location</b>.
</span>
</div>

<span style="font-size:24px;"><b>6.* Change storage location of your Conda envs:</b></span> <br>
<span style="color: #ff3870;font-weight: 500;">( <b>Do it only once!</b> <i>All your Conda envs are stored in .conda dir by default.</i> )</span><br>
*Remember to replace the placeholders in <> with the appropriate paths from your file system.*
```
cd ~
mkdir /project/<your_project_dir>/<account_name>
mv .conda /project/<your_project_dir>/<account_name>/
chmod -R g+s /project/<your_project_dir>/<account_name>/.conda
ln -s /project/<your_project_dir>/<account_name>/.conda .conda
```
*The* `mv` *command may take longer time depending on how much data you have in the <b>.conda</b> directory.* <br>
![move_conda_dir.png](../assets/images/move_conda_dir.png)

<div style="background: #cff4fc; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">
If you're unsure whether you've moved your <b>.conda directory</b> from <i>home</i> to the <i>project</i>, run <b>ls -lha</b> in your <i>home</i> directory to see the actual locations of all files, including the (eventually) <b>soft-linked .conda</b>. <br>
<img style="margin-top: 10px;" src="../assets/images/check_conda_location.png">
</span>
</div>

<span style="font-size:24px;"><b>7.* Create the storage directory for custom software and GitHub repos:</b></span> <br>
<span style="color: #ff3870;font-weight: 500;">( <b>Do it only once!</b> <i>You can keep all your self-installed useful tools here.</i> )</span> <br>
You can establish a SOFTWARE or TOOLS directory within your `/project/<account>` location on the cluster, ensuring a well-organized repository for your custom tools, making them easily locatable in the future.
```
cd /project/<your_project_dir>/
mkdir SOFTWARE
```
![create_software_dir.png](../assets/images/create_software_dir.png) <br>
*We will use this location later in this tutorial to* `git clone` *a few GitHub repositories with python utilities useful in land surveying tasks. You can also add your customized software here.*

<br><span style="font-size: 24px;"><b>B. On your local machine</b> <i>(alternatively)</i>:</span> <br>
* If you already have the Conda environment manager installed, skip step 1 and proceed with the instructions outlined above.
  * **NOTE:** On a local machine you will use `conda activate geospatial` instead of `source` command.


* For those unfamiliar with Conda, it's a valuable tool for computational tasks, and you can learn how to use it through the practical tutorials in the DataScience workbook: [Conda on Linux](https://datascience.101workbook.org/03-SetUpComputingMachine/03C-tutorial-installations-on-linux#conda), [Conda on macOS](https://datascience.101workbook.org/03-SetUpComputingMachine/03A-tutorial-installations-on-mac#-install-conda), [Python Setup on your computing machine](https://datascience.101workbook.org/04-DevelopmentEnvironment/02A-python-setup-locally#conda).


* If you choose not to use Conda, you can jump directly to step 4 in the guide, though <u>this is not recommended</u> because the necessary libraries will install system-wide, rather than in an isolated environment.

---

<br><span style="font-size: 24px;">A few tips before working with Conda and GitHub repos</span>

Once you've set up the `geospatial` environment, theoretically all the necessary dependencies for the repos listed below should already be installed. However, dependencies may change over time.
* If there are updates or changes to the repository, ensure that you activate the geospatial environment and install all the new requirements. The installation should be done once initially, or after each `git pull` (updates) from the repo.
* For subsequent usage of the repo's scripts, simply **activate the geospatial environment and execute the scripts**; there's no need to reinstall requirements every time you use them.

## **geo_utils** python utility: installation

**[geo_utils](https://github.com/ISUgenomics/geo_utils)** (GitHub repo) is an evolving collection of Python utilities tailored for geospatial analysis, developed at ISU as a part of the virtual research support for the USDA scientist. These utilities are designed to complement photogrammetry analysis using ODM software, enhancing the robustness of processing pipelines especially when calculations are executed on an HPC cluster. The GitHub repo contains a few small utilities useful in land surveying tasks:

* [gcp_to_aruco_mapper.py](https://github.com/ISUgenomics/geo_utils/tree/main#gcp-to-aruco-mapper) - maps custom GCP IDs to corresponding ArUco marker IDs in imagery based on the distance between GCP coordinates and image GPS

* [gcp_images_picker.py](https://github.com/ISUgenomics/geo_utils/tree/main#gcp-images-picker) - automatically selects the representative images for each GCP, minimizing manual inspection

**INSTALLATION:** <span style="color: #ff3870;font-weight: 500;">( <b>Do it only once!</b> <i>The cloned repo will persist in your file system.</i>)</span>

**On SCINet HPC: Atlas** (via ssh in CLI terminal or via OOD in JupyterLab Terminal app)

  1. navigate to the **SOFTWARE** directory in your `/project/<your_project_dir>/` path:
  ```
  cd /project/<your_project_dir>/SOFTWARE
  ```

  2. clone the `geo_utils` repo from GitHub:
  ```
  git clone https://github.com/ISUgenomics/geo_utils
  ```
  When you clone a repository from GitHub, it creates a new directory on your current path with the name of the repository. Inside this directory, you'll find the contents of the repository. <br>
  ![git_clone_geo_utils.png](../assets/images/git_clone_geo_utils.png)

  3.  check available **conda** modules and load selected one *(if not loaded yet in this session)*:
  ```
  module avail conda
  module load miniconda/4.12.0
  ```

  4. activate your `geospatial` environment *(if not activated yet in this session)*:
  ```
  source activate geospatial
  ```

  5. \*install required libraries *(optionally after* `git pull`*)*:
  ```
  cd geo_utils
  pip install -r requirements.txt
  ```

  6. start using the scripts from the repo! *(they are located in the TOOLS subdir)*
  ```
  # GCP images picker
  python gcp_images_picker.py [-h] -i DATA_FILE_PATH -w IMAGE_WIDTH -l IMAGE_HEIGHT [-n IMAGES_NUMBER] [-o OUTPUT]
  # GCP to ArUco mapper
  python gcp_to_aruco_mapper.py [-h] -g GCP_FILE -i IMAGERY_PATH -z ZONE [-o OUTPUT] [-d MAX_DIST]
  ```
  ![geo_utils_usage.png](../assets/images/geo_utils_usage.png)

  <div style="background: #cff4fc; padding: 15px; margin-left: 37px;">
  <span style="font-weight:800;">PRO TIP:</span>
  <br><span style="font-style:italic;">
  To gain practical experience with the use of these scripts, please follow the instructions provided in the subsequent sections of this tutorial.
  </span>
  </div>


## **Find-GCP** python utility: installation

**[Find-GCP](https://github.com/zsiki/Find-GCP)** (GitHub repo) is a Python tool leveraging the OpenCV library, designed to detect ArUco Ground Control Points in imagery and generate the corresponding GCP file required for photogrammetric programs such as Open Drone Map. The GitHub repo contains a few small utilities useful in land surveying tasks:

* [aruco_make.py](https://github.com/zsiki/Find-GCP#aruco_makepy) - generates aruco marker images using different standard dictionaries
* [gcp_find.py](https://github.com/zsiki/Find-GCP#gcp_findpy) - identifies Ground Control Points (GCP) in imagery
* [gcp_check.py](https://github.com/zsiki/Find-GCP#gcp_checkpy) - helps the visual check of the found GCPs by `gcp_find.py`

**INSTALLATION:** <span style="color: #ff3870;font-weight: 500;">( <b>Do it only once!</b> <i>The cloned repo will persist in your file system.</i>)</span>

**On SCINet HPC: Atlas** (via ssh in CLI terminal or via OOD in JupyterLab Terminal app)

  1. navigate to the **SOFTWARE** directory in your `/project/<your_project_dir>/` path:
  ```
  cd /project/<your_project_dir>/SOFTWARE
  ```

  2. clone the `Find-GCP` repo from GitHub:
  ```
  git clone https://github.com/zsiki/Find-GCP.git
  ```
  ![git_clone_find_gcp.png](../assets/images/git_clone_find_gcp.png)

  3.  check available **conda** modules and load selected one *(if not loaded yet in this session)*:
  ```
  module avail conda
  module load miniconda/4.12.0
  ```

  4. activate your `geospatial` environment *(if not activated yet in this session)*:
  ```
  source activate geospatial
  ```

  5. \*install required libraries *(optionally after* `git pull`*)*:
  ```
  pip install opencv-python opencv-contrib-python Pillow pil.imagetk numpy matplotlib
  ```
  <div style="background: mistyrose; padding: 15px; margin-bottom: 20px; margin-left: 37px;">
  <span style="font-weight:800;">WARNING:</span>
  <br><span style="font-style:italic;">
  Installing packages without specifying a version usually installs the latest version, which may be incompatible with older required ones, potentially causing scripts to malfunction.
  </span>
  </div>

<span style="margin-left:10px;">6. &nbsp; start using the scripts from the repo! <i>(they are placed directly in the directory)</i> </span><br>
Once you navigate into the newly created Find-GCP directory, you should see 6 files with the `.py` extension. These `.py` files are the **Find-GCP python utilities for working with ArUco markers in Land Surveying Tasks**.
```
cd Find-GCP
ls
python gcp_find.py -h
```
![find_gcp_repo](../assets/images/find_gcp_repo.png)<br><br>
![find_gcp_usage.png](../assets/images/find_gcp_usage.png)


## **Automatic generation of ArUco codes**

*ArUco markers provide known reference points in the imagery, enhancing the accuracy of photogrammetric analysis. This ensures that data derived from the imagery correctly corresponds to actual locations on the ground.*

**Available ArUco dictionaries** <br>
<span style="color: #ff3870;font-weight: 400;">ArUco markers come in various dictionaries that defines a set of distinct markers.</span> The choice of dictionary affects the size and resilience of the markers, as well as how many unique markers the dictionary contains.
In the naming convention like `DICT_4X4_100` or `DICT_6X6_250`: <br>
* The first part (4X4 or 6X6) represents the size of the marker grid. *For example, a 4X4 marker has a 4x4 grid of black or white squares, while a 6X6 marker has a 6x6 grid.* <br>
* The second part (100 or 250) indicates the number of unique markers available in that dictionary. *So, DICT_4X4_100 has 100 unique 4x4 markers, while DICT_6X6_250 contains 250 unique 6x6 markers.*

<div style="background: #cff4fc; padding: 15px; margin-bottom: 10px;">
<span style="font-weight:800;">PRO TIP:</span><br>
<b><i>When choosing a dictionary,</i></b> one must consider the application.
<li> Smaller markers (like 4x4) can be detected at shorter distances and might be harder to distinguish at low resolutions or with noise. Nevertheless, 4x4 markers are a popular and often sufficient choice for the majority of land surveying applications. <i>The contents of the 4X4_50 dictionary are displayed below.</i></li>
<li> Larger markers (like 6x6 or 7x7) can be detected from a greater distance, are generally more resilient to noise, but they also take up more space in the image. The number of unique markers needed will also influence the choice of dictionary.</li>
<li>In ArUco marker dictionaries, <b>each larger grid size includes all markers from its smaller counterparts in the same order</b>; <i>for example, the first 50 markers in all 4x4 dictionaries will be identical, and any additional markers in larger dictionaries are unique and extend the set.</i></li>
</div>

![aruco_dict](../assets/images/aruco_dict.png)

<div style="background: mistyrose; padding: 15px; margin-bottom: 30px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;"><b>ArUco markers within each dictionary are numbered starting from zero.</b> For accurate reference and data processing, always save both the selected marker's ID and the type of the source dictionary used.</span>
</div>

<span style="font-size: 24px;"><b>Generating markers using ready-made tools</b></span> <br>

**A. ArUco marker images in PNG** <br>
To produce ArUco markers for your land surveying project, start by installing the Find-GCP Python utility ([refer to the section above](#find-gcp-python-utility-installation)). Within the cloned Find-GCP repository directory, you'll locate the `aruco_make.py`. This tool assists you in generating markers from standard dictionaries, as well as more compact 3x3 square markers.

* While in the **Find-GCP** directory, use `pwd` command to print the path on the screen. *You will need this path to run python scripts from another location in the file system.*
![keep_software_path.png](../assets/images/keep_software_path.png)

* Navigate to the selected location in the file system and create the **markers** directory:
```
cd /project/<your_project_dir>/<user_account>/geospatial
mkdir markers
cd markers
```
![dir_for_markers.png](../assets/images/dir_for_markers.png)

* Then use the `aruco_make.py` script like this: <br>
`python aruco_make.py -d <DICT> -s <START> -e <END> -v`, for example:
```
source activate geospatial         # activate environment (if not activated yet)
python path_to_Find-GCP_dir/aruco_make.py -d 1 -s 0 -e 9 -v
conda deactivate                   # deactivate env when you are done with Find-GCP tasks
```

  ![aruco_make](../assets/images/aruco_make.png)

  <i>This command will create 10 markers, numbered from 0 to 9 (e.g., marker0.png), using the dictionary DICT_4x4_100.</i><br>
  * `-d <int>` option, the number determines the dictionary *(see the table below)*, default = 1

  |code  | dictionary|code  | dictionary |code  | dictionary |code  | dictionary  |
  |------|-----------|------|------------|------|------------|------|-------------|
  |**0** |DICT_4X4_50|**1** |DICT_4X4_100|**2** |DICT_4X4_250|**3** |DICT_4X4_1000|
  |**4** |DICT_5X5_50|**5** |DICT_5X5_100|**6** |DICT_5X5_250|**7** |DICT_5X5_1000|
  |**8** |DICT_6X6_50|**9** |DICT_6X6_100|**10**|DICT_6X6_250|**11**|DICT_6X6_1000|
  |**12**|DICT_7X7_50|**13**|DICT_7X7_100|**14**|DICT_7X7_250|**15**|DICT_7X7_1000|
  |**16**|DICT_ARUCO_ORIGINAL|
  |**17**|DICT_APRILTAG_16H5|**18**|DICT_APRILTAG_25H9|**19**|DICT_APRILTAG_36H10|**20**|DICT_APRILTAG_36H11|
  |**99**|DICT_3X3_32 custom|

  * `-s <int>` option, the number determines the index of the first marker, default = 0
  * `-e <int>` option, the number determines the index of the last marker, default = -1 <br> *(only one marker is generated with index 0)*
  * `-v` flag (optional) shows marker on monitor <br>*(when applicable, e.g., when working on a local machine)*
  * `-g` flag (optional) generates black/gray marker <br>*(instead black/white to reduce the effect of white burnt in)*
    * the optional `--value <VAL>` determines shade of background <br>*(use with `-g`, default=95)*
  * `-p <PAD>` (optional) determines border width around marker in inches, default= 0.5
<br><br>

**B. ArUco marker images in SVG** <br>
There is another GitHub repo, [gcp_aruco_generator](https://github.com/qaptadrone/gcp_aruco_generator), providing a simple python tool for generating ArUco markers with a real sizing of the image saved in SVG format. It also has a few more options, including `--print-id` in the corner of the marker and adding a watermark on the four borders. Follow the [Setup and use](https://github.com/qaptadrone/gcp_aruco_generator#setup-and-use) guide to get started with this tool.

<table>
<tr style="width: 100%">
  <td style="border: 1px solid white; width: 630px">
    <div style="background: #cff4fc; padding: 15px; height: 200px;">
    <span style="font-weight:800;">PRO TIP:</span><br>
    <span style="font-style:italic;">For good size recommendations, please see <a href="http://www.agt.bme.hu/on_line/gsd_calc/gsd_calc.html" target="_blank">http://www.agt.bme.hu/on_line/gsd_calc/gsd_calc.html</a>.<br><br>The generated ArUco markers are compatible with the Find-GCP tool, so you can use it after the flight to find the markers in your pictures.</span>
    </div><br>
    <div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
    <span style="font-weight:800;">WARNING:</span>
    <br><span style="font-style:italic;">When using the <b>gcp_aruco_generator</b> tool, be aware that the <b>ArUco marker IDs also start numbering from 0</b>, just like in standard ArUco dictionaries.</span>
    </div>
  </td>
  <td style="border: 1px solid white;"><img src="../assets/images/gcp_aruco_generator.png" style="width:270px;"></td>
</tr>
</table>

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px; margin-top: 0px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
If you're using <b>gcp_aruco_generator</b> to create ArUco codes for detection with the Find_GCP tool, <b>avoid using codes with IDs 9, 12, and 19</b>, as they are missing the central black square compared to the reference codes from the OpenCV library and will not be detectable. Check the repository's issue tracker (<a href="https://github.com/qaptadrone/gcp_aruco_generator/issues/3" target="_blank">issue #3</a>) to see if the developers have addressed this concern.
</span>
</div>

* generate a single marker from selected dictionary in size = 1000 mm (without margins):
```
python path/marker_generator.py -b -d 4X4_50 -s 1000 --id 0 --print-id
```

* generate 10 hand selected markers:
```
for i in 0 5 10 15 20 21 22 23 24 25
do
    python path/marker_generator.py -b -d 4X4_50 -s 1000 --id $i --print-id
done
```
![aruco_svg_1](../assets/images/aruco_svg_1.png)

* generate 10 consecutive markers with IDs in selected range:
```
for i in `seq 0 9`
do
    python path/marker_generator.py -b -d 4X4_50 -s 1000 --id $i --print-id
done
```
![aruco_svg_2](../assets/images/aruco_svg_2.png)


**C. ArUco marker images <u>generated online</u> (SVG or PDF)** <br>

<table>
<tr style="width: 100%">
<td style="border: 1px solid white; width: 600px; font-size: 20px;">
Finally, you can use the free online <br>ArUco markers generator: <a href="https://chev.me/arucogen/" target="_blank">https://chev.me/arucogen/</a> <br><br>
<i>See the corresponding GitHub repo:</i> <a href="https://github.com/okalachev/arucogen" target="_blank">https://github.com/okalachev/arucogen</a> <hr>
Save this marker as SVG, or open standard browser's print dialog to print or get the PDF.
</td>
<td style="border: 1px solid white;"><img src="../assets/images/online_generator.png" style="width:300px;"></td>
</tr>
</table>


## **Automatic recognition of ArUco codes**

For automatic recognition of ArUco markers, it's optimal to have your **land surveying imagery in the JPG format**. It's presumed that you've utilized printed ArUco markers as your Ground Control Points and **have diligently recorded GCPs precise positions**. This data should be saved in a text file, for instance, `GCP_reference.txt`. This file should feature four columns: `aruco_id`, `x`, `y`, and `z`, representing the marker ID and its three-dimensional coordinates respectively.

**INPUTS:** <br>
**-** imagery in JPG format <br>
**-** GCPs coordinate file, *e.g.,* `GCP_reference.txt`:
```
0 523287.368 4779588.335 1397.823
1 523394.376 4779529.525 1398.728
2 523350.181 4779492.395 1403.140
3 523363.938 4779530.027 1400.244
4 523364.648 4779587.932 1395.735
5 523329.480 4779525.642 1400.983
6 523347.074 4779571.424 1397.653
```
<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span><br>
<i><b>If your GCPs coordinate file uses custom IDs</b> (e.g., 131, 135, 143 when you used 4X4_50 ArUco dictionary)</i>, ensure you replace these with the appropriate ArUco marker IDs from the relevant dictionary before proceeding with automatic recognition of ArUco codes in your imagery.
</div>

<div style="background: #cff4fc; padding: 15px;">
<span style="font-weight:800;">PRO TIP:</span><br>
<i><b>If you're unable to match your custom IDs with the corresponding ArUco marker IDs</b></i>, you can still detect the markers in your imagery, but you won't be able to directly align them with the precise positions in your GCP coordinates file. <br><br>
<b>If you encounter this issue, you may follow one of the two scenarios:</b>
<ul><li>For simply identifying individual markers on your imagery and obtaining a list of images categorized by the detected markers, refer to section <a href="#scenario-3-no-gcp-file">SCENARIO 3: no GCP file</a>.</li>
<li>However, if your markers were placed at a sufficient distance apart, there's a good chance you can programmatically match the GCPs coordinates to the markers using the GPS coordinates embedded in the representative images <i>(I prepared a <u>ready-made Python script for this task</u>)</i>. To detect markers on imagery and subsequently assign them precise GCP coordinates, follow the guide in section <a href="#scenario-2-gcp-file-with-custom-ids">SCENARIO 2: GCP file with custom IDs</a>.</li>
<b>Note:</b> <i>Approach the results with caution, recognizing that the precision of such a method may be limited.</i>
</ul>
</div><br>

---

### **SCENARIO 1:** *GCP file with known ArUco IDs*

***i.e., Direct ArUco ID match***

This approach is for those possessing a GCP file with recognized ArUco IDs:
* by **inputting your imagery** and the `GCP_reference.txt` file along with the **known ArUco dictionary**,
* you'll receive an output in the form of `gcp_list.txt`. *This file provides precise world coordinates for the GCPs as well as coordinates on the corresponding images.*
  * ***This output file is ready for immediate use with OpenDroneMap (ODM) software.***

1. Login to the Atlas cluster using SSH protocol (command line) or OOD access (web-based).
2. Navigate to your ODM working directory. *Use the command below:*
```
cd /project/<path_to_your_project_directory>/ODM
```
![navigate_to_odm.png](../assets/images/navigate_to_odm.png)
<b>PRO TIP:</b> <i>If you haven't set up the ODM directory structure yet, please follow the guide provided in section [Create File Structure](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/02-ODM-modules#create-file-structure) in the tutorial [Command-line ODM modules](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/02-ODM-modules).</i> <br>
Create a subdirectory for your new project in the IMAGES directory and create soft links for your imagery and (eventually) the `GCP_reference.txt` file:
```
cd IMAGES
mkdir project-X
cd project-X
ln -s <source_path_to_imagery>/* ./
ls | head -10
pwd                                  # copy this path in the next step as the INPUTS_PATH variable
```
![/set_up_inputs.png](../assets/images/set_up_inputs.png) <br><br>
<i>If your GCP reference file (here: gcp_epsg32611_2021_wbs1_coresite.csv) has format different than <b>space-separated 4 columns: aruco_ID X Y Z</b>, then you should adjust it accordingly to get something like this:</i>
```
0 523287.368 4779588.335 1397.823
1 523394.376 4779529.525 1398.728
2 523350.181 4779492.395 1403.140
3 523363.938 4779530.027 1400.244
4 523364.648 4779587.932 1395.735
5 523329.480 4779525.642 1400.983
6 523347.074 4779571.424 1397.653
```
<i>You can use</i> `awk` *command to easily extract the columns you need.* <b>Note that GCP_reference.txt file should not have a header.</b>
![adjust_gcp_reference.png](../assets/images/adjust_gcp_reference.png)

3. Set paths as temporary variables or use them directly:
```
FIND_GCP_PATH=/path/to/Find-GCP_repo
INPUTS_PATH=/path/to/input_imagery
```
<i>In my case the path variables look like this:</i>
![paths_as_variables.png](../assets/images/paths_as_variables.png)

4. Activate the Conda environment (if not activated yet). *You should activate a specific conda environment related to this project (e.g., the geospatial env created in section <br>[Find-GCP python utility: installation](#find-gcp-python-utility-installation)):*
```
source activate geospatial
```
5. Run the `gcp_find.py` Python tool:
```
python $FIND_GCP_PATH/gcp_find.py -v -t ODM -i $INPUTS_PATH/GCP_reference.txt --epsg <code> -d <int> -o gcp_list.txt $INPUTS_PATH/*.JPG
```
<i>Replace \<code> with the EPSG of your GCP coordinate system and \<int> with an ID corresponding to the used ArUco dictionary. For example:</i>
```
python $FIND_GCP_PATH/gcp_find.py -v -t ODM -i $INPUTS_PATH/GCP_reference.txt --epsg 32611 -d 0 -o gcp_list.txt $INPUTS_PATH/*.JPG
```

<details><summary style="color: #ff3870;font-weight: 500; margin-left: 37px;"><b>Explore all options</b></summary>

<div style="background: #e6f0f0; padding: 5px; margin-top: 10px;">
usage: gcp_find.py [-h] [-d DICT] [-o OUTPUT] [-t {ODM,VisualSfM}] [-i INPUT] <br>
&emsp; &emsp; &emsp; [-s SEPARATOR] [-v] [--debug | --multi] [-l] [--epsg EPSG] <br>
&emsp; &emsp; &emsp; [-a] [--markersize MARKERSIZE] [--markerstyle MARKERSTYLE] <br>
&emsp; &emsp; &emsp; [--markerstyle1 MARKERSTYLE1] [--edgecolor EDGECOLOR] <br>
&emsp; &emsp; &emsp; [--edgewidth EDGEWIDTH] [--fontsize FONTSIZE] <br>
&emsp; &emsp; &emsp; [--fontcolor FONTCOLOR] [--fontcolor1 FONTCOLOR1] <br>
&emsp; &emsp; &emsp; [--fontweight FONTWEIGHT] [--fontweight1 FONTWEIGHT1] <br>
&emsp; &emsp; &emsp; [--limit LIMIT] [--nez] [-r] [--winmin WINMIN] <br>
&emsp; &emsp; &emsp; [--winmax WINMAX] [--winstep WINSTEP] [--thres THRES] <br>
&emsp; &emsp; &emsp; [--minrate MINRATE] [--maxrate MAXRATE] [--poly POLY] <br>
&emsp; &emsp; &emsp; [--corner CORNER] [--markerdist MARKERDIST] <br>
&emsp; &emsp; &emsp; [--borderdist BORDERDIST] [--borderbits BORDERBITS] <br>
&emsp; &emsp; &emsp; [--otsu OTSU] [--persp PERSP] [--ignore IGNORE] <br>
&emsp; &emsp; &emsp; [--error ERROR] [--correct CORRECT] <br>
&emsp; &emsp; &emsp; [--refinement REFINEMENT] [--refwin REFWIN] <br>
&emsp; &emsp; &emsp; [--maxiter MAXITER] [--minacc MINACC] <br>
&emsp; &emsp; &emsp; [file_names [file_names ...]]
</div><br>
<b>positional arguments:</b><br>
&emsp; file_names &emsp; &emsp; &emsp; image files to process <br><br>
<b>optional arguments (common):</b><br>

<table>
  <tr style="background-color:#f0f0f0; border-bottom: 1px solid black;">
    <th width="180">flag</th><th>values</th><th>default</th><th width="250">description</th><th>notes</th></tr>
  <tr>
    <td>-h <br>--help</td><td> </td><td> </td><td>show this help message and exit</td><td> </td></tr>
  <tr>
    <td><b>-d DICT <br>--dict DICT</b></td><td>integer</td><td>1</td><td>marker dictionary ID, default=1 (DICT_4X4_100)</td><td><i>ID determines which set of markers the program will recognize. Ensure you match this with the markers you're using. </i></td></tr>
  <tr>
    <td>-l <br>--list</td><td> </td><td> </td><td>output dictionary names and ids and exit</td><td><i>This is a quick way to view available dictionaries. It's useful if you're unsure which dictionary ID to use.</i></td></tr>
  <tr>
    <td><b>-o FILE <br>--output FILE</b></td><td>filename</td><td><i>stdout</i></td><td>name of output GCP list file, default stdout</td><td><i> If no output file name is specified, the results will be printed directly to the terminal (stdout).</i></td></tr>
  <tr>
    <td><b>-t {ODM,VisualSfM} <br>--type {ODM,VisualSfM}</b></td><td>ODM, VisualSfM</td><td> </td><td>target program ODM or VisualSfM, default</td><td><i>Make sure to select the target program that aligns with your project needs. Both ODM (OpenDroneMap) and VisualSfM have different formats.</i></td></tr>
  <tr>
    <td><b>-i GCP_FILE <br>--input GCP_FILE</b></td><td> </td><td><i>4-col file</i></td><td>name of input GCP coordinate file, default None</td><td><i>If not provided, the tool won't assign the reference coordinates to detected markers.</i></td></tr>
  <tr>
    <td><b>--epsg EPSG</b></td><td> </td><td> </td><td>epsg code for gcp coordinates, default None</td><td><i>Ensure the EPSG code matches the coordinate system of your GCPs. This ensures that coordinates are interpreted correctly.</i></td></tr>
  <tr>
    <td>-s SEPARATOR <br>--separator SEP</td><td>' ' , </td><td><i>space</i></td><td>input file separator</td><td><i>If you notice issues with reading your input file, double-check that the specified separator matches the one used in your file.</i></td></tr>
  <tr>
    <td>-v <br>--verbose</td><td> </td><td><i>off</i></td><td>verbose output to stdout</td><td><i>Use this option if you want detailed logs during processing. It can help with troubleshooting.</i></td></tr>
  <tr>
    <td>--debug</td><td> </td><td><i>off</i></td><td>show detected markers on image</td><td><i>This is particularly helpful for visual verification. When activated, you'll get images showing where markers were detected. Works on a local machine or with X11 forwarding.</i></td></tr>
  <tr>
    <td>--multi</td><td> </td><td><i>off</i></td><td>process images paralel</td><td><i>Use this for faster processing if you're working with multiple images, as it processes them in parallel.</i></td></tr>
  <tr>
    <td>-r, --inverted</td><td> </td><td><i>off</i></td><td>detect inverted markers</td><td><i>This ensures the tool will recognize markers even if they're inverted, increasing the robustness of your detection process.</i></td></tr>
</table>

<br><b>optional arguments (more customization):</b><br>
<a href="https://github.com/zsiki/Find-GCP#gcp_findpy" target="_blank">Additional customization options  ⤴</a> allow users to adjust color schemes, define marker styles and attributes, set font characteristics for debug images, limit record output, reorder coordinates, detect inverted markers, fine-tune adaptive thresholding, specify marker characteristics, determine marker-border relations, and refine marker detection through various parameters such as accuracy, error rates, and iteration limits.
</details>

<div style="margin-left: 37px; margin-top: 10px;">
<img src="../assets/images/run_gcp_find.png" alt="run_gcp_find.png">

This will search the ArUco markers from DICT_4x4_50 in your imagery and match them with corresponding IDs provided in your <b>GCP_reference.txt</b> file. Providing the exact EPSG code will ensure the returned coordinates of the GCPs detected in the imagery are in the correct coordinate system. The list of images with detected GCPs is saved to the <b>gcp_list.txt</b> file, which looks like this: <br>
<div style="background: #e6f0f0; padding: 15px;">
EPSG:32611 <br>
523287.368 4779588.335 1397.823 5041 91 R0036021.JPG 0 <br>
523287.368 4779588.335 1397.823 5190 1110 R0036023.JPG 0 <br>
523287.368 4779588.335 1397.823 5462 1856 R0036024.JPG 0 <br>
523287.368 4779588.335 1397.823 5680 2998 R0036026.JPG 0 <br>
523364.648 4779587.932 1395.735 3170 60 R0036061.JPG 4 <br>
523347.074 4779571.424 1397.653 624 700 R0036065.JPG 6 <br>
523347.074 4779571.424 1397.653 539 1349 R0036066.JPG 6 <br>
523305.976 4779572.588 1397.817 162 701 R0036073.JPG 10 <br>
523305.976 4779572.588 1397.817 87 1597 R0036074.JPG 10 <br>
523364.648 4779587.932 1395.735 4892 3940 R0036042.JPG 4 <br>
</div><br>
You can further refine the output file by sorting the records based on the ArUco marker ID, allowing you to <b>choose a subset of 5-10 images for each marker, required by the ODM software</b>. While manually reviewing the images is the most reliable approach to select the best representations, you can initially narrow down the number of images per marker programmatically. After this automatic reduction, a visual review is recommended to address any ambiguous images.
</div>


### ***Select representative images for a marker***

Optimally, a marker should be positioned near the center of the image. The marker's position in an image can be approximated using its target coordinates, which are located in the 4th and 5th columns of the output file, i.e., `gcp_list.txt`. <br>
In some cases, you might find that a given marker is detected in several dozens to even hundreds of images.
```
awk '{print $7}' < gcp_list.txt | sort | uniq -c
```
*The 7th column stores the ArUco marker ID. You can easily count the number of images matched with a given ID. In my case, the result is like this (counts, ID):*
```
  30 0
  40 1
  40 2
  20 3
  23 4
  37 5
  24 6
  25 7
  37 8
  33 10
  28 11
```

A practical strategy is to first **employ an automated filter to narrow down to approximately 10 images per marker**, and subsequently perform a visual check to ensure accuracy.
* To select the N=10 images per marker ID where the marker is placed closest to the center of the image, we can use a Python script `gcp_images_picker.py`. Here's the outline of the steps:
  * Calculate the distance of each marker from the center of the image using the Euclidean distance formula.
  * Sort the images for each marker ID based on this distance in ascending order.
  * Select the top N images for each marker ID.

1. Make sure you have your local copy of the **[geo_utils](https://github.com/ISUgenomics/geo_utils)** GitHub repository, placed in your SOFTWARE or TOOLS directory on Atlas (e.g., `project/<account>/<user>/SOFTWARE`). *(You can follow the instructions in section [geo_utils Python utility: installation](#geo_utils-python-utility-installation) to download this repository.)* We will use the Python script `gcp_images_picker.py` located in the TOOLS subdir of this repo:
  ```
  ls /project/<your_project_dir>/<user_account>/SOFTWARE
  ```

2. Make sure you navigate back to the IMAGES directory in your photogrammetry project. You can softlink the `gcp_images_picker.py` script for easy use:
  ```
  cd project/<your_project_dir>/<user_account>/ODM/IMAGES/<project-X>
  ln -s /project/<your_project_dir>/<user_account>/SOFTWARE/geo_utils/TOOLS/gcp_images_picker.py ./
  ```

3. Run the `gcp_images_picker.py` script to automate selection of representative GCP images, minimizing manual inspection:
  ```
  # Usage:
  python gcp_images_picker.py -i <data_file_path> -w <image_width> -l <image_height> -n <images_number> -o <custom_outfile>
  ```
  *for example:*
  ```
  python gcp_images_picker.py -i gcp_list.txt -w 6000 -l 4000 -n 10
  ```
  *The script will write the selected data to the file specified by the -o option. If the option isn't provided, it defaults to* `gcp_list_selected.txt`.
  ```
  523394.376 4779529.525 1398.728 3420 1497 R0036704.JPG 4
  523394.376 4779529.525 1398.728 2236 1940 R0036753.JPG 4
  523394.376 4779529.525 1398.728 2329 1549 R0036752.JPG 4
  523394.376 4779529.525 1398.728 3526 1127 R0036703.JPG 4
  523394.376 4779529.525 1398.728 3924 1274 R0036038.JPG 4
  523394.376 4779529.525 1398.728 4216 1930 R0036039.JPG 4
  523394.376 4779529.525 1398.728 4372 2125 R0036678.JPG 4
  523394.376 4779529.525 1398.728 4302 1536 R0036677.JPG 4
  523394.376 4779529.525 1398.728 4364 2421 R0036040.JPG 4
  523394.376 4779529.525 1398.728 4216 1199 R0036676.JPG 4
  ```

  <div style="background: #cff4fc; padding: 15px; margin-left: 37px;">
  <span style="font-weight:800;">PRO TIP:</span><br>
  While the script does a job of pre-selecting images, it's recommended taking a moment to <b>visually inspect the chosen images</b>. This ensures that markers are clearly visible and that the annotations with ArUco ID align with the correct pattern. A brief manual check can help enhance the accuracy and reliability of your dataset. <br>
  <i>Automation aids efficiency, but a human touch ensures precision!</i>
  </div><br>


### ***Visual check of representative images for a marker***

Ensuring that selected images are truly representative for each Ground Control Point (GCP) is a crucial step for accurate georeferencing. The `gcp_check.py` tool (from [Find-GCP repo](https://github.com/zsiki/Find-GCP)) offers a **user-friendly graphical interface to facilitate the visual check** of GCPs detected by `gcp_find.py`.

<div style="background: mistyrose; padding: 15px; margin-bottom: 30px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Proceeding with this section <b>requires the use of Secure SHell (SSH) paired with X11 forwarding</b>. X11 forwarding enables graphical applications run on the cluster to manifest visually on your local machine. <br><br>
When connecting via SSH, use the -X (or -Y, which is less secure but more permissive) option to enable X11 forwarding. <br>
<i>(For me, the -Y variant worked. Note it may be slow!)</i><br>
<b>ssh -Y user.name@atlas-login.hpc.msstate.edu</b><br><br>
Using <a href="https://atlas-ood.hpc.msstate.edu" target="_blank">Atlas Open OnDemand (OOD)  ⤴</a> (in-browser access) can offer a more responsive experience compared to traditional SSH with X11 forwarding, especially over slower connections. <br><br>
If the cluster doesn't permit the use of X11 forwarding and OOD also didn't work for you, you'll need to take a detour:<br>
1. Download the relevant images (only those from gcp_list.txt) to your local machine. <br>
2. Clone the Find-GCP repository again but this time on your local setup. <br>
3. Then continue with the subsequent steps of this section. <br>
*<i>Be sure to modify the paths to align with your local filesystem during this process.</i>
</span>
</div>

If you've followed this tutorial, you should have already cloned the Find-GCP repository (see section [Find-GCP Python utility: installation](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/03-ODM-georeferencing#find-gcp-python-utility-installation)). As a result, the `gcp_check.py` utility would be included within your cloned repo, ready for use.

Check your `SOFTWARE` path on the Atlas cluster (for reference, see step 7 *"Create the storage directory for custom software and GitHub repos"* in section [Create env for geospatial analysis](#create-env-for-geospatial-analysis)):
```
ls /project/<your_project_dir>/SOFTWARE
```
If you have the **Find-GCP** repo cloned already:
* set up its path as a local variable *(remember to adjuct the path)*:
```
FIND_GCP_PATH=/project/<your_project_dir>/SOFTWARE/Find-GCP
```
* activate your `geospatial` Conda environment *(if not activated yet)* or follow the guide provided in section [reate env for geospatial analysis](#create-env-for-geospatial-analysis) to make up for this step.
```
source activate geospatial
```

**INPUTS:**
* `project-X` directory with the complete imagery
* `gcp_list_selected.txt` file *(The (filtered) output file of the* `gcp_find.py` *is the input file of this program.)*

```
EPSG:32611
523287.368 4779588.335 1397.823 5041 91 R0036021.JPG 0
523287.368 4779588.335 1397.823 5190 1110 R0036023.JPG 0
523287.368 4779588.335 1397.823 5462 1856 R0036024.JPG 0
523287.368 4779588.335 1397.823 5680 2998 R0036026.JPG 0
523364.648 4779587.932 1395.735 3611 96 R0036036.JPG 4
523364.648 4779587.932 1395.735 3714 535 R0036037.JPG 4
523364.648 4779587.932 1395.735 3924 1274 R0036038.JPG 4
523364.648 4779587.932 1395.735 4216 1930 R0036039.JPG 4
```

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Please be cautious: the script <b>gcp_check.py can NOT be executed within the directory containing the input images</b>. Instead, navigate one level up in the directory structure before running it.
</span>
</div>

Let's assume you store your `gcp_list.txt` file along with your imagery at the `IMAGES/<project-X>` path. If so, navigate one level up:
```
cd /project/<your_project_dir>/<user_account>/ODM/IMAGES
```

Now, you can launch the `gcp_check.py` GUI by executing this command in the terminal:
```
python FIND_GCP_PATH/gcp_check.py --path ./<project-X>/ --edgewidth 5 --fontsize 300 ./<project-X>/gcp_list_selected.txt
```
<i>To modify how detected markers are highlighted,<b> explore the command-line parameters</b> outlined in the [official documentation](https://github.com/zsiki/Find-GCP#gcp_checkpy).</i>

After executing the command, a window showcasing the GUI will appear. Within this interface, you can use:
* forward and backward buttons (on the top) to navigate between images
* the mouse wheel to zoom in or out
* the left mouse button to pan across the image

![check_gcp.png](../assets/images/check_gcp.png) <br>
<i>The tool is designed to automatically identify markers within images. Once detected, it highlights these markers with a circle and displays the corresponding ArUco ID. This setup aids in easy visual verification, ensuring that markers are correctly recognized and labeled.</i>

---

### **SCENARIO 2:** *GCP file with custom IDs*

***i.e., Custom ID integration***

For cases where you have a GCP file with custom IDs, your inputs will be **the imagery**, a `GCP_reference.txt` file with <u>custom marker IDs</u>, a **known ArUco dictionary**, and the **EPSG code** for the registered GCPs.

**EPSG**: 32611 <br>

```
# GCP_reference.txt
131 523287.368 4779588.335 1397.823
135 523305.976 4779572.588 1397.817
137 523347.074 4779571.424 1397.653
141 523364.648 4779587.932 1395.735
134 523394.376 4779529.525 1398.728
133 523363.938 4779530.027 1400.244
138 523329.480 4779525.642 1400.983
136 523350.181 4779492.395 1403.140
132 523289.018 4779469.252 1407.142
139 523292.432 4779530.710 1401.051
143 523261.422 4779532.114 1401.978
```

**STEP 0.** Prepare your working space.

1. Login to the Atlas cluster using SSH protocol (command line) or OOD access (web-based).
2. Navigate to your project working directory. *Use the command below:*
```
cd /project/<path_to_your_project_directory>/ODM
```
![navigate_to_odm.png](../assets/images/navigate_to_odm.png)
<b>PRO TIP:</b> <i>If you haven't set up the ODM directory structure yet, please follow the guide provided in section [Create File Structure](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/02-ODM-modules#create-file-structure) in the tutorial [Command-line ODM modules](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/02-ODM-modules).</i> <br>
Create a subdirectory for your new project in the IMAGES directory and create soft links for your imagery and (eventually) the `GCP_reference.txt` file:
```
cd IMAGES
mkdir project-X
cd project-X
ln -s <source_path_to_imagery>/* ./
ls | head -10
pwd                                  # copy this path in the next step as the INPUTS_PATH variable
```

3. Set paths as temporary variables or use them directly:
```
FIND_GCP_PATH=/path/to/Find-GCP_repo
INPUTS_PATH=/path/to/input_imagery
```
<i>In my case the path variables look like this:</i>
![paths_as_variables.png](../assets/images/paths_as_variables.png)

4. Activate the Conda environment (if not activated yet). *You should activate a specific conda environment related to this project (e.g., the geospatial env created in section [Find-GCP python utility: installation](#find-gcp-python-utility-installation)):*
```
source activate geospatial
```

**STEP 1.** The first step involves identifying the ArUco IDs of markers detected on the imagery and selecting a representative image for each. Ideally, this image should predominantly feature the marker centrally placed.

1. Run the `gcp_find.py` Python tool with basic settings to detect ArUco markers:
```
python $FIND_GCP_PATH/gcp_find.py -d 0 $INPUTS_PATH/*.JPG >> markers_detected.txt
```
<i>Remember to use the correct ArUco dictionary with the <b>-d</b> option (for example, in this case, the -d 0 denotes the DICT_4x4_50 dictionary).</i><br>
The output `markers_detected.txt` should looks like this (*columns:* `x` `y` `image` `aruco_id`):
```
5041 91 R0036021.JPG 0
5190 1110 R0036023.JPG 0
5462 1856 R0036024.JPG 0
5680 2998 R0036026.JPG 0
3170 60 R0036061.JPG 4
624 700 R0036065.JPG 6
539 1349 R0036066.JPG 6
162 701 R0036073.JPG 10
```
Let's filter the output to include only the images with a single detected marker on them.
```
awk '{print $3}' markers_detected.txt | sort | uniq -c | awk '$1 == 1 {print $2}' | while read image; do grep "$image" markers_detected.txt; done > single_markers.txt
```
<i>The </i> `single_markers.txt` *has the same data structure but contains filtered records.*

2. Extract unique ArUco marker IDs from the 4th column:
```
awk '{print $4}' < markers_detected.txt | sort -n | uniq > marker_ids
```
<i>The </i> `marker_ids` *file contains unique marker IDs stored in a column.*

3. For each marker ID, select representative image, i.e., the image where the marker is placed closest to the center:
```
for i in `cat marker_ids`; do awk -v A=$i '{if ($4==A) print $0}' < single_markers.txt | awk 'BEGIN {min_dist = 1000000000} {dist = sqrt(($1-3000)^2 + ($2-2000)^2); if(dist < min_dist) {min_dist = dist; closest = $0}} END {print closest}' >> representatives; done
```
<i>This assumes that the image dimensions are 6000x4000 px, so the coordinates of the center of the picture are (x = 6000/2 = <b>3000</b>, y = 4000/2 = <b>2000</b>). <b>Remember to adjust the command for your values.</b></i><br>
The output `representatives` should contain the representative image for each marker ID: <br>*(with marker's XY coordinates in the image provided in the first 2 columns)*
```
3546 1937 R0036737.JPG 0
3631 2017 R0036909.JPG 1
3290 1831 R0036401.JPG 2
3118 1948 R0036914.JPG 3
3420 1497 R0036704.JPG 4
3001 1808 R0036953.JPG 5
3345 2661 R0036140.JPG 6
2531 2038 R0036933.JPG 7
2693 1783 R0036927.JPG 8
2993 2141 R0036789.JPG 10
3120 2112 R0037136.JPG 11
```
4. You can visually inspect the selected images to ensure they indeed showcase the distinct pattern of the detected ArUco marker ID and confirm that each image contains only one marker. *In my case, all detected markers match the pattern of a suggested ArUco ID.*
![](../assets/images/aruco_detected.png) <br>

5. Create a subdirectory and copy in or soft link the representative images:
```
mkdir representative
awk '{print $3}' < representatives > list
for i in `cat list`; do k=`echo $i | awk -F"." '{print $1}'`; n=`cat representatives | awk -v A=$i '{if ($3==A) print $4}'` ; cp $INPUTS_PATH/$i representative/$k"_"$n.JPG; done
```
<i>Now, the images should be copied into the <b>representative</b> subdirectory and their names should change from R0036737.JPG to R0036737<b>_0</b>.JPG denoting the ID of detected ArUco marker (which is required in the next step).</i><br>
![selected_representatives.png](../assets/images/selected_representatives.png)


**STEP 2.** In the second step, a Python script `gcp_to_aruco_mapper.py` automatically matches the GCP coordinates with the representative images (by calculating the distance between GCPs from reference file and GPS coordinates of each picture).
  * If there's a coordinate system difference between reference GCPs and imagery GPS, a conversion is carried out for the GCPs. <br>***(So, that's why you must provide the correct EPSG code.)***<br>

1. Make sure you have your local copy of the **[geo_utils](https://github.com/ISUgenomics/geo_utils)** GitHub repository, placed in your SOFTWARE or TOOLS directory on Atlas (e.g., `project/<account>/<user>/SOFTWARE`). *(You can follow the instructions in section [geo_utils Python utility: installation](#geo_utils-python-utility-installation) to download this repository.)* We will use the Python script `gcp_to_aruco_mapper.py` located in the TOOLS subdir of this repo:
  ```
  ls project/<your_project_dir>/<user_account>/SOFTWARE
  ```

2. Make sure you navigate back to the `representative` directory in your photogrammetry project. You can softlink the `gcp_to_aruco_mapper.py` script for easy use:
  ```
  cd project/<your_project_dir>/<user_account>/ODM/<project_X>/IMAGES/representative
  ln -s project/<your_project_dir>/<user_account>/SOFTWARE/geo_utils/TOOLS/gcp_to_aruco_mapper.py ./
  ```
  ![softlink_tool.png](../assets/images/softlink_tool.png)

3. Run the `gcp_to_aruco_mapper.py` script to match the GCP coordinates with the representative images:
```
python gcp_to_aruco_mapper.py -g ../GCP_reference.txt -i "./" -z 11 -o matching_results -d 50 > out_distances
grep "Match" < matching_results | sort -nk4 > ID_matches
cat ID_matches
```
<i>In this command, the <b>-g</b> argument specifies the GCP file with custom IDs, the <b>-b</b> option provides the path with selected representative images for unique ArUco codes detected, <b>-z</b> option expects you to provide the UTM zone (e.g., if EPSG is 32611 then the UTM zone is 11), <b>-d</b> option determines the maximum distance threshold between GCP coordinates and image GPS.</i>
```
Match found: GCP 131 (d=16.84m) is likely in image R0036737_0.JPG with ArUco marker 0.
Match found: GCP 132 (d=12.69m) is likely in image R0037136_11.JPG with ArUco marker 11.
Match found: GCP 133 (d=1.64m) is likely in image R0036914_3.JPG with ArUco marker 3.
Match found: GCP 134 (d=1.00m) is likely in image R0036909_1.JPG with ArUco marker 1.
Match found: GCP 135 (d=3.09m) is likely in image R0036789_10.JPG with ArUco marker 10.
Match found: GCP 136 (d=18.49m) is likely in image R0036401_2.JPG with ArUco marker 2.
Match found: GCP 137 (d=25.80m) is likely in image R0036140_6.JPG with ArUco marker 6.
Match found: GCP 138 (d=15.18m) is likely in image R0036953_5.JPG with ArUco marker 5.
Match found: GCP 139 (d=4.29m) is likely in image R0036927_8.JPG with ArUco marker 8.
Match found: GCP 141 (d=5.95m) is likely in image R0036704_4.JPG with ArUco marker 4.
Match found: GCP 143 (d=7.09m) is likely in image R0036933_7.JPG with ArUco marker 7.
```

<div style="background: mistyrose; padding: 15px; margin-bottom: 25px; margin-left: 37px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">Note that you should have activated a specific conda environment related to this project. See the <b>STEP 0</b> in this section. </span>
</div>

*  ***You might be curious about the success rate of matching GCPs with image GPS.***<br> Thankfully, I got reference data where GCPs were paired with ArUco codes during land surveying. As you can see below, the `gcp_to_aruco_mapper.py` accurately matched all GCPs with their corresponding ArUco markers. Thus, <b>this method serves as a reliable fallback when you're unsure of the ArUco IDs for your ground control coordinates.</b>
```
# programmatic detection                  |  reference created during land surveying
GCP 131 (d=16.84m) with ArUco marker  0   |   0 523287.368 4779588.335 1397.823 131
GCP 132 (d=12.69m) with ArUco marker 11   |  11 523289.018 4779469.252 1407.142 132
GCP 133  (d=1.64m) with ArUco marker  3   |   3 523363.938 4779530.027 1400.244 133
GCP 134  (d=1.00m) with ArUco marker  1   |   1 523394.376 4779529.525 1398.728 134
GCP 135  (d=3.09m) with ArUco marker 10   |  10 523305.976 4779572.588 1397.817 135
GCP 136 (d=18.49m) with ArUco marker  2   |   2 523350.181 4779492.395 1403.140 136
GCP 137 (d=25.80m) with ArUco marker  6   |   6 523347.074 4779571.424 1397.653 137
GCP 138 (d=15.18m) with ArUco marker  5   |   5 523329.480 4779525.642 1400.983 138
GCP 139  (d=4.29m) with ArUco marker  8   |   8 523292.432 4779530.710 1401.051 139
GCP 141  (d=5.95m) with ArUco marker  4   |   4 523364.648 4779587.932 1395.735 141
GCP 143  (d=7.09m) with ArUco marker  7   |   7 523261.422 4779532.114 1401.978 143
```

<div style="background: #cff4fc; padding: 15px;  margin-bottom: 25px; margin-left: 37px;">
<span style="font-weight:800;">PRO TIP:</span>
<br><span style="font-style:italic;">
<b>Always create a GCP-to-ArUco reference during land surveying.</b> This crucial step simplifies your geospatial analysis and georeferencing, allowing you to easily adhere to the guidelines outlined in <a href="https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/03-ODM-georeferencing#scenario-1-gcp-file-with-known-aruco-ids">SCENARIO 1: GCP file with known ArUco IDs</a>.
</span>
</div>

**STEP 3.** Once the matches are made, create a **new** `GCP_reference.txt` file replacing the custom IDs with ArUco IDs.

Create a one column `tmp` file with matching IDs in a format `GCP_ArUco`. Then, replace the custom GCP ID with the corresponding ArUco ID in your original `GCP_reference.txt` file.
```
awk '{print $4"_"$14}' < ID_matches | tr '.' ' ' > ../tmp
cd ../                               # navigate to the IMAGES dir with the GCP_reference.file

for i in `cat tmp`
do
    old=`echo $i | awk -F"_" '{print $1}'`
    new=`echo $i | awk -F"_" '{print $2}'`
    awk -v A=$old -v B=$new '{if ($1==A) print B,$2,$3,$4}' < GCP_reference.txt >> GCP_reference_aruco.txt
done
```
<i>The output from this operation is the </i> `GCP_reference_aruco.txt` file, used in STEP 4.
```
0 523287.368 4779588.335 1397.823
11 523289.018 4779469.252 1407.142
3 523363.938 4779530.027 1400.244
1 523394.376 4779529.525 1398.728
10 523305.976 4779572.588 1397.817
2 523350.181 4779492.395 1403.14
6 523347.074 4779571.424 1397.653
5 523329.48 4779525.642 1400.983
8 523292.432 4779530.71 1401.051
4 523364.648 4779587.932 1395.735
7 523261.422 4779532.114 1401.978
```

**STEP 4.** The `gcp_find.py` tool is then utilized again as in [SCENARIO 1: GCP file with known ArUco IDs](#scenario-1-gcp-file-with-known-aruco-ids). The end output, `gcp_list.txt`, is compatible with ODM software, but it should be used cautiously due to limited precision of GCP matching in this approach.

---

### **SCENARIO 3:** *no GCP file*

***i.e., No GCP reference, pure detection of ArUco markers***

In instances where no GCP file is available:
* The goal here is straightforward: ***How many and which ArUco markers are detectable within your imagery?***
* You only need **your imagery** and a **known ArUco dictionary**, *which can be discerned through a visual inspection of a sample marker in a photo*.
* The output will be a list that pairs ArUco IDs with the respective image names where they were spotted *(with marker coordinates in the picture)*.

**Identify marker IDs in the selected ArUco dictionary:**

Run the `gcp_find.py` Python tool with basic settings to detect ArUco markers:
```
$FIND_GCP_PATH/gcp_find.py -d 0 $INPUTS_PATH/*.JPG >> markers_detected.txt
```

<i>Remember to use the correct ArUco dictionary with the <b>-d</b> option (for example, in this case, the -d 0 denotes the DICT_4x4_50 dictionary).</i>
The output `markers_detected.txt` should looks like this (*columns:* `x` `y` `image` `aruco_id`):
```
5041 91 R0036021.JPG 0
5190 1110 R0036023.JPG 0
5462 1856 R0036024.JPG 0
5680 2998 R0036026.JPG 0
3170 60 R0036061.JPG 4
624 700 R0036065.JPG 6
539 1349 R0036066.JPG 6
162 701 R0036073.JPG 10
```
Let's filter the output to include only the images with a single detected marker on them.
```
awk '{print $3}' markers_detected.txt | sort | uniq -c | awk '$1 == 1 {print $2}' | while read image; do grep "$image" markers_detected.txt; done > single_markers.txt
```

You can further refine your results by selecting the top 10 most representative images for each identified ArUco marker, as outlined in section
[Select representative images for a marker](#select-representative-images-for-a-marker).





___
# Further Reading
<!-- * [Command-line ODM modules](02-ODM-modules) -->


___

[Homepage](../index.md){: .btn  .btn--primary}
[Section Index](../00-IntroPhotogrammetry-LandingPage){: .btn  .btn--primary}
[Previous](02-ODM-modules){: .btn  .btn--primary}
<!-- [Next](){: .btn  .btn--primary} -->
