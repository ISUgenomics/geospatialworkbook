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
In any case, to make geospatial localization of [aerial] photos, **GPS data** *(Global Positioning System)* is required. Most modern professional cameras record GPS coordinates automatically in the images, usually with an accuracy of 10-30 feet (3-9 meters) *[[source](https://www.blog.jimdoty.com/?p=14661)]*. While such a result is sufficient for most ordinary purposes, your research may be more precision demanding. For example, consider the case where the target area is even smaller than the geolocation threshold. The novel more accurate geolocation systems, such as VPS *(Visual Positioning System)* or CPS *(Camera Positioning Standard)*, are now being developed *[[learn more](https://www.mosaic51.com/community/alternative-to-gps-how-to-get-better-accuracy/)]* and will probably supplant GPS technology in the future. By this time though, the best patch for improving georeferencing process is still to use a **high-accuracy GPS point reference**, which will minimize the error.


## Georeferencing options in OpenDroneMap

The user has some level of control over the ODM settings for the photo georeferencing stage. By default ODM tries to use the GPS information embedded in the images automatically while recording the mission. If this is the case, you don't need to add any additional option for geolocation to be performed.

**1. Force the use of geolocation from images' EXIF metadata**

```
--force-gps --use-exif \
```

Use `--force-gps` and `--use-exif` flags when you have a GCP data file in the project file structure but <u>want to force</u> the use of the GPS data stored in the image metadata.

This is especially useful when the original imagery is not geotagged but you have GPS data in separate text files. In such a case you can add this information to the image EXIF metadata using [ExifTool](https://exiftool.org) software. Follow the instructions in the tutorial [Keep EXIF GEO metadata](https://geospatial.101workbook.org/IntroPhotogrammetry/OpenDroneMap/00-IntroODM#keep-exif-geo-metadata) *(section: add EXIF tags from a text file using exiftool)* to accomplish this step.

**2. Force the use of GPS data (e.g., RTK) from a text file**

```
--geo geo.txt \
```

Regardless of whether your imagery is geotagged, once you have alternative GPS information <u>stored in a text file</u>, you can force direct use of it with optin `--geo geo.txt`. This is especially useful when you have more **accurate geolocation data such as RTK** *(Real-Time Kinematic positioning)* that corrects some of common errors in current satellite navigation (GNSS) systems.

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
Keep in mind that the format of the text file containing the GPS data is strictly defined! <br><br>
If the <b>geo.txt</b> file is somewhere outside of your project's workdir or you have several GPS files, then provide the <b>absolute path</b> to the one you want.
</span>
</div>

<div style="background: #cff4fc; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">geo.txt</span> <i>(file content below)</i></span>
<span style="color:navy;"><i>[see details in the <a href="https://docs.opendronemap.org/geo/#image-geolocation-files" style="color: blue;">ODM Documentation: GPS data</a>]</i></span>
<br><br>
+proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs<br>
DJI_0028.JPG &emsp; -91.9942096 &emsp; 46.8425252 &emsp; 198.609 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; # GPS data <br>
DJI_0032.JPG &emsp; -91.9938293 &emsp; 46.8424584 &emsp; 198.609 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; # GPS data <br>
</div>

* The first line should contain the name of the projection used for the geo coordinates, in one of the following formats:

```
* PROJ string:   +proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs
*   EPSG code:   EPSG:4326
*   WGS84 UTM:   WGS84 UTM 16N
```

* Subsequent lines are the GPS information for a given image <u><i>(first 3 columns are obligatory)</i></u>:

```
     col-1 col-2 col-3  col-4              col-5          col-6             col-7                    col-8                    col-9      col-10
     _____ _____ _____  _____              _____          _____             _____                    _____                    _____      ______
image_name geo_x geo_y [geo_z] [omega (degrees)] [phi (degrees)] [kappa (degrees)] [horz accuracy (meters)] [vert accuracy (meters)] [extras...]
```


**3. Force fixed value of GPS Dilution of Precision** *(use along with variant 1 or 2)*

```
--gps-accuracy 10 .0
```

If you know the **estimated error of GPS** location determined by the camera in use, consider setting it as a value of the `--gps-accuracy value` option. The value is a positive float in metres and will be used as a GPS Dilution of Precision for all images. *The default is 10 meters.*

<div style="background: mistyrose; padding: 15px; margin-bottom: 20px;">
<span style="font-weight:800;">WARNING:</span>
<br><span style="font-style:italic;">
If you use high precision GPS (RTK), this value will be set automatically. You can manually set it in case the reconstruction fails. Lowering the value can help control bowling-effects over large areas.
</span>
</div>

**3. Force the use of GCP-based georeferencing**

**Ground Control Points** (GCPs) are clearly visible objects which can be easily identified in several images. Using the precise GPS position of these ground points is a good reference that improves significantly the accuracy of the project's geolocation. Ground control points can be any **steady structure** existing in the mission area, otherwise can be set using **targets placed on the ground**. *Learn more about recommended practices for GCPs in ODM workflow from the OpenDronMap Documentation: [Ground Control Points](https://docs.opendronemap.org/gcp/#ground-control-points).*

If you have a file with GCPs detected on the image collection, force georeferencing using it by option `--gcp gcp_list.txt`.

```
--gcp gcp_list.txt
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
<span style="font-weight:800;">geo.txt</span> <i>(file content below)</i></span>
<span style="color:navy;"><i>[see details in the <a href="https://docs.opendronemap.org/gcp/#ground-control-points" style="color: blue;">ODM Documentation: GCPs</a>]</i></span>
<br><br>
EPSG:4326 <br>
-116.7499838 &emsp; 43.06477015 &emsp; 2090.149 &emsp; 1559.4164501833 &emsp; 1372.84669162591 &emsp; DJI_0177.JPG &emsp; 100 <br>
-116.7499838 &emsp; 43.06477015 &emsp; 2090.149 &emsp; 1491.0163890586 &emsp; 2471.85207823960 &emsp; DJI_0355.JPG &emsp; 100 <br>
-116.7499838 &emsp; 43.06477015 &emsp; 2090.149 &emsp; 1524.1419621026 &emsp; 2214.43593367970 &emsp; DJI_0178.JPG &emsp; 100 <br>
-116.7499838 &emsp; 43.06477015 &emsp; 2090.149 &emsp; 1142.5991517178 &emsp; 1739.80028544801 &emsp; DJI_0152.JPG &emsp; 100 <br>
-116.7499838 &emsp; 43.06477015 &emsp; 2090.149 &emsp; 1207.8811697738 &emsp; 1863.28946363080 &emsp; DJI_0329.JPG &emsp; 100 <br>
-116.7504805 &emsp; 43.06475631 &emsp; 2088.227 &emsp; 1737.2264669926 &emsp; 1763.28507029339 &emsp; DJI_0172.JPG &emsp; 101 <br>
-116.7504805 &emsp; 43.06475631 &emsp; 2088.227 &emsp; 1660.2427796454 &emsp; 2912.50267420537 &emsp; DJI_0350.JPG &emsp; 101 <br>
-116.7504805 &emsp; 43.06475631 &emsp; 2088.227 &emsp; 1736.7057610024 &emsp; 1411.55810666259 &emsp; DJI_0171.JPG &emsp; 101 <br>
-116.7504805 &emsp; 43.06475631 &emsp; 2088.227 &emsp; 989.57852613080 &emsp; 1391.94185513447 &emsp; DJI_0157.JPG &emsp; 101 <br>
-116.7504805 &emsp; 43.06475631 &emsp; 2088.227 &emsp; 877.82625305623 &emsp; 2459.65369040342 &emsp; DJI_0335.JPG &emsp; 101 <br>
</div>

* The first line should contain the name of the projection used for the geo coordinates, in one of the following formats:

```
* PROJ string:   +proj=utm +zone=11 +ellps=WGS84 +datum=WGS84 +units=m +no_defs
*   EPSG code:   EPSG:4326
*   WGS84 UTM:   WGS84 UTM 16N
```

* Subsequent lines are the GPS information for a given image (first 3 columns are obligatory):

```
col-1 col-2 col-3 col4 col5    col-6     col-7     col-8    col-9
_____ _____ _____ ____ ____ __________ __________ ________ ________
geo_x geo_y geo_z im_x im_y image_name [gcp_name] [extra1] [extra2]
```


# Software for manual detection of GCPs

# Software for automatic detection of GCPs


___
# Further Reading
* [Command-line ODM modules](02-ODM-modules)


___

[Homepage](../index.md){: .btn  .btn--primary}
[Section Index](../00-IntroPhotogrammetry-LandingPage){: .btn  .btn--primary}
[Previous](00-IntroODM){: .btn  .btn--primary}
[Next](02-ODM-modules){: .btn  .btn--primary}
