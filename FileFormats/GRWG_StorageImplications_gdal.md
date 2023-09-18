---
title: "Raster file properties: computation time and storage implications"
layout: single
author: Heather Savoy
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

**Last Update:** 17 September 2023 <br />

## Overview

This tutorial explores raster file properties, how those properties affect file 
sizes, and how they affect geospatial processing execution times. Commands shown 
are for direct use of the Geospatial Data Abstraction Library (GDAL) which is a 
library that is often used by other geospatial software including R packages, 
Python packages, QGIS, and ESRI products. 

Both Atlas and Ceres have GDAL installed as a module. To start 
using GDAL on either cluster, load the module with the following command:

```bash
module load gdal
```

*Language:* `bash`

*Primary Libraries/Packages:*

| Name | Description | Link |
|:--|:--|:--|
| GDAL | Geospatial Data Abstraction Library | https://gdal.org/ |

## Nomenclature

* *Raster:* Data defined on a grid of geospatial coordinates
* *Block:* The configuration of how raster values are read from the file. 
* *Data type:* If raster values are stored as integers or floating point numbers, 
  if they are signed, and how many bits are used to store a value.  


## Sections

* Print file properties of raster datasets
* Effects of data types on file sizes
* Effects of tiles on cropping execution time
* Effects of virtual rasters on cropping execution time

## Section 1: Print file properties of raster datasets

First, we will look into what kind of raster file properties can be provided by GDAL. 
There is a command called `gdalinfo` which will print such information for whichever
raster filename you pass to it. 

```bash
gdalinfo <raster_filepath>
```

To illustrate a variety of raster file property combinations, we will look at the datasets
availabile in the Geospatial Common Data Library (GeoCDL) on Ceres (see 
[this tutorial](../ExampleGeoWorkflows/GRWG22_GeoCDL_R.md) 
to learn more). 

For example, calling `gdalinfo` on one of the source PRISM data files in the GeoCDL, 
the output looks like this:

```bash
gdalinfo prism/PRISM_ppt_stable_4kmM3_200807_bil.bil
```

```
Driver: EHdr/ESRI .hdr Labelled
Files: prism/PRISM_ppt_stable_4kmM3_200807_bil.bil
       prism/PRISM_ppt_stable_4kmM3_200807_bil.bil.aux.xml
       prism/PRISM_ppt_stable_4kmM3_200807_bil.hdr
       prism/PRISM_ppt_stable_4kmM3_200807_bil.stx
       prism/PRISM_ppt_stable_4kmM3_200807_bil.prj
Size is 1405, 621
Coordinate System is:
GEOGCRS["NAD83",
    DATUM["North American Datum 1983",
        ELLIPSOID["GRS 1980",6378137,298.257222101,
            LENGTHUNIT["metre",1]],
        ID["EPSG",6269]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["Degree",0.0174532925199433]],
    CS[ellipsoidal,2],
        AXIS["longitude",east,
            ORDER[1],
            ANGLEUNIT["Degree",0.0174532925199433]],
        AXIS["latitude",north,
            ORDER[2],
            ANGLEUNIT["Degree",0.0174532925199433]]]
Data axis to CRS axis mapping: 1,2
Origin = (-125.020833333333357,49.937499999999751)
Pixel Size = (0.041666666666700,-0.041666666666700)
Corner Coordinates:
Upper Left  (-125.0208333,  49.9375000) (125d 1'15.00"W, 49d56'15.00"N)
Lower Left  (-125.0208333,  24.0625000) (125d 1'15.00"W, 24d 3'45.00"N)
Upper Right ( -66.4791667,  49.9375000) ( 66d28'45.00"W, 49d56'15.00"N)
Lower Right ( -66.4791667,  24.0625000) ( 66d28'45.00"W, 24d 3'45.00"N)
Center      ( -95.7500000,  37.0000000) ( 95d45' 0.00"W, 37d 0' 0.00"N)
Band 1 Block=1405x1 Type=Float32, ColorInterp=Undefined
  Min=0.000 Max=556.523 
  Minimum=0.000, Maximum=556.523, Mean=72.033, StdDev=61.444
  NoData Value=-9999
  Metadata:
    STATISTICS_MAXIMUM=556.52301025391
    STATISTICS_MEAN=72.033263630928
    STATISTICS_MINIMUM=0
    STATISTICS_STDDEV=61.444120607318
```

This output has some recognizable geospatial metadata such as the coordinate reference 
system (`Coordinate System`) and the spatial resolution (`Pixel Size`). The raster file 
properties discernable from this output that we will be focusing on are:

* *File format* (`Driver`, `Files`): This is discerned by the file extension of the
  raster file being described and potentially also the GDAL driver used to read
  that type of file. Here, the PRISM file has an extension `.bil` for which GDAL 
  uses its `EHdr/ESRI .hdr Labelled` driver to read. The `.bil` format is developed 
  by ESRI, along with other similar raster file formats, so GDAL uses a combined 
  driver to read these formats. When reading the GDAL documentation about what 
  format-specific options there might be, 
  [the content will be organized by driver](https://gdal.org/drivers/raster/index.html). 
  For example, there is a [GeoTIFF driver](https://gdal.org/drivers/raster/gtiff.html).
* *Data type* (`Type`): If raster values are stored as integers or floating point numbers, 
  if they are signed, and how many bits are used to store a value.  
* *X-Y dimensions* (`Size`): This is the number of columns and rows of pixels in the
  image discretizing space (typically horizontal space). 
* *Block size* (`Block`): The dimensions (columns and rows) of the tile or strip 
  (or scanline) that represent the file access chunk size. When blocks are configured 
  as square subsets of a raster, they are referred to as tiles. 

Here is a table summarizing these properties from several of the GeoCDL datasets:

| Dataset     | File format | Data type           | X-Y dimensions  | Block size  |
|:--          |:--          |:--                  |:--              |:--          |
| DaymetV4    | .nc         | Float32             | 7814, 8075      | 1000x1000   | 
| GTOPO       | .dem        | Int16               | 4800, 6000[^1]  | 4800x1      | 
| NASS CDL    | .tif        | Byte                | 153811, 96523   | 512x512     | 
| NLCD        | .img        | Byte                | 161190, 104424  | 512x512     | 
| PRISM       | .bil        | Float32             | 1405, 621       | 1405x1      | 
| SMAP-HB 1km | .nc         | Float32             | 6996, 3120      | 1400x624    | 
| SRTM        | .bil        | Int16               | 3601,3601[^1]   | 3601x1      | 
| VIP         | .hdf        | Int16, UInt16, Int8 | 3600, 7200      | 512x512     | 


[^1]: This large dataset is stored in multiple individual files covering different geographic areas. The properties listed here are for one of the individual files. 


The following sections will explore some file storage implications for data types 
and computation time implications for block sizes.
 

## Section 2: Effects of data types on file sizes

Data types indicate the range and precision of values that can be stored in 
the file. There are three main components to differentiate data types:

* **Integer or floating point**: is it a whole number (integer) or a decimal 
  (floating point)? It takes more bits to store the extra precision of 1.0001
  than 1, and thus you can store a greater range of values as integers than 
  floats or consume less disk space storing similar ranges of values as integers
  than floats.   
* **Signed or unsigned**: do the values range from negative to positive, or 
  are they strictly non-negative? This effects the range of values that can be 
  stored.
* **Number of bits**: how many bits are used to store a value? This is typically 
  8-bit to 64-bit and dictates the ultimate file size. 

The values that are able to be stored in a raster depend on the data type. 
For example, the `UInt16` (unsigned 16-bit integer) data type can 
hold a range of 2^16 integers within [0, 65535]. The `Int16` 
(signed 16-bit integer) data type also covers 2^16 integer values, but within
[-32768, 32767]. 

The data types that GDAL supports in ascending order of size are: 
`Byte`, `Int8` <  `UInt16`, `Int16` < `UInt32`, `Int32`, `Float32` < `UInt64`, `Int64`, `Float64`. 
The support for some of these data types will depend on the version of GDAL 
installed. Based on our table of GeoCDL datasets, the `Byte`, `Int16`, and 
`Float32` are common data types. Each band in a raster data file can have 
its own data type, e.g., VIP has multiple data types in the table above. 
Note that these data type names are from GDAL's naming scheme. Other 
programs might have slightly different names, e.g. 'uint8' 
instead of 'Byte' for an unsigned 8-bit integer. 

Some datasets, like [VIP](https://doi.org/10.5067/MEaSUREs/VIP/VIP01.004), 
use scaling factors (e.g., dividing NDVI by 0.0001) 
to help reduce file size with the use of integer instead of float data types.  
We can explore this concept a bit further to see the effect of modifying a 
raster's data type (with and without scaling) on the file size and data loss.

For example, let's use our example PRISM data file again, which holds precipitation
(ppt) values as the `Float32` data type. Since precipitation is typically a measurement
of height and a continuous number, storing it as a floating point number makes sense.
However, depending on the precision needed, disk space could be saved by storing 
these precipitation values as scaled integers. In the `gdalinfo` output printed above,
the range of precipitation values in the file is [0, 556.523] mm. If we scale those values
by 100 ([0, 55652.3]) and then save them as integers ([0, 55652]), we could store them as 
`UInt16`, which would take up half the disk space (going from 32-bit to 16-bit data type) 
while also only losing precision around 0.001 mm of precipitation. We could not, however, 
successfully store all values as `Int16` since 55652 is outside the supported range 
of [-32768, 32767] (pixels with scaled values >32767 would become no-data pixels). 
Also, precipitation itself is non-negative, so an unsigned data type is sensible.  

The table below summarizes a range of scaling and data type conversion outcomes for
this PRISM file. Ultimately, scaling by 100 and saving the result as a `UInt16` has
the best balance of reducing file size and preventing data loss out of all scenarios
considered.

| Data type | Writing error | File size (MB)  | Scaling factor -> RMSE from <br />original file (% pixels lost) |
|:--        |:--            |:--              |:--                                                        |
| `Byte`      | Warning: detected values <br />outside of the limits of datatype <br />INT1U (for all scaling factors)| 0.873 | 1 (None) -> 0.56 (1.2%) <br />10 -> 0.054 (70%) <br />100 -> 0.0042 (92%) |
| `UInt16`    |               | 1.7             | 1 (None) -> 0.56 <br />10 -> 0.056 <br />100 -> 0.0057 |
| `Int16`     | Warning: detected values <br />outside of the limits of datatype <br />INT2S (for scaling factor = 100) | 1.7 | 1 (None) -> 0.56 <br />10 -> 0.056 <br />100 -> 0.0057 (0.3%) |
| `UInt32`    |               | 3.5             | 1 (None) -> 0.56 <br />10 -> 0.056 <br />100 -> 0.0057 |
| `Int32`     |               | 3.5             | 1 (None) -> 0.56 <br />10 -> 0.056 <br />100 -> 0.0057 |
| `Float32` <br />(original data type) | | 3.5  | N/A |


## Section 3: Effects of tiles on cropping execution time

Modifying the block size of a raster can improve the computation time of certain
geospatial analyses. For example, we can look at the effect of setting tiles for a
GeoTIFF so that values are read in by square (typically of dimensions that are 
multiples of 16) chunks. 

In GDAL, you can create tiles or change tile sizes for a raster with the `gdal_translate`
command with the following parameters: 

```bash
gdal_translate -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 input.tif interim.tif
```

Or remove or not create tiles:

```bash
gdal_translate -co TILED=NO input.tif interim.tif
```

To illustrate the effect of tiles on computation time, lets take the NASS CDL dataset.
From our dataset table above, we see that it is already tiled with blocks of 512x512 pixels. 
We can use the `gdal_translate -co TILED=NO` line above to make a copy that is not tiled, 
which results in a block size of 153811x1 (one row of pixels across the whole image.) 
This dataset covers CONUS and we will ask GDAL to crop it to some small county-sized area
specified by a shapefile:

```bash
gdalwarp -crop_to_cutline -cutline my_county.shp interim.tif output.tif
```

The table below shows how long that cropping command took to execute with the without
the tiles. With tiles, it took half the time! This is because GDAL had to read in a smaller
amount of the entire raster to cover the small cropping area, a small number of 
local 512x512 tiles instead of many 153811x1 strips across CONUS. So this computation time benefit
will depend on the relative sizes of the blocks, the cropping area, and the original raster extent. 

| Tile/block size | Computation time (s)  |
|:--              |:--                    |
| 512x512         | ~0.8                  |
| 153811x1        | ~1.5                  |


## Section 4: Effects of virtual rasters on cropping execution time

Another raster file concept that can save computation time is the GDAL Virtual Format raster (VRT). 
This is essentially a way to store interim raster operation *instructions*, as opposed to the
actualy interim raster *values*, which can save processing time if the total sequence of all
raster operations needed can be calculated in one combined GDAL operation. 

For example, let's revisit the process of creating tiled GeoTIFFs to improve cropping execution 
times. If we request first a tiled geoTIFF, then crop it: 

```bash
gdal_translate -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 input.tif interim.tif
gdalwarp -crop_to_cutline -cutline input.shp interim.tif output.tif
```
that can still take several minutes to tile if that GeoTIFF is large, e.g. several GB. 

What we can do instead is request a tiled VRT and pass that (essentially instructions, not data)
to the cropping command:

```bash
gdal_translate -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 input.tif interim.vrt
gdalwarp -crop_to_cutline -cutline input.shp interim.vrt output.tif
```

This generates the same result as making the interim GeoTIFF instead of the VRT, but the
VRT method takes less than 1 second instead of minutes. 

| VRT | Task                                    | Computation time (s)  |
|:--  |:--                                      |:--                    |
| N   | Creating the tiled GeoTIFF (several GB) | ~ 4.5 min             |
| N   | Cropping the tiled GeoTIFF              | < 1 sec               |
| Y   | Creating the tiled VRT (37 KB)          | < 1 sec               |
| Y   | Cropping the tiled VRT:                 | < 1 sec               |