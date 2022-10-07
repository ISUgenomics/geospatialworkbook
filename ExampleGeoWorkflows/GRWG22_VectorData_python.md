---
title: Handling vector data
layout: single
author: Heather Savoy
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

**Last Update:** 7 October 2022 <br />
**Download Jupyter Notebook**: [GRWG22_VectorData.ipynb](https://geospatial.101workbook.org/tutorials/GRWG22_VectorData.ipynb)

## Overview
This tutorial covers how to manipulate geospatial vector datasets in python. A 
dataset with polygon geometries representing wildfire perimeters is joined with 
a dataset with point geometries representing air quality monitoring stations to
determine which stations observed unhealthy concentrations of small particulate 
matter (PM2.5) in the atmosphere around the time of the Camp Fire in northern CA 
in 2018.  

*Language*: `Python`

*Primary Libraries/Packages*:

|Name|Description|Link|
|-|-|-|
| geopandas | GeoPandas is an open source project to make working with geospatial data in python easier. | https://geopandas.org/en/stable/ |
| plotnine | A Grammar of Graphics for Python | https://plotnine.readthedocs.io/en/stable/ |


## Nomenclature

* *Vector data:* Spatial data defined by simple geometry types (points, lines, 
  and polygons) where each geometry feature can be assigned non-spatial attributes.
* *CRS:* Coordinate Reference System, also known as a spatial reference system. A
  system for defining geospatial coordinates.
* *Spatial join:* Combining two spatial datasets by the relationship between their
  geometries.

## Data Details

* Data: National Interagency Fire Center's Historic Perimeters dataset
* Link: [https://data-nifc.opendata.arcgis.com/datasets/nifc::historic-perimeters-combined-2000-2018-geomac/explore](https://data-nifc.opendata.arcgis.com/datasets/nifc::historic-perimeters-combined-2000-2018-geomac/explore)
* Other Details: The original dataset contains perimeters of wildfires in the US 
  from 2000-2018 as a polygon feature collection. For this tutorial, the wildfire
  perimeters in CA during 2018 were extracted. 

* Data: US EPA's Air Quality System (AQS) database
* Link: [https://aqs.epa.gov/aqsweb/documents/data_api.html](https://aqs.epa.gov/aqsweb/documents/data_api.html)
* Other Details: PM2.5 concentration data from this database covering CA in 2018 
  were retrieved and pre-processed for this tutorial. 

## Analysis Steps

  1. **Fire perimeter data**
    - read in and visualize the wildfire perimeter data
       - Read in geojson file
       - Visualize perimeters on map of CA
       - Visualize non-spatial attributes
  2. **Air quality data**
    - read in the shapefile
  3. **Buffer and spatial join**
    - find the air quality stations within 200km of the fire perimeter
  4. **Visualize**
    -  air quality around the Camp Fire

### Step 0: Import Libraries / Packages

Below are commands to run to create a new Conda environment named 'geoenv' that contains the packages used in this tutorial series. To learn more about using Conda environments on Ceres, see [this guide](https://scinet.usda.gov/guide/conda/). NOTE: If you have used other Geospatial Workbook tutorials from the SCINet Geospatial Research Working Group Workshop 2022, you may have aleady created this environment and may skip to launching JupyterHub.

First, we allocate resources on a compute (Ceres) or development (Atlas) node so we do not burden the login node with the conda installations. 

On Ceres:
```bash
salloc
```

On Atlas (you will need to replace `yourProjectName` with one of your project's name):
```bash
srun -A yourProjectName -p development --pty --preserve-env bash
```

Then we load the `miniconda` conda module available on Ceres and Atlas to access the `Conda` commands to create environments, activate them, and install Python and packages.

```bash
salloc
module load miniconda
conda create --name geoenv
source activate geoenv
conda install geopandas rioxarray rasterstats plotnine ipython ipykernel dask dask-jobqueue -c conda-forge
```

To have JupyterLab use this conda environment, we will make a kernel.


```bash
ipython kernel install --user --name=geo_kernel
```

This tutorial assumes you are running this python notebook in JupyterLab. The 
easiest way to do that is with Open OnDemand (OoD) on [Ceres](http://ceres-ood.scinet.usda.gov/)
or [Atlas](https://atlas-ood.hpc.msstate.edu/). 
Select the following parameter values when requesting a JupyterLab
app to be launched depending on which cluster you choose. All other values can 
be left to their defaults. Note: on Atlas, we are using the development partition
so that we have internet access to download files since the regular compute nodes
on the `atlas` partition do not have internet access.

Ceres:
* `Slurm Partition`: short
* `Number of hours`: 1
* `Number of cores`: 2
* `Jupyer Notebook vs Lab`: Lab

Atlas:
* `Partition Name`: development 
* `QOS`: normal
* `Number of hours`: 1
* `Number of tasks`: 2

To download the python notebook file for this tutorial to either cluster within OoD, 
you can use the following lines in the python console:

```python
import urllib.request
tutorial_name = 'GRWG22_VectorData.ipynb'
urllib.request.urlretrieve('https://geospatial.101workbook.org/tutorials/' + tutorial_name, 
                           tutorial_name)
```

Once you are in JupyterLab with this notebook open, select your kernel by clicking on the *Switch kernel* button in the top right corner of the editor. A pop-up will appear with a dropdown menu containing the *geo_kernel* kernel we made above. Click on the *geo_kernel* kernel and click the *Select* button. 


```python
import geopandas as gpd
import numpy as np
from plotnine import ggplot, geom_map, aes, theme, geom_histogram, scale_x_datetime, geom_line, ylab, xlab, annotate, geom_vline
from datetime import datetime, date
```

### Step 1: Read in fire perimeter data and visualize

The National Interagency Fire Center's Historic Perimeters dataset has 23,776 
polygons representing wildfire perimeters from 2000-2018. A version of the dataset 
filtered to wildfires in CA in 2018, since that is when the 
destructive Camp Fire occurred, will be used in this tutorial. 

We will transform to planar coordinates for distance calculations. The 5070 EPSG
code is for the Equal Area CONUS Albers. 


```python
fire_f = 'Historic_Perimeters_Combined_2000-2018_GeoMAC_CA2018.geojson'
dnld_url = 'https://geospatial.101workbook.org/ExampleGeoWorkflows/assets/'
all_fires = gpd.read_file(dnld_url + fire_f)
fire_CA2018 = all_fires.to_crs(5070)
```

To get an idea of what the data look like, we can create a map. Since this 
feature collection has several attributes, we can also visualize, for example, 
when the fires occurred during the year.


```python
# CA boundary
census_states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip')
CA = census_states.loc[census_states['NAME'] == 'California'].to_crs(fire_CA2018.crs)

# The Camp Fire feature
camp_fire = fire_CA2018.loc[fire_CA2018['incidentname'] == 'CAMP']

# Plot a map of all wildfires and 
# highlight the Camp Fire.
ggplot() + \
    geom_map(CA, fill = 'white') + \
    geom_map(fire_CA2018, fill = 'black', color = None) + \
    geom_map(camp_fire, fill = 'red', color = None) + \
    theme(figure_size = (6,8))

```
![png](assets/Session8_Tutorial1_11_0.png)
    

Since this feature collection has several attributes, we can also visualize, 
for example, when the fires occurred during the year. Note that the dates are
when the fires' perimeters are established, not when the fires start. Since fires 
can endure for many days and change their spatial extent over that time, this
date is when the maximum extent of the fire was determined, which needs to be 
at the end of the fire.


```python
# Plotting when wildfires occurred throughout 2018
ggplot(fire_CA2018, aes('perimeterdatetime')) + \
  geom_histogram() + \
  scale_x_datetime(breaks = '1 month',
                   date_labels = '%b')
```
    
![png](assets/Session8_Tutorial1_13_1.png)
    

### Step 2: Read in air quality data

We will read in an air quality dataset to showcase combining multiple vector 
datasets. The geometry type is point, representing sampling stations. This 
dataset can be downloaded with an API, but you need to register an account, so 
the download process has been done already and the data are available in our 
GitHub repository. Some pre-processing for this dataset was done to make it into 
a feature collection.


```python
CA_PM25 = gpd.read_file(dnld_url + 'air_quality_CA2018.zip').to_crs(fire_CA2018.crs)
```

### Step 3: Find the air quality stations within 200km of the fire perimeter

We will buffer the Camp Fire polygon and join that to the air quality data to 
find the nearby stations. Note that the direction of joining is important: the 
geometry of the left dataset, in this case the air quality points, will be 
retained and the attributes of intersecting features of the right dataset will 
be retained. 


```python
camp_zone = camp_fire.copy()   
camp_zone['geometry'] = camp_fire.buffer(200000) # meters, buffer is in unit of CRS
air_fire = CA_PM25.sjoin(camp_zone, how="inner", predicate='intersects')
```

### Step 4: Visualize air quality around the Camp Fire

'Around' in this case should be in both space and time. We already found the 
air quality stations within 200 km from the fire perimeter. To make sure we are
looking at the right time of year, we will relate the non-spatial attributes of
the two feature collections. Both datasets come with date columns: when the air
quality was recorded and when the perimeter was determined (so the latter is 
an approximate time near the end of the fire). We will filter our spatially-
joined dataset to within 30 days of the fire perimeter date so we only consider
air quality within a 2-month window around the time of the fire. 

We can then plot the air quality metric of PM2.5 concentration (lower is better) 
as a function of time before/after the fire perimeter date. Each line is an 
individual air quality station. The figure indicates that there were several 
stations that recorded unhealthy to hazardous PM2.5 concentrations around the
time of this wildfire. 


```python
# Filter to 30 days from fire perimeter date -
# local date for air quality measurement
air_fire['dat_local'] = [datetime.strptime(dt, '%Y-%m-%d').date() for dt in list(air_fire['dat_lcl'])] 
# fire perimeter date
air_fire['perimeterdate'] = [dt.date() for dt in air_fire['perimeterdatetime']] 
air_fire['date_shift'] = air_fire['dat_local'] - air_fire['perimeterdate']
# Arithmetic mean of PM2.5 concentration
air_fire['arthmt_'] = air_fire['arthmt_'].astype('float64')
air_fire['station_id'] = air_fire['stat_cd'] + air_fire['cnty_cd'] + air_fire['st_nmbr']
air_near_fire = air_fire.loc[abs(air_fire['date_shift']).astype('timedelta64[D]') < 30]

# Define bounds of Camp Fire dates for illustrative purposes
# Camp Fire burned for 17 days- add a polygon to graph to visualize temporal overlap with low air quality
camp_dates = [datetime.strptime(dt, '%Y-%m-%d').date() for dt in ['2018-11-08','2018-11-25']]

# Camp Fire's maximum perimeter established at end of fire
fp_date = air_fire['perimeterdate'][0]

ggplot(air_near_fire, aes('date_shift','arthmt_')) + \
  annotate('rect', xmin = camp_dates[0], xmax = camp_dates[1],
           ymin = -np.inf, ymax = np.inf,
           fill = 'firebrick', alpha = 0.5) + \
  geom_line(aes(group='station_id')) + \
  geom_vline(xintercept = fp_date, 
             color = 'red',
             size = 1,
             linetype = 'dashed') + \
  scale_x_datetime(name = 'Date', date_breaks = "10 days", date_labels = "%m-%d-%y") + \
  ylab('PM2.5 [micrograms/cubic meter]')
```
 
![png](assets/Session8_Tutorial1_19_0.png)
    

We can then map where those stations were - quite an 
extensive impact on air quality!


```python
# 101 micrograms/cubic meter and higher is considered
# unhealthy for sensitive groups 
unhealthy_air = air_near_fire.loc[air_near_fire['arthmt_'] > 100]

ggplot() + \
   geom_map(CA, fill = 'white') + \
   geom_map(camp_zone, fill = None, color = 'red') + \
   geom_map(camp_fire, fill = 'red', color = None) + \
   geom_map(unhealthy_air, color = 'red') + \
   theme(figure_size = (6,8))

```
    
![png](assets/Session8_Tutorial1_21_0.png)
    
