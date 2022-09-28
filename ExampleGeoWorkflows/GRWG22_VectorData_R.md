---
title: Handling vector data
layout: single
author: Heather Savoy
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

**Last Update:** 8 September 2022 <br />
**Download RMarkdown**: [GRWG22_VectorData.Rmd](https://geospatial.101workbook.org/tutorials/GRWG22_VectorData.Rmd)

<!-- ToDo: would be great to have an R binder badge here -->

## Overview

This tutorial covers how to manipulate geospatial vector datasets in R. A 
dataset with polygon geometries representing wildfire perimeters is joined with 
a dataset with point geometries representing air quality monitoring stations to
determine which stations observed unhealthy concentrations of small particulate 
matter (PM2.5) in the atmosphere around the time
of the Camp Fire in northern CA in 2018. 

*Language:* `R`

*Primary Libraries/Packages:*

| Name | Description | Link |
|:--|:--|:--|
| `sf` | Simple features for R | https://cran.r-project.org/web/packages/sf/index.html |
| `USAboundaries` | Historical and Contemporary Boundaries of the United States of America | https://cran.r-project.org/web/packages/USAboundaries/index.html |
| `ggplot2` | Create Elegant Data Visualisations Using the Grammar of Graphics | https://cran.r-project.org/web/packages/ggplot2/index.html |

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

* Fire perimeter data - read in and visualize the wildfire perimeter data
  * Read in geojson file
  * Visualize perimeters on map of CA
  * Visualize non-spatial attributes
* Air quality data - read in the shapefile
* Buffer and spatial join - find the air quality stations within 200km of the 
  fire perimeter
* Visualize - air quality impact around the Camp Fire


## Step 0: Import Libraries/Packages

```r
library(sf)               # Handling vector data
library(USAboundaries)    # Mapping administrative boundaries
library(dplyr)            # General data manipulation
library(ggplot2)          # Visualizations
library(lubridate)        # Data manipulation
```

## Step 1: Read in fire perimeter data and visualize

The National Interagency Fire Center's Historic Perimeters dataset has 23,776 
polygons representing wildfire perimeters from 2000-2018. A version of the dataset 
filtered to wildfires in CA in 2018, since that is when the 
destructive Camp Fire occurred, will be used in this tutorial.

We will transform to planar coordinates for distance calculations. The 5070 EPSG
code is for the Equal Area CONUS Albers. If you kept the dataset in it's 
original CRS, WGS 84, `sf` would warn you about distance calculations not being 
accurate. 

```r
fire_f <- 'Historic_Perimeters_Combined_2000-2018_GeoMAC_CA2018.geojson'
dnld_url <- 'https://raw.githubusercontent.com/HeatherSavoy-USDA/geospatialworkbook/master/ExampleGeoWorkflows/assets/'
httr::GET(paste0(dnld_url,fire_f),
          httr::write_disk(fire_f,
                           overwrite=TRUE))

fire_CA2018 <- st_read(fire_f) %>%
  st_transform(5070)
```

To get an idea of what the data look like, we can create a map. 

```r
# CA boundary
CA <- us_states() %>%
            filter(state_abbr == 'CA') %>%
  st_transform(st_crs(fire_CA2018))

# The Camp Fire feature
camp_fire <- fire_CA2018 %>%
  filter(incidentname == 'CAMP')

# Plot a map of all wildfires and 
# highlight the Camp Fire.
fire_CA2018 %>%
  ggplot() +
  geom_sf(data = CA) +
  geom_sf(fill = 'black',
          color = NA) +
  geom_sf(fill = 'red',
          color = NA,
          data = camp_fire)

```

![fire_map](assets/R_fire_map-1.png)

Since this feature collection has several attributes, we can also visualize, 
for example, when the fires occurred during the year.

```r
# Plotting when wildfires occurred throughout 2018
fire_CA2018 %>%
  select(perimeterdatetime) %>%
  ggplot(aes(perimeterdatetime)) +
  geom_histogram() +
  scale_x_datetime(breaks = '1 month',
                   date_labels = '%b')
```
![fire_time](assets/R_fire_time-1.png)


## Step 2: Read in air quality data

We will read in an air quality dataset to showcase combining multiple vector 
datasets. The geometry type is point, representing sampling stations. This 
dataset can be downloaded with an API, but you need to register an account, so 
the download process has been done already and the data are available in our 
'session6' folder. Some pre-processing for this dataset was done to make it into 
a feature collection.

```r
aq_base <- 'air_quality_CA2018'
aq_zip <- paste0(aq_base,'.zip')
#dnld_url <- 'https://github.com/HeatherSavoy-USDA/geospatialworkbook/raw/master/ExampleGeoWorkflows/assets/'
httr::GET(paste0(dnld_url,aq_zip),
          httr::write_disk(aq_zip,
                           overwrite=TRUE))
unzip(aq_zip)

ca_PM25 <- st_read(paste0(aq_base,'.shp'))
```

## Step 3: Find the air quality stations within 200km of the fire perimeter

We will buffer the Camp Fire polygon and join that to the air quality data to 
find the nearby stations. Note that the direction of joining is important: the 
geometry of the left dataset, in this case the air quality points, will be 
retained and the attributes of intersecting features of the right dataset will 
be retained. 

```r
camp_zone <- st_buffer(camp_fire, 200000) # meters, buffer is in unit of CRS

air_fire <- st_join(ca_PM25, camp_zone)
```

## Step 4: Visualize air quality around the Camp Fire

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

```r
# Filter to 30 days from fire perimeter date
air_near_fire <- air_fire %>%
  mutate(date_shift = ymd(dat_lcl) - as.Date(perimeterdatetime),
         PM25 = as.numeric(arthmt_),
         station_id = paste(stat_cd, cnty_cd, st_nmbr)) %>% 
  filter(abs(date_shift) <= 30)

air_near_fire %>%
  ggplot(aes(date_shift,PM25)) +
  geom_line(aes(group = station_id)) +
  scale_x_continuous(name = 'Days before/after fire perimeter') +
  scale_y_continuous(name = 'PM2.5 [micrograms/cubic meter]')

```

![join_plot](assets/R_join_plot-1.png)

We can then map where those stations were - quite an 
extensive impact on air quality!

```r
# 101 micrograms/cubic meter and higher is considered
# unhealthy for sensitive groups 
unhealthy_air <- air_near_fire %>%
  filter(PM25 > 100)  

unhealthy_air %>%
  ggplot() +
  geom_sf(data = CA) +
  geom_sf(color = 'red') +
  geom_sf(fill = 'firebrick',
          color = NA,
          data = camp_fire) +
  geom_sf(color = 'firebrick',
          fill = NA,
          data = camp_zone)
```

![unhealthy](assets/R_unhealthy-1.png)
