---
title: "Metashape Photogrammetry Tutorial"
layout: single
header:
  overlay_color: "444444"
  overlay_image: /assets/images/nasa.png
---



## Introduction


### Requirements

* Python 3.5, 3.6, 3.7, or 3.8
* gcc/10.2.0
* metashape/1.7.3
* mesa/20.1.6

`module load <package>`


## Agisoft Metashape Tutorial

### Step 0: Data Collecting

#### Camera settings
Any high resolution digital camera (> 5 MPix) can be used for capturing images suitable for 3D model reconstruction in Metashape software. It is suggested to use focal length from 20 to 80 mm interval in 35mm equivalent while avoiding ultra-wide angle and fisheye lenses. Fixed lenses are preferred for more stable results.

#### Images settings
Take sharp photos at maximal possible resolution with sufficient focal depth and the lowest value of ISO. For the Metashape analysis use RAW data. The lossless conversion to the TIFF format is preferred over JPG, which induce more noise. Also, do not use any pre-processing (resize, rotate, crop, etc.) on your photos.

#### Optimal image sets
In general, a good set of images is not random. More than required number of photos is better than not enough but redundant or highly overlap pictures are not useful. However, the detail of the geometry should be visible from at least two different camera snapshots. To learn more tips & tricks see the *Capturing scenarios* and *Plan Mission* sections in the Metashape [user manual](https://www.agisoft.com/pdf/metashape-pro_1_5_en.pdf), pp 9-14.

### Step 1: Loading & Inspecting Photos

#### Creating application object
```
import Metashape as MS

doc = MS.app.document
```

#### CPU/GPU settings
> Metashape exploits GPU processing power that speeds up the process significantly.
If you have decided to switch on GPUs to boost the data processing with Metashape, it is recommended to uncheck "Use CPU when performing GPU accelerated processing" option, providing that at least one discrete GPU is utilized for processing. *(Preference settings in [Metashape User Manual](https://www.agisoft.com/pdf/metashape-pro_1_5_en.pdf), pp. 15)*

```
MS.app.gpu_mask = 2 ** (len(MS.app.enumGPUDevices()) - 1)         # activate all available GPUs
  if MS.app.gpu_mask <= 1:                                        # (faster with 1 no difference with 0 GPUs)
      MS.app.cpu_enable = True                                    # enable CPU for GPU accelerated processing
  elif MS.app.gpu_mask > 1:                                       # (faster when multiple GPUs are present)
      MS.app.cpu_enable = False                                   # disable CPU for GPU accelerated tasks
```

#### Loading images

```
datadir = "/absolute/path/to/your/input/directory/with/photos"    # directory with image inputs
photo_files = os.listdir(datadir)                                 # list of filenames for photo set
photos = [os.path.join(datadir, p) for p in photo_files]          # convert to full paths

```

### Step 2: Generating Sparse Point Cloud (SPC)


### Step 3: Generating Dense Point Cloud (DPC)


### Step 4: Generating of a Surface: Mesh or DEM
