# Tutorials

Can place jupyter notebook files here. Use the `jupyter_to_md.sh` script to convert the notebooks into Markdown files for the workbook.

```{bash}
bash jupyter_to_md.sh Tutorial1_Image_Processing_Essentials.ipynb
# Will generate Tutorial1_Image_Processing_Essentials.md
```

Remember to double check the following repos for updates. Notebook files are probably in the `tutorials` folder, also double check the repo branches.

* [kerriegeil/SCINET-GEOSPATIAL-RESEARCH-WG](https://github.com/kerriegeil/SCINET-GEOSPATIAL-RESEARCH-WG)

* [kerriegeil/NMSU-USDA-ARS-AI-Workshops](https://github.com/kerriegeil/NMSU-USDA-ARS-AI-Workshops)



## Running or editing tutorials locally



```bash
cd tutorial         # folder containing jupyter notebooks
# optional, start your conda environment which has the jupyter notebook package
jupyter notebook    # Opens a browser window where you can open, edit and run a notebook file
```



## Converting notebooks to python scripts

For example, Tutorial 1 loads python libraries, imports a file, and shows it in a plot. The main change is the fact that we are not showing in a plot but saving to an image. Therefore we can create a python script like the following:

```python
#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 
#%matplotlib inline                  #<== notice how this needs to be commented out
import imageio
import skimage.color
import skimage.transform
import scipy.ndimage as ndimage

I_camera = np.asarray(imageio.imread('data/cameraman.png'))
plt.figure()                      # open a new figure window
plt.imshow(I_camera, cmap='gray') # visualize the I_camera image with a grayscale colormap
# plt.show()                        # show the plot   #<== comment out
plt.savefig('cameraman_grayscale.png')   #<== Add exported file name
```

Then this script **tutorial1.py** can be called from command-line.

```bash
python tutorial1.py
ls cameraman_grayscale.png

#> -rw-r--r--@ 1 jenchang  staff   131K Nov 10 11:36 cameraman_grayscale.png
```

This script should **add argument handling** (take a different png image), and probably be renamed to **grayscale-ify.py**, so you can reuse it.