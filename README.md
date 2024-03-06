# wheat-field


This repository contains code that can serve as a starting point for further analysis of the Wheat field UAV data set. 
The code was also used for the reported technical validation and enables reproducing corresponding results. 

<img src="https://github.com/and-jonas/wheat-field/blob/master/imgs/fields_orthos.png" width="400"> <img src="https://github.com/and-jonas/wheat-field/blob/master/imgs/composite.jpg" width="500">


## Installation
All the necessary libraries and their dependencies can be installed with `conda env create -f environment.yml`. This creates a new conda environment named wheat-field.

## Data
Data sets are available via the ETH Zürich publications and research data repository:

The data set encompasses raw imagery, products from photogrammetric processing, and rich meta data characterizing the flights and the fields. 

## Models
The trained segmentation models can be downloaded from the same link. They are located in `./validation/models`

## Content

1. `00_segment_handheld.py` segments handheld images
2. `00_segment_uav.py` segments UAV images
3. `01_find.py` fetches images for given positions on each field. The implemented example fetches images giving the most-nadir view of reference areas by comparing real-world corner coordinates of the reference areas and estimated real-world image corner coordinates.
4. `02_align.py` aligns handheld images and UAV images (image co-registration) using the respective image coordinates for the corner of the reference areas. 

These scripts use functionalities from `Processors`, and `utils`.

The user has to adjust the root directory according to their setup. 
