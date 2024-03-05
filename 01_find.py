
# ======================================================================================================================

# Finds UAV images with the most-nadir perspective of a pre-defined position on a field

# The example below illustrates an example workflow for the identification of the image providing the most-nadir view
# of reference areas marked by rectangular frames, as well as the corresponding binary soil - vegetation mask.

# The UAV images are identified by loading the geojson files with containing the real-world coordinates of the reference
# areas, and matching them with the estimated real-world image corner coordinates, which are also available from the
# metadata for each field and each flight.

# In the below example, the identified images are copied to the validation folder.
# The identified images form the basis of the reported technical validation procedure.

# Authors: Jonas Anderegg, Flavian Tschurr; ETH Zürich.
# Last edited: 2024-03-05

# ======================================================================================================================

import os
import geojson
from utils import base as utils
from utils import FrameFunctions
from pathlib import Path
import pandas as pd
import shutil
import re

# ======================================================================================================================

# base directory
workdir = "O:/Evaluation/Hiwi/2023_herbifly_LTS"

# field names
fields = ["Nennigkofen1", "Nennigkofen2", "Grafenried", "Steinmaur", "Treytorrens-P", "Volken", "Opfertshofen",
          "Villars-le-G", "Oleyres"]
# flight height
fh = "50m"

# iterate over fields
for field in fields:

    # find shapefiles containing the reference area coordinates
    path_geojsons_folder = path_trainings = os.path.join(workdir, "Meta", field, fh, "frames")
    dates = os.listdir(path_geojsons_folder)

    # exclude late flights
    dates = [x for x in dates if "202007" not in x]
    dates = [x for x in dates if "202008" not in x]

    # iterate over measurement dates
    for date in dates:

        # load the image real world corner coordinates table
        try:
            cornersDF = pd.read_csv(f"{workdir}/Meta/{field}/{fh}/corners/{field}_{date}_{fh}_CameraCornerCoordinates.csv",
                                    index_col=0)
        except FileNotFoundError:
            print("FILE NOT FOUND!")
            continue

        # iterate over geojsons
        path_geojsons_date = os.path.join(path_geojsons_folder, date)
        geojsons = os.listdir(path_geojsons_date)
        for geoj in geojsons:
            if utils._check_geojson(geoj):
                path_geojson_current = os.path.join(path_geojsons_date)
                with open("{path_geo}/{pic_n}".format(path_geo=path_geojson_current, pic_n=geoj),
                          'r') as infile:
                    polygon_mask = geojson.load(infile)
                    polygons = polygon_mask["features"]
                    # iterate over the polygons within the geojson file
                for polygon in polygons:
                    frame_coverage = []
                    if len(polygon["geometry"]["coordinates"][0]) == 5:
                        coords = polygon["geometry"]["coordinates"][0]
                    else:
                        coords = polygon["geometry"]["coordinates"][0][0]
                    frame_label = polygon["properties"]["plot_label"]
                    print(frame_label)
                    # find the images of interest
                    img_name = FrameFunctions.image_finder(cornersDF, polygon_coords=coords)
                    print(img_name)

                    # copy and rename the identified image
                    pattern = r'[/\\]'
                    p = re.split(pattern, path_geojson_current)
                    date = p[-1]
                    fh = p[-3]

                    # make output dir
                    output_dir_images = Path(f"{workdir}/validation/{fh}/images_frames")
                    output_dir_masks = Path(f"{workdir}/validation/{fh}/masks_frames")
                    if not output_dir_images.exists():
                        output_dir_images.mkdir(exist_ok=True, parents=True)
                    if not output_dir_masks.exists():
                        output_dir_masks.mkdir(exist_ok=True, parents=True)

                    from_img_name = f'{workdir}/raw/{field}/{fh}/{date}/{img_name}.JPG'
                    to_img_name = f"{workdir}/validation/{fh}/images_frames/{date}_{frame_label}.JPG"
                    from_mask_name = f'{workdir}/raw/{field}/{fh}/{date}/vegmask/{img_name}.png'
                    to_mask_name = f"{workdir}/validation/{fh}/masks_frames/{date}_{frame_label}.png"

                    try:
                        shutil.copy(from_img_name, to_img_name)
                        shutil.copy(from_mask_name, to_mask_name)
                    except FileNotFoundError:
                        date = date
                        print("IMAGE NOT FOUND!")
                        continue

