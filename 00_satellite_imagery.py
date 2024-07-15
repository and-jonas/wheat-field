# ======================================================================================================================

# Downloads Sentinel-2 optical satellite imagery (10 m bands) from Microsoft Planetary Computer.
# The output is stored at a custom location, to be defined by the user.

# Author: Lukas Valentin Graf, ETH Zürich
# Last edited: 2024-07-15

# ======================================================================================================================

import geopandas as gpd
import json
import warnings

from datetime import datetime
from pathlib import Path
from typing import List

from eodal.config import get_settings
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs


Settings = get_settings()
Settings.USE_STAC = True

# suppress warnings from the COG translate command
warnings.filterwarnings('ignore')


def preprocess_sentinel2_scenes(
        ds: Sentinel2,
) -> Sentinel2:
    """
    Apply the Scene Classification Layer (SCL) to mask clouds, shadows, and
    snow.

    :returns:
        cloud-masked Sentinel-2 scene.
    """
    # resample the 20 m SCL band to 10 m
    ds.resample(inplace=True, target_resolution=10)
    # mask clouds, shadows, and snow
    ds.mask_clouds_and_shadows(inplace=True)
    return ds


def process_field_parcel(
        field_parcel: gpd.GeoDataFrame,
        output_dir: Path,
        scene_kwargs: dict,
        metadata_filters: List[Filter],
        collection: str,
        time_start: datetime,
        time_end: datetime,
) -> None:
    """
    Extract Sentinel-2 scenes for a field parcel. Store the scenes
    as individual GeoTiffs along the scene-specific viewing and illumination
    angles.

    :param field_parcel:
        GeoDataFrame with the field parcel geometry.
    :param output_dir:
        Path to the output directory.
    :param scene_kwargs:
        Dictionary with the scene processing options.
    :param metadata_filters:
        List of metadata filters to apply.
    :param collection:
        Collection to query.
    :param time_start:
        Start of the time range.
    :param time_end:
        End of the time range.
    """
    # convert the GeoDataFrame to a Feature
    feature = Feature.from_geoseries(field_parcel.geometry)

    # query the scenes available (no I/O of scenes, this only fetches metadata)
    mapper_configs = MapperConfigs(
        collection=collection,
        time_start=time_start,
        time_end=time_end,
        feature=feature,
        metadata_filters=metadata_filters)

    # to enhance reproducibility and provide proper documentation, the
    # MapperConfigs are saved as yaml (and also then be loaded again
    # from yaml)
    fpath_yaml = output_dir.joinpath(
        f'{collection}_{time_start.date()}-{time_end.date()}_'
        'mapper_configs.yaml'
    )
    mapper_configs.to_yaml(fpath_yaml)

    # create a new Mapper instance
    mapper = Mapper(mapper_configs)
    # query the scenes (fetches metadata)
    mapper.query_scenes()

    # load the scenes available from STAC (actual I/O of scenes)
    mapper.load_scenes(scene_kwargs=scene_kwargs)
    # the data loaded into `mapper.data` as a EOdal SceneCollection
    scoll = mapper.data

    # save scenes as cloud-optimized GeoTiff
    band_selection = ['blue', 'green', 'red', 'nir_1']
    band_str = '-'.join(band_selection)
    for timestamp, scene in scoll:
        # output file path for the imagery
        platform = scene.scene_properties.platform
        fpath = output_dir.joinpath(
            f'{platform}_{timestamp.date()}_{band_str}.tiff'
        )
        scene.to_rasterio(
            fpath,
            band_selection=band_selection,
            as_cog=True
        )

        # save the scene metadata as json
        fpath_metadata = output_dir.joinpath(
            f'{platform}_{timestamp.date()}_metadata.json'
        )
        scene_metadata = mapper.metadata[
            mapper.metadata['sensing_time'] ==
            scene.scene_properties.sensing_time
        ].copy()
        scene_metadata.sensing_time = scene_metadata.sensing_time.astype(str)
        scene_metadata.drop(columns='sensing_date', inplace=True)
        # drop columns that start with '_'
        scene_metadata = scene_metadata.loc[
            :, ~scene_metadata.columns.str.startswith('_')
        ]
        json_str = scene_metadata.to_json()

        with open(fpath_metadata, 'w') as f:
            f.write(json.dumps(json_str, ensure_ascii=False))


if __name__ == '__main__':

    import os
    cwd = Path(__file__).parents[1]
    os.chdir(cwd)

    # user-inputs
    # --------------------------- Collection ----------------------------------
    collection: str = 'sentinel2-msi'

    # --------------------------   Time Range ---------------------------------
    time_start: datetime = datetime(2020, 2, 1)  		# year, month, day (incl.)
    time_end: datetime = datetime(2020, 8, 30)   		# year, month, day (incl.)

    # --------------------------- Field Parcels  ------------------------------
    field_parcel_dir = Path(
        '/Volumes/green_groups_kp_public/Evaluation/Hiwi/2023_herbifly_LTS/extra_info/satellite_shapes'  # noqa E501
    )

    # ------------------------- Output Directory ------------------------------
    output_dir = Path(
        '/Volumes/green_groups_kp_public/Evaluation/Hiwi/2023_herbifly_LTS/sat_data'  # noqa E501
    )

    # ------------------------- Metadata Filters ------------------------------
    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<', 10),     # < 10% cloud cover
        Filter('processing_level', '==', 'Level-2A')    # only L2A products
    ]

    # --------------------   EOdal processing options -------------------------
    scene_kwargs = {
        'scene_constructor': Sentinel2.from_safe,
        'scene_constructor_kwargs': {'band_selection':
                                     ['B02', 'B03', 'B04', 'B08'],  # 10 m bands
                                     'apply_scaling': False,        # no scaling
                                     'read_scl': True},             # read SCL
        'scene_modifier': preprocess_sentinel2_scenes,
        'scene_modifier_kwargs': {}
    }

    # loop over the GeoJSON files in the directory
    for geojson in field_parcel_dir.glob('*.geojson'):
        field_parcel = gpd.read_file(geojson)
        # store the output in a directory named after the field
        output_dir_field = output_dir / geojson.stem.split('_')[0]
        output_dir_field.mkdir(exist_ok=True, parents=True)

        process_field_parcel(
            field_parcel=field_parcel,
            output_dir=output_dir_field,
            scene_kwargs=scene_kwargs,
            metadata_filters=metadata_filters,
            collection=collection,
            time_start=time_start,
            time_end=time_end
        )
