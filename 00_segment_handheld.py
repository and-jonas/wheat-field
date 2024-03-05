
# ======================================================================================================================

# Segments vegetation from soil in handheld images, based on a resnet34-unet++
# Creates output directories, see 'validation'
# this script was used to generate the output for the reported technical validation

# Author: Jonas Anderegg, ETH Zürich
# Last edited: 2024-03-05

# ======================================================================================================================

from Processors.ImageSegmentor import Segmentor

base_dir = "/home/anjonas/kp-public/Evaluation/validation/"


def run():
    dirs_to_process = base_dir
    dir_output = "/home/anjonas/kp-public/Evaluation/Hiwi/2023_herbifly_LTS/validation/seg"
    dir_vegetation_model = "vegAnn_herbifly.pt"
    dir_col_model = "segcol_rf.pkl"
    dir_patch_coordinates = None
    image_pre_segmentor = Segmentor(dirs_to_process=dirs_to_process,
                                    dir_vegetation_model=dir_vegetation_model,
                                    dir_col_model=dir_col_model,
                                    dir_patch_coordinates=dir_patch_coordinates,
                                    dir_output=dir_output,
                                    overwrite=False,
                                    img_type="JPG")
    image_pre_segmentor.process_images()


if __name__ == "__main__":
    run()
