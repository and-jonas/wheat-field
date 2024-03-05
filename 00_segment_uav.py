
# ======================================================================================================================

# Segments vegetation from soil in UAV images, based on color properties of pixels
# Data used to train the model is available in 'validation'.
# Creates an output directory called 'vegmask' within each flight folder containing the resulting binary masks

# Author: Jonas Anderegg, ETH Zürich
# Last edited: 2024-03-05

# ======================================================================================================================


from Processors.ImageSegmentor import ImageColorSegmentor

base_dir = '/home/anjonas/kp-public/Evaluation/Hiwi/2023_herbifly_LTS'


def run():
    dir_to_process = base_dir
    dir_model = f"{base_dir}/validation/models/rf_segmentation.pkl"
    image_color_segmentor = ImageColorSegmentor(
        dir_to_process=dir_to_process,
        dir_model=dir_model,
        overwrite=False,
        img_type="JPG",
        n_cpus=1,
    )
    image_color_segmentor.process_images()


if __name__ == "__main__":
    run()
