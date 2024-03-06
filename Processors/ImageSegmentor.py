
# ======================================================================================================================

# Segments vegetation from soil in UAV images, using either (i) a random forest and color properties of pixels for
# UAV-images or (ii) resnet34-unet++ followed by a pixels-wise color classification for high-res images.
# Data used to train the model random forest model is available in 'validation'.
# Creates an output directory called 'vegmask' within each flight folder containing the resulting binary masks or
# multiple output directories in the case of (ii); see 'validation'.

# Author: Jonas Anderegg, ETH Zürich
# Last edited: 2024-03-05

# ======================================================================================================================


import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import copy
import pickle
import imageio
import cv2
import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData
from utils.transforms import set_input_transform_options
from multiprocessing import Manager, Process
import SegmentationFunctions


transform = set_input_transform_options(
    train_size=512,
    crop_factor=0.688,
    p_color_jitter=0,
    blur_kernel_size=1,
    predict_size=(4000, 6000),
    predict_scale=0.5
)


class Segmentor:

    def __init__(self, dirs_to_process, dir_patch_coordinates, dir_output, dir_vegetation_model,
                 dir_col_model,
                 overwrite,
                 img_type):
        self.dirs_to_process = dirs_to_process
        self.dir_patch_coordinates = Path(dir_patch_coordinates) if dir_patch_coordinates is not None else None
        self.dir_vegetation_model = dir_vegetation_model
        self.dir_col_model = dir_col_model
        self.overwrite = overwrite
        # output paths
        self.path_output = Path(dir_output)
        self.path_mask = self.path_output / "SegVeg" / "Mask"
        self.path_overlay = self.path_output / "SegVeg" / "Overlay"
        self.path_col_mask = self.path_output / "SegVeg" / "ColMask"
        self.image_type = img_type
        # load the segmentation models
        self.vegetation_model = SemanticSegmentation.load_from_checkpoint(self.dir_vegetation_model)
        with open(self.dir_col_model, 'rb') as model:
            self.col_model = pickle.load(model)
        # instantiate trainer
        # self.trainer = flash.Trainer(max_epochs=1, accelerator='gpu', devices=[0])
        self.trainer = flash.Trainer(max_epochs=1, accelerator='cpu')

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        for path in [self.path_output, self.path_mask, self.path_overlay, self.path_col_mask]:
            path.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """
        # get all files and their paths
        files = []
        for d in self.dirs_to_process:
            files.extend(glob.glob(f'{d}/*.{self.image_type}'))
        # removes all processed images
        if not self.overwrite:
            processed = glob.glob(f'{self.path_col_mask}/*.png')
            existing = [os.path.basename(x).replace(".png", "") for x in processed]
            files_to_proc = [os.path.basename(f).replace(".JPG", "") for f in files]
            idx = [idx for idx, img in enumerate(files_to_proc) if img not in existing]
            files = [files[i] for i in idx]

        return files

    @staticmethod
    def make_overlay(patch, mask, colors=[(1, 0, 0, 0.25)]):
        img_ = Image.fromarray(patch, mode="RGB")
        img_ = img_.convert("RGBA")
        class_labels = np.unique(mask)
        for i, v in enumerate(class_labels[1:]):
            r, g, b, a = colors[i]
            M = np.where(mask == v, 255, 0)
            M = M.ravel()
            M = np.expand_dims(M, -1)
            out_mask = np.dot(M, np.array([[r, g, b, a]]))
            out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
            out_mask = out_mask.astype("uint8")
            M = Image.fromarray(out_mask, mode="RGBA")
            img_.paste(M, (0, 0), M)
        img_ = img_.convert('RGB')
        overlay = np.asarray(img_)

        return overlay

    def segment_image(self, patch, model, transform, colors):
        """
        Segments an image using a pre-trained semantic segmentation model.
        Creates probability maps, binary segmentation masks, and overlay
        :param image: The image to be processed as an numpy array.
        :param coordinates: A tuple of coordinates defining the ROI.
        :return: The resulting binary segmentation mask.
        """

        # image axes must be re-arranged
        patch_ = np.moveaxis(patch, 2, 0) / 255.0

        # create a datamodule from numpy array
        datamodule = SemanticSegmentationData.from_numpy(
            predict_data=[patch_],
            num_classes=2,
            train_transform=transform,
            val_transform=transform,
            test_transform=transform,
            predict_transform=transform,
            batch_size=1,  # required
        )

        # make predictions
        print("starting prediction")
        predictions = self.trainer.predict(
            model=model,
            datamodule=datamodule,
        )

        # extract predictions
        predictions = predictions[0][0]['preds']

        # transform predictions to probabilities and labels
        probabilities = torch.softmax(predictions, dim=0)
        probabilities_ = probabilities[0]
        mask = torch.argmax(probabilities, dim=0)
        mask_8bit = np.uint8((mask*255) / (len(np.unique(mask))-1))

        overlay = self.make_overlay(patch, mask_8bit, colors=colors)

        return probabilities_, np.asarray(mask_8bit), overlay


    def process_images(self):
        """
        Wrapper, processing all images
        """
        self.prepare_workspace()
        files = self.file_feed()

        for file in files:

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.image_type, ".png")
            img = Image.open(file)
            pix = np.array(img)

            # sample patch from image using coordinate file
            if self.dir_patch_coordinates is not None:
                c = pd.read_table(f'{self.dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
                patch = pix[c[2]:c[3], c[0]:c[1]]
            else:
                patch = pix

            # imageio.imwrite(self.path_patch / png_name, patch)

            # (2) segment vegetation ===================================================================================
            proba, pred_mask, overlay = self.segment_image(
                patch,
                model=self.vegetation_model,
                transform=transform,
                colors=[(0, 0, 1, 0.25)]
            )

            # output paths
            mask_name = self.path_mask / png_name
            overlay_name = self.path_overlay / base_name

            # save output
            imageio.imwrite(mask_name, pred_mask)
            imageio.imwrite(overlay_name, overlay)

            # (3) color-based segmentation =============================================================================

            # downscale
            x_new = int(patch.shape[0] * (1 / 2))
            y_new = int(patch.shape[1] * (1 / 2))
            patch_seg = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)

            # extract pixel features
            color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(patch_seg)
            descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

            # get pixel label probabilities
            segmented_flatten_probs = self.col_model.predict(descriptors_flatten)

            # restore image
            preds = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

            # convert to mask
            mask = np.zeros_like(patch_seg)
            mask[np.where(preds == "brown")] = (102, 61, 20)
            mask[np.where(preds == "yellow")] = (255, 204, 0)
            mask[np.where(preds == "green")] = (0, 100, 0)

            # upscale
            x_new = int(patch_seg.shape[0] * (2))
            y_new = int(patch_seg.shape[1] * (2))
            mask = cv2.resize(mask, (y_new, x_new), interpolation=cv2.INTER_NEAREST)

            # remove background
            col_mask_name = self.path_col_mask / png_name
            col_mask = copy.copy(mask)
            col_mask[np.where(pred_mask == 0)] = (0, 0, 0)
            imageio.imwrite(col_mask_name, col_mask)


class ImageColorSegmentor:

    def __init__(self, dir_to_process, dir_model, img_type, overwrite, n_cpus):
        self.dir_to_process = Path(dir_to_process)
        # self.directdir_to_process = Path(directdir_to_process)
        self.dir_model = Path(dir_model)
        # output paths
        self.image_type = img_type
        self.overwrite = overwrite
        # settings
        self.n_cpus = n_cpus
        # load model
        with open(self.dir_model, 'rb') as model:
            self.model = pickle.load(model)
        self.model.n_cpus = 12

    def prepare_workspace(self):
        """
        Creates all required output directories
        """

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """
        # get all files and their paths
        base_dir = self.dir_to_process
        dirs = glob.glob(f'{base_dir}/*/*/*0m/2020*')

        # get all files and their paths
        files = []
        for d in dirs:
            files.extend(glob.glob(f'{d}/*.{self.image_type}'))

        return files

    def segment_image(self, img, model):
        """
        Segments an image using a pre-trained pixel classification model.
        Creates probability maps, binary segmentation masks, and overlay
        :param veg_mask: vegetation mask (ground truth or flash predictions)
        :param img: The image to be processed.
        :return: The resulting binary segmentation mask.
        """

        # extract pixel features
        color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(img)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])
        # descriptors_flatten = descriptors_flatten[:, [3, 4, 7, 10, 14, 16, 18, 20]]

        # extract pixel label probabilities
        model.n_jobs = 1  # disable parallel; parallelize over images instead

        segmented_flatten_probs = model.predict(descriptors_flatten)

        # restore image
        mask = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))
        mask = (mask*255).astype("uint8")

        return mask

    def process_image(self, work_queue, result):
        """
        Wrapper, processing all images
        :param img_type: a character string, the file extension, e.g. "JPG"
        """
        for job in iter(work_queue.get, 'STOP'):

            file = job['file']

            # read image
            base_name = os.path.basename(file)
            png_name = base_name.replace("." + self.image_type, ".png")

            img = Image.open(file)
            img = np.array(img)

            # output paths
            current_dir = Path(os.path.dirname(file))
            mask_dir = current_dir / "vegmask"
            if not mask_dir.exists():
                mask_dir.mkdir(exist_ok=True, parents=True)
            mask_name = mask_dir / png_name

            # segment image
            mask = self.segment_image(img=img, model=self.model)

            # save mask
            imageio.imwrite(mask_name, mask)

            result.put(file)

    def process_images(self):

        self.prepare_workspace()
        files = self.file_feed()

        if len(files) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(files)
            count = 0

            # Build up job queue
            for file in files:
                print(file, "to queue")
                job = dict()
                job['file'] = file
                jobs.put(job)

            # Start processes
            for w in range(self.n_cpus):
                p = Process(target=self.process_image,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(files)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()
