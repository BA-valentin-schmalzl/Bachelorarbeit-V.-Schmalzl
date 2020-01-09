"""Ein Großteil des folgenden Codes basiert auf der Mask R-CNN Implementierung von Matterport Inc.
(siehe: https://github.com/matterport/Mask_RCNN)"""
"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
"""
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import skimage
import os
from skimage import io, draw
import json
from mrcnn import utils
from keras.preprocessing.image import img_to_array, load_img
from mrcnn.visualize import display_instances



class CustomDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        # Add classes. We have only one class to add.
        self.add_class("B_Schaden", 1, "Schaden")

        # Train or validation dataset?


        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations1 = json.load(open(os.path.join(dataset_dir, "Beton_schaden.json")))

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        via_1_check = annotations1.get('regions')
        via_2_check = annotations1.get('_via_img_metadata')

        # JSON is formatted with VIA-1.x
        if via_1_check:
            annotations = list(annotations1.values())
        # JSON is formatted with VIA-2.x
        elif via_2_check:
            annotations = list(annotations1['_via_img_metadata'].values())
        # Unknown JSON formatting
        else:
            raise ValueError('The JSON provided is not in a recognised via-1.x or via-2.x format.')

        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "B_Schaden",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "B_Schaden":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "B_Schaden":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "schaden_cfg"
    # number of classes (background + schaden)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85


test_set = CustomDataset()
test_set.load_dataset('Pfad zu dem Ordner mit den gewünschten Bildern')
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model_path = 'mask_rcnn_schaden_0010.h5'
model.load_weights(model_path, by_name=True)
img = load_img('Bildname')
img = img_to_array(img)
results = model.detect([img], verbose=0)
r = results[0]
print(r['masks'])
class_names = 'Schaden'
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], show_mask=True, show_bbox=False)