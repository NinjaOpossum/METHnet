''' Dataset for saved tiles '''
import os
import pickle

import numpy as np
import cv2
from progress.bar import Bar

from saved_tile_dataset import SavedTilesDataset
from mothi.tiling_projects import QuPathTilingProject
from paquo.projects import QuPathProject

# config
QUPATH_DIR = os.path.join("/workspaces", "METHnet_GBM_segmentation", "MountData", "Mothi-Testproject")
DATA_DIR = 'data'
SAVE_DIR = os.path.join(DATA_DIR, 'export')
# QUPATH_IMAGE_DIR = '/workspaces/GBM_QuPath_tiles/test/test_projects/slides'
EXPORT_IMG_IDS = [0] # numpy indexing

TILE_SIZE = (250, 250) # (width, height)

# Load the QuPath project
qp_project_test = QuPathProject(QUPATH_DIR, mode='r+')
# Update image paths
qp_project_test.update_image_paths(try_relative=True)


# extract tiles
qp_project = QuPathTilingProject(QUPATH_DIR)
img_meta = np.array(qp_project.images)[EXPORT_IMG_IDS]
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

for qupath_img in img_meta:
    img_id = int(qupath_img.entry_id) - 1
    # name to save tiles with
    save_tile_img_name = qupath_img.image_name.split('.')[0]
    # create subdirectory for each tiled image
    save_tiles_dir = os.path.join(SAVE_DIR, save_tile_img_name)
    if not os.path.exists(save_tiles_dir):
        os.mkdir(save_tiles_dir)
        os.mkdir(os.path.join(save_tiles_dir, 'labels'))

    ### check for existing exports
    if len(os.listdir(save_tiles_dir)) > 1:
        print(save_tile_img_name + ' skipped: already exported')
        continue
    ###

    print(save_tile_img_name + ': start export')
    y_steps = int(qupath_img.height / TILE_SIZE[1])
    with Bar('tile export: ' + save_tile_img_name, max = y_steps, suffix = '%(percent)d%%') as bar:
        for y_step in range(y_steps):
            location_y = y_step * TILE_SIZE[1]

            for x_step in range(int(qupath_img.width / TILE_SIZE[0])):
                location_x = x_step * TILE_SIZE[0]
                tilename = f'{save_tile_img_name}[x={location_x},y={location_y},size={TILE_SIZE}].tif'
                tile_path_name = os.path.join(save_tiles_dir, tilename)
                tile_mask_path_name = os.path.join(save_tiles_dir, 'labels',
                                                tilename.split('.')[0] + '_label.tif')

                tile = qp_project.get_tile(img_id, (location_x, location_y), TILE_SIZE,
                                           ret_array=True)
                tile_mask = qp_project.get_tile_annot_mask(img_id, (location_x, location_y),
                                                           TILE_SIZE,)

                # write tile
                cv2.imwrite(tile_path_name, tile)
                cv2.imwrite(tile_mask_path_name, tile_mask)
            bar.next()


# create and save dataset
dataset = SavedTilesDataset([os.path.join(SAVE_DIR, img) for img in os.listdir(SAVE_DIR)])
dataset_dir = os.path.join(DATA_DIR, 'dataset')
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
with open(os.path.join(dataset_dir, 'dataset.pkl'), 'wb') as dump_file:
    pickle.dump(dataset, dump_file)
