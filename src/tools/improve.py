from src import *
from src.tools.predict import patch_to_label, masks_to_submission, masks_to_submission

import os, os.path
import re

foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


def improve_test_set(model, path='data/test_set_images/'):
    path = PROJECT + path
    for i in range(1, 51):
        folder_path = path + 'test_' + str(i) + '/'
        image_path = folder_path + 'pred_' + str(i) + '.png'
        img = misc.imread(image_path)
        subm = model.predict(img)
        subm_path = folder_path + 'subm_' + str(i) + '.png'
        misc.imsave(subm_path, subm)
    print("DONE")


def submit(submission_filename='data/dummy_submission.csv', pred_path='data/test_set_images/'):
    submission_filename = PROJECT + submission_filename
    image_filenames = []
    pred_path = PROJECT + pred_path
    for i in range(1, 51):
        image_filename = pred_path + 'test_' + str(i) + '/subm_' + str(i) + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
    print("DONE")
