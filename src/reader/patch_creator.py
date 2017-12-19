import os, sys

if '.' not in sys.path:
    sys.path.append('.')
from src import *

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras

foreground_threshold = .25


def read_images(root_dir="data/training/"):
    """
    Read the images
    :param root_dir: the path to the directory containing the two subdirectories images/ and groundtruth/
    :return: number of images loaded, and two lists, images and ground truth. Values are between 0 and 1
    """

    root_dir = PROJECT + root_dir
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    files = list(filter(lambda s: not s.startswith('.'), files))
    n = len(files)
    print("Loading " + str(n) + " images")
    imgs = [mpimg.imread(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = [mpimg.imread(gt_dir + files[i]) for i in range(n)]

    return n, imgs, gt_imgs


def img_crop(im, w, h):
    """Create patches of size w*h from an image"""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h // 2):
        if i + h <= imgheight:
            for j in range(0, imgwidth, w // 2):
                if j + w <= imgwidth:
                    if is_2d:
                        im_patch = im[j:j + w, i:i + h]
                    else:
                        im_patch = im[j:j + w, i:i + h, :]
                    # if im_patch.shape[0] == h and im_patch.shape[1] == w:
                    list_patches.append(im_patch)
    return list_patches


def create_patches(imgs, gt_imgs, patch_size=48):
    """

    :param imgs: list of images
    :param gt_imgs: list of ground truth images
    :param patch_size: width and height of patches
    :return: two arrays of patches
    """

    # Extract patches from input images
    n = len(imgs)

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    return img_patches, gt_patches


def labelize_patch(patch):
    """
    Decide if a patch is a road or not
    :param patch: a ground truth patch
    :return: either 0 for background or 1 for road
    """

    patch_h, patch_w = patch.shape
    mid_h, mid_w = patch_h // 2, patch_w // 2
    sm_patch = patch[mid_h - 8:mid_h + 8, mid_w - 8:mid_w + 8]

    df = np.mean(sm_patch)

    return 1 if df > foreground_threshold else 0


def labelize_patches(gt_patches):
    """
    Labelize all patches in an array
    :param gt_patches: array of patches
    :return: array of labels
    """

    return np.asarray([labelize_patch(gt_patches[i]) for i in range(len(gt_patches))])


def save_to_file(imgs, gts, labels, save_file='data/patches_48/data.npz'):
    """
    Saves the images, ground truths and their labels to a npz file
    :param imgs: array of images
    :param gts: array of ground truths
    :param labels: array of labels
    :param save_file: file location
    """

    outfile = open(save_file, 'wb')
    np.savez_compressed(outfile, imgs=imgs, gt_imgs=gts, labels=labels)


def load_from_file(load_file='data/patches_48/data.npz'):
    """ Load the images, ground truths and labels from the given file.
    File must contain the arrays with labels:
      - imgs
      - gt_imgs
      - labels"""

    infile = open(load_file, 'rb')
    npzfile = np.load(infile)
    return npzfile['imgs'], npzfile['gt_imgs'], npzfile['labels']


if __name__ == '__main__':
    n, imgs, gts = read_images(root_dir="data/augmented-training/")
    img_patches, gt_patches = create_patches(imgs, gts, patch_size=80)
    gt_labels = keras.utils.to_categorical(labelize_patches(gt_patches), 2)

    save_to_file(img_patches, gt_patches, gt_labels, save_file='data/patches_80/aug-data.npz')
