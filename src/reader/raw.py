import numpy as np
from scipy import misc
import os, os.path
import cv2

DATA = '../data/training/'
IMAGES = DATA + 'images/'
GT = DATA + 'groundtruth/'
FORM = 'satImage_'

def load_data(path=DATA, width=400, height=400):
    """ Loads the data and returns a count and two arrays:

    :param path, the path to where the data is contained, the folder should contain two subfolders: grountruth and images
    :param width of the images
    :param height of the images

    :returns the number of loaded images, and the two arrays
    """

    if path[-1] != '/':
        path += '/'
    filenames = os.listdir(path + 'images/')
    filenames = list(filter(lambda s: not s.startswith('.'), filenames))

    size = len(filenames)

    imgs = np.empty([size, height, width, 3])
    gts = np.empty([size, height, width, 1])

    count = 0
    for filename in filenames:
        img = misc.imread(path + 'images/' + filename, mode='RGB')
        imgs[count] = normalize(img)

        gt_name = path + 'groundtruth/' + filename
        if os.path.exists(gt_name):
            gt = misc.imread(gt_name)
            gts[count] = np.reshape(polarize(normalize(gt)), [height, width, 1])

        count += 1
    return count, imgs, gts

def normalize(img):
    """ Normalize and image to the range [0., 1.]

    :param img: an np array containing an image
    :return: an np containing the normalized image
    """

    top = np.max(img)
    bot = np.min(img)

    return (img - bot) / (top - bot)

def polarize(img):
    """ Polarize a ground truth image as (0., 1.).
    """

    top = np.max(img)
    bot = np.min(img)

    mid = (top - bot) / 2

    polarized = np.zeros(img.shape)
    polarized[np.where(img > mid)] = 1.

    return polarized

def reshapeGT(polarized):
    """Reshapes the polarized GT image to a two channels vector"""
    return np.reshape(polarized, (polarized.shape[0] * polarized.shape[1], 2))

def normalizeHist(img):
    """Normalize the color histograms of the image,
    in order to increase the contrast.
    Values are floats between 0. and 1."""
    norm=np.zeros((img.shape[0], img.shape[1], 3),np.float32)

    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(r)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(b)

    return norm / 255
