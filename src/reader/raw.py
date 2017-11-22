import numpy as np
from scipy import misc
import os
import cv2

DATA = '../data/training/'
IMAGES = DATA + 'images/'
GT = DATA + 'groundtruth/'
FORM = 'satImage_'

def load_data():
    """ Loads the data and returns two arrays:
    One
    """
    count = 0
    imgs = np.empty([100, 400, 400, 3])
    gts = np.empty([100, 400*400, 2])
    for path in os.listdir(IMAGES):
        if path.startswith(FORM):
            img = misc.imread(IMAGES + path, mode='RGB')
            imgs[count] = normalizeHist(img)
            gt = misc.imread(GT + path) 
            gts[count] = reshapeGT(polarize(gt))
            count += 1
            if count > 100:
                print('something is wrong with the images')
                break
    return imgs, gts

def polarize(img):
    """ Polarize a ground truth image as (0., 1.).
    Returns two channels, one for road probability, one for background.
    """

    polarized = np.zeros([400, 400])
    polarized_inv = np.ones([400,400])
    polarized[np.where(img > 127)] = 1.
    polarized_inv[np.where(img > 127)] = 0.

    return np.stack([polarized, polarized_inv], axis=-1)

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
