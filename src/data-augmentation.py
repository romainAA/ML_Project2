import numpy as np
import os
from PIL import Image
from src import *

#make artificial dataset augmentation

def getAllFileNames(path, num):
    '''just list all the dataset path'''
    imageNames = []
    grTruthNames = []
    for i in range(1,num+1):
        imageNames.append(path + 'images/satImage_%03d.png'%i)
        grTruthNames.append(path + 'groundtruth/satImage_%03d.png'%i)
    return imageNames, grTruthNames


def getAllImages(imageNames, grNames):
    '''open all images as PIL Image objects'''
    images = []
    grTruths= []
    for imageName, grName in zip(imageNames, grNames):
        images.append(Image.open(imageName))
        grTruths.append(Image.open(grName))
    return images, grTruths

def augmentAllImages(images, grTruths, factor):
    '''for all images, create factor - 1 new images from the original'''
    resIm = []
    resGr = []
    for image, grTruth in zip(images, grTruths):
        ims, grs = augmentImage(image, grTruth, factor)
        resIm.extend(ims)
        resGr.extend(grs)
    return resIm, resGr

def augmentImage(image, grTruth, factor):
    '''create factor - 1 new images from the original by random transformation,
    apply the same transformation to the groundtruth'''
    resIm = [image]
    resGr = [grTruth]
    for i in range(factor - 1):
        transImage, transGrTruth = transformImage(image, grTruth)
        resIm.append(transImage)
        resGr.append(transGrTruth)
    return resIm, resGr

def transformImage(image, grTruth):
    '''take an image and create a new one by applying a random transformation,
    then apply the same transformation to the groundtruth '''
    degreeRotation = np.random.random_integers(-45,45)
    flip = np.random.randint(0,3,1)
    min_factor = 0.8
    scaling_factor = min_factor + (np.random.random_sample())*(1 - min_factor)
    trans = lambda x : scaling(flipImage(rotateImage(x, degreeRotation),flip), scaling_factor)
    newImage = trans(image)
    newGrTruth = trans(grTruth)
    return newImage, newGrTruth

def scaling(image, factor):
    ''' zoom in the photo, by a factor (between 0 and 1, 1 being the same image) '''
    size = nearestEven(factor*400)
    diff = (400 - size)/2
    new = image.crop((diff, diff, 400 - diff, 400 -diff))
    new = new.resize((400, 400), Image.BILINEAR)
    return new

def nearestEven(i):
    ''' just to get nice numbers '''
    return round(i*0.5)*2


def rotateImage(image, angle):
    ''' make a rotation of angle (in degrees), and crop so that we don't see the borders'''
    new = image.rotate(angle)
    # new = new.crop((60,60,340,340))
    # new = new.resize((400,400),Image.BILINEAR)
    return  new

def flipImage(image, flip):
    if flip == 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == 2:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def saveImages(path, newIms, newGrTruths, factor):
    '''save the new images in a folder'''
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'images/')
        os.makedirs(path+'groundtruth/')
    for i, (image, grTruth) in enumerate(zip(newIms,newGrTruths)):
        index = i/factor + 1
        j_index = (i % factor) + 1
        imPath = path + 'images/training_'
        grPath = path + 'groundtruth/groundtruth_'
        name = '%03d'%index + '_%03d'%j_index + '.png'
        image.save(imPath + name)
        grTruth.save(grPath + name)




#parameters
trainingPath = PROJECT + '/data/training/'
newFolder = PROJECT + '/data/augmented-training/'
numberOfImages = 100
factor = 3
seed = 57

if __name__ == '__main__':
    np.random.seed(seed)
    imNames, grNames = getAllFileNames(trainingPath, numberOfImages)
    images, grTruths = getAllImages(imNames,grNames)
    newIms, newGrTruths = augmentAllImages(images, grTruths, factor)
    saveImages(newFolder, newIms, newGrTruths, factor)
