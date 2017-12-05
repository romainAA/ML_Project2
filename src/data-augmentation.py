import numpy as np
import keras
import os
from PIL import Image

def getAllFileNames(path, num):
    imageNames = []
    grTruthNames = []
    for i in range(1,num+1):
        imageNames.append(path + 'images/satImage_%03d.png'%i)
        grTruthNames.append(path + 'groundtruth/satImage_%03d.png'%i)
    return imageNames, grTruthNames


def getAllImages(imageNames, grNames):
    images = []
    grTruths= []
    print(imageNames, grNames)
    for imageName, grName in zip(imageNames, grNames):
        images.append(Image.open(imageName))
        grTruths.append(Image.open(grName))

    print(len(images))
    return images, grTruths

def augmentAllImages(images, grTruths, factor):
    resIm = []
    resGr = []
    for image, grTruth in zip(images, grTruths):
        ims, grs = augmentImage(image, grTruth, factor)
        resIm.extend(ims)
        resGr.extend(grs)
    return resIm, resGr

def augmentImage(image, grTruth, factor):
    resIm = [image]
    resGr = [grTruth]
    for i in range(factor - 1):
        transImage, transGrTruth = transformImage(image, grTruth)
        resIm.append(transImage)
        resGr.append(transGrTruth)
    return resIm, resGr

def transformImage(image, grTruth):
    degreeRotation = np.random.random_integers(-45,45)
    min_factor = 0.7
    scaling_factor = min_factor + (np.random.random_sample())*(1 - min_factor)
    trans = lambda x : scaling(rotateImage(x, degreeRotation), scaling_factor)
    newImage = trans(image)
    newGrTruth = trans(grTruth)
    return newImage, newGrTruth

def scaling(image, factor):
    size = nearestEven(factor*400)
    diff = (400 - size)/2
    new = image.crop((diff, diff, 400 - diff, 400 -diff))
    new = new.resize((400, 400))
    return new

def nearestEven(i):
    return round(i*0.5)*2


def rotateImage(image, angle):
    new = image.rotate(angle)
    new = new.crop((60,60,340,340))
    new = new.resize((400,400))
    return  new

def saveImages(path, newIms, newGrTruths, factor):
    print(len(newIms))
    for i, (image, grTruth) in enumerate(zip(newIms,newGrTruths)):
        index = i/factor + 1
        j_index = (i % factor) + 1
        imPath = path + 'images/training_'
        grPath = path + 'groundtruth/groundtruth_'
        name = '%03d'%index + '_%03d'%j_index + '.png'
        print(imPath + name)
        image.save(imPath + name)
        grTruth.save(grPath + name)




#parameters
trainingPath = '../Data/training/'
newFolder = '../Data/augmented-training/'
numberOfImages = 100
factor = 3

if __name__ == '__main__':
    imNames, grNames = getAllFileNames(trainingPath, numberOfImages)
    images, grTruths = getAllImages(imNames,grNames)
    newIms, newGrTruths = augmentAllImages(images, grTruths, factor)
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
        os.makedirs(newFolder+'images/')
        os.makedirs(newFolder+'groundtruth/')
    saveImages(newFolder, newIms, newGrTruths, factor)
