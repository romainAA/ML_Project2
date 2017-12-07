import keras
import os
import glob
import numpy as np
from scipy import misc
from segnet import SegNet, preprocess_input, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator$
from src import *


def loadTrain(path, num, img_h, img_w):
    '''load all images in the traing data set and their groundtruth '''
    X = loadImages(path + 'images/', num, img_h, img_w)
    y = loadGroundTruth(path + 'groundtruth/', num, img_h, img_w)
    return X, y


def loadImages(path, num, img_h, img_w):
    '''load all images in the data set'''
    X = np.zeros((num, img_h, img_w, num_channels))
    for i, image_path in enumerate(glob.glob(path + "*.png")):
        if i >= num:
            break
        image = misc.imread(image_path)
        X[i] = image
    return X


def loadGroundTruth(path, num, img_h, img_w):
    '''load all groundtruth in the dataset'''
    Y = np.zeros((num, img_h, img_w))
    for i, image_path in enumerate(glob.glob(path + "*.png")):

        if i >= num:
            break
        image = misc.imread(image_path)
        Y[i] = toClassTab(image, img_h, img_w)
    return Y


def toClassTab(image, img_h, img_w):
    '''convert groundtruth to class [0,1]'''
    res = np.zeros((img_h, img_w))
    for i in range(img_h):
        for j in range(img_w):
            if image[i, j] >= 128:
                res[i, j] = 1  # not a road
            else:
                res[i, j] = 0  # road
    return res


def prepareInput(path, num_training, img_h, img_w, nb_classes):
    '''load and preprocess input data'''
    X, y = loadTrain(path, num_training, img_h, img_w)
    X_p = preprocess_input(X)
    Y = to_categorical(y, nb_classes)
    return X_p, Y


def buildModel(input_shape, nb_classes):
    ''' build a segnet model'''
    model = SegNet(input_shape=input_shape, classes=nb_classes)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    return model


def trainModel(X, Y, model, nb_epoch, validation_split=0.2, batch_size=1, save_path=""):
    ''' train the segnet model to fit the data, and save it for every epoch'''

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2
        )

    save_model = ModelCheckpoint(save_path + "weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)
    # model.fit(X, Y, callbacks=[save_model], validation_split=validation_split, batch_size=batch_size, epochs=nb_epoch)
    model.fit_generator(datagen.flow(X, Y, batch_size=batch_size), steps_per_epoch=len(X)/batch_size, epochs=nb_epoch)
    return model


def loadModel(path):
    '''load model from hdf5 file'''
    return keras.models.load_model(path)


def saveModel(model, path):
    '''save model to hdf5 file'''
    model.save(path)


# parameters
num_training = 100  # number of images
img_w = 400  # width in pixel of training data
img_h = 400
num_channels = 3  # rgb
num_classes = 2  # road or background

# param
input_shape = (400, 400, 3)
nb_classes = 2
nb_epoch = 20  # number of round of training
validation_split = 0.2  # separate training and validation
batch_size = 4  # ?

# data paths
path = '../../data/training/'  # path to the data
save_path = '../../results/keras-segnet/'
model_name = 'model3.hdf5'

# prepare input, build model, train model, save model
if __name__ == '__main__':
    X, Y = prepareInput(path, num_training, img_h, img_w, nb_classes)
    model = buildModel(input_shape, nb_classes)
    print('model built, training...')
    trainModel(X, Y, model, nb_epoch, validation_split, batch_size, save_path)
    print('finished training')
    saveModel(model, save_path + model_name)
