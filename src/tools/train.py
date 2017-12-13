import keras
import os
import glob
import numpy as np
from scipy import misc
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# import src.reader.raw as data


def load_train(path, height=400, width=400):
    """ load all images in the traing data set and their groundtruth """
    return data.load_data(path, height, width)


def train_model(X, Y, model, save_path, nb_epoch=20, batch_size=1, validation_split=.1):
    """ train the segnet model to fit the data, and save it for every epoch"""

    save_model = ModelCheckpoint(save_path + "weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(X, Y, callbacks=[save_model], batch_size=batch_size, validation_split=validation_split, epochs=nb_epoch)
    return model


def loadModel(path):
    """ load model from hdf5 file"""
    return keras.models.load_model(path)


def saveModel(model, path):
    """ save model to hdf5 file """
    model.save(path)