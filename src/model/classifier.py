from src.model.net import Net
from src import *
from src.tools.predict import pixel_to_class

from keras.applications import imagenet_utils
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential

import src.reader.patch_creator as pc

import glob
import os


class Classifier(Net):
    def __init__(self, rows=48, cols=48, channels=3):
        super().__init__(rows=rows, cols=cols, channels=channels)
        self.loss = "categorical_crossentropy"
        self.optimizer = "adadelta"
        self.metrics = ["accuracy"]
        self.result_path += 'classifier/'
        self.log_path += 'classifier/'

    def build(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.15))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.15))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.15))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.15))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dropout(.15))

        model.add(Dense(2, activation='softmax'))

        self.model = model

        return self.model

    def load_data(self, path='data/patches_48/data.npz'):
        path = PROJECT + path

        imgs, gt_imgs, labels = pc.load_from_file(path)

        self.X = imgs
        self.Y = labels

    def predict(self, img):
        crops = self.get_crops(img)
        preds = self.model.predict(crops)

        height, width = img.shape[0:2]
        crops_per_row = width // 16
        pred = np.zeros(img.shape[0:2])
        for c in range(len(crops)):
            ''' row and col are the indices of the top left pixel of the image '''
            row = (c // crops_per_row) * 16
            col = (c % crops_per_row) * 16
            if preds[c, 0] < preds[c, 1]:
                pred[row:row + 16, col:col + 16] = 1.
        return pred

    def get_crops(self, img):
        crop_size = self.input_shape[0]
        height, width = img.shape[0:2]
        padded = self.pad_image(img)
        crops_per_col = height // 16
        crops_per_row = width // 16
        nb_crops = crops_per_col * crops_per_row
        crops = np.empty([nb_crops, crop_size, crop_size, 3])
        for c in range(nb_crops):
            ''' row and col are the indices of the top left pixel of the padded image '''
            row = (c // crops_per_row) * 16
            col = (c % crops_per_row) * 16
            crops[c] = (padded[row:row + crop_size, col: col + crop_size]).copy()
        return crops

    def pad_image(self, img, mode='reflect'):
        pad_size = (self.input_shape[0] - 16) // 2
        return np.lib.pad(img, ((pad_size,), (pad_size,), (0,)), mode)
