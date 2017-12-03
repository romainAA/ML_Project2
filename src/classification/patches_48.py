import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam, Adadelta
from src.tools.train import train_model

from src import *


class Net(object):
    def __init__(self, rows=48, cols=48, channels=3):
        self.input_shape = (rows, cols, channels)
        self.model = None

    def model(self):
        return self.model

    def build(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dropout(.1))

        model.add(Dense(2, activation='softmax'))

        self.model = model
        return model

    def compile(self, lr):
        """
        Compiles the model but you should use build(self) first
        :param lr: the desired learning rate
        :return: the compiled model
        """

        if self.model is not None:
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adadelta(lr=lr),
                               metrics=['accuracy'])

        return self.model

    def change_lr(self, new_lr):
        """
        Changes the learning rate of the model and compile again, temporarily saving the weights
        :param new_lr: new learning rate
        :return: the model
        """
        self.save()
        self.compile(new_lr)
        return self.load()

    def train(self, X, Y, save_path, nb_epoch=20, batch_size=1, validation_split=.1):
        train_model(X, Y, self.model, save_path, nb_epoch=nb_epoch, batch_size=batch_size,
                    validation_split=validation_split)

    def save(self, path=PROJECT + 'results/patches_48/tmp.hdf5'):
        self.model.save(path)

    def load(self, path=PROJECT + 'results/patches_48/tmp.hdf5'):
        self.model.load_weights(path)
        return self.model


    def predict(self, img):


