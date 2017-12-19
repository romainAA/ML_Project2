from src.model.net import Net
from src.model.segnet import SegNet
from src import *
from src.tools.predict import pixel_to_class

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Reshape, Activation
from keras.models import Model
from keras.optimizers import Adam


class Unet(SegNet):
    def __init__(self):
        Net.__init__(self)
        # self.optimizer = Adam(lr=1e-2)
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.result_path += 'unet/'
        self.log_path += 'unet/'

    def build(self):
        inputs = Input((self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        initializer = 'he_normal'

        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv3)
        drop3 = Dropout(.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv4)
        drop4 = Dropout(.5)(conv4)

        up5 = UpSampling2D(size=(2, 2))(drop4)
        up5 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(up5)
        merge5 = concatenate([drop3, up5], axis=3)
        conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge5)
        conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv5)

        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(up6)
        merge6 = concatenate([conv2, up6], axis=3)
        conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge6)
        conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=initializer)(up7)
        merge7 = concatenate([conv1, up7], axis=3)
        conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv7)
        conv7 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv7)

        conv8 = Conv2D(2, (1, 1), padding='valid')(conv7)
        conv8 = Reshape((self.input_shape[0] * self.input_shape[1], 2))(conv8)
        out = Activation('sigmoid')(conv8)

        self.model = Model(input=inputs, output=out)
        return self.model
