from src.model.net import Net
from src import *
from src.tools.predict import pixel_to_class

import os, os.path
from keras.utils import np_utils
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Reshape, MaxPooling2D, Input
from keras.models import Model


class Improver(Net):
    """A subclass of Net that implement a netork improving the results of a first model"""
    def __init__(self, rows=400, cols=400, channels=1):
        """Instantiates the class"""
        super().__init__(rows, cols, channels)
        self.loss = "categorical_crossentropy"
        self.optimizer = "adadelta"
        self.metrics = ["accuracy"]
        self.result_path += 'improver/'
        self.log_path += 'improver/'

    def build(self):
        """Builds the model"""
        # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
        img_input = Input(shape=self.input_shape)
        x = img_input
        # Encoder
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Decoder
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(2, (1, 1), padding="valid")(x)
        x = Reshape((self.input_shape[0] * self.input_shape[1], 2))(x)
        x = Activation("softmax")(x)
        self.model = Model(img_input, x)

        return self.model

    def to_categorical(self, y):
        """Changes the shape of the groundtruth"""
        num_samples = len(y)
        Y = np_utils.to_categorical(y.flatten(), 2)
        return Y.reshape((num_samples, int(y.size / num_samples), 2))

    def to_class_tab(self, image):
        '''convert groundtruth to class [0,1]'''
        image[np.where(image < 128)] = 0
        image[np.where(image >= 128)] = 1
        return image

    def load_data(self, path='data/augmented-training/'):
        """
        Loads the training data.
        That is : the predictions of the first model and the groundtruth.
        """
        path = PROJECT + path
        img_path = path + '/preds/'
        gt_path = path + '/groundtruth/'

        img_names = list(filter(
            lambda name: not name.startswith('.') and name.endswith('.png') and os.path.exists(gt_path + name),
            os.listdir(img_path)))
        count = len(img_names)

        images = np.empty([count, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        gts = np.empty([count, self.input_shape[0], self.input_shape[1]])

        for i, name in enumerate(img_names):
            img = np.reshape(misc.imread(img_path + name), self.input_shape)
            images[i] = img
            gt = misc.imread(gt_path + name)
            gts[i] = self.to_class_tab(gt)

        self.X = images
        self.Y = self.to_categorical(gts)

    def predict(self, img):
        """
        Makes the prediction.
        """
        input_img = np.array(img, dtype=np.float64)
        height, width = input_img.shape[0:2]
        if height != self.input_shape[0] or width != self.input_shape[1]:
            return self.predict_diff_size(input_img)
        input_img = np.reshape(input_img, [1, height, width, 1])
        pred = self.model.predict(input_img, 1)[0]
        pred = np.array(list(map(pixel_to_class, pred)))
        return np.reshape(pred, (height, width))

    def predict_diff_size(self, img):
        "Adapts the prediction to images of different size."
        height, width = img.shape[0:2]
        patch_height, patch_width = self.input_shape[0:2]

        patches_per_row = (width // patch_width)
        if patches_per_row * patch_width < width:
            patches_per_row += 1
        patches_per_col = (height // patch_height)
        if patches_per_col * patch_height < height:
            patches_per_col += 1

        pred = np.zeros((height, width))

        for r in range(patches_per_col):
            row = r * patch_height
            if row + patch_height > height:
                row = height - patch_height
            for c in range(patches_per_row):
                col = c * patch_width
                if col + patch_width > width:
                    col = width - patch_width

                input_img = img[row: row + patch_height, col: col + patch_width]
                pred[row: row + patch_height, col: col + patch_width] = self.predict(input_img)

        return pred
