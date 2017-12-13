import keras
import os
import glob
import numpy as np
from scipy import misc
from segnet import SegNet, preprocess_input, to_categorical
from keras.callbacks import ModelCheckpoint


def predToClass(pred):
    ''' make classification based on the prediction '''
    return np.array(list(map(pixelToClass, pred[0])))


def pixelToClass(pixel):
    '''classify a pixel based on the model prediction'''
    if pixel[0] > pixel[1]:
        return 0
    else:
        return 1


def vectToMatrix(vect, img_h, img_w):
    ''' make the prediction a vector of the right size '''
    res = np.zeros((img_h, img_w))
    for i in range(img_h):
        res[i] = vect[i * (img_w): (i + 1) * (img_w)]
    return res


def smallPredictionFromModel(input_img, input_h, input_w, model):
    ''' make prediction on a 400*400 input'''
    img_p = preprocess_input(input_img)
    pred = model.predict(img_p)
    class_pred = predToClass(pred)
    prediction = vectToMatrix(class_pred, input_h, input_w)
    prediction[np.where(prediction == 1)] = 255
    return prediction


def predictionFromModel(image_path, img_h, img_w, input_h, input_w, model):
    '''load image 608*608 and make prediction on it'''
    img = misc.imread(image_path)
    pred = np.zeros((img_h, img_w))

    h_mid = (img_h - input_h)
    w_mid = (img_w - input_w)
    input_tab = np.zeros((4, input_h, input_w, 3))
    input_tab[0] = np.array([img[:input_h, :input_w]])
    input_tab[1] = np.array([img[h_mid:, :input_w]])
    input_tab[2] = np.array([img[:input_h, w_mid:]])
    input_tab[3] = np.array([img[h_mid:, w_mid:]])

    pred1 = smallPredictionFromModel(input_tab[0:1], input_h, input_w, model)
    pred2 = smallPredictionFromModel(input_tab[1:2], input_h, input_w, model)
    pred3 = smallPredictionFromModel(input_tab[2:3], input_h, input_w, model)
    pred4 = smallPredictionFromModel(input_tab[3:4], input_h, input_w, model)

    pred[:input_h, :input_w] = pred1
    pred[h_mid:, :input_w] = pred2
    pred[:input_h, w_mid:] = pred3
    pred[h_mid:, w_mid:] = pred4

    return pred


def loadModel(path):
    '''load model from hdf5 file'''
    return keras.models.load_model(path)


def savePrediction(file_name, prediction):
    '''save the prediction as a png image'''
    misc.imsave(file_name, prediction)


# parameters
num_test = 50
input_w = 400  # size of the input of the model
input_h = 400
img_w = 608  # size of the test data
img_h = 608
num_channels = 3  # rgb

# param
input_shape = (400, 400, 3)
nb_classes = 2  # road or background

# data paths
path = '../../data/test_set_images/'
model_path = '../../results/keras-segnet/aug-model3.hdf5'

if __name__ == '__main__':
    model = loadModel(model_path)
    for i in range(1, num_test + 1):
        folder_path = path + 'test_' + str(i) + '/'
        image_path = folder_path + 'test_' + str(i) + '.png'
        prediction = predictionFromModel(image_path, img_h, img_w, input_h, input_w, model)
        pred_path = folder_path + 'pred_' + str(i) + '.png'
        savePrediction(pred_path, prediction)
