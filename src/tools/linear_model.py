import src.reader.given as gi
import src.tools.features_managment as fm
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from sklearn import linear_model


def evaluateModel(X, Y, model):
    """ Evaluate the quality of a Model

    Takes a model, an input X and an output Y,
    use it to train the model with those X and Y,
    and return the percentage of correct guesses made on the training data
    """
    model.fit(X, Y)
    Ypr = model.predict(X)
    return 1 - sum(abs(Y - Ypr)) / Y.shape[0]


def print_img_from_model(model, imgs, gts):
    """ Use a trained model to print an image overlayed with the prediction on it."""

    img_idx = 2
    patch_size = 20

    Xi = gi.extract_img_features('../data/test_set_images/test_' + str(img_idx) + '/test_' + str(img_idx) + 'b.png')
    Xi = fm.add_neighbors(Xi, patch_size)
    Zi = model.predict(Xi)

    w = gts[img_idx].shape[0]
    h = gts[img_idx].shape[1]
    predicted_im = gi.label_to_img(w, h, patch_size, patch_size, Zi)
    new_img = gi.make_img_overlay(imgs[img_idx], predicted_im)
    plt.imshow(new_img)
    plt.show()


def find_best_LogisticRegression(X, Y):
    """ Look for the optimal parameter for a logistic regression

    Evaluate the a logistic regression model with different values for the parameter \'C\'
    and print the best parameter uncountered and the result it gave
    """
    max_tmp = 0
    cF = 0
    for c in range(10):
        correctnessLr = evaluateModel(X, Y, linear_model.LogisticRegression(C=100.8 + (10 ** (-2)) * c,
                                                                            class_weight="balanced"))
        if correctnessLr > max_tmp:
            max_tmp = correctnessLr
            cF = c
    print('Logistic Regression ', cF, max_tmp)  # 9
    return max_tmp


def find_best_BayesianRidge(X, Y):
    """ Look for the optimal parameters for a baysian ridge egression

    Evaluate the a baysian ridge regression model with different values for all of the parameters
    and print the best parameters uncountered and the result they gave
    """
    max_tmp = 0
    a1F = a2F = l1F = l2F = 0
    for a1 in range(10):
        for a2 in range(10):
            for l1 in range(10):
                for l2 in range(10):
                    correctnessBr = evaluateModel(X, Y, linear_model.BayesianRidge(alpha_1=10 ** (-1 * a1),
                                                                                   alpha_2=10 ** (-1 * a2),
                                                                                   lambda_1=10 ** (-1 * l1),
                                                                                   lambda_2=10 ** (-1 * l2)))
                    if correctnessBr > max_tmp:
                        max_tmp = correctnessBr
                        a1F = a1
                        a2F = a2
                        l1F = l1
                        l2F = l2
    print('Baysian Ridge ', a1F, a2F, l1F, l2F, max_tmp)  # 0 9 9 0
    return max_tmp


def find_best_Ridge(X, Y):
    """ Look for the optimal parameter for a ridge regression

    Evaluate the a ridge regression model with different values for the parameter \'C\'
    and print the best parameter uncountered and the result it gave
    """
    max_tmp = 0
    a1F = 0
    for a in range(10):
        correctnessR = evaluateModel(X, Y, linear_model.Ridge(alpha=10 ** (-1 * a)))
        if correctnessR > max_tmp:
            max_tmp = correctnessR
            a1F = a
    print('Ridge ', a1F, max_tmp)  # 9
    return max_tmp


def find_best_overall():
    """ Use the 3 previous methods to find the best of the 3 regressions with the best parameters"""
    imgs, gts = gi.load_all_images('data/training/')
    X, Y = gi.produce_XY(imgs, gts)

    find_best_LogisticRegression(X, Y)
    find_best_BayesianRidge(X, Y)
    find_best_Ridge(X, Y)


def example_linear_model():
    """ Train a ridge model and use it to draw the roads over an image"""
    imgs, gts = gi.load_all_images('data/training/')
    X, Y = gi.produce_XY(imgs, gts)

    model = linear_model.Ridge(alpha=10 ** (-9))
    print(evaluateModel(X, Y, linear_model.Ridge(alpha=10 ** (-9))))
    model.fit(X, Y)
    print_img_from_model(model, imgs, gts)


if __name__ == '__main__':
    find_best_overall()
