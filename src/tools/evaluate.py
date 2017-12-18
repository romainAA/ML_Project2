import reader.given as gi
import tools.featuresManagment as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import linear_model


def evaluateModel(X,Y,model):
    model.fit(X, Y)
    Ypr = model.predict(X)
    return 1 - sum(abs(Y-Ypr))/Y.shape[0]

def print_img_from_model(model,imgs,gts):
    img_idx = 2
    patch_size = 20

    Xi = gi.extract_img_features('../Data/test_set_images/test_' +str(img_idx) + '/test_'+ str(img_idx)+'b.png')
    Xi = fm.add_neighbors(Xi, patch_size)
    Zi = model.predict(Xi)

    w = gts[img_idx].shape[0]
    h = gts[img_idx].shape[1]
    predicted_im = gi.label_to_img(w, h, patch_size, patch_size, Zi)
    new_img = gi.make_img_overlay(imgs[img_idx], predicted_im)
    plt.imshow(new_img)
    plt.show()
