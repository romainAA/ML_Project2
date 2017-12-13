import reader.given as gi
import reader.featuresManagment as fm
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from sklearn import linear_model

imgs, gts = gi.load_all_images('../Data/training/')

patch_size = 20
img_patches = [gi.img_crop(imgs[i], patch_size, patch_size) for i in range(imgs.shape[0])]
gt_patches = [gi.img_crop(gts[i], patch_size, patch_size) for i in range(imgs.shape[0])]

img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

X = np.asarray([ gi.extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([gi.value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

X = fm.add_neighbors(X, patch_size)
print(X)
print(X.shape[0])
print(X[np.where(X[:][2]==1)])

'''
maxTmp = 0;
c = 0;
for c in range(10):
    logreg = linear_model.LogisticRegression(C=100.8+(10**(-2))*c, class_weight="balanced")
    logreg.fit(X, Y)
    Zlr = logreg.predict(X)

    correctLr = 1 - sum(abs(Y-Zlr))/Y.shape[0]
    if correctLr > maxTmp:
        maxTmp = correctLr
        cF = c
print(cF, maxTmp) # 9
'''

'''
maxTmp = 0;
a1F = a2F = l1F = l2F = 0;
for a1 in range(10):
    for a2 in range(10):
        for l1 in range(10):
            for l2 in range(10):
                br = linear_model.BayesianRidge(alpha_1=10**(-1*a1),alpha_2=10**(-1*a2),lambda_1=10**(-1*l1),lambda_2=10**(-1*l2))
                br.fit(X, Y)
                Zbr = br.predict(X)

                correctBr = 1 - sum(abs(Y-Zbr))/Y.shape[0]
                if correctBr > maxTmp:
                    maxTmp = correctBr
                    a1F = a1
                    a2F = a2
                    l1F = l1
                    l2F = l2
print(a1F, a2F, l1F, l2F, maxTmp) # 0 9 9 0
'''

'''
maxTmp = 0;
for a in range(10):
    r = linear_model.Ridge(alpha=10**(-1*a)) # normailzing doesn't seems to change
    r.fit(X, Y)
    Zr = r.predict(X)

    correctR = 1 - sum(abs(Y-Zr))/Y.shape[0]
    if correctR > maxTmp:
        maxTmp = correctR
        a1F = a
print(a1F, maxTmp) # 9
'''


logreg = linear_model.LogisticRegression(C=100.8, class_weight="balanced")
logreg.fit(X, Y)
Zlr = logreg.predict(X)

correctLr = 1 - sum(abs(Y-Zlr))/Y.shape[0]
print(correctLr) # 0.595147928994


# Run prediction on the img_idx-th image
img_idx = 2
Xi = gi.extract_img_features('../Data/test_set_images/test_' +str(img_idx) + '/test_'+ str(img_idx)+'b.png')
Xi = fm.add_neighbors(Xi, patch_size)
Zi = logreg.predict(Xi)

# Display prediction as an image
w = gts[img_idx].shape[0]
h = gts[img_idx].shape[1]
predicted_im = gi.label_to_img(w, h, patch_size, patch_size, Zi)
cimg = gi.concatenate_images(imgs[img_idx], predicted_im)
'''
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(cimg, cmap='Greys_r')

new_img = gi.make_img_overlay(imgs[img_idx], predicted_im)

plt.imshow(new_img)
plt.show()
'''

#im = imgs[0]
# im2 = gi.img_crop(im,100,100)
# plt.imshow(im,cmap='Greys_r')
# datas = np.array([gi.extract_features_2d(a) for a in imgs ])
# plt.imshow(gts[0],cmap='Greys_r')
# plt.show()
