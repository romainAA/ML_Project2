import reader.given as gi
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

imgs, gts = gi.load_all_images('../Data/training/')

patch_size = 32
img_patches = [gi.img_crop(imgs[i], patch_size, patch_size) for i in range(imgs.shape[0])]
gt_patches = [gi.img_crop(gts[i], patch_size, patch_size) for i in range(imgs.shape[0])]

img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

X = np.asarray([ gi.extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([gi.value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
logreg.fit(X, Y)
Z = logreg.predict(X)

correct = 1 - sum(abs(Y-Z))/Y.shape[0]

print(correct)

# Run prediction on the img_idx-th image
img_idx = 2
Xi = gi.extract_img_features('../Data/test_set_images/test_' +str(img_idx) + '/test_'+ str(img_idx)+'.png')
Zi = logreg.predict(Xi)

# Display prediction as an image
w = gts[img_idx].shape[0]
h = gts[img_idx].shape[1]
predicted_im = gi.label_to_img(w, h, patch_size, patch_size, Zi)
cimg = gi.concatenate_images(imgs[img_idx], predicted_im)
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(cimg, cmap='Greys_r')

new_img = gi.make_img_overlay(imgs[img_idx], predicted_im)

plt.imshow(new_img)
plt.show()
#im = imgs[0]
# im2 = gi.img_crop(im,100,100)
# plt.imshow(im,cmap='Greys_r')
# datas = np.array([gi.extract_features_2d(a) for a in imgs ])
# plt.imshow(gts[0],cmap='Greys_r')
# plt.show()
