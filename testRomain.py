import src.reader.raw as raw
import src.reader.given as given
import src.reader.tf_aerial_images as tfai
import matplotlib.pyplot as plt
import numpy as np

"""
imgs, gts = raw.load_data();
plt.imshow(imgs[0])
plt.show()
"""

imgs = tfai.extract_data('Data/training/images/', 4)
gts = tfai.extract_data('Data/training/grounftruth/', 4)
im = imgs[0]
gt = gts[0]

m = tfai.Sequential();
m.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

im2 =
plt.imshow()
plt.show()
