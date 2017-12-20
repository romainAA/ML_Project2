from src import *

from src.model.improver import Improver
from src.model.segnet import SegNet
from src.model.unet import Unet
from src.model.classifier import Classifier
from src.tools.improve import improve_test_set, submit

from keras.optimizers import Adadelta

model = SegNet()

model.build()

# 1 -> .5

# model.optimizer = Adadelta(lr=.5)
model.compile()

# model.load_weights('tue-19-168d.hdf5')

model.load_data('data/augmented-training/')
# model.load_data('data/patches_80/aug-data.npz')

model.train('wed-20-301d.hdf5', nb_epoch=301, batch_size=8, tb_path='wed-20-.15/', validation_split=0.01)

print("DONE")
