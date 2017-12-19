from src import *

from src.model.improver import Improver
from src.model.segnet import SegNet
from src.model.unet import Unet
from src.model.classifier import Classifier
from src.tools.improve import improve_test_set, submit

from keras.optimizers import Adadelta

model = SegNet()

model.build()

# 1 -> 0.1 -> 0.08 -> .02 -> .5

model.optimizer = Adadelta(lr=0.5)
model.compile()

model.load_weights('mon-18-136.hdf5')

model.load_data('data/augmented-training/')
# model.load_data('data/patches_80/aug-data.npz')

model.train('tue-19-178.hdf5', nb_epoch=42, batch_size=6, tb_path='tue-19/', validation_split=.01)

print("DONE")