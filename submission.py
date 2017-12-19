from src import *

from src.model.improver import Improver
from src.model.segnet import SegNet
from src.model.unet import Unet
from src.model.classifier import Classifier
from src.tools.predict import predict_test_set, predict_train_set, submit

model = SegNet()

model.build()

model.compile()

model.load_weights('tue-19-178.hdf5')
predict_test_set(model)

submit(submission_filename='data/submission_segnet-aug-178e.csv')
