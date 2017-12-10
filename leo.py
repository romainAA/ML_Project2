from src import *

from src.model.segnet import SegNet

model = SegNet(400, 400, 3)

model.load_data()

model.build()

model.compile()

model.train('results/segnet/model1.h5', nb_epoch=120, batch_size=4)
