from src import *

from src.model.improver import Improver
from src.tools.improve import improve_test_set, submit

model = Improver(400, 400, 1)

model.build()

model.compile()
model.load_weights('results/improver/model2.hdf5')

# model.load_data()
# model.train('results/improver/model2.hdf5', nb_epoch=10, batch_size=4)
improve_test_set(model)
submit(submission_filename='data/improve-net.csv')
