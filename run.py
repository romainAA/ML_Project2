from src import *
from src.model.segnet import SegNet
from src.tools.predict import predict_test_set, submit
from keras.optimizers import Adadelta
import src.reader.data_augmentation as da

USE_PRETRAINED_MODEL = True  # If you want to use the pretrained model, set true

if __name__ == '__main__':

    model = SegNet()
    model.build()
    model.compile()

    if USE_PRETRAINED_MODEL:
        model.load_weights('model_submit.hdf5')
    else:
        da.AugmentDataSet('data/training/', 'data/augmented_training/', 100, 3, 57)
        model.load_data('data/augmented_training/')
        model.train('submit_model_1.hdf5', nb_epoch=168, batch_size=4, tb_path='submit_1/', validation_split=.01)
        model.compile(optimizer=Adadelta(.5), save=True)
        model.train('submit_model_2.hdf5', nb_epoch=168, batch_size=4, tb_path='submit_2/', validation_split=.01)

    predict_test_set(model)
    submit(submission_filename='data/submission_final.csv')
