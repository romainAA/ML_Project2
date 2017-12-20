from src import *
from src.model.segnet import SegNet
from src.tools.predict import predict_test_set, predict_train_set, submit
import src.reader.data_augmentation as DA

USE_PRETRAINED_MODEL = True #If you want to use the pretrained model, set true


if __name__ == '__main__':

    model = SegNet()
    model.build()
    model.compile()

    if USE_PRETRAINED_MODEL:
        model.load_weights('mon-18-136.hdf5')
    else:
        DA.AugmentDataSet('data/training/', 'data/augmented-training/', 1, 1, 57)
        model.load_data('data/augmented-training/')
        model.train('submitModel.hdf5', nb_epoch=42, batch_size=4, tb_path='submit/', validation_split=.01)
    predict_test_set(model)
    submit(submission_filename='data/submissionFinal.csv')
