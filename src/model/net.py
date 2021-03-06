from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.applications import imagenet_utils
from keras.models import load_model

from src import *


class Net(object):
    """Abstract class used to encapsulate the models we try to fit"""
    def __init__(self, rows=400, cols=400, channels=3):
        """Instantiates the model"""
        self.input_shape = (rows, cols, channels)
        self.model = None
        self.optimizer = None
        self.loss = None
        self.metrics = []
        self.X = None
        self.Y = None
        self.result_path = PROJECT + 'results/'
        self.log_path = PROJECT + 'logs/'

    def model(self):
        """Simple getter for the model"""
        return self.model

    def build(self):
        """Builds the model: needs to be implemented in subclasses"""
        pass

    def compile(self, optimizer=None, loss=None, metrics=None, save=False):
        """
        Compiles the model.
        Sets all the learning parameters to the one set in __init__ by default.
        :param save: if True saves the current model before recompiling it.
        """
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss
        if metrics is None:
            metrics = self.metrics

        if save:
            self.save()

        if self.model is not None:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if save:
            self.load()

        return self.model

    def train(self, save_path, nb_epoch=20, batch_size=1, validation_split=.1, tb_path='tmp/'):
        save_model = ModelCheckpoint(self.result_path + save_path + "weights.{epoch:02d}-{loss:.2f}.hdf5",
                                     monitor='loss',
                                     verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
        reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min')
        tb_path = self.log_path + tb_path
        tensor_board = TensorBoard(log_dir=tb_path, histogram_freq=0, batch_size=batch_size, write_graph=True,
                                   write_grads=False, write_images=False, embeddings_freq=0,
                                   embeddings_layer_names=None, embeddings_metadata=None)
        callbacks = [save_model, reduce_lr, tensor_board]

        self.model.fit(self.X, self.Y, callbacks=callbacks, batch_size=batch_size, validation_split=validation_split,
                       epochs=nb_epoch)
        self.save(save_path)

        return self.model

    def load_data(self, path='data/augmented-training/'):
        """
        Loads the training data in memory.
        Needs to be implemented in subclasses
        """
        pass

    def save(self, path='tmp.hdf5'):
        """Saves the model in a file."""
        path = self.result_path + path
        self.model.save(path)

    def load(self, path='tmp.hdf5'):
        """Load the model from a file"""
        path = self.result_path + path
        self.model = load_model(path)

    def load_weights(self, path='tmp.hdf5'):
        "Loads the weights from a file into the model"
        path = self.result_path + path
        self.model.load_weights(path)
        return self.model
