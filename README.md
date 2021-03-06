# Machine Learning project on Road Segmentation

Using Machine Learning and Deep Learning viewed in the lectures to find road on satellite pictures.
The documentation to any method can be found in the corresponding file accordingly to PEP 257.
You can also access it though a python2/3 shell after importing it using `help(file.function)`.

## Environment Dependencies:

We used the libraries:
- hdf5 : 1.10.1
- keras : 2.1.1
- numpy : 1.13.3
- opencv-contrib-python : 3.3.0.10
- pillow : 4.2.1
- pip : 9.0.1
- python : 3.6.3
- scikit-image : 0.13.0
- scikit-learn : 0.19.1
- scipy : 0.19.1
- tensorflow : 1.4.0
- tensorflow-gpu : 1.4.0
- tensorflow-tensorboard : 0.4.0rc3

## Where to put the images:
- All the data should be in the folder $PROJECT/data/

### Training Data
- Training data should be in the folder $PROJECT/data/training/
- Plain images goes in $PROJECT/data/training/images/
- Groundtruth images goes in $PROJECT/data/training/groundtruth/

### Test Data
- All Test data should be in $PROJECT/data/test_set_images/
- Test image i should be in $PROJECT/data/test_set_images/test_${i}/

## Functionalities

### Recreate the final submission results:
- In the file 'src/\_\_init\_\_.py', change the variable PROJECT to your own path.
- Run the code using `python run.py`
*Making the prediction on our dataset (using a pretrained model) takes us 5 min on an intel i7 with 8Go of RAM*
- We provide a pre-trained model to make predictions, if you want to train a new model you can set the variable USE_PRETRAINED_MODEL to False in `run.py`.
*Please note that it took us 2 hours and a half to train on a GPU, a GPU of 6 GB memory was able to handle batches of size 8, you might need to change the batch size even lower than 4 depending on your GPU*

### Find the best linear result obtained:
- In a shell go in the directory ML_Project2
- Run the command `python -c 'import src.tools.linear_model; src.tools.linear_model.find_best_overall()'`
- You will obtain the best result for each regression as well as the parameters that gave the optimal result printed in the command line

## Packages and files:

### Model package
- `net.py`: An abstract class to make the code modular
- `classifier.py`: A CNN-model that performs classification on patches of the image
- `segnet.py`: A model inspired of the segnet architecture, that make segmentation on the whole image
- `improver.py`: A model that tries to correct the output of classifier using a similar technique to segnet
- `unet.py`: A model inspired of the unet architecture

### Reader package
- `data_aumgentation.py`: takes the training set and artificially augments it using random transformations.
- `given.py`: Contains methods given for this project
- `patch_creator.py`: Takes the image and divides it in patches of fixed size.

### Tools package
- `features_managment.py`: Contains methods to add features for the linear model
- `improve.py`: Helpers functions to create submissions for the improver.
- `linear_model.py`: Contains the methods used to look for the quality of the linear model
- `predict.py`: Helpers functions to create submissions.

## Credits:
We got inspired by some implementations found online:
- Segnet : https://github.com/namakemono/keras-segnet
- Unet : https://github.com/zhixuhao/unet
