# Machine Learning project on Road Segmentation

Using Machine Learning and Deep Learning viewed in the lectures to find road on satellite pictures.
The documentation to any method can be found in the corresponding file accordingly to PEP 257.
You can also access it though a python2/3 shell after importing it using `help(file.function)`.

## Functionalities

### Change the training and/or the testing Data
- Go to the directory ML_Project2/Data
- Replace the `test_set_images`and the `training`folder by your data

##Environment Dependencies:
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

##Running Instructions:
- In the file 'src/__init__.py' you need to change the variable PROJECT to your own path.
- run the code using python run.py
- We provide a pretrained model to make predictions if you want to train a new model you can set the variable USE_PRETRAINED_MODEL to False in run.py

### Find the best linear result obtained
- In a shell go in the directory ML_Project2/src
- run the command `python3 -c 'import tools.linear_model; tools.linear_model.find_best_overall()'`
- You will obtain the best result for each regression as well as the parameters that gave the optimal result

## Methods:

### Classification package

### Keras-Segnet package

### Model package

### Reader package

### Tools package

### Unet package
