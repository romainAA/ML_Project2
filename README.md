# Machine Learning project on Road Segmentation

Using Machine Learning and Deep Learning viewed in the lectures to find road on satellite pictures.
The documentation to any method can be found in the corresponding file accordingly to PEP 257.
You can also access it though a python2/3 shell after importing it using `help(file.function)`.

## Functionalities

### Change the training and/or the testing Data
- Go to the directory ML_Project2/Data
- Replace the `test_set_images` and/or the `training` folder by your data

### Find the best linear result obtained
- In a shell go in the directory ML_Project2
- run the command `python3 -c 'import src.tools.linear_model; src.tools.linear_model.find_best_overall()'`
- You will obtain the best result for each regression as well as the parameters that gave the optimal result

## Methods:

### Classification package

#### `classify.py`:

#### `patches_48.py`:

### Keras-Segnet package

#### `mask_to_submission.py`:

#### `prediction.py`:

#### `segnet.py`:

#### `train.py`:

### Model package

#### `improver.py`:

#### `losses.py`:

#### `net.py`:

#### `segnet.py`:

### Reader package

#### `data-aumgentation.py`:

#### `given.py`:

#### `patch_creator.py`:

#### `raw.py`:

#### `tf_aerial_images.py`:

### Tools package

#### `featuresManagment.py`:

#### `improve.py`:

#### `linear_model.py`:

#### `mask_to_submission.py`:

#### `predict.py`:

#### `submission_to_mask.py`:

#### `train.py`:

### Unet package

#### `submit.py`:

#### `unet.py`:
