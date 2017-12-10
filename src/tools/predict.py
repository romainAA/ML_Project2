from src import *
import os, os.path
import re

foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


def pixel_to_class(pixel):
    return 0 if pixel[0] > pixel[1] else 1


def predict_test_set(model, path='data/test_set_images/'):
    path = PROJECT + path
    for i in range(1, 51):
        folder_path = path + 'test_' + str(i) + '/'
        image_path = folder_path + 'test_' + str(i) + '.png'
        img = misc.imread(image_path)
        pred = model.predict(img)
        pred_path = folder_path + 'pred_' + str(i) + '.png'
        misc.imsave(pred_path, pred)
    print("DONE")


def predict_train_set(model, path='data/training/'):
    path = PROJECT + path
    pred_dir = path + 'preds/'
    img_dir = path + 'images/'
    gt_dir = path + 'groundtruth/'

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    img_names = list(filter(
        lambda name: not name.startswith('.') and name.endswith('.png') and os.path.exists(gt_dir + name),
        os.listdir(img_dir)))
    for name in img_names:
        img = misc.imread(img_dir + name)
        pred = model.predict(img)
        misc.imsave(pred_dir + name, pred)
    print("DONE")


def submit(submission_filename='data/dummy_submission.csv', pred_path='data/test_set_images/'):
    submission_filename = PROJECT + submission_filename
    image_filenames = []
    pred_path = PROJECT + pred_path
    for i in range(1, 51):
        image_filename = pred_path + 'test_' + str(i) + '/pred_' + str(i) + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
    print("DONE")


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))
