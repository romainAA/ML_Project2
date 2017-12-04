from src import *


def predict_image(model, test_path, img_num):
    img_path = test_path + 'test_' + str(img_num) + '/test_' + str(img_num) + '.png'
    pred_path = test_path + 'test_' + str(img_num) + '/pred_' + str(img_num) + '.png'
    ret = []
    img = misc.imread(img_path) / 255.0
    crop_size = model.input_shape[0]
    crops = get_crops(img, crop_size)
    preds = model.model.predict(crops)

    height, width = img.shape[0:2]
    crops_per_row = width // 16
    pred = np.zeros(img.shape[0:2])
    for c in range(len(crops)):
        ''' row and col are the indices of the top left pixel of the image '''
        row = (c // crops_per_row) * 16
        col = (c % crops_per_row) * 16
        string = "{:03d}_{}_{},".format(img_num, col, row)
        if preds[c, 0] < preds[c, 1]:
            pred[row:row + 16, col:col + 16] = 1.
            string += '1'
        else:
            string += '0'
        ret.append(string + '\n')
    misc.imsave(pred_path, pred)
    return ret


def submit(model, test_path, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(1, 51):
            ret = predict_image(model, test_path, i)
            for string in ret:
                f.write(string)
        print('DONE\n')


def get_crops(img, crop_size):
    height, width = img.shape[0:2]
    padded = pad_image(img, crop_size)
    crops_per_col = height // 16
    crops_per_row = width // 16
    nb_crops = crops_per_col * crops_per_row
    crops = np.empty([nb_crops, crop_size, crop_size, 3])
    for c in range(nb_crops):
        ''' row and col are the indices of the top left pixel of the padded image '''
        row = (c // crops_per_row) * 16
        col = (c % crops_per_row) * 16
        crops[c] = (padded[row:row + crop_size, col: col + crop_size]).copy()
    return crops


def pad_image(img, crop_size, mode='minimum'):
    pad_size = (crop_size - 16) // 2
    return np.lib.pad(img, ((pad_size,), (pad_size,), (0,)), mode)
