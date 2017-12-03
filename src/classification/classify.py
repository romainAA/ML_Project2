from src import *


def predict_image(model, img):
    height, width = img.shape[0:2]
    crop_height, crop_width = model.input_shape[0:2]
    pred = np.zeros(img.shape)
    for r in range(0, height, 16):
        for c in range(0, width, 16):
            crop = get_crop(img, r - 16, c - 16, crop_height, crop_width, height, width)
            is_road = model.predict(crop)

def get_crops(img, crop_size, )

def get_crop(img, r, c, crop_height, crop_width, img_height, img_width):
    crop = np.empty([crop_height, crop_width, 3])
    for row in range(crop_height):
        for col in range(crop_width):
            new_r = abs(r + row)
            if new_r >= img_height:
                new_r = img_height - (new_r - img_height) - 1
            new_c = abs(c + col)
            if new_c >= img_width:
                new_c = img_width - (new_c - img_width) - 1
            crop[row, col] = img[new_r, new_c]
    return crop
