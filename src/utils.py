
import json
import numpy as np


def load_json(json_path):
    with open(json_path, 'r') as outfile:
        return json.load(outfile)


def heatmap(x):
    x = np.squeeze(x)
    x = np.sum(x, axis=2)

    x_pos = np.maximum(0, x)
    x_neg = np.minimum(0, x)

    x_pos = x_pos / x_pos.max()
    x_pos = x_pos[..., np.newaxis]
    r = 0.9 - np.clip(x_pos-0.3,0,0.7)/0.7*0.5
    g = 0.9 - np.clip(x_pos-0.0,0,0.3)/0.3*0.5 - np.clip(x_pos-0.3,0,0.7)/0.7*0.4
    b = 0.9 - np.clip(x_pos-0.0,0,0.3)/0.3*0.5 - np.clip(x_pos-0.3,0,0.7)/0.7*0.4

    x_neg = x_neg * -1.0
    x_neg = x_neg / (x_neg.max() + 1e-9)
    x_neg = x_neg[..., np.newaxis]
    r2 = 0.9 - np.clip(x_neg-0.0,0,0.3)/0.3*0.5 - np.clip(x_neg-0.3,0,0.7)/0.7*0.4
    g2 = 0.9 - np.clip(x_neg-0.0,0,0.3)/0.3*0.5 - np.clip(x_neg-0.3,0,0.7)/0.7*0.4
    b2 = 0.9 - np.clip(x_neg-0.3,0,0.7)/0.7*0.5

    return np.concatenate([r, g, b], axis=-1) + np.concatenate([r2, g2, b2], axis=-1)


def vgg_addmean(image_batch):
    vgg_mean = [103.939, 116.779, 123.68]
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([blue + vgg_mean[0], green + vgg_mean[1], red + vgg_mean[2]], 2))
    return img_trans


def bgr_to_rgb(image_batch):
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([red, green, blue], 2))
    return img_trans
