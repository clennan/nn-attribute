
from src.modules.sequential import Sequential
from src.modules.linear import Linear
from src.modules.convolution import Convolution
from src.modules.convolution_ab import Convolution_ab
from src.modules.reshape import Reshape
from src.modules.proppool import PropPool
from src.utils import load_json
import numpy as np
import cv2


def vgg_addmean(images):
    mean = [103.939, 116.779, 123.68]
    img_trans = []
    for img in images:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate(
            [blue + mean[0], green + mean[1], red + mean[2]], 2),
            )
    return img_trans


def bgr_to_rgb(images):
    img_trans = []
    for img in images:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([red, green, blue], 2))
    return img_trans


class VGG16(object):
    def __init__(self, weights_file, classes_file):
        self.classes = load_json(classes_file)
        self.weights = np.load(weights_file, encoding='latin1').item()

    def vgg_addmean(self, images):
        mean = [103.939, 116.779, 123.68]
        img_trans = []
        for img in images:
            blue, green, red = np.split(img, 3, 2)
            img_trans.append(np.concatenate(
                [blue + mean[0], green + mean[1], red + mean[2]], 2),
                )
        return img_trans

    def load_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        return self.vgg_addmean(image)

    def build_model(self, batch_size, alpha):
        return Sequential([
            Convolution(batch_size=batch_size,
                        initializer=self.weights['conv1_1'],
                        first=True,
                        name='conv1_1_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv1_2'],
                        alpha=alpha,
                        name='conv1_2_'),
            PropPool(name='PropPool1'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv2_1'],
                        alpha=alpha,
                        name='conv2_1_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv2_2'],
                        alpha=alpha,
                        name='conv2_2_'),
            PropPool(name='PropPool2'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv3_1'],
                        alpha=alpha,
                        name='conv3_1_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv3_2'],
                        alpha=alpha,
                        name='conv3_2_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv3_3'],
                        alpha=alpha,
                        name='conv3_3_'),
            PropPool(name='PropPool3'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv4_1'],
                        alpha=alpha,
                        name='conv4_1_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv4_2'],
                        alpha=alpha,
                        name='conv4_2_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv4_3'],
                        alpha=alpha,
                        name='conv4_3_'),
            PropPool(name='PropPool4'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv5_1'],
                        alpha=alpha,
                        name='conv5_1_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv5_2'],
                        alpha=alpha,
                        name='conv5_2_'),
            Convolution_ab(
                        batch_size=batch_size,
                        initializer=self.weights['conv5_3'],
                        alpha=alpha,
                        name='conv5_3_'),
            PropPool(name='PropPool5'),
            Reshape(name='flat1'),
            Linear(batch_size=batch_size,
                   initializer=self.weights['fc6'],
                   alpha=alpha,
                   name='fc6_'),
            Linear(batch_size=batch_size,
                   initializer=self.weights['fc7'],
                   alpha=alpha,
                   name='fc7_'),
            Linear(batch_size=batch_size,
                   initializer=self.weights['fc8'],
                   alpha=alpha,
                   name='fc8_'),
                   ])
