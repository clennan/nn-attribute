
import argparse
from models.vgg16 import VGG16
from src.propagate import propagate
from src.utils import heatmap
import numpy as np
import matplotlib.pyplot as plt
import os


def main(model, weights_file, classes_file, alpha, image_file, target_file):
    if model.lower() == 'vgg16':
        vgg16 = VGG16(weights_file, classes_file)
        model = vgg16.build_model(1, alpha)
        image = vgg16.load_image(image_file)
        classes = vgg16.classes

    logits, relevances = propagate(image, model)
    prediction = classes[str(np.argmax(logits[0]))]
    print('predicted class: {}'.format(prediction))

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.imshow(heatmap(relevances[0]))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(os.path.join('results', target_file))
    print('heatmap saved as {}'.format(os.path.join('results', target_file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--image-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--target-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--model',
      help='',
      required=True
      )

    parser.add_argument(
      '--weights-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--classes-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--alpha',
      help='',
      required=True,
      type=int,
      )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
