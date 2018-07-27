#
# Handwritten number predictor
#

import os
import argparse
from PIL import Image

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

from .model import Net


"""Settings"""

package_dir = os.path.dirname(os.path.abspath(__file__))
default_img_path = os.path.join(package_dir,'test_2.png')

parser = argparse.ArgumentParser(description='PyTorch MNIST Predictor')
parser.add_argument('--image', type=str, default=default_img_path, metavar='IMG',
                            help='image for prediction (default: {})'.format(default_img_path))
args = parser.parse_args()


"""Make Prediction"""

# Load & transform image
img = tf.image.decode_png(tf.read_file(args.image), channels=1)
img = tf.image.resize_images(img, (28, 28))
img = ((img / 255) - 0.1307) / 0.3081 # Normalize
img = tf.expand_dims(img, 0) # Squeeze in batch_size dim

# Create model
model = Net()

# Load parameters; they will only be restored after the first run of the mode, in which variables in model are lazily created
checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_dir)):
    global_step = tf.train.get_or_create_global_step()

    # Predict
    output = model(img, training=False)
    pred = tf.argmax(output, 1) 
    print('Prediction: {}'.format(pred.numpy()[0]))
