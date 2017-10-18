#
# Handwritten number predictor
#

import os
import argparse
from PIL import Image
import torch
from torchvision import transforms

from .model import Net


"""Settings"""

package_dir = os.path.dirname(os.path.abspath(__file__))
default_img_path = os.path.join(package_dir,'test_2.png')

parser = argparse.ArgumentParser(description='PyTorch MNIST Predictor')
parser.add_argument('--image', type=str, default=default_img_path, metavar='IMG',
                            help='image for prediction (default: {})'.format(default_img_path))
args = parser.parse_args()


"""Make Prediction"""

# Load model
model_path = os.path.join(package_dir,'model')
model = Net()
model.load_state_dict(torch.load(model_path))

# Load & transform image
ori_img = Image.open(args.image).convert('L')
t = transforms.Compose([
    transforms.Scale((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))

# Predict
model.eval()
output = model(img)
pred = output.data.max(1, keepdim=True)[1][0][0]
print('Prediction: {}'.format(pred))

# Close image file
ori_img.close()
