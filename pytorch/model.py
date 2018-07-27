import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms    # torchvision contains common utilities for computer vision


class Net(nn.Module):  # Inherit from `nn.Module`, define `__init__` & `forward`
    def __init__(self):
        # Always call the init function of the parent class `nn.Module`
        # so that magics can be set up.
        super(Net, self).__init__()

        # Define the parameters in your network.
        # This is achieved by defining the shapes of the multiple layers in the network.

        # Define two 2D convolutional layers (1 x 10, 10 x 20 each)
        # with convolution kernel of size (5 x 5).
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Define a dropout layer
        self.conv2_drop = nn.Dropout2d()

        # Define a fully-connected layer (320 x 10)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # Define the network architecture.
        # This is achieved by defining how the network forward propagates your inputs

        # Input image size: 28 x 28, input channel: 1, batch size (training): 64 

        # Input (64 x 1 x 28 x 28) -> Conv1 (64 x 10 x 24 x 24) -> Max Pooling (64 x 10 x 12 x 12) -> ReLU -> ...
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # ... -> Conv2 (64 x 20 x 8 x 8) -> Dropout -> Max Pooling (64 x 20 x 4 x 4) -> ReLU -> ...
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # ... -> Flatten (64 x 320) -> ...
        x = x.view(-1, 320)

        # ... -> FC (64 x 10) -> ...
        x = self.fc(x)

        # ... -> Log Softmax -> Output
        return F.log_softmax(x, dim=1)
