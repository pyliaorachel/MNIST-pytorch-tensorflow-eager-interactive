import tensorflow as tf
from tensorflow import keras, nn


class Net(keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(10, kernel_size=5)
        self.conv2 = keras.layers.Conv2D(20, kernel_size=5)
        self.dense = keras.layers.Dense(10, activation='softmax')

        self.dropout = keras.layers.Dropout(0.5)
        self.max_pool = keras.layers.MaxPooling2D(2)

    def call(self, x, training=True):
        y = nn.relu(self.max_pool(self.conv1(x)))
        y = nn.relu(self.max_pool(self.dropout(self.conv2(y), training=training)))

        # Flatten feature matrix
        batch_size = y.shape[0]
        y = tf.reshape(y, (batch_size, -1))

        y = self.dense(y)
        return y
