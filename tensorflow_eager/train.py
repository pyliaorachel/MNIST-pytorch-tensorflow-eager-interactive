import os
import argparse

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

from .model import Net


def load_data(train_batch_size, test_batch_size):
    # Load data
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = ((x_train / 255.0) - 0.1307) / 0.3081, ((x_test / 255.0 - 0.1307)) / 0.3081 # Normalize
    x_train = tf.expand_dims(x_train, -1) # Append channel dim
    x_test = tf.expand_dims(x_test, -1)

    train_size = x_train.shape[0].value
    test_size = x_test.shape[0].value

    # Wrap in tf dataset; type casting so that tf.equal can work in cal_acc
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
                                        .map(lambda x, y: (x, tf.cast(y, tf.int64)))\
                                        .shuffle(1000)\
                                        .batch(train_batch_size)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
                                        .map(lambda x, y: (x, tf.cast(y, tf.int64)))\
                                        .shuffle(1000)\
                                        .batch(train_batch_size)

    return (dataset_train, dataset_test, train_size, test_size)

def cal_loss(model, data, target, training=True):
    # Forward propagation
    output = model(data, training=training)
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, output) # The order is correct, as specified in Keras doc
    return (loss, output)

def train(model, optimizer, epoch, train_loader, batch_size, train_size, log_interval):
    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Calculate loss & gradient
        with tf.GradientTape() as tape:
            loss, output = cal_loss(model, data, target)
            loss_value = tf.reduce_mean(loss)
        gradients = tape.gradient(loss_value, model.variables)
        grads_and_vars = zip(gradients, model.variables)

        # Backward propagation
        optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())

        # Output debug message
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, train_size,
                100. * batch_idx * batch_size / train_size, loss_value.numpy()))

def test(model, test_size, test_loader):
    # Init loss & correct prediction accumulators
    test_loss = 0
    correct = 0

    # Iterate over data, accumulate total loss & correct predictions
    for data, target in test_loader:
        # Get loss and output from network
        loss, output = cal_loss(model, data, target, training=False)
        test_loss += tf.reduce_sum(loss).numpy()

        # Get correct number of predictions
        pred = tf.argmax(output, 1)
        correct += tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.int32))

    # Print out average test loss
    test_loss /= test_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_size,
        100. * correct / test_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Configure tf
    tf.set_random_seed(args.seed)

    # Load data
    train_loader, test_loader, train_size, test_size = load_data(args.batch_size, args.test_batch_size)

    # Build model
    model = Net()
    optimizer = tf.train.MomentumOptimizer(args.lr, args.momentum)

    # Train & test model
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, epoch, train_loader, args.batch_size, train_size, log_interval=args.log_interval)
        test(model, test_size, test_loader)

    # Save model for future use
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    global_step = tf.train.get_or_create_global_step()
    all_variables = (model.variables + [global_step])
    tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)
