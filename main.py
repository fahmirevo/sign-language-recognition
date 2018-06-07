from PIL import Image
import tensorflow as tf
import numpy as np
import scipy.signal as sg

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

INPUTS = 4096
HIDDEN_1 = 1024
HIDDEN_2 = 1024
OUTPUTS = 10

EPOCHS = 10000
DISPLAY_EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.001


def gaussian_noise(arr, th=10):
    rand = np.random.randint(0, 100, size=arr.shape)
    nrand = np.random.randint(-128, 128, size=arr.shape)
    mask = rand < th

    arr[mask] += nrand[mask]

    mask = arr > 255
    arr[mask] = 255

    mask = arr < 0
    arr[mask] = 0

    return arr


def normalize(arr):
        arr = (arr - np.min(arr)) * (np.max(arr) - np.min(arr))
        return arr


def display_datum(datum):
    arr = datum * 255
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    im.show()


def display_arr(arr):
    im = Image.fromarray(arr.astype(np.uint8))
    im.show()


def sobel(datum):
    kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    kernel_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])

    dy = sg.convolve2d(datum, kernel_y, boundary='wrap', mode='same')
    dx = sg.convolve2d(datum, kernel_x, boundary='wrap', mode='same')

    dist = (dy ** 2 + dx ** 2) ** 0.5
    dist[dist > 255] = 255
    return dist


def apply_sobel(data):
    for datum in data:
        datum[:] = sobel(datum)

    return data


class Data:

    _batch_counter = 0
    _train_idx = None

    def __init__(self):
        self._build_dataset()
        self._set_train_idx()

    def _build_dataset(self):
        # real = np.load("dataset/X.npy").reshape((-1, INPUTS))
        convolved = np.load("dataset/convolved.npy").reshape(-1, INPUTS)
        # convolved = np.load("dataset/convolved.npy")
        Y = np.load("dataset/Y.npy")

        flipped = convolved[:, ::-1]

        # real_noise = gaussian_noise(real.copy() * 255)
        convolved_noise = gaussian_noise(convolved.copy())
        flipped_noise = gaussian_noise(flipped.copy())

        # real = normalize(real)
        convolved = normalize(convolved)
        # real_noise = normalize(real_noise)
        convolved_noise = normalize(convolved_noise)
        flipped_noise = normalize(flipped_noise)

        # X = (real, convolved, real_noise, convolved_noise)
        X = (convolved, convolved_noise, flipped, flipped_noise)
        self.X = np.concatenate(X, axis=0)
        self.Y = np.concatenate((Y,) * 4, axis=0)

    @property
    def train(self):
        return self.X[self._train_idx], self.Y[self._train_idx]

    @property
    def test(self):
        idx = np.arange(len(self.X))
        mask = np.isin(idx, self._train_idx)

        return self.X[~mask], self.Y[~mask]

    def _set_train_idx(self):
        size = int(len(self.Y) * 0.75)
        self._train_idx = np.random.randint(0, len(self.Y), size)

    def next_batch(self, batch_size):
        max_length = len(self._train_idx)

        if self._batch_counter >= max_length:
            np.random.shuffle(self._train_idx)
            idx = self._train_idx[:batch_size]
            self._batch_counter = batch_size
        elif self._batch_counter + batch_size > max_length:
            idx = self._train_idx[self._batch_counter:]
            self._batch_counter = max_length
        else:
            idx = self._train_idx[self._batch_counter:self._batch_counter + batch_size]
            self._batch_counter += batch_size

        return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    data = Data()

    x_data = tf.placeholder(dtype=tf.float32,shape=[None, INPUTS],name="input")
    y_data = tf.placeholder(dtype=tf.float32,shape=[None, OUTPUTS],name="output")

    # Weights - Input to Hidden 1
    weight1 = tf.random_normal([INPUTS, HIDDEN_1])
    weight1 = tf.Variable(weight1, name='W1')

    bias1 = tf.random_normal([HIDDEN_1])
    bias1 = tf.Variable(bias1, name='B1')

    # Weights - Hidden 1 to Hidden 2
    weight2 = tf.random_normal([HIDDEN_1, HIDDEN_2])
    weight2 = tf.Variable(weight2, name='W2')

    bias2 = tf.random_normal([HIDDEN_2])
    bias2 = tf.Variable(bias2, name='B2')

    # Weights - Hidden 2 to Output
    weight3 = tf.random_normal([HIDDEN_2, OUTPUTS])
    weight3 = tf.Variable(weight3, name='W2')

    bias3 = tf.random_normal([OUTPUTS])
    bias3 = tf.Variable(bias3, name='B2')

    #input to hidden 1
    hidden1 = tf.nn.relu(tf.add(tf.matmul(x_data, weight1), bias1))

    #hidden 1 to hidden 2
    hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weight2), bias2))

    #hidden2 to output
    y = tf.add(tf.matmul(hidden2, weight3), bias3)


    #apply final activation
    result = tf.nn.softmax(y)


    #loss and training
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_data))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session.run(init)

    for step in range(EPOCHS):
        batch_x, batch_y = data.next_batch(BATCH_SIZE)
        out_training , out_loss = session.run([train,loss],feed_dict={x_data: batch_x, y_data: batch_y})

        print("Step: %d error: %g "%(step,out_loss))

    print("complete")
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    x_test, y_test = data.test
    accuracy = session.run(accuracy,feed_dict={x_data: x_test, y_data: y_test})
    print("Accuracy:", "{:.0%}".format(accuracy))
