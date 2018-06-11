import numpy as np
import scipy.signal as sg

TRAIN_SIZE = 0.75


def standardize(data):
    data = (data - np.mean(data)) / np.std(data)
    return data


def sobel(images):
    kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    kernel_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])

    for image in images:
        dy = sg.convolve2d(image, kernel_y, boundary='wrap', mode='same')
        dx = sg.convolve2d(image, kernel_x, boundary='wrap', mode='same')

        dist = (dy ** 2 + dx ** 2) ** 0.5
        dist[dist > 255] = 255

        image[:] = dist

    return images


def add_channel(data):
    return data.reshape(data.shape + (1,))


def clean_save(X, Y, name=""):
    X = standardize(X)
    X = add_channel(X)

    np.save("dataset/X_" + name, X)
    np.save("dataset/Y_" + name, Y)


if __name__ == '__main__':
    X = np.load("dataset/X.npy")
    Y = np.load("dataset/Y.npy")

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    splitter = int(len(X) * TRAIN_SIZE)

    X_train, X_test = X[:splitter], X[splitter:]
    Y_train, Y_test = Y[:splitter], Y[splitter:]

    clean_save(X_train, Y_train, name="train")
    clean_save(X_test, Y_test, name="test")
