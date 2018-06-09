import numpy as np
import scipy.signal as sg


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


def split_save(X, Y, train_size=0.75):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    splitter = int(len(X) * train_size)
    X_train, Y_train = X[:splitter], Y[:splitter]
    X_test, Y_test = X[splitter:], Y[splitter:]

    np.save("dataset/X_train", X_train)
    np.save("dataset/Y_train", Y_train)
    np.save("dataset/X_test", X_test)
    np.save("dataset/Y_test", Y_test)


if __name__ == '__main__':
    X = np.load("dataset/X.npy")
    Y = np.load("dataset/Y.npy")

    X = sobel(X * 255)
    X = standardize(X)
    X = add_channel(X)

    split_save(X, Y)
