import numpy as np
import scipy.signal as sg

from random import randint

CHUNK_SIZE = 1800


def copy_and_convert(func):

    def wrapper(arr):
        arr = arr.copy()

        if np.max(arr) <= 1:
            arr *= 255

        arr = func(arr)

        if np.max(arr) > 1:
            arr = normalize(arr)

        return arr

    return wrapper


@copy_and_convert
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
    arr = arr.copy()
    arr = (arr - np.min(arr)) * 1 / (np.max(arr) - np.min(arr))
    return arr


def sobel(datum):
    kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    kernel_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])

    dy = sg.convolve2d(datum, kernel_y, boundary='wrap', mode='same')
    dx = sg.convolve2d(datum, kernel_x, boundary='wrap', mode='same')

    dist = (dy ** 2 + dx ** 2) ** 0.5
    dist[dist > 255] = 255
    return dist


@copy_and_convert
def apply_sobel(data):
    for datum in data:
        datum[:] = sobel(datum)

    return data


class DataBuilder:

    def __init__(self):
        self.X = np.load("dataset/X.npy")
        self.Y = np.load("dataset/Y.npy")

        self.n_modifs = 10

    def _combine(self, *args, Y_chunk=None):
        n = len(args)

        X_chunk = np.concatenate(args, axis=0)
        Y_chunk = np.concatenate((Y_chunk,) * n, axis=0)

        return X_chunk, Y_chunk

    def _modify(self, X_chunk, Y_chunk):
        real = X_chunk
        convolved = apply_sobel(real)

        noised = gaussian_noise(real)
        noised_convolved = apply_sobel(gaussian_noise(real))

        real = normalize(real)

        flipped = real[:, :, ::-1]
        flipped_convolved = convolved[:, :, ::-1]

        l_blocked = real.copy()
        r_blocked = real.copy()
        lc_blocked = convolved.copy()
        rc_blocked = convolved.copy()

        l_blocked[:, :, :20] = 0
        r_blocked[:, :, -20:] = 0
        lc_blocked[:, :, :20] = 0
        rc_blocked[:, :, -20:] = 0

        return self._combine(
            real,
            convolved,
            noised,
            noised_convolved,
            flipped,
            flipped_convolved,
            l_blocked,
            r_blocked,
            lc_blocked,
            rc_blocked,
            Y_chunk=Y_chunk
        )

    def build(self):
        shuffled_idx = np.arange(len(self.X))
        np.random.shuffle(shuffled_idx)

        self.X = self.X[shuffled_idx]
        self.Y = self.Y[shuffled_idx]

        n_chunks = len(self.X) * self.n_modifs // CHUNK_SIZE
        X_chunks = np.array_split(self.X, n_chunks)
        Y_chunks = np.array_split(self.Y, n_chunks)

        for i in range(len(X_chunks)):
            X_chunk, Y_chunk = self._modify(X_chunks[i], Y_chunks[i])
            np.save("dataset/X_data" + str(i), X_chunk)
            np.save("dataset/Y_data" + str(i), Y_chunk)


def data_generator(train_level=9, batch_size=128):
    while True:
        rand = randint(0, train_level)
        X = np.load("dataset/X_data" + str(rand) + ".npy")
        Y = np.load("dataset/Y_data" + str(rand) + ".npy")

        X = X.reshape(X.shape + (1,))

        idxs = np.arange(len(X))
        np.random.shuffle(idxs)

        for batch_idx in np.array_split(idxs, len(X) // batch_size):
            yield (X[batch_idx], Y[batch_idx])


if __name__ == '__main__':
    dbuilder = DataBuilder()
    dbuilder.build()
