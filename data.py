import numpy as np
import abc


class ModifierFramework(metaclass=abc.ABCMeta):

    default_probability = 0.1

    def __init__(self, probability=None):
        if probability is None:
            self.probability = self.default_probability
        else:
            self.probability = probability

    def do(self, data):
        mask = self.select(data)
        data[mask] = self.modify(data[mask])

    def select(self, data):
        n_data = data.shape[0]
        return np.random.random(n_data) < self.probability

    @abc.abstractmethod
    def modify(self, data):
        pass


class Blocker(ModifierFramework):

    default_probability = 0.2

    def modify(self, data):
        n_data = data.shape[0]
        blocker_type = np.random.random(n_data)
        blocker_range = np.random.randint(0, 20, n_data)

        mask = blocker_type < 0.3
        data[mask, blocker_range[mask]:] = 0

        mask = (blocker_type >= 0.3) & (blocker_type < 0.6)
        data[mask, :blocker_range[mask]] = 0

        mask = (blocker_type >= 0.6) & (blocker_type < 0.8)
        data[mask, :, blocker_range[mask]:] = 0

        mask = (blocker_type >= 0.8) & (blocker_type < 1)
        data[mask, :, :blocker_range[mask]] = 0

        return data


class PixelKiller(ModifierFramework):

    th = 0.1

    def modify(self, data):
        mask = np.random.random(data.shape)
        data[mask] = 0
        return data


class Rotator(ModifierFramework):

    def modify(self, data):
        n_data = data.shape[0]
        rotate_type = np.random.randint(0, 4, n_data)

        for i in range(4):
            mask = rotate_type >= i
            data[mask] = np.rot90(data[mask])

        return data


class RandomModifier:

    def __init__(self, generator):
        self.generator = generator
        self.modifiers = [Blocker(), PixelKiller(), Rotator()]

    def __next__(self):
        data = next(self.generator)
        for modifier in self.modifiers:
            data = self.modifiers.do(data)

        return data


@RandomModifier
def data_generator(batch_size=128):
    X = np.load("dataset/X_train.npy")
    Y = np.load("dataset/Y_train.npy")

    idxs = np.arange(len(X))

    while True:
        np.random.shuffle(idxs)
        X = X[idxs[:batch_size]].copy()
        Y = Y[idxs[:batch_size]].copy()

        yield X, Y
