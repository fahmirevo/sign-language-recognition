import preprocess as pre
from PIL import Image
from keras.models import load_model

import numpy as np


def train_standardize(arr):
    X = np.load("dataset/X_train.npy")

    return (arr - np.mean(X)) / np.std(X)


def get_image(path):
    im = Image.open(path).convert("L")
    im = im.resize((64, 64))
    return im


def clean(arr):
    arr = arr.reshape((1,) + arr.shape)
    arr = pre.sobel(arr)
    arr = train_standardize(arr)
    arr = pre.add_channel(arr)

    return arr


def predict(arr):
    model = load_model("model")
    prob = model.predict(arr)
    label = np.argmax(prob)

    return label, prob


if __name__ == '__main__':
    while True:
        path = input("path : ")

        im = get_image(path)

        arr = np.array(im)
        arr = clean(arr)

        label, prob = predict(arr)

        print("label : " + str(label))
        print("prob : ")
        print(prob)
