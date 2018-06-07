from PIL import Image
import numpy as np


def display_datum(datum):
    arr = datum * 255
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    im.show()


def display_arr(arr):
    im = Image.fromarray(arr.astype(np.uint8))
    im.show()
