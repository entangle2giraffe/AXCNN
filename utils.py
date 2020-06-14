import cv2
import pickle
import matplotlib.pyplot as plt
from preprocessing.translate import translate
import numpy as np


def pload():
    pickle_in = open('preprocessing/X_train.pickle', 'rb')
    X_train = pickle.load(pickle_in)

    pickle_in = open('preprocessing/y_train.pickle', 'rb')
    y_train = pickle.load(pickle_in)

    pickle_in = open('preprocessing/X_val.pickle', 'rb')
    X_val = pickle.load(pickle_in)

    pickle_in = open('preprocessing/y_val.pickle', 'rb')
    y_val = pickle.load(pickle_in)

    # Normalize the image array
    X_train, X_val = (X_train/255, X_val/255)

    IMG_SIZE = 57

    # Turn it into numpy array
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y_train = np.array(y_train)

    X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y_val = np.array(y_val)

    return (X_train, y_train, X_val, y_val)


def clist():
    categories = ['cane', 'cavallo', 'elefante', 'farfalla',
                  'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
    predicts = []

    for ccategory in categories:
        for tkey, tvalue in translate.items():
            if ccategory == tkey:
                predicts.append(tvalue)

    predicts.insert(-1, 'spider')

    return predicts


def prepare(filepath):
    IMG_SIZE = 57
    img_array = cv2.imread(file, CV2.IMREAD_COLOR)
    norm_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return norm_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


def image(filepath):
    plt.switch_backend('tkagg')
    plt.imshow(img_array)
    plt.show()


if __name__ == 'utils':
    pload()
    clist()
    prepare()
    image()
