import pickle
import numpy as np

def load():
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

if __name__ == 'pickle_load':
  load()