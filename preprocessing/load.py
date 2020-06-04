import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import chdir, listdir
import cv2
from tqdm import tqdm
import random
import pickle

# Directories
prep_path = Path("/home/giraffe/Desktop/Animal_CNN/preprocessing")
chdir(prep_path.parent)
data_path = Path.cwd() / 'dataset'
train_path, val_path = (data_path / 'train', data_path / 'val')


categories = ['cane', 'cavallo', 'elefante', 'farfalla',
              'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# Image size
IMG_SIZE = 57

# Empty list
train_data = []
val_data = []

# Create dataset


def create_train():
    for category in categories:
        path = train_path / category
        classes = categories.index(category)
        # Iterate over each image
        for img in tqdm(listdir(path)):
            img_array = cv2.imread(str(path / img), cv2.IMREAD_COLOR)
            norm_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_data.append([norm_array, classes])


def create_val():
    for category in categories:
        path = val_path / str(category)
        classes = categories.index(category)
        # Iterate over each image
        for img in tqdm(listdir(path)):
            img_array = cv2.imread(str(path / img), cv2.IMREAD_COLOR)
            norm_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            val_data.append([norm_array, classes])


create_train()
create_val()

print("Size of train samples ", len(train_data))
print("Size of validation samples ", len(val_data))

# Shuffle dataset
random.shuffle(train_data)
random.shuffle(val_data)

X_train = []
y_train = []
X_val = []
y_val = []

for features, label in train_data:
    X_train.append(features)
    y_train.append(label)

for features, label in val_data:
    X_val.append(features)
    y_val.append(label)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Save x,y as .pickle
pickle_out = open('preprocessing/X_train.pickle', 'wb')
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open('preprocessing/y_train.pickle', 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open('preprocessing/X_val.pickle', 'wb')
pickle.dump(X_val, pickle_out)
pickle_out.close()

pickle_out = open('preprocessing/y_val.pickle', 'wb')
pickle.dump(y_val, pickle_out)
pickle_out.close()
