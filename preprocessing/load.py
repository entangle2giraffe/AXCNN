import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import chdir, listdir
import cv2
from tqdm import tqdm
import random
import pickle
# Import translate for categories
from translate import translate

plt.switch_backend('tkagg')

# Directories
prep_path = Path("/home/giraffe/Desktop/Animal_CNN/preprocessing")
chdir(prep_path.parent)
data_path = Path.cwd() / 'dataset/raw-img'

categories = list(translate.keys())
labels = list(translate.values())


# Create dataset
def create_dataset():
    for category in categories:
        path = data_path / category
        # Iterate over each image
        for img in tqdm(listdir(path)):
            img_array = cv2.imread(str(path / img), cv2.IMREAD_COLOR)
            img_array = cv2.resize(img_array, (57, 57))
