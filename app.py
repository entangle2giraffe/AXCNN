import tensorflow as tf
from pathlib import Path
import cv2
from utils import clist, prepare, image
import matplotlib.pyplot as plt

# Path to the model
path = Path.cwd() / save_model


def main():
    # Load Model
    model = tf.keras.models.load_model(str(path)+"/Nadam-10E-65.5P")
    pic = input("picture to predict:")
    prediction = model.predict([prepare(pic)])
    print(clist[int(predict[0][0])])
    image(pic)


if '__name__' == '__main__':
    main()
