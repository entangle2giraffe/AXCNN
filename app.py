from pathlib import Path
import cv2
from os import chdir, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from utils import clist, prepare
import matplotlib.pyplot as plt

# Path to the model
model_path = Path.cwd() / 'save_model'

clist = clist()



# Load Model
model = tf.keras.models.load_model(str(model_path)+"/Nadam-10E-65.5P")
pic = input("picture to predict:")
chdir(str(Path.cwd() / 'test'))
prediction = model.predict([prepare(str(Path.cwd() / pic))])
print("prediction: ",clist[int(prediction[0][0])])


if '__name__' == '__main__':
    pass
