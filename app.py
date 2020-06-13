import tensorflow as tf
from pathlib import Path
import cv2
from preprocessing.translate import translate
import matplotlib.pyplot as plt
plt.switch_backend('tkaggg')

# Convert prediction
categories = ['cane', 'cavallo', 'elefante', 'farfalla',
              'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
predicts = []

for tkey, tvalue in translate.items():
    for ccategory in categories:
        if ccategory == tkey:
            predicts.append(tvalue)
        else:
            pass


def prepare(filepath):
    IMG_SIZE = 57
    img_array = cv2.imread(file, CV2.IMREAD_COLOR)
    norm_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return norm_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


def image(array):
    plt.imshow(img_array)
    plt.show()


def predict():
    # Load Model
    model = tf.keras.models.load_model("Nadam-10E-65.5P")
    pic = input("picture to predict:")
    prepare = prepare(pic)
    prediction = model.predict([prepare])
    print(predicts[int(predict[0][0])])
    image(prepare)


predict()
