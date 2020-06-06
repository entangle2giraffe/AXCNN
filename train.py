import tensorflow as tf
from pickle_load import load
from model import create_model
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import sys
import os
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def LOAD():
    # load in X and y
    X_train, y_train, X_val, y_val = load()

    return (X_train, y_train, X_val, y_val)
def COMPILE():
    
    # Import model from model.py
    model = create_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
      

model = COMPILE()
X_train, y_train, X_val, y_val = LOAD()
# Callbacks
# Comment out if you want to use callbacks
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
#checkpoint_path = "training_1/cp.ckpt"
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  #save_weights_only=True,
                                                  #verbose=1, period=10)
# After validation_data add the following:
# ,callbacks=[tensorboard_callback, cp_callback]
Epochs = input("epochs: ")                       
history = model.fit(X_train, y_train,
                    epochs=int(Epochs),
                    validation_data=(X_val, y_val))
  
def SAVE():
  if '-h5' in sys.argv:
    current_dir = Path.cwd()
    save_dir = 'save_model/my_model.h5'
    model.save(save_dir)
    print("The model has been saved")
    print(f"At {current_dir / save_dir}")
  elif '-s' in sys.argv:
    current_dir = Path.cwd()
    save_dir = 'save_model/my_model'
    model.save(save_dir)
    print("The model has been saved")
    print(f"At {current_dir / save_dir}")
  else:
    pass

def EVALUATE():
  if '-e' in sys.argv:
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_val,  y_val, verbose=2)

    plt.show()
  else:
    pass

if __name__ == '__main__':
  LOAD()
  EVALUATE()
  SAVE()