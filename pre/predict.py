
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model('diplomaORnot.h5')

## 實測
def main():
    img = cv2.imread('A.jpg')

    img = np.resize(img, (300,300,3))

    img = img.reshape(1,300,300,3)

    res = model.predict(img)

    if res[0][1] > 0.7:
        res = 1
    else:
        res =0

    print(res)

if __name__ == '__main__':
    main()
