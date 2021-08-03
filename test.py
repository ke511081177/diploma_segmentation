
import cv2
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="subtle-fulcrum-319206-415ab8f59c71.json"
from google.cloud import vision
client = vision.ImageAnnotatorClient()
import jieba
from tensorflow import keras
import tensorflow as tf
import numpy as np 


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def textDetect():
    jieba.load_userdict('./jiabaDictionary/school.txt')
    jieba.load_userdict('./jiabaDictionary/department.txt')
    jieba.load_userdict('./jiabaDictionary/diplomaNumber.txt')
    img = cv2.imread('20_stamp.jpg')
    success, encoded_image = cv2.imencode('.jpg', img)

    content = encoded_image.tobytes()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    text = texts[0].description


    text = text.replace("\n"," ").strip()

    school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']
    department = ['系','所','科']
    degree = ['學士','碩士','博士']

def Dornot():
    model = keras.models.load_model('diplomaORnot.h5',custom_objects={'Functional':tf.keras.models.Model})
    img  = cv2.imread('./image/10.jpg')
    img = cv2.resize(img, (300, 300))

    img = img.reshape(1,300,300,3)

    res = model.predict(img)
    
    if res[0][1] > 0.7:
        res = 1
    else:
        res =0
    return res

if __name__ == "__main__":
    print(Dornot())
