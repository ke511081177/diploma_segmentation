from tensorflow import keras
import cv2
import numpy as np 
import os
import jieba.analyse

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="subtle-fulcrum-319206-415ab8f59c71.json"
from google.cloud import vision
#import io
import skimage.io as io
client = vision.ImageAnnotatorClient()

import jieba
from matplotlib import pyplot as plt


class diplomaClassifier():
    def __init__(self):
        self.cutModel = keras.models.load_model('unet_diploma_4.hdf5')
        self.stampModel = keras.models.load_model('unet_stamp_4.hdf5')
        self.text =[]
    def backgroundCut(self):
        

        rimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        rimg = cv2.resize(rimg, (256, 256))

        #binary = cv2.threshold(rimg, 0, 255, cv2.THRESH_OTSU)[1]

        binary = np.reshape(rimg,rimg.shape+(1,))

        binary = np.reshape(binary,(1,)+binary.shape)
        
        mask = self.cutModel.predict(binary)
        
        
        
        mask = np.stack((mask,)*3, axis=-1)
        
        mask = np.reshape(mask,(256,256,3))
        print(mask)
        print(self.img)
        
        mask = cv2.resize(mask,(self.img.shape[1],self.img.shape[0]))
        
        mask = mask[:,:,0]
    
        #res,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

        mask = np.uint8(mask)
    
        #cv2.waitKey(0)
        self.img = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        cv2.imwrite('output.jpg',self.img)
    
    def stampCut(self):
        
        rimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        rimg = cv2.resize(rimg, (256, 256))

        #binary = cv2.threshold(rimg, 0, 255, cv2.THRESH_OTSU)[1]

        binary = np.reshape(rimg,rimg.shape+(1,))

        binary = np.reshape(binary,(1,)+binary.shape)
        
        mask = self.stampModel.predict(binary)

        
        
        mask = np.stack((mask,)*3, axis=-1)
        
        
        mask = np.reshape(mask,(256,256,3))
        _,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
        
        # print(mask)
        # print(self.img)
        
        mask = cv2.resize(mask,(self.img.shape[1],self.img.shape[0]))
        
        mask = mask[:,:,0]
    
        #res,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        cv2.imwrite('mask.png',mask)
        mask = np.uint8(mask)
        
        #cv2.waitKey(0)
        self.img = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        cv2.imwrite('outpuataaaa.jpg',self.img)

    def detect_text(self):

        """Detects text in the file."""


        # with io.open(path, 'rb') as image_file:
        #     content = image_file.read()
        success, encoded_image = cv2.imencode('.jpg', self.img)

        content = encoded_image.tobytes()
        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        text = texts[0].description
        
        #self.text = text.splitlines()
        self.text = text.replace("\n"," ").strip()

        school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']
        department = ['系','所','科']
        degree = ['學士','碩士','博士']

        #print(len(classifiler.text)
        jieba.load_userdict('./jiabaDictionary/school.txt')
        jieba.load_userdict('./jiabaDictionary/department.txt')
        jieba.load_userdict('./jiabaDictionary/diplomaNumber.txt')
        print(self.text)
        #print(jieba.analyse.extract_tags(self.text, topK=20, withWeight=False, allowPOS=()))
        seg_list = jieba.cut(self.text)
        for seg in seg_list:
            print(seg)
        # seg_list= str('/'.join(seg_list))
        # print(seg_list)
        # seg_list = str('\n'.join(seg_list))
        # seg_list = seg_list.replace(" ","").strip()
        # seg_list = seg_list.splitlines()
        # #print(seg_list)
        # seg = []
        # for i in seg_list:
        #     seg.append(i)
        # #print(seg)
        # ## 學校
        # flag = False
        # for c in school:
        
        #     if flag == True:
        #         break
        #     for i in seg:
                
        #         if c in i:
        #             if i.find(c)+2 != len(i):
        #                 print('school: Not Found')
        #                 flag = True
        #                 break
        #             print('School: ',i)
        #             flag = True
        #             break

        # ## 系所
        # flag = False     
        # for c in department:
        
        #     if flag == True:
        #         break

        #     for i in seg:
        #         if c in i:
        #             print('Department: ',i)
        #             flag = True
        #             break

        # ## 學位
        # flag = False   
        # for c in degree:  
        
        #     if flag == True:
        #         break

        #     for i in seg:
        #         if c in i:
        #             print('Degree: ',c)
        #             flag = True
        #             break
    
    def run(self,img):

        self.img = cv2.imread(img)
        #self.backgroundCut()
        #self.stampCut()
        self.detect_text()
        
    
jieba.case_sensitive = True

if __name__ == '__main__':
    classifiler = diplomaClassifier()
    classifiler.run('./image/57.jpg')
