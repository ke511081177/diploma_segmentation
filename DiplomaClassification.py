
from tensorflow import keras
import cv2
import numpy as np 
import os
import jieba
import re

from loss import*
from func import*
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="subtle-fulcrum-319206-415ab8f59c71.json"

from google.cloud import vision

client = vision.ImageAnnotatorClient()


school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']


class diplomaClassifier():
    def __init__(self):
        
        self.stampModel = keras.models.load_model('unet_stamp_rgb_300_15_binary.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
        self.preModel = keras.models.load_model('diplomaORnot.h5')
        self.text = []
        self.tempschool = ''
        #self.folderlist = ['unsorted']
        self.folderpath = './classify'
        self.folderlist = os.listdir(self.folderpath)
        #load 自建自典
        jieba.load_userdict('./jiabaDictionary/school.txt')
        jieba.load_userdict('./jiabaDictionary/department.txt')



    def checkDiploma(self):
        temp = self.img
        temp = cv2.resize(temp, (300,300))
        
        temp = temp.reshape(1,300,300,3)

        temp = self.preModel.predict(temp)
        
        if temp[0][1] > 0.7:
            temp = 1
        else:
            temp = 0

        return temp

        


    def run(self,file):
        #try:
        import time
        self.text = []
        self.tempschool = ''

        path = './image'
        self.img = cv2.imread('{}/{}'.format(path,file))
        self.tempImg = self.img
        
        file = re.sub('[.jpg]', '', file)

        #篩掉非畢業證書
        temp = self.checkDiploma()

        if temp  == 0:
            
            self.Unqualified('notdiploma',file)
            # print('not: ',file)
            # os.makedirs('./{}/notdiploma/{}'.format(self.folderpath,file))
            # cv2.imwrite('{}/notdiploma/{}/{}'.format(self.folderpath,file,file),self.tempImg)
            
            return 


        self.scanText(file)

        # 偵測不到文字
        if len(self.texts) == 0:
            self.Unqualified(file)
            return 

        self.texts = self.texts[0].description
        self.texts = self.texts.replace("\n","").strip()
        self.texts = self.texts.replace(" ","").strip()
        if '證書' not in self.texts and '學位' not in self.texts:
            self.Unqualified('ungraduate',file)
            return
        temp = jieba.cut(self.texts)

        for i in temp:
            self.text.append(i)

        buffer = []

        for c in school:
            for i in self.text:
                if c in i:
                    buffer.append(i)

        #有偵測到大學
        if len(buffer) != 0:
            for i in buffer:
                if len(self.tempschool) < len(i):
                    self.tempschool = i
            
            #查無大學
            if len(self.tempschool) < 4:
                self.Unqualified('unsorted',file)
                return
            
            self.qualified(file)
        # except:

        #     print(file)
        #     with open('error.txt', 'w',encoding="utf-8") as f:
        #         f.writelines(file) 
                

    def scanText(self,file):
        

        success, encoded_image = cv2.imencode('.jpg', self.img)

        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        self.texts = response.text_annotations

    def qualified(self,file):
        print('School: ',self.tempschool)

        if self.tempschool not in self.folderlist:
            os.makedirs('./{}/{}'.format(self.folderpath, self.tempschool))
            self.folderlist.append(self.tempschool)

        self.stampCut()
        os.makedirs('./{}/{}/{}'.format(self.folderpath, self.tempschool, file))
        
        # 中文路徑要解碼
        cv2.imencode('.jpg',self.tempImg)[1].tofile('{}/{}/{}/{}.jpg'.format(self.folderpath, self.tempschool, file, file)) 
        
        cv2.imencode('.jpg',self.img)[1].tofile('{}/{}/{}/{}_stamp.jpg'.format(self.folderpath, self.tempschool, file, file))

        with open('{}/{}/{}/{}_text.txt'.format(self.folderpath,self.tempschool,file,file), 'w',encoding="utf-8") as f:
            f.write(self.texts) 

    def Unqualified(self,folder,file):
        print('school: Not Found')

        os.makedirs('./{}/{}/{}'.format(self.folderpath,folder,file))
        cv2.imwrite('{}/{}/{}/{}.jpg'.format(self.folderpath,folder,file,file),self.tempImg)

    def stampCut(self):
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)

        rimg = cv2.resize(self.img, (256, 256))

        binary = np.reshape(rimg,(1,)+rimg.shape)
        
        mask = self.stampModel.predict(binary)
        
        mask = np.stack((mask,)*3, axis=-1)
        
        mask = np.reshape(mask,(256,256,3))
        
        mask = cv2.resize(mask,(self.img.shape[1],self.img.shape[0]))
        
        mask = mask[:,:,0]

        mask = np.uint8(mask)

        self.img = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)

        self.img = cv2.cvtColor(self.img,cv2.COLOR_RGB2BGR)

        squares = find_squares(self.img)

        if squares == 0:
            print('----ERROR------')
            return 0

        # 截印章位置
        self.img = self.tempImg[min(squares[0][0][1],squares[0][2][1]):max(squares[0][0][1],squares[0][2][1]),
                    min( squares[0][2][0],squares[0][0][0]):max(squares[0][2][0],squares[0][0][0])]

if __name__ == '__main__':
    classifiler = diplomaClassifier()

    jieba.case_sensitive = True

    classifiler.run('0.jpg')
    classifiler.run('329.jpg')
    classifiler.run('295.jpg')
    classifiler.run('110a.jpg')
    



    

    # for file in filelist:
    #     classifiler.img  = cv2.imread('{}/{}'.format(path,file))
    #     classifiler.run(file)

