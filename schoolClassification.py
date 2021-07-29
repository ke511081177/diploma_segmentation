
from tensorflow import keras
import cv2
import numpy as np 
import os
import jieba
import re

from loss import*
from func import*


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="subtle-fulcrum-319206-415ab8f59c71.json"

from google.cloud import vision

client = vision.ImageAnnotatorClient()

folderlist = ['unsorted']

folderpath = './classify'

school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']


class diplomaClassifier():
    def __init__(self):
        self.stampModel = keras.models.load_model('unet_stamp_rgb_300_15_binary.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
        self.text = []
        self.tempschool = ''

    def run(self,file):
        self.text = []
        self.tempschool = ''

        path = './image'
        self.img = cv2.imread('{}/{}'.format(path,file))
        
        """Detects text in the file."""
        self.tempImg = self.img
        
        file = re.sub('[.jpg]', '', file)
 
        success, encoded_image = cv2.imencode('.jpg', self.img)

        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        texts = response.text_annotations
        
        text = texts[0].description
        
        text = text.replace("\n","").strip()
        text = text.replace(" ","").strip()
        #print(text)

        jieba.load_userdict('./jiabaDictionary/school.txt')
        jieba.load_userdict('./jiabaDictionary/department.txt')

        seg_list = jieba.cut(text)
       
        for i in seg_list:
            self.text.append(i)

        buffer = []

        for c in school:
            for i in self.text:
                if c in i:
                    buffer.append(i)

        #print(buffer)

        if len(buffer) != 0:
            for i in buffer:
                if len(self.tempschool) < len(i):
                    self.tempschool = i
            
            #查無大學
            if len(self.tempschool) < 4:
                print('school: Not Found')

                os.makedirs('./{}/unsorted/{}'.format(folderpath,file))


                cv2.imwrite('{}/unsorted/{}/{}.jpg'.format(folderpath,file,file),self.tempImg)

                self.stampCut()

                cv2.imwrite('{}/unsorted/{}/{}_stamp.jpg'.format(folderpath,file,file),self.img)

                with open('{}/unsorted/{}/{}_text.txt'.format(folderpath,file,file), 'w',encoding="utf-8") as f:
                    f.write(text)

            else:
                print('School: ',self.tempschool)

                if self.tempschool not in folderlist:
                    os.makedirs('./{}/{}'.format(folderpath,self.tempschool))
                    folderlist.append(self.tempschool)

                self.stampCut()

                # 中文路徑要解碼
                cv2.imencode('.jpg',self.tempImg)[1].tofile('classify/{}/{}.jpg'.format(self.tempschool,file)) 
                
                cv2.imencode('.jpg',self.img)[1].tofile('{}/{}/{}_stamp.jpg'.format(folderpath,self.tempschool,file))

                with open('{}/{}/{}_text.txt'.format(folderpath,self.tempschool,file), 'w',encoding="utf-8") as f:
                    f.write(text) 

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
            print(file)
            print('----ERROR------')
            return 0

        # 截印章位置
        self.img = self.tempImg[min(squares[0][0][1],squares[0][2][1]):max(squares[0][0][1],squares[0][2][1]),
                    min( squares[0][2][0],squares[0][0][0]):max(squares[0][2][0],squares[0][0][0])]

if __name__ == '__main__':
    classifiler = diplomaClassifier()

    path = './image'
    filelist = os.listdir(path)

    jieba.case_sensitive = True

    for file in filelist:
        classifiler.img  = cv2.imread('{}/{}'.format(path,file))
        classifiler.run(file)
