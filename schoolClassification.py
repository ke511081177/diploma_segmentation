
from tensorflow import keras
import cv2
import numpy as np 
import os


from loss import*
from func import*

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="subtle-fulcrum-319206-415ab8f59c71.json"

from google.cloud import vision
client = vision.ImageAnnotatorClient()
import jieba


import re


folderlist = ['unsorted']

folderpath = './classify'

school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']





class diplomaClassifier():
    def __init__(self):

        self.stampModel = keras.models.load_model('unet_stamp_rgb_300_15_binary.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
        self.text =[]

    def run(self,file):

        path = './image'
        self.img = cv2.imread('{}/{}'.format(path,file))
        self.imgSave = self.img

        """Detects text in the file."""
        tempImg = self.img
        
        file = re.sub('[.jpg]', '', file)
 

        success, encoded_image = cv2.imencode('.jpg', self.img)

        content = encoded_image.tobytes()
        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        text = texts[0].description
        

        text = text.replace("\n"," ").strip()

        jieba.load_userdict('./jiabaDictionary/school.txt')
        jieba.load_userdict('./jiabaDictionary/department.txt')

        seg_list = jieba.cut(text)
        


       
        for i in seg_list:
            self.text.append(i)



        tempSchool = ''
        



   
        buffer = []
        for c in school:
            for i in self.text:
                if c in i:
                    buffer.append(i)

        #print(buffer)

        if len(buffer) != 0:
            
            
            c = ''
            for i in buffer:
                if len(c) < len(i):
                    c = i
            
            if len(buffer) == 0 or len(c) < 4:
                print('school: Not Found')

                tempSchool = 'unsorted'
                os.makedirs('./{}/unsorted/{}'.format(folderpath,file))
             

                cv2.imwrite('{}/{}/{}/{}.jpg'.format(folderpath,tempSchool,file,file),tempImg)
                self.stampCut()
                squares = find_squares(self.img)
                
                if squares == 0:
                    print(file)
                    print('----ERROR------')
                    return 0

                self.img = self.imgSave[min(squares[0][0][1],squares[0][2][1]):max(squares[0][0][1],squares[0][2][1]),
                            min( squares[0][2][0],squares[0][0][0]):max(squares[0][2][0],squares[0][0][0])]

                cv2.imwrite('{}/{}/{}/{}_stamp.jpg'.format(folderpath,tempSchool,file,file),self.img)


            else:
                tempschool = c
                if tempschool not in folderlist:
                    os.makedirs('./{}/{}'.format(folderpath,tempschool))
                    folderlist.append(tempschool)
                    

                print('School: ',tempschool)

                

                self.stampCut()

                # 中文路徑要解碼
                cv2.imencode('.jpg',tempImg)[1].tofile('classify/{}/{}.jpg'.format(tempschool,file)) 
                squares = find_squares(self.img)

                if squares == 0:
                    print(file)
                    print('----ERROR------')
                    return 0

                self.img = self.imgSave[min(squares[0][0][1],squares[0][2][1]):max(squares[0][0][1],squares[0][2][1]),
                            min( squares[0][2][0],squares[0][0][0]):max(squares[0][2][0],squares[0][0][0])]
                cv2.imencode('.jpg',self.img)[1].tofile('{}/{}/{}_stamp.jpg'.format(folderpath,tempschool,file))



        
    
    
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
    






        
    
jieba.case_sensitive = True

if __name__ == '__main__':
    classifiler = diplomaClassifier()

    path = './image'
    filelist = os.listdir(path)

    jieba.case_sensitive = True

    for file in filelist:
        classifiler.img  = cv2.imread('{}/{}'.format(path,file))
        classifiler.run(file)
        
