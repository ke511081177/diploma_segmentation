from tensorflow import keras
import cv2
import numpy as np 
import os
import jieba.analyse

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="XXXXX.json"   # GCP金鑰
from google.cloud import vision
client = vision.ImageAnnotatorClient()
import jieba
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten
import skimage.io as io
#from matplotlib import pyplot as plt
import re
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

folderlist = ['unsorted']

foolderpath = './classify'

school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']

department = ['系','所','科']

degree = ['學士','碩士','博士']

unsortedCount = 0
smooth = 1. # 用於防止分母爲0.

def dice_coef(y_true, y_pred):
    flatten_layer = Flatten()                           
    y_true_f = flatten_layer(y_true) # 將 y_true 拉伸爲一維.
    y_pred_f = flatten_layer(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255






class diplomaClassifier():
    def __init__(self):
        #self.cutModel = keras.models.load_model('unet_diploma_4.hdf5')
        self.stampModel = keras.models.load_model('unet_stamp_400_5.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
        #self.stampModel = keras.models.load_model('unet_stamp_4.hdf5')
        self.text =[]

    def detect_all_text(self,file):
        global unsortedCount
        path = './image'
        self.img = cv2.imread('{}/{}'.format(path,file))

        #self.backgroundCut()     ## 去背

        """Detects text in the file."""
        tempImg = self.img
        
        file = re.sub('[.jpg]', '', file)
        #print(file)

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
        


        seg = []
        for i in seg_list:
            seg.append(i)

        flag = False

        tempSchool = ''
        

        ## 系所
        flag = False     
        tempDepartment = ''
        for c in department:
        
            if flag == True:
                break

            for i in seg:
                if c in i:
                    tempDepartment = i
                    print('Department: ',i)
                    flag = True
                    break

        ## 學位
        flag = False   
        tempDegree = ''
        for c in degree:  
        
            if flag == True:
                break

            for i in seg:
                if c in i:
                    tempDegree = c
                    print('Degree: ',c)
                    flag = True
                    break

        flag = False

        buffer = []
        for c in school:
        
            if flag == True:
                break
            
            for i in seg:
                
                if c in i:
                    #print(c,i)
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
                os.makedirs('./classify/unsorted/{}'.format(file))
             

                cv2.imwrite('{}/{}/{}/{}.png'.format(foolderpath,tempSchool,file,file),tempImg)
                self.stampCut()
                cv2.imwrite('{}/{}/{}/{}_stamp.png'.format(foolderpath,tempSchool,file,file),self.img)
                f = open('{}/{}/{}/{}_info.txt'.format(foolderpath,tempSchool,file,file), 'w')
                
                f.writelines('{}\n{}\n{}'.format(tempSchool,tempDepartment,tempDegree))
                unsortedCount = unsortedCount+1
            else:
                tempschool = c
                if tempschool not in folderlist:
                    os.makedirs('./classify/{}'.format(tempschool))
                    folderlist.append(tempschool)
                    

                print('School: ',tempschool)

                

                self.stampCut()

                # 中文路徑要解碼
                cv2.imencode('.jpg',tempImg)[1].tofile('classify/{}/{}.jpg'.format(tempschool,file)) 
                cv2.imencode('.jpg',self.img)[1].tofile('classify/{}/{}_stamp.jpg'.format(tempschool,file))


                f = open('./classify/{}/{}_info.txt'.format( tempschool,file), 'w')
                #print('./classify/{}/{}_info.txt'.format( tempschool,file))
                f.writelines('{}\n{}\n{}'.format(tempSchool,tempDepartment,tempDegree))
                
                f.close
            flag = True
        
    def backgroundCut(self):
        

        rimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        rimg = cv2.resize(rimg, (256, 256))

        #binary = cv2.threshold(rimg, 0, 255, cv2.THRESH_OTSU)[1]

        binary = np.reshape(rimg,rimg.shape+(1,))

        binary = np.reshape(binary,(1,)+binary.shape)
        
        mask = self.cutModel.predict(binary)
        
        
        
        mask = np.stack((mask,)*3, axis=-1)
        
        mask = np.reshape(mask,(256,256,3))

        
        mask = cv2.resize(mask,(self.img.shape[1],self.img.shape[0]))
        
        mask = mask[:,:,0]
    
        #res,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

        mask = np.uint8(mask)
    
        
        self.img = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        cv2.imwrite('output.jpg',self.img)
    
    def stampCut(self):
        
        rimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        rimg = cv2.resize(rimg, (256, 256))

        #binary = cv2.threshold(rimg, 0, 255, cv2.THRESH_OTSU)[1]

        binary = np.reshape(rimg,rimg.shape+(1,))

        binary = np.reshape(binary,(1,)+binary.shape)
        
        mask = self.stampModel.predict(binary)
        mask = np.reshape(mask,(256,256,1))
        # print(mask.shape)
        # io.imsave(os.path.join('./aa.png'),mask)
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  
        # cv2.imshow('aa',mask)
        # cv2.waitKey()
        # io.imsave(os.path.join('./aaa.jpg'),mask)
        
        mask = np.stack((mask,)*3, axis=-1)
        
        mask = np.reshape(mask,(256,256,3))

        
        mask = cv2.resize(mask,(self.img.shape[1],self.img.shape[0]))
        

        #h, w, _ = mask.shape  
  
        # gray = cv2.cvtColor( mask, cv2.COLOR_BGR2GRAY)  
        
        # ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  
        
        # # Find Contour  
        # thresh = np.uint8(thresh)
        # _, contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
  
 
        # c_max = []
        # max_area = 0
        # max_cnt = 0
        # for i in range(len(contours)):
        #     cnt = contours[i]
        #     area = cv2.contourArea(cnt)
        #     # find max countour
        #     if (area>max_area):
        #         if(max_area!=0):
        #             c_min = []
        #             c_min.append(max_cnt)
        #             cv2.drawContours( mask, c_min, -1, (0,0,0), cv2.FILLED)
        #         max_area = area
        #         max_cnt = cnt
        #     else:
        #         c_min = []
        #         c_min.append(cnt)
        #         cv2.drawContours( mask, c_min, -1, (0,0,0), cv2.FILLED)

        # c_max.append(max_cnt)
        # # cv2.imshow('A',mask)
        
        # if len(c_max)>1:
        #     cv2.drawContours( mask, c_max, -1, (255, 255, 255), thickness=-1)  
        #     # cv2.imshow('B',mask)
            
        mask = mask[:,:,0]
        #cv2.waitKey(0)

        mask = np.uint8(mask)
       # cv2.imwrite('output.jpg',mask)
        #cv2.waitKey(0)
        self.img = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        #cv2.imshow('aa',self.img)
        #cv2.waitKey()
        #cv2.imwrite('output.jpg',self.img)

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
        
        
        self.text = text.replace("\n"," ").strip()

        school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']
        department = ['系','所','科']
        degree = ['學士','碩士','博士']

        
        jieba.load_userdict('./jiabaDictionary/school.txt')
        jieba.load_userdict('./jiabaDictionary/department.txt')
        jieba.load_userdict('./jiabaDictionary/diplomaNumber.txt')
        print(self.text)
        

        seg_list = jieba.cut(self.text)
        # for seg in seg_list:
        #     print(seg)

    
    def run(self,img):

        self.img = cv2.imread(img)
        #self.backgroundCut()
        #self.stampCut()
        self.detect_text()
        
    
jieba.case_sensitive = True

if __name__ == '__main__':
    classifiler = diplomaClassifier()
    #classifiler.run('./image/57.jpg')

    path = './image'
    filelist = os.listdir(path)
    jieba.case_sensitive = True
    for file in filelist:
        classifiler.img  = cv2.imread('{}/{}'.format(path,file))
        #classifiler.backgroundCut()
        #classifiler.stampCut()
        classifiler.detect_all_text(file)
