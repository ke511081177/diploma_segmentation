# -*- coding: UTF-8 -*-


import os 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key"  ## GCP 金鑰
from google.cloud import vision
client = vision.ImageAnnotatorClient()
import cv2
import jieba
import jieba.analyse
from time import sleep
import codecs
import numpy as np 
import re

folderlist = ['unsorted']

foolderpath = './classify'

school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']

department = ['系','所','科']

degree = ['學士','碩士','博士']




unsortedCount = 0

def detect_text(file):
    global unsortedCount
    path = './image'
    img = cv2.imread('{}/{}'.format(path,file))

    """Detects text in the file."""
    tempImg = img
    
    file = re.sub('[.jpg]', '', file)
    print(file)

    success, encoded_image = cv2.imencode('.jpg', img)

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
                print(c,i)
                buffer.append(i)
    print(buffer)
    if len(buffer) != 0:
        
        
        c = ''
        for i in buffer:
            if len(c) < len(i):
                c = i
        
        if len(buffer) == 0:
            print('school: Not Found')

            tempSchool = 'unsorted'
            os.makedirs('./classify/unsorted/{}'.format(unsortedCount))
            sleep(0.5)

            cv2.imwrite('{}/{}/{}/{}.png'.format(foolderpath,tempSchool,unsortedCount,file),tempImg)
            
            f = open('{}/{}/{}/{}_info.txt'.format(foolderpath,tempSchool,unsortedCount,file), 'w')
            
            f.writelines('{}\n{}\n{}'.format(tempSchool,tempDepartment,tempDegree))
            unsortedCount = unsortedCount+1
        else:
            tempschool = c
            if tempschool not in folderlist:
                os.makedirs('./classify/{}'.format(tempschool))
                folderlist.append(tempschool)
                sleep(0.5)

            print('School: ',tempschool)

            
            cv2.imwrite('./classify/{}/{}.png'.format(tempschool,file),tempImg)
            cv2.imencode('.jpg',tempImg)[1].tofile('classify/{}/{}.jpg'.format(tempschool,file)) 



            f = open('./classify/{}/{}_info.txt'.format( tempschool,file), 'w')
            print('./classify/{}/{}_info.txt'.format( tempschool,file))
            f.writelines('{}\n{}\n{}'.format(tempSchool,tempDepartment,tempDegree))
            
            f.close
        flag = True
        
                                        

    print('done')



if __name__ == '__main__':

    path = './image'
    filelist = os.listdir(path)
    jieba.case_sensitive = True
    detect_text('57.jpg')
