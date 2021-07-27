import os
import shutil
import cv2
 
 
path = r"StampLabel"
img_output = "./stampTrain/full_image/"
label_output = "./stampTrain/full_label/"
 
dirs = os.listdir(path)
# (1920/1440)/8
for dir_name in dirs:
    filenames = os.listdir(os.path.join(path, dir_name))
    for filename in filenames:
        if filename == "img.png":
            img = cv2.imread(os.path.join(path, dir_name, filename), 0)
            #img = cv2.resize(img, (240, 180))
            cv2.imwrite(os.path.join(img_output, dir_name+".png"), img)
 
        if filename == "label.png":
            img = cv2.imread(os.path.join(path, dir_name, filename), 0)
           # img = cv2.resize(img, (240, 180))
            binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
            cv2.imwrite(os.path.join(label_output, dir_name+".png"), binary)
 
        print("Generate "+os.path.join(img_output, dir_name+".png"))

