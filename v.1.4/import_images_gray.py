import os

import cv2
#import pyheif
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np

import winsound
import shutil

register_heif_opener()

# Set path to photos

#Ruta donde esta la foto
path = "C:\\Users\\marco.lanese\\Documents\\Temp\\"
#Ruta donde se guarda la foto recortada y gris.
output_path = "C:\\Users\\marco.lanese\\Documents\\Temp\\output\\"
# Commented for thios to work in REPL
script_path = os.path.dirname(__file__) + "\\"
print(script_path)

# Load the cascade
casc_path = script_path + "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(casc_path)


def crop_img(raw_img, face_classifier, name, number, output_path):
    img = cv2.imread(raw_img)
    # Detect the face
    face = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)
    for x, y, w, h in face:
        face = img[y : y + h, x : x + w]
        gray_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (30,30))
        number = str(number)
        cv2.imwrite(output_path + name + "-" + number + ".jpg", gray_img)        
    # cv2.imshow("img", img)
    cv2.waitKey(0)

np.random.seed(1957)

current_file = path + "Gugui_1.jpeg"
#convertir las .HEIC
if '.HEIC' in current_file: 
    temp_img = Image.open(current_file)
    jpg_photo = current_file.replace('.HEIC','.jpg')
    temp_img.save(current_file + jpg_photo)
    print(os.listdir(current_file))


crop_img(current_file,face_cascade,'Gugui',1, output_path)
    
 
  
   
#winsound.Beep(640,2000)