import os

import cv2
#import pyheif
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np

import winsound

register_heif_opener()

# Set path to photos

path = "C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\Fotos\\"
output_path = "C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\output\\"
# Commented for thios to work in REPL
script_path = os.path.dirname(__file__) + "\\"
print(script_path)

# Read names
names = []
for dir in os.listdir(path):
    names.append(dir)
names.count


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
        # cv2.imwrite(output_path + name + "-" + number + ".jpg", gray_img)        
    # cv2.imshow("img", img)
    cv2.waitKey(0)

flat_faces = []
names_faces = []
files_faces = []
position_faces = []

for person in names:
    current_dir = path + person + "\\"
    print("Current person is: " + person)
    #convertir las .HEIC
    heic_files = [photo for photo in os.listdir(current_dir) if '.HEIC' in photo]
    print(heic_files)
    for photo in heic_files:
        temp_img = Image.open(current_dir + photo)
        jpg_photo = photo.replace('.HEIC','.jpg')
        temp_img.save(current_dir + jpg_photo)
    i = 0
    for file in os.listdir(current_dir):
        if file[-4:] != "HEIC":
            i += 1
            current_img = current_dir + file
            print("Current file is: " + file)
            crop_img(current_img, face_cascade, person, i, output_path)
            img = cv2.imread(current_img)
            face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(250,250))
            print(face)
            for x, y, w, h in face:
                face = img[y : y + h, x : x + w]
                gray_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.resize(gray_img, (30,30))
                
                number = str(x)
                cv2.imwrite(output_path + file + "-" + number + ".jpg", gray_img)
                
                gray_img_flat = gray_img.flatten()
                
                print(gray_img_flat[0], file, person)
                flat_faces.append(gray_img_flat)
                files_faces.append(file)
                names_faces.append(person)
                
# Convert the faces list to a NumPy array
flat_faces_array = np.array(flat_faces)
names_faces_array = np.array(names_faces)
files_faces_array = np.array(files_faces)

np.savetxt("C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\faces_numpy.csv", flat_faces_array, delimiter=',')
np.savetxt("C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\names_numpy.csv", names_faces_array, delimiter=',', fmt='%s')
np.savetxt("C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\files_numpy.csv", files_faces_array, delimiter=',', fmt='%s')

winsound.Beep(640,2000)