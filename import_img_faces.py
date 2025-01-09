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

path = "C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\Fotos\\"
output_path = "C:\\DMA-Grupo3\\output\\"
# Commented for thios to work in REPL
script_path = os.path.dirname(__file__) + "\\"
print(script_path)

# Read names
names = []
for dir in os.listdir(path):
    names.append(dir)
names.count
names = np.unique(names)


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

flat_faces = np.array([])
names_faces = np.array([])
files_faces = np.array([])

flat_faces_train = np.array([])
names_faces_train = np.array([])
files_faces_train = np.array([])

flat_faces_test = np.array([])
names_faces_test = np.array([])
files_faces_test = np.array([])

np.random.seed(1957)
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
    
    os.makedirs(output_path + person)
    output_path_person = output_path + person + "\\"
    
    for file in os.listdir(current_dir):
        if file[-4:] != "HEIC":
            i += 1
            current_img = current_dir + file
            print("Current file is: " + file)
            #crop_img(current_img, face_cascade, person, i, output_path)
            img = cv2.imread(current_img)
            face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(250,250))
            print(face)
            
            j = 0
            for x, y, w, h in face:
                j += 1
                face = img[y : y + h, x : x + w]
                gray_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.resize(gray_img, (30,30))
                
                # number = str(x)
                # cv2.imwrite(output_path + file + "-" + number + ".jpg", gray_img)
                cv2.imwrite(output_path_person + file + "-" + str(i) + "-" + str(j) + ".jpg", gray_img)
                
                gray_img_flat = gray_img.flatten()
                
                print(gray_img_flat[0], file, person)
                flat_faces = np.append(flat_faces, gray_img_flat)
                files_faces = np.append(files_faces, file + "-" + str(i) + "-" + str(j) + ".jpg")
                names_faces = np.append(names_faces, person)
                
    face_position = np.where(names_faces == person)[0]
    aux_for_test = np.random.choice(face_position, 4, replace=False)
    aux_for_train = np.setdiff1d(face_position, aux_for_test)
    
    flat_faces_test = np.append(flat_faces_test, flat_faces[aux_for_test])
    flat_faces_train = np.append(flat_faces_train, flat_faces[aux_for_train])
    
    files_faces_test = np.append(files_faces_test, files_faces[aux_for_test])
    files_faces_train = np.append(files_faces_train, files_faces[aux_for_train])
    
    names_faces_test = np.append(names_faces_test, names_faces[aux_for_test])
    names_faces_train = np.append(names_faces_train, names_faces[aux_for_train])
    
    lista_imgs_temp = os.listdir(output_path_person)
    os.makedirs(output_path_person + "train")
    os.makedirs(output_path_person + "test")
    
    for img_for_train in lista_imgs_temp:
        imagen_actual = os.path.join(output_path_person, img_for_train)
        
        if img_for_train in files_faces_train:
            shutil.move(imagen_actual, output_path_person + "train")
        elif img_for_train in files_faces_test:
            shutil.move(imagen_actual, output_path_person + "test")

# Exportacion    
np.savetxt("C:\\DMA-Grupo3\\faces_test.csv", flat_faces_test, delimiter=',')
np.savetxt("C:\\DMA-Grupo3\\names_test.csv", names_faces_test, delimiter=',', fmt='%s')
np.savetxt("C:\\DMA-Grupo3\\files_test.csv", files_faces_test, delimiter=',', fmt='%s')
np.savetxt("C:\\DMA-Grupo3\\faces_train.csv", flat_faces_train, delimiter=',')
np.savetxt("C:\\DMA-Grupo3\\names_train.csv", names_faces_train, delimiter=',', fmt='%s')
np.savetxt("C:\\DMA-Grupo3\\files_train.csv", files_faces_train, delimiter=',', fmt='%s')
   
winsound.Beep(640,2000)