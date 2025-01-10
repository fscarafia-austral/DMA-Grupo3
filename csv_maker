import os
import shutil
import Augmentor
import numpy as np
import cv2
from PIL import Image
 
# Ruta a la carpeta con las imágenes originales
ruta_imagenes = "C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\output\\"

# Read names
names = []
for dir in os.listdir(ruta_imagenes):
    names.append(dir)
names.count
names = np.unique(names)

# Load the cascade
casc_path = "C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\" + "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(casc_path)

flat_faces = np.array([])
names_faces = np.array([])
files_faces = np.array([])

flat_faces_train = np.array([])
names_faces_train = np.array([])
files_faces_train = np.array([])

for person in names:
    current_dir = ruta_imagenes + person + "\\train\\"
    print("Current person is: " + person)
    for raw_img in os.listdir(current_dir):            
        j = 0
        img = cv2.imread(raw_img)
        face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(250,250))
    # Detect the face
        # face = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)
        for x, y, w, h in face:
            j += 1
            face = img[y : y + h, x : x + w]
            face_flat = face.flatten()
            
            print(face_flat[0], raw_img, person)
            flat_faces_train = np.append(flat_faces, face_flat)
            files_faces_train = np.append(files_faces, raw_img)
            names_faces_train = np.append(names_faces, person)

# Exportacion    
np.savetxt("C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\csvs\\faces_train.csv", flat_faces_train, delimiter=',')
np.savetxt("C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\csvs\\names_train.csv", names_faces_train, delimiter=',', fmt='%s')
np.savetxt("C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\csvs\\files_train.csv", files_faces_train, delimiter=',', fmt='%s')