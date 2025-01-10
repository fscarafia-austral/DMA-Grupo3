import os
import shutil
import Augmentor
import numpy as np
import cv2
 
# Ruta a la carpeta con las imágenes originales
ruta_imagenes = "C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\output\\"

# Read names
names = []
for dir in os.listdir(ruta_imagenes):
    names.append(dir)
names.count
names = np.unique(names)

# Número total de imágenes deseadas
numero_total_imagenes = 10000

# Número de imágenes originales
numero_imagenes_originales = len(os.listdir(ruta_imagenes))

# Cantidad de imágenes a generar por cada imagen original
imagenes_por_imagen_original = numero_total_imagenes // numero_imagenes_originales

print(f"Se generarán {imagenes_por_imagen_original} imágenes por cada imagen original.")

# Crear una carpeta de salida para las imágenes aumentadas
# carpeta_output = "C:\DMA-Grupo3\output_aumentadas"
# os.makedirs(carpeta_output, exist_ok=True)

# Crear una carpeta temporal para procesar una imagen a la vez
carpeta_temporal = "C:\\Repositorios-Ing.Carlos-Cicconi\\DMA-Grupo3\\temp_imagen"
# os.makedirs(carpeta_temporal, exist_ok=True)

# Iterar sobre cada imagen en la carpeta original
for person in names:
    person_path = ruta_imagenes + person + "\\train\\"
    print(person_path)    
    for imagen in os.listdir(person_path):
        print(imagen)
        for archivo in os.listdir(carpeta_temporal):
            os.remove(os.path.join(carpeta_temporal, archivo))
        # Copiar la imagen actual a la carpeta temporal
        imagen_actual = os.path.join(person_path, imagen)
        shutil.copy(imagen_actual, carpeta_temporal)
        # Crear un pipeline para procesar solo la imagen actual
        p = Augmentor.Pipeline(source_directory=carpeta_temporal, output_directory=person_path)
        # Agregar transformaciones
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)  # Rotación
        p.flip_left_right(probability=0.5)  # Flip horizontal
        p.zoom_random(probability=0.5, percentage_area=0.8)  # Zoom
        p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)  # Brillo
        p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)  # Contraste
        # Generar imágenes aumentadas para la imagen actual
        p.sample(imagenes_por_imagen_original)
    
    print("Generación completada.")
    
flat_faces = np.array([])
names_faces = np.array([])
files_faces = np.array([])

flat_faces_train = np.array([])
names_faces_train = np.array([])
files_faces_train = np.array([])

for person in names:
    current_dir = path + person + "\\"
    print("Current person is: " + person)
    for raw_img in os.listdir(current_dir):            
        j = 0
        img = cv2.imread(raw_img)
    # Detect the face
        face = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)
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