import os
import numpy as np
import cv2
import csv

# Definir el directorio base
base_dir = "C:\\DMA-Grupo3"
ruta_imagenes = os.path.join(base_dir, "output")

print(f"Directorio base: {base_dir}")

# Leer nombres de personas (carpetas)
names = []
for dir in os.listdir(ruta_imagenes):
    names.append(dir)
names = np.unique(names)

# Definir directorio de salida
output_dir = os.path.join(base_dir, "csvs")
os.makedirs(output_dir, exist_ok=True)

# Rutas de archivos CSV
faces_csv_path = os.path.join(output_dir, "faces_test.csv")
names_csv_path = os.path.join(output_dir, "names_test.csv")
files_csv_path = os.path.join(output_dir, "files_test.csv")

# Inicializar contadores
total_imagenes = 0

# Crear los archivos CSV y escribir los encabezados (si es necesario)
with open(faces_csv_path, mode='w', newline='') as faces_csv, \
     open(names_csv_path, mode='w', newline='') as names_csv, \
     open(files_csv_path, mode='w', newline='') as files_csv:
    
    faces_writer = csv.writer(faces_csv)
    names_writer = csv.writer(names_csv)
    files_writer = csv.writer(files_csv)

    for person in names:
        current_dir = os.path.join(ruta_imagenes, person, "test")
        print(f"\nProcesando persona: {person}")
        
        if not os.path.exists(current_dir):
            print(f"Directorio no encontrado: {current_dir}")
            continue

        person_images = 0
        for raw_img in os.listdir(current_dir):
            img_path = os.path.join(current_dir, raw_img)
            
            if not os.path.isfile(img_path):
                continue
                
            # Leer la imagen en escala de grises
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"No se pudo cargar la imagen: {img_path}")
                continue
            
            # Verificar tamaño de la imagen
            if img.shape != (30, 30):
                print(f"Tamaño incorrecto en {img_path}: {img.shape}, se omite.")
                continue
            
            # Aplanar la imagen
            face_flat = img.flatten()
            
            # Escribir directamente en los archivos CSV
            faces_writer.writerow(face_flat)
            names_writer.writerow([person])
            files_writer.writerow([raw_img])
            
            person_images += 1
            total_imagenes += 1
        
        print(f"Procesadas {person_images} imágenes para {person}")

print(f"\nTotal de imágenes procesadas: {total_imagenes}")
print(f"Archivos CSV guardados en: {output_dir}")
