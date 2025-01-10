import os
import numpy as np
import cv2

# Definir el directorio base
base_dir = os.path.expanduser("~/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main")
ruta_imagenes = os.path.join(base_dir, "output")

print(f"Directorio base: {base_dir}")

# Read names
names = []
for dir in os.listdir(ruta_imagenes):
    names.append(dir)
names = np.unique(names)

# Inicializar listas
flat_faces_train = []
names_faces_train = []
files_faces_train = []

total_imagenes = 0

for person in names:
    current_dir = os.path.join(ruta_imagenes, person, "train")
    print(f"\nProcesando persona: {person}")
    
    if not os.path.exists(current_dir):
        print(f"Directorio no encontrado: {current_dir}")
        continue

    person_images = 0
    for raw_img in os.listdir(current_dir):
        img_path = os.path.join(current_dir, raw_img)
        
        if not os.path.isfile(img_path):
            continue
            
        # Leer la imagen directamente
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"No se pudo cargar la imagen: {img_path}")
            continue
        
        # Aplanar la imagen
        face_flat = img.flatten()
        
        # Guardar datos
        flat_faces_train.append(face_flat)
        files_faces_train.append(raw_img)
        names_faces_train.append(person)
        
        person_images += 1
        total_imagenes += 1
        
    print(f"Procesadas {person_images} imágenes para {person}")

# Convertir a arrays numpy
flat_faces_train = np.array(flat_faces_train)
files_faces_train = np.array(files_faces_train)
names_faces_train = np.array(names_faces_train)

# Verificar y guardar
if total_imagenes > 0:
    print(f"\nTotal de imágenes procesadas: {total_imagenes}")
    print(f"Dimensiones del array de rostros: {flat_faces_train.shape}")
    
    output_dir = os.path.join(base_dir, "csvs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar los archivos CSV
    np.savetxt(
        os.path.join(output_dir, "faces_train.csv"),
        flat_faces_train,
        delimiter=','
    )
    np.savetxt(
        os.path.join(output_dir, "names_train.csv"),
        names_faces_train,
        delimiter=',',
        fmt='%s'
    )
    np.savetxt(
        os.path.join(output_dir, "files_train.csv"),
        files_faces_train,
        delimiter=',',
        fmt='%s'
    )
    
    print(f"\nArchivos CSV guardados en: {output_dir}")
    print("Dimensiones esperadas del array de rostros:")
    print(f"- Número de imágenes: {total_imagenes}")
    print(f"- Píxeles por imagen: {flat_faces_train.shape[1]}")
else:
    print("No se encontraron imágenes para procesar")