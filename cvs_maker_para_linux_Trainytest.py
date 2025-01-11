import os
import numpy as np
import cv2
from tqdm import tqdm  # Barra de progreso


def procesar_y_guardar_imagenes(ruta_imagenes, tipo, output_dir, batch_size=100):
    """
    Procesa imágenes en lotes y guarda directamente en archivos CSV para minimizar el uso de memoria.
    
    Args:
        ruta_imagenes (str): Ruta base de las imágenes.
        tipo (str): Puede ser "train" o "test".
        output_dir (str): Directorio donde se guardarán los CSVs.
        batch_size (int): Número de imágenes por lote.
    """
    nombres_personas = sorted(os.listdir(ruta_imagenes))
    ruta_csv_faces = os.path.join(output_dir, f"faces_{tipo}.csv")
    ruta_csv_names = os.path.join(output_dir, f"names_{tipo}.csv")
    ruta_csv_files = os.path.join(output_dir, f"files_{tipo}.csv")

    # Crear o truncar los archivos CSV antes de empezar
    open(ruta_csv_faces, 'w').close()
    open(ruta_csv_names, 'w').close()
    open(ruta_csv_files, 'w').close()

    # Procesar cada persona
    for persona in tqdm(nombres_personas, desc=f"Procesando {tipo} (personas)", unit="persona"):
        directorio_actual = os.path.join(ruta_imagenes, persona, tipo)
        if not os.path.exists(directorio_actual):
            tqdm.write(f"Directorio no encontrado: {directorio_actual}")
            continue

        imagenes = os.listdir(directorio_actual)
        batch_faces = []
        batch_names = []
        batch_files = []

        # Procesar imágenes dentro del lote
        for raw_img in tqdm(imagenes, desc=f"  {persona} ({tipo})", leave=False, unit="imagen"):
            img_path = os.path.join(directorio_actual, raw_img)
            if not os.path.isfile(img_path):
                continue

            try:
                # Leer la imagen
                img = cv2.imread(img_path)
                if img is None:
                    tqdm.write(f"No se pudo cargar la imagen: {img_path}")
                    continue

                # Aplanar la imagen
                face_flat = img.flatten()

                # Agregar al lote
                batch_faces.append(face_flat)
                batch_names.append(persona)
                batch_files.append(raw_img)

                # Escribir lote en los CSV si se alcanza el tamaño
                if len(batch_faces) >= batch_size:
                    guardar_batch_en_csv(batch_faces, batch_names, batch_files, ruta_csv_faces, ruta_csv_names, ruta_csv_files)
                    batch_faces, batch_names, batch_files = [], [], []  # Limpiar el lote
            except Exception as e:
                tqdm.write(f"Error procesando {img_path}: {e}")

        # Escribir cualquier remanente del lote
        if batch_faces:
            guardar_batch_en_csv(batch_faces, batch_names, batch_files, ruta_csv_faces, ruta_csv_names, ruta_csv_files)

        tqdm.write(f"Procesadas imágenes para {persona}")


def guardar_batch_en_csv(batch_faces, batch_names, batch_files, ruta_csv_faces, ruta_csv_names, ruta_csv_files):
    """
    Guarda un lote de datos en archivos CSV.
    
    Args:
        batch_faces (list): Lista de imágenes aplanadas.
        batch_names (list): Lista de nombres asociados a las imágenes.
        batch_files (list): Lista de nombres de archivo de las imágenes.
        ruta_csv_faces (str): Ruta del archivo CSV para las imágenes.
        ruta_csv_names (str): Ruta del archivo CSV para los nombres.
        ruta_csv_files (str): Ruta del archivo CSV para los archivos.
    """
    try:
        # Guardar imágenes aplanadas
        with open(ruta_csv_faces, 'ab') as f_faces:
            np.savetxt(f_faces, np.array(batch_faces), delimiter=",")
        # Guardar nombres
        with open(ruta_csv_names, 'a') as f_names:
            f_names.writelines(f"{name}\n" for name in batch_names)
        # Guardar archivos
        with open(ruta_csv_files, 'a') as f_files:
            f_files.writelines(f"{file}\n" for file in batch_files)
    except Exception as e:
        print(f"Error guardando lote en CSV: {e}")


def main():
    # Definir directorios
    base_dir = os.path.expanduser("~/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main")
    ruta_imagenes = os.path.join(base_dir, "output")
    output_dir = os.path.join(base_dir, "csvs")

    print(f"Directorio base: {base_dir}")
    if not os.path.exists(ruta_imagenes):
        print(f"Directorio de imágenes no encontrado: {ruta_imagenes}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Procesar datos de entrenamiento y prueba
    procesar_y_guardar_imagenes(ruta_imagenes, tipo="train", output_dir=output_dir, batch_size=100)
    procesar_y_guardar_imagenes(ruta_imagenes, tipo="test", output_dir=output_dir, batch_size=100)

    print("\nProcesamiento completado. Archivos CSV guardados en:", output_dir)


if __name__ == "__main__":
    main()

