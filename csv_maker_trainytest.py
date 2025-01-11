import os
import numpy as np
import cv2
from tqdm import tqdm
import tempfile
import shutil
import gc
from contextlib import contextmanager
import logging
from typing import Iterator, List, Tuple
import mmap
import psutil  # Agregamos psutil para mejor manejo de memoria
import sys

class MemoryEfficientImageProcessor:
    def __init__(self, base_dir: str, temp_dir: str = None):
        self.base_dir = os.path.expanduser(base_dir)
        self.temp_dir = temp_dir or os.path.join(self.base_dir, "temp")
        self.output_dir = os.path.join(self.base_dir, "csvs")
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Lista para mantener registro de objetos grandes
        self._large_objects = []

    def __del__(self):
        """Destructor para limpieza explícita"""
        self.cleanup()

    def cleanup(self):
        """Limpieza explícita de recursos"""
        # Limpiar objetos grandes
        for obj in self._large_objects:
            del obj
        self._large_objects.clear()
        
        # Forzar recolección de basura
        gc.collect()
        
        # Liberar memoria no utilizada al sistema
        if psutil and hasattr(psutil.Process(), 'memory_info'):
            process = psutil.Process(os.getpid())
            for obj in process.memory_maps():
                if hasattr(obj, 'rss'):
                    del obj
        
        # Limpiar caché de numpy
        #np.clear_npyio_cache()
        
        # Cerrar todos los archivos abiertos por CV2
        cv2.destroyAllWindows()

    @contextmanager
    def temp_file(self) -> Iterator[str]:
        temp = tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir)
        try:
            yield temp.name
        finally:
            try:
                os.unlink(temp.name)
            except OSError:
                pass
   
           
    def read_image_in_chunks(self, img_path: str) -> np.ndarray:
        try:
            # Leer la imagen directamente en escala de grises (ya debería ser 30x30)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.logger.error(f"No se pudo cargar la imagen: {img_path}")
                return None
            
            # Verificar que los valores estén dentro del rango [0, 255]
            if not np.all((img >= 0) & (img <= 255)):
                self.logger.error(f"Valores fuera de rango en la imagen: {img_path}")
            
            return img.astype(np.uint8)  # Asegurarse de que la imagen esté en formato uint8

        except Exception as e:
            self.logger.error(f"Error leyendo imagen {img_path}: {e}")
            return None
        finally:
            gc.collect()


    def process_image_batch(self, image_paths: List[str], person_names: List[str], pbar: tqdm) -> Tuple[List[np.ndarray], List[str], List[str]]:
        batch_faces = []
        batch_names = []
        batch_files = []
        
        try:
            for img_path, persona in zip(image_paths, person_names):
                try:
                    img = self.read_image_in_chunks(img_path)
                    if img is None:
                        pbar.write(f"⚠️ No se pudo cargar la imagen: {img_path}")
                        continue
                    
                    face_flat = img.flatten()
                    del img
                    
                    batch_faces.append(face_flat)
                    batch_names.append(persona)
                    batch_files.append(os.path.basename(img_path))
                    
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.write(f"❌ Error procesando {img_path}: {e}")
                finally:
                    gc.collect()
            
            return batch_faces, batch_names, batch_files
        finally:
            # Limpiar variables que ya no se necesitan
            del img_path, persona
            gc.collect()

            
    def save_batch_to_disk(self, batch_faces: List[np.ndarray], batch_names: List[str], batch_files: List[str], csv_paths: Tuple[str, str, str]) -> None:
        faces_path, names_path, files_path = csv_paths
        
        try:
            # Guardar las caras como enteros de 0 a 255
            with open(faces_path, 'a') as f_faces:
                for face in batch_faces:
                    # Asegurarnos de que la imagen esté en tipo uint8 y escribirla como una fila en el CSV
                    face = face.astype(np.uint8)
                    face_str = ','.join(map(str, face))  # Convertir cada valor de la cara a string
                    f_faces.write(f"{face_str}\n")
            
            # Escribir los nombres y archivos
            with open(names_path, 'a') as f_names:
                for name in batch_names:
                    f_names.write(f"{name}\n")
            
            with open(files_path, 'a') as f_files:
                for file in batch_files:
                    f_files.write(f"{file}\n")
                    
        finally:
            # Limpiar variables
            del batch_faces
            gc.collect()
            
            

    def process_dataset(self, tipo: str, batch_size: int = 500) -> None:
        try:
            ruta_imagenes = os.path.join(self.base_dir, "output")
            csv_paths = (
                os.path.join(self.output_dir, f"faces_{tipo}.csv"),
                os.path.join(self.output_dir, f"names_{tipo}.csv"),
                os.path.join(self.output_dir, f"files_{tipo}.csv")
            )
            
            for path in csv_paths:
                open(path, 'w').close()
            
            nombres_personas = sorted(os.listdir(ruta_imagenes))
            
            
            with tqdm(total=len(nombres_personas), desc=f"👤 Procesando personas ({tipo})", 
                     unit="persona", position=0) as pbar_personas:
                
                for persona in nombres_personas:
                    try:
                        directorio_actual = os.path.join(ruta_imagenes, persona, tipo)
                        if not os.path.exists(directorio_actual):
                            pbar_personas.write(f"⚠️ Directorio no encontrado: {directorio_actual}")
                            pbar_personas.update(1)
                            continue
                        
                        imagenes = [img for img in os.listdir(directorio_actual) 
                                  if os.path.isfile(os.path.join(directorio_actual, img))]
                        total_imagenes = len(imagenes)
                        
                        with tqdm(total=total_imagenes, 
                                 desc=f"🖼️  {persona} ({tipo})", 
                                 unit="img",
                                 position=1,
                                 leave=False) as pbar_imagenes:
                            
                            # Procesar en lotes más pequeños
                            for i in range(0, len(imagenes), batch_size):
                                batch_imgs = imagenes[i:i + batch_size]
                                image_paths = [os.path.join(directorio_actual, img) for img in batch_imgs]
                                person_names = [persona] * len(batch_imgs)
                                
                                try:
                                    batch_faces, batch_names, batch_files = self.process_image_batch(
                                        image_paths, person_names, pbar_imagenes
                                    )
                                    self.save_batch_to_disk(batch_faces, batch_names, batch_files, csv_paths)
                                finally:
                                    # Limpiar variables del lote
                                    del batch_imgs, image_paths, person_names
                                    if 'batch_faces' in locals():
                                        del batch_faces, batch_names, batch_files
                                    gc.collect()
                        
                        pbar_personas.write(f"✅ Completado: {persona} - {total_imagenes} imágenes procesadas")
                        pbar_personas.update(1)
                        
                    finally:
                        # Limpiar variables de la persona actual
                        if 'imagenes' in locals():
                            del imagenes
                        gc.collect()
                        
        finally:
            # Limpieza final
            self.cleanup()

def main():
    base_dir = os.path.expanduser("~/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main")
    temp_dir = os.path.join(base_dir, "temp_processing")
    
    try:
        processor = MemoryEfficientImageProcessor(base_dir, temp_dir)
        
        print("🚀 Iniciando procesamiento de imágenes...")
        processor.process_dataset("train", batch_size=500)
        print("\n✨ Datos de entrenamiento completados")
        
        processor.process_dataset("test", batch_size=500)
        print("\n✨ Datos de prueba completados")
        
        print("\n🎉 Procesamiento completado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        raise
    finally:
        # Limpieza final
        if 'processor' in locals():
            processor.cleanup()
            del processor
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("🧹 Directorio temporal limpiado")
        
        # Forzar limpieza final
        gc.collect()
        if psutil and hasattr(psutil.Process(), 'memory_info'):
            process = psutil.Process(os.getpid())
            if hasattr(process, 'memory_clear'):
                process.memory_clear()

if __name__ == "__main__":
    main()