import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image
import cv2

# -----------------------------
# Carga de datos
# -----------------------------

# Cargar los datos de entrenamiento
try:
    training_faces = np.loadtxt(
        "/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/csvs/faces_train.csv",
        delimiter=",",
    )
    training_names = pd.read_csv(
        "/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/csvs/names_train.csv",
        delimiter=",",
        header=None,
    ).values
    training_files = pd.read_csv(
        "/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/csvs/files_train.csv",
        delimiter=",",
        header=None,
    ).values
except Exception as e:
    print(f"Error al leer los archivos: {e}")
    raise

# Normalizar las imágenes a rango [0, 1]
training_faces = training_faces.astype("float64") / 255.0

# Aplicar PCA para reducir dimensionalidad
pca = PCA(n_components=60, svd_solver="full")
training_faces = pca.fit_transform(training_faces)
training_faces = training_faces[:, 3:]  # Eliminar las primeras 3 componentes

# Codificar las etiquetas
label_encoder = LabelEncoder()
label_binarizer = LabelBinarizer()
training_names_num = label_binarizer.fit_transform(training_names)
training_int_num = label_encoder.fit_transform(training_names)

# Dividir el conjunto en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    training_faces, training_names_num, test_size=0.2, random_state=42
)

# -----------------------------
# Construcción del modelo
# -----------------------------

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Entrenamiento del modelo
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Predicción con nuevas imágenes
# -----------------------------

def predecir_imagen(imagen_path):
    """
    Carga una imagen, la preprocesa y realiza una predicción.
    """
    # Cargar la imagen desde la ruta
    img = Image.open(imagen_path)
    
    # Convertir la imagen a escala de grises y redimensionar
    img = img.convert("L")
    img = img.resize((30, 30))  # Redimensionamos la imagen a 30x30 píxeles
    
    # Convertir la imagen en un array numpy
    img_array = np.array(img).astype("float64") / 255.0

    # Aplanar la imagen para que tenga la misma forma que las imágenes de entrenamiento
    img_array = img_array.flatten().reshape(1, -1) # Crear un vector de 900 características

    # Aplicar la misma transformación PCA (ahora transformamos a 57 componentes)
    img_pca = pca.transform(img_array)[:, 3:]  # Eliminamos las primeras 3 componentes para obtener 57 características

    # Realizar la predicción
    pred = model.predict(img_pca)
    pred_class = np.argmax(pred, axis=1)

    # Obtener el nombre de la persona predicha
    predicted_name = label_binarizer.classes_[pred_class][0]

    # Mostrar el resultado
    print(f"Predicción: {predicted_name}")


# -----------------------------
# Evaluación en el conjunto de test
# -----------------------------

# Cargar el conjunto de test
test_faces = np.loadtxt(
    "./csvs/faces_test.csv", delimiter=","
).astype("float64") / 255.0

test_names = pd.read_csv("./csvs/names_test.csv", delimiter=",", header=None).values
test_faces = pca.transform(test_faces)[:, 3:]

# Codificar etiquetas del test
test_names_num = label_binarizer.transform(test_names)

# Evaluar el modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(test_faces, test_names_num, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

# -----------------------------
# Realizar predicciones y mostrar resultados
# -----------------------------

# Hacer predicciones sobre el conjunto de test
y_pred = model.predict(test_faces)

# Convertir las predicciones en las clases correspondientes
y_pred_classes = np.argmax(y_pred, axis=1)

# Obtener los nombres de las personas predichas
predicted_names = label_binarizer.classes_[y_pred_classes]

# Obtener los nombres reales
true_names = label_binarizer.classes_[np.argmax(test_names_num, axis=1)]

# Mostrar resultados durante la evaluación
print("\nResultados de la evaluación en el conjunto de test:")
for true_name, predicted_name in zip(true_names, predicted_names):
    is_correct = true_name == predicted_name
    print(f"Persona Real: {true_name} - Predicción: {predicted_name} - Correcto: {is_correct}")

# -----------------------------
# Reporte de clasificación y matriz de confusión con nombres
# -----------------------------

# Imprimir el reporte de clasificación con los nombres de las personas
print("\nClassification Report:")
print(classification_report(true_names, predicted_names))

# Imprimir la matriz de confusión
conf_matrix = confusion_matrix(true_names, predicted_names)
print("Confusion Matrix:")
print(conf_matrix)



# -----------------------------
# Uso de la función para predecir una imagen
# -----------------------------

# Ruta de la imagen en la carpeta de test
imagen_path = "/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/test/IMG_20240309_112032.jpg-6-1.jpg"  # Cambia esta ruta a una imagen en tu carpeta

# Realizar la predicción sobre la imagen
predecir_imagen(imagen_path)


