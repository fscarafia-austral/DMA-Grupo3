import numpy as np
import pandas as pd
from utils import *
import math
import joblib 
from iprocess import *
import sys
import calendar
import time


if __name__ == "__main__":

    argumentos = sys.argv   
    if len(argumentos) < 2:
        print("Debe completar todos los argumentos para ejecutar el progrma")
        exit()
    
    # Carga del modelo.
    modeloImport = joblib.load('./modelo_entrenado.pkl') 
    # Carga la lista de nombres
    unique_names = pd.read_csv("./csvs/names.csv", delimiter=",", header=None).values
    
    path = argumentos[1]
    output_path = argumentos[2]
    
    # Carga la imagen para predecir
    salida = calendar.timegm(time.gmtime())
    image_path = crop_img(path,str(salida),1, output_path)
    print(image_path)
    # Predicción con ensemble
    image_processed = modeloImport.preprocess_image(image_path).reshape(1, -1)
    
    # Hacer múltiples predicciones con dropout
    n_predictions = 5
    predictions = []
    for _ in range(n_predictions):
        pred = modeloImport.predict(image_processed, dropout=True)
        predictions.append(pred)
    
    # Promedio de predicciones
    avg_prediction = np.mean(predictions, axis=0)
    predicted_person = unique_names[np.argmax(avg_prediction)]
    confidence = np.max(avg_prediction) * 100
    
    print(f"\nLa persona predicha es: {predicted_person} (Confianza: {confidence:.2f}%)")