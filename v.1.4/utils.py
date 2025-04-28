import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from tqdm import tqdm
import math
from PIL import Image
import joblib 

def load_data(faces_path, names_path, files_path):
    """Carga y preprocesa los datos desde archivos CSV"""
    try:
        with tqdm(total=3, desc="Cargando archivos CSV") as pbar:
            faces = np.loadtxt(faces_path, delimiter=",")
            pbar.update(1)
            names = pd.read_csv(names_path, delimiter=",", header=None).values
            pbar.update(1)
            files = pd.read_csv(files_path, delimiter=",", header=None).values
            pbar.update(1)
            
        # Convertir a float64 y normalizar
        faces = faces.astype('float64') / 255.0
        
        # Preparar las etiquetas
        label_encoder = LabelEncoder()
        label_binarizer = LabelBinarizer()
        names_encoded = label_binarizer.fit_transform(names)
        names_int = label_encoder.fit_transform(names)
        
        return faces, names_encoded, names_int, names, label_encoder.classes_
        
    except Exception as e:
        print(f"Error al cargar los archivos: {e}")
        raise

class FaceRecognitionModel:
    def __init__(self, hidden_layers=[120, 80, 40], learning_rate=0.005, epochs=50, 
                 batch_size=16, pca_components=100):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.pca_components = pca_components
        
    def _initialize_weights(self, input_size, output_size):
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights.append(np.random.uniform(-limit, limit, 
                              (layer_sizes[i+1], layer_sizes[i])))
            self.biases.append(np.zeros((layer_sizes[i+1], 1)))

    def _activation(self, x, derivative=False):
        # Leaky ReLU para mejor gradiente
        alpha = 0.01
        if derivative:
            return np.where(x > 0, 1, alpha)
        return np.where(x > 0, x, alpha * x)

    def _output_activation(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def fit(self, X_train, y_train, names_int, validation_split=0.1):
        # Separar conjunto de validación
        n_val = int(X_train.shape[0] * validation_split)
        indices = np.random.permutation(X_train.shape[0])
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        names_int_train = names_int[train_idx]
        names_int_val = names_int[val_idx]

        # Preprocesamiento
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # PCA con whitening
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        
        input_size = X_train_pca.shape[1]
        output_size = y_train.shape[1]
        
        self._initialize_weights(input_size, output_size)
        
        best_val_accuracy = 0
        patience = 15
        no_improve = 0
        
        for epoch in range(self.epochs):
            # Learning rate decay
            current_lr = self.learning_rate / (1 + epoch * 0.01)
            
            # Training
            n_batches = len(X_train_pca) // self.batch_size
            with tqdm(total=n_batches, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for i in range(0, len(X_train_pca), self.batch_size):
                    batch_X = X_train_pca[i:i+self.batch_size].T
                    batch_y = y_train[i:i+self.batch_size].T
                    
                    # Forward pass con dropout
                    activations = [batch_X]
                    masks = []
                    dropout_rate = 0.2
                    
                    for j in range(len(self.weights)):
                        z = self.weights[j] @ activations[-1] + self.biases[j]
                        
                        if j != len(self.weights) - 1:  # No dropout en la capa de salida
                            mask = (np.random.rand(*z.shape) > dropout_rate) / (1 - dropout_rate)
                            masks.append(mask)
                            a = self._activation(z) * mask
                        else:
                            a = self._output_activation(z)
                        
                        activations.append(a)
                    
                    # Backward pass con L2 regularization
                    deltas = []
                    error = activations[-1] - batch_y
                    delta = error
                    deltas.append(delta)
                    
                    for j in range(len(self.weights)-1, 0, -1):
                        delta = (self.weights[j].T @ delta)
                        if j != 0:  # No aplicar dropout en la entrada
                            delta = delta * masks[j-1]
                        delta = delta * self._activation(activations[j], derivative=True)
                        deltas.append(delta)
                    deltas = deltas[::-1]
                    
                    # Update weights con L2 regularization
                    l2_lambda = 0.0001
                    for j in range(len(self.weights)):
                        weight_update = (deltas[j] @ activations[j].T) / batch_X.shape[1]
                        bias_update = np.mean(deltas[j], axis=1, keepdims=True)
                        
                        # L2 regularization
                        weight_update += l2_lambda * self.weights[j]
                        
                        self.weights[j] -= current_lr * weight_update
                        self.biases[j] -= current_lr * bias_update
                    
                    pbar.update(1)
            
            # Evaluar en validación
            val_predictions = self.predict(X_val)
            val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == names_int_val)
            
            # Evaluar en training
            train_predictions = self.predict(X_train)
            train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == names_int_train)
            
            print(f"Train accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improve = 0
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered")
                    self.weights = self.best_weights
                    self.biases = self.best_biases
                    break

    def predict(self, X, dropout=False):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        current_activation = X_pca.T
        dropout_rate = 0.2 if dropout else 0
        
        for i in range(len(self.weights)):
            z = self.weights[i] @ current_activation + self.biases[i]
            if i == len(self.weights) - 1:
                current_activation = self._output_activation(z)
            else:
                current_activation = self._activation(z)
                if dropout:
                    mask = (np.random.rand(*current_activation.shape) > dropout_rate) / (1 - dropout_rate)
                    current_activation *= mask
        
        return current_activation.T

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')  # Convertir a escala de grises
        image = image.resize((30, 30))  # Asegurarse de que las imágenes tengan el tamaño correcto
        image = np.array(image) / 255.0  # Normalizar la imagen
        return image.flatten()  # Retornar la imagen como un vector plano


def train_and_evaluate():
    # Cargar datos
    print("Cargando datos...")
    faces_train, names_encoded_train, names_int_train, names_train, unique_names = load_data(
        "./csvs/faces_train.csv",
        "./csvs/names_train.csv",
        "./csvs/files_train.csv"
    )
    
    # Crear y entrenar modelo
    model = FaceRecognitionModel(
        hidden_layers=[120, 80, 40],  # Arquitectura más profunda
        learning_rate=0.005,          # Learning rate más bajo
        epochs=50,                    # Más épocas
        batch_size=16,                # Batch size más pequeño
        pca_components=100            # Más componentes PCA
    )
    
    print("Entrenando modelo...")
    model.fit(faces_train, names_encoded_train, names_int_train)
    
    # Evaluar en test
    faces_test, names_encoded_test, names_int_test, names_test, _ = load_data(
        "./csvs/faces_test.csv",
        "./csvs/names_test.csv",
        "./csvs/files_test.csv"
    )
    
    print("\nEvaluando en test...")
    predictions_test = model.predict(faces_test)
    pred_classes_test = np.argmax(predictions_test, axis=1)
    accuracy_test = np.mean(pred_classes_test == names_int_test)
    print(f"Test accuracy: {accuracy_test:.4f}")
    
    # Mostrar matriz de confusión
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(names_int_test, pred_classes_test)
    print("\nMatriz de confusión:")
    for i, name in enumerate(unique_names):
        print(f"{name}: {cm[i].tolist()}")
    
    return model, unique_names