import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchNormalization:
    def __init__(self, input_dim):
        self.epsilon = 1e-8
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        self.momentum = 0.9

    def forward(self, x, training=True):
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # Actualizar running statistics
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            mu = self.running_mean
            var = self.running_var
        
        x_norm = (x - mu) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_norm + self.beta
        
        if training:
            self.cache = (x, x_norm, mu, var)
        
        return out

@dataclass
class NetworkConfig:
    """Configuración de la red neuronal"""
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    learning_rate: float = 0.1
    epoch_limit: int = 50
    error_threshold: float = 1.0e-6
    activation_functions: List[str] = None
    batch_size: int = 32
    
    def __post_init__(self):
        if self.activation_functions is None:
            self.activation_functions = ['tansig'] * (len(self.hidden_sizes)) + ['softmax']
    
    def to_dict(self):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'epoch_limit': self.epoch_limit,
            'error_threshold': self.error_threshold,
            'activation_functions': self.activation_functions,
            'batch_size': self.batch_size
        }

class ActivationFunctions:
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)
    
    @staticmethod
    def tansig(x: np.ndarray) -> np.ndarray:
        return 2.0 / (1.0 + np.exp(-2.0 * np.clip(x, -500, 500))) - 1.0
    
    @staticmethod
    def get_derivative(fname: str) -> Callable:
        derivatives = {
            "softmax": lambda y: y * (1.0 - y),  # Aproximación simplificada
            "tansig": lambda y: 1.0 - y**2
        }
        return derivatives.get(fname)

class DataPreprocessor:
    def __init__(self):
        self.mean = None
        self.zca_matrix = None
        self.pca = None
        
    def zca_whitening(self, X: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        logger.info("Aplicando ZCA whitening...")
        with tqdm(total=4, desc="ZCA Whitening") as pbar:
            self.mean = np.mean(X, axis=0)
            X_centered = X - self.mean
            pbar.update(1)
            
            n_samples = X_centered.shape[0]
            cov = np.dot(X_centered.T, X_centered) / n_samples
            pbar.update(1)
            
            U, S, V = np.linalg.svd(cov)
            pbar.update(1)
            
            self.zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
            X_whitened = np.dot(X_centered, self.zca_matrix)
            pbar.update(1)
        
        return X_whitened

def load_data(file_paths: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray, DataPreprocessor, PCA]:
    """Cargar y preparar datos"""
    try:
        logger.info(f"Cargando datos desde {file_paths['train_faces']} y {file_paths['train_names']}")
        
        # Validar si los archivos existen
        if not file_paths["train_faces"].exists() or not file_paths["train_names"].exists():
            raise FileNotFoundError("No se encuentran los archivos de entrenamiento.")
        
        # Leer datos con barra de progreso
        logger.info("Leyendo datos de entrenamiento...")
        total_lines = sum(1 for line in open(file_paths["train_faces"]))
        
        training_faces = []
        with tqdm(total=total_lines, desc="Cargando imágenes") as pbar:
            for chunk in pd.read_csv(file_paths["train_faces"], header=None, chunksize=100):
                training_faces.append(chunk.values)
                pbar.update(len(chunk))
        
        training_faces = np.vstack(training_faces).astype('float64')
        training_names = pd.read_csv(file_paths["train_names"], header=None).values
        
        # Validar datos
        if training_faces.shape[0] <= 1 or training_faces.shape[1] <= 1:
            raise ValueError("El conjunto de datos no tiene suficientes filas o columnas.")
        
        # Normalizar datos a [0,1]
        logger.info("Normalizando datos...")
        training_faces = training_faces / 255.0
        
        # Aplicar ZCA whitening
        preprocessor = DataPreprocessor()
        training_faces_whitened = preprocessor.zca_whitening(training_faces)
        
        # Aplicar PCA
        logger.info("Aplicando PCA...")
        n_components = min(60, training_faces_whitened.shape[0], training_faces_whitened.shape[1])
        pca = PCA(n_components=n_components, svd_solver='full')
        with tqdm(total=1, desc="PCA") as pbar:
            training_Z = pca.fit_transform(training_faces_whitened)
            pbar.update(1)
        
        # Preparar etiquetas
        label_binarizer = LabelBinarizer()
        training_names_num = label_binarizer.fit_transform(training_names)
        
        logger.info("Datos cargados y preparados exitosamente")
        return training_Z, training_names_num, preprocessor, pca
    
    except Exception as e:
        logger.error(f"Error al cargar los datos: {str(e)}")
        raise

class ResultManager:
    """Clase para manejar el guardado de resultados"""
    def __init__(self, base_path: str = 'resultados_backpropagation'):
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_path = self.base_path / self.timestamp
        self.setup_directories()
        
    def setup_directories(self):
        """Crea las carpetas necesarias"""
        self.result_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de resultados creado en: {self.result_path}")
    
    def save_config(self, config: NetworkConfig):
        """Guarda la configuración de la red"""
        config_path = self.result_path / 'network_config.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        logger.info(f"Configuración guardada en: {config_path}")
    
    def save_training_results(self, results: List[Dict]):
        """Guarda los resultados del entrenamiento"""
        results_path = self.result_path / 'training_results.csv'
        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)
        logger.info(f"Resultados de entrenamiento guardados en: {results_path}")
    
    def save_weights(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        """Guarda los pesos y biases de la red"""
        weights_path = self.result_path / 'weights'
        weights_path.mkdir(exist_ok=True)
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            np.save(weights_path / f'weight_layer_{i}.npy', w)
            np.save(weights_path / f'bias_layer_{i}.npy', b)
        logger.info(f"Pesos guardados en: {weights_path}")
        
        
class SimpleNN:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.weights = []
        self.biases = []
        self.batch_norms = []
        self.initialize_weights()
        self.result_manager = ResultManager()
        self.result_manager.save_config(config)
    
    def initialize_weights(self):
        logger.info("Inicializando pesos con inicialización Xavier/Glorot...")
        layer_sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i + 1], layer_sizes[i])))
            self.biases.append(np.zeros((layer_sizes[i + 1], 1)))
            
            # Agregar batch normalization para cada capa oculta (excepto la última)
            if i < len(layer_sizes) - 2:
                self.batch_norms.append(BatchNormalization(layer_sizes[i + 1]))
    
    def forward_pass(self, X: np.ndarray, training: bool = True) -> List[np.ndarray]:
        activations = [X.T]
        pre_activations = []
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = np.dot(W, activations[-1]) + b
            pre_activations.append(Z)
            
            # Aplicar batch normalization en capas ocultas
            if i < len(self.weights) - 1:
                Z = self.batch_norms[i].forward(Z.T, training).T
            
            activation_fn = getattr(ActivationFunctions, self.config.activation_functions[i])
            activations.append(activation_fn(Z))
        
        return activations, pre_activations
    
    def get_batches(self, X, Y, batch_size):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], Y[batch_indices]
    
    def train(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray):
        results = []
        best_error = float('inf')
        patience = 5
        patience_counter = 0
        
        logger.info("Iniciando entrenamiento...")
        with tqdm(total=self.config.epoch_limit, desc="Entrenamiento") as pbar:
            for epoch in range(self.config.epoch_limit):
                epoch_error = 0
                batch_count = 0
                
                for batch_X, batch_Y in self.get_batches(X, Y, self.config.batch_size):
                    activations, pre_activations = self.forward_pass(batch_X, training=True)
                    batch_error = self.backpropagation(activations, pre_activations, batch_Y)
                    epoch_error += batch_error
                    batch_count += 1
                
                epoch_error /= batch_count
                
                # Evaluar en conjunto de validación
                val_activations, _ = self.forward_pass(X_val, training=False)
                val_predictions = np.argmax(val_activations[-1], axis=0)
                val_true = np.argmax(Y_val, axis=1)
                val_accuracy = np.mean(val_predictions == val_true)
                
                results.append({
                    'epoch': epoch + 1,
                    'train_error': float(epoch_error),
                    'val_accuracy': float(val_accuracy)
                })
                
                if epoch_error < best_error:
                    best_error = epoch_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                pbar.set_postfix({'error': f'{epoch_error:.6f}', 'val_acc': f'{val_accuracy:.4f}'})
                pbar.update(1)
                
                if patience_counter >= patience or epoch_error < self.config.error_threshold:
                    logger.info("Criterio de parada alcanzado.")
                    break
        
        return results
    
    def save_final_state(self):
        """Guarda los pesos y sesgos finales de la red."""
        self.result_manager.save_weights(self.weights, self.biases)
        logger.info("Pesos finales guardados.")        
    
    def backpropagation(self, activations: List[np.ndarray], pre_activations: List[np.ndarray], Y: np.ndarray):
        m = Y.shape[0]
        delta = (activations[-1] - Y.T) * ActivationFunctions.get_derivative(
            self.config.activation_functions[-1])(activations[-1])
        
        for l in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(delta, activations[l].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * ActivationFunctions.get_derivative(
                    self.config.activation_functions[l-1])(activations[l])
            
            # Actualizar pesos con momentum
            self.weights[l] -= self.config.learning_rate * dW
            self.biases[l] -= self.config.learning_rate * db
        
        return np.mean((Y.T - activations[-1])**2)

def main():
    load_dotenv()
    base_path = Path(os.getenv('DATA_PATH', './csvs'))
    file_paths = {
        "train_faces": base_path / "faces_train.csv",
        "train_names": base_path / "names_train.csv",
    }
    
    try:
        # Cargar y preparar datos
        X, Y, preprocessor, pca = load_data(file_paths)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Configurar la red para 19 clases
        config = NetworkConfig(
            input_size=X.shape[1],
            hidden_sizes=[120, 80, 40],  # Arquitectura piramidal
            output_size=Y.shape[1],      # 19 clases
            learning_rate=0.0003,        # Learning rate más bajo
            epoch_limit=400,             # Más épocas
            batch_size=32,               # Tamaño de batch
            activation_functions=['tansig', 'tansig', 'tansig', 'softmax']
        )
        
        nn = SimpleNN(config)
        results = nn.train(X_train, Y_train, X_val, Y_val)
        
        nn.result_manager.save_training_results(results)
        nn.save_final_state() # Llamada a la función para guardar el estado final
        logger.info("Entrenamiento completado y resultados guardados.")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    main()