import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV 
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional, Union
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import psutil
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchNormalization:
    """Implementación de Batch Normalization."""
    def __init__(self, input_dim: int):
        self.epsilon = 1e-8
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        self.momentum = 0.9
        self.input_dim = input_dim

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Aplica batch normalization al input.
        
        Args:
            x: Input data de shape (batch_size, input_dim)
            training: Si es True, actualiza running statistics
            
        Returns:
            Data normalizada
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {x.shape[1]}")
            
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
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
    activation_functions: Optional[List[str]] = None
    batch_size: int = 16
    lambda_: float = 0.01  # L2 regularization parameter
    early_stopping_patience: int = 5
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        if self.activation_functions is None:
            self.activation_functions = ['tansig'] * (len(self.hidden_sizes)) + ['softmax']
        if len(self.activation_functions) != len(self.hidden_sizes) + 1:
            raise ValueError("Number of activation functions must match number of layers")

    def to_dict(self) -> Dict:
        """Convierte la configuración a diccionario."""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'epoch_limit': self.epoch_limit,
            'error_threshold': self.error_threshold,
            'activation_functions': self.activation_functions,
            'batch_size': self.batch_size,
            'lambda_': self.lambda_,
            'early_stopping_patience': self.early_stopping_patience,
            'checkpoint_frequency': self.checkpoint_frequency
        }

class ActivationFunctions:
    """Implementación de funciones de activación y sus derivadas."""
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x_safe = x - np.max(x, axis=0, keepdims=True)  # Para estabilidad numérica
        exps = np.exp(x_safe)
        return exps / np.sum(exps, axis=0, keepdims=True)
    
    @staticmethod
    def tansig(x: np.ndarray) -> np.ndarray:
        return 2.0 / (1.0 + np.exp(-2.0 * np.clip(x, -500, 500))) - 1.0
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def get_derivative(fname: str) -> Callable:
        derivatives = {
            "softmax": lambda y: y * (1.0 - y),
            "tansig": lambda y: 1.0 - y**2,
            "relu": lambda y: np.where(y > 0, 1, 0)
        }
        if fname not in derivatives:
            raise ValueError(f"Unknown activation function: {fname}")
        return derivatives[fname]

class DataPreprocessor:
    """Clase para preprocesamiento de datos."""
    def __init__(self):
        self.mean = None
        self.zca_matrix = None
        self.pca = None
        self.label_binarizer = None
        
    def zca_whitening(self, X: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """Aplica ZCA whitening a los datos."""
        if X.ndim != 2:
            raise ValueError("Input debe ser 2D array")
            
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
    
    def save(self, path: Path):
        """Guarda el estado del preprocessor."""
        with open(path / 'preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'zca_matrix': self.zca_matrix,
                'pca': self.pca,
                'label_binarizer': self.label_binarizer
            }, f)

    def load(self, path: Path):
        """Carga el estado del preprocessor."""
        with open(path / 'preprocessor.pkl', 'rb') as f:
            state = pickle.load(f)
            self.mean = state['mean']
            self.zca_matrix = state['zca_matrix']
            self.pca = state['pca']
            self.label_binarizer = state['label_binarizer']

class ResultManager:
    """Clase para manejar el guardado de resultados y checkpoints."""
    def __init__(self, base_path: str = 'resultados_backpropagation'):
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_path = self.base_path / self.timestamp
        self.setup_directories()
        
    def setup_directories(self):
        """Crea las carpetas necesarias."""
        self.result_path.mkdir(parents=True, exist_ok=True)
        (self.result_path / 'checkpoints').mkdir(exist_ok=True)
        logger.info(f"Directorio de resultados creado en: {self.result_path}")
    
    def save_config(self, config: NetworkConfig):
        """Guarda la configuración de la red."""
        config_path = self.result_path / 'network_config.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        logger.info(f"Configuración guardada en: {config_path}")
    
    def save_training_results(self, results: List[Dict]):
        """Guarda los resultados del entrenamiento."""
        results_path = self.result_path / 'training_results.csv'
        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)
        logger.info(f"Resultados de entrenamiento guardados en: {results_path}")
    
    def save_checkpoint(self, epoch: int, model_state: Dict):
        """Guarda un checkpoint del modelo."""
        checkpoint_path = self.result_path / 'checkpoints' / f'checkpoint_epoch_{epoch}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model_state, f)
        logger.info(f"Checkpoint guardado en: {checkpoint_path}")
    
    def load_checkpoint(self, epoch: int) -> Dict:
        """Carga un checkpoint del modelo."""
        checkpoint_path = self.result_path / 'checkpoints' / f'checkpoint_epoch_{epoch}.pkl'
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)

class SimpleNN:
    """Implementación mejorada de una red neuronal feed-forward con barras de progreso."""
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.weights = []
        self.biases = []
        self.batch_norms = []
        self.initialize_weights()
        self.result_manager = ResultManager()
        self.result_manager.save_config(config)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def initialize_weights(self):
        """Inicializa los pesos usando inicialización Xavier/Glorot con barra de progreso."""
        logger.info("Inicializando pesos con inicialización Xavier/Glorot...")
        layer_sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        
        with tqdm(total=len(layer_sizes)-1, desc="Inicializando capas") as pbar:
            for i in range(len(layer_sizes) - 1):
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i + 1], layer_sizes[i])))
                self.biases.append(np.zeros((layer_sizes[i + 1], 1)))
                
                if i < len(layer_sizes) - 2:
                    self.batch_norms.append(BatchNormalization(layer_sizes[i + 1]))
                pbar.update(1)
    
    def get_batches(self, X: np.ndarray, Y: np.ndarray, batch_size: int):
        """Generator para obtener mini-batches con barra de progreso."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        with tqdm(total=n_batches, desc="Procesando batches", leave=False) as pbar:
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                yield X[batch_indices], Y[batch_indices]
                pbar.update(1)
    
    def train(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray) -> List[Dict]:
        """Entrena la red neuronal con barras de progreso detalladas."""
        results = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Iniciando entrenamiento...")
        
        # Barra de progreso principal para las épocas
        with tqdm(total=self.config.epoch_limit, desc="Épocas de entrenamiento") as epoch_pbar:
            for epoch in range(self.config.epoch_limit):
                # Training metrics
                epoch_loss = 0
                batch_count = 0
                
                # Barra de progreso para las métricas de entrenamiento
                train_metrics = tqdm(desc="Métricas de entrenamiento", leave=False)
                
                for batch_X, batch_Y in self.get_batches(X, Y, self.config.batch_size):
                    activations, pre_activations = self.forward_pass(batch_X, training=True)
                    batch_loss = self.backpropagation(activations, pre_activations, batch_Y)
                    epoch_loss += batch_loss
                    batch_count += 1
                    
                    # Actualizar métricas en tiempo real
                    train_metrics.set_postfix({
                        'batch_loss': f'{batch_loss:.6f}',
                        'avg_loss': f'{epoch_loss/batch_count:.6f}'
                    })
                
                epoch_loss /= batch_count
                train_metrics.close()
                
                # Validation con barra de progreso
                with tqdm(desc="Validación", leave=False) as val_pbar:
                    val_metrics = self.compute_metrics(X_val, Y_val)
                    val_pbar.set_postfix(val_metrics)
                    val_pbar.update(1)
                
                # Actualizar resultados
                results.append({
                    'epoch': epoch + 1,
                    'train_loss': float(epoch_loss),
                    'val_loss': float(val_metrics['total_loss']),
                    'val_accuracy': float(val_metrics['accuracy'])
                })
                
                # Early stopping check con barra de progreso
                with tqdm(desc="Verificando early stopping", total=1, leave=False) as es_pbar:
                    if val_metrics['total_loss'] < best_val_loss:
                        best_val_loss = val_metrics['total_loss']
                        patience_counter = 0
                        # Guardar checkpoint
                        self.save_checkpoint(epoch)
                    else:
                        patience_counter += 1
                    es_pbar.update(1)
                
                # Actualizar barra de progreso principal
                epoch_pbar.set_postfix({
                    'train_loss': f'{epoch_loss:.6f}',
                    'val_loss': f'{val_metrics["total_loss"]:.6f}',
                    'val_acc': f'{val_metrics["accuracy"]:.4f}'
                })
                epoch_pbar.update(1)
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping en época {epoch + 1}")
                    break
                
                # Guardar checkpoint periódico
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint(epoch)
        
        return results
    
    def save_checkpoint(self, epoch: int):
        """Guarda checkpoint con barra de progreso."""
        with tqdm(desc=f"Guardando checkpoint época {epoch}", total=1, leave=False) as cp_pbar:
            state = {
                'epoch': epoch,
                'weights': self.weights,
                'biases': self.biases,
                'batch_norms': self.batch_norms,
                'best_val_loss': self.best_val_loss
            }
            self.result_manager.save_checkpoint(epoch, state)
            cp_pbar.update(1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones con barra de progreso."""
        with tqdm(desc="Realizando predicciones", total=1) as pred_pbar:
            activations, _ = self.forward_pass(X, training=False)
            predictions = np.argmax(activations[-1], axis=0)
            pred_pbar.update(1)
        return predictions
    
    def compute_metrics(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """Calcula métricas con barra de progreso."""
        metrics = {}
        with tqdm(total=3, desc="Calculando métricas", leave=False) as metric_pbar:
            # Calcular predicciones
            activations, _ = self.forward_pass(X, training=False)
            predictions = np.argmax(activations[-1], axis=0)
            true_labels = np.argmax(Y, axis=1)
            metric_pbar.update(1)
            
            # Calcular accuracy
            metrics['accuracy'] = np.mean(predictions == true_labels)
            metric_pbar.update(1)
            
            # Calcular pérdidas
            loss = np.mean((Y.T - activations[-1])**2)
            l2_reg = sum(np.sum(np.square(W)) for W in self.weights)
            metrics['loss'] = loss
            metrics['total_loss'] = loss + self.config.lambda_ * l2_reg / (2 * X.shape[0])
            metric_pbar.update(1)
            
        return metrics

    def forward_pass(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza forward pass a través de la red.
        
        Args:
            X: Input data de shape (n_samples, input_size)
            training: Si es True, actualiza batch normalization
            
        Returns:
            Tuple de (activaciones, pre_activaciones)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
                
        with tqdm(total=len(self.weights), desc="Forward pass", leave=False) as fwd_pbar:
            activations = [X.T]  # Transponer para compatibilidad (input_size, n_samples)
            pre_activations = []
            
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                # Cálculo de la pre-activación
                Z = np.dot(W, activations[-1]) + b
                pre_activations.append(Z)
                
                # Aplicar batch normalization en capas ocultas
                if i < len(self.weights) - 1 and self.batch_norms:
                    Z = self.batch_norms[i].forward(Z.T, training).T
                
                # Aplicar función de activación
                activation_fn = getattr(ActivationFunctions, self.config.activation_functions[i])
                activations.append(activation_fn(Z))
                
                fwd_pbar.update(1)
                fwd_pbar.set_postfix({'layer': f'{i+1}/{len(self.weights)}'})
            
        return activations, pre_activations

    def backpropagation(self, activations: List[np.ndarray], pre_activations: List[np.ndarray], Y: np.ndarray) -> float:
        """
        Realiza backpropagation y actualiza los pesos.
        
        Args:
            activations: Lista de activaciones de cada capa
            pre_activations: Lista de pre-activaciones de cada capa
            Y: Labels objetivo
            
        Returns:
            Pérdida del batch actual
        """
        m = Y.shape[0]
        
        with tqdm(total=len(self.weights), desc="Backpropagation", leave=False) as back_pbar:
            # Calcular delta inicial
            delta = (activations[-1] - Y.T) * ActivationFunctions.get_derivative(
                self.config.activation_functions[-1])(activations[-1])
            
            # Calcular regularización L2
            L2_reg = 0
            for W in self.weights:
                L2_reg += np.sum(np.square(W))
            
            # Backward pass por cada capa
            for l in range(len(self.weights) - 1, -1, -1):
                # Calcular gradientes
                dW = np.dot(delta, activations[l].T) / m + self.config.lambda_ * self.weights[l] / m
                db = np.sum(delta, axis=1, keepdims=True) / m
                
                # Calcular delta para la siguiente capa
                if l > 0:
                    delta = np.dot(self.weights[l].T, delta) * ActivationFunctions.get_derivative(
                        self.config.activation_functions[l-1])(activations[l])
                
                # Actualizar pesos y biases
                self.weights[l] -= self.config.learning_rate * dW
                self.biases[l] -= self.config.learning_rate * db
                
                back_pbar.update(1)
                back_pbar.set_postfix({'layer': f'{l}/{len(self.weights)-1}'})
            
            # Calcular pérdida total incluyendo regularización
            loss = np.mean((Y.T - activations[-1])**2) + self.config.lambda_ * L2_reg / (2 * m)
            
        return loss

    def save_final_state(self):
        """Guarda el estado final del modelo."""
        with tqdm(desc="Guardando estado final", total=1) as save_pbar:
            state = {
                'weights': self.weights,
                'biases': self.biases,
                'batch_norms': self.batch_norms,
                'config': self.config
            }
            self.result_manager.save_weights(self.weights, self.biases)
            save_pbar.update(1)
            
def load_data(file_paths: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray, DataPreprocessor, PCA]:
    """
    Carga y prepara los datos con barras de progreso detalladas.
    
    Args:
        file_paths: Diccionario con rutas a los archivos de datos
        
    Returns:
        Tuple con datos procesados, preprocessor y PCA
    """
    try:
        logger.info(f"Cargando datos desde {file_paths['train_faces']} y {file_paths['train_names']}")
        
        # Validar archivos
        if not file_paths["train_faces"].exists() or not file_paths["train_names"].exists():
            raise FileNotFoundError("No se encuentran los archivos de entrenamiento.")
        
        # Barra de progreso principal para carga de datos
        with tqdm(total=6, desc="Preparación de datos") as data_pbar:
            # 1. Leer datos
            logger.info("Leyendo datos de entrenamiento...")
            total_lines = sum(1 for line in open(file_paths["train_faces"]))
            
            training_faces = []
            with tqdm(total=total_lines, desc="Cargando imágenes", leave=False) as load_pbar:
                for chunk in pd.read_csv(file_paths["train_faces"], header=None, chunksize=50):
                    chunk_array = chunk.values
                    training_faces.append(chunk.values)
                    load_pbar.update(len(chunk))
            
                    # Limpiar memoria después de cada chunk
                    del chunk
                    gc.collect()
                    
            training_faces = np.vstack(training_faces).astype('float64')
            training_names = pd.read_csv(file_paths["train_names"], header=None).values
            data_pbar.update(1)
            
            # 2. Validar datos
            with tqdm(desc="Validando datos", total=1, leave=False) as val_pbar:
                if training_faces.shape[0] <= 1 or training_faces.shape[1] <= 1:
                    raise ValueError("El conjunto de datos no tiene suficientes filas o columnas.")
                val_pbar.update(1)
            data_pbar.update(1)
            
            # 3. Normalizar datos
            with tqdm(desc="Normalizando datos", total=1, leave=False) as norm_pbar:
                training_faces = training_faces / 255.0
                norm_pbar.update(1)
            data_pbar.update(1)
            
            # 4. Aplicar ZCA whitening
            preprocessor = DataPreprocessor()
            training_faces_whitened = preprocessor.zca_whitening(training_faces)
            data_pbar.update(1)
            
            # 5. Aplicar PCA
            logger.info("Aplicando PCA...")
            with tqdm(desc="PCA", total=1, leave=False) as pca_pbar:
                n_components = min(60, training_faces_whitened.shape[0], training_faces_whitened.shape[1])
                pca = PCA(n_components=n_components, svd_solver='full')
                training_Z = pca.fit_transform(training_faces_whitened)
                pca_pbar.update(1)
            data_pbar.update(1)
            
            # 6. Preparar etiquetas
            with tqdm(desc="Preparando etiquetas", total=1, leave=False) as label_pbar:
                label_binarizer = LabelBinarizer()
                training_names_num = label_binarizer.fit_transform(training_names)
                label_pbar.update(1)
            data_pbar.update(1)
            
            logger.info("Datos cargados y preparados exitosamente")
            return training_Z, training_names_num, preprocessor, pca
    
    except Exception as e:
        logger.error(f"Error al cargar los datos: {str(e)}")
        raise
    def forward_pass(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza el forward pass a través de la red neuronal.
        
        Args:
            X: Input data de shape (n_samples, input_size)
            training: Si es True, actualiza batch normalization
            
        Returns:
            Tuple con (lista de activaciones, lista de pre-activaciones)
        """
        with tqdm(total=len(self.weights), desc="Forward pass", leave=False) as fp_pbar:
            if X.ndim == 1:
                X = X.reshape(1, -1)
                
            activations = [X.T]  # Transponer para mantener consistencia (input_size, n_samples)
            pre_activations = []
            
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                # Calcular pre-activación
                Z = np.dot(W, activations[-1]) + b
                pre_activations.append(Z)
                
                # Aplicar batch normalization en capas ocultas
                if i < len(self.weights) - 1 and len(self.batch_norms) > i:
                    Z = self.batch_norms[i].forward(Z.T, training).T
                
                # Aplicar función de activación
                activation_fn = getattr(ActivationFunctions, self.config.activation_functions[i])
                activations.append(activation_fn(Z))
                
                fp_pbar.update(1)
                fp_pbar.set_postfix({
                    'layer': f'{i+1}/{len(self.weights)}',
                    'shape': str(activations[-1].shape)
                })

                if not training:
                        for i in range(len(activations)-1):
                            del activations[i]
                            gc.collect()        
        
        return activations, pre_activations

def main():
    """Función principal con barras de progreso para cada etapa."""
    load_dotenv()
    base_path = Path(os.getenv('DATA_PATH', './csvs'))
    file_paths = {
        "train_faces": base_path / "faces_train.csv",
        "train_names": base_path / "names_train.csv",
    }
    
    with tqdm(total=5, desc="Proceso completo") as main_pbar:
        try:
            # 1. Cargar y preparar datos
            logger.info("Cargando datos...")
            X, Y, preprocessor, pca = load_data(file_paths)
            main_pbar.update(1)
            
            # 2. Split de datos
            logger.info("Dividiendo datos...")
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
            main_pbar.update(1)
            
            # 3. Configurar la red
            logger.info("Configurando red neuronal...")
            config = NetworkConfig(
                input_size=X.shape[1],
                hidden_sizes=[120, 80, 40],
                output_size=Y.shape[1],
                learning_rate=0.0003,
                epoch_limit=400,
                batch_size=16,
                activation_functions=['tansig', 'tansig', 'tansig', 'softmax']
            )
            main_pbar.update(1)
            
            # 4. Entrenar modelo
            logger.info("Iniciando entrenamiento...")
            nn = SimpleNN(config)
            results = nn.train(X_train, Y_train, X_val, Y_val)
            main_pbar.update(1)
            
            # 5. Guardar resultados
            logger.info("Guardando resultados...")
            nn.result_manager.save_training_results(results)
            nn.save_final_state()
            main_pbar.update(1)
            
            logger.info("¡Entrenamiento completado exitosamente!")
            
        except Exception as e:
            logger.error(f"Error durante la ejecución: {str(e)}")
            raise

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    memory_percent = process.memory_percent()
    
    print(f"Memoria usada: {memory_usage:.2f} MB ({memory_percent:.1f}%)")
    print(f"Memoria disponible: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
    
    if memory_usage > 1000:  # 1GB
        print("Limpiando memoria...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()