"""
Clase base para todos los modelos de trading.
Define la interfaz común para modelos baseline y ML.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Clase base abstracta para todos los modelos de trading.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Inicializa el modelo base.
        
        Args:
            name: Nombre del modelo
            config: Configuración del modelo
        """
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.feature_importance = {}
        self.performance_metrics = {}
        self.created_at = datetime.now()
        
        logger.info(f"Modelo '{name}' inicializado")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Entrena el modelo.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones del modelo
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de predicción
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene importancia de features.
        
        Returns:
            Diccionario con importancia de features
        """
        return self.feature_importance
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Obtiene métricas de performance.
        
        Returns:
            Diccionario con métricas de performance
        """
        return self.performance_metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en archivo.
        
        Args:
            filepath: Ruta del archivo
        """
        model_data = {
            'name': self.name,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat(),
            'model_object': self._get_model_object()
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado: {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Carga el modelo desde archivo.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Modelo cargado
        """
        model_data = joblib.load(filepath)
        
        self.name = model_data['name']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.feature_importance = model_data['feature_importance']
        self.performance_metrics = model_data['performance_metrics']
        self.created_at = datetime.fromisoformat(model_data['created_at'])
        
        self._set_model_object(model_data['model_object'])
        
        logger.info(f"Modelo cargado: {filepath}")
        return self
    
    @abstractmethod
    def _get_model_object(self) -> Any:
        """
        Obtiene el objeto del modelo para guardar.
        
        Returns:
            Objeto del modelo
        """
        pass
    
    @abstractmethod
    def _set_model_object(self, model_object: Any) -> None:
        """
        Establece el objeto del modelo desde carga.
        
        Args:
            model_object: Objeto del modelo
        """
        pass
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series = None) -> bool:
        """
        Valida los datos de entrada.
        
        Args:
            X: Features
            y: Target (opcional)
            
        Returns:
            True si los datos son válidos
        """
        if X.empty:
            logger.error("Features vacíos")
            return False
        
        if y is not None and len(y) != len(X):
            logger.error("Longitud de features y target no coincide")
            return False
        
        if X.isnull().all().any():
            logger.error("Algunas columnas están completamente vacías")
            return False
        
        return True
    
    def calculate_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calcula retornos para diferentes períodos.
        
        Args:
            prices: Serie de precios
            periods: Número de períodos
            
        Returns:
            Serie de retornos
        """
        return prices.pct_change(periods=periods)
    
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calcula volatilidad rolling.
        
        Args:
            returns: Serie de retornos
            window: Ventana de cálculo
            
        Returns:
            Serie de volatilidad
        """
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sharpe ratio.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anual
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calcula maximum drawdown.
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_hit_ratio(self, returns: pd.Series) -> float:
        """
        Calcula hit ratio (porcentaje de trades ganadores).
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Hit ratio
        """
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).sum() / len(returns)
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calcula profit factor.
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Profit factor
        """
        if len(returns) == 0:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss


class TradingSignal:
    """
    Clase para manejar señales de trading.
    """
    
    def __init__(self, symbol: str, timestamp: datetime, signal: float, 
                 confidence: float = 1.0, metadata: Dict[str, Any] = None):
        """
        Inicializa una señal de trading.
        
        Args:
            symbol: Símbolo del instrumento
            timestamp: Timestamp de la señal
            signal: Señal (-1, 0, 1)
            confidence: Confianza de la señal (0-1)
            metadata: Metadatos adicionales
        """
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal = signal  # -1: Short, 0: Neutral, 1: Long
        self.confidence = confidence
        self.metadata = metadata or {}
        
        # Validar señal
        if signal not in [-1, 0, 1]:
            raise ValueError(f"Señal inválida: {signal}. Debe ser -1, 0, o 1")
        
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confianza inválida: {confidence}. Debe estar entre 0 y 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la señal a diccionario.
        
        Returns:
            Diccionario con la señal
        """
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """Representación string de la señal."""
        signal_text = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}[self.signal]
        return f"TradingSignal({self.symbol}, {self.timestamp}, {signal_text}, conf={self.confidence:.2f})"


class ModelEnsemble:
    """
    Ensemble de múltiples modelos.
    """
    
    def __init__(self, name: str, models: List[BaseModel] = None):
        """
        Inicializa el ensemble.
        
        Args:
            name: Nombre del ensemble
            models: Lista de modelos
        """
        self.name = name
        self.models = models or []
        self.weights = None
        self.is_fitted = False
        
        logger.info(f"Ensemble '{name}' inicializado con {len(self.models)} modelos")
    
    def add_model(self, model: BaseModel) -> None:
        """
        Agrega un modelo al ensemble.
        
        Args:
            model: Modelo a agregar
        """
        self.models.append(model)
        logger.info(f"Modelo '{model.name}' agregado al ensemble")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelEnsemble':
        """
        Entrena todos los modelos del ensemble.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Ensemble entrenado
        """
        logger.info(f"Entrenando ensemble con {len(self.models)} modelos")
        
        for model in self.models:
            try:
                model.fit(X, y)
                logger.info(f"Modelo '{model.name}' entrenado exitosamente")
            except Exception as e:
                logger.error(f"Error entrenando modelo '{model.name}': {str(e)}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """
        Genera predicciones del ensemble.
        
        Args:
            X: Features para predicción
            method: Método de combinación ('average', 'weighted', 'voting')
            
        Returns:
            Predicciones del ensemble
        """
        if not self.is_fitted:
            raise ValueError("Ensemble no está entrenado")
        
        if not self.models:
            raise ValueError("No hay modelos en el ensemble")
        
        predictions = []
        for model in self.models:
            if model.is_fitted:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("Ningún modelo está entrenado")
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            if self.weights is None:
                self.weights = np.ones(len(predictions)) / len(predictions)
            return np.average(predictions, axis=0, weights=self.weights)
        elif method == 'voting':
            # Para clasificación binaria
            votes = np.sum(predictions > 0, axis=0)
            return np.where(votes > len(predictions) / 2, 1, -1)
        else:
            raise ValueError(f"Método de combinación inválido: {method}")
    
    def predict_proba(self, X: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """
        Genera probabilidades del ensemble.
        
        Args:
            X: Features para predicción
            method: Método de combinación
            
        Returns:
            Probabilidades del ensemble
        """
        if not self.is_fitted:
            raise ValueError("Ensemble no está entrenado")
        
        probabilities = []
        for model in self.models:
            if model.is_fitted:
                prob = model.predict_proba(X)
                probabilities.append(prob)
        
        if not probabilities:
            raise ValueError("Ningún modelo está entrenado")
        
        probabilities = np.array(probabilities)
        
        if method == 'average':
            return np.mean(probabilities, axis=0)
        elif method == 'weighted':
            if self.weights is None:
                self.weights = np.ones(len(probabilities)) / len(probabilities)
            return np.average(probabilities, axis=0, weights=self.weights)
        else:
            raise ValueError(f"Método de combinación inválido: {method}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene importancia de features del ensemble.
        
        Returns:
            Importancia promedio de features
        """
        if not self.models:
            return {}
        
        all_importance = []
        for model in self.models:
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                all_importance.append(importance)
        
        if not all_importance:
            return {}
        
        # Promediar importancia de todos los modelos
        feature_names = set()
        for importance in all_importance:
            feature_names.update(importance.keys())
        
        ensemble_importance = {}
        for feature in feature_names:
            values = [imp.get(feature, 0) for imp in all_importance]
            ensemble_importance[feature] = np.mean(values)
        
        return ensemble_importance
    
    def save_ensemble(self, filepath: str) -> None:
        """
        Guarda el ensemble en archivo.
        
        Args:
            filepath: Ruta del archivo
        """
        ensemble_data = {
            'name': self.name,
            'models': [model._get_model_object() for model in self.models],
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble guardado: {filepath}")
    
    def load_ensemble(self, filepath: str) -> 'ModelEnsemble':
        """
        Carga el ensemble desde archivo.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Ensemble cargado
        """
        ensemble_data = joblib.load(filepath)
        
        self.name = ensemble_data['name']
        self.weights = ensemble_data['weights']
        self.is_fitted = ensemble_data['is_fitted']
        
        # Cargar modelos (esto requeriría implementación específica)
        # self.models = [load_model_from_object(obj) for obj in ensemble_data['models']]
        
        logger.info(f"Ensemble cargado: {filepath}")
        return self

