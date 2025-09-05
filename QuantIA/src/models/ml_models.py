"""
Modelos ML parsimoniosos para trading.
Incluye Lasso, XGBoost, Random Forest y Ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

from .base import BaseModel, ModelEnsemble

logger = logging.getLogger(__name__)


class LassoModel(BaseModel):
    """
    Modelo Lasso para trading con regularización L1.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el modelo Lasso.
        
        Args:
            config: Configuración del modelo
        """
        default_config = {
            'alpha': 0.01,
            'max_iter': 1000,
            'tol': 1e-4,
            'random_state': 42,
            'selection': 'cyclic',
            'normalize': False,
            'feature_selection': True,
            'n_features': 50,
            'scaling': True
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("lasso", config)
        
        self.alpha = config['alpha']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.random_state = config['random_state']
        self.selection = config['selection']
        self.normalize = config['normalize']
        self.feature_selection = config['feature_selection']
        self.n_features = config['n_features']
        self.scaling = config['scaling']
        
        # Inicializar componentes
        self.model = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            selection=self.selection
        )
        
        self.scaler = StandardScaler() if self.scaling else None
        self.feature_selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LassoModel':
        """
        Entrena el modelo Lasso.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando modelo Lasso con {len(X)} muestras y {len(X.columns)} features")
        
        # Preparar datos
        X_processed, y_processed = self._prepare_data(X, y, fit=True)
        
        # Entrenar modelo
        self.model.fit(X_processed, y_processed)
        
        # Calcular importancia de features
        self._calculate_feature_importance(X.columns)
        
        # Calcular métricas de entrenamiento
        self._calculate_training_metrics(X_processed, y_processed)
        
        self.is_fitted = True
        logger.info("Modelo Lasso entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones del modelo Lasso.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos
        X_processed, _ = self._prepare_data(X, None, fit=False)
        
        # Generar predicciones continuas
        predictions_continuous = self.model.predict(X_processed)
        
        # Convertir a señales discretas
        predictions = self._continuous_to_signals(predictions_continuous)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos
        X_processed, _ = self._prepare_data(X, None, fit=False)
        
        # Generar predicciones continuas
        predictions_continuous = self.model.predict(X_processed)
        
        # Convertir a probabilidades
        probabilities = self._continuous_to_probabilities(predictions_continuous)
        
        return probabilities
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento/predicción.
        
        Args:
            X: Features
            y: Target (opcional)
            fit: Si es entrenamiento o predicción
            
        Returns:
            Tupla con (X_processed, y_processed)
        """
        # Remover NaN
        X_clean = X.fillna(X.median())
        
        if y is not None:
            y_clean = y.fillna(0)
        else:
            y_clean = None
        
        # Feature selection
        if self.feature_selection and fit:
            # Seleccionar mejores features
            self.feature_selector = SelectKBest(f_classif, k=min(self.n_features, len(X_clean.columns)))
            X_selected = self.feature_selector.fit_transform(X_clean, y_clean)
            self.selected_features = X_clean.columns[self.feature_selector.get_support()].tolist()
            logger.info(f"Seleccionadas {len(self.selected_features)} features de {len(X_clean.columns)}")
        elif self.feature_selection and not fit:
            # Usar features seleccionadas
            if self.selected_features is None:
                raise ValueError("Features no seleccionadas durante entrenamiento")
            X_selected = X_clean[self.selected_features].values
        else:
            X_selected = X_clean.values
            self.selected_features = X_clean.columns.tolist()
        
        # Scaling
        if self.scaling:
            if fit:
                X_processed = self.scaler.fit_transform(X_selected)
            else:
                X_processed = self.scaler.transform(X_selected)
        else:
            X_processed = X_selected
        
        return X_processed, y_clean.values if y_clean is not None else None
    
    def _continuous_to_signals(self, predictions_continuous: np.ndarray) -> np.ndarray:
        """
        Convierte predicciones continuas a señales discretas.
        
        Args:
            predictions_continuous: Predicciones continuas
            
        Returns:
            Señales discretas (-1, 0, 1)
        """
        # Usar percentiles para determinar umbrales
        p25 = np.percentile(predictions_continuous, 25)
        p75 = np.percentile(predictions_continuous, 75)
        
        signals = np.zeros_like(predictions_continuous)
        signals[predictions_continuous > p75] = 1  # Long
        signals[predictions_continuous < p25] = -1  # Short
        
        return signals.astype(int)
    
    def _continuous_to_probabilities(self, predictions_continuous: np.ndarray) -> np.ndarray:
        """
        Convierte predicciones continuas a probabilidades.
        
        Args:
            predictions_continuous: Predicciones continuas
            
        Returns:
            Probabilidades de cada clase
        """
        # Normalizar a [0, 1]
        min_val = np.min(predictions_continuous)
        max_val = np.max(predictions_continuous)
        
        if max_val == min_val:
            # Todas las predicciones son iguales
            return np.ones((len(predictions_continuous), 3)) / 3
        
        normalized = (predictions_continuous - min_val) / (max_val - min_val)
        
        # Convertir a probabilidades
        probabilities = []
        for val in normalized:
            if val > 0.6:
                prob = [0.1, 0.1, 0.8]  # Long
            elif val < 0.4:
                prob = [0.8, 0.1, 0.1]  # Short
            else:
                prob = [0.1, 0.8, 0.1]  # Neutral
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> None:
        """
        Calcula importancia de features.
        
        Args:
            feature_names: Nombres de features
        """
        if self.selected_features is None:
            return
        
        # Obtener coeficientes
        coefficients = self.model.coef_
        
        # Crear diccionario de importancia
        self.feature_importance = {}
        for i, feature in enumerate(self.selected_features):
            self.feature_importance[feature] = abs(coefficients[i])
        
        # Normalizar importancia
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total_importance
    
    def _calculate_training_metrics(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calcula métricas de entrenamiento.
        
        Args:
            X: Features procesadas
            y: Target procesado
        """
        predictions = self.predict(pd.DataFrame(X, columns=self.selected_features))
        
        # Calcular métricas básicas
        hit_ratio = self.calculate_hit_ratio(pd.Series(predictions * y))
        profit_factor = self.calculate_profit_factor(pd.Series(predictions * y))
        
        self.performance_metrics = {
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            'total_signals': len(predictions),
            'long_signals': (predictions == 1).sum(),
            'short_signals': (predictions == -1).sum(),
            'neutral_signals': (predictions == 0).sum(),
            'n_features_used': len(self.selected_features) if self.selected_features else len(X[0]),
            'alpha': self.alpha
        }
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.model = model_object['model']
        self.scaler = model_object['scaler']
        self.feature_selector = model_object['feature_selector']
        self.selected_features = model_object['selected_features']
        self.feature_importance = model_object['feature_importance']
        self.performance_metrics = model_object['performance_metrics']


class XGBoostModel(BaseModel):
    """
    Modelo XGBoost para trading con regularización.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el modelo XGBoost.
        
        Args:
            config: Configuración del modelo
        """
        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'feature_selection': True,
            'n_features': 100,
            'early_stopping_rounds': 10
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("xgboost", config)
        
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.learning_rate = config['learning_rate']
        self.subsample = config['subsample']
        self.colsample_bytree = config['colsample_bytree']
        self.reg_alpha = config['reg_alpha']
        self.reg_lambda = config['reg_lambda']
        self.random_state = config['random_state']
        self.n_jobs = config['n_jobs']
        self.feature_selection = config['feature_selection']
        self.n_features = config['n_features']
        self.early_stopping_rounds = config['early_stopping_rounds']
        
        # Inicializar modelo
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective='multi:softprob',
            num_class=3
        )
        
        self.feature_selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """
        Entrena el modelo XGBoost.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando modelo XGBoost con {len(X)} muestras y {len(X.columns)} features")
        
        # Preparar datos
        X_processed, y_processed = self._prepare_data(X, y, fit=True)
        
        # Entrenar modelo
        self.model.fit(X_processed, y_processed)
        
        # Calcular importancia de features
        self._calculate_feature_importance(X.columns)
        
        # Calcular métricas de entrenamiento
        self._calculate_training_metrics(X_processed, y_processed)
        
        self.is_fitted = True
        logger.info("Modelo XGBoost entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones del modelo XGBoost.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos
        X_processed, _ = self._prepare_data(X, None, fit=False)
        
        # Generar predicciones
        predictions = self.model.predict(X_processed)
        
        # Convertir de [0, 1, 2] a [-1, 0, 1]
        predictions = predictions - 1
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos
        X_processed, _ = self._prepare_data(X, None, fit=False)
        
        # Generar probabilidades
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento/predicción.
        
        Args:
            X: Features
            y: Target (opcional)
            fit: Si es entrenamiento o predicción
            
        Returns:
            Tupla con (X_processed, y_processed)
        """
        # Remover NaN
        X_clean = X.fillna(X.median())
        
        if y is not None:
            y_clean = y.fillna(0)
        else:
            y_clean = None
        
        # Feature selection
        if self.feature_selection and fit:
            # Usar importancia de XGBoost para selección
            temp_model = xgb.XGBClassifier(n_estimators=50, random_state=self.random_state)
            temp_model.fit(X_clean, y_clean)
            
            # Obtener importancia de features
            importance = temp_model.feature_importances_
            feature_importance = list(zip(X_clean.columns, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Seleccionar top features
            self.selected_features = [f[0] for f in feature_importance[:self.n_features]]
            logger.info(f"Seleccionadas {len(self.selected_features)} features de {len(X_clean.columns)}")
        
        if self.selected_features is not None:
            X_processed = X_clean[self.selected_features].values
        else:
            X_processed = X_clean.values
            self.selected_features = X_clean.columns.tolist()
        
        if y_clean is not None:
            # Convertir target de [-1, 0, 1] a [0, 1, 2]
            y_processed = y_clean.values + 1
        else:
            y_processed = None
        
        return X_processed, y_processed
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> None:
        """
        Calcula importancia de features.
        
        Args:
            feature_names: Nombres de features
        """
        if self.selected_features is None:
            return
        
        # Obtener importancia de XGBoost
        importance = self.model.feature_importances_
        
        # Crear diccionario de importancia
        self.feature_importance = {}
        for i, feature in enumerate(self.selected_features):
            self.feature_importance[feature] = importance[i]
        
        # Normalizar importancia
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total_importance
    
    def _calculate_training_metrics(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calcula métricas de entrenamiento.
        
        Args:
            X: Features procesadas
            y: Target procesado
        """
        predictions = self.predict(pd.DataFrame(X, columns=self.selected_features))
        
        # Convertir y de [0, 1, 2] a [-1, 0, 1]
        y_original = y - 1
        
        # Calcular métricas básicas
        hit_ratio = self.calculate_hit_ratio(pd.Series(predictions * y_original))
        profit_factor = self.calculate_profit_factor(pd.Series(predictions * y_original))
        
        self.performance_metrics = {
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            'total_signals': len(predictions),
            'long_signals': (predictions == 1).sum(),
            'short_signals': (predictions == -1).sum(),
            'neutral_signals': (predictions == 0).sum(),
            'n_features_used': len(self.selected_features) if self.selected_features else len(X[0]),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'model': self.model,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.model = model_object['model']
        self.feature_selector = model_object['feature_selector']
        self.selected_features = model_object['selected_features']
        self.feature_importance = model_object['feature_importance']
        self.performance_metrics = model_object['performance_metrics']


class RandomForestModel(BaseModel):
    """
    Modelo Random Forest para trading.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el modelo Random Forest.
        
        Args:
            config: Configuración del modelo
        """
        default_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'feature_selection': True,
            'n_features': 100
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("random_forest", config)
        
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.min_samples_split = config['min_samples_split']
        self.min_samples_leaf = config['min_samples_leaf']
        self.max_features = config['max_features']
        self.bootstrap = config['bootstrap']
        self.random_state = config['random_state']
        self.n_jobs = config['n_jobs']
        self.feature_selection = config['feature_selection']
        self.n_features = config['n_features']
        
        # Inicializar modelo
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight='balanced'
        )
        
        self.feature_selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """
        Entrena el modelo Random Forest.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando modelo Random Forest con {len(X)} muestras y {len(X.columns)} features")
        
        # Preparar datos
        X_processed, y_processed = self._prepare_data(X, y, fit=True)
        
        # Entrenar modelo
        self.model.fit(X_processed, y_processed)
        
        # Calcular importancia de features
        self._calculate_feature_importance(X.columns)
        
        # Calcular métricas de entrenamiento
        self._calculate_training_metrics(X_processed, y_processed)
        
        self.is_fitted = True
        logger.info("Modelo Random Forest entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones del modelo Random Forest.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos
        X_processed, _ = self._prepare_data(X, None, fit=False)
        
        # Generar predicciones
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Preparar datos
        X_processed, _ = self._prepare_data(X, None, fit=False)
        
        # Generar probabilidades
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento/predicción.
        
        Args:
            X: Features
            y: Target (opcional)
            fit: Si es entrenamiento o predicción
            
        Returns:
            Tupla con (X_processed, y_processed)
        """
        # Remover NaN
        X_clean = X.fillna(X.median())
        
        if y is not None:
            y_clean = y.fillna(0)
        else:
            y_clean = None
        
        # Feature selection
        if self.feature_selection and fit:
            # Usar RFE con Random Forest
            self.feature_selector = RFE(
                RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                n_features_to_select=min(self.n_features, len(X_clean.columns))
            )
            X_selected = self.feature_selector.fit_transform(X_clean, y_clean)
            self.selected_features = X_clean.columns[self.feature_selector.get_support()].tolist()
            logger.info(f"Seleccionadas {len(self.selected_features)} features de {len(X_clean.columns)}")
        elif self.feature_selection and not fit:
            # Usar features seleccionadas
            if self.selected_features is None:
                raise ValueError("Features no seleccionadas durante entrenamiento")
            X_selected = X_clean[self.selected_features].values
        else:
            X_selected = X_clean.values
            self.selected_features = X_clean.columns.tolist()
        
        return X_selected, y_clean.values if y_clean is not None else None
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> None:
        """
        Calcula importancia de features.
        
        Args:
            feature_names: Nombres de features
        """
        if self.selected_features is None:
            return
        
        # Obtener importancia de Random Forest
        importance = self.model.feature_importances_
        
        # Crear diccionario de importancia
        self.feature_importance = {}
        for i, feature in enumerate(self.selected_features):
            self.feature_importance[feature] = importance[i]
        
        # Normalizar importancia
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total_importance
    
    def _calculate_training_metrics(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calcula métricas de entrenamiento.
        
        Args:
            X: Features procesadas
            y: Target procesado
        """
        predictions = self.predict(pd.DataFrame(X, columns=self.selected_features))
        
        # Calcular métricas básicas
        hit_ratio = self.calculate_hit_ratio(pd.Series(predictions * y))
        profit_factor = self.calculate_profit_factor(pd.Series(predictions * y))
        
        self.performance_metrics = {
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            'total_signals': len(predictions),
            'long_signals': (predictions == 1).sum(),
            'short_signals': (predictions == -1).sum(),
            'neutral_signals': (predictions == 0).sum(),
            'n_features_used': len(self.selected_features) if self.selected_features else len(X[0]),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'model': self.model,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.model = model_object['model']
        self.feature_selector = model_object['feature_selector']
        self.selected_features = model_object['selected_features']
        self.feature_importance = model_object['feature_importance']
        self.performance_metrics = model_object['performance_metrics']


class MLEnsemble(BaseModel):
    """
    Ensemble de modelos ML con diferentes algoritmos.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el ensemble ML.
        
        Args:
            config: Configuración del ensemble
        """
        default_config = {
            'models': ['lasso', 'xgboost', 'random_forest'],
            'weights': None,  # Se calcularán automáticamente
            'voting_method': 'soft',  # 'hard' o 'soft'
            'performance_weighting': True,
            'min_models': 2
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("ml_ensemble", config)
        
        self.models_config = config['models']
        self.weights = config['weights']
        self.voting_method = config['voting_method']
        self.performance_weighting = config['performance_weighting']
        self.min_models = config['min_models']
        
        # Inicializar modelos
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Inicializa los modelos del ensemble."""
        for model_name in self.models_config:
            if model_name == 'lasso':
                self.models[model_name] = LassoModel()
            elif model_name == 'xgboost':
                self.models[model_name] = XGBoostModel()
            elif model_name == 'random_forest':
                self.models[model_name] = RandomForestModel()
            else:
                logger.warning(f"Modelo desconocido: {model_name}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLEnsemble':
        """
        Entrena el ensemble ML.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Ensemble entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando ensemble ML con {len(self.models)} modelos")
        
        # Entrenar cada modelo
        for model_name, model in self.models.items():
            try:
                logger.info(f"Entrenando modelo: {model_name}")
                model.fit(X, y)
                logger.info(f"Modelo {model_name} entrenado exitosamente")
            except Exception as e:
                logger.error(f"Error entrenando modelo {model_name}: {str(e)}")
        
        # Calcular pesos basados en performance
        if self.performance_weighting:
            self._calculate_weights(X, y)
        
        # Calcular métricas de entrenamiento
        self._calculate_training_metrics(X, y)
        
        self.is_fitted = True
        logger.info("Ensemble ML entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones del ensemble ML.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble no está entrenado")
        
        # Obtener predicciones de todos los modelos
        all_predictions = []
        valid_models = []
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    all_predictions.append(pred)
                    valid_models.append(model_name)
                except Exception as e:
                    logger.error(f"Error prediciendo con modelo {model_name}: {str(e)}")
        
        if len(all_predictions) < self.min_models:
            raise ValueError(f"Se necesitan al menos {self.min_models} modelos válidos")
        
        # Combinar predicciones
        if self.voting_method == 'hard':
            # Votación mayoritaria
            predictions = self._hard_voting(all_predictions, valid_models)
        else:
            # Votación suave (promedio ponderado)
            predictions = self._soft_voting(all_predictions, valid_models)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades del ensemble ML.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        if not self.is_fitted:
            raise ValueError("Ensemble no está entrenado")
        
        # Obtener probabilidades de todos los modelos
        all_probabilities = []
        valid_models = []
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    prob = model.predict_proba(X)
                    all_probabilities.append(prob)
                    valid_models.append(model_name)
                except Exception as e:
                    logger.error(f"Error prediciendo probabilidades con modelo {model_name}: {str(e)}")
        
        if len(all_probabilities) < self.min_models:
            raise ValueError(f"Se necesitan al menos {self.min_models} modelos válidos")
        
        # Combinar probabilidades
        probabilities = self._combine_probabilities(all_probabilities, valid_models)
        
        return probabilities
    
    def _hard_voting(self, predictions: List[np.ndarray], model_names: List[str]) -> np.ndarray:
        """
        Votación mayoritaria (hard voting).
        
        Args:
            predictions: Lista de predicciones
            model_names: Nombres de modelos
            
        Returns:
            Predicciones combinadas
        """
        predictions_array = np.array(predictions)
        
        # Votación mayoritaria
        final_predictions = []
        for i in range(len(predictions[0])):
            votes = predictions_array[:, i]
            # Contar votos
            vote_counts = np.bincount(votes + 1, minlength=3)  # [-1, 0, 1] -> [0, 1, 2]
            final_pred = np.argmax(vote_counts) - 1  # Convertir de vuelta a [-1, 0, 1]
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def _soft_voting(self, predictions: List[np.ndarray], model_names: List[str]) -> np.ndarray:
        """
        Votación suave (promedio ponderado).
        
        Args:
            predictions: Lista de predicciones
            model_names: Nombres de modelos
            
        Returns:
            Predicciones combinadas
        """
        if self.weights is None:
            # Pesos uniformes
            weights = np.ones(len(predictions)) / len(predictions)
        else:
            # Usar pesos calculados
            weights = [self.weights.get(name, 0) for name in model_names]
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalizar
        
        # Promedio ponderado
        predictions_array = np.array(predictions)
        weighted_predictions = np.average(predictions_array, axis=0, weights=weights)
        
        # Convertir a señales discretas
        final_predictions = np.round(weighted_predictions).astype(int)
        
        return final_predictions
    
    def _combine_probabilities(self, probabilities: List[np.ndarray], model_names: List[str]) -> np.ndarray:
        """
        Combina probabilidades de múltiples modelos.
        
        Args:
            probabilities: Lista de probabilidades
            model_names: Nombres de modelos
            
        Returns:
            Probabilidades combinadas
        """
        if self.weights is None:
            # Pesos uniformes
            weights = np.ones(len(probabilities)) / len(probabilities)
        else:
            # Usar pesos calculados
            weights = [self.weights.get(name, 0) for name in model_names]
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalizar
        
        # Promedio ponderado de probabilidades
        probabilities_array = np.array(probabilities)
        combined_probabilities = np.average(probabilities_array, axis=0, weights=weights)
        
        return combined_probabilities
    
    def _calculate_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calcula pesos basados en performance de cada modelo.
        
        Args:
            X: Features
            y: Target
        """
        model_performances = {}
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    # Calcular Sharpe ratio como métrica de performance
                    predictions = model.predict(X)
                    strategy_returns = predictions * y
                    
                    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
                        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                    else:
                        sharpe = 0.0
                    
                    model_performances[model_name] = max(0, sharpe)  # Solo valores positivos
                    
                except Exception as e:
                    logger.error(f"Error calculando performance de {model_name}: {str(e)}")
                    model_performances[model_name] = 0.0
        
        # Normalizar pesos
        total_performance = sum(model_performances.values())
        if total_performance > 0:
            self.weights = {name: perf / total_performance for name, perf in model_performances.items()}
        else:
            # Pesos uniformes si no hay performance positiva
            self.weights = {name: 1.0 / len(model_performances) for name in model_performances.keys()}
        
        logger.info(f"Pesos calculados: {self.weights}")
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calcula métricas de entrenamiento del ensemble.
        
        Args:
            X: Features
            y: Target
        """
        predictions = self.predict(X)
        
        # Calcular métricas básicas
        hit_ratio = self.calculate_hit_ratio(pd.Series(predictions * y))
        profit_factor = self.calculate_profit_factor(pd.Series(predictions * y))
        
        self.performance_metrics = {
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            'total_signals': len(predictions),
            'long_signals': (predictions == 1).sum(),
            'short_signals': (predictions == -1).sum(),
            'neutral_signals': (predictions == 0).sum(),
            'n_models': len([m for m in self.models.values() if m.is_fitted]),
            'weights': self.weights,
            'voting_method': self.voting_method
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene importancia de features del ensemble.
        
        Returns:
            Importancia promedio de features
        """
        all_importance = []
        
        for model_name, model in self.models.items():
            if model.is_fitted and hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                if importance:
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
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'models': {name: model._get_model_object() for name, model in self.models.items()},
            'weights': self.weights,
            'performance_metrics': self.performance_metrics
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.weights = model_object['weights']
        self.performance_metrics = model_object['performance_metrics']
        
        # Cargar modelos individuales
        for model_name, model_data in model_object['models'].items():
            if model_name in self.models:
                self.models[model_name]._set_model_object(model_data)

