"""
Entrenador especializado para modelos ML.
Maneja hyperparameter tuning, feature selection y validación avanzada.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')

from .ml_models import LassoModel, XGBoostModel, RandomForestModel, MLEnsemble
from .base import BaseModel
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MLModelTrainer:
    """
    Entrenador especializado para modelos ML con hyperparameter tuning.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el entrenador ML.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.models = {}
        self.training_results = {}
        self.validation_results = {}
        self.hyperparameter_results = {}
        
        # Configuración de entrenamiento ML
        self.ml_config = self.config.get('modelos', {}).get('ml', {})
        self.hyperparameter_tuning = self.ml_config.get('hyperparameter_tuning', True)
        self.feature_selection = self.ml_config.get('feature_selection', True)
        self.cv_folds = self.ml_config.get('cv_folds', 5)
        self.n_trials = self.ml_config.get('n_trials', 50)
        
        logger.info("MLModelTrainer inicializado")
    
    def create_ml_models(self) -> Dict[str, BaseModel]:
        """
        Crea modelos ML con configuración optimizada.
        
        Returns:
            Diccionario con modelos ML
        """
        models = {}
        
        # Modelo Lasso
        lasso_config = {
            'alpha': 0.01,
            'max_iter': 1000,
            'feature_selection': True,
            'n_features': 50,
            'scaling': True
        }
        models['lasso'] = LassoModel(lasso_config)
        
        # Modelo XGBoost
        xgb_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'feature_selection': True,
            'n_features': 100
        }
        models['xgboost'] = XGBoostModel(xgb_config)
        
        # Modelo Random Forest
        rf_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'feature_selection': True,
            'n_features': 100
        }
        models['random_forest'] = RandomForestModel(rf_config)
        
        # Ensemble ML
        ensemble_config = {
            'models': ['lasso', 'xgboost', 'random_forest'],
            'voting_method': 'soft',
            'performance_weighting': True
        }
        models['ml_ensemble'] = MLEnsemble(ensemble_config)
        
        self.models = models
        logger.info(f"Modelos ML creados: {list(models.keys())}")
        
        return models
    
    def prepare_training_data(self, features: Dict[str, pd.DataFrame], 
                            target_column: str = 'returns_1') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos de entrenamiento para modelos ML.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            
        Returns:
            Tupla con (X, y) para entrenamiento
        """
        logger.info("Preparando datos de entrenamiento ML...")
        
        all_X = []
        all_y = []
        
        for symbol, feature_df in features.items():
            if target_column not in feature_df.columns:
                logger.warning(f"Target '{target_column}' no encontrado en {symbol}")
                continue
            
            # Remover columnas no numéricas y target
            X = feature_df.drop(columns=[target_column])
            y = feature_df[target_column]
            
            # Remover filas con NaN
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) > 0:
                all_X.append(X_clean)
                all_y.append(y_clean)
                logger.info(f"Datos preparados para {symbol}: {len(X_clean)} muestras")
        
        if not all_X:
            raise ValueError("No se encontraron datos válidos para entrenamiento")
        
        # Combinar datos de todos los instrumentos
        X_combined = pd.concat(all_X, axis=0, ignore_index=True)
        y_combined = pd.concat(all_y, axis=0, ignore_index=True)
        
        # Alinear índices
        X_combined = X_combined.reset_index(drop=True)
        y_combined = y_combined.reset_index(drop=True)
        
        logger.info(f"Datos ML preparados: {len(X_combined)} muestras, {len(X_combined.columns)} features")
        
        return X_combined, y_combined
    
    def create_target_signals(self, returns: pd.Series, 
                            long_threshold: float = 0.01, 
                            short_threshold: float = -0.01) -> pd.Series:
        """
        Crea señales target para modelos ML.
        
        Args:
            returns: Serie de retornos
            long_threshold: Umbral para señal long
            short_threshold: Umbral para señal short
            
        Returns:
            Serie con señales (-1, 0, 1)
        """
        signals = pd.Series(0, index=returns.index)
        signals[returns > long_threshold] = 1  # Long
        signals[returns < short_threshold] = -1  # Short
        
        return signals
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                               model_name: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimiza hyperparámetros usando Optuna.
        
        Args:
            X: Features
            y: Target
            model_name: Nombre del modelo
            n_trials: Número de trials para optimización
            
        Returns:
            Mejores hyperparámetros
        """
        logger.info(f"Optimizando hyperparámetros para {model_name} con {n_trials} trials")
        
        def objective(trial):
            if model_name == 'lasso':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True),
                    'max_iter': trial.suggest_int('max_iter', 500, 2000),
                    'n_features': trial.suggest_int('n_features', 20, 100)
                }
                model = LassoModel(params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'n_features': trial.suggest_int('n_features', 50, 150)
                }
                model = XGBoostModel(params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                    'n_features': trial.suggest_int('n_features', 50, 150)
                }
                model = RandomForestModel(params)
                
            else:
                raise ValueError(f"Modelo no soportado para optimización: {model_name}")
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_val)
                    
                    # Calcular Sharpe ratio como métrica
                    strategy_returns = predictions * y_val
                    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
                        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                    else:
                        sharpe = 0.0
                    
                    scores.append(sharpe)
                    
                except Exception as e:
                    logger.warning(f"Error en trial: {str(e)}")
                    scores.append(0.0)
            
            return np.mean(scores)
        
        # Crear estudio Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Mejores hyperparámetros para {model_name}: {best_params}")
        logger.info(f"Mejor score: {best_score:.4f}")
        
        return best_params
    
    def train_models(self, features: Dict[str, pd.DataFrame], 
                    target_column: str = 'returns_1',
                    optimize_hyperparams: bool = True) -> Dict[str, BaseModel]:
        """
        Entrena todos los modelos ML.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            optimize_hyperparams: Si optimizar hyperparámetros
            
        Returns:
            Diccionario con modelos entrenados
        """
        logger.info("Iniciando entrenamiento de modelos ML...")
        
        # Crear modelos si no existen
        if not self.models:
            self.create_ml_models()
        
        # Preparar datos de entrenamiento
        X, y_returns = self.prepare_training_data(features, target_column)
        y_signals = self.create_target_signals(y_returns)
        
        # Entrenar cada modelo
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Entrenando modelo ML: {model_name}")
                
                # Optimizar hyperparámetros si se solicita
                if optimize_hyperparams and model_name != 'ml_ensemble':
                    logger.info(f"Optimizando hyperparámetros para {model_name}")
                    best_params = self.optimize_hyperparameters(X, y_signals, model_name, self.n_trials)
                    
                    # Crear nuevo modelo con mejores parámetros
                    if model_name == 'lasso':
                        model = LassoModel(best_params)
                    elif model_name == 'xgboost':
                        model = XGBoostModel(best_params)
                    elif model_name == 'random_forest':
                        model = RandomForestModel(best_params)
                    
                    self.hyperparameter_results[model_name] = best_params
                
                # Entrenar modelo
                model.fit(X, y_signals)
                
                # Evaluar en entrenamiento
                train_predictions = model.predict(X)
                train_metrics = self._calculate_ml_metrics(y_signals, train_predictions, y_returns)
                
                self.training_results[model_name] = {
                    'metrics': train_metrics,
                    'predictions': train_predictions,
                    'target': y_signals.values,
                    'returns': y_returns.values
                }
                
                trained_models[model_name] = model
                logger.info(f"Modelo ML {model_name} entrenado exitosamente")
                
            except Exception as e:
                logger.error(f"Error entrenando modelo ML {model_name}: {str(e)}")
        
        self.models = trained_models
        logger.info(f"Entrenamiento ML completado: {len(trained_models)} modelos")
        
        return trained_models
    
    def validate_models(self, features: Dict[str, pd.DataFrame], 
                       target_column: str = 'returns_1',
                       validation_split: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Valida modelos ML usando split temporal.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            validation_split: Proporción de datos para validación
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Iniciando validación de modelos ML...")
        
        if not self.models:
            raise ValueError("No hay modelos ML entrenados para validar")
        
        # Preparar datos
        X, y_returns = self.prepare_training_data(features, target_column)
        y_signals = self.create_target_signals(y_returns)
        
        # Split temporal
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y_signals.iloc[:split_idx]
        y_val = y_signals.iloc[split_idx:]
        y_returns_val = y_returns.iloc[split_idx:]
        
        logger.info(f"Split temporal ML: {len(X_train)} train, {len(X_val)} validation")
        
        validation_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Validando modelo ML: {model_name}")
                
                # Re-entrenar en datos de entrenamiento
                model.fit(X_train, y_train)
                
                # Predecir en validación
                val_predictions = model.predict(X_val)
                val_metrics = self._calculate_ml_metrics(y_val, val_predictions, y_returns_val)
                
                # Calcular métricas de trading
                trading_metrics = self._calculate_trading_metrics(y_val, val_predictions, y_returns_val)
                
                validation_results[model_name] = {
                    'metrics': val_metrics,
                    'trading_metrics': trading_metrics,
                    'predictions': val_predictions,
                    'target': y_val.values,
                    'returns': y_returns_val.values
                }
                
                logger.info(f"Modelo ML {model_name} validado exitosamente")
                
            except Exception as e:
                logger.error(f"Error validando modelo ML {model_name}: {str(e)}")
        
        self.validation_results = validation_results
        logger.info("Validación ML completada")
        
        return validation_results
    
    def cross_validate_models(self, features: Dict[str, pd.DataFrame], 
                            target_column: str = 'returns_1',
                            n_splits: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Realiza cross-validation temporal de los modelos ML.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            n_splits: Número de splits para CV
            
        Returns:
            Diccionario con resultados de CV
        """
        logger.info(f"Iniciando cross-validation ML con {n_splits} splits...")
        
        if not self.models:
            raise ValueError("No hay modelos ML entrenados para validar")
        
        # Preparar datos
        X, y_returns = self.prepare_training_data(features, target_column)
        y_signals = self.create_target_signals(y_returns)
        
        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Cross-validando modelo ML: {model_name}")
            
            fold_metrics = []
            fold_trading_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"Fold {fold + 1}/{n_splits}")
                
                # Split datos
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y_signals.iloc[train_idx]
                y_val_fold = y_signals.iloc[val_idx]
                y_returns_fold = y_returns.iloc[val_idx]
                
                try:
                    # Entrenar en fold
                    model.fit(X_train_fold, y_train_fold)
                    
                    # Predecir en validación
                    val_predictions = model.predict(X_val_fold)
                    
                    # Calcular métricas
                    metrics = self._calculate_ml_metrics(y_val_fold, val_predictions, y_returns_fold)
                    trading_metrics = self._calculate_trading_metrics(y_val_fold, val_predictions, y_returns_fold)
                    
                    fold_metrics.append(metrics)
                    fold_trading_metrics.append(trading_metrics)
                    
                except Exception as e:
                    logger.error(f"Error en fold {fold + 1} para modelo ML {model_name}: {str(e)}")
            
            if fold_metrics:
                # Promediar métricas de todos los folds
                avg_metrics = {}
                for key in fold_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
                
                avg_trading_metrics = {}
                for key in fold_trading_metrics[0].keys():
                    avg_trading_metrics[key] = np.mean([m[key] for m in fold_trading_metrics])
                
                cv_results[model_name] = {
                    'avg_metrics': avg_metrics,
                    'avg_trading_metrics': avg_trading_metrics,
                    'fold_metrics': fold_metrics,
                    'fold_trading_metrics': fold_trading_metrics
                }
                
                logger.info(f"Cross-validation ML completada para {model_name}")
        
        logger.info("Cross-validation ML completada")
        return cv_results
    
    def _calculate_ml_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                            returns: pd.Series) -> Dict[str, float]:
        """
        Calcula métricas específicas para modelos ML.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            returns: Retornos reales
            
        Returns:
            Diccionario con métricas
        """
        # Convertir a arrays numpy
        y_true = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        returns = returns.values if hasattr(returns, 'values') else returns
        
        # Filtrar NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(returns))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        returns_clean = returns[valid_mask]
        
        if len(y_true_clean) == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'sharpe_ratio': 0.0, 'hit_ratio': 0.0, 'profit_factor': 0.0
            }
        
        try:
            # Métricas de clasificación
            accuracy = accuracy_score(y_true_clean, y_pred_clean)
            precision = precision_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            recall = recall_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            f1 = f1_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            
            # Métricas de trading
            strategy_returns = y_pred_clean * returns_clean
            
            if len(strategy_returns) > 1 and strategy_returns.std() > 0:
                sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            hit_ratio = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
            
            gross_profit = strategy_returns[strategy_returns > 0].sum()
            gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sharpe_ratio': sharpe_ratio,
                'hit_ratio': hit_ratio,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculando métricas ML: {str(e)}")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'sharpe_ratio': 0.0, 'hit_ratio': 0.0, 'profit_factor': 0.0
            }
    
    def _calculate_trading_metrics(self, signals: pd.Series, predictions: np.ndarray, 
                                 returns: pd.Series) -> Dict[str, float]:
        """
        Calcula métricas específicas de trading.
        
        Args:
            signals: Señales reales
            predictions: Predicciones del modelo
            returns: Retornos reales
            
        Returns:
            Diccionario con métricas de trading
        """
        # Filtrar NaN
        valid_mask = ~(signals.isnull() | pd.Series(predictions).isnull() | returns.isnull())
        signals_clean = signals[valid_mask]
        predictions_clean = pd.Series(predictions)[valid_mask]
        returns_clean = returns[valid_mask]
        
        if len(signals_clean) == 0:
            return {
                'hit_ratio': 0.0, 'profit_factor': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'total_return': 0.0
            }
        
        # Calcular retornos de la estrategia
        strategy_returns = predictions_clean * returns_clean
        
        # Métricas básicas
        hit_ratio = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
        
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Sharpe ratio
        if len(strategy_returns) > 1 and strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Total return
        total_return = (1 + strategy_returns).prod() - 1
        
        return {
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Tuple[str, BaseModel]:
        """
        Obtiene el mejor modelo ML basado en una métrica.
        
        Args:
            metric: Métrica para comparar
            
        Returns:
            Tupla con (nombre_del_modelo, modelo)
        """
        if not self.validation_results:
            raise ValueError("No hay resultados de validación ML disponibles")
        
        best_model_name = None
        best_score = float('-inf')
        
        for model_name, results in self.validation_results.items():
            if 'trading_metrics' in results and metric in results['trading_metrics']:
                score = results['trading_metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"No se encontró métrica '{metric}' en los resultados ML")
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, filepath: str = "models/ml_models") -> None:
        """
        Guarda todos los modelos ML entrenados.
        
        Args:
            filepath: Ruta base para guardar modelos
        """
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_filepath = save_path / f"{model_name}_model.pkl"
            model.save_model(str(model_filepath))
            logger.info(f"Modelo ML {model_name} guardado: {model_filepath}")
        
        # Guardar resultados
        results_filepath = save_path / "ml_training_results.pkl"
        results_data = {
            'training_results': self.training_results,
            'validation_results': self.validation_results,
            'hyperparameter_results': self.hyperparameter_results,
            'config': self.config
        }
        joblib.dump(results_data, results_filepath)
        logger.info(f"Resultados ML guardados: {results_filepath}")
    
    def load_models(self, filepath: str = "models/ml_models") -> Dict[str, BaseModel]:
        """
        Carga modelos ML entrenados.
        
        Args:
            filepath: Ruta base de los modelos
            
        Returns:
            Diccionario con modelos cargados
        """
        load_path = Path(filepath)
        
        if not load_path.exists():
            raise ValueError(f"Directorio no encontrado: {filepath}")
        
        models = {}
        
        # Cargar modelos individuales
        for model_file in load_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            
            if model_name == 'lasso':
                model = LassoModel()
            elif model_name == 'xgboost':
                model = XGBoostModel()
            elif model_name == 'random_forest':
                model = RandomForestModel()
            elif model_name == 'ml_ensemble':
                model = MLEnsemble()
            else:
                logger.warning(f"Tipo de modelo ML desconocido: {model_name}")
                continue
            
            model.load_model(str(model_file))
            models[model_name] = model
            logger.info(f"Modelo ML {model_name} cargado: {model_file}")
        
        # Cargar resultados
        results_file = load_path / "ml_training_results.pkl"
        if results_file.exists():
            results_data = joblib.load(results_file)
            self.training_results = results_data.get('training_results', {})
            self.validation_results = results_data.get('validation_results', {})
            self.hyperparameter_results = results_data.get('hyperparameter_results', {})
            logger.info("Resultados de entrenamiento ML cargados")
        
        self.models = models
        logger.info(f"Modelos ML cargados: {list(models.keys())}")
        
        return models
    
    def generate_model_report(self) -> pd.DataFrame:
        """
        Genera reporte de performance de todos los modelos ML.
        
        Returns:
            DataFrame con métricas de todos los modelos
        """
        if not self.validation_results:
            logger.warning("No hay resultados de validación ML para generar reporte")
            return pd.DataFrame()
        
        report_data = []
        
        for model_name, results in self.validation_results.items():
            if 'trading_metrics' in results:
                metrics = results['trading_metrics']
                report_data.append({
                    'model': model_name,
                    'hit_ratio': metrics.get('hit_ratio', 0.0),
                    'profit_factor': metrics.get('profit_factor', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'total_return': metrics.get('total_return', 0.0)
                })
        
        if not report_data:
            return pd.DataFrame()
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('sharpe_ratio', ascending=False)
        
        return report_df
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Compara modelos ML con modelos baseline.
        
        Args:
            baseline_results: Resultados de modelos baseline
            
        Returns:
            DataFrame con comparación
        """
        comparison_data = []
        
        # Agregar resultados baseline
        for model_name, results in baseline_results.items():
            if 'trading_metrics' in results:
                metrics = results['trading_metrics']
                comparison_data.append({
                    'model': f"baseline_{model_name}",
                    'type': 'baseline',
                    'hit_ratio': metrics.get('hit_ratio', 0.0),
                    'profit_factor': metrics.get('profit_factor', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'total_return': metrics.get('total_return', 0.0)
                })
        
        # Agregar resultados ML
        for model_name, results in self.validation_results.items():
            if 'trading_metrics' in results:
                metrics = results['trading_metrics']
                comparison_data.append({
                    'model': f"ml_{model_name}",
                    'type': 'ml',
                    'hit_ratio': metrics.get('hit_ratio', 0.0),
                    'profit_factor': metrics.get('profit_factor', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'total_return': metrics.get('total_return', 0.0)
                })
        
        if not comparison_data:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df

