"""
Entrenador de modelos para el sistema de trading.
Maneja el entrenamiento, validación y selección de modelos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base import BaseModel, ModelEnsemble
from .baseline import MomentumModel, MeanReversionModel, HybridModel
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Entrenador principal para modelos de trading.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.models = {}
        self.training_results = {}
        self.validation_results = {}
        
        # Configuración de entrenamiento
        self.train_config = self.config.get('modelos', {})
        self.lookback_periods = self.train_config.get('baseline', {}).get('momentum_periods', [5, 10, 20, 50])
        
        logger.info("ModelTrainer inicializado")
    
    def create_baseline_models(self) -> Dict[str, BaseModel]:
        """
        Crea modelos baseline.
        
        Returns:
            Diccionario con modelos baseline
        """
        models = {}
        
        # Modelo de Momentum
        momentum_config = {
            'short_period': 5,
            'medium_period': 20,
            'long_period': 50,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volatility_period': 20,
            'volatility_threshold': 0.02,
            'min_confidence': 0.3
        }
        models['momentum'] = MomentumModel(momentum_config)
        
        # Modelo de Mean Reversion
        mean_reversion_config = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'williams_period': 14,
            'williams_oversold': -80,
            'williams_overbought': -20,
            'stoch_period': 14,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'min_confidence': 0.3
        }
        models['mean_reversion'] = MeanReversionModel(mean_reversion_config)
        
        # Modelo Híbrido
        hybrid_config = {
            'momentum_weight': 0.6,
            'mean_reversion_weight': 0.4,
            'volatility_threshold': 0.02,
            'regime_detection_period': 50,
            'min_confidence': 0.3,
            'momentum_config': momentum_config,
            'mean_reversion_config': mean_reversion_config
        }
        models['hybrid'] = HybridModel(hybrid_config)
        
        self.models = models
        logger.info(f"Modelos baseline creados: {list(models.keys())}")
        
        return models
    
    def prepare_training_data(self, features: Dict[str, pd.DataFrame], 
                            target_column: str = 'returns_1') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos de entrenamiento combinando features de todos los instrumentos.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            
        Returns:
            Tupla con (X, y) para entrenamiento
        """
        logger.info("Preparando datos de entrenamiento...")
        
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
        
        logger.info(f"Datos de entrenamiento preparados: {len(X_combined)} muestras, {len(X_combined.columns)} features")
        
        return X_combined, y_combined
    
    def create_target_signals(self, returns: pd.Series, 
                            long_threshold: float = 0.01, 
                            short_threshold: float = -0.01) -> pd.Series:
        """
        Crea señales target basadas en retornos.
        
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
    
    def train_models(self, features: Dict[str, pd.DataFrame], 
                    target_column: str = 'returns_1') -> Dict[str, BaseModel]:
        """
        Entrena todos los modelos baseline.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            
        Returns:
            Diccionario con modelos entrenados
        """
        logger.info("Iniciando entrenamiento de modelos...")
        
        # Crear modelos si no existen
        if not self.models:
            self.create_baseline_models()
        
        # Preparar datos de entrenamiento
        X, y_returns = self.prepare_training_data(features, target_column)
        
        # Crear señales target
        y_signals = self.create_target_signals(y_returns)
        
        # Entrenar cada modelo
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Entrenando modelo: {model_name}")
                
                # Entrenar modelo
                model.fit(X, y_signals)
                
                # Evaluar en entrenamiento
                train_predictions = model.predict(X)
                train_metrics = self._calculate_metrics(y_signals, train_predictions)
                
                self.training_results[model_name] = {
                    'metrics': train_metrics,
                    'predictions': train_predictions,
                    'target': y_signals.values
                }
                
                trained_models[model_name] = model
                logger.info(f"Modelo {model_name} entrenado exitosamente")
                
            except Exception as e:
                logger.error(f"Error entrenando modelo {model_name}: {str(e)}")
        
        self.models = trained_models
        logger.info(f"Entrenamiento completado: {len(trained_models)} modelos")
        
        return trained_models
    
    def validate_models(self, features: Dict[str, pd.DataFrame], 
                       target_column: str = 'returns_1',
                       validation_split: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Valida modelos usando split temporal.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            validation_split: Proporción de datos para validación
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Iniciando validación de modelos...")
        
        if not self.models:
            raise ValueError("No hay modelos entrenados para validar")
        
        # Preparar datos
        X, y_returns = self.prepare_training_data(features, target_column)
        y_signals = self.create_target_signals(y_returns)
        
        # Split temporal
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y_signals.iloc[:split_idx]
        y_val = y_signals.iloc[split_idx:]
        
        logger.info(f"Split temporal: {len(X_train)} train, {len(X_val)} validation")
        
        validation_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Validando modelo: {model_name}")
                
                # Re-entrenar en datos de entrenamiento
                model.fit(X_train, y_train)
                
                # Predecir en validación
                val_predictions = model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_predictions)
                
                # Calcular métricas de trading
                trading_metrics = self._calculate_trading_metrics(y_val, val_predictions, y_returns.iloc[split_idx:])
                
                validation_results[model_name] = {
                    'metrics': val_metrics,
                    'trading_metrics': trading_metrics,
                    'predictions': val_predictions,
                    'target': y_val.values,
                    'returns': y_returns.iloc[split_idx:].values
                }
                
                logger.info(f"Modelo {model_name} validado exitosamente")
                
            except Exception as e:
                logger.error(f"Error validando modelo {model_name}: {str(e)}")
        
        self.validation_results = validation_results
        logger.info("Validación completada")
        
        return validation_results
    
    def cross_validate_models(self, features: Dict[str, pd.DataFrame], 
                            target_column: str = 'returns_1',
                            n_splits: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Realiza cross-validation temporal de los modelos.
        
        Args:
            features: Diccionario con features por instrumento
            target_column: Columna target
            n_splits: Número de splits para CV
            
        Returns:
            Diccionario con resultados de CV
        """
        logger.info(f"Iniciando cross-validation con {n_splits} splits...")
        
        if not self.models:
            raise ValueError("No hay modelos entrenados para validar")
        
        # Preparar datos
        X, y_returns = self.prepare_training_data(features, target_column)
        y_signals = self.create_target_signals(y_returns)
        
        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Cross-validando modelo: {model_name}")
            
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
                    metrics = self._calculate_metrics(y_val_fold, val_predictions)
                    trading_metrics = self._calculate_trading_metrics(y_val_fold, val_predictions, y_returns_fold)
                    
                    fold_metrics.append(metrics)
                    fold_trading_metrics.append(trading_metrics)
                    
                except Exception as e:
                    logger.error(f"Error en fold {fold + 1} para modelo {model_name}: {str(e)}")
            
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
                
                logger.info(f"Cross-validation completada para {model_name}")
        
        logger.info("Cross-validation completada")
        return cv_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de clasificación.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            
        Returns:
            Diccionario con métricas
        """
        # Convertir a arrays numpy
        y_true = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        # Filtrar NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        try:
            accuracy = accuracy_score(y_true_clean, y_pred_clean)
            precision = precision_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            recall = recall_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            f1 = f1_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            logger.error(f"Error calculando métricas: {str(e)}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
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
                'hit_ratio': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0
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
        Obtiene el mejor modelo basado en una métrica.
        
        Args:
            metric: Métrica para comparar ('sharpe_ratio', 'hit_ratio', 'profit_factor')
            
        Returns:
            Tupla con (nombre_del_modelo, modelo)
        """
        if not self.validation_results:
            raise ValueError("No hay resultados de validación disponibles")
        
        best_model_name = None
        best_score = float('-inf')
        
        for model_name, results in self.validation_results.items():
            if 'trading_metrics' in results and metric in results['trading_metrics']:
                score = results['trading_metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"No se encontró métrica '{metric}' en los resultados")
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, filepath: str = "models/baseline_models") -> None:
        """
        Guarda todos los modelos entrenados.
        
        Args:
            filepath: Ruta base para guardar modelos
        """
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_filepath = save_path / f"{model_name}_model.pkl"
            model.save_model(str(model_filepath))
            logger.info(f"Modelo {model_name} guardado: {model_filepath}")
        
        # Guardar resultados
        results_filepath = save_path / "training_results.pkl"
        results_data = {
            'training_results': self.training_results,
            'validation_results': self.validation_results,
            'config': self.config
        }
        joblib.dump(results_data, results_filepath)
        logger.info(f"Resultados guardados: {results_filepath}")
    
    def load_models(self, filepath: str = "models/baseline_models") -> Dict[str, BaseModel]:
        """
        Carga modelos entrenados.
        
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
            
            if model_name == 'momentum':
                model = MomentumModel()
            elif model_name == 'mean_reversion':
                model = MeanReversionModel()
            elif model_name == 'hybrid':
                model = HybridModel()
            else:
                logger.warning(f"Tipo de modelo desconocido: {model_name}")
                continue
            
            model.load_model(str(model_file))
            models[model_name] = model
            logger.info(f"Modelo {model_name} cargado: {model_file}")
        
        # Cargar resultados
        results_file = load_path / "training_results.pkl"
        if results_file.exists():
            results_data = joblib.load(results_file)
            self.training_results = results_data.get('training_results', {})
            self.validation_results = results_data.get('validation_results', {})
            logger.info("Resultados de entrenamiento cargados")
        
        self.models = models
        logger.info(f"Modelos cargados: {list(models.keys())}")
        
        return models
    
    def generate_model_report(self) -> pd.DataFrame:
        """
        Genera reporte de performance de todos los modelos.
        
        Returns:
            DataFrame con métricas de todos los modelos
        """
        if not self.validation_results:
            logger.warning("No hay resultados de validación para generar reporte")
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

