"""
Modelos baseline transparentes para trading.
Incluye momentum, mean-reversion y reglas híbridas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from .base import BaseModel, TradingSignal

logger = logging.getLogger(__name__)


class MomentumModel(BaseModel):
    """
    Modelo baseline de momentum.
    Basado en tendencias de precios y momentum indicators.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el modelo de momentum.
        
        Args:
            config: Configuración del modelo
        """
        default_config = {
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
        
        config = {**default_config, **(config or {})}
        super().__init__("momentum", config)
        
        self.short_period = config['short_period']
        self.medium_period = config['medium_period']
        self.long_period = config['long_period']
        self.rsi_period = config['rsi_period']
        self.rsi_oversold = config['rsi_oversold']
        self.rsi_overbought = config['rsi_overbought']
        self.volatility_period = config['volatility_period']
        self.volatility_threshold = config['volatility_threshold']
        self.min_confidence = config['min_confidence']
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MomentumModel':
        """
        Entrena el modelo de momentum.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando modelo momentum con {len(X)} muestras")
        
        # Calcular métricas de performance en entrenamiento
        self._calculate_training_metrics(X, y)
        
        self.is_fitted = True
        logger.info("Modelo momentum entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones de momentum.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        predictions = []
        
        for idx, row in X.iterrows():
            signal = self._calculate_momentum_signal(row)
            predictions.append(signal)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        predictions = self.predict(X)
        
        # Convertir señales a probabilidades
        probabilities = []
        for pred in predictions:
            if pred == 1:  # Long
                prob = [0.1, 0.1, 0.8]  # [Short, Neutral, Long]
            elif pred == -1:  # Short
                prob = [0.8, 0.1, 0.1]
            else:  # Neutral
                prob = [0.1, 0.8, 0.1]
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def _calculate_momentum_signal(self, features: pd.Series) -> int:
        """
        Calcula señal de momentum basada en features.
        
        Args:
            features: Features de una muestra
            
        Returns:
            Señal (-1, 0, 1)
        """
        signal_score = 0
        confidence_factors = []
        
        # 1. Momentum de precios
        if f'returns_{self.short_period}' in features:
            short_momentum = features[f'returns_{self.short_period}']
            if not pd.isna(short_momentum):
                if short_momentum > 0.01:  # 1% threshold
                    signal_score += 1
                    confidence_factors.append(abs(short_momentum))
                elif short_momentum < -0.01:
                    signal_score -= 1
                    confidence_factors.append(abs(short_momentum))
        
        # 2. RSI
        if f'rsi_{self.rsi_period}' in features:
            rsi = features[f'rsi_{self.rsi_period}']
            if not pd.isna(rsi):
                if rsi < self.rsi_oversold:
                    signal_score += 1
                    confidence_factors.append((self.rsi_oversold - rsi) / self.rsi_oversold)
                elif rsi > self.rsi_overbought:
                    signal_score -= 1
                    confidence_factors.append((rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
        
        # 3. Moving averages
        if f'price_sma_ratio_{self.medium_period}' in features:
            sma_ratio = features[f'price_sma_ratio_{self.medium_period}']
            if not pd.isna(sma_ratio):
                if sma_ratio > 1.02:  # 2% above SMA
                    signal_score += 1
                    confidence_factors.append(sma_ratio - 1)
                elif sma_ratio < 0.98:  # 2% below SMA
                    signal_score -= 1
                    confidence_factors.append(1 - sma_ratio)
        
        # 4. Volatility filter
        if f'volatility_{self.volatility_period}' in features:
            volatility = features[f'volatility_{self.volatility_period}']
            if not pd.isna(volatility) and volatility > self.volatility_threshold:
                # Reducir confianza en alta volatilidad
                confidence_factors = [cf * 0.5 for cf in confidence_factors]
        
        # 5. MACD
        if 'macd' in features and 'macd_signal' in features:
            macd = features['macd']
            macd_signal = features['macd_signal']
            if not pd.isna(macd) and not pd.isna(macd_signal):
                if macd > macd_signal:
                    signal_score += 1
                    confidence_factors.append(abs(macd - macd_signal))
                elif macd < macd_signal:
                    signal_score -= 1
                    confidence_factors.append(abs(macd - macd_signal))
        
        # Determinar señal final
        if not confidence_factors:
            return 0
        
        avg_confidence = np.mean(confidence_factors)
        
        if avg_confidence < self.min_confidence:
            return 0
        
        if signal_score > 0:
            return 1  # Long
        elif signal_score < 0:
            return -1  # Short
        else:
            return 0  # Neutral
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calcula métricas de entrenamiento."""
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
            'neutral_signals': (predictions == 0).sum()
        }
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.performance_metrics = model_object['performance_metrics']


class MeanReversionModel(BaseModel):
    """
    Modelo baseline de reversión a la media.
    Basado en indicadores de sobrecompra/sobreventa.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el modelo de mean reversion.
        
        Args:
            config: Configuración del modelo
        """
        default_config = {
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
        
        config = {**default_config, **(config or {})}
        super().__init__("mean_reversion", config)
        
        self.bb_period = config['bb_period']
        self.bb_std = config['bb_std']
        self.rsi_period = config['rsi_period']
        self.rsi_oversold = config['rsi_oversold']
        self.rsi_overbought = config['rsi_overbought']
        self.williams_period = config['williams_period']
        self.williams_oversold = config['williams_oversold']
        self.williams_overbought = config['williams_overbought']
        self.stoch_period = config['stoch_period']
        self.stoch_oversold = config['stoch_oversold']
        self.stoch_overbought = config['stoch_overbought']
        self.min_confidence = config['min_confidence']
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MeanReversionModel':
        """
        Entrena el modelo de mean reversion.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando modelo mean reversion con {len(X)} muestras")
        
        # Calcular métricas de performance en entrenamiento
        self._calculate_training_metrics(X, y)
        
        self.is_fitted = True
        logger.info("Modelo mean reversion entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones de mean reversion.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        predictions = []
        
        for idx, row in X.iterrows():
            signal = self._calculate_mean_reversion_signal(row)
            predictions.append(signal)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        predictions = self.predict(X)
        
        # Convertir señales a probabilidades
        probabilities = []
        for pred in predictions:
            if pred == 1:  # Long (oversold)
                prob = [0.1, 0.1, 0.8]  # [Short, Neutral, Long]
            elif pred == -1:  # Short (overbought)
                prob = [0.8, 0.1, 0.1]
            else:  # Neutral
                prob = [0.1, 0.8, 0.1]
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def _calculate_mean_reversion_signal(self, features: pd.Series) -> int:
        """
        Calcula señal de mean reversion basada en features.
        
        Args:
            features: Features de una muestra
            
        Returns:
            Señal (-1, 0, 1)
        """
        signal_score = 0
        confidence_factors = []
        
        # 1. Bollinger Bands
        if f'bb_position_{self.bb_period}_{self.bb_std}' in features:
            bb_position = features[f'bb_position_{self.bb_period}_{self.bb_std}']
            if not pd.isna(bb_position):
                if bb_position < 0.1:  # Near lower band (oversold)
                    signal_score += 1
                    confidence_factors.append(0.1 - bb_position)
                elif bb_position > 0.9:  # Near upper band (overbought)
                    signal_score -= 1
                    confidence_factors.append(bb_position - 0.9)
        
        # 2. RSI
        if f'rsi_{self.rsi_period}' in features:
            rsi = features[f'rsi_{self.rsi_period}']
            if not pd.isna(rsi):
                if rsi < self.rsi_oversold:
                    signal_score += 1
                    confidence_factors.append((self.rsi_oversold - rsi) / self.rsi_oversold)
                elif rsi > self.rsi_overbought:
                    signal_score -= 1
                    confidence_factors.append((rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
        
        # 3. Williams %R
        if f'williams_r_{self.williams_period}' in features:
            williams_r = features[f'williams_r_{self.williams_period}']
            if not pd.isna(williams_r):
                if williams_r < self.williams_oversold:
                    signal_score += 1
                    confidence_factors.append((self.williams_oversold - williams_r) / abs(self.williams_oversold))
                elif williams_r > self.williams_overbought:
                    signal_score -= 1
                    confidence_factors.append((williams_r - self.williams_overbought) / abs(self.williams_overbought))
        
        # 4. Stochastic Oscillator
        if f'stoch_k_{self.stoch_period}_3' in features:
            stoch_k = features[f'stoch_k_{self.stoch_period}_3']
            if not pd.isna(stoch_k):
                if stoch_k < self.stoch_oversold:
                    signal_score += 1
                    confidence_factors.append((self.stoch_oversold - stoch_k) / self.stoch_oversold)
                elif stoch_k > self.stoch_overbought:
                    signal_score -= 1
                    confidence_factors.append((stoch_k - self.stoch_overbought) / (100 - self.stoch_overbought))
        
        # 5. Z-Score (precio vs media)
        if f'zscore_{self.bb_period}' in features:
            zscore = features[f'zscore_{self.bb_period}']
            if not pd.isna(zscore):
                if zscore < -2:  # 2 std below mean
                    signal_score += 1
                    confidence_factors.append(abs(zscore) - 2)
                elif zscore > 2:  # 2 std above mean
                    signal_score -= 1
                    confidence_factors.append(zscore - 2)
        
        # Determinar señal final
        if not confidence_factors:
            return 0
        
        avg_confidence = np.mean(confidence_factors)
        
        if avg_confidence < self.min_confidence:
            return 0
        
        if signal_score > 0:
            return 1  # Long (oversold)
        elif signal_score < 0:
            return -1  # Short (overbought)
        else:
            return 0  # Neutral
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calcula métricas de entrenamiento."""
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
            'neutral_signals': (predictions == 0).sum()
        }
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.performance_metrics = model_object['performance_metrics']


class HybridModel(BaseModel):
    """
    Modelo híbrido que combina momentum y mean reversion.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el modelo híbrido.
        
        Args:
            config: Configuración del modelo
        """
        default_config = {
            'momentum_weight': 0.6,
            'mean_reversion_weight': 0.4,
            'volatility_threshold': 0.02,
            'regime_detection_period': 50,
            'min_confidence': 0.3
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("hybrid", config)
        
        self.momentum_weight = config['momentum_weight']
        self.mean_reversion_weight = config['mean_reversion_weight']
        self.volatility_threshold = config['volatility_threshold']
        self.regime_detection_period = config['regime_detection_period']
        self.min_confidence = config['min_confidence']
        
        # Inicializar sub-modelos
        self.momentum_model = MomentumModel(config.get('momentum_config', {}))
        self.mean_reversion_model = MeanReversionModel(config.get('mean_reversion_config', {}))
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HybridModel':
        """
        Entrena el modelo híbrido.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Modelo entrenado
        """
        if not self.validate_data(X, y):
            raise ValueError("Datos de entrenamiento inválidos")
        
        logger.info(f"Entrenando modelo híbrido con {len(X)} muestras")
        
        # Entrenar sub-modelos
        self.momentum_model.fit(X, y)
        self.mean_reversion_model.fit(X, y)
        
        # Calcular métricas de performance en entrenamiento
        self._calculate_training_metrics(X, y)
        
        self.is_fitted = True
        logger.info("Modelo híbrido entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones híbridas.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones (-1, 0, 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Obtener predicciones de sub-modelos
        momentum_preds = self.momentum_model.predict(X)
        mean_reversion_preds = self.mean_reversion_model.predict(X)
        
        # Combinar predicciones
        hybrid_preds = []
        
        for i in range(len(X)):
            # Detectar régimen basado en volatilidad
            regime = self._detect_regime(X.iloc[i])
            
            if regime == 'trending':
                # En trending, usar más momentum
                weight_momentum = 0.8
                weight_mean_reversion = 0.2
            elif regime == 'ranging':
                # En ranging, usar más mean reversion
                weight_momentum = 0.2
                weight_mean_reversion = 0.8
            else:
                # Régimen mixto
                weight_momentum = self.momentum_weight
                weight_mean_reversion = self.mean_reversion_weight
            
            # Combinar señales
            combined_signal = (weight_momentum * momentum_preds[i] + 
                             weight_mean_reversion * mean_reversion_preds[i])
            
            # Convertir a señal discreta
            if combined_signal > 0.5:
                hybrid_preds.append(1)
            elif combined_signal < -0.5:
                hybrid_preds.append(-1)
            else:
                hybrid_preds.append(0)
        
        return np.array(hybrid_preds)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades de predicción.
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de cada clase
        """
        predictions = self.predict(X)
        
        # Convertir señales a probabilidades
        probabilities = []
        for pred in predictions:
            if pred == 1:  # Long
                prob = [0.1, 0.1, 0.8]  # [Short, Neutral, Long]
            elif pred == -1:  # Short
                prob = [0.8, 0.1, 0.1]
            else:  # Neutral
                prob = [0.1, 0.8, 0.1]
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def _detect_regime(self, features: pd.Series) -> str:
        """
        Detecta el régimen de mercado basado en features.
        
        Args:
            features: Features de una muestra
            
        Returns:
            Régimen detectado ('trending', 'ranging', 'mixed')
        """
        # Usar volatilidad para detectar régimen
        if f'volatility_{self.regime_detection_period}' in features:
            volatility = features[f'volatility_{self.regime_detection_period}']
            if not pd.isna(volatility):
                if volatility > self.volatility_threshold * 1.5:
                    return 'trending'
                elif volatility < self.volatility_threshold * 0.5:
                    return 'ranging'
        
        # Usar ATR para confirmar
        if 'atr_14' in features:
            atr = features['atr_14']
            if not pd.isna(atr):
                if atr > 0.02:  # High ATR
                    return 'trending'
                elif atr < 0.005:  # Low ATR
                    return 'ranging'
        
        return 'mixed'
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calcula métricas de entrenamiento."""
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
            'momentum_metrics': self.momentum_model.performance_metrics,
            'mean_reversion_metrics': self.mean_reversion_model.performance_metrics
        }
    
    def _get_model_object(self) -> Dict[str, Any]:
        """Obtiene objeto del modelo para guardar."""
        return {
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'momentum_model': self.momentum_model._get_model_object(),
            'mean_reversion_model': self.mean_reversion_model._get_model_object()
        }
    
    def _set_model_object(self, model_object: Dict[str, Any]) -> None:
        """Establece objeto del modelo desde carga."""
        self.config = model_object['config']
        self.performance_metrics = model_object['performance_metrics']
        self.momentum_model._set_model_object(model_object['momentum_model'])
        self.mean_reversion_model._set_model_object(model_object['mean_reversion_model'])

