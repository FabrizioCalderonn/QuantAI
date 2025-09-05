"""
Feature extractors para indicadores técnicos.
Incluye momentum, reversión a la media, volatilidad y microestructura.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class TechnicalFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor de features técnicos básicos.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        super().__init__("technical", lookback_periods)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae features técnicos básicos.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features técnicos
        """
        if not self.validate_data(data):
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        # Precios
        close = data['close']
        high = data['high']
        low = data['low']
        open_price = data['open']
        volume = data['volume']
        
        # 1. RETORNOS Y MOMENTUM
        returns_features = self._calculate_returns(close, [1, 5, 10, 20])
        features = pd.concat([features, pd.DataFrame(returns_features)], axis=1)
        
        # Log retornos
        log_returns_features = self._calculate_log_returns(close, [1, 5, 10, 20])
        features = pd.concat([features, pd.DataFrame(log_returns_features)], axis=1)
        
        # 2. VOLATILIDAD
        volatility_features = self._calculate_volatility(returns_features['returns_1'], [5, 10, 20, 50])
        features = pd.concat([features, pd.DataFrame(volatility_features)], axis=1)
        
        # ATR (Average True Range)
        atr_features = self._calculate_atr_features(high, low, close)
        features = pd.concat([features, pd.DataFrame(atr_features)], axis=1)
        
        # 3. MOMENTUM INDICATORS
        momentum_features = self._calculate_momentum_features(close)
        features = pd.concat([features, pd.DataFrame(momentum_features)], axis=1)
        
        # 4. MEAN REVERSION INDICATORS
        mean_reversion_features = self._calculate_mean_reversion_features(close, high, low)
        features = pd.concat([features, pd.DataFrame(mean_reversion_features)], axis=1)
        
        # 5. VOLUME INDICATORS
        volume_features = self._calculate_volume_features(close, volume)
        features = pd.concat([features, pd.DataFrame(volume_features)], axis=1)
        
        # 6. PRICE POSITION FEATURES
        price_position_features = self._calculate_price_position_features(close, high, low)
        features = pd.concat([features, pd.DataFrame(price_position_features)], axis=1)
        
        # 7. Z-SCORES Y PERCENTILE RANKS
        zscore_features = self._calculate_zscore(close, [20, 50, 100])
        features = pd.concat([features, pd.DataFrame(zscore_features)], axis=1)
        
        percentile_features = self._calculate_percentile_rank(close, [20, 50, 100])
        features = pd.concat([features, pd.DataFrame(percentile_features)], axis=1)
        
        # Almacenar nombres de features
        self.features = {col: col for col in features.columns}
        
        return features
    
    def _calculate_atr_features(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features basados en ATR."""
        features = {}
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR para diferentes períodos
        for period in [14, 21, 50]:
            atr = self._safe_rolling(true_range, period, lambda x: x.mean())
            features[f'atr_{period}'] = atr
            
            # ATR como porcentaje del precio
            features[f'atr_pct_{period}'] = atr / close
        
        return features
    
    def _calculate_momentum_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de momentum."""
        features = {}
        
        # RSI (Relative Strength Index)
        for period in [14, 21, 50]:
            rsi = self._calculate_rsi(close, period)
            features[f'rsi_{period}'] = rsi
            
            # RSI normalizado (0-1)
            features[f'rsi_norm_{period}'] = rsi / 100
        
        # MACD
        macd_features = self._calculate_macd(close)
        features.update(macd_features)
        
        # Rate of Change
        for period in [5, 10, 20, 50]:
            roc = (close / close.shift(period) - 1) * 100
            features[f'roc_{period}'] = roc
        
        # Momentum
        for period in [5, 10, 20, 50]:
            momentum = close - close.shift(period)
            features[f'momentum_{period}'] = momentum
        
        return features
    
    def _calculate_mean_reversion_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de reversión a la media."""
        features = {}
        
        # Bollinger Bands
        for period in [20, 50]:
            for std_dev in [1, 2]:
                bb_features = self._calculate_bollinger_bands(close, period, std_dev)
                features.update({f'bb_{k}_{period}_{std_dev}': v for k, v in bb_features.items()})
        
        # Williams %R
        for period in [14, 21, 50]:
            williams_r = self._calculate_williams_r(high, low, close, period)
            features[f'williams_r_{period}'] = williams_r
        
        # Stochastic Oscillator
        for k_period in [14, 21]:
            for d_period in [3, 5]:
                stoch_features = self._calculate_stochastic(high, low, close, k_period, d_period)
                features.update({f'stoch_{k}_{k_period}_{d_period}': v for k, v in stoch_features.items()})
        
        # Price distance from moving averages
        for period in [20, 50, 100, 200]:
            sma = self._safe_rolling(close, period, lambda x: x.mean())
            features[f'price_sma_ratio_{period}'] = close / sma
            features[f'price_sma_diff_{period}'] = (close - sma) / sma
        
        return features
    
    def _calculate_volume_features(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de volumen."""
        features = {}
        
        # Volume moving averages
        for period in [10, 20, 50]:
            vol_sma = self._safe_rolling(volume, period, lambda x: x.mean())
            features[f'volume_sma_{period}'] = vol_sma
            features[f'volume_ratio_{period}'] = volume / vol_sma
        
        # Volume-weighted average price (VWAP)
        for period in [20, 50]:
            typical_price = (close + close.shift(1) + close.shift(2)) / 3  # Simplified
            vwap = self._safe_rolling(typical_price * volume, period, lambda x: x.sum()) / \
                   self._safe_rolling(volume, period, lambda x: x.sum())
            features[f'vwap_{period}'] = vwap
            features[f'price_vwap_ratio_{period}'] = close / vwap
        
        # On-Balance Volume (OBV)
        obv = self._calculate_obv(close, volume)
        features['obv'] = obv
        
        # OBV moving averages
        for period in [10, 20, 50]:
            obv_sma = self._safe_rolling(obv, period, lambda x: x.mean())
            features[f'obv_sma_{period}'] = obv_sma
            features[f'obv_ratio_{period}'] = obv / obv_sma
        
        return features
    
    def _calculate_price_position_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de posición de precio."""
        features = {}
        
        # Price position within daily range
        daily_range = high - low
        price_position = (close - low) / daily_range
        price_position = price_position.replace([np.inf, -np.inf], np.nan)
        features['price_position_daily'] = price_position
        
        # Price position within rolling range
        for period in [20, 50]:
            rolling_high = self._safe_rolling(high, period, lambda x: x.max())
            rolling_low = self._safe_rolling(low, period, lambda x: x.min())
            rolling_range = rolling_high - rolling_low
            price_position_rolling = (close - rolling_low) / rolling_range
            price_position_rolling = price_position_rolling.replace([np.inf, -np.inf], np.nan)
            features[f'price_position_{period}'] = price_position_rolling
        
        # High-Low ratio
        for period in [5, 10, 20]:
            high_ratio = high / self._safe_rolling(high, period, lambda x: x.mean())
            low_ratio = low / self._safe_rolling(low, period, lambda x: x.mean())
            features[f'high_ratio_{period}'] = high_ratio
            features[f'low_ratio_{period}'] = low_ratio
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calcula MACD."""
        ema_fast = self._safe_ewm(prices, fast)
        ema_slow = self._safe_ewm(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._safe_ewm(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calcula Bollinger Bands."""
        sma = self._safe_rolling(prices, period, lambda x: x.mean())
        std = self._safe_rolling(prices, period, lambda x: x.std())
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Bollinger Band position
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        bb_position = bb_position.replace([np.inf, -np.inf], np.nan)
        
        # Bollinger Band width
        bb_width = (upper_band - lower_band) / sma
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_position': bb_position,
            'bb_width': bb_width
        }
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula Williams %R."""
        highest_high = self._safe_rolling(high, period, lambda x: x.max())
        lowest_low = self._safe_rolling(low, period, lambda x: x.min())
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
        
        return williams_r
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calcula Stochastic Oscillator."""
        lowest_low = self._safe_rolling(low, k_period, lambda x: x.min())
        highest_high = self._safe_rolling(high, k_period, lambda x: x.max())
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k_percent = k_percent.replace([np.inf, -np.inf], np.nan)
        
        d_percent = self._safe_rolling(k_percent, d_period, lambda x: x.mean())
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula On-Balance Volume."""
        price_change = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

