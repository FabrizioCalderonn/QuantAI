"""
Feature extractors para features estadísticos avanzados.
Incluye análisis de regímenes, estacionalidad y correlaciones.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class StatisticalFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor de features estadísticos avanzados.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        super().__init__("statistical", lookback_periods)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae features estadísticos avanzados.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features estadísticos
        """
        if not self.validate_data(data):
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calcular retornos
        returns = close.pct_change().dropna()
        
        # 1. FEATURES DE DISTRIBUCIÓN
        distribution_features = self._calculate_distribution_features(returns)
        features = pd.concat([features, pd.DataFrame(distribution_features)], axis=1)
        
        # 2. FEATURES DE AUTOCORRELACIÓN
        autocorr_features = self._calculate_autocorrelation_features(returns)
        features = pd.concat([features, pd.DataFrame(autocorr_features)], axis=1)
        
        # 3. FEATURES DE VOLATILIDAD AVANZADOS
        volatility_features = self._calculate_advanced_volatility_features(returns, close)
        features = pd.concat([features, pd.DataFrame(volatility_features)], axis=1)
        
        # 4. FEATURES DE REGÍMENES
        regime_features = self._calculate_regime_features(returns, close)
        features = pd.concat([features, pd.DataFrame(regime_features)], axis=1)
        
        # 5. FEATURES DE ESTACIONALIDAD
        seasonality_features = self._calculate_seasonality_features(close, returns)
        features = pd.concat([features, pd.DataFrame(seasonality_features)], axis=1)
        
        # 6. FEATURES DE MOMENTOS ESTADÍSTICOS
        moments_features = self._calculate_moments_features(returns)
        features = pd.concat([features, pd.DataFrame(moments_features)], axis=1)
        
        # 7. FEATURES DE PERSISTENCIA
        persistence_features = self._calculate_persistence_features(returns)
        features = pd.concat([features, pd.DataFrame(persistence_features)], axis=1)
        
        # 8. FEATURES DE FRACTALES
        fractal_features = self._calculate_fractal_features(close, high, low)
        features = pd.concat([features, pd.DataFrame(fractal_features)], axis=1)
        
        # Almacenar nombres de features
        self.features = {col: col for col in features.columns}
        
        return features
    
    def _calculate_distribution_features(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de distribución de retornos."""
        features = {}
        
        for period in [20, 50, 100]:
            # Skewness
            skew = self._safe_rolling(returns, period, lambda x: stats.skew(x))
            features[f'skewness_{period}'] = skew
            
            # Kurtosis
            kurtosis = self._safe_rolling(returns, period, lambda x: stats.kurtosis(x))
            features[f'kurtosis_{period}'] = kurtosis
            
            # Jarque-Bera test statistic
            def jb_stat(x):
                if len(x) < 4:
                    return np.nan
                return stats.jarque_bera(x)[0]
            
            jb = self._safe_rolling(returns, period, jb_stat)
            features[f'jarque_bera_{period}'] = jb
            
            # Percentiles
            for pct in [5, 10, 25, 75, 90, 95]:
                pct_val = self._safe_rolling(returns, period, lambda x: np.percentile(x, pct))
                features[f'percentile_{pct}_{period}'] = pct_val
            
            # Interquartile Range
            iqr = self._safe_rolling(returns, period, lambda x: np.percentile(x, 75) - np.percentile(x, 25))
            features[f'iqr_{period}'] = iqr
        
        return features
    
    def _calculate_autocorrelation_features(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de autocorrelación."""
        features = {}
        
        for period in [20, 50, 100]:
            for lag in [1, 2, 3, 5, 10]:
                def autocorr_func(x):
                    if len(x) < lag + 5:
                        return np.nan
                    return x.autocorr(lag=lag)
                
                autocorr = self._safe_rolling(returns, period, autocorr_func)
                features[f'autocorr_lag{lag}_{period}'] = autocorr
            
            # Ljung-Box test statistic
            def ljung_box_stat(x):
                if len(x) < 10:
                    return np.nan
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    result = acorr_ljungbox(x, lags=min(10, len(x)//4), return_df=False)
                    return result[0][0] if len(result[0]) > 0 else np.nan
                except:
                    return np.nan
            
            lb_stat = self._safe_rolling(returns, period, ljung_box_stat)
            features[f'ljung_box_{period}'] = lb_stat
        
        return features
    
    def _calculate_advanced_volatility_features(self, returns: pd.Series, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de volatilidad avanzados."""
        features = {}
        
        for period in [20, 50, 100]:
            # GARCH-like volatility (EWMA)
            vol_ewma = self._safe_ewm(returns**2, period)
            features[f'vol_ewma_{period}'] = np.sqrt(vol_ewma * 252)
            
            # Parkinson volatility (using high-low)
            # Note: This would need high-low data, simplified here
            vol_parkinson = self._safe_rolling(returns, period, lambda x: np.sqrt(np.mean(x**2) * 252))
            features[f'vol_parkinson_{period}'] = vol_parkinson
            
            # Volatility of volatility
            vol_series = self._safe_rolling(returns, 20, lambda x: x.std() * np.sqrt(252))
            vol_of_vol = self._safe_rolling(vol_series, period, lambda x: x.std())
            features[f'vol_of_vol_{period}'] = vol_of_vol
            
            # Volatility clustering
            vol_clustering = self._safe_rolling(returns**2, period, lambda x: x.autocorr(lag=1))
            features[f'vol_clustering_{period}'] = vol_clustering
        
        return features
    
    def _calculate_regime_features(self, returns: pd.Series, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de detección de regímenes."""
        features = {}
        
        for period in [50, 100, 200]:
            # Rolling volatility regime
            vol = self._safe_rolling(returns, 20, lambda x: x.std() * np.sqrt(252))
            vol_ma = self._safe_rolling(vol, period, lambda x: x.mean())
            vol_regime = (vol > vol_ma * 1.5).astype(int)
            features[f'high_vol_regime_{period}'] = vol_regime
            
            # Trend regime (simplified)
            price_ma = self._safe_rolling(prices, period, lambda x: x.mean())
            trend_regime = (prices > price_ma).astype(int)
            features[f'uptrend_regime_{period}'] = trend_regime
            
            # Momentum regime
            momentum = prices / prices.shift(period) - 1
            momentum_regime = (momentum > 0.05).astype(int)  # 5% threshold
            features[f'positive_momentum_regime_{period}'] = momentum_regime
        
        # Hidden Markov Model-like features (simplified)
        hmm_features = self._calculate_simple_hmm_features(returns)
        features.update(hmm_features)
        
        return features
    
    def _calculate_seasonality_features(self, prices: pd.Series, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de estacionalidad."""
        features = {}
        
        # Day of week effects
        if hasattr(prices.index, 'dayofweek'):
            for day in range(7):
                day_mask = prices.index.dayofweek == day
                day_returns = returns[day_mask]
                
                if len(day_returns) > 10:
                    day_mean = day_returns.rolling(50).mean()
                    features[f'day_{day}_mean'] = day_mean.reindex(prices.index, method='ffill')
        
        # Month effects
        if hasattr(prices.index, 'month'):
            for month in range(1, 13):
                month_mask = prices.index.month == month
                month_returns = returns[month_mask]
                
                if len(month_returns) > 5:
                    month_mean = month_returns.rolling(20).mean()
                    features[f'month_{month}_mean'] = month_mean.reindex(prices.index, method='ffill')
        
        # Hour effects (for intraday data)
        if hasattr(prices.index, 'hour'):
            for hour in range(24):
                hour_mask = prices.index.hour == hour
                hour_returns = returns[hour_mask]
                
                if len(hour_returns) > 10:
                    hour_mean = hour_returns.rolling(50).mean()
                    features[f'hour_{hour}_mean'] = hour_mean.reindex(prices.index, method='ffill')
        
        return features
    
    def _calculate_moments_features(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de momentos estadísticos."""
        features = {}
        
        for period in [20, 50, 100]:
            # Higher moments
            def moment_func(x, order):
                if len(x) < 3:
                    return np.nan
                return np.mean((x - np.mean(x))**order)
            
            # 3rd moment (skewness)
            moment3 = self._safe_rolling(returns, period, lambda x: moment_func(x, 3))
            features[f'moment3_{period}'] = moment3
            
            # 4th moment (kurtosis)
            moment4 = self._safe_rolling(returns, period, lambda x: moment_func(x, 4))
            features[f'moment4_{period}'] = moment4
            
            # 5th moment
            moment5 = self._safe_rolling(returns, period, lambda x: moment_func(x, 5))
            features[f'moment5_{period}'] = moment5
            
            # 6th moment
            moment6 = self._safe_rolling(returns, period, lambda x: moment_func(x, 6))
            features[f'moment6_{period}'] = moment6
        
        return features
    
    def _calculate_persistence_features(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features de persistencia."""
        features = {}
        
        for period in [20, 50, 100]:
            # Hurst exponent (simplified)
            def hurst_func(x):
                if len(x) < 10:
                    return np.nan
                try:
                    # Simplified Hurst calculation
                    lags = range(2, min(10, len(x)//2))
                    tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] * 2.0
                except:
                    return np.nan
            
            hurst = self._safe_rolling(returns, period, hurst_func)
            features[f'hurst_{period}'] = hurst
            
            # Variance ratio
            def variance_ratio_func(x):
                if len(x) < 20:
                    return np.nan
                try:
                    # Simplified variance ratio
                    var_1 = np.var(x)
                    var_2 = np.var(x[::2])  # Every other observation
                    return var_2 / (2 * var_1) if var_1 > 0 else np.nan
                except:
                    return np.nan
            
            var_ratio = self._safe_rolling(returns, period, variance_ratio_func)
            features[f'variance_ratio_{period}'] = var_ratio
        
        return features
    
    def _calculate_fractal_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features fractales."""
        features = {}
        
        for period in [20, 50, 100]:
            # Fractal dimension (simplified)
            def fractal_dim_func(x):
                if len(x) < 10:
                    return np.nan
                try:
                    # Simplified box-counting dimension
                    n = len(x)
                    scales = [2, 4, 8, 16]
                    counts = []
                    
                    for scale in scales:
                        if scale >= n:
                            continue
                        boxes = n // scale
                        count = 0
                        for i in range(boxes):
                            start = i * scale
                            end = min(start + scale, n)
                            box_data = x.iloc[start:end]
                            if len(box_data) > 0:
                                count += 1
                        counts.append(count)
                    
                    if len(counts) >= 2:
                        poly = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
                        return -poly[0]
                    return np.nan
                except:
                    return np.nan
            
            fractal_dim = self._safe_rolling(close, period, fractal_dim_func)
            features[f'fractal_dimension_{period}'] = fractal_dim
        
        return features
    
    def _calculate_simple_hmm_features(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calcula features simplificados tipo HMM."""
        features = {}
        
        # Volatility states
        vol = self._safe_rolling(returns, 20, lambda x: x.std() * np.sqrt(252))
        vol_quantiles = vol.rolling(100).quantile([0.33, 0.67])
        
        if not vol_quantiles.empty:
            low_vol_threshold = vol_quantiles.iloc[:, 0]
            high_vol_threshold = vol_quantiles.iloc[:, 1]
            
            # State 0: Low volatility
            state_0 = (vol <= low_vol_threshold).astype(int)
            features['hmm_state_low_vol'] = state_0
            
            # State 1: Medium volatility
            state_1 = ((vol > low_vol_threshold) & (vol <= high_vol_threshold)).astype(int)
            features['hmm_state_medium_vol'] = state_1
            
            # State 2: High volatility
            state_2 = (vol > high_vol_threshold).astype(int)
            features['hmm_state_high_vol'] = state_2
        
        return features

