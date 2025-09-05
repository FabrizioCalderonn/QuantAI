"""
Feature extractors para features cross-asset.
Incluye correlaciones, carry, term structure y factores macro.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy.stats import pearsonr, spearmanr

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class CrossAssetFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor de features cross-asset.
    Requiere datos de múltiples instrumentos.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        super().__init__("cross_asset", lookback_periods)
        self.all_data = {}  # Almacenará datos de todos los instrumentos
    
    def set_all_data(self, all_data: Dict[str, pd.DataFrame]) -> None:
        """
        Establece datos de todos los instrumentos.
        
        Args:
            all_data: Diccionario con DataFrames por instrumento
        """
        self.all_data = all_data
        logger.info(f"Datos cross-asset establecidos para {len(all_data)} instrumentos")
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae features cross-asset.
        
        Args:
            data: DataFrame con datos OHLCV del instrumento actual
            
        Returns:
            DataFrame con features cross-asset
        """
        if not self.validate_data(data):
            return pd.DataFrame()
        
        if len(self.all_data) < 2:
            logger.warning("Se necesitan al menos 2 instrumentos para features cross-asset")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        # Obtener símbolo actual (asumiendo que está en los metadatos)
        current_symbol = data.attrs.get('symbol', 'UNKNOWN')
        
        # 1. FEATURES DE CORRELACIÓN
        correlation_features = self._calculate_correlation_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(correlation_features)], axis=1)
        
        # 2. FEATURES DE CARRY
        carry_features = self._calculate_carry_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(carry_features)], axis=1)
        
        # 3. FEATURES DE TERM STRUCTURE
        term_structure_features = self._calculate_term_structure_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(term_structure_features)], axis=1)
        
        # 4. FEATURES DE FACTORES MACRO
        macro_features = self._calculate_macro_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(macro_features)], axis=1)
        
        # 5. FEATURES DE MOMENTUM CROSS-ASSET
        cross_momentum_features = self._calculate_cross_momentum_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(cross_momentum_features)], axis=1)
        
        # 6. FEATURES DE VOLATILIDAD CROSS-ASSET
        cross_vol_features = self._calculate_cross_volatility_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(cross_vol_features)], axis=1)
        
        # 7. FEATURES DE REGÍMENES CROSS-ASSET
        cross_regime_features = self._calculate_cross_regime_features(data, current_symbol)
        features = pd.concat([features, pd.DataFrame(cross_regime_features)], axis=1)
        
        # Almacenar nombres de features
        self.features = {col: col for col in features.columns}
        
        return features
    
    def _calculate_correlation_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features de correlación con otros activos."""
        features = {}
        
        current_returns = data['close'].pct_change()
        
        for other_symbol, other_data in self.all_data.items():
            if other_symbol == current_symbol:
                continue
            
            if 'close' not in other_data.columns:
                continue
            
            other_returns = other_data['close'].pct_change()
            
            # Alinear índices
            common_index = current_returns.index.intersection(other_returns.index)
            if len(common_index) < 50:
                continue
            
            current_aligned = current_returns.loc[common_index]
            other_aligned = other_returns.loc[common_index]
            
            # Correlaciones rolling
            for period in [20, 50, 100]:
                def corr_func(x, y):
                    if len(x) < 10 or len(y) < 10:
                        return np.nan
                    try:
                        # Alinear x e y
                        common_len = min(len(x), len(y))
                        x_vals = x.iloc[-common_len:]
                        y_vals = y.iloc[-common_len:]
                        
                        if len(x_vals) < 5:
                            return np.nan
                        
                        corr, _ = pearsonr(x_vals, y_vals)
                        return corr if not np.isnan(corr) else np.nan
                    except:
                        return np.nan
                
                # Correlación rolling
                corr_series = pd.Series(index=common_index, dtype=float)
                for i in range(period, len(common_index)):
                    start_idx = i - period
                    end_idx = i
                    x_window = current_aligned.iloc[start_idx:end_idx]
                    y_window = other_aligned.iloc[start_idx:end_idx]
                    corr_series.iloc[i] = corr_func(x_window, y_window)
                
                # Reindexar al índice original
                corr_series = corr_series.reindex(data.index, method='ffill')
                features[f'corr_{other_symbol}_{period}'] = corr_series
                
                # Correlación de volatilidad
                current_vol = current_returns.rolling(20).std()
                other_vol = other_returns.rolling(20).std()
                
                vol_corr_series = pd.Series(index=common_index, dtype=float)
                for i in range(period, len(common_index)):
                    start_idx = i - period
                    end_idx = i
                    x_vol_window = current_vol.loc[common_index].iloc[start_idx:end_idx]
                    y_vol_window = other_vol.loc[common_index].iloc[start_idx:end_idx]
                    vol_corr_series.iloc[i] = corr_func(x_vol_window, y_vol_window)
                
                vol_corr_series = vol_corr_series.reindex(data.index, method='ffill')
                features[f'vol_corr_{other_symbol}_{period}'] = vol_corr_series
        
        return features
    
    def _calculate_carry_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features de carry (simplificado)."""
        features = {}
        
        # Carry simplificado basado en momentum de largo plazo vs corto plazo
        close = data['close']
        
        for short_period in [20, 50]:
            for long_period in [100, 200]:
                if long_period <= short_period:
                    continue
                
                short_ma = self._safe_rolling(close, short_period, lambda x: x.mean())
                long_ma = self._safe_rolling(close, long_period, lambda x: x.mean())
                
                # Carry como diferencia entre medias móviles
                carry = (short_ma - long_ma) / long_ma
                features[f'carry_{short_period}_{long_period}'] = carry
                
                # Carry momentum
                carry_momentum = carry.diff(5)
                features[f'carry_momentum_{short_period}_{long_period}'] = carry_momentum
        
        # Carry vs otros activos
        current_returns = data['close'].pct_change()
        
        for other_symbol, other_data in self.all_data.items():
            if other_symbol == current_symbol or 'close' not in other_data.columns:
                continue
            
            other_returns = other_data['close'].pct_change()
            
            # Alinear datos
            common_index = current_returns.index.intersection(other_returns.index)
            if len(common_index) < 100:
                continue
            
            current_aligned = current_returns.loc[common_index]
            other_aligned = other_returns.loc[common_index]
            
            # Carry relativo (retorno del activo vs retorno promedio del mercado)
            market_returns = other_aligned.rolling(20).mean()
            relative_carry = current_aligned - market_returns
            relative_carry = relative_carry.reindex(data.index, method='ffill')
            
            features[f'relative_carry_{other_symbol}'] = relative_carry
        
        return features
    
    def _calculate_term_structure_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features de term structure (simplificado)."""
        features = {}
        
        close = data['close']
        
        # Term structure basado en diferentes timeframes (simplificado)
        # En un sistema real, esto usaría datos de futuros con diferentes vencimientos
        
        for short_period in [5, 10, 20]:
            for long_period in [50, 100, 200]:
                if long_period <= short_period:
                    continue
                
                # Slope de la curva (simplificado)
                short_vol = self._safe_rolling(close.pct_change(), short_period, lambda x: x.std())
                long_vol = self._safe_rolling(close.pct_change(), long_period, lambda x: x.std())
                
                term_slope = (short_vol - long_vol) / long_vol
                features[f'term_slope_{short_period}_{long_period}'] = term_slope
                
                # Curvature
                mid_period = (short_period + long_period) // 2
                mid_vol = self._safe_rolling(close.pct_change(), mid_period, lambda x: x.std())
                
                curvature = 2 * mid_vol - short_vol - long_vol
                features[f'term_curvature_{short_period}_{long_period}'] = curvature
        
        return features
    
    def _calculate_macro_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features macro (simplificado)."""
        features = {}
        
        close = data['close']
        returns = close.pct_change()
        
        # Risk-on/Risk-off indicator
        risk_assets = []
        safe_assets = []
        
        for symbol, other_data in self.all_data.items():
            if symbol == current_symbol or 'close' not in other_data.columns:
                continue
            
            # Clasificar activos por volatilidad promedio
            other_returns = other_data['close'].pct_change()
            avg_vol = other_returns.rolling(50).std().mean()
            
            if avg_vol > returns.rolling(50).std().mean():
                risk_assets.append(other_returns)
            else:
                safe_assets.append(other_returns)
        
        if risk_assets and safe_assets:
            # Risk-on/Risk-off ratio
            risk_performance = pd.concat(risk_assets, axis=1).mean(axis=1)
            safe_performance = pd.concat(safe_assets, axis=1).mean(axis=1)
            
            risk_on_off = risk_performance - safe_performance
            risk_on_off = risk_on_off.reindex(data.index, method='ffill')
            
            features['risk_on_off'] = risk_on_off
            
            # Rolling correlation con risk-on/risk-off
            for period in [20, 50]:
                def corr_with_risk(x, risk_series):
                    if len(x) < 10:
                        return np.nan
                    try:
                        common_len = min(len(x), len(risk_series))
                        x_vals = x.iloc[-common_len:]
                        risk_vals = risk_series.iloc[-common_len:]
                        corr, _ = pearsonr(x_vals, risk_vals)
                        return corr if not np.isnan(corr) else np.nan
                    except:
                        return np.nan
                
                corr_risk = pd.Series(index=data.index, dtype=float)
                for i in range(period, len(data)):
                    start_idx = i - period
                    end_idx = i
                    returns_window = returns.iloc[start_idx:end_idx]
                    risk_window = risk_on_off.iloc[start_idx:end_idx]
                    corr_risk.iloc[i] = corr_with_risk(returns_window, risk_window)
                
                features[f'corr_risk_on_off_{period}'] = corr_risk
        
        return features
    
    def _calculate_cross_momentum_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features de momentum cross-asset."""
        features = {}
        
        current_returns = data['close'].pct_change()
        
        # Momentum relativo vs otros activos
        for other_symbol, other_data in self.all_data.items():
            if other_symbol == current_symbol or 'close' not in other_data.columns:
                continue
            
            other_returns = other_data['close'].pct_change()
            
            # Alinear datos
            common_index = current_returns.index.intersection(other_returns.index)
            if len(common_index) < 50:
                continue
            
            current_aligned = current_returns.loc[common_index]
            other_aligned = other_returns.loc[common_index]
            
            for period in [20, 50, 100]:
                # Momentum relativo
                current_momentum = current_aligned.rolling(period).sum()
                other_momentum = other_aligned.rolling(period).sum()
                
                relative_momentum = current_momentum - other_momentum
                relative_momentum = relative_momentum.reindex(data.index, method='ffill')
                
                features[f'relative_momentum_{other_symbol}_{period}'] = relative_momentum
                
                # Momentum ranking
                all_momentums = []
                for sym, sym_data in self.all_data.items():
                    if 'close' in sym_data.columns:
                        sym_returns = sym_data['close'].pct_change()
                        sym_aligned = sym_returns.loc[common_index]
                        sym_momentum = sym_aligned.rolling(period).sum()
                        all_momentums.append(sym_momentum)
                
                if len(all_momentums) > 1:
                    all_momentums_df = pd.concat(all_momentums, axis=1)
                    momentum_rank = all_momentums_df.rank(axis=1, pct=True).iloc[:, 0]
                    momentum_rank = momentum_rank.reindex(data.index, method='ffill')
                    features[f'momentum_rank_{period}'] = momentum_rank
        
        return features
    
    def _calculate_cross_volatility_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features de volatilidad cross-asset."""
        features = {}
        
        current_returns = data['close'].pct_change()
        current_vol = current_returns.rolling(20).std()
        
        # Volatilidad relativa vs otros activos
        all_vols = []
        
        for other_symbol, other_data in self.all_data.items():
            if 'close' not in other_data.columns:
                continue
            
            other_returns = other_data['close'].pct_change()
            other_vol = other_returns.rolling(20).std()
            
            # Alinear datos
            common_index = current_vol.index.intersection(other_vol.index)
            if len(common_index) < 50:
                continue
            
            other_vol_aligned = other_vol.loc[common_index]
            all_vols.append(other_vol_aligned)
        
        if all_vols:
            # Volatilidad promedio del mercado
            market_vol = pd.concat(all_vols, axis=1).mean(axis=1)
            market_vol = market_vol.reindex(data.index, method='ffill')
            
            # Volatilidad relativa
            relative_vol = current_vol / market_vol
            features['relative_volatility'] = relative_vol
            
            # Volatility clustering cross-asset
            for period in [20, 50]:
                vol_clustering = self._safe_rolling(relative_vol, period, lambda x: x.autocorr(lag=1))
                features[f'vol_clustering_cross_{period}'] = vol_clustering
        
        return features
    
    def _calculate_cross_regime_features(self, data: pd.DataFrame, current_symbol: str) -> Dict[str, pd.Series]:
        """Calcula features de regímenes cross-asset."""
        features = {}
        
        current_returns = data['close'].pct_change()
        
        # Regímenes de correlación
        for other_symbol, other_data in self.all_data.items():
            if other_symbol == current_symbol or 'close' not in other_data.columns:
                continue
            
            other_returns = other_data['close'].pct_change()
            
            # Alinear datos
            common_index = current_returns.index.intersection(other_returns.index)
            if len(common_index) < 100:
                continue
            
            current_aligned = current_returns.loc[common_index]
            other_aligned = other_returns.loc[common_index]
            
            # Regímenes de alta correlación
            for period in [50, 100]:
                def high_corr_regime(x, y):
                    if len(x) < 20 or len(y) < 20:
                        return np.nan
                    try:
                        common_len = min(len(x), len(y))
                        x_vals = x.iloc[-common_len:]
                        y_vals = y.iloc[-common_len:]
                        corr, _ = pearsonr(x_vals, y_vals)
                        return 1 if corr > 0.7 else 0
                    except:
                        return np.nan
                
                regime_series = pd.Series(index=common_index, dtype=float)
                for i in range(period, len(common_index)):
                    start_idx = i - period
                    end_idx = i
                    x_window = current_aligned.iloc[start_idx:end_idx]
                    y_window = other_aligned.iloc[start_idx:end_idx]
                    regime_series.iloc[i] = high_corr_regime(x_window, y_window)
                
                regime_series = regime_series.reindex(data.index, method='ffill')
                features[f'high_corr_regime_{other_symbol}_{period}'] = regime_series
        
        return features

