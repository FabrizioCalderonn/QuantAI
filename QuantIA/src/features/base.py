"""
Clase base para feature engineering.
Define la interfaz común para todos los feature extractors.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Clase base abstracta para todos los feature extractors.
    """
    
    def __init__(self, name: str, lookback_periods: List[int] = None):
        """
        Inicializa el feature extractor.
        
        Args:
            name: Nombre del feature extractor
            lookback_periods: Lista de períodos de lookback para calcular features
        """
        self.name = name
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.features = {}
        
        logger.info(f"Feature extractor '{name}' inicializado con períodos: {self.lookback_periods}")
    
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae features del DataFrame de datos.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features calculados
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Valida que los datos tengan las columnas necesarias.
        
        Args:
            data: DataFrame a validar
            
        Returns:
            True si los datos son válidos
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Columnas faltantes para {self.name}: {missing_columns}")
            return False
        
        if data.empty:
            logger.error(f"Datos vacíos para {self.name}")
            return False
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna la lista de nombres de features generados.
        
        Returns:
            Lista de nombres de features
        """
        return list(self.features.keys())
    
    def _safe_rolling(self, series: pd.Series, window: int, func, **kwargs) -> pd.Series:
        """
        Aplica función rolling de forma segura, manejando ventanas insuficientes.
        
        Args:
            series: Serie de datos
            window: Tamaño de ventana
            func: Función a aplicar
            **kwargs: Argumentos adicionales para la función
            
        Returns:
            Serie con resultados
        """
        if len(series) < window:
            logger.warning(f"Serie demasiado corta para ventana {window}: {len(series)} < {window}")
            return pd.Series(index=series.index, dtype=float)
        
        return series.rolling(window=window, **kwargs).apply(func)
    
    def _safe_ewm(self, series: pd.Series, span: int, **kwargs) -> pd.Series:
        """
        Aplica función ewm de forma segura.
        
        Args:
            series: Serie de datos
            span: Span para ewm
            **kwargs: Argumentos adicionales
            
        Returns:
            Serie con resultados
        """
        if len(series) < 2:
            logger.warning(f"Serie demasiado corta para ewm: {len(series)}")
            return pd.Series(index=series.index, dtype=float)
        
        return series.ewm(span=span, **kwargs).mean()
    
    def _calculate_returns(self, prices: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calcula retornos para diferentes períodos.
        
        Args:
            prices: Serie de precios
            periods: Lista de períodos para calcular retornos
            
        Returns:
            Diccionario con retornos por período
        """
        if periods is None:
            periods = [1, 5, 10, 20]
        
        returns = {}
        for period in periods:
            if period == 1:
                returns[f'returns_{period}'] = prices.pct_change()
            else:
                returns[f'returns_{period}'] = prices.pct_change(periods=period)
        
        return returns
    
    def _calculate_log_returns(self, prices: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calcula log retornos para diferentes períodos.
        
        Args:
            prices: Serie de precios
            periods: Lista de períodos para calcular log retornos
            
        Returns:
            Diccionario con log retornos por período
        """
        if periods is None:
            periods = [1, 5, 10, 20]
        
        log_returns = {}
        for period in periods:
            if period == 1:
                log_returns[f'log_returns_{period}'] = np.log(prices / prices.shift(1))
            else:
                log_returns[f'log_returns_{period}'] = np.log(prices / prices.shift(periods=period))
        
        return log_returns
    
    def _calculate_volatility(self, returns: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calcula volatilidad para diferentes períodos.
        
        Args:
            returns: Serie de retornos
            periods: Lista de períodos para calcular volatilidad
            
        Returns:
            Diccionario con volatilidad por período
        """
        if periods is None:
            periods = [5, 10, 20, 50]
        
        volatility = {}
        for period in periods:
            vol = self._safe_rolling(returns, period, lambda x: x.std() * np.sqrt(252))
            volatility[f'volatility_{period}'] = vol
        
        return volatility
    
    def _calculate_zscore(self, series: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calcula z-scores para diferentes períodos.
        
        Args:
            series: Serie de datos
            periods: Lista de períodos para calcular z-scores
            
        Returns:
            Diccionario con z-scores por período
        """
        if periods is None:
            periods = [20, 50, 100]
        
        zscores = {}
        for period in periods:
            rolling_mean = self._safe_rolling(series, period, lambda x: x.mean())
            rolling_std = self._safe_rolling(series, period, lambda x: x.std())
            zscores[f'zscore_{period}'] = (series - rolling_mean) / rolling_std
        
        return zscores
    
    def _calculate_percentile_rank(self, series: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calcula percentile rank para diferentes períodos.
        
        Args:
            series: Serie de datos
            periods: Lista de períodos para calcular percentile rank
            
        Returns:
            Diccionario con percentile ranks por período
        """
        if periods is None:
            periods = [20, 50, 100]
        
        percentile_ranks = {}
        for period in periods:
            def percentile_rank_func(x):
                if len(x) < 2:
                    return np.nan
                return (x < x.iloc[-1]).sum() / (len(x) - 1) * 100
            
            percentile_ranks[f'percentile_rank_{period}'] = self._safe_rolling(
                series, period, percentile_rank_func
            )
        
        return percentile_ranks


class FeatureStore:
    """
    Store centralizado para features del sistema de trading.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el Feature Store.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config or {}
        self.extractors = {}
        self.features = {}
        self.feature_metadata = {}
        
        logger.info("Feature Store inicializado")
    
    def register_extractor(self, extractor: BaseFeatureExtractor) -> None:
        """
        Registra un feature extractor.
        
        Args:
            extractor: Instancia del feature extractor
        """
        self.extractors[extractor.name] = extractor
        logger.info(f"Feature extractor '{extractor.name}' registrado")
    
    def extract_all_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Extrae todos los features de todos los instrumentos.
        
        Args:
            data: Diccionario con DataFrames por instrumento
            
        Returns:
            Diccionario con features por instrumento
        """
        logger.info(f"Extrayendo features para {len(data)} instrumentos")
        
        all_features = {}
        
        for symbol, df in data.items():
            logger.info(f"Procesando features para {symbol}")
            
            # Combinar features de todos los extractors
            combined_features = pd.DataFrame(index=df.index)
            
            for extractor_name, extractor in self.extractors.items():
                try:
                    if extractor.validate_data(df):
                        features = extractor.extract_features(df)
                        
                        # Agregar prefijo al nombre de las columnas
                        features.columns = [f"{extractor_name}_{col}" for col in features.columns]
                        
                        # Combinar features
                        combined_features = pd.concat([combined_features, features], axis=1)
                        
                        logger.info(f"Features extraídos de {extractor_name} para {symbol}: {len(features.columns)}")
                    else:
                        logger.warning(f"Datos inválidos para {extractor_name} en {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error extrayendo features de {extractor_name} para {symbol}: {str(e)}")
            
            # Remover filas con todos NaN
            combined_features = combined_features.dropna(how='all')
            
            all_features[symbol] = combined_features
            logger.info(f"Features completados para {symbol}: {len(combined_features.columns)} columnas, {len(combined_features)} filas")
        
        self.features = all_features
        return all_features
    
    def get_features(self, symbol: str = None) -> Dict[str, pd.DataFrame]:
        """
        Obtiene features almacenados.
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            Features por instrumento o para un instrumento específico
        """
        if symbol:
            return {symbol: self.features.get(symbol, pd.DataFrame())}
        return self.features
    
    def get_feature_names(self) -> List[str]:
        """
        Obtiene lista de todos los nombres de features.
        
        Returns:
            Lista de nombres de features
        """
        all_features = set()
        for features_df in self.features.values():
            all_features.update(features_df.columns)
        return sorted(list(all_features))
    
    def save_features(self, filepath: str) -> None:
        """
        Guarda features en archivo parquet.
        
        Args:
            filepath: Ruta del archivo
        """
        import os
        from pathlib import Path
        
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, features_df in self.features.items():
            symbol_filepath = save_dir / f"{symbol}_features.parquet"
            features_df.to_parquet(symbol_filepath)
            logger.info(f"Features guardados para {symbol}: {symbol_filepath}")
    
    def load_features(self, filepath: str) -> None:
        """
        Carga features desde archivo parquet.
        
        Args:
            filepath: Ruta del archivo
        """
        from pathlib import Path
        
        load_dir = Path(filepath)
        
        if not load_dir.exists():
            logger.error(f"Directorio no encontrado: {filepath}")
            return
        
        self.features = {}
        
        for feature_file in load_dir.glob("*_features.parquet"):
            symbol = feature_file.stem.replace("_features", "")
            features_df = pd.read_parquet(feature_file)
            self.features[symbol] = features_df
            logger.info(f"Features cargados para {symbol}: {len(features_df.columns)} columnas")
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Genera resumen de features disponibles.
        
        Returns:
            DataFrame con resumen de features
        """
        summary_data = []
        
        for symbol, features_df in self.features.items():
            for feature_name in features_df.columns:
                feature_series = features_df[feature_name]
                
                summary_data.append({
                    'symbol': symbol,
                    'feature_name': feature_name,
                    'count': feature_series.count(),
                    'mean': feature_series.mean(),
                    'std': feature_series.std(),
                    'min': feature_series.min(),
                    'max': feature_series.max(),
                    'null_count': feature_series.isnull().sum(),
                    'null_pct': feature_series.isnull().sum() / len(feature_series) * 100
                })
        
        return pd.DataFrame(summary_data)

