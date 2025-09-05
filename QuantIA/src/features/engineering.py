"""
Módulo principal de feature engineering.
Combina todos los extractores de features y maneja el pipeline completo.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime

from .base import FeatureStore
from .technical import TechnicalFeatureExtractor
from .statistical import StatisticalFeatureExtractor
from .cross_asset import CrossAssetFeatureExtractor
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Ingeniero de features principal que coordina todos los extractores.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el feature engineer.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.feature_store = FeatureStore(self.config)
        
        # Inicializar extractores
        self._initialize_extractors()
        
        # Configuración de features
        self.feature_config = self.config.get('features', {})
        self.lookback_periods = self.feature_config.get('lookback_periods', [5, 10, 20, 50, 100])
        
        logger.info("Feature Engineer inicializado")
    
    def _initialize_extractors(self):
        """Inicializa todos los extractores de features."""
        # Technical features
        technical_extractor = TechnicalFeatureExtractor(self.lookback_periods)
        self.feature_store.register_extractor(technical_extractor)
        
        # Statistical features
        statistical_extractor = StatisticalFeatureExtractor(self.lookback_periods)
        self.feature_store.register_extractor(statistical_extractor)
        
        # Cross-asset features
        cross_asset_extractor = CrossAssetFeatureExtractor(self.lookback_periods)
        self.feature_store.register_extractor(cross_asset_extractor)
        
        logger.info("Extractores de features inicializados")
    
    def create_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Crea features para todos los instrumentos.
        
        Args:
            data: Diccionario con DataFrames por instrumento
            
        Returns:
            Diccionario con features por instrumento
        """
        logger.info(f"Creando features para {len(data)} instrumentos")
        
        # Configurar datos cross-asset
        cross_asset_extractor = self.feature_store.extractors.get('cross_asset')
        if cross_asset_extractor:
            cross_asset_extractor.set_all_data(data)
        
        # Extraer features
        features = self.feature_store.extract_all_features(data)
        
        # Post-procesamiento
        features = self._post_process_features(features)
        
        # Validar features
        features = self._validate_features(features)
        
        logger.info("Features creados exitosamente")
        return features
    
    def _post_process_features(self, features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Post-procesa features (normalización, limpieza, etc.).
        
        Args:
            features: Diccionario con features por instrumento
            
        Returns:
            Features post-procesados
        """
        logger.info("Post-procesando features...")
        
        processed_features = {}
        
        for symbol, feature_df in features.items():
            if feature_df.empty:
                continue
            
            processed_df = feature_df.copy()
            
            # 1. Remover outliers extremos
            processed_df = self._remove_outliers(processed_df)
            
            # 2. Normalización por volatilidad
            processed_df = self._normalize_by_volatility(processed_df, symbol)
            
            # 3. Winsorización
            processed_df = self._winsorize_features(processed_df)
            
            # 4. Imputación de valores faltantes
            processed_df = self._impute_missing_values(processed_df)
            
            # 5. Agregar metadatos
            processed_df.attrs['symbol'] = symbol
            processed_df.attrs['created_at'] = datetime.now().isoformat()
            processed_df.attrs['feature_count'] = len(processed_df.columns)
            
            processed_features[symbol] = processed_df
            
            logger.info(f"Features post-procesados para {symbol}: {len(processed_df.columns)} features")
        
        return processed_features
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remueve outliers de los features.
        
        Args:
            df: DataFrame con features
            method: Método para detectar outliers ('iqr', 'zscore', 'modified_zscore')
            threshold: Umbral para detección de outliers
            
        Returns:
            DataFrame sin outliers
        """
        df_clean = df.copy()
        
        for column in df.columns:
            if df[column].dtype in ['object', 'bool']:
                continue
            
            series = df[column].dropna()
            if len(series) < 10:
                continue
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
            
            else:
                continue
            
            # Reemplazar outliers con NaN
            df_clean.loc[outliers, column] = np.nan
        
        return df_clean
    
    def _normalize_by_volatility(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normaliza features por volatilidad.
        
        Args:
            df: DataFrame con features
            symbol: Símbolo del instrumento
            
        Returns:
            DataFrame normalizado
        """
        df_normalized = df.copy()
        
        # Buscar columnas de volatilidad
        vol_columns = [col for col in df.columns if 'volatility' in col.lower() or 'vol_' in col.lower()]
        
        if not vol_columns:
            return df_normalized
        
        # Usar la primera columna de volatilidad encontrada
        vol_series = df[vol_columns[0]]
        
        # Normalizar features por volatilidad
        for column in df.columns:
            if column in vol_columns or df[column].dtype in ['object', 'bool']:
                continue
            
            # Normalizar solo si la volatilidad no es cero
            vol_mask = vol_series > 0
            if vol_mask.any():
                df_normalized.loc[vol_mask, column] = df.loc[vol_mask, column] / vol_series.loc[vol_mask]
        
        return df_normalized
    
    def _winsorize_features(self, df: pd.DataFrame, limits: tuple = (0.01, 0.01)) -> pd.DataFrame:
        """
        Aplica winsorización a los features.
        
        Args:
            df: DataFrame con features
            limits: Límites de winsorización (lower, upper)
            
        Returns:
            DataFrame winsorizado
        """
        df_winsorized = df.copy()
        
        for column in df.columns:
            if df[column].dtype in ['object', 'bool']:
                continue
            
            series = df[column].dropna()
            if len(series) < 10:
                continue
            
            # Calcular percentiles
            lower_limit = series.quantile(limits[0])
            upper_limit = series.quantile(1 - limits[1])
            
            # Aplicar winsorización
            df_winsorized[column] = df[column].clip(lower=lower_limit, upper=upper_limit)
        
        return df_winsorized
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores faltantes en los features.
        
        Args:
            df: DataFrame con features
            
        Returns:
            DataFrame con valores imputados
        """
        df_imputed = df.copy()
        
        for column in df.columns:
            if df[column].dtype in ['object', 'bool']:
                continue
            
            # Estrategias de imputación
            if df[column].isnull().sum() > 0:
                # Forward fill primero
                df_imputed[column] = df[column].fillna(method='ffill')
                
                # Luego backward fill
                df_imputed[column] = df_imputed[column].fillna(method='bfill')
                
                # Finalmente, media rolling
                if df_imputed[column].isnull().sum() > 0:
                    rolling_mean = df_imputed[column].rolling(window=20, min_periods=1).mean()
                    df_imputed[column] = df_imputed[column].fillna(rolling_mean)
        
        return df_imputed
    
    def _validate_features(self, features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Valida la calidad de los features.
        
        Args:
            features: Diccionario con features por instrumento
            
        Returns:
            Features validados
        """
        logger.info("Validando features...")
        
        validated_features = {}
        
        for symbol, feature_df in features.items():
            if feature_df.empty:
                logger.warning(f"Features vacíos para {symbol}")
                continue
            
            # Validaciones
            issues = []
            
            # 1. Verificar que no todas las columnas sean NaN
            all_nan_columns = feature_df.columns[feature_df.isnull().all()].tolist()
            if all_nan_columns:
                issues.append(f"Columnas completamente NaN: {all_nan_columns}")
            
            # 2. Verificar que no todas las filas sean NaN
            all_nan_rows = feature_df.isnull().all(axis=1).sum()
            if all_nan_rows > len(feature_df) * 0.5:
                issues.append(f"Demasiadas filas NaN: {all_nan_rows}")
            
            # 3. Verificar valores infinitos
            inf_columns = []
            for col in feature_df.columns:
                if feature_df[col].dtype in ['float64', 'float32']:
                    if np.isinf(feature_df[col]).any():
                        inf_columns.append(col)
            
            if inf_columns:
                issues.append(f"Columnas con valores infinitos: {inf_columns}")
            
            # Reportar issues
            if issues:
                logger.warning(f"Issues en features de {symbol}: {issues}")
                
                # Limpiar issues
                clean_df = feature_df.copy()
                
                # Remover columnas completamente NaN
                clean_df = clean_df.dropna(axis=1, how='all')
                
                # Reemplazar infinitos con NaN
                clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
                
                # Remover filas con demasiados NaN
                max_nan_pct = 0.8
                clean_df = clean_df.dropna(thresh=int(len(clean_df.columns) * (1 - max_nan_pct)))
                
                validated_features[symbol] = clean_df
            else:
                validated_features[symbol] = feature_df
            
            logger.info(f"Features validados para {symbol}: {len(validated_features[symbol].columns)} features, {len(validated_features[symbol])} filas")
        
        return validated_features
    
    def get_feature_importance(self, features: Dict[str, pd.DataFrame], 
                             target: str = 'returns_1') -> Dict[str, pd.DataFrame]:
        """
        Calcula importancia de features usando correlación con target.
        
        Args:
            features: Diccionario con features por instrumento
            target: Nombre de la columna target
            
        Returns:
            Diccionario con importancia de features por instrumento
        """
        importance_results = {}
        
        for symbol, feature_df in features.items():
            if target not in feature_df.columns:
                logger.warning(f"Target '{target}' no encontrado en {symbol}")
                continue
            
            target_series = feature_df[target]
            feature_columns = [col for col in feature_df.columns if col != target]
            
            importance_data = []
            
            for feature_col in feature_columns:
                feature_series = feature_df[feature_col]
                
                # Calcular correlación
                correlation = feature_series.corr(target_series)
                
                # Calcular información mutua (simplificado)
                mutual_info = abs(correlation)  # Simplificación
            
                importance_data.append({
                    'feature': feature_col,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation),
                    'mutual_info': mutual_info
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('abs_correlation', ascending=False)
            
            importance_results[symbol] = importance_df
            
            logger.info(f"Importancia calculada para {symbol}: {len(importance_df)} features")
        
        return importance_results
    
    def save_features(self, features: Dict[str, pd.DataFrame], 
                     filepath: str = "data/processed/features") -> None:
        """
        Guarda features en archivos parquet.
        
        Args:
            features: Diccionario con features por instrumento
            filepath: Ruta base para guardar archivos
        """
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for symbol, feature_df in features.items():
            symbol_filepath = save_path / f"{symbol}_features.parquet"
            feature_df.to_parquet(symbol_filepath)
            logger.info(f"Features guardados para {symbol}: {symbol_filepath}")
        
        # Guardar metadatos
        metadata = {
            'created_at': datetime.now().isoformat(),
            'symbols': list(features.keys()),
            'feature_counts': {symbol: len(df.columns) for symbol, df in features.items()},
            'total_features': sum(len(df.columns) for df in features.values())
        }
        
        metadata_filepath = save_path / "features_metadata.json"
        import json
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadatos guardados: {metadata_filepath}")
    
    def load_features(self, filepath: str = "data/processed/features") -> Dict[str, pd.DataFrame]:
        """
        Carga features desde archivos parquet.
        
        Args:
            filepath: Ruta base de los archivos
            
        Returns:
            Diccionario con features por instrumento
        """
        load_path = Path(filepath)
        
        if not load_path.exists():
            logger.error(f"Directorio no encontrado: {filepath}")
            return {}
        
        features = {}
        
        for feature_file in load_path.glob("*_features.parquet"):
            symbol = feature_file.stem.replace("_features", "")
            feature_df = pd.read_parquet(feature_file)
            features[symbol] = feature_df
            logger.info(f"Features cargados para {symbol}: {len(feature_df.columns)} features")
        
        return features
    
    def get_feature_summary(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Genera resumen completo de features.
        
        Args:
            features: Diccionario con features por instrumento
            
        Returns:
            DataFrame con resumen de features
        """
        return self.feature_store.get_feature_summary()


def main():
    """
    Función principal para demostrar el feature engineering.
    """
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear feature engineer
    engineer = FeatureEngineer()
    
    # Cargar datos de ejemplo (esto requeriría datos reales)
    print("Feature Engineer inicializado correctamente")
    print(f"Extractores registrados: {list(engineer.feature_store.extractors.keys())}")
    print(f"Períodos de lookback: {engineer.lookback_periods}")


if __name__ == "__main__":
    main()

