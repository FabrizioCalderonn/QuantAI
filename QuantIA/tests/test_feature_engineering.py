"""
Tests para el sistema de feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.features.engineering import FeatureEngineer
from src.features.technical import TechnicalFeatureExtractor
from src.features.statistical import StatisticalFeatureExtractor
from src.features.cross_asset import CrossAssetFeatureExtractor
from src.features.base import BaseFeatureExtractor


class TestTechnicalFeatureExtractor:
    """Tests para TechnicalFeatureExtractor."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        # Generar datos OHLCV realistas
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.02, 200)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 200)
        }, index=dates)
        
        # Asegurar lógica OHLC
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def test_technical_extractor_initialization(self):
        """Test de inicialización del extractor técnico."""
        extractor = TechnicalFeatureExtractor([5, 10, 20])
        
        assert extractor.name == "technical"
        assert extractor.lookback_periods == [5, 10, 20]
        assert isinstance(extractor.features, dict)
    
    def test_technical_feature_extraction(self, sample_data):
        """Test de extracción de features técnicos."""
        extractor = TechnicalFeatureExtractor([5, 10, 20])
        
        features = extractor.extract_features(sample_data)
        
        assert not features.empty
        assert len(features) == len(sample_data)
        
        # Verificar que se generaron features esperados
        expected_features = ['returns_1', 'returns_5', 'volatility_5', 'rsi_14', 'atr_14']
        for feature in expected_features:
            assert any(feature in col for col in features.columns), f"Feature {feature} no encontrado"
    
    def test_rsi_calculation(self, sample_data):
        """Test de cálculo de RSI."""
        extractor = TechnicalFeatureExtractor()
        rsi = extractor._calculate_rsi(sample_data['close'], 14)
        
        assert len(rsi) == len(sample_data)
        assert all(0 <= val <= 100 for val in rsi.dropna())
        assert not rsi.isna().all()
    
    def test_bollinger_bands_calculation(self, sample_data):
        """Test de cálculo de Bollinger Bands."""
        extractor = TechnicalFeatureExtractor()
        bb = extractor._calculate_bollinger_bands(sample_data['close'], 20, 2)
        
        assert 'bb_upper' in bb
        assert 'bb_middle' in bb
        assert 'bb_lower' in bb
        assert 'bb_position' in bb
        
        # Verificar lógica de Bollinger Bands
        assert all(bb['bb_upper'] >= bb['bb_middle'])
        assert all(bb['bb_middle'] >= bb['bb_lower'])
        assert all(0 <= val <= 1 for val in bb['bb_position'].dropna())


class TestStatisticalFeatureExtractor:
    """Tests para StatisticalFeatureExtractor."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.02, 200)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 200)
        }, index=dates)
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def test_statistical_extractor_initialization(self):
        """Test de inicialización del extractor estadístico."""
        extractor = StatisticalFeatureExtractor([20, 50])
        
        assert extractor.name == "statistical"
        assert extractor.lookback_periods == [20, 50]
    
    def test_statistical_feature_extraction(self, sample_data):
        """Test de extracción de features estadísticos."""
        extractor = StatisticalFeatureExtractor([20, 50])
        
        features = extractor.extract_features(sample_data)
        
        assert not features.empty
        assert len(features) == len(sample_data)
        
        # Verificar features estadísticos
        expected_features = ['skewness_20', 'kurtosis_20', 'autocorr_lag1_20']
        for feature in expected_features:
            assert any(feature in col for col in features.columns), f"Feature {feature} no encontrado"
    
    def test_distribution_features(self, sample_data):
        """Test de features de distribución."""
        extractor = StatisticalFeatureExtractor()
        returns = sample_data['close'].pct_change().dropna()
        
        dist_features = extractor._calculate_distribution_features(returns)
        
        assert 'skewness_20' in dist_features
        assert 'kurtosis_20' in dist_features
        assert 'jarque_bera_20' in dist_features
        
        # Verificar que los valores son razonables
        assert not dist_features['skewness_20'].isna().all()
        assert not dist_features['kurtosis_20'].isna().all()


class TestCrossAssetFeatureExtractor:
    """Tests para CrossAssetFeatureExtractor."""
    
    @pytest.fixture
    def sample_multi_data(self):
        """Datos de muestra para múltiples instrumentos."""
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        data = {}
        symbols = ['SPY', 'QQQ', 'GLD']
        
        for i, symbol in enumerate(symbols):
            np.random.seed(42 + i)
            base_price = 100 + i * 10
            returns = np.random.normal(0, 0.02, 200)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 10000, 200)
            }, index=dates)
            
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            df.attrs['symbol'] = symbol
            
            data[symbol] = df
        
        return data
    
    def test_cross_asset_extractor_initialization(self):
        """Test de inicialización del extractor cross-asset."""
        extractor = CrossAssetFeatureExtractor([20, 50])
        
        assert extractor.name == "cross_asset"
        assert extractor.lookback_periods == [20, 50]
        assert isinstance(extractor.all_data, dict)
    
    def test_cross_asset_feature_extraction(self, sample_multi_data):
        """Test de extracción de features cross-asset."""
        extractor = CrossAssetFeatureExtractor([20, 50])
        extractor.set_all_data(sample_multi_data)
        
        # Probar con un instrumento
        spy_data = sample_multi_data['SPY']
        features = extractor.extract_features(spy_data)
        
        assert not features.empty
        assert len(features) == len(spy_data)
        
        # Verificar features cross-asset
        expected_features = ['corr_QQQ_20', 'relative_carry_QQQ']
        for feature in expected_features:
            assert any(feature in col for col in features.columns), f"Feature {feature} no encontrado"
    
    def test_correlation_calculation(self, sample_multi_data):
        """Test de cálculo de correlaciones."""
        extractor = CrossAssetFeatureExtractor()
        extractor.set_all_data(sample_multi_data)
        
        spy_data = sample_multi_data['SPY']
        corr_features = extractor._calculate_correlation_features(spy_data, 'SPY')
        
        # Verificar que se calcularon correlaciones
        assert len(corr_features) > 0
        
        # Verificar que las correlaciones están en rango [-1, 1]
        for feature_name, feature_series in corr_features.items():
            if 'corr_' in feature_name:
                valid_values = feature_series.dropna()
                if len(valid_values) > 0:
                    assert all(-1 <= val <= 1 for val in valid_values), f"Correlación fuera de rango en {feature_name}"


class TestFeatureEngineer:
    """Tests para FeatureEngineer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Directorio temporal para tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Configuración de muestra."""
        return {
            'features': {
                'lookback_periods': [5, 10, 20, 50]
            },
            'capital_inicial': 1000000,
            'apalancamiento_max': 3,
            'instrumentos': {
                'indices': ['SPY', 'QQQ']
            },
            'timeframes': {'principal': '1H'},
            'zona_horaria': 'UTC',
            'ventana_historica_anos': 1,
            'costos': {'comision_bps': 2},
            'vol_objetivo_anualizada': 0.15,
            'position_sizing_max_pct': 5,
            'max_daily_loss_pct': 2,
            'max_drawdown_pct': 10
        }
    
    @pytest.fixture
    def sample_multi_data(self):
        """Datos de muestra para múltiples instrumentos."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        data = {}
        symbols = ['SPY', 'QQQ']
        
        for i, symbol in enumerate(symbols):
            np.random.seed(42 + i)
            base_price = 100 + i * 10
            returns = np.random.normal(0, 0.02, 100)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 10000, 100)
            }, index=dates)
            
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            df.attrs['symbol'] = symbol
            
            data[symbol] = df
        
        return data
    
    def test_feature_engineer_initialization(self, temp_dir, sample_config):
        """Test de inicialización del feature engineer."""
        # Crear archivo de configuración temporal
        config_file = temp_dir + "/test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        engineer = FeatureEngineer(config_file)
        
        assert engineer.config == sample_config
        assert len(engineer.feature_store.extractors) == 3  # technical, statistical, cross_asset
        assert engineer.lookback_periods == [5, 10, 20, 50]
    
    def test_feature_creation(self, temp_dir, sample_config, sample_multi_data):
        """Test de creación de features."""
        # Crear archivo de configuración temporal
        config_file = temp_dir + "/test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        engineer = FeatureEngineer(config_file)
        features = engineer.create_features(sample_multi_data)
        
        assert len(features) == len(sample_multi_data)
        
        for symbol, feature_df in features.items():
            assert not feature_df.empty
            assert len(feature_df) <= len(sample_multi_data[symbol])  # Puede ser menor por NaN
            assert len(feature_df.columns) > 10  # Debería tener muchos features
    
    def test_feature_validation(self, temp_dir, sample_config, sample_multi_data):
        """Test de validación de features."""
        config_file = temp_dir + "/test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        engineer = FeatureEngineer(config_file)
        
        # Crear features con problemas intencionalmente
        problematic_data = sample_multi_data.copy()
        for symbol in problematic_data:
            # Agregar columnas con todos NaN
            problematic_data[symbol]['bad_feature'] = np.nan
            # Agregar valores infinitos
            problematic_data[symbol]['inf_feature'] = np.inf
        
        features = engineer.create_features(problematic_data)
        
        # Verificar que los features problemáticos fueron manejados
        for symbol, feature_df in features.items():
            assert 'bad_feature' not in feature_df.columns  # Debería ser removido
            assert not np.isinf(feature_df.values).any()  # No debería haber infinitos
    
    def test_feature_importance(self, temp_dir, sample_config, sample_multi_data):
        """Test de cálculo de importancia de features."""
        config_file = temp_dir + "/test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        engineer = FeatureEngineer(config_file)
        features = engineer.create_features(sample_multi_data)
        
        # Calcular importancia
        importance = engineer.get_feature_importance(features, 'returns_1')
        
        assert len(importance) == len(features)
        
        for symbol, importance_df in importance.items():
            assert not importance_df.empty
            assert 'feature' in importance_df.columns
            assert 'correlation' in importance_df.columns
            assert 'abs_correlation' in importance_df.columns
    
    def test_save_load_features(self, temp_dir, sample_config, sample_multi_data):
        """Test de guardado y carga de features."""
        config_file = temp_dir + "/test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        engineer = FeatureEngineer(config_file)
        features = engineer.create_features(sample_multi_data)
        
        # Guardar features
        save_path = temp_dir + "/features"
        engineer.save_features(features, save_path)
        
        # Verificar que se crearon archivos
        import os
        assert os.path.exists(save_path)
        assert os.path.exists(save_path + "/features_metadata.json")
        
        # Cargar features
        loaded_features = engineer.load_features(save_path)
        
        assert len(loaded_features) == len(features)
        
        for symbol in features:
            assert symbol in loaded_features
            assert len(loaded_features[symbol].columns) == len(features[symbol].columns)


class TestBaseFeatureExtractor:
    """Tests para BaseFeatureExtractor."""
    
    def test_base_extractor_initialization(self):
        """Test de inicialización del extractor base."""
        class TestExtractor(BaseFeatureExtractor):
            def extract_features(self, data):
                return pd.DataFrame({'test': [1, 2, 3]})
        
        extractor = TestExtractor("test", [5, 10])
        
        assert extractor.name == "test"
        assert extractor.lookback_periods == [5, 10]
        assert isinstance(extractor.features, dict)
    
    def test_data_validation(self):
        """Test de validación de datos."""
        class TestExtractor(BaseFeatureExtractor):
            def extract_features(self, data):
                return pd.DataFrame()
        
        extractor = TestExtractor("test")
        
        # Datos válidos
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        assert extractor.validate_data(valid_data) == True
        
        # Datos inválidos (faltan columnas)
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [102, 103, 104]
        })
        
        assert extractor.validate_data(invalid_data) == False
        
        # Datos vacíos
        empty_data = pd.DataFrame()
        assert extractor.validate_data(empty_data) == False
    
    def test_safe_rolling(self):
        """Test de rolling seguro."""
        class TestExtractor(BaseFeatureExtractor):
            def extract_features(self, data):
                return pd.DataFrame()
        
        extractor = TestExtractor("test")
        
        # Serie normal
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = extractor._safe_rolling(series, 5, lambda x: x.mean())
        
        assert len(result) == len(series)
        assert not result.iloc[:4].isna().all()  # Primeros valores pueden ser NaN
        assert not result.iloc[4:].isna().all()  # Valores posteriores no deberían ser NaN
        
        # Serie muy corta
        short_series = pd.Series([1, 2])
        result_short = extractor._safe_rolling(short_series, 5, lambda x: x.mean())
        
        assert len(result_short) == len(short_series)
        assert result_short.isna().all()  # Debería ser todo NaN


if __name__ == "__main__":
    pytest.main([__file__])

