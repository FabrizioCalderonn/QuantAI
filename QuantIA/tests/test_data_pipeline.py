"""
Tests para el pipeline de datos.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.data.pipeline import DataPipeline
from src.utils.config import load_config, validate_config


class TestDataPipeline:
    """Tests para la clase DataPipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Crea directorio temporal para tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Configuración de muestra para tests."""
        return {
            'capital_inicial': 1000000,
            'apalancamiento_max': 3,
            'instrumentos': {
                'indices': ['SPY', 'QQQ'],
                'fx_majors': ['EURUSD']
            },
            'timeframes': {
                'principal': '1H'
            },
            'zona_horaria': 'UTC',
            'ventana_historica_anos': 1,
            'costos': {
                'comision_bps': 2
            },
            'vol_objetivo_anualizada': 0.15,
            'position_sizing_max_pct': 5,
            'max_daily_loss_pct': 2,
            'max_drawdown_pct': 10
        }
    
    def test_pipeline_initialization(self, temp_dir, sample_config):
        """Test de inicialización del pipeline."""
        # Crear archivo de configuración temporal
        config_file = Path(temp_dir) / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Inicializar pipeline
        pipeline = DataPipeline(str(config_file))
        
        assert pipeline.config == sample_config
        assert pipeline.data_dir.name == "data"
        assert pipeline.raw_dir.name == "raw"
        assert pipeline.processed_dir.name == "processed"
    
    def test_data_validation(self):
        """Test de validación de datos."""
        # Crear datos de prueba
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Datos válidos
        valid_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Asegurar lógica OHLC
        valid_data['high'] = valid_data[['open', 'high', 'close']].max(axis=1)
        valid_data['low'] = valid_data[['open', 'low', 'close']].min(axis=1)
        
        pipeline = DataPipeline()
        validated_data = pipeline._validate_data(valid_data, 'TEST')
        
        assert len(validated_data) == 100
        assert all(validated_data['high'] >= validated_data['low'])
        assert all(validated_data['high'] >= validated_data['open'])
        assert all(validated_data['high'] >= validated_data['close'])
        assert all(validated_data['low'] <= validated_data['open'])
        assert all(validated_data['low'] <= validated_data['close'])
    
    def test_data_validation_invalid_ohlc(self):
        """Test de validación con datos OHLC inválidos."""
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        
        # Datos inválidos (high < low)
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 104, 103, 102, 101, 100, 99, 98, 97, 96],  # high < low
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000] * 10
        }, index=dates)
        
        pipeline = DataPipeline()
        validated_data = pipeline._validate_data(invalid_data, 'TEST')
        
        # Debería remover todos los registros inválidos
        assert len(validated_data) == 0
    
    def test_atr_calculation(self):
        """Test de cálculo de ATR."""
        dates = pd.date_range('2023-01-01', periods=20, freq='H')
        
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 20),
            'high': np.random.uniform(110, 120, 20),
            'low': np.random.uniform(90, 100, 20),
            'close': np.random.uniform(100, 110, 20),
            'volume': np.random.uniform(1000, 10000, 20)
        }, index=dates)
        
        pipeline = DataPipeline()
        atr = pipeline._calculate_atr(data, 14)
        
        assert len(atr) == 20
        assert not atr.isna().all()  # Al menos algunos valores no deberían ser NaN
        assert all(atr >= 0)  # ATR no puede ser negativo
    
    def test_rsi_calculation(self):
        """Test de cálculo de RSI."""
        # Crear serie de precios con tendencia
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101] * 5)
        
        pipeline = DataPipeline()
        rsi = pipeline._calculate_rsi(prices, 14)
        
        assert len(rsi) == len(prices)
        assert all(0 <= rsi.dropna() <= 100)  # RSI debe estar entre 0 y 100
    
    def test_data_hash_calculation(self):
        """Test de cálculo de hash de datos."""
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000] * 10
        }, index=dates)
        
        pipeline = DataPipeline()
        hash1 = pipeline._calculate_data_hash(data)
        hash2 = pipeline._calculate_data_hash(data)
        
        # Mismos datos deberían producir mismo hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Datos diferentes deberían producir hash diferente
        data_modified = data.copy()
        data_modified.iloc[0, 0] = 999
        hash3 = pipeline._calculate_data_hash(data_modified)
        
        assert hash1 != hash3


class TestConfigUtils:
    """Tests para utilidades de configuración."""
    
    def test_load_config(self, temp_dir):
        """Test de carga de configuración."""
        config_data = {
            'test_key': 'test_value',
            'numeric_key': 123,
            'nested': {
                'inner_key': 'inner_value'
            }
        }
        
        config_file = Path(temp_dir) / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config == config_data
    
    def test_validate_config_valid(self, sample_config):
        """Test de validación con configuración válida."""
        # sample_config fixture ya está definido arriba
        assert validate_config(sample_config) == True
    
    def test_validate_config_missing_fields(self):
        """Test de validación con campos faltantes."""
        invalid_config = {
            'capital_inicial': 1000000,
            # Faltan otros campos requeridos
        }
        
        with pytest.raises(ValueError, match="Campos faltantes"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_numeric(self):
        """Test de validación con valores numéricos inválidos."""
        invalid_config = {
            'capital_inicial': -1000000,  # Negativo
            'apalancamiento_max': 3,
            'instrumentos': {'test': ['SPY']},
            'timeframes': {'test': '1H'},
            'zona_horaria': 'UTC',
            'ventana_historica_anos': 1,
            'costos': {'test': 1},
            'vol_objetivo_anualizada': 0.15,
            'position_sizing_max_pct': 5,
            'max_daily_loss_pct': 2,
            'max_drawdown_pct': 10
        }
        
        with pytest.raises(ValueError, match="debe ser un número positivo"):
            validate_config(invalid_config)
    
    def test_get_config_value(self):
        """Test de obtención de valores de configuración."""
        config = {
            'level1': {
                'level2': {
                    'value': 'test'
                }
            },
            'simple_value': 123
        }
        
        from src.utils.config import get_config_value
        
        assert get_config_value(config, 'level1.level2.value') == 'test'
        assert get_config_value(config, 'simple_value') == 123
        assert get_config_value(config, 'nonexistent', 'default') == 'default'


if __name__ == "__main__":
    pytest.main([__file__])

