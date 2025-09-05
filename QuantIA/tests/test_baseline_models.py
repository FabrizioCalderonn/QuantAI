"""
Tests para modelos baseline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.models.baseline import MomentumModel, MeanReversionModel, HybridModel
from src.models.trainer import ModelTrainer
from src.models.base import BaseModel, TradingSignal


class TestMomentumModel:
    """Tests para MomentumModel."""
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Crear features realistas
        np.random.seed(42)
        features = pd.DataFrame({
            'returns_1': np.random.normal(0, 0.02, 100),
            'returns_5': np.random.normal(0, 0.05, 100),
            'returns_10': np.random.normal(0, 0.08, 100),
            'returns_20': np.random.normal(0, 0.12, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'rsi_21': np.random.uniform(20, 80, 100),
            'rsi_50': np.random.uniform(20, 80, 100),
            'price_sma_ratio_20': np.random.uniform(0.95, 1.05, 100),
            'price_sma_ratio_50': np.random.uniform(0.95, 1.05, 100),
            'volatility_20': np.random.uniform(0.01, 0.05, 100),
            'volatility_50': np.random.uniform(0.01, 0.05, 100),
            'macd': np.random.normal(0, 0.01, 100),
            'macd_signal': np.random.normal(0, 0.01, 100),
            'macd_histogram': np.random.normal(0, 0.005, 100)
        }, index=dates)
        
        return features
    
    @pytest.fixture
    def sample_target(self):
        """Target de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Crear target con señales
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        # Crear señales basadas en retornos
        signals = pd.Series(0, index=dates)
        signals[returns > 0.01] = 1  # Long
        signals[returns < -0.01] = -1  # Short
        
        return signals
    
    def test_momentum_model_initialization(self):
        """Test de inicialización del modelo momentum."""
        model = MomentumModel()
        
        assert model.name == "momentum"
        assert model.short_period == 5
        assert model.medium_period == 20
        assert model.long_period == 50
        assert model.rsi_period == 14
        assert not model.is_fitted
    
    def test_momentum_model_fit(self, sample_features, sample_target):
        """Test de entrenamiento del modelo momentum."""
        model = MomentumModel()
        
        # Entrenar modelo
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert 'hit_ratio' in model.performance_metrics
        assert 'profit_factor' in model.performance_metrics
        assert 'total_signals' in model.performance_metrics
    
    def test_momentum_model_predict(self, sample_features, sample_target):
        """Test de predicción del modelo momentum."""
        model = MomentumModel()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
        assert isinstance(predictions, np.ndarray)
    
    def test_momentum_model_predict_proba(self, sample_features, sample_target):
        """Test de probabilidades del modelo momentum."""
        model = MomentumModel()
        model.fit(sample_features, sample_target)
        
        probabilities = model.predict_proba(sample_features)
        
        assert len(probabilities) == len(sample_features)
        assert probabilities.shape[1] == 3  # [Short, Neutral, Long]
        assert all(abs(prob.sum() - 1.0) < 1e-6 for prob in probabilities)
    
    def test_momentum_signal_calculation(self, sample_features):
        """Test de cálculo de señales momentum."""
        model = MomentumModel()
        
        # Test con features que deberían generar señal long
        test_features = sample_features.iloc[0].copy()
        test_features['returns_5'] = 0.02  # 2% return
        test_features['rsi_14'] = 25  # Oversold
        test_features['price_sma_ratio_20'] = 1.03  # Above SMA
        test_features['volatility_20'] = 0.015  # Low volatility
        
        signal = model._calculate_momentum_signal(test_features)
        assert signal in [-1, 0, 1]
    
    def test_momentum_model_save_load(self, sample_features, sample_target, temp_dir):
        """Test de guardado y carga del modelo momentum."""
        model = MomentumModel()
        model.fit(sample_features, sample_target)
        
        # Guardar modelo
        model_path = temp_dir + "/momentum_model.pkl"
        model.save_model(model_path)
        
        # Cargar modelo
        loaded_model = MomentumModel()
        loaded_model.load_model(model_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.name == "momentum"
        assert loaded_model.performance_metrics == model.performance_metrics


class TestMeanReversionModel:
    """Tests para MeanReversionModel."""
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        features = pd.DataFrame({
            'bb_position_20_2': np.random.uniform(0, 1, 100),
            'bb_position_50_2': np.random.uniform(0, 1, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'rsi_21': np.random.uniform(20, 80, 100),
            'rsi_50': np.random.uniform(20, 80, 100),
            'williams_r_14': np.random.uniform(-100, 0, 100),
            'williams_r_21': np.random.uniform(-100, 0, 100),
            'williams_r_50': np.random.uniform(-100, 0, 100),
            'stoch_k_14_3': np.random.uniform(0, 100, 100),
            'stoch_k_14_5': np.random.uniform(0, 100, 100),
            'stoch_d_14_3': np.random.uniform(0, 100, 100),
            'stoch_d_14_5': np.random.uniform(0, 100, 100),
            'zscore_20': np.random.uniform(-3, 3, 100),
            'zscore_50': np.random.uniform(-3, 3, 100)
        }, index=dates)
        
        return features
    
    @pytest.fixture
    def sample_target(self):
        """Target de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        signals = pd.Series(0, index=dates)
        signals[returns > 0.01] = 1
        signals[returns < -0.01] = -1
        
        return signals
    
    def test_mean_reversion_model_initialization(self):
        """Test de inicialización del modelo mean reversion."""
        model = MeanReversionModel()
        
        assert model.name == "mean_reversion"
        assert model.bb_period == 20
        assert model.bb_std == 2
        assert model.rsi_period == 14
        assert not model.is_fitted
    
    def test_mean_reversion_model_fit(self, sample_features, sample_target):
        """Test de entrenamiento del modelo mean reversion."""
        model = MeanReversionModel()
        
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert 'hit_ratio' in model.performance_metrics
        assert 'profit_factor' in model.performance_metrics
    
    def test_mean_reversion_model_predict(self, sample_features, sample_target):
        """Test de predicción del modelo mean reversion."""
        model = MeanReversionModel()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
    
    def test_mean_reversion_signal_calculation(self, sample_features):
        """Test de cálculo de señales mean reversion."""
        model = MeanReversionModel()
        
        # Test con features que deberían generar señal long (oversold)
        test_features = sample_features.iloc[0].copy()
        test_features['bb_position_20_2'] = 0.05  # Near lower band
        test_features['rsi_14'] = 25  # Oversold
        test_features['williams_r_14'] = -85  # Oversold
        test_features['stoch_k_14_3'] = 15  # Oversold
        test_features['zscore_20'] = -2.5  # Below mean
        
        signal = model._calculate_mean_reversion_signal(test_features)
        assert signal in [-1, 0, 1]


class TestHybridModel:
    """Tests para HybridModel."""
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        features = pd.DataFrame({
            # Momentum features
            'returns_1': np.random.normal(0, 0.02, 100),
            'returns_5': np.random.normal(0, 0.05, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'price_sma_ratio_20': np.random.uniform(0.95, 1.05, 100),
            'volatility_20': np.random.uniform(0.01, 0.05, 100),
            'macd': np.random.normal(0, 0.01, 100),
            'macd_signal': np.random.normal(0, 0.01, 100),
            # Mean reversion features
            'bb_position_20_2': np.random.uniform(0, 1, 100),
            'williams_r_14': np.random.uniform(-100, 0, 100),
            'stoch_k_14_3': np.random.uniform(0, 100, 100),
            'zscore_20': np.random.uniform(-3, 3, 100),
            # Regime detection
            'volatility_50': np.random.uniform(0.01, 0.05, 100),
            'atr_14': np.random.uniform(0.005, 0.02, 100)
        }, index=dates)
        
        return features
    
    @pytest.fixture
    def sample_target(self):
        """Target de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        signals = pd.Series(0, index=dates)
        signals[returns > 0.01] = 1
        signals[returns < -0.01] = -1
        
        return signals
    
    def test_hybrid_model_initialization(self):
        """Test de inicialización del modelo híbrido."""
        model = HybridModel()
        
        assert model.name == "hybrid"
        assert model.momentum_weight == 0.6
        assert model.mean_reversion_weight == 0.4
        assert hasattr(model, 'momentum_model')
        assert hasattr(model, 'mean_reversion_model')
        assert not model.is_fitted
    
    def test_hybrid_model_fit(self, sample_features, sample_target):
        """Test de entrenamiento del modelo híbrido."""
        model = HybridModel()
        
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert model.momentum_model.is_fitted
        assert model.mean_reversion_model.is_fitted
        assert 'hit_ratio' in model.performance_metrics
    
    def test_hybrid_model_predict(self, sample_features, sample_target):
        """Test de predicción del modelo híbrido."""
        model = HybridModel()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
    
    def test_regime_detection(self, sample_features):
        """Test de detección de regímenes."""
        model = HybridModel()
        
        # Test trending regime
        test_features = sample_features.iloc[0].copy()
        test_features['volatility_50'] = 0.05  # High volatility
        test_features['atr_14'] = 0.025  # High ATR
        
        regime = model._detect_regime(test_features)
        assert regime in ['trending', 'ranging', 'mixed']
        
        # Test ranging regime
        test_features['volatility_50'] = 0.005  # Low volatility
        test_features['atr_14'] = 0.002  # Low ATR
        
        regime = model._detect_regime(test_features)
        assert regime in ['trending', 'ranging', 'mixed']


class TestModelTrainer:
    """Tests para ModelTrainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Directorio temporal para tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        np.random.seed(42)
        features = pd.DataFrame({
            'returns_1': np.random.normal(0, 0.02, 200),
            'returns_5': np.random.normal(0, 0.05, 200),
            'rsi_14': np.random.uniform(20, 80, 200),
            'price_sma_ratio_20': np.random.uniform(0.95, 1.05, 200),
            'volatility_20': np.random.uniform(0.01, 0.05, 200),
            'bb_position_20_2': np.random.uniform(0, 1, 200),
            'williams_r_14': np.random.uniform(-100, 0, 200),
            'stoch_k_14_3': np.random.uniform(0, 100, 200),
            'zscore_20': np.random.uniform(-3, 3, 200)
        }, index=dates)
        
        return features
    
    def test_trainer_initialization(self):
        """Test de inicialización del trainer."""
        trainer = ModelTrainer()
        
        assert isinstance(trainer.models, dict)
        assert isinstance(trainer.training_results, dict)
        assert isinstance(trainer.validation_results, dict)
    
    def test_create_baseline_models(self):
        """Test de creación de modelos baseline."""
        trainer = ModelTrainer()
        models = trainer.create_baseline_models()
        
        assert 'momentum' in models
        assert 'mean_reversion' in models
        assert 'hybrid' in models
        
        assert isinstance(models['momentum'], MomentumModel)
        assert isinstance(models['mean_reversion'], MeanReversionModel)
        assert isinstance(models['hybrid'], HybridModel)
    
    def test_prepare_training_data(self, sample_features):
        """Test de preparación de datos de entrenamiento."""
        trainer = ModelTrainer()
        
        # Simular features de múltiples instrumentos
        features_dict = {
            'SPY': sample_features,
            'QQQ': sample_features.copy()
        }
        
        X, y = trainer.prepare_training_data(features_dict, 'returns_1')
        
        assert len(X) == len(sample_features) * 2  # 2 instrumentos
        assert len(y) == len(sample_features) * 2
        assert len(X.columns) == len(sample_features.columns) - 1  # Sin target
    
    def test_create_target_signals(self):
        """Test de creación de señales target."""
        trainer = ModelTrainer()
        
        # Crear retornos de prueba
        returns = pd.Series([0.02, -0.02, 0.005, -0.005, 0.0])
        
        signals = trainer.create_target_signals(returns)
        
        expected = pd.Series([1, -1, 0, 0, 0])  # 1% threshold
        pd.testing.assert_series_equal(signals, expected)
    
    def test_calculate_metrics(self):
        """Test de cálculo de métricas."""
        trainer = ModelTrainer()
        
        y_true = pd.Series([1, -1, 0, 1, -1])
        y_pred = np.array([1, -1, 0, 0, -1])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert all(0 <= val <= 1 for val in metrics.values())
    
    def test_calculate_trading_metrics(self):
        """Test de cálculo de métricas de trading."""
        trainer = ModelTrainer()
        
        signals = pd.Series([1, -1, 0, 1, -1])
        predictions = np.array([1, -1, 0, 1, -1])
        returns = pd.Series([0.02, -0.01, 0.0, 0.015, -0.02])
        
        metrics = trainer._calculate_trading_metrics(signals, predictions, returns)
        
        assert 'hit_ratio' in metrics
        assert 'profit_factor' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_return' in metrics
    
    def test_save_load_models(self, temp_dir, sample_features):
        """Test de guardado y carga de modelos."""
        trainer = ModelTrainer()
        
        # Crear y entrenar modelos
        models = trainer.create_baseline_models()
        
        # Simular features con target
        features_with_target = sample_features.copy()
        features_with_target['returns_1'] = np.random.normal(0, 0.02, len(sample_features))
        
        features_dict = {'SPY': features_with_target}
        
        # Entrenar modelos
        trained_models = trainer.train_models(features_dict)
        
        # Guardar modelos
        save_path = temp_dir + "/models"
        trainer.save_models(save_path)
        
        # Cargar modelos
        loaded_models = trainer.load_models(save_path)
        
        assert len(loaded_models) == len(trained_models)
        assert all(model.is_fitted for model in loaded_models.values())


class TestTradingSignal:
    """Tests para TradingSignal."""
    
    def test_trading_signal_initialization(self):
        """Test de inicialización de señal de trading."""
        timestamp = datetime.now()
        signal = TradingSignal('SPY', timestamp, 1, 0.8)
        
        assert signal.symbol == 'SPY'
        assert signal.timestamp == timestamp
        assert signal.signal == 1
        assert signal.confidence == 0.8
        assert signal.metadata == {}
    
    def test_trading_signal_validation(self):
        """Test de validación de señal de trading."""
        timestamp = datetime.now()
        
        # Señal válida
        signal = TradingSignal('SPY', timestamp, 1, 0.8)
        assert signal.signal == 1
        
        # Señal inválida
        with pytest.raises(ValueError):
            TradingSignal('SPY', timestamp, 2, 0.8)  # Señal inválida
        
        with pytest.raises(ValueError):
            TradingSignal('SPY', timestamp, 1, 1.5)  # Confianza inválida
    
    def test_trading_signal_to_dict(self):
        """Test de conversión a diccionario."""
        timestamp = datetime.now()
        metadata = {'model': 'momentum', 'features': ['rsi', 'macd']}
        
        signal = TradingSignal('SPY', timestamp, 1, 0.8, metadata)
        signal_dict = signal.to_dict()
        
        assert signal_dict['symbol'] == 'SPY'
        assert signal_dict['signal'] == 1
        assert signal_dict['confidence'] == 0.8
        assert signal_dict['metadata'] == metadata
        assert 'timestamp' in signal_dict
    
    def test_trading_signal_str(self):
        """Test de representación string."""
        timestamp = datetime.now()
        signal = TradingSignal('SPY', timestamp, 1, 0.8)
        
        signal_str = str(signal)
        
        assert 'SPY' in signal_str
        assert 'LONG' in signal_str
        assert '0.80' in signal_str


if __name__ == "__main__":
    pytest.main([__file__])

