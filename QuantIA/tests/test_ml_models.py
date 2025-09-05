"""
Tests para modelos ML.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.models.ml_models import LassoModel, XGBoostModel, RandomForestModel, MLEnsemble
from src.models.ml_trainer import MLModelTrainer
from src.models.base import BaseModel


class TestLassoModel:
    """Tests para LassoModel."""
    
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
            'macd_histogram': np.random.normal(0, 0.005, 100),
            'bb_position_20_2': np.random.uniform(0, 1, 100),
            'williams_r_14': np.random.uniform(-100, 0, 100),
            'stoch_k_14_3': np.random.uniform(0, 100, 100),
            'zscore_20': np.random.uniform(-3, 3, 100)
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
    
    def test_lasso_model_initialization(self):
        """Test de inicialización del modelo Lasso."""
        model = LassoModel()
        
        assert model.name == "lasso"
        assert model.alpha == 0.01
        assert model.max_iter == 1000
        assert model.feature_selection == True
        assert model.n_features == 50
        assert not model.is_fitted
    
    def test_lasso_model_fit(self, sample_features, sample_target):
        """Test de entrenamiento del modelo Lasso."""
        model = LassoModel()
        
        # Entrenar modelo
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert 'hit_ratio' in model.performance_metrics
        assert 'profit_factor' in model.performance_metrics
        assert 'n_features_used' in model.performance_metrics
        assert model.selected_features is not None
    
    def test_lasso_model_predict(self, sample_features, sample_target):
        """Test de predicción del modelo Lasso."""
        model = LassoModel()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
        assert isinstance(predictions, np.ndarray)
    
    def test_lasso_model_predict_proba(self, sample_features, sample_target):
        """Test de probabilidades del modelo Lasso."""
        model = LassoModel()
        model.fit(sample_features, sample_target)
        
        probabilities = model.predict_proba(sample_features)
        
        assert len(probabilities) == len(sample_features)
        assert probabilities.shape[1] == 3  # [Short, Neutral, Long]
        assert all(abs(prob.sum() - 1.0) < 1e-6 for prob in probabilities)
    
    def test_lasso_feature_selection(self, sample_features, sample_target):
        """Test de selección de features."""
        model = LassoModel({'n_features': 10})
        model.fit(sample_features, sample_target)
        
        assert len(model.selected_features) <= 10
        assert model.feature_importance is not None
        assert len(model.feature_importance) == len(model.selected_features)
    
    def test_lasso_model_save_load(self, sample_features, sample_target, temp_dir):
        """Test de guardado y carga del modelo Lasso."""
        model = LassoModel()
        model.fit(sample_features, sample_target)
        
        # Guardar modelo
        model_path = temp_dir + "/lasso_model.pkl"
        model.save_model(model_path)
        
        # Cargar modelo
        loaded_model = LassoModel()
        loaded_model.load_model(model_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.name == "lasso"
        assert loaded_model.selected_features == model.selected_features


class TestXGBoostModel:
    """Tests para XGBoostModel."""
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        features = pd.DataFrame({
            'returns_1': np.random.normal(0, 0.02, 100),
            'returns_5': np.random.normal(0, 0.05, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'price_sma_ratio_20': np.random.uniform(0.95, 1.05, 100),
            'volatility_20': np.random.uniform(0.01, 0.05, 100),
            'macd': np.random.normal(0, 0.01, 100),
            'macd_signal': np.random.normal(0, 0.01, 100),
            'bb_position_20_2': np.random.uniform(0, 1, 100),
            'williams_r_14': np.random.uniform(-100, 0, 100),
            'stoch_k_14_3': np.random.uniform(0, 100, 100),
            'zscore_20': np.random.uniform(-3, 3, 100)
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
    
    def test_xgboost_model_initialization(self):
        """Test de inicialización del modelo XGBoost."""
        model = XGBoostModel()
        
        assert model.name == "xgboost"
        assert model.n_estimators == 100
        assert model.max_depth == 6
        assert model.learning_rate == 0.1
        assert not model.is_fitted
    
    def test_xgboost_model_fit(self, sample_features, sample_target):
        """Test de entrenamiento del modelo XGBoost."""
        model = XGBoostModel()
        
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert 'hit_ratio' in model.performance_metrics
        assert 'profit_factor' in model.performance_metrics
        assert model.selected_features is not None
    
    def test_xgboost_model_predict(self, sample_features, sample_target):
        """Test de predicción del modelo XGBoost."""
        model = XGBoostModel()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
    
    def test_xgboost_model_predict_proba(self, sample_features, sample_target):
        """Test de probabilidades del modelo XGBoost."""
        model = XGBoostModel()
        model.fit(sample_features, sample_target)
        
        probabilities = model.predict_proba(sample_features)
        
        assert len(probabilities) == len(sample_features)
        assert probabilities.shape[1] == 3  # [Short, Neutral, Long]
        assert all(abs(prob.sum() - 1.0) < 1e-6 for prob in probabilities)
    
    def test_xgboost_feature_importance(self, sample_features, sample_target):
        """Test de importancia de features."""
        model = XGBoostModel()
        model.fit(sample_features, sample_target)
        
        assert model.feature_importance is not None
        assert len(model.feature_importance) == len(model.selected_features)
        assert all(imp >= 0 for imp in model.feature_importance.values())


class TestRandomForestModel:
    """Tests para RandomForestModel."""
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        features = pd.DataFrame({
            'returns_1': np.random.normal(0, 0.02, 100),
            'returns_5': np.random.normal(0, 0.05, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'price_sma_ratio_20': np.random.uniform(0.95, 1.05, 100),
            'volatility_20': np.random.uniform(0.01, 0.05, 100),
            'macd': np.random.normal(0, 0.01, 100),
            'macd_signal': np.random.normal(0, 0.01, 100),
            'bb_position_20_2': np.random.uniform(0, 1, 100),
            'williams_r_14': np.random.uniform(-100, 0, 100),
            'stoch_k_14_3': np.random.uniform(0, 100, 100),
            'zscore_20': np.random.uniform(-3, 3, 100)
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
    
    def test_random_forest_model_initialization(self):
        """Test de inicialización del modelo Random Forest."""
        model = RandomForestModel()
        
        assert model.name == "random_forest"
        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.min_samples_split == 5
        assert not model.is_fitted
    
    def test_random_forest_model_fit(self, sample_features, sample_target):
        """Test de entrenamiento del modelo Random Forest."""
        model = RandomForestModel()
        
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert 'hit_ratio' in model.performance_metrics
        assert 'profit_factor' in model.performance_metrics
        assert model.selected_features is not None
    
    def test_random_forest_model_predict(self, sample_features, sample_target):
        """Test de predicción del modelo Random Forest."""
        model = RandomForestModel()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
    
    def test_random_forest_model_predict_proba(self, sample_features, sample_target):
        """Test de probabilidades del modelo Random Forest."""
        model = RandomForestModel()
        model.fit(sample_features, sample_target)
        
        probabilities = model.predict_proba(sample_features)
        
        assert len(probabilities) == len(sample_features)
        assert probabilities.shape[1] == 3  # [Short, Neutral, Long]
        assert all(abs(prob.sum() - 1.0) < 1e-6 for prob in probabilities)
    
    def test_random_forest_feature_importance(self, sample_features, sample_target):
        """Test de importancia de features."""
        model = RandomForestModel()
        model.fit(sample_features, sample_target)
        
        assert model.feature_importance is not None
        assert len(model.feature_importance) == len(model.selected_features)
        assert all(imp >= 0 for imp in model.feature_importance.values())


class TestMLEnsemble:
    """Tests para MLEnsemble."""
    
    @pytest.fixture
    def sample_features(self):
        """Features de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        np.random.seed(42)
        features = pd.DataFrame({
            'returns_1': np.random.normal(0, 0.02, 100),
            'returns_5': np.random.normal(0, 0.05, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'price_sma_ratio_20': np.random.uniform(0.95, 1.05, 100),
            'volatility_20': np.random.uniform(0.01, 0.05, 100),
            'macd': np.random.normal(0, 0.01, 100),
            'macd_signal': np.random.normal(0, 0.01, 100),
            'bb_position_20_2': np.random.uniform(0, 1, 100),
            'williams_r_14': np.random.uniform(-100, 0, 100),
            'stoch_k_14_3': np.random.uniform(0, 100, 100),
            'zscore_20': np.random.uniform(-3, 3, 100)
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
    
    def test_ml_ensemble_initialization(self):
        """Test de inicialización del ensemble ML."""
        model = MLEnsemble()
        
        assert model.name == "ml_ensemble"
        assert 'lasso' in model.models
        assert 'xgboost' in model.models
        assert 'random_forest' in model.models
        assert model.voting_method == 'soft'
        assert not model.is_fitted
    
    def test_ml_ensemble_fit(self, sample_features, sample_target):
        """Test de entrenamiento del ensemble ML."""
        model = MLEnsemble()
        
        model.fit(sample_features, sample_target)
        
        assert model.is_fitted
        assert all(submodel.is_fitted for submodel in model.models.values())
        assert 'hit_ratio' in model.performance_metrics
        assert 'n_models' in model.performance_metrics
        assert model.weights is not None
    
    def test_ml_ensemble_predict(self, sample_features, sample_target):
        """Test de predicción del ensemble ML."""
        model = MLEnsemble()
        model.fit(sample_features, sample_target)
        
        predictions = model.predict(sample_features)
        
        assert len(predictions) == len(sample_features)
        assert all(pred in [-1, 0, 1] for pred in predictions)
    
    def test_ml_ensemble_predict_proba(self, sample_features, sample_target):
        """Test de probabilidades del ensemble ML."""
        model = MLEnsemble()
        model.fit(sample_features, sample_target)
        
        probabilities = model.predict_proba(sample_features)
        
        assert len(probabilities) == len(sample_features)
        assert probabilities.shape[1] == 3  # [Short, Neutral, Long]
        assert all(abs(prob.sum() - 1.0) < 1e-6 for prob in probabilities)
    
    def test_ml_ensemble_voting_methods(self, sample_features, sample_target):
        """Test de métodos de votación."""
        # Test hard voting
        model_hard = MLEnsemble({'voting_method': 'hard'})
        model_hard.fit(sample_features, sample_target)
        predictions_hard = model_hard.predict(sample_features)
        
        # Test soft voting
        model_soft = MLEnsemble({'voting_method': 'soft'})
        model_soft.fit(sample_features, sample_target)
        predictions_soft = model_soft.predict(sample_features)
        
        assert len(predictions_hard) == len(predictions_soft)
        assert all(pred in [-1, 0, 1] for pred in predictions_hard)
        assert all(pred in [-1, 0, 1] for pred in predictions_soft)
    
    def test_ml_ensemble_feature_importance(self, sample_features, sample_target):
        """Test de importancia de features del ensemble."""
        model = MLEnsemble()
        model.fit(sample_features, sample_target)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) > 0
        assert all(imp >= 0 for imp in importance.values())


class TestMLModelTrainer:
    """Tests para MLModelTrainer."""
    
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
            'macd': np.random.normal(0, 0.01, 200),
            'macd_signal': np.random.normal(0, 0.01, 200),
            'bb_position_20_2': np.random.uniform(0, 1, 200),
            'williams_r_14': np.random.uniform(-100, 0, 200),
            'stoch_k_14_3': np.random.uniform(0, 100, 200),
            'zscore_20': np.random.uniform(-3, 3, 200)
        }, index=dates)
        
        return features
    
    def test_ml_trainer_initialization(self):
        """Test de inicialización del trainer ML."""
        trainer = MLModelTrainer()
        
        assert isinstance(trainer.models, dict)
        assert isinstance(trainer.training_results, dict)
        assert isinstance(trainer.validation_results, dict)
        assert isinstance(trainer.hyperparameter_results, dict)
    
    def test_create_ml_models(self):
        """Test de creación de modelos ML."""
        trainer = MLModelTrainer()
        models = trainer.create_ml_models()
        
        assert 'lasso' in models
        assert 'xgboost' in models
        assert 'random_forest' in models
        assert 'ml_ensemble' in models
        
        assert isinstance(models['lasso'], LassoModel)
        assert isinstance(models['xgboost'], XGBoostModel)
        assert isinstance(models['random_forest'], RandomForestModel)
        assert isinstance(models['ml_ensemble'], MLEnsemble)
    
    def test_prepare_training_data(self, sample_features):
        """Test de preparación de datos de entrenamiento ML."""
        trainer = MLModelTrainer()
        
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
        trainer = MLModelTrainer()
        
        # Crear retornos de prueba
        returns = pd.Series([0.02, -0.02, 0.005, -0.005, 0.0])
        
        signals = trainer.create_target_signals(returns)
        
        expected = pd.Series([1, -1, 0, 0, 0])  # 1% threshold
        pd.testing.assert_series_equal(signals, expected)
    
    def test_calculate_ml_metrics(self):
        """Test de cálculo de métricas ML."""
        trainer = MLModelTrainer()
        
        y_true = pd.Series([1, -1, 0, 1, -1])
        y_pred = np.array([1, -1, 0, 0, -1])
        returns = pd.Series([0.02, -0.01, 0.0, 0.015, -0.02])
        
        metrics = trainer._calculate_ml_metrics(y_true, y_pred, returns)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'hit_ratio' in metrics
        assert 'profit_factor' in metrics
        assert all(0 <= val <= 1 for val in [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])
    
    def test_calculate_trading_metrics(self):
        """Test de cálculo de métricas de trading."""
        trainer = MLModelTrainer()
        
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
        """Test de guardado y carga de modelos ML."""
        trainer = MLModelTrainer()
        
        # Crear y entrenar modelos
        models = trainer.create_ml_models()
        
        # Simular features con target
        features_with_target = sample_features.copy()
        features_with_target['returns_1'] = np.random.normal(0, 0.02, len(sample_features))
        
        features_dict = {'SPY': features_with_target}
        
        # Entrenar modelos
        trained_models = trainer.train_models(features_dict, optimize_hyperparams=False)
        
        # Guardar modelos
        save_path = temp_dir + "/ml_models"
        trainer.save_models(save_path)
        
        # Cargar modelos
        loaded_models = trainer.load_models(save_path)
        
        assert len(loaded_models) == len(trained_models)
        assert all(model.is_fitted for model in loaded_models.values())


if __name__ == "__main__":
    pytest.main([__file__])

