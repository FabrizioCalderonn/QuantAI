"""
Tests para sistema de explainability.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import os

from src.explainability.shap_explainer import SHAPExplainer, SHAPExplanation, ExplanationType, FeatureImportance
from src.explainability.audit_logger import AuditLogger, EventType, LogLevel, AuditEvent, ModelDecision, TradeDecision
from src.explainability.explainability_manager import ExplainabilityManager


class TestSHAPExplanation:
    """Tests para SHAPExplanation."""
    
    def test_shap_explanation_creation(self):
        """Test de creación de SHAPExplanation."""
        explanation = SHAPExplanation(
            explanation_type=ExplanationType.GLOBAL,
            values=np.array([[0.1, 0.2, 0.3]]),
            base_values=0.5,
            data=np.array([[1.0, 2.0, 3.0]]),
            feature_names=['feature1', 'feature2', 'feature3'],
            timestamp=datetime.now(),
            model_name="TestModel",
            metadata={'test': 'value'}
        )
        
        assert explanation.explanation_type == ExplanationType.GLOBAL
        assert explanation.values.shape == (1, 3)
        assert explanation.base_values == 0.5
        assert explanation.data.shape == (1, 3)
        assert explanation.feature_names == ['feature1', 'feature2', 'feature3']
        assert explanation.model_name == "TestModel"
        assert explanation.metadata == {'test': 'value'}


class TestFeatureImportance:
    """Tests para FeatureImportance."""
    
    def test_feature_importance_creation(self):
        """Test de creación de FeatureImportance."""
        fi = FeatureImportance(
            feature_name="test_feature",
            importance=0.5,
            rank=1,
            contribution=0.3,
            direction="positive"
        )
        
        assert fi.feature_name == "test_feature"
        assert fi.importance == 0.5
        assert fi.rank == 1
        assert fi.contribution == 0.3
        assert fi.direction == "positive"


class TestSHAPExplainer:
    """Tests para SHAPExplainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        y = X['feature1'] + X['feature2'] + np.random.normal(0, 0.1, 100)
        return X, y
    
    @pytest.fixture
    def sample_model(self, sample_data):
        """Modelo de muestra para tests."""
        X, y = sample_data
        
        # Crear modelo simple
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_shap_explainer_initialization(self):
        """Test de inicialización de SHAPExplainer."""
        explainer = SHAPExplainer()
        
        assert explainer.sample_size == 1000
        assert explainer.max_features == 50
        assert explainer.random_state == 42
        assert explainer.explainer is None
        assert explainer.model is None
        assert explainer.feature_names == []
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_fit_explainer(self, sample_data, sample_model):
        """Test de ajuste del explainer."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        
        assert explainer.model == sample_model
        assert explainer.feature_names == list(X.columns)
        assert explainer.explainer is not None
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_explain_global(self, sample_data, sample_model):
        """Test de explicación global."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        explanation = explainer.explain_global(X.head(10))
        
        assert isinstance(explanation, SHAPExplanation)
        assert explanation.explanation_type == ExplanationType.GLOBAL
        assert explanation.values.shape[1] == len(X.columns)
        assert explanation.data.shape[1] == len(X.columns)
        assert explanation.feature_names == list(X.columns)
        assert explanation.model_name == "LinearRegression"
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_explain_local(self, sample_data, sample_model):
        """Test de explicación local."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        explanation = explainer.explain_local(X.head(1), 0)
        
        assert isinstance(explanation, SHAPExplanation)
        assert explanation.explanation_type == ExplanationType.LOCAL
        assert explanation.values.shape[1] == len(X.columns)
        assert explanation.data.shape[1] == len(X.columns)
        assert explanation.feature_names == list(X.columns)
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_get_feature_importance(self, sample_data, sample_model):
        """Test de obtención de importancia de features."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        explanation = explainer.explain_global(X.head(10))
        feature_importance = explainer.get_feature_importance(explanation)
        
        assert isinstance(feature_importance, list)
        assert len(feature_importance) == len(X.columns)
        
        for fi in feature_importance:
            assert isinstance(fi, FeatureImportance)
            assert fi.feature_name in X.columns
            assert fi.importance >= 0
            assert fi.rank > 0
            assert fi.direction in ["positive", "negative", "neutral"]
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_get_explanation_summary(self, sample_data, sample_model):
        """Test de obtención de resumen de explicación."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        explanation = explainer.explain_global(X.head(10))
        summary = explainer.get_explanation_summary(explanation)
        
        assert isinstance(summary, dict)
        assert 'explanation_type' in summary
        assert 'model_name' in summary
        assert 'features_count' in summary
        assert 'samples_count' in summary
        assert 'base_value' in summary
        assert 'top_features' in summary
        assert 'statistics' in summary
        assert 'timestamp' in summary
        assert 'metadata' in summary
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_save_load_explanation(self, sample_data, sample_model, temp_dir):
        """Test de guardado y carga de explicación."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        explanation = explainer.explain_global(X.head(10))
        
        # Guardar explicación
        save_path = temp_dir + "/explanation.pkl"
        explainer.save_explanation(explanation, save_path)
        
        # Cargar explicación
        loaded_explanation = explainer.load_explanation(save_path)
        
        assert loaded_explanation.explanation_type == explanation.explanation_type
        assert np.array_equal(loaded_explanation.values, explanation.values)
        assert loaded_explanation.base_values == explanation.base_values
        assert np.array_equal(loaded_explanation.data, explanation.data)
        assert loaded_explanation.feature_names == explanation.feature_names
        assert loaded_explanation.model_name == explanation.model_name
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_get_summary(self, sample_data, sample_model):
        """Test de obtención de resumen."""
        X, y = sample_data
        explainer = SHAPExplainer()
        
        explainer.fit_explainer(sample_model, X, model_type="linear")
        summary = explainer.get_summary()
        
        assert isinstance(summary, dict)
        assert 'explainer_type' in summary
        assert 'model_name' in summary
        assert 'feature_names' in summary
        assert 'explanations_count' in summary
        assert 'config' in summary


class TestAuditEvent:
    """Tests para AuditEvent."""
    
    def test_audit_event_creation(self):
        """Test de creación de AuditEvent."""
        event = AuditEvent(
            event_id="test_event",
            event_type=EventType.MODEL_TRAINING,
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            data={"test": "data"},
            user_id="test_user",
            session_id="test_session",
            metadata={"meta": "data"}
        )
        
        assert event.event_id == "test_event"
        assert event.event_type == EventType.MODEL_TRAINING
        assert event.level == LogLevel.INFO
        assert event.message == "Test message"
        assert event.data == {"test": "data"}
        assert event.user_id == "test_user"
        assert event.session_id == "test_session"
        assert event.metadata == {"meta": "data"}


class TestModelDecision:
    """Tests para ModelDecision."""
    
    def test_model_decision_creation(self):
        """Test de creación de ModelDecision."""
        decision = ModelDecision(
            decision_id="test_decision",
            model_name="TestModel",
            timestamp=datetime.now(),
            input_data={"feature1": 1.0, "feature2": 2.0},
            prediction=0.8,
            confidence=0.9,
            features_used=["feature1", "feature2"],
            feature_importance={"feature1": 0.6, "feature2": 0.4},
            explanation={"reasoning": "test"},
            metadata={"meta": "data"}
        )
        
        assert decision.decision_id == "test_decision"
        assert decision.model_name == "TestModel"
        assert decision.prediction == 0.8
        assert decision.confidence == 0.9
        assert decision.features_used == ["feature1", "feature2"]
        assert decision.feature_importance == {"feature1": 0.6, "feature2": 0.4}
        assert decision.explanation == {"reasoning": "test"}
        assert decision.metadata == {"meta": "data"}


class TestTradeDecision:
    """Tests para TradeDecision."""
    
    def test_trade_decision_creation(self):
        """Test de creación de TradeDecision."""
        decision = TradeDecision(
            trade_id="test_trade",
            timestamp=datetime.now(),
            symbol="SPY",
            action="buy",
            quantity=100.0,
            price=400.0,
            reasoning="Test reasoning",
            model_decision_id="test_decision",
            risk_assessment={"risk": "low"},
            metadata={"meta": "data"}
        )
        
        assert decision.trade_id == "test_trade"
        assert decision.symbol == "SPY"
        assert decision.action == "buy"
        assert decision.quantity == 100.0
        assert decision.price == 400.0
        assert decision.reasoning == "Test reasoning"
        assert decision.model_decision_id == "test_decision"
        assert decision.risk_assessment == {"risk": "low"}
        assert decision.metadata == {"meta": "data"}


class TestAuditLogger:
    """Tests para AuditLogger."""
    
    @pytest.fixture
    def temp_db(self, temp_dir):
        """Base de datos temporal para tests."""
        return temp_dir + "/test_audit.db"
    
    def test_audit_logger_initialization(self, temp_db):
        """Test de inicialización de AuditLogger."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        assert logger.db_path == temp_db
        assert logger.enable_database == True
        assert logger.enable_console == False
        assert logger.enable_file == False
        assert logger.session_id is not None
        assert logger.events_count == 0
    
    def test_log_event(self, temp_db):
        """Test de registro de evento."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        event_id = logger.log_event(
            event_type=EventType.MODEL_TRAINING,
            message="Test event",
            data={"test": "data"},
            level=LogLevel.INFO
        )
        
        assert event_id is not None
        assert logger.events_count == 1
    
    def test_log_model_decision(self, temp_db):
        """Test de registro de decisión de modelo."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        decision_id = logger.log_model_decision(
            model_name="TestModel",
            input_data={"feature1": 1.0},
            prediction=0.8,
            confidence=0.9,
            features_used=["feature1"],
            feature_importance={"feature1": 1.0},
            explanation={"reasoning": "test"}
        )
        
        assert decision_id is not None
    
    def test_log_trade_decision(self, temp_db):
        """Test de registro de decisión de trading."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        trade_id = logger.log_trade_decision(
            symbol="SPY",
            action="buy",
            quantity=100.0,
            price=400.0,
            reasoning="Test reasoning"
        )
        
        assert trade_id is not None
    
    def test_get_events(self, temp_db):
        """Test de obtención de eventos."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        # Registrar algunos eventos
        logger.log_event(EventType.MODEL_TRAINING, "Event 1")
        logger.log_event(EventType.MODEL_PREDICTION, "Event 2")
        
        # Obtener eventos
        events = logger.get_events()
        
        assert len(events) == 2
        assert events[0]['event_type'] == "model_prediction"  # Más reciente primero
        assert events[1]['event_type'] == "model_training"
    
    def test_get_model_decisions(self, temp_db):
        """Test de obtención de decisiones de modelo."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        # Registrar decisión de modelo
        logger.log_model_decision(
            model_name="TestModel",
            input_data={"feature1": 1.0},
            prediction=0.8,
            confidence=0.9,
            features_used=["feature1"],
            feature_importance={"feature1": 1.0}
        )
        
        # Obtener decisiones
        decisions = logger.get_model_decisions()
        
        assert len(decisions) == 1
        assert decisions[0]['model_name'] == "TestModel"
        assert decisions[0]['prediction'] == 0.8
        assert decisions[0]['confidence'] == 0.9
    
    def test_get_trade_decisions(self, temp_db):
        """Test de obtención de decisiones de trading."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        # Registrar decisión de trading
        logger.log_trade_decision(
            symbol="SPY",
            action="buy",
            quantity=100.0,
            price=400.0,
            reasoning="Test reasoning"
        )
        
        # Obtener decisiones
        decisions = logger.get_trade_decisions()
        
        assert len(decisions) == 1
        assert decisions[0]['symbol'] == "SPY"
        assert decisions[0]['action'] == "buy"
        assert decisions[0]['quantity'] == 100.0
        assert decisions[0]['price'] == 400.0
    
    def test_create_audit_report(self, temp_db):
        """Test de creación de reporte de auditoría."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        # Registrar algunos eventos
        logger.log_event(EventType.MODEL_TRAINING, "Event 1")
        logger.log_model_decision(
            model_name="TestModel",
            input_data={"feature1": 1.0},
            prediction=0.8,
            confidence=0.9,
            features_used=["feature1"],
            feature_importance={"feature1": 1.0}
        )
        logger.log_trade_decision(
            symbol="SPY",
            action="buy",
            quantity=100.0,
            price=400.0,
            reasoning="Test reasoning"
        )
        
        # Crear reporte
        report = logger.create_audit_report()
        
        assert 'report_period' in report
        assert 'summary' in report
        assert 'event_types' in report
        assert 'model_usage' in report
        assert 'symbols_traded' in report
        assert report['summary']['total_events'] == 1
        assert report['summary']['total_model_decisions'] == 1
        assert report['summary']['total_trade_decisions'] == 1
    
    def test_cleanup_old_logs(self, temp_db):
        """Test de limpieza de logs antiguos."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        # Registrar evento
        logger.log_event(EventType.MODEL_TRAINING, "Event 1")
        
        # Limpiar logs antiguos (muy agresivo)
        deleted = logger.cleanup_old_logs(days=0)
        
        assert deleted > 0
    
    def test_get_summary(self, temp_db):
        """Test de obtención de resumen."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        
        summary = logger.get_summary()
        
        assert 'session_id' in summary
        assert 'events_count' in summary
        assert 'config' in summary
        assert 'database_enabled' in summary
        assert 'database_path' in summary
    
    def test_close(self, temp_db):
        """Test de cierre del logger."""
        config = {
            'db_path': temp_db,
            'enable_database': True,
            'enable_console': False,
            'enable_file': False
        }
        
        logger = AuditLogger(config)
        logger.close()  # No debería lanzar excepción


class TestExplainabilityManager:
    """Tests para ExplainabilityManager."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        y = X['feature1'] + X['feature2'] + np.random.normal(0, 0.1, 100)
        return X, y
    
    @pytest.fixture
    def sample_model(self, sample_data):
        """Modelo de muestra para tests."""
        X, y = sample_data
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def test_explainability_manager_initialization(self):
        """Test de inicialización de ExplainabilityManager."""
        manager = ExplainabilityManager()
        
        assert manager.shap_explainer is not None
        assert manager.audit_logger is not None
        assert manager.explanations == {}
        assert manager.model_explanations == {}
        assert manager.last_explanation is None
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_explain_model(self, sample_data, sample_model):
        """Test de explicación de modelo."""
        X, y = sample_data
        manager = ExplainabilityManager()
        
        explanations = manager.explain_model(sample_model, X, X.head(10), "TestModel")
        
        assert 'global' in explanations
        assert 'feature_importance' in explanations
        assert 'interactions' in explanations
        assert 'summary' in explanations
        assert 'TestModel' in manager.model_explanations
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP no está disponible")
    def test_explain_prediction(self, sample_data, sample_model):
        """Test de explicación de predicción."""
        X, y = sample_data
        manager = ExplainabilityManager()
        
        prediction_explanation = manager.explain_prediction(
            sample_model, X, X.head(1), "TestModel"
        )
        
        assert 'prediction' in prediction_explanation
        assert 'confidence' in prediction_explanation
        assert 'explanation' in prediction_explanation
        assert 'feature_importance' in prediction_explanation
        assert 'top_features' in prediction_explanation
        assert 'reasoning' in prediction_explanation
        assert 'decision_id' in prediction_explanation
    
    def test_explain_trade_decision(self):
        """Test de explicación de decisión de trading."""
        manager = ExplainabilityManager()
        
        trade_explanation = manager.explain_trade_decision(
            symbol="SPY",
            action="buy",
            quantity=100.0,
            price=400.0,
            reasoning="Test reasoning"
        )
        
        assert 'trade_id' in trade_explanation
        assert trade_explanation['symbol'] == "SPY"
        assert trade_explanation['action'] == "buy"
        assert trade_explanation['quantity'] == 100.0
        assert trade_explanation['price'] == 400.0
        assert 'reasoning' in trade_explanation
        assert 'timestamp' in trade_explanation
    
    def test_create_explanation_report(self):
        """Test de creación de reporte de explicación."""
        manager = ExplainabilityManager()
        
        report = manager.create_explanation_report()
        
        assert isinstance(report, str)
        assert "REPORTE COMPRENSIVO DE EXPLICABILIDAD" in report
    
    def test_export_explanations(self, temp_dir):
        """Test de exportación de explicaciones."""
        manager = ExplainabilityManager()
        
        # Agregar explicación de muestra
        manager.model_explanations['TestModel'] = {
            'summary': {'test': 'data'},
            'feature_importance': []
        }
        
        # Exportar a JSON
        json_path = temp_dir + "/explanations.json"
        manager.export_explanations(json_path, "json")
        
        assert os.path.exists(json_path)
        
        # Exportar a pickle
        pickle_path = temp_dir + "/explanations.pkl"
        manager.export_explanations(pickle_path, "pickle")
        
        assert os.path.exists(pickle_path)
    
    def test_get_audit_report(self):
        """Test de obtención de reporte de auditoría."""
        manager = ExplainabilityManager()
        
        report = manager.get_audit_report()
        
        assert isinstance(report, dict)
        assert 'report_period' in report
        assert 'summary' in report
        assert 'event_types' in report
        assert 'model_usage' in report
        assert 'symbols_traded' in report
    
    def test_get_model_explanations(self):
        """Test de obtención de explicaciones de modelos."""
        manager = ExplainabilityManager()
        
        # Agregar explicación de muestra
        manager.model_explanations['TestModel'] = {'test': 'data'}
        
        # Obtener todas las explicaciones
        all_explanations = manager.get_model_explanations()
        assert 'TestModel' in all_explanations
        
        # Obtener explicación específica
        specific_explanation = manager.get_model_explanations('TestModel')
        assert specific_explanation == {'test': 'data'}
        
        # Obtener explicación inexistente
        non_existent = manager.get_model_explanations('NonExistent')
        assert non_existent == {}
    
    def test_cleanup_old_logs(self):
        """Test de limpieza de logs antiguos."""
        manager = ExplainabilityManager()
        
        deleted = manager.cleanup_old_logs(days=0)
        
        assert isinstance(deleted, int)
        assert deleted >= 0
    
    def test_close(self):
        """Test de cierre del manager."""
        manager = ExplainabilityManager()
        manager.close()  # No debería lanzar excepción
    
    def test_get_summary(self):
        """Test de obtención de resumen."""
        manager = ExplainabilityManager()
        
        summary = manager.get_summary()
        
        assert 'config' in summary
        assert 'model_explanations_count' in summary
        assert 'models_explained' in summary
        assert 'audit_summary' in summary
        assert 'shap_summary' in summary


if __name__ == "__main__":
    pytest.main([__file__])

