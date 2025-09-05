"""
Tests para sistema de validación.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.validation.metrics import ValidationMetrics, MetricType, ValidationLevel
from src.validation.validator import StrategyValidator, ValidationResult, ValidationStatus, ValidationCriteria, MetricThreshold
from src.validation.validation_manager import ValidationManager


class TestValidationMetrics:
    """Tests para ValidationMetrics."""
    
    @pytest.fixture
    def sample_returns(self):
        """Retornos de muestra para tests."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Retornos de benchmark para tests."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(43)
        returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_trades(self):
        """Trades de muestra para tests."""
        trades = []
        for i in range(50):
            trade = {
                'symbol': 'SPY',
                'entry_time': datetime.now() - timedelta(days=i),
                'exit_time': datetime.now() - timedelta(days=i-1),
                'entry_price': 100 + i,
                'exit_price': 100 + i + np.random.normal(0, 2),
                'status': 'closed',
                'quantity': 100
            }
            trades.append(trade)
        return trades
    
    def test_validation_metrics_initialization(self):
        """Test de inicialización de ValidationMetrics."""
        metrics = ValidationMetrics()
        
        assert metrics.risk_free_rate == 0.02
        assert metrics.trading_days == 252
    
    def test_calculate_performance_metrics(self, sample_returns, sample_benchmark_returns):
        """Test de cálculo de métricas de performance."""
        metrics = ValidationMetrics()
        
        performance_metrics = metrics.calculate_performance_metrics(
            sample_returns, sample_benchmark_returns
        )
        
        assert 'total_return' in performance_metrics
        assert 'annualized_return' in performance_metrics
        assert 'volatility' in performance_metrics
        assert 'sharpe_ratio' in performance_metrics
        assert 'sortino_ratio' in performance_metrics
        assert 'calmar_ratio' in performance_metrics
        assert 'beta' in performance_metrics
        assert 'alpha' in performance_metrics
        assert 'information_ratio' in performance_metrics
    
    def test_calculate_risk_metrics(self, sample_returns):
        """Test de cálculo de métricas de riesgo."""
        metrics = ValidationMetrics()
        
        risk_metrics = metrics.calculate_risk_metrics(sample_returns)
        
        assert 'var_95' in risk_metrics
        assert 'var_99' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'cvar_99' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'avg_drawdown' in risk_metrics
        assert 'max_drawdown_duration' in risk_metrics
        assert 'skewness' in risk_metrics
        assert 'kurtosis' in risk_metrics
        assert 'tail_ratio' in risk_metrics
        assert 'ulcer_index' in risk_metrics
    
    def test_calculate_trading_metrics(self, sample_trades):
        """Test de cálculo de métricas de trading."""
        metrics = ValidationMetrics()
        
        trading_metrics = metrics.calculate_trading_metrics(sample_trades)
        
        assert 'total_trades' in trading_metrics
        assert 'winning_trades' in trading_metrics
        assert 'losing_trades' in trading_metrics
        assert 'win_rate' in trading_metrics
        assert 'avg_win' in trading_metrics
        assert 'avg_loss' in trading_metrics
        assert 'profit_factor' in trading_metrics
        assert 'expectancy' in trading_metrics
        assert 'max_consecutive_wins' in trading_metrics
        assert 'max_consecutive_losses' in trading_metrics
        assert 'avg_trade_duration' in trading_metrics
    
    def test_calculate_stability_metrics(self, sample_returns):
        """Test de cálculo de métricas de estabilidad."""
        metrics = ValidationMetrics()
        
        stability_metrics = metrics.calculate_stability_metrics(sample_returns)
        
        assert 'return_stability' in stability_metrics
        assert 'volatility_stability' in stability_metrics
        assert 'positive_months' in stability_metrics
        assert 'positive_quarters' in stability_metrics
        assert 'positive_years' in stability_metrics
    
    def test_calculate_robustness_metrics(self, sample_returns, sample_benchmark_returns):
        """Test de cálculo de métricas de robustez."""
        metrics = ValidationMetrics()
        
        robustness_metrics = metrics.calculate_robustness_metrics(
            sample_returns, sample_benchmark_returns
        )
        
        assert 'outlier_ratio' in robustness_metrics
        assert 'fat_tail_ratio' in robustness_metrics
        assert 'normality_test' in robustness_metrics
        assert 'autocorrelation' in robustness_metrics
        assert 'heteroscedasticity' in robustness_metrics
        assert 'correlation_stability' in robustness_metrics
        assert 'beta_stability' in robustness_metrics
    
    def test_empty_returns(self):
        """Test con retornos vacíos."""
        metrics = ValidationMetrics()
        empty_returns = pd.Series(dtype=float)
        
        performance_metrics = metrics.calculate_performance_metrics(empty_returns)
        risk_metrics = metrics.calculate_risk_metrics(empty_returns)
        stability_metrics = metrics.calculate_stability_metrics(empty_returns)
        robustness_metrics = metrics.calculate_robustness_metrics(empty_returns)
        
        # Verificar que se retornan métricas vacías
        assert all(v == 0.0 for v in performance_metrics.values())
        assert all(v == 0.0 for v in risk_metrics.values())
        assert all(v == 0.0 for v in stability_metrics.values())
        assert all(v == 0.0 for v in robustness_metrics.values())


class TestMetricThreshold:
    """Tests para MetricThreshold."""
    
    def test_metric_threshold_creation(self):
        """Test de creación de MetricThreshold."""
        threshold = MetricThreshold(
            metric_name="sharpe_ratio",
            threshold_value=1.0,
            comparison_operator=">=",
            weight=1.0,
            required=True,
            description="Sharpe ratio mínimo"
        )
        
        assert threshold.metric_name == "sharpe_ratio"
        assert threshold.threshold_value == 1.0
        assert threshold.comparison_operator == ">="
        assert threshold.weight == 1.0
        assert threshold.required == True
        assert threshold.description == "Sharpe ratio mínimo"


class TestValidationCriteria:
    """Tests para ValidationCriteria."""
    
    def test_validation_criteria_creation(self):
        """Test de creación de ValidationCriteria."""
        thresholds = [
            MetricThreshold("sharpe_ratio", 1.0, ">=", 1.0, True, "Sharpe ratio mínimo"),
            MetricThreshold("max_drawdown", -0.1, ">=", 1.0, True, "Drawdown máximo")
        ]
        
        criteria = ValidationCriteria(
            level=ValidationLevel.INTERMEDIATE,
            thresholds=thresholds,
            min_score=0.7,
            required_metrics=["sharpe_ratio", "max_drawdown"],
            description="Validación intermedia"
        )
        
        assert criteria.level == ValidationLevel.INTERMEDIATE
        assert len(criteria.thresholds) == 2
        assert criteria.min_score == 0.7
        assert criteria.required_metrics == ["sharpe_ratio", "max_drawdown"]
        assert criteria.description == "Validación intermedia"


class TestStrategyValidator:
    """Tests para StrategyValidator."""
    
    @pytest.fixture
    def sample_returns(self):
        """Retornos de muestra para tests."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Retornos de benchmark para tests."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(43)
        returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_trades(self):
        """Trades de muestra para tests."""
        trades = []
        for i in range(50):
            trade = {
                'symbol': 'SPY',
                'entry_time': datetime.now() - timedelta(days=i),
                'exit_time': datetime.now() - timedelta(days=i-1),
                'entry_price': 100 + i,
                'exit_price': 100 + i + np.random.normal(0, 2),
                'status': 'closed',
                'quantity': 100
            }
            trades.append(trade)
        return trades
    
    def test_strategy_validator_initialization(self):
        """Test de inicialización de StrategyValidator."""
        validator = StrategyValidator()
        
        assert validator.metrics_calculator is not None
        assert len(validator.validation_criteria) == 4  # 4 niveles de validación
    
    def test_validate_strategy(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de validación de estrategia."""
        validator = StrategyValidator()
        
        result = validator.validate_strategy(
            strategy_name="test_strategy",
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            trades=sample_trades,
            validation_level=ValidationLevel.BASIC
        )
        
        assert isinstance(result, ValidationResult)
        assert result.strategy_name == "test_strategy"
        assert result.validation_level == ValidationLevel.BASIC
        assert isinstance(result.overall_score, float)
        assert isinstance(result.passed, bool)
        assert isinstance(result.metrics_results, dict)
        assert isinstance(result.failed_metrics, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.timestamp, datetime)
    
    def test_validate_multiple_strategies(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de validación de múltiples estrategias."""
        validator = StrategyValidator()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,  # Mejor performance
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        results = validator.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        
        assert len(results) == 2
        assert "strategy1" in results
        assert "strategy2" in results
        
        for strategy_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert result.strategy_name == strategy_name
    
    def test_get_validation_summary(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de obtención de resumen de validación."""
        validator = StrategyValidator()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        results = validator.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        summary = validator.get_validation_summary(results)
        
        assert 'total_strategies' in summary
        assert 'passed_strategies' in summary
        assert 'failed_strategies' in summary
        assert 'pass_rate' in summary
        assert 'avg_score' in summary
        assert 'min_score' in summary
        assert 'max_score' in summary
        assert 'levels' in summary
        assert 'most_problematic_metrics' in summary
        assert 'timestamp' in summary
    
    def test_get_validation_criteria(self):
        """Test de obtención de criterios de validación."""
        validator = StrategyValidator()
        
        # Obtener todos los criterios
        all_criteria = validator.get_validation_criteria()
        assert len(all_criteria) == 4
        
        # Obtener criterios específicos
        basic_criteria = validator.get_validation_criteria(ValidationLevel.BASIC)
        assert basic_criteria.level == ValidationLevel.BASIC
        assert len(basic_criteria.thresholds) > 0
    
    def test_update_validation_criteria(self):
        """Test de actualización de criterios de validación."""
        validator = StrategyValidator()
        
        # Crear nuevos criterios
        new_thresholds = [
            MetricThreshold("sharpe_ratio", 2.0, ">=", 1.0, True, "Sharpe ratio muy alto")
        ]
        
        new_criteria = ValidationCriteria(
            level=ValidationLevel.BASIC,
            thresholds=new_thresholds,
            min_score=0.9,
            required_metrics=["sharpe_ratio"],
            description="Criterios muy estrictos"
        )
        
        validator.update_validation_criteria(ValidationLevel.BASIC, new_criteria)
        
        updated_criteria = validator.get_validation_criteria(ValidationLevel.BASIC)
        assert updated_criteria.min_score == 0.9
        assert len(updated_criteria.thresholds) == 1
        assert updated_criteria.thresholds[0].threshold_value == 2.0
    
    def test_add_custom_threshold(self):
        """Test de agregar umbral personalizado."""
        validator = StrategyValidator()
        
        custom_threshold = MetricThreshold(
            metric_name="custom_metric",
            threshold_value=0.5,
            comparison_operator=">=",
            weight=0.5,
            required=False,
            description="Métrica personalizada"
        )
        
        validator.add_custom_threshold(ValidationLevel.BASIC, custom_threshold)
        
        criteria = validator.get_validation_criteria(ValidationLevel.BASIC)
        threshold_names = [t.metric_name for t in criteria.thresholds]
        assert "custom_metric" in threshold_names
    
    def test_remove_threshold(self):
        """Test de remover umbral."""
        validator = StrategyValidator()
        
        # Agregar umbral personalizado
        custom_threshold = MetricThreshold(
            metric_name="temp_metric",
            threshold_value=0.5,
            comparison_operator=">=",
            weight=0.5,
            required=False,
            description="Métrica temporal"
        )
        
        validator.add_custom_threshold(ValidationLevel.BASIC, custom_threshold)
        
        # Verificar que se agregó
        criteria = validator.get_validation_criteria(ValidationLevel.BASIC)
        threshold_names = [t.metric_name for t in criteria.thresholds]
        assert "temp_metric" in threshold_names
        
        # Remover umbral
        validator.remove_threshold(ValidationLevel.BASIC, "temp_metric")
        
        # Verificar que se removió
        criteria = validator.get_validation_criteria(ValidationLevel.BASIC)
        threshold_names = [t.metric_name for t in criteria.thresholds]
        assert "temp_metric" not in threshold_names


class TestValidationResult:
    """Tests para ValidationResult."""
    
    def test_validation_result_creation(self):
        """Test de creación de ValidationResult."""
        result = ValidationResult(
            strategy_name="test_strategy",
            validation_level=ValidationLevel.INTERMEDIATE,
            overall_score=0.8,
            passed=True,
            metrics_results={"performance": {"sharpe_ratio": 1.2}},
            failed_metrics=[],
            warnings=[],
            recommendations=["Estrategia aprobada"],
            timestamp=datetime.now()
        )
        
        assert result.strategy_name == "test_strategy"
        assert result.validation_level == ValidationLevel.INTERMEDIATE
        assert result.overall_score == 0.8
        assert result.passed == True
        assert result.metrics_results == {"performance": {"sharpe_ratio": 1.2}}
        assert result.failed_metrics == []
        assert result.warnings == []
        assert result.recommendations == ["Estrategia aprobada"]
        assert isinstance(result.timestamp, datetime)


class TestValidationManager:
    """Tests para ValidationManager."""
    
    @pytest.fixture
    def sample_returns(self):
        """Retornos de muestra para tests."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Retornos de benchmark para tests."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(43)
        returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
        return returns
    
    @pytest.fixture
    def sample_trades(self):
        """Trades de muestra para tests."""
        trades = []
        for i in range(50):
            trade = {
                'symbol': 'SPY',
                'entry_time': datetime.now() - timedelta(days=i),
                'exit_time': datetime.now() - timedelta(days=i-1),
                'entry_price': 100 + i,
                'exit_price': 100 + i + np.random.normal(0, 2),
                'status': 'closed',
                'quantity': 100
            }
            trades.append(trade)
        return trades
    
    def test_validation_manager_initialization(self):
        """Test de inicialización de ValidationManager."""
        manager = ValidationManager()
        
        assert manager.metrics_calculator is not None
        assert manager.validator is not None
        assert manager.validation_results == {}
        assert manager.last_validation is None
    
    def test_validate_strategy(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de validación de estrategia."""
        manager = ValidationManager()
        
        result = manager.validate_strategy(
            strategy_name="test_strategy",
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            trades=sample_trades,
            validation_level=ValidationLevel.BASIC
        )
        
        assert isinstance(result, ValidationResult)
        assert result.strategy_name == "test_strategy"
        assert "test_strategy" in manager.validation_results
        assert manager.last_validation is not None
    
    def test_validate_multiple_strategies(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de validación de múltiples estrategias."""
        manager = ValidationManager()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        results = manager.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        
        assert len(results) == 2
        assert len(manager.validation_results) == 2
        assert manager.last_validation is not None
    
    def test_validate_backtest_results(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de validación de resultados de backtesting."""
        manager = ValidationManager()
        
        backtest_results = {
            "walk_forward": {
                "consolidated_returns": sample_returns,
                "consolidated_benchmark_returns": sample_benchmark_returns,
                "consolidated_portfolio_values": pd.Series([1000, 1100, 1200]),
                "all_trades": sample_trades
            },
            "purged_cv": {
                "consolidated_returns": sample_returns * 1.05,
                "consolidated_benchmark_returns": sample_benchmark_returns,
                "consolidated_portfolio_values": pd.Series([1000, 1120, 1250]),
                "all_trades": sample_trades
            }
        }
        
        results = manager.validate_backtest_results(backtest_results, ValidationLevel.BASIC)
        
        assert len(results) == 2
        assert "walk_forward" in results
        assert "purged_cv" in results
    
    def test_get_validation_summary(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de obtención de resumen de validación."""
        manager = ValidationManager()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        manager.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        summary = manager.get_validation_summary()
        
        assert 'total_strategies' in summary
        assert 'passed_strategies' in summary
        assert 'failed_strategies' in summary
        assert 'pass_rate' in summary
        assert 'avg_score' in summary
    
    def test_get_passed_strategies(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de obtención de estrategias que pasaron."""
        manager = ValidationManager()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        manager.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        passed_strategies = manager.get_passed_strategies()
        
        assert isinstance(passed_strategies, list)
        assert all(isinstance(name, str) for name in passed_strategies)
    
    def test_get_failed_strategies(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de obtención de estrategias que fallaron."""
        manager = ValidationManager()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        manager.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        failed_strategies = manager.get_failed_strategies()
        
        assert isinstance(failed_strategies, list)
        assert all(isinstance(name, str) for name in failed_strategies)
    
    def test_get_strategies_by_score(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de obtención de estrategias por score."""
        manager = ValidationManager()
        
        strategies = {
            "strategy1": {
                "returns": sample_returns,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            },
            "strategy2": {
                "returns": sample_returns * 1.1,
                "benchmark_returns": sample_benchmark_returns,
                "trades": sample_trades
            }
        }
        
        manager.validate_multiple_strategies(strategies, ValidationLevel.BASIC)
        high_score_strategies = manager.get_strategies_by_score(min_score=0.5)
        
        assert isinstance(high_score_strategies, list)
        assert all(isinstance(name, str) for name in high_score_strategies)
    
    def test_create_validation_report(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de creación de reporte de validación."""
        manager = ValidationManager()
        
        manager.validate_strategy(
            strategy_name="test_strategy",
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            trades=sample_trades,
            validation_level=ValidationLevel.BASIC
        )
        
        # Reporte de estrategia específica
        single_report = manager.create_validation_report("test_strategy")
        assert isinstance(single_report, str)
        assert "test_strategy" in single_report
        
        # Reporte comprensivo
        comprehensive_report = manager.create_validation_report()
        assert isinstance(comprehensive_report, str)
        assert "REPORTE COMPRENSIVO" in comprehensive_report
    
    def test_export_validation_results(self, sample_returns, sample_benchmark_returns, sample_trades, temp_dir):
        """Test de exportación de resultados de validación."""
        manager = ValidationManager()
        
        manager.validate_strategy(
            strategy_name="test_strategy",
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            trades=sample_trades,
            validation_level=ValidationLevel.BASIC
        )
        
        # Exportar a CSV
        csv_path = temp_dir + "/validation_results.csv"
        manager.export_validation_results(csv_path, "csv")
        
        # Verificar que se creó el archivo
        import os
        assert os.path.exists(csv_path)
        
        # Exportar a Excel
        excel_path = temp_dir + "/validation_results.xlsx"
        manager.export_validation_results(excel_path, "excel")
        
        # Verificar que se creó el archivo
        assert os.path.exists(excel_path)
        
        # Exportar a JSON
        json_path = temp_dir + "/validation_results.json"
        manager.export_validation_results(json_path, "json")
        
        # Verificar que se creó el archivo
        assert os.path.exists(json_path)
    
    def test_save_load_validation_state(self, sample_returns, sample_benchmark_returns, sample_trades, temp_dir):
        """Test de guardado y carga de estado de validación."""
        manager = ValidationManager()
        
        manager.validate_strategy(
            strategy_name="test_strategy",
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            trades=sample_trades,
            validation_level=ValidationLevel.BASIC
        )
        
        # Guardar estado
        save_path = temp_dir + "/validation_state.pkl"
        manager.save_validation_state(save_path)
        
        # Crear nuevo manager y cargar estado
        new_manager = ValidationManager()
        new_manager.load_validation_state(save_path)
        
        # Verificar que se cargó el estado
        assert len(new_manager.validation_results) == 1
        assert "test_strategy" in new_manager.validation_results
        assert new_manager.last_validation is not None
    
    def test_get_summary(self, sample_returns, sample_benchmark_returns, sample_trades):
        """Test de obtención de resumen."""
        manager = ValidationManager()
        
        manager.validate_strategy(
            strategy_name="test_strategy",
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            trades=sample_trades,
            validation_level=ValidationLevel.BASIC
        )
        
        summary = manager.get_summary()
        
        assert 'config' in summary
        assert 'validation_results_count' in summary
        assert 'last_validation' in summary
        assert 'validation_summary' in summary


if __name__ == "__main__":
    pytest.main([__file__])

