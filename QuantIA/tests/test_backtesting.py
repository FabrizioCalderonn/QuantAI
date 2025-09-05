"""
Tests para sistema de backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.backtesting.base import (
    BaseBacktester, BacktestConfig, BacktestResult, BacktestMetrics, 
    Trade, TradeDirection, TradeStatus, BacktestType
)
from src.backtesting.walk_forward import WalkForwardBacktester, WalkForwardConfig
from src.backtesting.purged_cv import PurgedCVBacktester, PurgedCVConfig
from src.backtesting.backtest_manager import BacktestManager


class TestBacktestConfig:
    """Tests para BacktestConfig."""
    
    def test_backtest_config_creation(self):
        """Test de creación de configuración de backtesting."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=1000000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.1,
            min_trade_size=100.0,
            max_trades_per_day=10,
            risk_free_rate=0.02,
            benchmark_symbol="SPY",
            rebalance_frequency="1D",
            lookback_window=252,
            min_periods=60,
            purge_period=1,
            gap_period=0,
            enable_short_selling=True,
            enable_leverage=False,
            max_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        
        assert config.start_date == datetime(2020, 1, 1)
        assert config.end_date == datetime(2023, 12, 31)
        assert config.initial_capital == 1000000.0
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0005
        assert config.max_position_size == 0.1
        assert config.min_trade_size == 100.0
        assert config.max_trades_per_day == 10
        assert config.risk_free_rate == 0.02
        assert config.benchmark_symbol == "SPY"
        assert config.rebalance_frequency == "1D"
        assert config.lookback_window == 252
        assert config.min_periods == 60
        assert config.purge_period == 1
        assert config.gap_period == 0
        assert config.enable_short_selling == True
        assert config.enable_leverage == False
        assert config.max_leverage == 1.0
        assert config.stop_loss_pct == 0.05
        assert config.take_profit_pct == 0.15


class TestTrade:
    """Tests para Trade."""
    
    def test_trade_creation(self):
        """Test de creación de trade."""
        trade = Trade(
            symbol="SPY",
            direction=TradeDirection.LONG,
            entry_time=datetime.now(),
            entry_price=400.0,
            quantity=100.0,
            exit_time=datetime.now() + timedelta(days=1),
            exit_price=410.0,
            status=TradeStatus.CLOSED,
            pnl=1000.0,
            pnl_pct=0.025,
            commission=10.0,
            slippage=5.0,
            stop_loss=380.0,
            take_profit=420.0
        )
        
        assert trade.symbol == "SPY"
        assert trade.direction == TradeDirection.LONG
        assert trade.entry_price == 400.0
        assert trade.quantity == 100.0
        assert trade.exit_price == 410.0
        assert trade.status == TradeStatus.CLOSED
        assert trade.pnl == 1000.0
        assert trade.pnl_pct == 0.025
        assert trade.commission == 10.0
        assert trade.slippage == 5.0
        assert trade.stop_loss == 380.0
        assert trade.take_profit == 420.0


class TestBacktestMetrics:
    """Tests para BacktestMetrics."""
    
    def test_backtest_metrics_creation(self):
        """Test de creación de métricas de backtesting."""
        metrics = BacktestMetrics(
            total_return=0.15,
            annualized_return=0.12,
            volatility=0.18,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown=-0.08,
            max_drawdown_duration=30,
            avg_drawdown=-0.03,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=-0.015,
            profit_factor=1.5,
            expectancy=0.005,
            var_95=-0.03,
            var_99=-0.05,
            cvar_95=-0.04,
            cvar_99=-0.06,
            beta=1.1,
            alpha=0.02,
            information_ratio=0.8,
            benchmark_return=0.10,
            excess_return=0.05,
            tracking_error=0.12,
            skewness=0.2,
            kurtosis=3.5,
            tail_ratio=1.2,
            common_sense_ratio=1.3,
            ulcer_index=0.05,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_days=1095
        )
        
        assert metrics.total_return == 0.15
        assert metrics.annualized_return == 0.12
        assert metrics.volatility == 0.18
        assert metrics.sharpe_ratio == 1.2
        assert metrics.sortino_ratio == 1.5
        assert metrics.calmar_ratio == 2.0
        assert metrics.max_drawdown == -0.08
        assert metrics.max_drawdown_duration == 30
        assert metrics.avg_drawdown == -0.03
        assert metrics.total_trades == 100
        assert metrics.winning_trades == 60
        assert metrics.losing_trades == 40
        assert metrics.win_rate == 0.6
        assert metrics.avg_win == 0.02
        assert metrics.avg_loss == -0.015
        assert metrics.profit_factor == 1.5
        assert metrics.expectancy == 0.005
        assert metrics.var_95 == -0.03
        assert metrics.var_99 == -0.05
        assert metrics.cvar_95 == -0.04
        assert metrics.cvar_99 == -0.06
        assert metrics.beta == 1.1
        assert metrics.alpha == 0.02
        assert metrics.information_ratio == 0.8
        assert metrics.benchmark_return == 0.10
        assert metrics.excess_return == 0.05
        assert metrics.tracking_error == 0.12
        assert metrics.skewness == 0.2
        assert metrics.kurtosis == 3.5
        assert metrics.tail_ratio == 1.2
        assert metrics.common_sense_ratio == 1.3
        assert metrics.ulcer_index == 0.05
        assert metrics.start_date == datetime(2020, 1, 1)
        assert metrics.end_date == datetime(2023, 12, 31)
        assert metrics.total_days == 1095


class TestWalkForwardConfig:
    """Tests para WalkForwardConfig."""
    
    def test_walk_forward_config_creation(self):
        """Test de creación de configuración de walk-forward."""
        config = WalkForwardConfig(
            train_period=252,
            test_period=63,
            step_size=21,
            min_train_periods=126,
            purge_period=1,
            gap_period=0,
            max_periods=10,
            retrain_frequency=1,
            validation_method="expanding"
        )
        
        assert config.train_period == 252
        assert config.test_period == 63
        assert config.step_size == 21
        assert config.min_train_periods == 126
        assert config.purge_period == 1
        assert config.gap_period == 0
        assert config.max_periods == 10
        assert config.retrain_frequency == 1
        assert config.validation_method == "expanding"


class TestPurgedCVConfig:
    """Tests para PurgedCVConfig."""
    
    def test_purged_cv_config_creation(self):
        """Test de creación de configuración de purged CV."""
        config = PurgedCVConfig(
            n_splits=5,
            test_size=0.2,
            purge_period=1,
            gap_period=0,
            min_train_size=252,
            max_train_size=1000,
            validation_method="purged",
            shuffle=False,
            random_state=42
        )
        
        assert config.n_splits == 5
        assert config.test_size == 0.2
        assert config.purge_period == 1
        assert config.gap_period == 0
        assert config.min_train_size == 252
        assert config.max_train_size == 1000
        assert config.validation_method == "purged"
        assert config.shuffle == False
        assert config.random_state == 42


class TestWalkForwardBacktester:
    """Tests para WalkForwardBacktester."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'SPY': 100 + np.cumsum(np.random.normal(0, 1, 1000)),
            'QQQ': 200 + np.cumsum(np.random.normal(0, 1.5, 1000)),
            'IWM': 150 + np.cumsum(np.random.normal(0, 1.2, 1000))
        }, index=dates)
        
        # Agregar columnas OHLCV
        for symbol in data.columns:
            data[f'{symbol}_open'] = data[symbol] * (1 + np.random.normal(0, 0.001, 1000))
            data[f'{symbol}_high'] = data[symbol] * (1 + np.abs(np.random.normal(0, 0.002, 1000)))
            data[f'{symbol}_low'] = data[symbol] * (1 - np.abs(np.random.normal(0, 0.002, 1000)))
            data[f'{symbol}_close'] = data[symbol]
            data[f'{symbol}_volume'] = np.random.randint(1000000, 10000000, 1000)
        
        return data
    
    @pytest.fixture
    def sample_config(self):
        """Configuración de muestra para tests."""
        return BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2022, 12, 31),
            initial_capital=1000000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.1,
            min_trade_size=100.0,
            max_trades_per_day=10,
            risk_free_rate=0.02,
            benchmark_symbol="SPY",
            rebalance_frequency="1D",
            lookback_window=252,
            min_periods=60,
            purge_period=1,
            gap_period=0,
            enable_short_selling=True,
            enable_leverage=False,
            max_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
    
    def test_walk_forward_initialization(self, sample_config):
        """Test de inicialización del walk-forward backtester."""
        wf_config = WalkForwardConfig()
        backtester = WalkForwardBacktester(sample_config, wf_config)
        
        assert backtester.config == sample_config
        assert backtester.walk_forward_config == wf_config
        assert backtester.periods == []
        assert backtester.period_results == []
        assert backtester.consolidated_results is None
    
    def test_generate_walk_forward_periods(self, sample_data, sample_config):
        """Test de generación de períodos de walk-forward."""
        wf_config = WalkForwardConfig(
            train_period=100,
            test_period=20,
            step_size=10,
            min_train_periods=50,
            max_periods=5
        )
        backtester = WalkForwardBacktester(sample_config, wf_config)
        
        backtester._generate_walk_forward_periods(sample_data)
        
        assert len(backtester.periods) > 0
        assert len(backtester.periods) <= wf_config.max_periods
        
        for period in backtester.periods:
            assert 'period_id' in period
            assert 'train_start' in period
            assert 'train_end' in period
            assert 'test_start' in period
            assert 'test_end' in period
            assert period['train_start'] < period['train_end']
            assert period['train_end'] < period['test_start']
            assert period['test_start'] < period['test_end']
    
    def test_run_walk_forward_backtest(self, sample_data, sample_config):
        """Test de ejecución de walk-forward backtesting."""
        wf_config = WalkForwardConfig(
            train_period=100,
            test_period=20,
            step_size=10,
            min_train_periods=50,
            max_periods=3
        )
        backtester = WalkForwardBacktester(sample_config, wf_config)
        
        results = backtester.run_walk_forward_backtest(sample_data)
        
        assert 'walk_forward_config' in results
        assert 'periods' in results
        assert 'period_results' in results
        assert 'consolidated_metrics' in results
        assert 'summary' in results
        
        assert len(results['periods']) > 0
        assert len(results['period_results']) > 0
        assert len(results['period_results']) == len(results['periods'])
    
    def test_consolidate_results(self, sample_data, sample_config):
        """Test de consolidación de resultados."""
        wf_config = WalkForwardConfig(
            train_period=100,
            test_period=20,
            step_size=10,
            min_train_periods=50,
            max_periods=2
        )
        backtester = WalkForwardBacktester(sample_config, wf_config)
        
        # Ejecutar backtesting
        backtester.run_walk_forward_backtest(sample_data)
        
        # Verificar consolidación
        assert backtester.consolidated_results is not None
        assert 'consolidated_metrics' in backtester.consolidated_results
        assert 'summary' in backtester.consolidated_results
        
        # Verificar métricas consolidadas
        consolidated_metrics = backtester.consolidated_results['consolidated_metrics']
        assert 'total_return' in consolidated_metrics
        assert 'volatility' in consolidated_metrics
        assert 'sharpe_ratio' in consolidated_metrics
        assert 'max_drawdown' in consolidated_metrics
    
    def test_get_summary(self, sample_data, sample_config):
        """Test de obtención de resumen."""
        wf_config = WalkForwardConfig(
            train_period=100,
            test_period=20,
            step_size=10,
            min_train_periods=50,
            max_periods=2
        )
        backtester = WalkForwardBacktester(sample_config, wf_config)
        
        # Ejecutar backtesting
        backtester.run_walk_forward_backtest(sample_data)
        
        # Obtener resumen
        summary = backtester.get_summary()
        
        assert 'config' in summary
        assert 'walk_forward_config' in summary
        assert 'periods_count' in summary
        assert 'period_results_count' in summary
        assert 'consolidated_results' in summary


class TestPurgedCVBacktester:
    """Tests para PurgedCVBacktester."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'SPY': 100 + np.cumsum(np.random.normal(0, 1, 1000)),
            'QQQ': 200 + np.cumsum(np.random.normal(0, 1.5, 1000)),
            'IWM': 150 + np.cumsum(np.random.normal(0, 1.2, 1000))
        }, index=dates)
        
        # Agregar columnas OHLCV
        for symbol in data.columns:
            data[f'{symbol}_open'] = data[symbol] * (1 + np.random.normal(0, 0.001, 1000))
            data[f'{symbol}_high'] = data[symbol] * (1 + np.abs(np.random.normal(0, 0.002, 1000)))
            data[f'{symbol}_low'] = data[symbol] * (1 - np.abs(np.random.normal(0, 0.002, 1000)))
            data[f'{symbol}_close'] = data[symbol]
            data[f'{symbol}_volume'] = np.random.randint(1000000, 10000000, 1000)
        
        return data
    
    @pytest.fixture
    def sample_config(self):
        """Configuración de muestra para tests."""
        return BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2022, 12, 31),
            initial_capital=1000000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.1,
            min_trade_size=100.0,
            max_trades_per_day=10,
            risk_free_rate=0.02,
            benchmark_symbol="SPY",
            rebalance_frequency="1D",
            lookback_window=252,
            min_periods=60,
            purge_period=1,
            gap_period=0,
            enable_short_selling=True,
            enable_leverage=False,
            max_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
    
    def test_purged_cv_initialization(self, sample_config):
        """Test de inicialización del purged CV backtester."""
        cv_config = PurgedCVConfig()
        backtester = PurgedCVBacktester(sample_config, cv_config)
        
        assert backtester.config == sample_config
        assert backtester.purged_cv_config == cv_config
        assert backtester.cv_splits == []
        assert backtester.cv_results == []
        assert backtester.consolidated_results is None
    
    def test_generate_cv_splits(self, sample_data, sample_config):
        """Test de generación de splits de cross-validation."""
        cv_config = PurgedCVConfig(
            n_splits=3,
            test_size=0.2,
            min_train_size=100,
            max_train_size=500
        )
        backtester = PurgedCVBacktester(sample_config, cv_config)
        
        backtester._generate_cv_splits(sample_data)
        
        assert len(backtester.cv_splits) > 0
        assert len(backtester.cv_splits) <= cv_config.n_splits
        
        for split in backtester.cv_splits:
            assert 'fold_id' in split
            assert 'train_start' in split
            assert 'train_end' in split
            assert 'test_start' in split
            assert 'test_end' in split
            assert split['train_start'] < split['train_end']
            assert split['train_end'] < split['test_start']
            assert split['test_start'] < split['test_end']
    
    def test_run_purged_cv_backtest(self, sample_data, sample_config):
        """Test de ejecución de purged CV backtesting."""
        cv_config = PurgedCVConfig(
            n_splits=3,
            test_size=0.2,
            min_train_size=100,
            max_train_size=500
        )
        backtester = PurgedCVBacktester(sample_config, cv_config)
        
        results = backtester.run_purged_cv_backtest(sample_data)
        
        assert 'purged_cv_config' in results
        assert 'cv_splits' in results
        assert 'cv_results' in results
        assert 'consolidated_metrics' in results
        assert 'summary' in results
        
        assert len(results['cv_splits']) > 0
        assert len(results['cv_results']) > 0
        assert len(results['cv_results']) == len(results['cv_splits'])
    
    def test_consolidate_cv_results(self, sample_data, sample_config):
        """Test de consolidación de resultados de CV."""
        cv_config = PurgedCVConfig(
            n_splits=3,
            test_size=0.2,
            min_train_size=100,
            max_train_size=500
        )
        backtester = PurgedCVBacktester(sample_config, cv_config)
        
        # Ejecutar backtesting
        backtester.run_purged_cv_backtest(sample_data)
        
        # Verificar consolidación
        assert backtester.consolidated_results is not None
        assert 'consolidated_metrics' in backtester.consolidated_results
        assert 'summary' in backtester.consolidated_results
        
        # Verificar métricas consolidadas
        consolidated_metrics = backtester.consolidated_results['consolidated_metrics']
        assert 'total_return' in consolidated_metrics
        assert 'volatility' in consolidated_metrics
        assert 'sharpe_ratio' in consolidated_metrics
        assert 'max_drawdown' in consolidated_metrics
    
    def test_get_summary(self, sample_data, sample_config):
        """Test de obtención de resumen."""
        cv_config = PurgedCVConfig(
            n_splits=3,
            test_size=0.2,
            min_train_size=100,
            max_train_size=500
        )
        backtester = PurgedCVBacktester(sample_config, cv_config)
        
        # Ejecutar backtesting
        backtester.run_purged_cv_backtest(sample_data)
        
        # Obtener resumen
        summary = backtester.get_summary()
        
        assert 'config' in summary
        assert 'purged_cv_config' in summary
        assert 'cv_splits_count' in summary
        assert 'cv_results_count' in summary
        assert 'consolidated_results' in summary


class TestBacktestManager:
    """Tests para BacktestManager."""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'SPY': 100 + np.cumsum(np.random.normal(0, 1, 1000)),
            'QQQ': 200 + np.cumsum(np.random.normal(0, 1.5, 1000)),
            'IWM': 150 + np.cumsum(np.random.normal(0, 1.2, 1000))
        }, index=dates)
        
        # Agregar columnas OHLCV
        for symbol in data.columns:
            data[f'{symbol}_open'] = data[symbol] * (1 + np.random.normal(0, 0.001, 1000))
            data[f'{symbol}_high'] = data[symbol] * (1 + np.abs(np.random.normal(0, 0.002, 1000)))
            data[f'{symbol}_low'] = data[symbol] * (1 - np.abs(np.random.normal(0, 0.002, 1000)))
            data[f'{symbol}_close'] = data[symbol]
            data[f'{symbol}_volume'] = np.random.randint(1000000, 10000000, 1000)
        
        return data
    
    def test_backtest_manager_initialization(self):
        """Test de inicialización del gestor de backtesting."""
        manager = BacktestManager()
        
        assert manager.default_config is not None
        assert manager.walk_forward_config is not None
        assert manager.purged_cv_config is not None
        assert manager.results == {}
        assert manager.current_backtest is None
        assert manager.last_run is None
    
    def test_run_backtest_walk_forward(self, sample_data):
        """Test de ejecución de backtesting walk-forward."""
        manager = BacktestManager()
        
        results = manager.run_backtest(
            sample_data, 
            backtest_type=BacktestType.WALK_FORWARD
        )
        
        assert 'walk_forward' in manager.results
        assert 'walk_forward_config' in results
        assert 'periods' in results
        assert 'period_results' in results
        assert 'consolidated_metrics' in results
        assert 'summary' in results
    
    def test_run_backtest_purged_cv(self, sample_data):
        """Test de ejecución de backtesting purged CV."""
        manager = BacktestManager()
        
        results = manager.run_backtest(
            sample_data, 
            backtest_type=BacktestType.PURGED_CV
        )
        
        assert 'purged_cv' in manager.results
        assert 'purged_cv_config' in results
        assert 'cv_splits' in results
        assert 'cv_results' in results
        assert 'consolidated_metrics' in results
        assert 'summary' in results
    
    def test_run_comprehensive_backtest(self, sample_data):
        """Test de ejecución de backtesting comprensivo."""
        manager = BacktestManager()
        
        results = manager.run_comprehensive_backtest(sample_data)
        
        assert 'walk_forward' in results
        assert 'purged_cv' in results
        assert 'comparison' in results
        assert 'summary' in results
        
        # Verificar que se guardaron los resultados
        assert 'walk_forward' in manager.results
        assert 'purged_cv' in manager.results
    
    def test_get_results(self, sample_data):
        """Test de obtención de resultados."""
        manager = BacktestManager()
        
        # Ejecutar backtesting
        manager.run_backtest(sample_data, backtest_type=BacktestType.WALK_FORWARD)
        
        # Obtener resultados
        results = manager.get_results()
        assert 'walk_forward' in results
        
        # Obtener resultados específicos
        wf_results = manager.get_results('walk_forward')
        assert 'walk_forward_config' in wf_results
    
    def test_get_summary(self, sample_data):
        """Test de obtención de resumen."""
        manager = BacktestManager()
        
        # Ejecutar backtesting
        manager.run_backtest(sample_data, backtest_type=BacktestType.WALK_FORWARD)
        
        # Obtener resumen
        summary = manager.get_summary()
        
        assert 'config' in summary
        assert 'walk_forward_config' in summary
        assert 'purged_cv_config' in summary
        assert 'results_count' in summary
        assert 'last_run' in summary
        assert 'current_backtest' in summary
    
    def test_create_report(self, sample_data):
        """Test de creación de reporte."""
        manager = BacktestManager()
        
        # Ejecutar backtesting
        manager.run_backtest(sample_data, backtest_type=BacktestType.WALK_FORWARD)
        
        # Crear reporte
        report = manager.create_report('walk_forward')
        
        assert isinstance(report, str)
        assert 'REPORTE DE BACKTESTING' in report
        assert 'WALK_FORWARD' in report
    
    def test_save_load_results(self, sample_data, temp_dir):
        """Test de guardado y carga de resultados."""
        manager = BacktestManager()
        
        # Ejecutar backtesting
        manager.run_backtest(sample_data, backtest_type=BacktestType.WALK_FORWARD)
        
        # Guardar resultados
        save_path = temp_dir + "/backtest_results.pkl"
        manager.save_results(save_path, 'walk_forward')
        
        # Crear nuevo manager y cargar resultados
        new_manager = BacktestManager()
        new_manager.load_results(save_path, 'walk_forward')
        
        # Verificar que se cargaron los resultados
        assert 'walk_forward' in new_manager.results
        assert 'walk_forward_config' in new_manager.results['walk_forward']
    
    def test_export_results(self, sample_data, temp_dir):
        """Test de exportación de resultados."""
        manager = BacktestManager()
        
        # Ejecutar backtesting
        manager.run_backtest(sample_data, backtest_type=BacktestType.WALK_FORWARD)
        
        # Exportar a CSV
        csv_path = temp_dir + "/backtest_results.csv"
        manager.export_results(csv_path, 'csv', 'walk_forward')
        
        # Verificar que se creó el archivo
        import os
        assert os.path.exists(csv_path)
        
        # Exportar a Excel
        excel_path = temp_dir + "/backtest_results.xlsx"
        manager.export_results(excel_path, 'excel', 'walk_forward')
        
        # Verificar que se creó el archivo
        assert os.path.exists(excel_path)
        
        # Exportar a JSON
        json_path = temp_dir + "/backtest_results.json"
        manager.export_results(json_path, 'json', 'walk_forward')
        
        # Verificar que se creó el archivo
        assert os.path.exists(json_path)


if __name__ == "__main__":
    pytest.main([__file__])

