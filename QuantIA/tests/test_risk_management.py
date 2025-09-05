"""
Tests para gestión de riesgo.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.risk.base import BaseRiskManager, RiskLimits, RiskAlert, Position, RiskType, RiskLevel, RiskMetrics
from src.risk.volatility_targeting import VolatilityTargetingRiskManager
from src.risk.circuit_breakers import CircuitBreakerRiskManager, CircuitBreakerType, CircuitBreakerStatus
from src.risk.risk_manager import RiskManager


class TestRiskLimits:
    """Tests para RiskLimits."""
    
    def test_risk_limits_creation(self):
        """Test de creación de límites de riesgo."""
        limits = RiskLimits(
            max_position_size=0.1,
            max_portfolio_volatility=0.20,
            max_drawdown=0.10,
            max_leverage=2.0,
            max_concentration=0.3,
            max_correlation=0.7,
            var_limit_95=-0.05,
            var_limit_99=-0.10,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        
        assert limits.max_position_size == 0.1
        assert limits.max_portfolio_volatility == 0.20
        assert limits.max_drawdown == 0.10
        assert limits.max_leverage == 2.0
        assert limits.max_concentration == 0.3


class TestRiskAlert:
    """Tests para RiskAlert."""
    
    def test_risk_alert_creation(self):
        """Test de creación de alerta de riesgo."""
        alert = RiskAlert(
            risk_type=RiskType.MARKET,
            risk_level=RiskLevel.HIGH,
            message="Test alert",
            value=0.15,
            threshold=0.10,
            timestamp=datetime.now(),
            symbol="SPY",
            action_required=True
        )
        
        assert alert.risk_type == RiskType.MARKET
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.message == "Test alert"
        assert alert.value == 0.15
        assert alert.threshold == 0.10
        assert alert.symbol == "SPY"
        assert alert.action_required == True
    
    def test_risk_alert_to_dict(self):
        """Test de conversión a diccionario."""
        alert = RiskAlert(
            risk_type=RiskType.MARKET,
            risk_level=RiskLevel.HIGH,
            message="Test alert",
            value=0.15,
            threshold=0.10,
            timestamp=datetime.now()
        )
        
        alert_dict = alert.to_dict()
        
        assert 'risk_type' in alert_dict
        assert 'risk_level' in alert_dict
        assert 'message' in alert_dict
        assert 'value' in alert_dict
        assert 'threshold' in alert_dict
        assert 'timestamp' in alert_dict


class TestPosition:
    """Tests para Position."""
    
    def test_position_creation(self):
        """Test de creación de posición."""
        position = Position(
            symbol="SPY",
            quantity=100.0,
            price=400.0,
            timestamp=datetime.now(),
            side="long",
            stop_loss=380.0,
            take_profit=420.0
        )
        
        assert position.symbol == "SPY"
        assert position.quantity == 100.0
        assert position.price == 400.0
        assert position.side == "long"
        assert position.stop_loss == 380.0
        assert position.take_profit == 420.0


class TestVolatilityTargetingRiskManager:
    """Tests para VolatilityTargetingRiskManager."""
    
    @pytest.fixture
    def sample_returns(self):
        """Retornos de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        return returns
    
    @pytest.fixture
    def sample_positions(self):
        """Posiciones de muestra para tests."""
        positions = [
            Position(
                symbol="SPY",
                quantity=100.0,
                price=400.0,
                timestamp=datetime.now(),
                side="long"
            ),
            Position(
                symbol="QQQ",
                quantity=50.0,
                price=300.0,
                timestamp=datetime.now(),
                side="long"
            )
        ]
        return positions
    
    def test_volatility_targeting_initialization(self):
        """Test de inicialización del gestor de volatility targeting."""
        manager = VolatilityTargetingRiskManager()
        
        assert manager.name == "volatility_targeting"
        assert manager.target_volatility == 0.15
        assert manager.volatility_window == 20
        assert manager.volatility_method == "ewma"
    
    def test_calculate_volatility_simple(self, sample_returns):
        """Test de cálculo de volatilidad simple."""
        manager = VolatilityTargetingRiskManager()
        
        volatility = manager._calculate_simple_volatility(sample_returns)
        
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_calculate_volatility_ewma(self, sample_returns):
        """Test de cálculo de volatilidad EWMA."""
        manager = VolatilityTargetingRiskManager()
        
        volatility = manager._calculate_ewma_volatility(sample_returns)
        
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_adjust_position_size(self, sample_returns):
        """Test de ajuste de tamaño de posición."""
        manager = VolatilityTargetingRiskManager()
        
        # Actualizar volatilidad estimada
        manager.update_volatility_estimates("SPY", sample_returns)
        
        # Ajustar posición
        position_size = manager.adjust_position_size(
            signal=1.0,
            symbol="SPY",
            current_price=400.0,
            portfolio_value=100000.0
        )
        
        assert position_size > 0
        assert isinstance(position_size, float)
    
    def test_check_risk_limits(self, sample_positions, sample_returns):
        """Test de verificación de límites de riesgo."""
        manager = VolatilityTargetingRiskManager()
        
        alerts = manager.check_risk_limits(sample_positions, sample_returns)
        
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, RiskAlert)
    
    def test_rebalance_portfolio(self, sample_positions):
        """Test de rebalance del portafolio."""
        manager = VolatilityTargetingRiskManager()
        
        current_prices = {"SPY": 410.0, "QQQ": 310.0}
        target_weights = {"SPY": 0.6, "QQQ": 0.4}
        
        rebalanced_positions = manager.rebalance_portfolio(
            sample_positions, current_prices, target_weights
        )
        
        assert len(rebalanced_positions) == 2
        assert all(isinstance(p, Position) for p in rebalanced_positions)
    
    def test_set_volatility_target(self):
        """Test de establecimiento de volatilidad objetivo."""
        manager = VolatilityTargetingRiskManager()
        
        new_target = 0.20
        manager.set_volatility_target(new_target)
        
        assert manager.target_volatility == new_target
    
    def test_get_risk_summary(self):
        """Test de obtención de resumen de riesgo."""
        manager = VolatilityTargetingRiskManager()
        
        summary = manager.get_risk_summary()
        
        assert 'target_volatility' in summary
        assert 'volatility_estimates' in summary
        assert 'positions_count' in summary
        assert 'alerts_count' in summary


class TestCircuitBreakerRiskManager:
    """Tests para CircuitBreakerRiskManager."""
    
    @pytest.fixture
    def sample_positions(self):
        """Posiciones de muestra para tests."""
        positions = [
            Position(
                symbol="SPY",
                quantity=100.0,
                price=400.0,
                timestamp=datetime.now(),
                side="long"
            )
        ]
        return positions
    
    @pytest.fixture
    def sample_returns(self):
        """Retornos de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        return returns
    
    def test_circuit_breaker_initialization(self):
        """Test de inicialización del gestor de circuit breakers."""
        manager = CircuitBreakerRiskManager()
        
        assert manager.name == "circuit_breakers"
        assert len(manager.circuit_breakers) > 0
        assert 'price_drop' in manager.circuit_breakers
        assert 'volatility_spike' in manager.circuit_breakers
    
    def test_check_price_breaker(self, sample_positions):
        """Test de verificación de circuit breaker de precio."""
        manager = CircuitBreakerRiskManager()
        
        # Simular caída de precio
        manager.price_history["SPY"] = [400.0, 380.0, 360.0]  # 5% caída
        
        state = manager.circuit_breakers['price_drop']
        is_triggered = manager._check_price_breaker(state, sample_positions)
        
        assert isinstance(is_triggered, bool)
    
    def test_check_volatility_breaker(self, sample_returns):
        """Test de verificación de circuit breaker de volatilidad."""
        manager = CircuitBreakerRiskManager()
        
        # Crear retornos con alta volatilidad
        high_vol_returns = pd.Series(np.random.normal(0, 0.1, 100))  # 10% volatilidad
        
        state = manager.circuit_breakers['volatility_spike']
        is_triggered = manager._check_volatility_breaker(state, high_vol_returns)
        
        assert isinstance(is_triggered, bool)
    
    def test_check_drawdown_breaker(self, sample_returns):
        """Test de verificación de circuit breaker de drawdown."""
        manager = CircuitBreakerRiskManager()
        
        # Crear retornos con drawdown
        drawdown_returns = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05] * 20)
        
        state = manager.circuit_breakers['drawdown_limit']
        is_triggered = manager._check_drawdown_breaker(state, drawdown_returns)
        
        assert isinstance(is_triggered, bool)
    
    def test_adjust_position_size(self, sample_returns):
        """Test de ajuste de tamaño de posición."""
        manager = CircuitBreakerRiskManager()
        
        position_size = manager.adjust_position_size(
            signal=1.0,
            symbol="SPY",
            current_price=400.0,
            portfolio_value=100000.0
        )
        
        assert position_size > 0
        assert isinstance(position_size, float)
    
    def test_update_market_data(self):
        """Test de actualización de datos de mercado."""
        manager = CircuitBreakerRiskManager()
        
        manager.update_market_data("SPY", 400.0, volume=1000000, volatility=0.2)
        
        assert "SPY" in manager.price_history
        assert "SPY" in manager.volume_history
        assert "SPY" in manager.volatility_history
    
    def test_enable_disable_circuit_breaker(self):
        """Test de habilitación/deshabilitación de circuit breakers."""
        manager = CircuitBreakerRiskManager()
        
        # Deshabilitar
        manager.disable_circuit_breaker("price_drop")
        assert not manager.circuit_breakers["price_drop"].rule.enabled
        
        # Habilitar
        manager.enable_circuit_breaker("price_drop")
        assert manager.circuit_breakers["price_drop"].rule.enabled
    
    def test_reset_circuit_breaker(self):
        """Test de reset de circuit breaker."""
        manager = CircuitBreakerRiskManager()
        
        # Activar circuit breaker
        state = manager.circuit_breakers["price_drop"]
        state.status = CircuitBreakerStatus.TRIGGERED
        state.triggered_at = datetime.now()
        
        # Resetear
        manager.reset_circuit_breaker("price_drop")
        
        assert state.status == CircuitBreakerStatus.NORMAL
        assert state.triggered_at is None
    
    def test_get_circuit_breaker_status(self):
        """Test de obtención de estado de circuit breakers."""
        manager = CircuitBreakerRiskManager()
        
        status = manager.get_circuit_breaker_status()
        
        assert isinstance(status, dict)
        assert 'price_drop' in status
        assert 'volatility_spike' in status
        
        for breaker_status in status.values():
            assert 'status' in breaker_status
            assert 'rule' in breaker_status


class TestRiskManager:
    """Tests para RiskManager principal."""
    
    @pytest.fixture
    def sample_positions(self):
        """Posiciones de muestra para tests."""
        positions = [
            Position(
                symbol="SPY",
                quantity=100.0,
                price=400.0,
                timestamp=datetime.now(),
                side="long"
            )
        ]
        return positions
    
    @pytest.fixture
    def sample_returns(self):
        """Retornos de muestra para tests."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        return returns
    
    def test_risk_manager_initialization(self):
        """Test de inicialización del gestor principal."""
        manager = RiskManager()
        
        assert manager.volatility_manager is not None
        assert manager.circuit_breaker_manager is not None
        assert manager.positions == []
        assert manager.alerts == []
    
    def test_update_portfolio(self, sample_positions, sample_returns):
        """Test de actualización del portafolio."""
        manager = RiskManager()
        
        current_prices = {"SPY": 410.0}
        
        summary = manager.update_portfolio(
            sample_positions, current_prices, sample_returns
        )
        
        assert 'portfolio_value' in summary
        assert 'positions_count' in summary
        assert 'risk_metrics' in summary
        assert 'alerts' in summary
        assert 'timestamp' in summary
    
    def test_adjust_position_size(self, sample_returns):
        """Test de ajuste de tamaño de posición."""
        manager = RiskManager()
        manager.portfolio_value = 100000.0
        
        position_size = manager.adjust_position_size(
            signal=1.0,
            symbol="SPY",
            current_price=400.0
        )
        
        assert isinstance(position_size, float)
    
    def test_add_remove_position(self):
        """Test de agregar/remover posiciones."""
        manager = RiskManager()
        
        position = Position(
            symbol="SPY",
            quantity=100.0,
            price=400.0,
            timestamp=datetime.now(),
            side="long"
        )
        
        # Agregar posición
        manager.add_position(position)
        assert len(manager.positions) == 1
        
        # Remover posición
        manager.remove_position("SPY")
        assert len(manager.positions) == 0
    
    def test_get_positions(self):
        """Test de obtención de posiciones."""
        manager = RiskManager()
        
        position = Position(
            symbol="SPY",
            quantity=100.0,
            price=400.0,
            timestamp=datetime.now(),
            side="long"
        )
        
        manager.add_position(position)
        
        positions = manager.get_positions()
        assert len(positions) == 1
        
        spy_positions = manager.get_positions("SPY")
        assert len(spy_positions) == 1
        
        qqq_positions = manager.get_positions("QQQ")
        assert len(qqq_positions) == 0
    
    def test_rebalance_portfolio(self, sample_positions):
        """Test de rebalance del portafolio."""
        manager = RiskManager()
        
        current_prices = {"SPY": 410.0}
        target_weights = {"SPY": 1.0}
        
        rebalanced_positions = manager.rebalance_portfolio(
            target_weights, current_prices
        )
        
        assert isinstance(rebalanced_positions, list)
    
    def test_emergency_stop(self, sample_positions):
        """Test de parada de emergencia."""
        manager = RiskManager()
        
        manager.positions = sample_positions
        
        result = manager.emergency_stop()
        
        assert result['status'] == 'executed'
        assert result['closed_positions'] == len(sample_positions)
        assert len(manager.positions) == 0
    
    def test_get_risk_summary(self):
        """Test de obtención de resumen de riesgo."""
        manager = RiskManager()
        
        summary = manager.get_risk_summary()
        
        assert 'portfolio' in summary
        assert 'volatility_targeting' in summary
        assert 'circuit_breakers' in summary
        assert 'alerts' in summary
        assert 'risk_metrics' in summary
    
    def test_save_load_state(self, temp_dir):
        """Test de guardado y carga de estado."""
        manager = RiskManager()
        
        # Agregar posición
        position = Position(
            symbol="SPY",
            quantity=100.0,
            price=400.0,
            timestamp=datetime.now(),
            side="long"
        )
        manager.add_position(position)
        
        # Guardar estado
        save_path = temp_dir + "/risk_state.pkl"
        manager.save_state(save_path)
        
        # Crear nuevo manager y cargar estado
        new_manager = RiskManager()
        new_manager.load_state(save_path)
        
        assert len(new_manager.positions) == 1
        assert new_manager.positions[0].symbol == "SPY"


class TestRiskMetrics:
    """Tests para RiskMetrics."""
    
    def test_risk_metrics_creation(self):
        """Test de creación de métricas de riesgo."""
        metrics = RiskMetrics(
            volatility=0.15,
            var_95=-0.05,
            var_99=-0.10,
            cvar_95=-0.06,
            cvar_99=-0.12,
            max_drawdown=-0.08,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.0,
            beta=1.1,
            correlation=0.7,
            concentration_risk=0.3,
            leverage_ratio=1.5,
            liquidity_risk=0.2,
            timestamp=datetime.now()
        )
        
        assert metrics.volatility == 0.15
        assert metrics.var_95 == -0.05
        assert metrics.max_drawdown == -0.08
        assert metrics.sharpe_ratio == 1.2
        assert metrics.beta == 1.1


if __name__ == "__main__":
    pytest.main([__file__])

