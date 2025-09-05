"""
Tests para sistema de paper trading.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import os
import time

from src.paper_trading.portfolio import (
    PaperTradingPortfolio, Order, OrderType, OrderStatus, PositionSide, Position, Trade
)
from src.paper_trading.execution_engine import (
    PaperTradingExecutionEngine, ExecutionConfig, ExecutionMode, ExecutionResult
)
from src.paper_trading.paper_trading_manager import PaperTradingManager


class TestOrder:
    """Tests para Order."""
    
    def test_order_creation(self):
        """Test de creación de Order."""
        order = Order(
            order_id="test_order",
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        assert order.order_id == "test_order"
        assert order.symbol == "SPY"
        assert order.side == PositionSide.LONG
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100.0
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0


class TestPosition:
    """Tests para Position."""
    
    def test_position_creation(self):
        """Test de creación de Position."""
        position = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=405.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        assert position.symbol == "SPY"
        assert position.quantity == 100.0
        assert position.side == PositionSide.LONG
        assert position.entry_price == 400.0
        assert position.current_price == 405.0
        assert position.unrealized_pnl == 500.0  # (405 - 400) * 100


class TestTrade:
    """Tests para Trade."""
    
    def test_trade_creation(self):
        """Test de creación de Trade."""
        trade = Trade(
            trade_id="test_trade",
            symbol="SPY",
            side=PositionSide.LONG,
            quantity=100.0,
            price=400.0,
            timestamp=datetime.now(),
            commission=0.4,
            order_id="test_order"
        )
        
        assert trade.trade_id == "test_trade"
        assert trade.symbol == "SPY"
        assert trade.side == PositionSide.LONG
        assert trade.quantity == 100.0
        assert trade.price == 400.0
        assert trade.commission == 0.4
        assert trade.order_id == "test_order"


class TestPaperTradingPortfolio:
    """Tests para PaperTradingPortfolio."""
    
    @pytest.fixture
    def portfolio(self):
        """Portfolio de prueba."""
        return PaperTradingPortfolio(initial_cash=100000.0, commission_rate=0.001)
    
    def test_portfolio_initialization(self):
        """Test de inicialización del portfolio."""
        portfolio = PaperTradingPortfolio(initial_cash=50000.0, commission_rate=0.002)
        
        assert portfolio.initial_cash == 50000.0
        assert portfolio.cash == 50000.0
        assert portfolio.commission_rate == 0.002
        assert len(portfolio.positions) == 0
        assert len(portfolio.orders) == 0
        assert len(portfolio.trades) == 0
        assert portfolio.is_trading_enabled == True
    
    def test_place_market_order(self, portfolio):
        """Test de colocación de orden de mercado."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        assert order_id is not None
        assert order_id in portfolio.orders
        assert portfolio.orders[order_id].symbol == "SPY"
        assert portfolio.orders[order_id].side == PositionSide.LONG
        assert portfolio.orders[order_id].order_type == OrderType.MARKET
        assert portfolio.orders[order_id].quantity == 100.0
    
    def test_place_limit_order(self, portfolio):
        """Test de colocación de orden limit."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=400.0
        )
        
        assert order_id is not None
        assert portfolio.orders[order_id].order_type == OrderType.LIMIT
        assert portfolio.orders[order_id].price == 400.0
    
    def test_place_stop_order(self, portfolio):
        """Test de colocación de orden stop."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.STOP,
            quantity=100.0,
            stop_price=390.0
        )
        
        assert order_id is not None
        assert portfolio.orders[order_id].order_type == OrderType.STOP
        assert portfolio.orders[order_id].stop_price == 390.0
    
    def test_place_stop_limit_order(self, portfolio):
        """Test de colocación de orden stop-limit."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.STOP_LIMIT,
            quantity=100.0,
            price=400.0,
            stop_price=390.0
        )
        
        assert order_id is not None
        assert portfolio.orders[order_id].order_type == OrderType.STOP_LIMIT
        assert portfolio.orders[order_id].price == 400.0
        assert portfolio.orders[order_id].stop_price == 390.0
    
    def test_validate_order_insufficient_cash(self, portfolio):
        """Test de validación de orden con cash insuficiente."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=1000000.0  # Cantidad muy grande
        )
        
        order = portfolio.orders[order_id]
        assert order.status == OrderStatus.REJECTED
    
    def test_validate_order_insufficient_position(self, portfolio):
        """Test de validación de orden con posición insuficiente."""
        # Primero crear una posición
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=400.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        # Intentar vender más de lo que se tiene
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.SHORT,
            order_type=OrderType.MARKET,
            quantity=200.0  # Más de lo que se tiene
        )
        
        order = portfolio.orders[order_id]
        assert order.status == OrderStatus.REJECTED
    
    def test_execute_order_buy(self, portfolio):
        """Test de ejecución de orden de compra."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        # Ejecutar orden
        success = portfolio.execute_order(order_id, 400.0)
        
        assert success == True
        order = portfolio.orders[order_id]
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100.0
        assert order.filled_price == 400.0
        
        # Verificar posición
        assert "SPY" in portfolio.positions
        position = portfolio.positions["SPY"]
        assert position.quantity == 100.0
        assert position.side == PositionSide.LONG
        assert position.entry_price == 400.0
        
        # Verificar cash
        expected_cash = 100000.0 - (100.0 * 400.0 * 1.001)  # Incluyendo comisión
        assert abs(portfolio.cash - expected_cash) < 0.01
    
    def test_execute_order_sell(self, portfolio):
        """Test de ejecución de orden de venta."""
        # Primero crear una posición
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=400.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.SHORT,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        # Ejecutar orden
        success = portfolio.execute_order(order_id, 405.0)
        
        assert success == True
        order = portfolio.orders[order_id]
        assert order.status == OrderStatus.FILLED
        
        # Verificar que la posición se cerró
        assert "SPY" not in portfolio.positions
        
        # Verificar cash
        expected_cash = 100000.0 + (100.0 * 405.0 * 0.999)  # Incluyendo comisión
        assert abs(portfolio.cash - expected_cash) < 0.01
    
    def test_update_prices(self, portfolio):
        """Test de actualización de precios."""
        # Crear posición
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=400.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        # Actualizar precios
        portfolio.update_prices({"SPY": 405.0})
        
        position = portfolio.positions["SPY"]
        assert position.current_price == 405.0
        assert position.unrealized_pnl == 500.0  # (405 - 400) * 100
    
    def test_get_portfolio_value(self, portfolio):
        """Test de obtención de valor del portfolio."""
        # Crear posición
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=405.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        portfolio_value = portfolio.get_portfolio_value()
        expected_value = portfolio.cash + (100.0 * 405.0)
        assert portfolio_value == expected_value
    
    def test_get_total_pnl(self, portfolio):
        """Test de obtención de PnL total."""
        # Crear posición
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=405.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        total_pnl = portfolio.get_total_pnl()
        expected_pnl = 500.0 - portfolio.total_commission_paid  # PnL no realizado - comisiones
        assert total_pnl == expected_pnl
    
    def test_get_performance_metrics(self, portfolio):
        """Test de obtención de métricas de performance."""
        metrics = portfolio.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'current_value' in metrics
        assert 'cash' in metrics
        assert 'positions_value' in metrics
        assert 'total_pnl' in metrics
        assert 'realized_pnl' in metrics
        assert 'unrealized_pnl' in metrics
        assert 'commission_paid' in metrics
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'losing_trades' in metrics
        assert 'win_rate' in metrics
        assert 'avg_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'positions_count' in metrics
    
    def test_cancel_order(self, portfolio):
        """Test de cancelación de orden."""
        order_id = portfolio.place_order(
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        success = portfolio.cancel_order(order_id)
        
        assert success == True
        order = portfolio.orders[order_id]
        assert order.status == OrderStatus.CANCELLED
    
    def test_enable_disable_trading(self, portfolio):
        """Test de habilitación/deshabilitación de trading."""
        portfolio.disable_trading()
        assert portfolio.is_trading_enabled == False
        
        portfolio.enable_trading()
        assert portfolio.is_trading_enabled == True
    
    def test_reset_portfolio(self, portfolio):
        """Test de reset del portfolio."""
        # Crear algunas posiciones y órdenes
        portfolio.place_order("SPY", PositionSide.LONG, OrderType.MARKET, 100.0)
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=400.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        # Resetear portfolio
        portfolio.reset_portfolio(50000.0)
        
        assert portfolio.initial_cash == 50000.0
        assert portfolio.cash == 50000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.orders) == 0
        assert len(portfolio.trades) == 0
    
    def test_get_summary(self, portfolio):
        """Test de obtención de resumen."""
        summary = portfolio.get_summary()
        
        assert 'initial_cash' in summary
        assert 'current_value' in summary
        assert 'cash' in summary
        assert 'positions_count' in summary
        assert 'orders_count' in summary
        assert 'trades_count' in summary
        assert 'total_commission_paid' in summary
        assert 'trading_enabled' in summary
        assert 'last_update' in summary


class TestExecutionConfig:
    """Tests para ExecutionConfig."""
    
    def test_execution_config_creation(self):
        """Test de creación de ExecutionConfig."""
        config = ExecutionConfig(
            mode=ExecutionMode.REALISTIC,
            base_latency_ms=100.0,
            slippage_rate=0.0002
        )
        
        assert config.mode == ExecutionMode.REALISTIC
        assert config.base_latency_ms == 100.0
        assert config.slippage_rate == 0.0002


class TestExecutionResult:
    """Tests para ExecutionResult."""
    
    def test_execution_result_creation(self):
        """Test de creación de ExecutionResult."""
        result = ExecutionResult(
            order_id="test_order",
            execution_price=400.0,
            execution_quantity=100.0,
            execution_time=datetime.now(),
            slippage=0.0001,
            commission=0.4,
            latency_ms=50.0,
            partial_fills=0,
            success=True
        )
        
        assert result.order_id == "test_order"
        assert result.execution_price == 400.0
        assert result.execution_quantity == 100.0
        assert result.slippage == 0.0001
        assert result.commission == 0.4
        assert result.latency_ms == 50.0
        assert result.partial_fills == 0
        assert result.success == True


class TestPaperTradingExecutionEngine:
    """Tests para PaperTradingExecutionEngine."""
    
    @pytest.fixture
    def portfolio(self):
        """Portfolio de prueba."""
        return PaperTradingPortfolio(initial_cash=100000.0, commission_rate=0.001)
    
    @pytest.fixture
    def execution_engine(self, portfolio):
        """Motor de ejecución de prueba."""
        config = ExecutionConfig(mode=ExecutionMode.IMMEDIATE)
        return PaperTradingExecutionEngine(portfolio, config)
    
    def test_execution_engine_initialization(self, portfolio):
        """Test de inicialización del motor de ejecución."""
        config = ExecutionConfig()
        engine = PaperTradingExecutionEngine(portfolio, config)
        
        assert engine.portfolio == portfolio
        assert engine.config == config
        assert engine.is_running == False
        assert len(engine.execution_queue) == 0
        assert len(engine.execution_results) == 0
        assert engine.total_executions == 0
        assert engine.successful_executions == 0
        assert engine.failed_executions == 0
    
    def test_start_stop_engine(self, execution_engine):
        """Test de inicio y parada del motor."""
        execution_engine.start()
        assert execution_engine.is_running == True
        
        execution_engine.stop()
        assert execution_engine.is_running == False
    
    def test_update_prices(self, execution_engine):
        """Test de actualización de precios."""
        execution_engine.update_prices({"SPY": 400.0, "QQQ": 300.0})
        
        assert "SPY" in execution_engine.current_prices
        assert "QQQ" in execution_engine.current_prices
        assert execution_engine.current_prices["SPY"] == 400.0
        assert execution_engine.current_prices["QQQ"] == 300.0
    
    def test_update_volume(self, execution_engine):
        """Test de actualización de volumen."""
        execution_engine.update_volume("SPY", 1000000.0)
        
        assert "SPY" in execution_engine.volume_history
        assert len(execution_engine.volume_history["SPY"]) == 1
        assert execution_engine.volume_history["SPY"][0][1] == 1000000.0
    
    def test_should_execute_order_market(self, execution_engine):
        """Test de evaluación de orden de mercado."""
        execution_engine.update_prices({"SPY": 400.0})
        
        order = Order(
            order_id="test_order",
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        should_execute = execution_engine._should_execute_order(order)
        assert should_execute == True
    
    def test_should_execute_order_limit(self, execution_engine):
        """Test de evaluación de orden limit."""
        execution_engine.update_prices({"SPY": 400.0})
        
        # Orden limit de compra con precio por encima del mercado
        order = Order(
            order_id="test_order",
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=405.0
        )
        
        should_execute = execution_engine._should_execute_order(order)
        assert should_execute == True
        
        # Orden limit de compra con precio por debajo del mercado
        order.price = 395.0
        should_execute = execution_engine._should_execute_order(order)
        assert should_execute == False
    
    def test_should_execute_order_stop(self, execution_engine):
        """Test de evaluación de orden stop."""
        execution_engine.update_prices({"SPY": 400.0})
        
        # Orden stop de compra con precio por debajo del mercado
        order = Order(
            order_id="test_order",
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.STOP,
            quantity=100.0,
            stop_price=395.0
        )
        
        should_execute = execution_engine._should_execute_order(order)
        assert should_execute == False
        
        # Orden stop de compra con precio por encima del mercado
        order.stop_price = 405.0
        should_execute = execution_engine._should_execute_order(order)
        assert should_execute == True
    
    def test_calculate_latency(self, execution_engine):
        """Test de cálculo de latencia."""
        # Modo inmediato
        execution_engine.config.mode = ExecutionMode.IMMEDIATE
        latency = execution_engine._calculate_latency()
        assert latency == 0.0
        
        # Modo realista
        execution_engine.config.mode = ExecutionMode.REALISTIC
        latency = execution_engine._calculate_latency()
        assert latency >= 0.0
    
    def test_calculate_slippage(self, execution_engine):
        """Test de cálculo de slippage."""
        order = Order(
            order_id="test_order",
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        # Modo inmediato
        execution_engine.config.mode = ExecutionMode.IMMEDIATE
        slippage = execution_engine._calculate_slippage(order, 400.0)
        assert slippage == 0.0
        
        # Modo realista
        execution_engine.config.mode = ExecutionMode.REALISTIC
        slippage = execution_engine._calculate_slippage(order, 400.0)
        assert slippage >= 0.0  # Para órdenes de compra, slippage debe ser positivo
    
    def test_calculate_execution_quantity(self, execution_engine):
        """Test de cálculo de cantidad de ejecución."""
        order = Order(
            order_id="test_order",
            symbol="SPY",
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        # Modo inmediato
        execution_engine.config.mode = ExecutionMode.IMMEDIATE
        quantity = execution_engine._calculate_execution_quantity(order)
        assert quantity == 100.0
        
        # Modo realista
        execution_engine.config.mode = ExecutionMode.REALISTIC
        quantity = execution_engine._calculate_execution_quantity(order)
        assert 0.0 <= quantity <= 100.0
    
    def test_get_execution_stats(self, execution_engine):
        """Test de obtención de estadísticas de ejecución."""
        stats = execution_engine.get_execution_stats()
        
        assert 'total_executions' in stats
        assert 'successful_executions' in stats
        assert 'failed_executions' in stats
        assert 'success_rate' in stats
        assert 'avg_latency_ms' in stats
        assert 'avg_slippage' in stats
        assert 'is_running' in stats
        assert 'config' in stats
    
    def test_get_recent_executions(self, execution_engine):
        """Test de obtención de ejecuciones recientes."""
        executions = execution_engine.get_recent_executions(10)
        assert isinstance(executions, list)
    
    def test_get_summary(self, execution_engine):
        """Test de obtención de resumen."""
        summary = execution_engine.get_summary()
        
        assert 'is_running' in summary
        assert 'total_executions' in summary
        assert 'success_rate' in summary
        assert 'avg_latency_ms' in summary
        assert 'avg_slippage' in summary
        assert 'symbols_tracked' in summary
        assert 'config' in summary


class TestPaperTradingManager:
    """Tests para PaperTradingManager."""
    
    @pytest.fixture
    def manager(self):
        """Gestor de paper trading de prueba."""
        return PaperTradingManager()
    
    def test_manager_initialization(self):
        """Test de inicialización del gestor."""
        manager = PaperTradingManager()
        
        assert manager.portfolio is not None
        assert manager.execution_engine is not None
        assert manager.is_running == False
        assert manager.start_time is None
        assert manager.last_update is None
        assert len(manager.market_data) == 0
    
    def test_start_stop_trading(self, manager):
        """Test de inicio y parada de trading."""
        manager.start_trading()
        assert manager.is_running == True
        assert manager.start_time is not None
        
        manager.stop_trading()
        assert manager.is_running == False
    
    def test_place_market_order(self, manager):
        """Test de colocación de orden de mercado."""
        manager.start_trading()
        
        order_id = manager.place_market_order("SPY", PositionSide.LONG, 100.0)
        
        assert order_id is not None
        assert order_id in manager.portfolio.orders
    
    def test_place_limit_order(self, manager):
        """Test de colocación de orden limit."""
        manager.start_trading()
        
        order_id = manager.place_limit_order("SPY", PositionSide.LONG, 100.0, 400.0)
        
        assert order_id is not None
        order = manager.portfolio.orders[order_id]
        assert order.order_type == OrderType.LIMIT
        assert order.price == 400.0
    
    def test_place_stop_order(self, manager):
        """Test de colocación de orden stop."""
        manager.start_trading()
        
        order_id = manager.place_stop_order("SPY", PositionSide.LONG, 100.0, 390.0)
        
        assert order_id is not None
        order = manager.portfolio.orders[order_id]
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 390.0
    
    def test_place_stop_limit_order(self, manager):
        """Test de colocación de orden stop-limit."""
        manager.start_trading()
        
        order_id = manager.place_stop_limit_order("SPY", PositionSide.LONG, 100.0, 390.0, 400.0)
        
        assert order_id is not None
        order = manager.portfolio.orders[order_id]
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.stop_price == 390.0
        assert order.price == 400.0
    
    def test_cancel_order(self, manager):
        """Test de cancelación de orden."""
        manager.start_trading()
        
        order_id = manager.place_market_order("SPY", PositionSide.LONG, 100.0)
        success = manager.cancel_order(order_id)
        
        assert success == True
        order = manager.portfolio.orders[order_id]
        assert order.status == OrderStatus.CANCELLED
    
    def test_update_market_data(self, manager):
        """Test de actualización de datos de mercado."""
        manager.update_market_data("SPY", 400.0, 1000000.0)
        
        assert "SPY" in manager.market_data
        assert manager.market_data["SPY"]["price"] == 400.0
        assert manager.market_data["SPY"]["volume"] == 1000000.0
        assert manager.last_update is not None
    
    def test_get_portfolio_summary(self, manager):
        """Test de obtención de resumen del portfolio."""
        summary = manager.get_portfolio_summary()
        
        assert 'portfolio_value' in summary
        assert 'initial_cash' in summary
        assert 'cash' in summary
        assert 'positions' in summary
        assert 'metrics' in summary
        assert 'last_update' in summary
        assert 'trading_enabled' in summary
    
    def test_get_performance_metrics(self, manager):
        """Test de obtención de métricas de performance."""
        metrics = manager.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'current_value' in metrics
        assert 'cash' in metrics
        assert 'positions_value' in metrics
        assert 'total_pnl' in metrics
        assert 'realized_pnl' in metrics
        assert 'unrealized_pnl' in metrics
        assert 'commission_paid' in metrics
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'losing_trades' in metrics
        assert 'win_rate' in metrics
        assert 'avg_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'positions_count' in metrics
    
    def test_get_positions(self, manager):
        """Test de obtención de posiciones."""
        positions = manager.get_positions()
        assert isinstance(positions, dict)
    
    def test_get_orders(self, manager):
        """Test de obtención de órdenes."""
        orders = manager.get_orders()
        assert isinstance(orders, list)
        
        # Test con filtro de estado
        pending_orders = manager.get_orders(OrderStatus.PENDING)
        assert isinstance(pending_orders, list)
    
    def test_get_trades(self, manager):
        """Test de obtención de trades."""
        trades = manager.get_trades()
        assert isinstance(trades, list)
    
    def test_get_execution_stats(self, manager):
        """Test de obtención de estadísticas de ejecución."""
        stats = manager.get_execution_stats()
        
        assert 'total_executions' in stats
        assert 'successful_executions' in stats
        assert 'failed_executions' in stats
        assert 'success_rate' in stats
        assert 'avg_latency_ms' in stats
        assert 'avg_slippage' in stats
        assert 'is_running' in stats
        assert 'config' in stats
    
    def test_get_recent_executions(self, manager):
        """Test de obtención de ejecuciones recientes."""
        executions = manager.get_recent_executions(10)
        assert isinstance(executions, list)
    
    def test_create_performance_report(self, manager):
        """Test de creación de reporte de performance."""
        report = manager.create_performance_report()
        
        assert isinstance(report, str)
        assert "REPORTE DE PERFORMANCE - PAPER TRADING" in report
        assert "RESUMEN DEL PORTFOLIO" in report
        assert "MÉTRICAS DE PERFORMANCE" in report
        assert "ESTADÍSTICAS DE EJECUCIÓN" in report
    
    def test_export_data(self, manager, temp_dir):
        """Test de exportación de datos."""
        # Exportar a CSV
        csv_path = temp_dir + "/test_data.csv"
        manager.export_data(csv_path, "csv")
        
        # Verificar que se crearon archivos
        assert os.path.exists(csv_path.with_suffix('.trades.csv'))
        assert os.path.exists(csv_path.with_suffix('.orders.csv'))
        assert os.path.exists(csv_path.with_suffix('.executions.csv'))
        
        # Exportar a Excel
        excel_path = temp_dir + "/test_data.xlsx"
        manager.export_data(excel_path, "excel")
        assert os.path.exists(excel_path)
        
        # Exportar a JSON
        json_path = temp_dir + "/test_data.json"
        manager.export_data(json_path, "json")
        assert os.path.exists(json_path)
    
    def test_save_load_state(self, manager, temp_dir):
        """Test de guardado y carga de estado."""
        # Guardar estado
        save_path = temp_dir + "/test_state.pkl"
        manager.save_state(save_path)
        assert os.path.exists(save_path)
        
        # Cargar estado
        new_manager = PaperTradingManager()
        new_manager.load_state(save_path)
        
        # Verificar que se cargó el estado
        assert new_manager.portfolio.initial_cash == manager.portfolio.initial_cash
    
    def test_reset_portfolio(self, manager):
        """Test de reset del portfolio."""
        # Crear algunas posiciones
        manager.portfolio.positions["SPY"] = Position(
            symbol="SPY",
            quantity=100.0,
            side=PositionSide.LONG,
            entry_price=400.0,
            current_price=400.0,
            entry_time=datetime.now(),
            last_update=datetime.now()
        )
        
        # Resetear portfolio
        manager.reset_portfolio(50000.0)
        
        assert manager.portfolio.initial_cash == 50000.0
        assert manager.portfolio.cash == 50000.0
        assert len(manager.portfolio.positions) == 0
    
    def test_get_summary(self, manager):
        """Test de obtención de resumen."""
        summary = manager.get_summary()
        
        assert 'config' in summary
        assert 'is_running' in summary
        assert 'start_time' in summary
        assert 'last_update' in summary
        assert 'portfolio_summary' in summary
        assert 'execution_stats' in summary
        assert 'market_data_symbols' in summary


if __name__ == "__main__":
    pytest.main([__file__])

