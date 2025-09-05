"""
Portfolio de paper trading.
Maneja posiciones, cash, y métricas de performance en tiempo real.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Tipos de órdenes."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Estados de órdenes."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Lados de posición."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Orden de trading."""
    order_id: str
    symbol: str
    side: PositionSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Posición en el portafolio."""
    symbol: str
    quantity: float
    side: PositionSide
    entry_price: float
    current_price: float
    entry_time: datetime
    last_update: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Trade ejecutado."""
    trade_id: str
    symbol: str
    side: PositionSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    order_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaperTradingPortfolio:
    """
    Portfolio de paper trading.
    """
    
    def __init__(self, initial_cash: float = 100000.0, 
                 commission_rate: float = 0.001,
                 config: Dict[str, Any] = None):
        """
        Inicializa el portfolio de paper trading.
        
        Args:
            initial_cash: Cash inicial
            commission_rate: Tasa de comisión
            config: Configuración adicional
        """
        self.config = config or {}
        
        # Estado del portfolio
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        
        # Posiciones y órdenes
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Historial de performance
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        
        # Métricas
        self.total_commission_paid = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Estado
        self.last_update = datetime.now()
        self.is_trading_enabled = True
        
        logger.info(f"PaperTradingPortfolio inicializado con ${initial_cash:,.2f}")
    
    def place_order(self, symbol: str, side: PositionSide, 
                   order_type: OrderType, quantity: float,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Coloca una orden.
        
        Args:
            symbol: Símbolo del activo
            side: Lado de la posición
            order_type: Tipo de orden
            quantity: Cantidad
            price: Precio (para órdenes limit)
            stop_price: Precio de stop
            metadata: Metadatos adicionales
            
        Returns:
            ID de la orden
        """
        if not self.is_trading_enabled:
            raise ValueError("Trading no está habilitado")
        
        # Generar ID de orden
        order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.orders):06d}"
        
        # Crear orden
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        # Validar orden
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Orden rechazada: {order_id}")
        else:
            logger.info(f"Orden colocada: {order_id} - {side.value} {quantity} {symbol}")
        
        # Guardar orden
        self.orders[order_id] = order
        
        return order_id
    
    def _validate_order(self, order: Order) -> bool:
        """
        Valida una orden.
        
        Args:
            order: Orden a validar
            
        Returns:
            True si la orden es válida
        """
        # Validar cantidad
        if order.quantity <= 0:
            return False
        
        # Validar precio para órdenes limit
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False
        
        # Validar stop price para órdenes stop
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False
        
        # Validar cash para órdenes de compra
        if order.side == PositionSide.LONG:
            required_cash = order.quantity * (order.price or 0) * (1 + self.commission_rate)
            if required_cash > self.cash:
                return False
        
        # Validar posición para órdenes de venta
        if order.side == PositionSide.SHORT:
            if order.symbol not in self.positions or self.positions[order.symbol].quantity < order.quantity:
                return False
        
        return True
    
    def execute_order(self, order_id: str, execution_price: float) -> bool:
        """
        Ejecuta una orden.
        
        Args:
            order_id: ID de la orden
            execution_price: Precio de ejecución
            
        Returns:
            True si la orden fue ejecutada
        """
        if order_id not in self.orders:
            logger.error(f"Orden no encontrada: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Orden no está pendiente: {order_id}")
            return False
        
        # Calcular comisión
        commission = order.quantity * execution_price * self.commission_rate
        
        # Ejecutar orden
        if order.side == PositionSide.LONG:
            # Compra
            total_cost = order.quantity * execution_price + commission
            
            if total_cost > self.cash:
                logger.error(f"Cash insuficiente para ejecutar orden: {order_id}")
                order.status = OrderStatus.REJECTED
                return False
            
            self.cash -= total_cost
            self._update_position(order.symbol, order.quantity, execution_price, PositionSide.LONG)
            
        else:
            # Venta
            if order.symbol not in self.positions or self.positions[order.symbol].quantity < order.quantity:
                logger.error(f"Posición insuficiente para ejecutar orden: {order_id}")
                order.status = OrderStatus.REJECTED
                return False
            
            proceeds = order.quantity * execution_price - commission
            self.cash += proceeds
            self._update_position(order.symbol, -order.quantity, execution_price, PositionSide.SHORT)
        
        # Actualizar orden
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission
        
        # Crear trade
        trade = Trade(
            trade_id=f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.trades):06d}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.now(),
            commission=commission,
            order_id=order_id,
            metadata=order.metadata
        )
        
        self.trades.append(trade)
        self.total_commission_paid += commission
        self.total_trades += 1
        
        logger.info(f"Orden ejecutada: {order_id} - {order.quantity} {order.symbol} @ {execution_price:.2f}")
        
        return True
    
    def _update_position(self, symbol: str, quantity: float, 
                        price: float, side: PositionSide) -> None:
        """
        Actualiza una posición.
        
        Args:
            symbol: Símbolo del activo
            quantity: Cantidad (positiva para compra, negativa para venta)
            price: Precio de ejecución
            side: Lado de la posición
        """
        if symbol not in self.positions:
            # Crear nueva posición
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                side=side,
                entry_price=price,
                current_price=price,
                entry_time=datetime.now(),
                last_update=datetime.now()
            )
        else:
            # Actualizar posición existente
            position = self.positions[symbol]
            
            if position.side == side:
                # Misma dirección - promediar precio
                total_quantity = position.quantity + quantity
                if total_quantity != 0:
                    position.entry_price = (position.entry_price * position.quantity + price * quantity) / total_quantity
                position.quantity = total_quantity
            else:
                # Dirección opuesta - reducir posición
                if abs(quantity) >= abs(position.quantity):
                    # Cerrar posición completamente
                    realized_pnl = (price - position.entry_price) * position.quantity
                    if position.side == PositionSide.SHORT:
                        realized_pnl = -realized_pnl
                    
                    position.realized_pnl += realized_pnl
                    
                    # Crear nueva posición si queda cantidad
                    remaining_quantity = quantity + position.quantity
                    if remaining_quantity != 0:
                        position.quantity = remaining_quantity
                        position.entry_price = price
                        position.side = side
                        position.entry_time = datetime.now()
                    else:
                        # Eliminar posición
                        del self.positions[symbol]
                        return
                else:
                    # Reducir posición parcialmente
                    realized_pnl = (price - position.entry_price) * abs(quantity)
                    if position.side == PositionSide.SHORT:
                        realized_pnl = -realized_pnl
                    
                    position.realized_pnl += realized_pnl
                    position.quantity += quantity
            
            position.last_update = datetime.now()
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Actualiza precios de posiciones.
        
        Args:
            prices: Diccionario con precios actuales
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.last_update = datetime.now()
                
                # Calcular PnL no realizado
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - price) * position.quantity
        
        # Actualizar valor del portfolio
        self._update_portfolio_value()
    
    def _update_portfolio_value(self) -> None:
        """Actualiza el valor del portfolio."""
        # Calcular valor de posiciones
        positions_value = 0.0
        for position in self.positions.values():
            positions_value += position.current_price * position.quantity
        
        # Valor total del portfolio
        total_value = self.cash + positions_value
        
        # Guardar valor
        self.portfolio_values.append((datetime.now(), total_value))
        
        # Calcular retorno diario
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2][1]
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append((datetime.now(), daily_return))
    
    def get_portfolio_value(self) -> float:
        """
        Obtiene el valor actual del portfolio.
        
        Returns:
            Valor del portfolio
        """
        positions_value = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """
        Obtiene el PnL total.
        
        Returns:
            PnL total (realizado + no realizado)
        """
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return realized_pnl + unrealized_pnl - self.total_commission_paid
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Obtiene todas las posiciones.
        
        Returns:
            Diccionario con posiciones
        """
        return self.positions.copy()
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Obtiene órdenes.
        
        Args:
            status: Estado de órdenes (opcional)
            
        Returns:
            Lista de órdenes
        """
        if status is None:
            return list(self.orders.values())
        return [order for order in self.orders.values() if order.status == status]
    
    def get_trades(self) -> List[Trade]:
        """
        Obtiene todos los trades.
        
        Returns:
            Lista de trades
        """
        return self.trades.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Obtiene métricas de performance.
        
        Returns:
            Diccionario con métricas
        """
        if not self.portfolio_values:
            return {}
        
        # Métricas básicas
        initial_value = self.initial_cash
        current_value = self.get_portfolio_value()
        total_return = (current_value - initial_value) / initial_value
        
        # Métricas de trading
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Métricas de retorno
        if len(self.daily_returns) > 1:
            returns = [ret for _, ret in self.daily_returns]
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        else:
            avg_return = 0.0
            volatility = 0.0
            sharpe_ratio = 0.0
        
        # Drawdown
        max_drawdown = 0.0
        if len(self.portfolio_values) > 1:
            values = [val for _, val in self.portfolio_values]
            peak = values[0]
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'current_value': current_value,
            'cash': self.cash,
            'positions_value': current_value - self.cash,
            'total_pnl': self.get_total_pnl(),
            'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values()),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'commission_paid': self.total_commission_paid,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'positions_count': len(self.positions)
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del portfolio.
        
        Returns:
            Diccionario con resumen
        """
        metrics = self.get_performance_metrics()
        
        return {
            'portfolio_value': self.get_portfolio_value(),
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.positions.items()
            },
            'metrics': metrics,
            'last_update': self.last_update.isoformat(),
            'trading_enabled': self.is_trading_enabled
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            True si la orden fue cancelada
        """
        if order_id not in self.orders:
            logger.error(f"Orden no encontrada: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Orden no está pendiente: {order_id}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Orden cancelada: {order_id}")
        
        return True
    
    def enable_trading(self) -> None:
        """Habilita trading."""
        self.is_trading_enabled = True
        logger.info("Trading habilitado")
    
    def disable_trading(self) -> None:
        """Deshabilita trading."""
        self.is_trading_enabled = False
        logger.info("Trading deshabilitado")
    
    def reset_portfolio(self, initial_cash: Optional[float] = None) -> None:
        """
        Resetea el portfolio.
        
        Args:
            initial_cash: Cash inicial (opcional)
        """
        if initial_cash is not None:
            self.initial_cash = initial_cash
        
        self.cash = self.initial_cash
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.portfolio_values.clear()
        self.daily_returns.clear()
        
        self.total_commission_paid = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.last_update = datetime.now()
        
        logger.info(f"Portfolio reseteado con ${self.initial_cash:,.2f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del portfolio.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'initial_cash': self.initial_cash,
            'current_value': self.get_portfolio_value(),
            'cash': self.cash,
            'positions_count': len(self.positions),
            'orders_count': len(self.orders),
            'trades_count': len(self.trades),
            'total_commission_paid': self.total_commission_paid,
            'trading_enabled': self.is_trading_enabled,
            'last_update': self.last_update.isoformat()
        }

