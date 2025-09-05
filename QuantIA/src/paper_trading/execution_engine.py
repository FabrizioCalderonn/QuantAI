"""
Motor de ejecución de paper trading.
Simula ejecución de órdenes con slippage, latencia y otros factores realistas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import asyncio
import threading
import time

from .portfolio import PaperTradingPortfolio, Order, OrderType, OrderStatus, PositionSide

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Modos de ejecución."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    REALISTIC = "realistic"


@dataclass
class ExecutionConfig:
    """Configuración de ejecución."""
    mode: ExecutionMode = ExecutionMode.REALISTIC
    base_latency_ms: float = 50.0
    latency_std_ms: float = 20.0
    slippage_rate: float = 0.0001
    slippage_std: float = 0.00005
    fill_probability: float = 0.95
    partial_fill_probability: float = 0.1
    max_partial_fills: int = 3
    market_impact_rate: float = 0.0002
    volume_impact_threshold: float = 0.01


@dataclass
class ExecutionResult:
    """Resultado de ejecución."""
    order_id: str
    execution_price: float
    execution_quantity: float
    execution_time: datetime
    slippage: float
    commission: float
    latency_ms: float
    partial_fills: int
    success: bool
    error_message: Optional[str] = None


class PaperTradingExecutionEngine:
    """
    Motor de ejecución de paper trading.
    """
    
    def __init__(self, portfolio: PaperTradingPortfolio, 
                 config: ExecutionConfig = None):
        """
        Inicializa el motor de ejecución.
        
        Args:
            portfolio: Portfolio de paper trading
            config: Configuración de ejecución
        """
        self.portfolio = portfolio
        self.config = config or ExecutionConfig()
        
        # Estado del motor
        self.is_running = False
        self.execution_thread = None
        self.execution_queue = []
        self.execution_results = []
        
        # Datos de mercado
        self.current_prices = {}
        self.price_history = {}
        self.volume_history = {}
        
        # Estadísticas
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.avg_latency_ms = 0.0
        self.avg_slippage = 0.0
        
        logger.info("PaperTradingExecutionEngine inicializado")
    
    def start(self) -> None:
        """Inicia el motor de ejecución."""
        if self.is_running:
            logger.warning("Motor de ejecución ya está corriendo")
            return
        
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        logger.info("Motor de ejecución iniciado")
    
    def stop(self) -> None:
        """Detiene el motor de ejecución."""
        if not self.is_running:
            logger.warning("Motor de ejecución no está corriendo")
            return
        
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5.0)
        
        logger.info("Motor de ejecución detenido")
    
    def _execution_loop(self) -> None:
        """Loop principal de ejecución."""
        while self.is_running:
            try:
                # Procesar órdenes pendientes
                self._process_pending_orders()
                
                # Actualizar precios
                self._update_prices()
                
                # Dormir un poco
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error en loop de ejecución: {str(e)}")
                time.sleep(1.0)
    
    def _process_pending_orders(self) -> None:
        """Procesa órdenes pendientes."""
        pending_orders = self.portfolio.get_orders(OrderStatus.PENDING)
        
        for order in pending_orders:
            try:
                # Verificar si la orden debe ejecutarse
                if self._should_execute_order(order):
                    # Ejecutar orden
                    result = self._execute_order(order)
                    self.execution_results.append(result)
                    
                    if result.success:
                        # Actualizar portfolio
                        self.portfolio.execute_order(order.order_id, result.execution_price)
                        self.successful_executions += 1
                    else:
                        # Marcar orden como rechazada
                        order.status = OrderStatus.REJECTED
                        self.failed_executions += 1
                    
                    self.total_executions += 1
                    
            except Exception as e:
                logger.error(f"Error procesando orden {order.order_id}: {str(e)}")
                order.status = OrderStatus.REJECTED
                self.failed_executions += 1
    
    def _should_execute_order(self, order: Order) -> bool:
        """
        Determina si una orden debe ejecutarse.
        
        Args:
            order: Orden a evaluar
            
        Returns:
            True si la orden debe ejecutarse
        """
        if order.symbol not in self.current_prices:
            return False
        
        current_price = self.current_prices[order.symbol]
        
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == PositionSide.LONG:
                return current_price <= order.price
            else:
                return current_price >= order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == PositionSide.LONG:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == PositionSide.LONG:
                return current_price >= order.stop_price and current_price <= order.price
            else:
                return current_price <= order.stop_price and current_price >= order.price
        
        return False
    
    def _execute_order(self, order: Order) -> ExecutionResult:
        """
        Ejecuta una orden.
        
        Args:
            order: Orden a ejecutar
            
        Returns:
            Resultado de ejecución
        """
        start_time = time.time()
        
        try:
            # Obtener precio actual
            current_price = self.current_prices[order.symbol]
            
            # Calcular latencia
            latency_ms = self._calculate_latency()
            
            # Simular latencia
            if self.config.mode == ExecutionMode.REALISTIC:
                time.sleep(latency_ms / 1000.0)
            
            # Calcular slippage
            slippage = self._calculate_slippage(order, current_price)
            
            # Calcular precio de ejecución
            execution_price = current_price * (1 + slippage)
            
            # Calcular cantidad de ejecución
            execution_quantity = self._calculate_execution_quantity(order)
            
            # Calcular comisión
            commission = execution_quantity * execution_price * self.portfolio.commission_rate
            
            # Crear resultado
            result = ExecutionResult(
                order_id=order.order_id,
                execution_price=execution_price,
                execution_quantity=execution_quantity,
                execution_time=datetime.now(),
                slippage=slippage,
                commission=commission,
                latency_ms=latency_ms,
                partial_fills=0,
                success=True
            )
            
            # Actualizar estadísticas
            self._update_execution_stats(result)
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                execution_price=0.0,
                execution_quantity=0.0,
                execution_time=datetime.now(),
                slippage=0.0,
                commission=0.0,
                latency_ms=0.0,
                partial_fills=0,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_latency(self) -> float:
        """
        Calcula latencia de ejecución.
        
        Returns:
            Latencia en milisegundos
        """
        if self.config.mode == ExecutionMode.IMMEDIATE:
            return 0.0
        
        # Latencia con distribución normal
        latency = np.random.normal(self.config.base_latency_ms, self.config.latency_std_ms)
        return max(0.0, latency)
    
    def _calculate_slippage(self, order: Order, current_price: float) -> float:
        """
        Calcula slippage de ejecución.
        
        Args:
            order: Orden a ejecutar
            current_price: Precio actual
            
        Returns:
            Slippage como fracción del precio
        """
        if self.config.mode == ExecutionMode.IMMEDIATE:
            return 0.0
        
        # Slippage base
        base_slippage = np.random.normal(self.config.slippage_rate, self.config.slippage_std)
        
        # Impacto de mercado
        market_impact = self._calculate_market_impact(order, current_price)
        
        # Slippage total
        total_slippage = base_slippage + market_impact
        
        # Asegurar que el slippage sea positivo para órdenes de compra
        if order.side == PositionSide.LONG:
            total_slippage = max(0.0, total_slippage)
        else:
            total_slippage = min(0.0, total_slippage)
        
        return total_slippage
    
    def _calculate_market_impact(self, order: Order, current_price: float) -> float:
        """
        Calcula impacto de mercado.
        
        Args:
            order: Orden a ejecutar
            current_price: Precio actual
            
        Returns:
            Impacto de mercado como fracción del precio
        """
        if order.symbol not in self.volume_history:
            return 0.0
        
        # Obtener volumen reciente
        recent_volume = self._get_recent_volume(order.symbol)
        
        if recent_volume == 0:
            return 0.0
        
        # Calcular ratio de volumen
        volume_ratio = order.quantity / recent_volume
        
        # Impacto de mercado basado en volumen
        if volume_ratio > self.config.volume_impact_threshold:
            impact = self.config.market_impact_rate * (volume_ratio / self.config.volume_impact_threshold)
            return impact
        
        return 0.0
    
    def _calculate_execution_quantity(self, order: Order) -> float:
        """
        Calcula cantidad de ejecución.
        
        Args:
            order: Orden a ejecutar
            
        Returns:
            Cantidad de ejecución
        """
        # Verificar probabilidad de llenado completo
        if np.random.random() < self.config.fill_probability:
            return order.quantity
        
        # Verificar probabilidad de llenado parcial
        if np.random.random() < self.config.partial_fill_probability:
            # Llenado parcial
            partial_ratio = np.random.uniform(0.3, 0.8)
            return order.quantity * partial_ratio
        
        # No llenado
        return 0.0
    
    def _get_recent_volume(self, symbol: str, minutes: int = 5) -> float:
        """
        Obtiene volumen reciente.
        
        Args:
            symbol: Símbolo del activo
            minutes: Minutos hacia atrás
            
        Returns:
            Volumen reciente
        """
        if symbol not in self.volume_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_volume = 0.0
        
        for timestamp, volume in self.volume_history[symbol]:
            if timestamp >= cutoff_time:
                recent_volume += volume
        
        return recent_volume
    
    def _update_prices(self) -> None:
        """Actualiza precios de mercado."""
        # En un sistema real, esto vendría de un feed de datos
        # Por ahora, simulamos actualizaciones de precio
        
        for symbol in self.current_prices:
            # Simular movimiento de precio
            current_price = self.current_prices[symbol]
            change = np.random.normal(0, 0.001)  # 0.1% de volatilidad
            new_price = current_price * (1 + change)
            
            self.current_prices[symbol] = new_price
            
            # Guardar historial
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append((datetime.now(), new_price))
            
            # Limpiar historial antiguo
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.price_history[symbol] = [
                (ts, price) for ts, price in self.price_history[symbol]
                if ts >= cutoff_time
            ]
    
    def _update_execution_stats(self, result: ExecutionResult) -> None:
        """
        Actualiza estadísticas de ejecución.
        
        Args:
            result: Resultado de ejecución
        """
        # Actualizar latencia promedio
        if self.total_executions > 0:
            self.avg_latency_ms = (self.avg_latency_ms * (self.total_executions - 1) + result.latency_ms) / self.total_executions
        else:
            self.avg_latency_ms = result.latency_ms
        
        # Actualizar slippage promedio
        if self.total_executions > 0:
            self.avg_slippage = (self.avg_slippage * (self.total_executions - 1) + abs(result.slippage)) / self.total_executions
        else:
            self.avg_slippage = abs(result.slippage)
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Actualiza precios de mercado.
        
        Args:
            prices: Diccionario con precios actuales
        """
        self.current_prices.update(prices)
        
        # Actualizar portfolio
        self.portfolio.update_prices(prices)
    
    def update_volume(self, symbol: str, volume: float) -> None:
        """
        Actualiza volumen de mercado.
        
        Args:
            symbol: Símbolo del activo
            volume: Volumen
        """
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        self.volume_history[symbol].append((datetime.now(), volume))
        
        # Limpiar historial antiguo
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.volume_history[symbol] = [
            (ts, vol) for ts, vol in self.volume_history[symbol]
            if ts >= cutoff_time
        ]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de ejecución.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0,
            'avg_latency_ms': self.avg_latency_ms,
            'avg_slippage': self.avg_slippage,
            'is_running': self.is_running,
            'config': {
                'mode': self.config.mode.value,
                'base_latency_ms': self.config.base_latency_ms,
                'slippage_rate': self.config.slippage_rate,
                'fill_probability': self.config.fill_probability
            }
        }
    
    def get_recent_executions(self, limit: int = 100) -> List[ExecutionResult]:
        """
        Obtiene ejecuciones recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de ejecuciones recientes
        """
        return self.execution_results[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del motor de ejecución.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'is_running': self.is_running,
            'total_executions': self.total_executions,
            'success_rate': self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0,
            'avg_latency_ms': self.avg_latency_ms,
            'avg_slippage': self.avg_slippage,
            'symbols_tracked': len(self.current_prices),
            'config': self.config.__dict__
        }

