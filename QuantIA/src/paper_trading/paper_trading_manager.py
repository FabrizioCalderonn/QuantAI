"""
Gestor principal de paper trading que coordina todos los componentes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings

from .portfolio import PaperTradingPortfolio, OrderType, PositionSide, OrderStatus
from .execution_engine import PaperTradingExecutionEngine, ExecutionConfig, ExecutionMode
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PaperTradingManager:
    """
    Gestor principal de paper trading que coordina todos los componentes.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el gestor de paper trading.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.paper_trading_config = self.config.get('paper_trading', {})
        
        # Configuración del portfolio
        initial_cash = self.paper_trading_config.get('initial_cash', 100000.0)
        commission_rate = self.paper_trading_config.get('commission_rate', 0.001)
        
        # Configuración de ejecución
        execution_config = ExecutionConfig(
            mode=ExecutionMode(self.paper_trading_config.get('execution_mode', 'realistic')),
            base_latency_ms=self.paper_trading_config.get('base_latency_ms', 50.0),
            latency_std_ms=self.paper_trading_config.get('latency_std_ms', 20.0),
            slippage_rate=self.paper_trading_config.get('slippage_rate', 0.0001),
            slippage_std=self.paper_trading_config.get('slippage_std', 0.00005),
            fill_probability=self.paper_trading_config.get('fill_probability', 0.95),
            partial_fill_probability=self.paper_trading_config.get('partial_fill_probability', 0.1),
            max_partial_fills=self.paper_trading_config.get('max_partial_fills', 3),
            market_impact_rate=self.paper_trading_config.get('market_impact_rate', 0.0002),
            volume_impact_threshold=self.paper_trading_config.get('volume_impact_threshold', 0.01)
        )
        
        # Inicializar componentes
        self.portfolio = PaperTradingPortfolio(initial_cash, commission_rate, self.paper_trading_config)
        self.execution_engine = PaperTradingExecutionEngine(self.portfolio, execution_config)
        
        # Estado del sistema
        self.is_running = False
        self.start_time = None
        self.last_update = None
        
        # Datos de mercado
        self.market_data = {}
        self.price_feed = None
        
        logger.info("PaperTradingManager inicializado")
    
    def start_trading(self) -> None:
        """Inicia el paper trading."""
        if self.is_running:
            logger.warning("Paper trading ya está corriendo")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        
        # Iniciar motor de ejecución
        self.execution_engine.start()
        
        logger.info("Paper trading iniciado")
    
    def stop_trading(self) -> None:
        """Detiene el paper trading."""
        if not self.is_running:
            logger.warning("Paper trading no está corriendo")
            return
        
        self.is_running = False
        
        # Detener motor de ejecución
        self.execution_engine.stop()
        
        logger.info("Paper trading detenido")
    
    def place_market_order(self, symbol: str, side: PositionSide, 
                          quantity: float, metadata: Dict[str, Any] = None) -> str:
        """
        Coloca una orden de mercado.
        
        Args:
            symbol: Símbolo del activo
            side: Lado de la posición
            quantity: Cantidad
            metadata: Metadatos adicionales
            
        Returns:
            ID de la orden
        """
        if not self.is_running:
            raise ValueError("Paper trading no está corriendo")
        
        order_id = self.portfolio.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            metadata=metadata
        )
        
        logger.info(f"Orden de mercado colocada: {order_id} - {side.value} {quantity} {symbol}")
        
        return order_id
    
    def place_limit_order(self, symbol: str, side: PositionSide, 
                         quantity: float, price: float,
                         metadata: Dict[str, Any] = None) -> str:
        """
        Coloca una orden limit.
        
        Args:
            symbol: Símbolo del activo
            side: Lado de la posición
            quantity: Cantidad
            price: Precio límite
            metadata: Metadatos adicionales
            
        Returns:
            ID de la orden
        """
        if not self.is_running:
            raise ValueError("Paper trading no está corriendo")
        
        order_id = self.portfolio.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            metadata=metadata
        )
        
        logger.info(f"Orden limit colocada: {order_id} - {side.value} {quantity} {symbol} @ {price}")
        
        return order_id
    
    def place_stop_order(self, symbol: str, side: PositionSide, 
                        quantity: float, stop_price: float,
                        metadata: Dict[str, Any] = None) -> str:
        """
        Coloca una orden stop.
        
        Args:
            symbol: Símbolo del activo
            side: Lado de la posición
            quantity: Cantidad
            stop_price: Precio de stop
            metadata: Metadatos adicionales
            
        Returns:
            ID de la orden
        """
        if not self.is_running:
            raise ValueError("Paper trading no está corriendo")
        
        order_id = self.portfolio.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_price,
            metadata=metadata
        )
        
        logger.info(f"Orden stop colocada: {order_id} - {side.value} {quantity} {symbol} @ {stop_price}")
        
        return order_id
    
    def place_stop_limit_order(self, symbol: str, side: PositionSide, 
                              quantity: float, stop_price: float, limit_price: float,
                              metadata: Dict[str, Any] = None) -> str:
        """
        Coloca una orden stop-limit.
        
        Args:
            symbol: Símbolo del activo
            side: Lado de la posición
            quantity: Cantidad
            stop_price: Precio de stop
            limit_price: Precio límite
            metadata: Metadatos adicionales
            
        Returns:
            ID de la orden
        """
        if not self.is_running:
            raise ValueError("Paper trading no está corriendo")
        
        order_id = self.portfolio.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LIMIT,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            metadata=metadata
        )
        
        logger.info(f"Orden stop-limit colocada: {order_id} - {side.value} {quantity} {symbol} @ {stop_price}/{limit_price}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            True si la orden fue cancelada
        """
        success = self.portfolio.cancel_order(order_id)
        
        if success:
            logger.info(f"Orden cancelada: {order_id}")
        else:
            logger.warning(f"No se pudo cancelar orden: {order_id}")
        
        return success
    
    def update_market_data(self, symbol: str, price: float, volume: float = None) -> None:
        """
        Actualiza datos de mercado.
        
        Args:
            symbol: Símbolo del activo
            price: Precio actual
            volume: Volumen (opcional)
        """
        # Actualizar precios
        self.execution_engine.update_prices({symbol: price})
        
        # Actualizar volumen si se proporciona
        if volume is not None:
            self.execution_engine.update_volume(symbol, volume)
        
        # Guardar datos de mercado
        self.market_data[symbol] = {
            'price': price,
            'volume': volume,
            'timestamp': datetime.now()
        }
        
        self.last_update = datetime.now()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del portfolio.
        
        Returns:
            Diccionario con resumen del portfolio
        """
        return self.portfolio.get_portfolio_summary()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Obtiene métricas de performance.
        
        Returns:
            Diccionario con métricas
        """
        return self.portfolio.get_performance_metrics()
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Obtiene posiciones actuales.
        
        Returns:
            Diccionario con posiciones
        """
        positions = self.portfolio.get_positions()
        
        return {
            symbol: {
                'quantity': pos.quantity,
                'side': pos.side.value,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'entry_time': pos.entry_time.isoformat(),
                'last_update': pos.last_update.isoformat()
            }
            for symbol, pos in positions.items()
        }
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """
        Obtiene órdenes.
        
        Args:
            status: Estado de órdenes (opcional)
            
        Returns:
            Lista de órdenes
        """
        orders = self.portfolio.get_orders(status)
        
        return [
            {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'filled_price': order.filled_price,
                'commission': order.commission,
                'timestamp': order.timestamp.isoformat(),
                'metadata': order.metadata
            }
            for order in orders
        ]
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Obtiene todos los trades.
        
        Returns:
            Lista de trades
        """
        trades = self.portfolio.get_trades()
        
        return [
            {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'order_id': trade.order_id,
                'timestamp': trade.timestamp.isoformat(),
                'metadata': trade.metadata
            }
            for trade in trades
        ]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de ejecución.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.execution_engine.get_execution_stats()
    
    def get_recent_executions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene ejecuciones recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de ejecuciones recientes
        """
        executions = self.execution_engine.get_recent_executions(limit)
        
        return [
            {
                'order_id': exec_result.order_id,
                'execution_price': exec_result.execution_price,
                'execution_quantity': exec_result.execution_quantity,
                'execution_time': exec_result.execution_time.isoformat(),
                'slippage': exec_result.slippage,
                'commission': exec_result.commission,
                'latency_ms': exec_result.latency_ms,
                'partial_fills': exec_result.partial_fills,
                'success': exec_result.success,
                'error_message': exec_result.error_message
            }
            for exec_result in executions
        ]
    
    def create_performance_report(self) -> str:
        """
        Crea reporte de performance.
        
        Returns:
            Reporte en formato texto
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE PERFORMANCE - PAPER TRADING")
        report.append("=" * 80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Tiempo de ejecución: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}")
        report.append("")
        
        # Resumen del portfolio
        portfolio_summary = self.get_portfolio_summary()
        report.append("RESUMEN DEL PORTFOLIO:")
        report.append("-" * 40)
        report.append(f"Valor inicial: ${portfolio_summary['initial_cash']:,.2f}")
        report.append(f"Valor actual: ${portfolio_summary['portfolio_value']:,.2f}")
        report.append(f"Cash: ${portfolio_summary['cash']:,.2f}")
        report.append(f"Valor de posiciones: ${portfolio_summary['portfolio_value'] - portfolio_summary['cash']:,.2f}")
        report.append("")
        
        # Métricas de performance
        metrics = self.get_performance_metrics()
        report.append("MÉTRICAS DE PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Retorno total: {metrics.get('total_return', 0):.2%}")
        report.append(f"PnL total: ${metrics.get('total_pnl', 0):,.2f}")
        report.append(f"PnL realizado: ${metrics.get('realized_pnl', 0):,.2f}")
        report.append(f"PnL no realizado: ${metrics.get('unrealized_pnl', 0):,.2f}")
        report.append(f"Comisiones pagadas: ${metrics.get('commission_paid', 0):,.2f}")
        report.append(f"Total de trades: {metrics.get('total_trades', 0)}")
        report.append(f"Trades ganadores: {metrics.get('winning_trades', 0)}")
        report.append(f"Trades perdedores: {metrics.get('losing_trades', 0)}")
        report.append(f"Win rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"Retorno promedio: {metrics.get('avg_return', 0):.2%}")
        report.append(f"Volatilidad: {metrics.get('volatility', 0):.2%}")
        report.append(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Maximum drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append("")
        
        # Posiciones actuales
        positions = self.get_positions()
        if positions:
            report.append("POSICIONES ACTUALES:")
            report.append("-" * 40)
            for symbol, pos in positions.items():
                report.append(f"{symbol}: {pos['quantity']} {pos['side']} @ {pos['entry_price']:.2f} (PnL: ${pos['unrealized_pnl']:,.2f})")
            report.append("")
        
        # Estadísticas de ejecución
        exec_stats = self.get_execution_stats()
        report.append("ESTADÍSTICAS DE EJECUCIÓN:")
        report.append("-" * 40)
        report.append(f"Total de ejecuciones: {exec_stats['total_executions']}")
        report.append(f"Ejecuciones exitosas: {exec_stats['successful_executions']}")
        report.append(f"Ejecuciones fallidas: {exec_stats['failed_executions']}")
        report.append(f"Tasa de éxito: {exec_stats['success_rate']:.2%}")
        report.append(f"Latencia promedio: {exec_stats['avg_latency_ms']:.2f} ms")
        report.append(f"Slippage promedio: {exec_stats['avg_slippage']:.4f}")
        report.append("")
        
        # Configuración
        report.append("CONFIGURACIÓN:")
        report.append("-" * 40)
        report.append(f"Modo de ejecución: {exec_stats['config']['mode']}")
        report.append(f"Latencia base: {exec_stats['config']['base_latency_ms']} ms")
        report.append(f"Tasa de slippage: {exec_stats['config']['slippage_rate']:.4f}")
        report.append(f"Probabilidad de llenado: {exec_stats['config']['fill_probability']:.2%}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_data(self, filepath: str, format: str = 'csv') -> None:
        """
        Exporta datos del paper trading.
        
        Args:
            filepath: Ruta del archivo
            format: Formato de exportación ('csv', 'excel', 'json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            self._export_to_csv(filepath)
        elif format == 'excel':
            self._export_to_excel(filepath)
        elif format == 'json':
            self._export_to_json(filepath)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Datos de paper trading exportados: {filepath}")
    
    def _export_to_csv(self, filepath: Path) -> None:
        """
        Exporta datos a CSV.
        
        Args:
            filepath: Ruta del archivo
        """
        # Exportar trades
        trades = self.get_trades()
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(filepath.with_suffix('.trades.csv'), index=False)
        
        # Exportar órdenes
        orders = self.get_orders()
        if orders:
            orders_df = pd.DataFrame(orders)
            orders_df.to_csv(filepath.with_suffix('.orders.csv'), index=False)
        
        # Exportar ejecuciones
        executions = self.get_recent_executions()
        if executions:
            executions_df = pd.DataFrame(executions)
            executions_df.to_csv(filepath.with_suffix('.executions.csv'), index=False)
    
    def _export_to_excel(self, filepath: Path) -> None:
        """
        Exporta datos a Excel.
        
        Args:
            filepath: Ruta del archivo
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Trades
            trades = self.get_trades()
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            # Órdenes
            orders = self.get_orders()
            if orders:
                orders_df = pd.DataFrame(orders)
                orders_df.to_excel(writer, sheet_name='Orders', index=False)
            
            # Ejecuciones
            executions = self.get_recent_executions()
            if executions:
                executions_df = pd.DataFrame(executions)
                executions_df.to_excel(writer, sheet_name='Executions', index=False)
            
            # Métricas
            metrics = self.get_performance_metrics()
            if metrics:
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    
    def _export_to_json(self, filepath: Path) -> None:
        """
        Exporta datos a JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        import json
        
        data = {
            'portfolio_summary': self.get_portfolio_summary(),
            'performance_metrics': self.get_performance_metrics(),
            'positions': self.get_positions(),
            'orders': self.get_orders(),
            'trades': self.get_trades(),
            'execution_stats': self.get_execution_stats(),
            'recent_executions': self.get_recent_executions(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_state(self, filepath: str) -> None:
        """
        Guarda estado del paper trading.
        
        Args:
            filepath: Ruta del archivo
        """
        state = {
            'portfolio': self.portfolio,
            'execution_engine': self.execution_engine,
            'is_running': self.is_running,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'market_data': self.market_data,
            'config': self.config
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(state, filepath)
        logger.info(f"Estado de paper trading guardado: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Carga estado del paper trading.
        
        Args:
            filepath: Ruta del archivo
        """
        state = joblib.load(filepath)
        
        self.portfolio = state.get('portfolio', self.portfolio)
        self.execution_engine = state.get('execution_engine', self.execution_engine)
        self.is_running = state.get('is_running', False)
        self.start_time = state.get('start_time')
        self.last_update = state.get('last_update')
        self.market_data = state.get('market_data', {})
        
        logger.info(f"Estado de paper trading cargado: {filepath}")
    
    def reset_portfolio(self, initial_cash: Optional[float] = None) -> None:
        """
        Resetea el portfolio.
        
        Args:
            initial_cash: Cash inicial (opcional)
        """
        self.portfolio.reset_portfolio(initial_cash)
        logger.info("Portfolio reseteado")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del gestor de paper trading.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'portfolio_summary': self.portfolio.get_summary(),
            'execution_stats': self.execution_engine.get_summary(),
            'market_data_symbols': list(self.market_data.keys())
        }

