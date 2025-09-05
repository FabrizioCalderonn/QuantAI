"""
Gestor principal de riesgo que coordina todos los componentes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib

from .base import BaseRiskManager, RiskLimits, RiskAlert, Position, RiskType, RiskLevel, RiskMetrics
from .volatility_targeting import VolatilityTargetingRiskManager
from .circuit_breakers import CircuitBreakerRiskManager
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Gestor principal de riesgo que coordina todos los componentes.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el gestor principal de riesgo.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.risk_config = self.config.get('risk_management', {})
        
        # Inicializar componentes de riesgo
        self.volatility_manager = VolatilityTargetingRiskManager(
            self.risk_config.get('volatility_targeting', {})
        )
        
        self.circuit_breaker_manager = CircuitBreakerRiskManager(
            self.risk_config.get('circuit_breakers', {})
        )
        
        # Estado del sistema
        self.positions = []
        self.alerts = []
        self.risk_metrics = {}
        self.portfolio_value = 0.0
        self.last_update = None
        
        # Configuración
        self.update_frequency = self.risk_config.get('update_frequency', 60)  # segundos
        self.alert_threshold = self.risk_config.get('alert_threshold', 0.8)  # 80% de límite
        self.auto_rebalance = self.risk_config.get('auto_rebalance', True)
        self.emergency_stop = self.risk_config.get('emergency_stop', True)
        
        logger.info("Risk Manager principal inicializado")
    
    def update_portfolio(self, positions: List[Position], 
                        current_prices: Dict[str, float],
                        returns: pd.Series = None) -> Dict[str, Any]:
        """
        Actualiza el portafolio y calcula métricas de riesgo.
        
        Args:
            positions: Lista de posiciones
            current_prices: Precios actuales
            returns: Serie de retornos (opcional)
            
        Returns:
            Diccionario con métricas de riesgo
        """
        logger.info(f"Actualizando portafolio con {len(positions)} posiciones")
        
        # Actualizar posiciones
        self.positions = positions.copy()
        
        # Calcular valor del portafolio
        self.portfolio_value = self._calculate_portfolio_value(current_prices)
        
        # Actualizar datos de mercado
        self._update_market_data(current_prices)
        
        # Calcular métricas de riesgo
        risk_metrics = self._calculate_risk_metrics(returns)
        
        # Verificar límites de riesgo
        alerts = self._check_risk_limits(returns)
        
        # Actualizar timestamp
        self.last_update = datetime.now()
        
        # Generar resumen
        summary = {
            'portfolio_value': self.portfolio_value,
            'positions_count': len(positions),
            'risk_metrics': risk_metrics,
            'alerts': [alert.to_dict() for alert in alerts],
            'timestamp': self.last_update.isoformat()
        }
        
        logger.info(f"Portafolio actualizado: ${self.portfolio_value:,.2f}, {len(alerts)} alertas")
        
        return summary
    
    def adjust_position_size(self, signal: float, symbol: str, 
                           current_price: float) -> float:
        """
        Ajusta el tamaño de posición considerando todos los factores de riesgo.
        
        Args:
            signal: Señal de trading (-1, 0, 1)
            symbol: Símbolo del instrumento
            current_price: Precio actual
            
        Returns:
            Tamaño de posición ajustado
        """
        if signal == 0 or self.portfolio_value <= 0:
            return 0.0
        
        # Verificar circuit breakers primero
        circuit_breaker_size = self.circuit_breaker_manager.adjust_position_size(
            signal, symbol, current_price, self.portfolio_value
        )
        
        if circuit_breaker_size == 0:
            logger.warning(f"Circuit breaker bloquea posición para {symbol}")
            return 0.0
        
        # Ajustar por volatility targeting
        volatility_size = self.volatility_manager.adjust_position_size(
            signal, symbol, current_price, self.portfolio_value
        )
        
        # Tomar el mínimo de ambos ajustes
        final_size = min(abs(circuit_breaker_size), abs(volatility_size))
        
        # Aplicar señal
        final_size *= signal
        
        # Verificar límites finales
        final_size = self._apply_final_limits(final_size, symbol, current_price)
        
        logger.info(f"Posición final para {symbol}: {final_size:.2f}")
        
        return final_size
    
    def check_risk_limits(self, returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica todos los límites de riesgo.
        
        Args:
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas de riesgo
        """
        all_alerts = []
        
        # Verificar límites de volatility targeting
        volatility_alerts = self.volatility_manager.check_risk_limits(
            self.positions, returns
        )
        all_alerts.extend(volatility_alerts)
        
        # Verificar circuit breakers
        circuit_breaker_alerts = self.circuit_breaker_manager.check_risk_limits(
            self.positions, returns
        )
        all_alerts.extend(circuit_breaker_alerts)
        
        # Verificar límites globales
        global_alerts = self._check_global_limits(returns)
        all_alerts.extend(global_alerts)
        
        # Agregar alertas al gestor principal
        for alert in all_alerts:
            self.add_alert(alert)
        
        return all_alerts
    
    def add_position(self, position: Position) -> None:
        """
        Agrega una posición al portafolio.
        
        Args:
            position: Posición a agregar
        """
        self.positions.append(position)
        
        # Agregar a gestores individuales
        self.volatility_manager.add_position(position)
        self.circuit_breaker_manager.add_position(position)
        
        logger.info(f"Posición agregada: {position.symbol} {position.quantity} @ {position.price}")
    
    def remove_position(self, symbol: str) -> None:
        """
        Remueve una posición del portafolio.
        
        Args:
            symbol: Símbolo de la posición a remover
        """
        self.positions = [p for p in self.positions if p.symbol != symbol]
        
        # Remover de gestores individuales
        self.volatility_manager.remove_position(symbol)
        self.circuit_breaker_manager.remove_position(symbol)
        
        logger.info(f"Posición removida: {symbol}")
    
    def get_positions(self, symbol: str = None) -> List[Position]:
        """
        Obtiene posiciones del portafolio.
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            Lista de posiciones
        """
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return self.positions.copy()
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """
        Obtiene el valor del portafolio.
        
        Args:
            current_prices: Precios actuales (opcional)
            
        Returns:
            Valor del portafolio
        """
        if current_prices:
            return self._calculate_portfolio_value(current_prices)
        return self.portfolio_value
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Obtiene métricas de riesgo actuales.
        
        Returns:
            Métricas de riesgo
        """
        if not self.risk_metrics:
            return RiskMetrics(
                volatility=0.0, var_95=0.0, var_99=0.0,
                cvar_95=0.0, cvar_99=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                beta=0.0, correlation=0.0, concentration_risk=0.0,
                leverage_ratio=0.0, liquidity_risk=0.0,
                timestamp=datetime.now()
            )
        
        return self.risk_metrics
    
    def get_alerts(self, risk_type: RiskType = None, 
                   risk_level: RiskLevel = None) -> List[RiskAlert]:
        """
        Obtiene alertas de riesgo.
        
        Args:
            risk_type: Tipo de riesgo (opcional)
            risk_level: Nivel de riesgo (opcional)
            
        Returns:
            Lista de alertas
        """
        alerts = self.alerts.copy()
        
        if risk_type:
            alerts = [a for a in alerts if a.risk_type == risk_type]
        
        if risk_level:
            alerts = [a for a in alerts if a.risk_level == risk_level]
        
        return alerts
    
    def add_alert(self, alert: RiskAlert) -> None:
        """
        Agrega una alerta de riesgo.
        
        Args:
            alert: Alerta de riesgo
        """
        self.alerts.append(alert)
        logger.warning(f"Alerta de riesgo: {alert.risk_type.value} - {alert.message}")
    
    def clear_alerts(self) -> None:
        """Limpia todas las alertas."""
        self.alerts.clear()
        self.volatility_manager.clear_alerts()
        self.circuit_breaker_manager.clear_alerts()
        logger.info("Todas las alertas de riesgo limpiadas")
    
    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                          current_prices: Dict[str, float]) -> List[Position]:
        """
        Rebalancea el portafolio.
        
        Args:
            target_weights: Pesos objetivo
            current_prices: Precios actuales
            
        Returns:
            Lista de posiciones rebalanceadas
        """
        if not self.auto_rebalance:
            logger.info("Auto-rebalance deshabilitado")
            return self.positions
        
        logger.info("Iniciando rebalance del portafolio")
        
        # Rebalancear usando volatility targeting
        rebalanced_positions = self.volatility_manager.rebalance_portfolio(
            self.positions, current_prices, target_weights
        )
        
        # Actualizar posiciones
        self.positions = rebalanced_positions
        
        # Actualizar gestores individuales
        for position in rebalanced_positions:
            self.circuit_breaker_manager.add_position(position)
        
        logger.info(f"Portafolio rebalanceado: {len(rebalanced_positions)} posiciones")
        
        return rebalanced_positions
    
    def emergency_stop(self) -> Dict[str, Any]:
        """
        Ejecuta parada de emergencia.
        
        Returns:
            Diccionario con resultado de la parada
        """
        if not self.emergency_stop:
            logger.warning("Emergency stop deshabilitado")
            return {'status': 'disabled'}
        
        logger.critical("EJECUTANDO PARADA DE EMERGENCIA")
        
        # Cerrar todas las posiciones
        closed_positions = []
        for position in self.positions:
            position.metadata = position.metadata or {}
            position.metadata['emergency_close'] = True
            position.metadata['emergency_timestamp'] = datetime.now().isoformat()
            closed_positions.append(position)
        
        # Limpiar posiciones
        self.positions.clear()
        
        # Limpiar alertas
        self.clear_alerts()
        
        # Agregar alerta de parada de emergencia
        emergency_alert = RiskAlert(
            risk_type=RiskType.OPERATIONAL,
            risk_level=RiskLevel.CRITICAL,
            message="Parada de emergencia ejecutada",
            value=1.0,
            threshold=1.0,
            timestamp=datetime.now(),
            action_required=True
        )
        self.add_alert(emergency_alert)
        
        result = {
            'status': 'executed',
            'closed_positions': len(closed_positions),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.critical(f"Parada de emergencia completada: {len(closed_positions)} posiciones cerradas")
        
        return result
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen completo de riesgo.
        
        Returns:
            Diccionario con resumen de riesgo
        """
        return {
            'portfolio': {
                'value': self.portfolio_value,
                'positions_count': len(self.positions),
                'last_update': self.last_update.isoformat() if self.last_update else None
            },
            'volatility_targeting': self.volatility_manager.get_risk_summary(),
            'circuit_breakers': self.circuit_breaker_manager.get_risk_summary(),
            'alerts': {
                'total': len(self.alerts),
                'by_type': self._count_alerts_by_type(),
                'by_level': self._count_alerts_by_level(),
                'action_required': len([a for a in self.alerts if a.action_required])
            },
            'risk_metrics': self.risk_metrics.__dict__ if self.risk_metrics else {}
        }
    
    def save_state(self, filepath: str) -> None:
        """
        Guarda el estado del gestor de riesgo.
        
        Args:
            filepath: Ruta del archivo
        """
        state = {
            'positions': self.positions,
            'alerts': self.alerts,
            'risk_metrics': self.risk_metrics,
            'portfolio_value': self.portfolio_value,
            'last_update': self.last_update,
            'config': self.config
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(state, filepath)
        logger.info(f"Estado del gestor de riesgo guardado: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Carga el estado del gestor de riesgo.
        
        Args:
            filepath: Ruta del archivo
        """
        state = joblib.load(filepath)
        
        self.positions = state.get('positions', [])
        self.alerts = state.get('alerts', [])
        self.risk_metrics = state.get('risk_metrics', {})
        self.portfolio_value = state.get('portfolio_value', 0.0)
        self.last_update = state.get('last_update')
        
        logger.info(f"Estado del gestor de riesgo cargado: {filepath}")
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calcula el valor del portafolio.
        
        Args:
            current_prices: Precios actuales
            
        Returns:
            Valor del portafolio
        """
        total_value = 0.0
        
        for position in self.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def _update_market_data(self, current_prices: Dict[str, float]) -> None:
        """
        Actualiza datos de mercado en los gestores.
        
        Args:
            current_prices: Precios actuales
        """
        for symbol, price in current_prices.items():
            # Actualizar en circuit breaker manager
            self.circuit_breaker_manager.update_market_data(symbol, price)
    
    def _calculate_risk_metrics(self, returns: pd.Series = None) -> RiskMetrics:
        """
        Calcula métricas de riesgo.
        
        Args:
            returns: Serie de retornos (opcional)
            
        Returns:
            Métricas de riesgo
        """
        if returns is None or len(returns) == 0:
            return RiskMetrics(
                volatility=0.0, var_95=0.0, var_99=0.0,
                cvar_95=0.0, cvar_99=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                beta=0.0, correlation=0.0, concentration_risk=0.0,
                leverage_ratio=0.0, liquidity_risk=0.0,
                timestamp=datetime.now()
            )
        
        # Calcular métricas básicas
        volatility = returns.std() * np.sqrt(252)
        var_95 = self.volatility_manager.calculate_var(returns, 0.95)
        var_99 = self.volatility_manager.calculate_var(returns, 0.99)
        cvar_95 = self.volatility_manager.calculate_cvar(returns, 0.95)
        cvar_99 = self.volatility_manager.calculate_cvar(returns, 0.99)
        max_drawdown = self.volatility_manager.calculate_max_drawdown(returns)
        sharpe_ratio = self.volatility_manager.calculate_sharpe_ratio(returns)
        sortino_ratio = self.volatility_manager.calculate_sortino_ratio(returns)
        calmar_ratio = self.volatility_manager.calculate_calmar_ratio(returns)
        
        # Calcular métricas específicas
        concentration_risk = self.volatility_manager._calculate_portfolio_concentration()
        leverage_ratio = self.volatility_manager._calculate_portfolio_leverage()
        liquidity_risk = self.volatility_manager._calculate_portfolio_liquidity()
        
        risk_metrics = RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            beta=0.0,  # Se calculará cuando se tenga referencia
            correlation=0.0,  # Se calculará cuando se tenga otra serie
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio,
            liquidity_risk=liquidity_risk,
            timestamp=datetime.now()
        )
        
        self.risk_metrics = risk_metrics
        return risk_metrics
    
    def _check_global_limits(self, returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica límites globales de riesgo.
        
        Args:
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        # Verificar límites de alerta temprana
        if returns is not None and len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            max_volatility = self.volatility_manager.risk_limits.max_portfolio_volatility
            
            if volatility > max_volatility * self.alert_threshold:
                alerts.append(RiskAlert(
                    risk_type=RiskType.MARKET,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Volatilidad ({volatility:.3f}) cerca del límite ({max_volatility:.3f})",
                    value=volatility,
                    threshold=max_volatility * self.alert_threshold,
                    timestamp=datetime.now(),
                    action_required=False
                ))
        
        return alerts
    
    def _apply_final_limits(self, position_size: float, symbol: str, 
                          current_price: float) -> float:
        """
        Aplica límites finales a la posición.
        
        Args:
            position_size: Tamaño de posición
            symbol: Símbolo del instrumento
            current_price: Precio actual
            
        Returns:
            Tamaño de posición con límites aplicados
        """
        # Verificar límite de posición máxima
        max_position_value = self.volatility_manager.risk_limits.max_position_size * self.portfolio_value
        max_position_size = max_position_value / current_price
        
        if abs(position_size) > max_position_size:
            position_size = max_position_size * np.sign(position_size)
        
        return position_size
    
    def _count_alerts_by_type(self) -> Dict[str, int]:
        """
        Cuenta alertas por tipo.
        
        Returns:
            Diccionario con conteo por tipo
        """
        counts = {}
        for alert in self.alerts:
            alert_type = alert.risk_type.value
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts
    
    def _count_alerts_by_level(self) -> Dict[str, int]:
        """
        Cuenta alertas por nivel.
        
        Returns:
            Diccionario con conteo por nivel
        """
        counts = {}
        for alert in self.alerts:
            alert_level = alert.risk_level.value
            counts[alert_level] = counts.get(alert_level, 0) + 1
        return counts

