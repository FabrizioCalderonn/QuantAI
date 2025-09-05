"""
Sistema de circuit breakers para gestión de riesgo.
Implementa stops dinámicos y circuit breakers automáticos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base import BaseRiskManager, RiskLimits, RiskAlert, Position, RiskType, RiskLevel, RiskMetrics

logger = logging.getLogger(__name__)


class CircuitBreakerType(Enum):
    """Tipos de circuit breakers."""
    PRICE = "price"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"


class CircuitBreakerStatus(Enum):
    """Estados de circuit breakers."""
    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"
    HALTED = "halted"


@dataclass
class CircuitBreakerRule:
    """Regla de circuit breaker."""
    name: str
    breaker_type: CircuitBreakerType
    threshold: float
    duration: int  # minutos
    action: str  # 'halt', 'reduce', 'alert'
    severity: RiskLevel
    enabled: bool = True
    cooldown: int = 60  # minutos de cooldown


@dataclass
class CircuitBreakerState:
    """Estado de un circuit breaker."""
    rule: CircuitBreakerRule
    status: CircuitBreakerStatus
    triggered_at: Optional[datetime] = None
    last_check: Optional[datetime] = None
    trigger_count: int = 0
    cooldown_until: Optional[datetime] = None


class CircuitBreakerRiskManager(BaseRiskManager):
    """
    Gestor de riesgo con circuit breakers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el gestor de circuit breakers.
        
        Args:
            config: Configuración del gestor
        """
        default_config = {
            'max_position_size': 0.1,
            'max_portfolio_volatility': 0.20,
            'max_drawdown': 0.10,
            'max_leverage': 2.0,
            'max_concentration': 0.3,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15,
            'circuit_breakers': {
                'price_drop': {
                    'threshold': 0.05,  # 5% caída de precio
                    'duration': 15,     # 15 minutos
                    'action': 'halt',
                    'severity': 'high'
                },
                'volatility_spike': {
                    'threshold': 0.30,  # 30% volatilidad
                    'duration': 30,     # 30 minutos
                    'action': 'reduce',
                    'severity': 'medium'
                },
                'volume_drop': {
                    'threshold': 0.5,   # 50% caída de volumen
                    'duration': 10,     # 10 minutos
                    'action': 'alert',
                    'severity': 'low'
                },
                'drawdown_limit': {
                    'threshold': 0.08,  # 8% drawdown
                    'duration': 60,     # 60 minutos
                    'action': 'halt',
                    'severity': 'critical'
                },
                'correlation_spike': {
                    'threshold': 0.9,   # 90% correlación
                    'duration': 20,     # 20 minutos
                    'action': 'reduce',
                    'severity': 'medium'
                },
                'liquidity_crisis': {
                    'threshold': 0.8,   # 80% riesgo de liquidez
                    'duration': 45,     # 45 minutos
                    'action': 'halt',
                    'severity': 'high'
                }
            },
            'position_sizing': {
                'base_size': 0.05,      # 5% tamaño base
                'max_size': 0.15,       # 15% tamaño máximo
                'volatility_adjustment': True,
                'correlation_adjustment': True,
                'liquidity_adjustment': True
            },
            'stop_loss': {
                'enabled': True,
                'trailing': True,
                'trailing_pct': 0.02,   # 2% trailing
                'min_profit_pct': 0.01, # 1% ganancia mínima para trailing
                'max_loss_pct': 0.08    # 8% pérdida máxima
            },
            'take_profit': {
                'enabled': True,
                'levels': [0.10, 0.20, 0.30],  # 10%, 20%, 30%
                'partial_close': True,
                'close_pct': [0.33, 0.33, 0.34]  # 33%, 33%, 34%
            }
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("circuit_breakers", config)
        
        # Configuración específica
        self.circuit_breakers_config = config['circuit_breakers']
        self.position_sizing_config = config['position_sizing']
        self.stop_loss_config = config['stop_loss']
        self.take_profit_config = config['take_profit']
        
        # Inicializar circuit breakers
        self.circuit_breakers = self._initialize_circuit_breakers()
        
        # Estado interno
        self.breach_history = []
        self.position_history = []
        self.volatility_history = {}
        self.price_history = {}
        self.volume_history = {}
        self.correlation_history = {}
        
        logger.info(f"Circuit Breaker Risk Manager inicializado con {len(self.circuit_breakers)} circuit breakers")
    
    def _create_risk_limits(self) -> RiskLimits:
        """
        Crea límites de riesgo para circuit breakers.
        
        Returns:
            Límites de riesgo
        """
        return RiskLimits(
            max_position_size=self.config.get('max_position_size', 0.1),
            max_portfolio_volatility=self.config.get('max_portfolio_volatility', 0.20),
            max_drawdown=self.config.get('max_drawdown', 0.10),
            max_leverage=self.config.get('max_leverage', 2.0),
            max_concentration=self.config.get('max_concentration', 0.3),
            max_correlation=self.config.get('max_correlation', 0.7),
            var_limit_95=self.config.get('var_limit_95', -0.05),
            var_limit_99=self.config.get('var_limit_99', -0.10),
            stop_loss_pct=self.config.get('stop_loss_pct', 0.05),
            take_profit_pct=self.config.get('take_profit_pct', 0.15)
        )
    
    def _initialize_circuit_breakers(self) -> Dict[str, CircuitBreakerState]:
        """
        Inicializa circuit breakers basados en configuración.
        
        Returns:
            Diccionario con circuit breakers
        """
        circuit_breakers = {}
        
        for name, config in self.circuit_breakers_config.items():
            # Mapear tipo
            type_mapping = {
                'price_drop': CircuitBreakerType.PRICE,
                'volatility_spike': CircuitBreakerType.VOLATILITY,
                'volume_drop': CircuitBreakerType.VOLUME,
                'drawdown_limit': CircuitBreakerType.DRAWDOWN,
                'correlation_spike': CircuitBreakerType.CORRELATION,
                'liquidity_crisis': CircuitBreakerType.LIQUIDITY
            }
            
            # Mapear severidad
            severity_mapping = {
                'low': RiskLevel.LOW,
                'medium': RiskLevel.MEDIUM,
                'high': RiskLevel.HIGH,
                'critical': RiskLevel.CRITICAL
            }
            
            rule = CircuitBreakerRule(
                name=name,
                breaker_type=type_mapping.get(name, CircuitBreakerType.PRICE),
                threshold=config['threshold'],
                duration=config['duration'],
                action=config['action'],
                severity=severity_mapping.get(config['severity'], RiskLevel.MEDIUM),
                enabled=True
            )
            
            state = CircuitBreakerState(
                rule=rule,
                status=CircuitBreakerStatus.NORMAL
            )
            
            circuit_breakers[name] = state
        
        return circuit_breakers
    
    def calculate_risk_metrics(self, returns: pd.Series, 
                             prices: pd.Series = None) -> RiskMetrics:
        """
        Calcula métricas de riesgo con circuit breakers.
        
        Args:
            returns: Serie de retornos
            prices: Serie de precios (opcional)
            
        Returns:
            Métricas de riesgo
        """
        if len(returns) == 0:
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
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        max_drawdown = self.calculate_max_drawdown(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)
        
        # Calcular métricas específicas de circuit breakers
        concentration_risk = self._calculate_portfolio_concentration()
        leverage_ratio = self._calculate_portfolio_leverage()
        liquidity_risk = self._calculate_portfolio_liquidity()
        
        return RiskMetrics(
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
    
    def check_risk_limits(self, positions: List[Position], 
                         returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica límites de riesgo y circuit breakers.
        
        Args:
            positions: Lista de posiciones
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas de riesgo
        """
        alerts = []
        
        # Verificar circuit breakers
        circuit_breaker_alerts = self._check_circuit_breakers(positions, returns)
        alerts.extend(circuit_breaker_alerts)
        
        # Verificar límites básicos
        basic_alerts = self._check_basic_limits(positions, returns)
        alerts.extend(basic_alerts)
        
        # Verificar stops y takes
        stop_take_alerts = self._check_stops_and_takes(positions)
        alerts.extend(stop_take_alerts)
        
        # Agregar alertas al gestor
        for alert in alerts:
            self.add_alert(alert)
        
        return alerts
    
    def _check_circuit_breakers(self, positions: List[Position], 
                               returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica circuit breakers.
        
        Args:
            positions: Lista de posiciones
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas de circuit breakers
        """
        alerts = []
        current_time = datetime.now()
        
        for name, state in self.circuit_breakers.items():
            if not state.rule.enabled:
                continue
            
            # Verificar cooldown
            if state.cooldown_until and current_time < state.cooldown_until:
                continue
            
            # Verificar si el circuit breaker está activo
            if state.status == CircuitBreakerStatus.TRIGGERED:
                if state.triggered_at:
                    time_since_trigger = current_time - state.triggered_at
                    if time_since_trigger.total_seconds() / 60 >= state.rule.duration:
                        # Circuit breaker expirado, volver a normal
                        state.status = CircuitBreakerStatus.NORMAL
                        state.triggered_at = None
                        logger.info(f"Circuit breaker {name} expirado, volviendo a normal")
                continue
            
            # Verificar condición del circuit breaker
            is_triggered = self._check_circuit_breaker_condition(name, state, positions, returns)
            
            if is_triggered:
                # Circuit breaker activado
                state.status = CircuitBreakerStatus.TRIGGERED
                state.triggered_at = current_time
                state.trigger_count += 1
                
                # Crear alerta
                alert = RiskAlert(
                    risk_type=RiskType.MARKET,
                    risk_level=state.rule.severity,
                    message=f"Circuit breaker {name} activado: {state.rule.action}",
                    value=state.rule.threshold,
                    threshold=state.rule.threshold,
                    timestamp=current_time,
                    action_required=True,
                    metadata={
                        'circuit_breaker': name,
                        'action': state.rule.action,
                        'trigger_count': state.trigger_count
                    }
                )
                
                alerts.append(alert)
                
                # Ejecutar acción
                self._execute_circuit_breaker_action(name, state, positions)
                
                logger.warning(f"Circuit breaker {name} activado: {state.rule.action}")
        
        return alerts
    
    def _check_circuit_breaker_condition(self, name: str, state: CircuitBreakerState, 
                                       positions: List[Position], 
                                       returns: pd.Series = None) -> bool:
        """
        Verifica condición específica de un circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            state: Estado del circuit breaker
            positions: Lista de posiciones
            returns: Serie de retornos (opcional)
            
        Returns:
            True si la condición se cumple
        """
        if state.rule.breaker_type == CircuitBreakerType.PRICE:
            return self._check_price_breaker(state, positions)
        elif state.rule.breaker_type == CircuitBreakerType.VOLATILITY:
            return self._check_volatility_breaker(state, returns)
        elif state.rule.breaker_type == CircuitBreakerType.VOLUME:
            return self._check_volume_breaker(state, positions)
        elif state.rule.breaker_type == CircuitBreakerType.DRAWDOWN:
            return self._check_drawdown_breaker(state, returns)
        elif state.rule.breaker_type == CircuitBreakerType.CORRELATION:
            return self._check_correlation_breaker(state, positions)
        elif state.rule.breaker_type == CircuitBreakerType.LIQUIDITY:
            return self._check_liquidity_breaker(state, positions)
        
        return False
    
    def _check_price_breaker(self, state: CircuitBreakerState, 
                           positions: List[Position]) -> bool:
        """
        Verifica circuit breaker de precio.
        
        Args:
            state: Estado del circuit breaker
            positions: Lista de posiciones
            
        Returns:
            True si se activa el circuit breaker
        """
        for position in positions:
            if position.symbol in self.price_history:
                price_history = self.price_history[position.symbol]
                if len(price_history) >= 2:
                    current_price = price_history[-1]
                    previous_price = price_history[-2]
                    
                    price_change = (current_price - previous_price) / previous_price
                    
                    if price_change <= -state.rule.threshold:
                        return True
        
        return False
    
    def _check_volatility_breaker(self, state: CircuitBreakerState, 
                                returns: pd.Series = None) -> bool:
        """
        Verifica circuit breaker de volatilidad.
        
        Args:
            state: Estado del circuit breaker
            returns: Serie de retornos (opcional)
            
        Returns:
            True si se activa el circuit breaker
        """
        if returns is None or len(returns) == 0:
            return False
        
        volatility = returns.std() * np.sqrt(252)
        return volatility >= state.rule.threshold
    
    def _check_volume_breaker(self, state: CircuitBreakerState, 
                            positions: List[Position]) -> bool:
        """
        Verifica circuit breaker de volumen.
        
        Args:
            state: Estado del circuit breaker
            positions: Lista de posiciones
            
        Returns:
            True si se activa el circuit breaker
        """
        for position in positions:
            if position.symbol in self.volume_history:
                volume_history = self.volume_history[position.symbol]
                if len(volume_history) >= 2:
                    current_volume = volume_history[-1]
                    previous_volume = volume_history[-2]
                    
                    volume_change = (current_volume - previous_volume) / previous_volume
                    
                    if volume_change <= -state.rule.threshold:
                        return True
        
        return False
    
    def _check_drawdown_breaker(self, state: CircuitBreakerState, 
                              returns: pd.Series = None) -> bool:
        """
        Verifica circuit breaker de drawdown.
        
        Args:
            state: Estado del circuit breaker
            returns: Serie de retornos (opcional)
            
        Returns:
            True si se activa el circuit breaker
        """
        if returns is None or len(returns) == 0:
            return False
        
        max_drawdown = self.calculate_max_drawdown(returns)
        return abs(max_drawdown) >= state.rule.threshold
    
    def _check_correlation_breaker(self, state: CircuitBreakerState, 
                                 positions: List[Position]) -> bool:
        """
        Verifica circuit breaker de correlación.
        
        Args:
            state: Estado del circuit breaker
            positions: Lista de posiciones
            
        Returns:
            True si se activa el circuit breaker
        """
        if len(positions) < 2:
            return False
        
        # Calcular correlación promedio entre posiciones
        symbols = [p.symbol for p in positions]
        correlations = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                if symbol1 in self.correlation_history and symbol2 in self.correlation_history[symbol1]:
                    correlation = self.correlation_history[symbol1][symbol2]
                    correlations.append(abs(correlation))
        
        if correlations:
            avg_correlation = np.mean(correlations)
            return avg_correlation >= state.rule.threshold
        
        return False
    
    def _check_liquidity_breaker(self, state: CircuitBreakerState, 
                               positions: List[Position]) -> bool:
        """
        Verifica circuit breaker de liquidez.
        
        Args:
            state: Estado del circuit breaker
            positions: Lista de posiciones
            
        Returns:
            True si se activa el circuit breaker
        """
        for position in positions:
            if position.symbol in self.volatility_history:
                # Usar volatilidad como proxy de liquidez
                volatility = self.volatility_history[position.symbol]
                liquidity_risk = min(1.0, volatility / 0.5)  # Normalizar
                
                if liquidity_risk >= state.rule.threshold:
                    return True
        
        return False
    
    def _execute_circuit_breaker_action(self, name: str, state: CircuitBreakerState, 
                                      positions: List[Position]) -> None:
        """
        Ejecuta acción del circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            state: Estado del circuit breaker
            positions: Lista de posiciones
        """
        action = state.rule.action
        
        if action == 'halt':
            # Detener todas las operaciones
            self._halt_trading(positions)
        elif action == 'reduce':
            # Reducir tamaño de posiciones
            self._reduce_positions(positions, 0.5)  # Reducir a 50%
        elif action == 'alert':
            # Solo alertar, no tomar acción
            pass
        
        # Establecer cooldown
        state.cooldown_until = datetime.now() + timedelta(minutes=state.rule.cooldown)
    
    def _halt_trading(self, positions: List[Position]) -> None:
        """
        Detiene todas las operaciones de trading.
        
        Args:
            positions: Lista de posiciones
        """
        # En producción, esto detendría el sistema de trading
        logger.critical("TRADING HALTED - Circuit breaker activado")
        
        # Marcar todas las posiciones para cierre
        for position in positions:
            position.metadata = position.metadata or {}
            position.metadata['halt_reason'] = 'circuit_breaker'
            position.metadata['halt_timestamp'] = datetime.now().isoformat()
    
    def _reduce_positions(self, positions: List[Position], reduction_factor: float) -> None:
        """
        Reduce el tamaño de las posiciones.
        
        Args:
            positions: Lista de posiciones
            reduction_factor: Factor de reducción (0.0 a 1.0)
        """
        for position in positions:
            position.quantity *= reduction_factor
            position.metadata = position.metadata or {}
            position.metadata['reduced_by'] = reduction_factor
            position.metadata['reduction_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Posiciones reducidas por factor {reduction_factor}")
    
    def _check_basic_limits(self, positions: List[Position], 
                          returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica límites básicos de riesgo.
        
        Args:
            positions: Lista de posiciones
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        # Verificar tamaño de posiciones
        for position in positions:
            position_size = abs(position.quantity * position.price)
            portfolio_value = self._estimate_portfolio_value(positions)
            
            if portfolio_value > 0:
                position_weight = position_size / portfolio_value
                
                if position_weight > self.risk_limits.max_position_size:
                    alerts.append(RiskAlert(
                        risk_type=RiskType.CONCENTRATION,
                        risk_level=RiskLevel.HIGH,
                        message=f"Posición {position.symbol} ({position_weight:.3f}) excede límite",
                        value=position_weight,
                        threshold=self.risk_limits.max_position_size,
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        action_required=True
                    ))
        
        # Verificar volatilidad del portafolio
        if returns is not None and len(returns) > 0:
            portfolio_volatility = returns.std() * np.sqrt(252)
            
            if portfolio_volatility > self.risk_limits.max_portfolio_volatility:
                alerts.append(RiskAlert(
                    risk_type=RiskType.MARKET,
                    risk_level=RiskLevel.HIGH,
                    message=f"Volatilidad del portafolio ({portfolio_volatility:.3f}) excede límite",
                    value=portfolio_volatility,
                    threshold=self.risk_limits.max_portfolio_volatility,
                    timestamp=datetime.now(),
                    action_required=True
                ))
        
        return alerts
    
    def _check_stops_and_takes(self, positions: List[Position]) -> List[RiskAlert]:
        """
        Verifica stops y takes de las posiciones.
        
        Args:
            positions: Lista de posiciones
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        for position in positions:
            if position.stop_loss is not None:
                # Verificar stop loss
                if position.side == 'long' and position.price <= position.stop_loss:
                    alerts.append(RiskAlert(
                        risk_type=RiskType.MARKET,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"Stop loss activado para {position.symbol}",
                        value=position.price,
                        threshold=position.stop_loss,
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        action_required=True
                    ))
                elif position.side == 'short' and position.price >= position.stop_loss:
                    alerts.append(RiskAlert(
                        risk_type=RiskType.MARKET,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"Stop loss activado para {position.symbol}",
                        value=position.price,
                        threshold=position.stop_loss,
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        action_required=True
                    ))
            
            if position.take_profit is not None:
                # Verificar take profit
                if position.side == 'long' and position.price >= position.take_profit:
                    alerts.append(RiskAlert(
                        risk_type=RiskType.MARKET,
                        risk_level=RiskLevel.LOW,
                        message=f"Take profit activado para {position.symbol}",
                        value=position.price,
                        threshold=position.take_profit,
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        action_required=True
                    ))
                elif position.side == 'short' and position.price <= position.take_profit:
                    alerts.append(RiskAlert(
                        risk_type=RiskType.MARKET,
                        risk_level=RiskLevel.LOW,
                        message=f"Take profit activado para {position.symbol}",
                        value=position.price,
                        threshold=position.take_profit,
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        action_required=True
                    ))
        
        return alerts
    
    def adjust_position_size(self, signal: float, symbol: str, 
                           current_price: float, 
                           portfolio_value: float) -> float:
        """
        Ajusta el tamaño de posición con circuit breakers.
        
        Args:
            signal: Señal de trading (-1, 0, 1)
            symbol: Símbolo del instrumento
            current_price: Precio actual
            portfolio_value: Valor del portafolio
            
        Returns:
            Tamaño de posición ajustado
        """
        if signal == 0 or portfolio_value <= 0:
            return 0.0
        
        # Verificar si hay circuit breakers activos
        if self._has_active_circuit_breakers():
            logger.warning(f"Circuit breakers activos, reduciendo tamaño de posición para {symbol}")
            return 0.0
        
        # Calcular tamaño base
        base_size = self.position_sizing_config['base_size']
        max_size = self.position_sizing_config['max_size']
        
        # Ajustar por volatilidad
        if self.position_sizing_config['volatility_adjustment']:
            volatility = self._get_instrument_volatility(symbol)
            if volatility > 0:
                volatility_adjustment = min(1.0, 0.2 / volatility)  # Reducir si alta volatilidad
                base_size *= volatility_adjustment
        
        # Ajustar por correlación
        if self.position_sizing_config['correlation_adjustment']:
            correlation = self._get_instrument_correlation(symbol)
            if correlation > 0.7:  # Alta correlación
                base_size *= 0.5  # Reducir a la mitad
        
        # Ajustar por liquidez
        if self.position_sizing_config['liquidity_adjustment']:
            liquidity_risk = self._get_instrument_liquidity_risk(symbol)
            if liquidity_risk > 0.5:  # Alta liquidez
                base_size *= 0.7  # Reducir
        
        # Aplicar límites
        base_size = np.clip(base_size, 0.01, max_size)  # Entre 1% y 15%
        
        # Calcular tamaño final
        position_value = base_size * portfolio_value
        position_size = position_value / current_price
        
        # Aplicar señal
        position_size *= signal
        
        logger.info(f"Posición ajustada para {symbol}: {position_size:.2f} (base: {base_size:.3f})")
        
        return position_size
    
    def _has_active_circuit_breakers(self) -> bool:
        """
        Verifica si hay circuit breakers activos.
        
        Returns:
            True si hay circuit breakers activos
        """
        for state in self.circuit_breakers.values():
            if state.status == CircuitBreakerStatus.TRIGGERED:
                return True
        return False
    
    def _get_instrument_volatility(self, symbol: str) -> float:
        """
        Obtiene volatilidad de un instrumento.
        
        Args:
            symbol: Símbolo del instrumento
            
        Returns:
            Volatilidad
        """
        return self.volatility_history.get(symbol, 0.2)  # 20% por defecto
    
    def _get_instrument_correlation(self, symbol: str) -> float:
        """
        Obtiene correlación promedio de un instrumento.
        
        Args:
            symbol: Símbolo del instrumento
            
        Returns:
            Correlación promedio
        """
        if symbol in self.correlation_history:
            correlations = list(self.correlation_history[symbol].values())
            return np.mean(correlations) if correlations else 0.0
        return 0.0
    
    def _get_instrument_liquidity_risk(self, symbol: str) -> float:
        """
        Obtiene riesgo de liquidez de un instrumento.
        
        Args:
            symbol: Símbolo del instrumento
            
        Returns:
            Riesgo de liquidez
        """
        # Usar volatilidad como proxy
        volatility = self._get_instrument_volatility(symbol)
        return min(1.0, volatility / 0.5)  # Normalizar
    
    def _estimate_portfolio_value(self, positions: List[Position]) -> float:
        """
        Estima el valor del portafolio.
        
        Args:
            positions: Lista de posiciones
            
        Returns:
            Valor estimado del portafolio
        """
        total_value = 0.0
        
        for position in positions:
            position_value = position.quantity * position.price
            total_value += position_value
        
        return total_value
    
    def _calculate_portfolio_concentration(self) -> float:
        """
        Calcula concentración del portafolio.
        
        Returns:
            Riesgo de concentración
        """
        if not self.positions:
            return 0.0
        
        total_value = self._estimate_portfolio_value(self.positions)
        
        if total_value <= 0:
            return 0.0
        
        weights = []
        for position in self.positions:
            position_value = abs(position.quantity * position.price)
            weight = position_value / total_value
            weights.append(weight)
        
        if not weights:
            return 0.0
        
        # Índice de Herfindahl
        herfindahl_index = sum(w**2 for w in weights)
        return herfindahl_index
    
    def _calculate_portfolio_leverage(self) -> float:
        """
        Calcula leverage del portafolio.
        
        Returns:
            Ratio de leverage
        """
        if not self.positions:
            return 0.0
        
        total_exposure = 0.0
        for position in self.positions:
            position_value = abs(position.quantity * position.price)
            total_exposure += position_value
        
        capital = 1000000  # $1M por defecto
        
        if capital <= 0:
            return 0.0
        
        return total_exposure / capital
    
    def _calculate_portfolio_liquidity(self) -> float:
        """
        Calcula riesgo de liquidez del portafolio.
        
        Returns:
            Riesgo de liquidez promedio
        """
        if not self.positions:
            return 0.0
        
        liquidity_risks = []
        for position in self.positions:
            liquidity_risk = self._get_instrument_liquidity_risk(position.symbol)
            liquidity_risks.append(liquidity_risk)
        
        if not liquidity_risks:
            return 0.0
        
        return np.mean(liquidity_risks)
    
    def update_market_data(self, symbol: str, price: float, 
                          volume: float = None, volatility: float = None) -> None:
        """
        Actualiza datos de mercado para circuit breakers.
        
        Args:
            symbol: Símbolo del instrumento
            price: Precio actual
            volume: Volumen (opcional)
            volatility: Volatilidad (opcional)
        """
        # Actualizar precio
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Mantener solo últimos 100 precios
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Actualizar volumen
        if volume is not None:
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            
            self.volume_history[symbol].append(volume)
            
            if len(self.volume_history[symbol]) > 100:
                self.volume_history[symbol] = self.volume_history[symbol][-100:]
        
        # Actualizar volatilidad
        if volatility is not None:
            self.volatility_history[symbol] = volatility
    
    def update_correlation(self, symbol1: str, symbol2: str, 
                          correlation: float) -> None:
        """
        Actualiza correlación entre instrumentos.
        
        Args:
            symbol1: Primer símbolo
            symbol2: Segundo símbolo
            correlation: Correlación
        """
        if symbol1 not in self.correlation_history:
            self.correlation_history[symbol1] = {}
        
        self.correlation_history[symbol1][symbol2] = correlation
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Obtiene estado de todos los circuit breakers.
        
        Returns:
            Diccionario con estado de circuit breakers
        """
        status = {}
        
        for name, state in self.circuit_breakers.items():
            status[name] = {
                'status': state.status.value,
                'triggered_at': state.triggered_at.isoformat() if state.triggered_at else None,
                'trigger_count': state.trigger_count,
                'cooldown_until': state.cooldown_until.isoformat() if state.cooldown_until else None,
                'rule': {
                    'type': state.rule.breaker_type.value,
                    'threshold': state.rule.threshold,
                    'duration': state.rule.duration,
                    'action': state.rule.action,
                    'severity': state.rule.severity.value,
                    'enabled': state.rule.enabled
                }
            }
        
        return status
    
    def enable_circuit_breaker(self, name: str) -> None:
        """
        Habilita un circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
        """
        if name in self.circuit_breakers:
            self.circuit_breakers[name].rule.enabled = True
            logger.info(f"Circuit breaker {name} habilitado")
    
    def disable_circuit_breaker(self, name: str) -> None:
        """
        Deshabilita un circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
        """
        if name in self.circuit_breakers:
            self.circuit_breakers[name].rule.enabled = False
            logger.info(f"Circuit breaker {name} deshabilitado")
    
    def reset_circuit_breaker(self, name: str) -> None:
        """
        Resetea un circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
        """
        if name in self.circuit_breakers:
            state = self.circuit_breakers[name]
            state.status = CircuitBreakerStatus.NORMAL
            state.triggered_at = None
            state.cooldown_until = None
            logger.info(f"Circuit breaker {name} reseteado")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de riesgo con circuit breakers.
        
        Returns:
            Diccionario con resumen de riesgo
        """
        return {
            'circuit_breakers': self.get_circuit_breaker_status(),
            'positions_count': len(self.positions),
            'alerts_count': len(self.alerts),
            'active_circuit_breakers': len([s for s in self.circuit_breakers.values() if s.status == CircuitBreakerStatus.TRIGGERED]),
            'portfolio_concentration': self._calculate_portfolio_concentration(),
            'portfolio_leverage': self._calculate_portfolio_leverage(),
            'portfolio_liquidity': self._calculate_portfolio_liquidity()
        }

