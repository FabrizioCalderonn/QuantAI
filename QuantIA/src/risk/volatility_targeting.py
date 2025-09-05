"""
Gestor de riesgo con volatility targeting.
Ajusta posiciones para mantener volatilidad objetivo.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from .base import BaseRiskManager, RiskLimits, RiskAlert, Position, RiskType, RiskLevel, RiskMetrics

logger = logging.getLogger(__name__)


class VolatilityTargetingRiskManager(BaseRiskManager):
    """
    Gestor de riesgo con volatility targeting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el gestor de volatility targeting.
        
        Args:
            config: Configuración del gestor
        """
        default_config = {
            'target_volatility': 0.15,  # 15% anual
            'volatility_window': 20,    # 20 períodos
            'rebalance_frequency': 1,   # 1 período
            'max_position_size': 0.1,   # 10% del portafolio
            'max_portfolio_volatility': 0.20,  # 20% anual
            'max_drawdown': 0.10,       # 10% máximo drawdown
            'max_leverage': 2.0,        # 2x leverage máximo
            'max_concentration': 0.3,   # 30% máximo concentración
            'stop_loss_pct': 0.05,      # 5% stop loss
            'take_profit_pct': 0.15,    # 15% take profit
            'volatility_floor': 0.05,   # 5% volatilidad mínima
            'volatility_ceiling': 0.30, # 30% volatilidad máxima
            'lookback_periods': [5, 10, 20, 50],  # Períodos para volatilidad
            'volatility_method': 'ewma',  # 'ewma', 'garch', 'simple'
            'ewma_lambda': 0.94,        # Lambda para EWMA
            'garch_params': {'p': 1, 'q': 1},  # Parámetros GARCH
            'correlation_threshold': 0.7,  # Umbral de correlación
            'liquidity_threshold': 0.1,    # Umbral de liquidez
            'min_volume': 1000000,        # Volumen mínimo
            'max_spread_pct': 0.002       # 0.2% spread máximo
        }
        
        config = {**default_config, **(config or {})}
        super().__init__("volatility_targeting", config)
        
        # Configuración específica
        self.target_volatility = config['target_volatility']
        self.volatility_window = config['volatility_window']
        self.rebalance_frequency = config['rebalance_frequency']
        self.volatility_floor = config['volatility_floor']
        self.volatility_ceiling = config['volatility_ceiling']
        self.lookback_periods = config['lookback_periods']
        self.volatility_method = config['volatility_method']
        self.ewma_lambda = config['ewma_lambda']
        self.garch_params = config['garch_params']
        self.correlation_threshold = config['correlation_threshold']
        self.liquidity_threshold = config['liquidity_threshold']
        self.min_volume = config['min_volume']
        self.max_spread_pct = config['max_spread_pct']
        
        # Estado interno
        self.last_rebalance = None
        self.volatility_estimates = {}
        self.correlation_matrix = {}
        self.liquidity_scores = {}
        
        logger.info(f"Volatility Targeting Risk Manager inicializado con volatilidad objetivo: {self.target_volatility}")
    
    def _create_risk_limits(self) -> RiskLimits:
        """
        Crea límites de riesgo para volatility targeting.
        
        Returns:
            Límites de riesgo
        """
        return RiskLimits(
            max_position_size=self.config.get('max_position_size', 0.1),
            max_portfolio_volatility=self.config.get('max_portfolio_volatility', 0.20),
            max_drawdown=self.config.get('max_drawdown', 0.10),
            max_leverage=self.config.get('max_leverage', 2.0),
            max_concentration=self.config.get('max_concentration', 0.3),
            max_correlation=self.config.get('correlation_threshold', 0.7),
            var_limit_95=self.config.get('var_limit_95', -0.05),
            var_limit_99=self.config.get('var_limit_99', -0.10),
            stop_loss_pct=self.config.get('stop_loss_pct', 0.05),
            take_profit_pct=self.config.get('take_profit_pct', 0.15)
        )
    
    def calculate_risk_metrics(self, returns: pd.Series, 
                             prices: pd.Series = None) -> RiskMetrics:
        """
        Calcula métricas de riesgo con enfoque en volatilidad.
        
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
        
        # Calcular volatilidad usando método configurado
        volatility = self._calculate_volatility(returns)
        
        # Calcular VaR y CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Calcular drawdown
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Calcular ratios de riesgo
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)
        
        # Calcular beta (requiere retornos del mercado)
        beta = 0.0  # Se calculará cuando se tenga referencia de mercado
        
        # Calcular correlación (requiere otra serie)
        correlation = 0.0  # Se calculará cuando se tenga otra serie
        
        # Calcular riesgo de concentración
        concentration_risk = self._calculate_portfolio_concentration()
        
        # Calcular leverage ratio
        leverage_ratio = self._calculate_portfolio_leverage()
        
        # Calcular riesgo de liquidez
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
            beta=beta,
            correlation=correlation,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio,
            liquidity_risk=liquidity_risk,
            timestamp=datetime.now()
        )
    
    def check_risk_limits(self, positions: List[Position], 
                         returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica límites de riesgo con enfoque en volatilidad.
        
        Args:
            positions: Lista de posiciones
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas de riesgo
        """
        alerts = []
        
        # Verificar volatilidad del portafolio
        if returns is not None and len(returns) > 0:
            portfolio_volatility = self._calculate_volatility(returns)
            
            if portfolio_volatility > self.risk_limits.max_portfolio_volatility:
                alerts.append(RiskAlert(
                    risk_type=RiskType.MARKET,
                    risk_level=RiskLevel.HIGH,
                    message=f"Volatilidad del portafolio ({portfolio_volatility:.3f}) excede límite ({self.risk_limits.max_portfolio_volatility:.3f})",
                    value=portfolio_volatility,
                    threshold=self.risk_limits.max_portfolio_volatility,
                    timestamp=datetime.now(),
                    action_required=True
                ))
            
            # Verificar volatilidad vs objetivo
            vol_deviation = abs(portfolio_volatility - self.target_volatility) / self.target_volatility
            
            if vol_deviation > 0.2:  # 20% desviación
                alerts.append(RiskAlert(
                    risk_type=RiskType.MARKET,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Volatilidad del portafolio ({portfolio_volatility:.3f}) se desvía significativamente del objetivo ({self.target_volatility:.3f})",
                    value=portfolio_volatility,
                    threshold=self.target_volatility,
                    timestamp=datetime.now(),
                    action_required=True
                ))
        
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
                        message=f"Posición {position.symbol} ({position_weight:.3f}) excede límite ({self.risk_limits.max_position_size:.3f})",
                        value=position_weight,
                        threshold=self.risk_limits.max_position_size,
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        action_required=True
                    ))
        
        # Verificar concentración del portafolio
        concentration_risk = self._calculate_portfolio_concentration()
        
        if concentration_risk > self.risk_limits.max_concentration:
            alerts.append(RiskAlert(
                risk_type=RiskType.CONCENTRATION,
                risk_level=RiskLevel.HIGH,
                message=f"Riesgo de concentración ({concentration_risk:.3f}) excede límite ({self.risk_limits.max_concentration:.3f})",
                value=concentration_risk,
                threshold=self.risk_limits.max_concentration,
                timestamp=datetime.now(),
                action_required=True
            ))
        
        # Verificar leverage
        leverage_ratio = self._calculate_portfolio_leverage()
        
        if leverage_ratio > self.risk_limits.max_leverage:
            alerts.append(RiskAlert(
                risk_type=RiskType.LEVERAGE,
                risk_level=RiskLevel.CRITICAL,
                message=f"Leverage del portafolio ({leverage_ratio:.3f}) excede límite ({self.risk_limits.max_leverage:.3f})",
                value=leverage_ratio,
                threshold=self.risk_limits.max_leverage,
                timestamp=datetime.now(),
                action_required=True
            ))
        
        # Verificar drawdown
        if returns is not None and len(returns) > 0:
            max_drawdown = self.calculate_max_drawdown(returns)
            
            if abs(max_drawdown) > self.risk_limits.max_drawdown:
                alerts.append(RiskAlert(
                    risk_type=RiskType.MARKET,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Maximum drawdown ({abs(max_drawdown):.3f}) excede límite ({self.risk_limits.max_drawdown:.3f})",
                    value=abs(max_drawdown),
                    threshold=self.risk_limits.max_drawdown,
                    timestamp=datetime.now(),
                    action_required=True
                ))
        
        # Agregar alertas al gestor
        for alert in alerts:
            self.add_alert(alert)
        
        return alerts
    
    def adjust_position_size(self, signal: float, symbol: str, 
                           current_price: float, 
                           portfolio_value: float) -> float:
        """
        Ajusta el tamaño de posición basado en volatility targeting.
        
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
        
        # Calcular volatilidad del instrumento
        instrument_volatility = self._get_instrument_volatility(symbol)
        
        if instrument_volatility <= 0:
            logger.warning(f"No se pudo calcular volatilidad para {symbol}")
            return 0.0
        
        # Calcular tamaño de posición basado en volatilidad
        # Fórmula: position_size = (target_volatility / instrument_volatility) * portfolio_value
        volatility_scalar = self.target_volatility / instrument_volatility
        
        # Aplicar límites
        volatility_scalar = np.clip(volatility_scalar, 0.1, 2.0)  # Entre 0.1x y 2.0x
        
        # Calcular tamaño base
        base_position_size = (volatility_scalar * portfolio_value) / current_price
        
        # Aplicar límite de posición máxima
        max_position_value = self.risk_limits.max_position_size * portfolio_value
        max_position_size = max_position_value / current_price
        
        # Aplicar límite de concentración
        concentration_limit = self.risk_limits.max_concentration * portfolio_value
        concentration_position_size = concentration_limit / current_price
        
        # Tomar el mínimo de todos los límites
        final_position_size = min(base_position_size, max_position_size, concentration_position_size)
        
        # Aplicar señal
        final_position_size *= signal
        
        # Verificar límites adicionales
        if abs(final_position_size * current_price) > max_position_value:
            final_position_size = (max_position_value / current_price) * np.sign(final_position_size)
        
        logger.info(f"Posición ajustada para {symbol}: {final_position_size:.2f} (vol: {instrument_volatility:.3f}, scalar: {volatility_scalar:.3f})")
        
        return final_position_size
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calcula volatilidad usando el método configurado.
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Volatilidad anualizada
        """
        if len(returns) == 0:
            return 0.0
        
        if self.volatility_method == 'ewma':
            return self._calculate_ewma_volatility(returns)
        elif self.volatility_method == 'garch':
            return self._calculate_garch_volatility(returns)
        else:  # simple
            return self._calculate_simple_volatility(returns)
    
    def _calculate_simple_volatility(self, returns: pd.Series) -> float:
        """
        Calcula volatilidad simple (rolling window).
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Volatilidad anualizada
        """
        if len(returns) < self.volatility_window:
            window = len(returns)
        else:
            window = self.volatility_window
        
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility * np.sqrt(252)  # Anualizar
    
    def _calculate_ewma_volatility(self, returns: pd.Series) -> float:
        """
        Calcula volatilidad EWMA.
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Volatilidad anualizada
        """
        if len(returns) == 0:
            return 0.0
        
        # Calcular EWMA de varianza
        ewma_var = returns.ewm(alpha=1-self.ewma_lambda).var().iloc[-1]
        volatility = np.sqrt(ewma_var)
        
        return volatility * np.sqrt(252)  # Anualizar
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """
        Calcula volatilidad GARCH (simplificado).
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Volatilidad anualizada
        """
        if len(returns) < 10:
            return self._calculate_simple_volatility(returns)
        
        # Implementación simplificada de GARCH(1,1)
        # En producción, usar librería especializada como arch
        try:
            # Calcular varianza condicional
            returns_squared = returns ** 2
            mean_return = returns.mean()
            mean_return_squared = mean_return ** 2
            
            # Parámetros GARCH simplificados
            alpha = 0.1  # Coeficiente para shocks
            beta = 0.85  # Coeficiente para varianza pasada
            omega = 0.01  # Constante
            
            # Calcular varianza GARCH
            garch_var = omega + alpha * returns_squared.iloc[-1] + beta * returns_squared.rolling(5).mean().iloc[-1]
            volatility = np.sqrt(garch_var)
            
            return volatility * np.sqrt(252)  # Anualizar
            
        except Exception as e:
            logger.warning(f"Error calculando volatilidad GARCH: {str(e)}")
            return self._calculate_simple_volatility(returns)
    
    def _get_instrument_volatility(self, symbol: str) -> float:
        """
        Obtiene volatilidad estimada para un instrumento.
        
        Args:
            symbol: Símbolo del instrumento
            
        Returns:
            Volatilidad estimada
        """
        if symbol in self.volatility_estimates:
            return self.volatility_estimates[symbol]
        
        # Volatilidad por defecto si no se tiene estimación
        return self.target_volatility
    
    def update_volatility_estimates(self, symbol: str, returns: pd.Series) -> None:
        """
        Actualiza estimaciones de volatilidad para un instrumento.
        
        Args:
            symbol: Símbolo del instrumento
            returns: Serie de retornos
        """
        volatility = self._calculate_volatility(returns)
        self.volatility_estimates[symbol] = volatility
        
        logger.info(f"Volatilidad actualizada para {symbol}: {volatility:.3f}")
    
    def _calculate_portfolio_concentration(self) -> float:
        """
        Calcula riesgo de concentración del portafolio.
        
        Returns:
            Riesgo de concentración
        """
        if not self.positions:
            return 0.0
        
        # Calcular pesos de posiciones
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
        
        # Calcular exposición total
        total_exposure = 0.0
        for position in self.positions:
            position_value = abs(position.quantity * position.price)
            total_exposure += position_value
        
        # Asumir capital base (en producción, esto vendría de configuración)
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
            # En producción, esto vendría de datos de mercado
            liquidity_risk = 0.1  # Valor por defecto
            liquidity_risks.append(liquidity_risk)
        
        if not liquidity_risks:
            return 0.0
        
        return np.mean(liquidity_risks)
    
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
    
    def should_rebalance(self) -> bool:
        """
        Determina si el portafolio debe rebalancearse.
        
        Returns:
            True si debe rebalancearse
        """
        if self.last_rebalance is None:
            return True
        
        # Rebalancear cada N períodos
        time_since_rebalance = datetime.now() - self.last_rebalance
        rebalance_interval = timedelta(days=self.rebalance_frequency)
        
        return time_since_rebalance >= rebalance_interval
    
    def rebalance_portfolio(self, positions: List[Position], 
                          current_prices: Dict[str, float],
                          target_weights: Dict[str, float]) -> List[Position]:
        """
        Rebalancea el portafolio para mantener volatilidad objetivo.
        
        Args:
            positions: Posiciones actuales
            current_prices: Precios actuales
            target_weights: Pesos objetivo
            
        Returns:
            Lista de posiciones rebalanceadas
        """
        if not positions:
            return positions
        
        # Calcular valor total del portafolio
        total_value = self._estimate_portfolio_value(positions)
        
        if total_value <= 0:
            return positions
        
        # Calcular nuevas posiciones
        new_positions = []
        
        for symbol, target_weight in target_weights.items():
            if symbol in current_prices:
                target_value = target_weight * total_value
                target_quantity = target_value / current_prices[symbol]
                
                # Crear nueva posición
                new_position = Position(
                    symbol=symbol,
                    quantity=target_quantity,
                    price=current_prices[symbol],
                    timestamp=datetime.now(),
                    side='long' if target_quantity > 0 else 'short'
                )
                
                new_positions.append(new_position)
        
        # Actualizar timestamp de rebalance
        self.last_rebalance = datetime.now()
        
        logger.info(f"Portafolio rebalanceado: {len(new_positions)} posiciones")
        
        return new_positions
    
    def get_volatility_target(self) -> float:
        """
        Obtiene la volatilidad objetivo.
        
        Returns:
            Volatilidad objetivo
        """
        return self.target_volatility
    
    def set_volatility_target(self, target_volatility: float) -> None:
        """
        Establece nueva volatilidad objetivo.
        
        Args:
            target_volatility: Nueva volatilidad objetivo
        """
        if target_volatility <= 0:
            raise ValueError("La volatilidad objetivo debe ser positiva")
        
        self.target_volatility = target_volatility
        logger.info(f"Volatilidad objetivo actualizada: {target_volatility:.3f}")
    
    def get_volatility_estimates(self) -> Dict[str, float]:
        """
        Obtiene estimaciones de volatilidad por instrumento.
        
        Returns:
            Diccionario con estimaciones de volatilidad
        """
        return self.volatility_estimates.copy()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de riesgo del portafolio.
        
        Returns:
            Diccionario con resumen de riesgo
        """
        return {
            'target_volatility': self.target_volatility,
            'volatility_estimates': self.volatility_estimates,
            'positions_count': len(self.positions),
            'alerts_count': len(self.alerts),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'should_rebalance': self.should_rebalance(),
            'portfolio_concentration': self._calculate_portfolio_concentration(),
            'portfolio_leverage': self._calculate_portfolio_leverage(),
            'portfolio_liquidity': self._calculate_portfolio_liquidity()
        }

