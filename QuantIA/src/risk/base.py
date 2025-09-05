"""
Clase base para gestión de riesgo.
Define la interfaz común para todos los componentes de riesgo.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Niveles de riesgo."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """Tipos de riesgo."""
    MARKET = "market"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    OPERATIONAL = "operational"


@dataclass
class RiskMetrics:
    """Métricas de riesgo."""
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation: float
    concentration_risk: float
    leverage_ratio: float
    liquidity_risk: float
    timestamp: datetime


@dataclass
class RiskAlert:
    """Alerta de riesgo."""
    risk_type: RiskType
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    symbol: str = None
    action_required: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class Position:
    """Posición de trading."""
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class RiskLimits:
    """Límites de riesgo."""
    max_position_size: float
    max_portfolio_volatility: float
    max_drawdown: float
    max_leverage: float
    max_concentration: float
    max_correlation: float
    var_limit_95: float
    var_limit_99: float
    stop_loss_pct: float
    take_profit_pct: float


class BaseRiskManager(ABC):
    """
    Clase base abstracta para gestión de riesgo.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Inicializa el gestor de riesgo.
        
        Args:
            name: Nombre del gestor de riesgo
            config: Configuración del gestor
        """
        self.name = name
        self.config = config or {}
        self.risk_limits = self._create_risk_limits()
        self.alerts = []
        self.positions = []
        self.risk_metrics = {}
        self.created_at = datetime.now()
        
        logger.info(f"Risk Manager '{name}' inicializado")
    
    @abstractmethod
    def _create_risk_limits(self) -> RiskLimits:
        """
        Crea límites de riesgo por defecto.
        
        Returns:
            Límites de riesgo
        """
        pass
    
    @abstractmethod
    def calculate_risk_metrics(self, returns: pd.Series, 
                             prices: pd.Series = None) -> RiskMetrics:
        """
        Calcula métricas de riesgo.
        
        Args:
            returns: Serie de retornos
            prices: Serie de precios (opcional)
            
        Returns:
            Métricas de riesgo
        """
        pass
    
    @abstractmethod
    def check_risk_limits(self, positions: List[Position], 
                         returns: pd.Series = None) -> List[RiskAlert]:
        """
        Verifica límites de riesgo.
        
        Args:
            positions: Lista de posiciones
            returns: Serie de retornos (opcional)
            
        Returns:
            Lista de alertas de riesgo
        """
        pass
    
    @abstractmethod
    def adjust_position_size(self, signal: float, symbol: str, 
                           current_price: float, 
                           portfolio_value: float) -> float:
        """
        Ajusta el tamaño de posición basado en riesgo.
        
        Args:
            signal: Señal de trading (-1, 0, 1)
            symbol: Símbolo del instrumento
            current_price: Precio actual
            portfolio_value: Valor del portafolio
            
        Returns:
            Tamaño de posición ajustado
        """
        pass
    
    def add_position(self, position: Position) -> None:
        """
        Agrega una posición.
        
        Args:
            position: Posición a agregar
        """
        self.positions.append(position)
        logger.info(f"Posición agregada: {position.symbol} {position.quantity} @ {position.price}")
    
    def remove_position(self, symbol: str) -> None:
        """
        Remueve una posición.
        
        Args:
            symbol: Símbolo de la posición a remover
        """
        self.positions = [p for p in self.positions if p.symbol != symbol]
        logger.info(f"Posición removida: {symbol}")
    
    def get_positions(self, symbol: str = None) -> List[Position]:
        """
        Obtiene posiciones.
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            Lista de posiciones
        """
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return self.positions.copy()
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calcula el valor del portafolio.
        
        Args:
            current_prices: Precios actuales por símbolo
            
        Returns:
            Valor total del portafolio
        """
        total_value = 0.0
        
        for position in self.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def get_portfolio_returns(self, current_prices: Dict[str, float]) -> pd.Series:
        """
        Calcula retornos del portafolio.
        
        Args:
            current_prices: Precios actuales por símbolo
            
        Returns:
            Serie de retornos del portafolio
        """
        returns = []
        
        for position in self.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                position_return = (current_price - position.price) / position.price
                returns.append(position_return)
        
        return pd.Series(returns)
    
    def add_alert(self, alert: RiskAlert) -> None:
        """
        Agrega una alerta de riesgo.
        
        Args:
            alert: Alerta de riesgo
        """
        self.alerts.append(alert)
        logger.warning(f"Alerta de riesgo: {alert.risk_type.value} - {alert.message}")
    
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
    
    def clear_alerts(self) -> None:
        """Limpia todas las alertas."""
        self.alerts.clear()
        logger.info("Alertas de riesgo limpiadas")
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Value at Risk (VaR).
        
        Args:
            returns: Serie de retornos
            confidence_level: Nivel de confianza
            
        Returns:
            VaR
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Conditional Value at Risk (CVaR).
        
        Args:
            returns: Serie de retornos
            confidence_level: Nivel de confianza
            
        Returns:
            CVaR
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calcula maximum drawdown.
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Maximum drawdown
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sharpe ratio.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anual
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                               risk_free_rate: float = 0.02) -> float:
        """
        Calcula Sortino ratio.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anual
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calcula Calmar ratio.
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    def calculate_beta(self, asset_returns: pd.Series, 
                      market_returns: pd.Series) -> float:
        """
        Calcula beta.
        
        Args:
            asset_returns: Retornos del activo
            market_returns: Retornos del mercado
            
        Returns:
            Beta
        """
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Alinear series
        aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
        
        asset_ret = aligned_returns.iloc[:, 0]
        market_ret = aligned_returns.iloc[:, 1]
        
        covariance = np.cov(asset_ret, market_ret)[0, 1]
        market_variance = np.var(market_ret)
        
        if market_variance == 0:
            return 0.0
        
        return covariance / market_variance
    
    def calculate_correlation(self, returns1: pd.Series, 
                            returns2: pd.Series) -> float:
        """
        Calcula correlación entre dos series de retornos.
        
        Args:
            returns1: Primera serie de retornos
            returns2: Segunda serie de retornos
            
        Returns:
            Correlación
        """
        if len(returns1) == 0 or len(returns2) == 0:
            return 0.0
        
        # Alinear series
        aligned_returns = pd.concat([returns1, returns2], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
        
        return aligned_returns.corr().iloc[0, 1]
    
    def calculate_concentration_risk(self, positions: List[Position], 
                                   current_prices: Dict[str, float]) -> float:
        """
        Calcula riesgo de concentración.
        
        Args:
            positions: Lista de posiciones
            current_prices: Precios actuales
            
        Returns:
            Riesgo de concentración (índice de Herfindahl)
        """
        if not positions:
            return 0.0
        
        total_value = self.get_portfolio_value(current_prices)
        
        if total_value == 0:
            return 0.0
        
        weights = []
        for position in positions:
            if position.symbol in current_prices:
                position_value = position.quantity * current_prices[position.symbol]
                weight = position_value / total_value
                weights.append(weight)
        
        if not weights:
            return 0.0
        
        # Índice de Herfindahl
        herfindahl_index = sum(w**2 for w in weights)
        return herfindahl_index
    
    def calculate_leverage_ratio(self, positions: List[Position], 
                               current_prices: Dict[str, float],
                               capital: float) -> float:
        """
        Calcula ratio de leverage.
        
        Args:
            positions: Lista de posiciones
            current_prices: Precios actuales
            capital: Capital disponible
            
        Returns:
            Ratio de leverage
        """
        if capital == 0:
            return 0.0
        
        total_exposure = 0.0
        
        for position in positions:
            if position.symbol in current_prices:
                position_value = abs(position.quantity * current_prices[position.symbol])
                total_exposure += position_value
        
        return total_exposure / capital
    
    def calculate_liquidity_risk(self, symbol: str, 
                               volume: pd.Series,
                               price: pd.Series) -> float:
        """
        Calcula riesgo de liquidez.
        
        Args:
            symbol: Símbolo del instrumento
            volume: Serie de volúmenes
            price: Serie de precios
            
        Returns:
            Riesgo de liquidez
        """
        if len(volume) == 0 or len(price) == 0:
            return 1.0  # Máximo riesgo
        
        # Calcular bid-ask spread aproximado basado en volatilidad
        returns = price.pct_change().dropna()
        if len(returns) == 0:
            return 1.0
        
        volatility = returns.std() * np.sqrt(252)
        
        # Calcular impacto de mercado basado en volumen
        avg_volume = volume.mean()
        if avg_volume == 0:
            return 1.0
        
        # Riesgo de liquidez inversamente proporcional al volumen
        liquidity_risk = min(1.0, volatility / (avg_volume / 1e6))
        
        return liquidity_risk
    
    def validate_config(self) -> bool:
        """
        Valida la configuración del gestor de riesgo.
        
        Returns:
            True si la configuración es válida
        """
        if not self.risk_limits:
            logger.error("Límites de riesgo no definidos")
            return False
        
        # Validar límites
        if self.risk_limits.max_position_size <= 0:
            logger.error("max_position_size debe ser positivo")
            return False
        
        if self.risk_limits.max_portfolio_volatility <= 0:
            logger.error("max_portfolio_volatility debe ser positivo")
            return False
        
        if self.risk_limits.max_drawdown <= 0:
            logger.error("max_drawdown debe ser positivo")
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del gestor de riesgo.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'positions_count': len(self.positions),
            'alerts_count': len(self.alerts),
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_portfolio_volatility': self.risk_limits.max_portfolio_volatility,
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_leverage': self.risk_limits.max_leverage,
                'max_concentration': self.risk_limits.max_concentration
            },
            'active_alerts': len([a for a in self.alerts if a.action_required])
        }

