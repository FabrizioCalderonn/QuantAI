"""
Clase base para backtesting.
Define la interfaz común para todos los componentes de backtesting.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class BacktestType(Enum):
    """Tipos de backtesting."""
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    MONTE_CARLO = "monte_carlo"


class TradeDirection(Enum):
    """Dirección de trades."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TradeStatus(Enum):
    """Estado de trades."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Representa un trade individual."""
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    quantity: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class BacktestConfig:
    """Configuración de backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_position_size: float = 0.1  # 10% del capital
    min_trade_size: float = 100.0   # $100 mínimo
    max_trades_per_day: int = 10
    risk_free_rate: float = 0.02    # 2% anual
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "1D"  # Diario
    lookback_window: int = 252      # 1 año
    min_periods: int = 60           # 3 meses mínimo
    purge_period: int = 1           # 1 día de purga
    gap_period: int = 0             # Sin gap
    enable_short_selling: bool = True
    enable_leverage: bool = False
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.05     # 5%
    take_profit_pct: float = 0.15   # 15%


@dataclass
class BacktestMetrics:
    """Métricas de backtesting."""
    # Métricas de retorno
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Métricas de drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    
    # Métricas de trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Métricas de riesgo
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: float
    alpha: float
    information_ratio: float
    
    # Métricas de benchmark
    benchmark_return: float
    excess_return: float
    tracking_error: float
    
    # Métricas adicionales
    skewness: float
    kurtosis: float
    tail_ratio: float
    common_sense_ratio: float
    ulcer_index: float
    
    # Timestamps
    start_date: datetime
    end_date: datetime
    total_days: int


@dataclass
class BacktestResult:
    """Resultado completo de backtesting."""
    config: BacktestConfig
    trades: List[Trade]
    portfolio_values: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    metrics: BacktestMetrics
    equity_curve: pd.DataFrame
    drawdown_curve: pd.Series
    monthly_returns: pd.Series
    yearly_returns: pd.Series
    trade_analysis: pd.DataFrame
    risk_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class BaseBacktester(ABC):
    """
    Clase base abstracta para backtesting.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Inicializa el backtester.
        
        Args:
            config: Configuración de backtesting
        """
        self.config = config
        self.trades = []
        self.portfolio_values = pd.Series(dtype=float)
        self.returns = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.current_capital = config.initial_capital
        self.positions = {}  # symbol -> quantity
        self.trade_count = 0
        self.daily_trades = {}  # date -> count
        
        # Validar configuración
        self._validate_config()
        
        logger.info(f"Backtester inicializado: {config.start_date} a {config.end_date}")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Valida la configuración de backtesting.
        """
        pass
    
    @abstractmethod
    def _generate_signals(self, data: pd.DataFrame, 
                         features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Genera señales de trading.
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            
        Returns:
            DataFrame con señales
        """
        pass
    
    @abstractmethod
    def _calculate_position_size(self, signal: float, price: float, 
                               symbol: str, current_capital: float) -> float:
        """
        Calcula el tamaño de posición.
        
        Args:
            signal: Señal de trading (-1, 0, 1)
            price: Precio actual
            symbol: Símbolo
            current_capital: Capital actual
            
        Returns:
            Tamaño de posición
        """
        pass
    
    def run_backtest(self, data: pd.DataFrame, 
                    features: pd.DataFrame = None,
                    benchmark_data: pd.DataFrame = None) -> BacktestResult:
        """
        Ejecuta el backtesting.
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            benchmark_data: Datos del benchmark (opcional)
            
        Returns:
            Resultado del backtesting
        """
        logger.info("Iniciando backtesting...")
        
        # Validar datos
        self._validate_data(data)
        
        # Filtrar datos por rango de fechas
        data = self._filter_data_by_date(data)
        
        # Generar señales
        signals = self._generate_signals(data, features)
        
        # Obtener datos del benchmark
        if benchmark_data is not None:
            benchmark_data = self._filter_data_by_date(benchmark_data)
            self.benchmark_returns = self._calculate_benchmark_returns(benchmark_data)
        
        # Ejecutar simulación
        self._run_simulation(data, signals)
        
        # Calcular métricas
        metrics = self._calculate_metrics()
        
        # Crear resultado
        result = self._create_result(metrics)
        
        logger.info(f"Backtesting completado: {len(self.trades)} trades")
        
        return result
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Valida los datos de entrada.
        
        Args:
            data: Datos de precios
        """
        if data.empty:
            raise ValueError("Los datos están vacíos")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El índice debe ser DatetimeIndex")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Verificar valores faltantes
        if data.isnull().any().any():
            logger.warning("Los datos contienen valores faltantes")
    
    def _filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra datos por rango de fechas.
        
        Args:
            data: Datos de precios
            
        Returns:
            Datos filtrados
        """
        return data.loc[self.config.start_date:self.config.end_date]
    
    def _calculate_benchmark_returns(self, benchmark_data: pd.DataFrame) -> pd.Series:
        """
        Calcula retornos del benchmark.
        
        Args:
            benchmark_data: Datos del benchmark
            
        Returns:
            Serie de retornos del benchmark
        """
        if benchmark_data.empty:
            return pd.Series(dtype=float)
        
        benchmark_prices = benchmark_data['close']
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        return benchmark_returns
    
    def _run_simulation(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        """
        Ejecuta la simulación de trading.
        
        Args:
            data: Datos de precios
            signals: Señales de trading
        """
        logger.info("Ejecutando simulación de trading...")
        
        # Inicializar portafolio
        self.portfolio_values = pd.Series(index=data.index, dtype=float)
        self.portfolio_values.iloc[0] = self.config.initial_capital
        
        # Procesar cada día
        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0:
                continue
            
            # Obtener señales del día
            day_signals = signals.loc[date] if date in signals.index else pd.Series(dtype=float)
            
            # Procesar trades
            self._process_trades(date, row, day_signals)
            
            # Actualizar valor del portafolio
            self._update_portfolio_value(date, row)
        
        # Calcular retornos
        self.returns = self.portfolio_values.pct_change().dropna()
    
    def _process_trades(self, date: datetime, price_data: pd.Series, 
                       signals: pd.Series) -> None:
        """
        Procesa trades para un día específico.
        
        Args:
            date: Fecha actual
            price_data: Datos de precios del día
            signals: Señales del día
        """
        # Verificar límite de trades por día
        if self._is_trade_limit_reached(date):
            return
        
        # Procesar cada símbolo
        for symbol in signals.index:
            if symbol not in price_data.index:
                continue
            
            signal = signals[symbol]
            if pd.isna(signal) or signal == 0:
                continue
            
            # Obtener precio
            price = price_data[symbol]
            if pd.isna(price):
                continue
            
            # Calcular tamaño de posición
            position_size = self._calculate_position_size(
                signal, price, symbol, self.current_capital
            )
            
            if abs(position_size) < self.config.min_trade_size:
                continue
            
            # Ejecutar trade
            self._execute_trade(date, symbol, signal, price, position_size)
    
    def _is_trade_limit_reached(self, date: datetime) -> bool:
        """
        Verifica si se alcanzó el límite de trades por día.
        
        Args:
            date: Fecha actual
            
        Returns:
            True si se alcanzó el límite
        """
        date_str = date.date()
        daily_count = self.daily_trades.get(date_str, 0)
        return daily_count >= self.config.max_trades_per_day
    
    def _execute_trade(self, date: datetime, symbol: str, signal: float, 
                      price: float, position_size: float) -> None:
        """
        Ejecuta un trade.
        
        Args:
            date: Fecha del trade
            symbol: Símbolo
            signal: Señal de trading
            price: Precio
            position_size: Tamaño de posición
        """
        # Determinar dirección
        direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
        
        # Calcular cantidad
        quantity = position_size / price
        
        # Aplicar slippage
        slippage = price * self.config.slippage_rate
        if direction == TradeDirection.LONG:
            entry_price = price + slippage
        else:
            entry_price = price - slippage
        
        # Calcular comisión
        commission = abs(position_size) * self.config.commission_rate
        
        # Crear trade
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_time=date,
            entry_price=entry_price,
            quantity=quantity,
            commission=commission,
            slippage=slippage,
            stop_loss=self._calculate_stop_loss(entry_price, direction),
            take_profit=self._calculate_take_profit(entry_price, direction)
        )
        
        # Agregar trade
        self.trades.append(trade)
        self.trade_count += 1
        
        # Actualizar posición
        if symbol in self.positions:
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] = quantity
        
        # Actualizar contador diario
        date_str = date.date()
        self.daily_trades[date_str] = self.daily_trades.get(date_str, 0) + 1
        
        # Actualizar capital
        self.current_capital -= commission
        
        logger.debug(f"Trade ejecutado: {symbol} {direction.value} {quantity:.2f} @ {entry_price:.2f}")
    
    def _calculate_stop_loss(self, entry_price: float, direction: TradeDirection) -> float:
        """
        Calcula stop loss.
        
        Args:
            entry_price: Precio de entrada
            direction: Dirección del trade
            
        Returns:
            Precio de stop loss
        """
        if direction == TradeDirection.LONG:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: float, direction: TradeDirection) -> float:
        """
        Calcula take profit.
        
        Args:
            entry_price: Precio de entrada
            direction: Dirección del trade
            
        Returns:
            Precio de take profit
        """
        if direction == TradeDirection.LONG:
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)
    
    def _update_portfolio_value(self, date: datetime, price_data: pd.Series) -> None:
        """
        Actualiza el valor del portafolio.
        
        Args:
            date: Fecha actual
            price_data: Datos de precios del día
        """
        # Calcular valor de posiciones
        position_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if symbol in price_data.index:
                price = price_data[symbol]
                position_value += quantity * price
        
        # Valor total del portafolio
        total_value = self.current_capital + position_value
        
        # Actualizar serie
        self.portfolio_values.loc[date] = total_value
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """
        Calcula métricas de backtesting.
        
        Returns:
            Métricas calculadas
        """
        if self.returns.empty:
            return self._create_empty_metrics()
        
        # Métricas básicas
        total_return = (self.portfolio_values.iloc[-1] / self.config.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(self.returns)) - 1
        volatility = self.returns.std() * np.sqrt(252)
        
        # Ratios de riesgo
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        calmar_ratio = self._calculate_calmar_ratio()
        
        # Drawdown
        max_drawdown, max_dd_duration, avg_drawdown = self._calculate_drawdown_metrics()
        
        # Métricas de trading
        trade_metrics = self._calculate_trade_metrics()
        
        # Métricas de riesgo
        risk_metrics = self._calculate_risk_metrics()
        
        # Métricas de benchmark
        benchmark_metrics = self._calculate_benchmark_metrics()
        
        # Métricas adicionales
        additional_metrics = self._calculate_additional_metrics()
        
        return BacktestMetrics(
            # Métricas de retorno
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            
            # Métricas de drawdown
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_drawdown,
            
            # Métricas de trading
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate=trade_metrics['win_rate'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            profit_factor=trade_metrics['profit_factor'],
            expectancy=trade_metrics['expectancy'],
            
            # Métricas de riesgo
            var_95=risk_metrics['var_95'],
            var_99=risk_metrics['var_99'],
            cvar_95=risk_metrics['cvar_95'],
            cvar_99=risk_metrics['cvar_99'],
            beta=risk_metrics['beta'],
            alpha=risk_metrics['alpha'],
            information_ratio=risk_metrics['information_ratio'],
            
            # Métricas de benchmark
            benchmark_return=benchmark_metrics['benchmark_return'],
            excess_return=benchmark_metrics['excess_return'],
            tracking_error=benchmark_metrics['tracking_error'],
            
            # Métricas adicionales
            skewness=additional_metrics['skewness'],
            kurtosis=additional_metrics['kurtosis'],
            tail_ratio=additional_metrics['tail_ratio'],
            common_sense_ratio=additional_metrics['common_sense_ratio'],
            ulcer_index=additional_metrics['ulcer_index'],
            
            # Timestamps
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_days=len(self.returns)
        )
    
    def _create_empty_metrics(self) -> BacktestMetrics:
        """
        Crea métricas vacías.
        
        Returns:
            Métricas vacías
        """
        return BacktestMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, max_drawdown_duration=0, avg_drawdown=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, expectancy=0.0,
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            beta=0.0, alpha=0.0, information_ratio=0.0,
            benchmark_return=0.0, excess_return=0.0, tracking_error=0.0,
            skewness=0.0, kurtosis=0.0, tail_ratio=0.0,
            common_sense_ratio=0.0, ulcer_index=0.0,
            start_date=self.config.start_date, end_date=self.config.end_date,
            total_days=0
        )
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calcula Sharpe ratio.
        
        Returns:
            Sharpe ratio
        """
        if self.returns.empty or self.returns.std() == 0:
            return 0.0
        
        excess_returns = self.returns - self.config.risk_free_rate / 252
        return excess_returns.mean() / self.returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self) -> float:
        """
        Calcula Sortino ratio.
        
        Returns:
            Sortino ratio
        """
        if self.returns.empty:
            return 0.0
        
        excess_returns = self.returns - self.config.risk_free_rate / 252
        downside_returns = self.returns[self.returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self) -> float:
        """
        Calcula Calmar ratio.
        
        Returns:
            Calmar ratio
        """
        if self.returns.empty:
            return 0.0
        
        annual_return = self.returns.mean() * 252
        max_dd = abs(self._calculate_max_drawdown())
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calcula maximum drawdown.
        
        Returns:
            Maximum drawdown
        """
        if self.portfolio_values.empty:
            return 0.0
        
        cumulative = self.portfolio_values / self.portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_drawdown_metrics(self) -> Tuple[float, int, float]:
        """
        Calcula métricas de drawdown.
        
        Returns:
            Tuple con max_drawdown, max_duration, avg_drawdown
        """
        if self.portfolio_values.empty:
            return 0.0, 0, 0.0
        
        cumulative = self.portfolio_values / self.portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Calcular duración máxima
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0.0
        
        return max_drawdown, max_duration, avg_drawdown
    
    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de trading.
        
        Returns:
            Diccionario con métricas de trading
        """
        if not self.trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'profit_factor': 0.0, 'expectancy': 0.0
            }
        
        # Calcular PnL para trades cerrados
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return {
                'total_trades': len(self.trades), 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'profit_factor': 0.0, 'expectancy': 0.0
            }
        
        # Calcular PnL
        for trade in closed_trades:
            if trade.exit_price is not None:
                if trade.direction == TradeDirection.LONG:
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - trade.commission
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity - trade.commission
                
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
        
        # Métricas
        pnls = [t.pnl for t in closed_trades if t.pnl is not None]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_trades = len(closed_trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = winning_count / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        expectancy = np.mean(pnls) if pnls else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de riesgo.
        
        Returns:
            Diccionario con métricas de riesgo
        """
        if self.returns.empty:
            return {
                'var_95': 0.0, 'var_99': 0.0, 'cvar_95': 0.0, 'cvar_99': 0.0,
                'beta': 0.0, 'alpha': 0.0, 'information_ratio': 0.0
            }
        
        # VaR y CVaR
        var_95 = np.percentile(self.returns, 5)
        var_99 = np.percentile(self.returns, 1)
        cvar_95 = self.returns[self.returns <= var_95].mean()
        cvar_99 = self.returns[self.returns <= var_99].mean()
        
        # Beta y Alpha (requiere benchmark)
        beta = 0.0
        alpha = 0.0
        information_ratio = 0.0
        
        if not self.benchmark_returns.empty:
            # Alinear series
            aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
            
            if len(aligned_returns) > 1:
                portfolio_ret = aligned_returns.iloc[:, 0]
                benchmark_ret = aligned_returns.iloc[:, 1]
                
                # Beta
                covariance = np.cov(portfolio_ret, benchmark_ret)[0, 1]
                benchmark_variance = np.var(benchmark_ret)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                
                # Alpha
                alpha = portfolio_ret.mean() - beta * benchmark_ret.mean()
                
                # Information ratio
                excess_returns = portfolio_ret - benchmark_ret
                tracking_error = excess_returns.std()
                information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio
        }
    
    def _calculate_benchmark_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de benchmark.
        
        Returns:
            Diccionario con métricas de benchmark
        """
        if self.benchmark_returns.empty:
            return {
                'benchmark_return': 0.0,
                'excess_return': 0.0,
                'tracking_error': 0.0
            }
        
        benchmark_return = (1 + self.benchmark_returns).prod() - 1
        portfolio_return = (1 + self.returns).prod() - 1
        excess_return = portfolio_return - benchmark_return
        
        # Tracking error
        aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(aligned_returns) > 1:
            excess_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
            tracking_error = excess_returns.std() * np.sqrt(252)
        else:
            tracking_error = 0.0
        
        return {
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'tracking_error': tracking_error
        }
    
    def _calculate_additional_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas adicionales.
        
        Returns:
            Diccionario con métricas adicionales
        """
        if self.returns.empty:
            return {
                'skewness': 0.0, 'kurtosis': 0.0, 'tail_ratio': 0.0,
                'common_sense_ratio': 0.0, 'ulcer_index': 0.0
            }
        
        # Skewness y Kurtosis
        skewness = self.returns.skew()
        kurtosis = self.returns.kurtosis()
        
        # Tail ratio
        tail_ratio = self._calculate_tail_ratio()
        
        # Common sense ratio
        common_sense_ratio = self._calculate_common_sense_ratio()
        
        # Ulcer index
        ulcer_index = self._calculate_ulcer_index()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'common_sense_ratio': common_sense_ratio,
            'ulcer_index': ulcer_index
        }
    
    def _calculate_tail_ratio(self) -> float:
        """
        Calcula tail ratio.
        
        Returns:
            Tail ratio
        """
        if self.returns.empty:
            return 0.0
        
        # Percentiles 95 y 5
        p95 = np.percentile(self.returns, 95)
        p5 = np.percentile(self.returns, 5)
        
        return abs(p95 / p5) if p5 != 0 else 0.0
    
    def _calculate_common_sense_ratio(self) -> float:
        """
        Calcula common sense ratio.
        
        Returns:
            Common sense ratio
        """
        if self.returns.empty:
            return 0.0
        
        # Retorno promedio de los mejores y peores días
        best_days = self.returns.nlargest(5).mean()
        worst_days = self.returns.nsmallest(5).mean()
        
        return abs(best_days / worst_days) if worst_days != 0 else 0.0
    
    def _calculate_ulcer_index(self) -> float:
        """
        Calcula ulcer index.
        
        Returns:
            Ulcer index
        """
        if self.portfolio_values.empty:
            return 0.0
        
        cumulative = self.portfolio_values / self.portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Ulcer index es la raíz cuadrada del promedio de drawdowns al cuadrado
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        
        return ulcer_index
    
    def _create_result(self, metrics: BacktestMetrics) -> BacktestResult:
        """
        Crea el resultado del backtesting.
        
        Args:
            metrics: Métricas calculadas
            
        Returns:
            Resultado del backtesting
        """
        # Crear equity curve
        equity_curve = pd.DataFrame({
            'portfolio_value': self.portfolio_values,
            'returns': self.returns,
            'benchmark_returns': self.benchmark_returns
        })
        
        # Crear drawdown curve
        if not self.portfolio_values.empty:
            cumulative = self.portfolio_values / self.portfolio_values.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown_curve = (cumulative - running_max) / running_max
        else:
            drawdown_curve = pd.Series(dtype=float)
        
        # Crear retornos mensuales y anuales
        monthly_returns = self._calculate_period_returns('M')
        yearly_returns = self._calculate_period_returns('Y')
        
        # Crear análisis de trades
        trade_analysis = self._create_trade_analysis()
        
        # Crear métricas de riesgo
        risk_metrics = {
            'var_95': metrics.var_95,
            'var_99': metrics.var_99,
            'cvar_95': metrics.cvar_95,
            'cvar_99': metrics.cvar_99,
            'beta': metrics.beta,
            'alpha': metrics.alpha,
            'information_ratio': metrics.information_ratio
        }
        
        # Crear metadata
        metadata = {
            'backtest_type': 'base',
            'created_at': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'trade_count': len(self.trades),
            'total_days': len(self.returns)
        }
        
        return BacktestResult(
            config=self.config,
            trades=self.trades,
            portfolio_values=self.portfolio_values,
            returns=self.returns,
            benchmark_returns=self.benchmark_returns,
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            trade_analysis=trade_analysis,
            risk_metrics=risk_metrics,
            metadata=metadata
        )
    
    def _calculate_period_returns(self, period: str) -> pd.Series:
        """
        Calcula retornos por período.
        
        Args:
            period: Período ('M' para mensual, 'Y' para anual)
            
        Returns:
            Serie de retornos por período
        """
        if self.returns.empty:
            return pd.Series(dtype=float)
        
        period_returns = self.returns.resample(period).apply(lambda x: (1 + x).prod() - 1)
        return period_returns.dropna()
    
    def _create_trade_analysis(self) -> pd.DataFrame:
        """
        Crea análisis de trades.
        
        Returns:
            DataFrame con análisis de trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'symbol': trade.symbol,
                'direction': trade.direction.value,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'quantity': trade.quantity,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'status': trade.status.value,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'commission': trade.commission,
                'slippage': trade.slippage
            })
        
        return pd.DataFrame(trade_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del backtesting.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config.__dict__,
            'trades_count': len(self.trades),
            'portfolio_value': self.current_capital,
            'returns_count': len(self.returns),
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat()
        }

