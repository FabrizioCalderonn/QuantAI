"""
Métricas de validación para estrategias de trading.
Implementa métricas robustas y criterios de aprobación.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Tipos de métricas."""
    PERFORMANCE = "performance"
    RISK = "risk"
    TRADING = "trading"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"


class ValidationLevel(Enum):
    """Niveles de validación."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    INSTITUTIONAL = "institutional"


@dataclass
class MetricThreshold:
    """Umbral para una métrica."""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    weight: float = 1.0
    required: bool = True
    description: str = ""


@dataclass
class ValidationResult:
    """Resultado de validación."""
    strategy_name: str
    validation_level: ValidationLevel
    overall_score: float
    passed: bool
    metrics_results: Dict[str, Dict[str, Any]]
    failed_metrics: List[str]
    warnings: List[str]
    recommendations: List[str]
    timestamp: datetime


class ValidationMetrics:
    """
    Calculadora de métricas de validación.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el calculador de métricas.
        
        Args:
            config: Configuración de métricas
        """
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.trading_days = self.config.get('trading_days', 252)
        
        logger.info("ValidationMetrics inicializado")
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calcula métricas de performance.
        
        Args:
            returns: Serie de retornos
            benchmark_returns: Retornos del benchmark (opcional)
            
        Returns:
            Diccionario con métricas de performance
        """
        if returns.empty:
            return self._empty_performance_metrics()
        
        metrics = {}
        
        # Métricas básicas
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns).prod() ** (self.trading_days / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(self.trading_days)
        
        # Ratios de riesgo
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)
        
        # Métricas de benchmark
        if benchmark_returns is not None and not benchmark_returns.empty:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return metrics
    
    def calculate_risk_metrics(self, returns: pd.Series, 
                             portfolio_values: pd.Series = None) -> Dict[str, float]:
        """
        Calcula métricas de riesgo.
        
        Args:
            returns: Serie de retornos
            portfolio_values: Valores del portafolio (opcional)
            
        Returns:
            Diccionario con métricas de riesgo
        """
        if returns.empty:
            return self._empty_risk_metrics()
        
        metrics = {}
        
        # VaR y CVaR
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Drawdown
        if portfolio_values is not None and not portfolio_values.empty:
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_values)
            metrics.update(drawdown_metrics)
        else:
            metrics.update(self._calculate_drawdown_metrics_from_returns(returns))
        
        # Métricas adicionales
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
        metrics['ulcer_index'] = self._calculate_ulcer_index(returns)
        
        return metrics
    
    def calculate_trading_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calcula métricas de trading.
        
        Args:
            trades: Lista de trades
            
        Returns:
            Diccionario con métricas de trading
        """
        if not trades:
            return self._empty_trading_metrics()
        
        # Convertir trades a DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Filtrar trades cerrados
        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return self._empty_trading_metrics()
        
        # Calcular PnL
        closed_trades['pnl'] = closed_trades['exit_price'] - closed_trades['entry_price']
        closed_trades['pnl_pct'] = closed_trades['pnl'] / closed_trades['entry_price']
        
        metrics = {}
        
        # Métricas básicas
        metrics['total_trades'] = len(closed_trades)
        metrics['winning_trades'] = len(closed_trades[closed_trades['pnl'] > 0])
        metrics['losing_trades'] = len(closed_trades[closed_trades['pnl'] < 0])
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        
        # Métricas de PnL
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        metrics['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
        metrics['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
        metrics['profit_factor'] = self._calculate_profit_factor(winning_trades, losing_trades)
        metrics['expectancy'] = closed_trades['pnl'].mean()
        
        # Métricas adicionales
        metrics['max_consecutive_wins'] = self._calculate_max_consecutive_wins(closed_trades)
        metrics['max_consecutive_losses'] = self._calculate_max_consecutive_losses(closed_trades)
        metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(closed_trades)
        
        return metrics
    
    def calculate_stability_metrics(self, returns: pd.Series, 
                                  period_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calcula métricas de estabilidad.
        
        Args:
            returns: Serie de retornos
            period_returns: Retornos por período (opcional)
            
        Returns:
            Diccionario con métricas de estabilidad
        """
        if returns.empty:
            return self._empty_stability_metrics()
        
        metrics = {}
        
        # Estabilidad de retornos
        metrics['return_stability'] = 1 - (returns.std() / (abs(returns.mean()) + 1e-8))
        metrics['volatility_stability'] = self._calculate_volatility_stability(returns)
        
        # Consistencia
        metrics['positive_months'] = self._calculate_positive_periods(returns, 'M')
        metrics['positive_quarters'] = self._calculate_positive_periods(returns, 'Q')
        metrics['positive_years'] = self._calculate_positive_periods(returns, 'Y')
        
        # Métricas de período
        if period_returns is not None and not period_returns.empty:
            metrics['period_consistency'] = len(period_returns[period_returns > 0]) / len(period_returns)
            metrics['period_stability'] = 1 - (period_returns.std() / (abs(period_returns.mean()) + 1e-8))
        
        return metrics
    
    def calculate_robustness_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calcula métricas de robustez.
        
        Args:
            returns: Serie de retornos
            benchmark_returns: Retornos del benchmark (opcional)
            
        Returns:
            Diccionario con métricas de robustez
        """
        if returns.empty:
            return self._empty_robustness_metrics()
        
        metrics = {}
        
        # Robustez estadística
        metrics['outlier_ratio'] = self._calculate_outlier_ratio(returns)
        metrics['fat_tail_ratio'] = self._calculate_fat_tail_ratio(returns)
        metrics['normality_test'] = self._calculate_normality_test(returns)
        
        # Robustez temporal
        metrics['autocorrelation'] = self._calculate_autocorrelation(returns)
        metrics['heteroscedasticity'] = self._calculate_heteroscedasticity(returns)
        
        # Robustez vs benchmark
        if benchmark_returns is not None and not benchmark_returns.empty:
            metrics['correlation_stability'] = self._calculate_correlation_stability(returns, benchmark_returns)
            metrics['beta_stability'] = self._calculate_beta_stability(returns, benchmark_returns)
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcula Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days
        return excess_returns.mean() / returns.std() * np.sqrt(self.trading_days)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calcula Sortino ratio."""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days)
        return excess_returns.mean() / downside_deviation * np.sqrt(self.trading_days)
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calcula Calmar ratio."""
        if returns.empty:
            return 0.0
        
        annual_return = returns.mean() * self.trading_days
        max_dd = abs(self._calculate_max_drawdown_from_returns(returns))
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calcula métricas vs benchmark."""
        # Alinear series
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return {}
        
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
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calcula métricas de drawdown."""
        if portfolio_values.empty:
            return {}
        
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0.0,
            'max_drawdown_duration': self._calculate_max_drawdown_duration(drawdown)
        }
    
    def _calculate_drawdown_metrics_from_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calcula métricas de drawdown desde retornos."""
        if returns.empty:
            return {}
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0.0,
            'max_drawdown_duration': self._calculate_max_drawdown_duration(drawdown)
        }
    
    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """Calcula maximum drawdown desde retornos."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calcula duración máxima del drawdown."""
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calcula tail ratio."""
        if returns.empty:
            return 0.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        return abs(p95 / p5) if p5 != 0 else 0.0
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calcula ulcer index."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return np.sqrt((drawdown ** 2).mean())
    
    def _calculate_profit_factor(self, winning_trades: pd.DataFrame, 
                               losing_trades: pd.DataFrame) -> float:
        """Calcula profit factor."""
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        
        return total_wins / total_losses if total_losses > 0 else float('inf')
    
    def _calculate_max_consecutive_wins(self, trades: pd.DataFrame) -> int:
        """Calcula máximo de victorias consecutivas."""
        if trades.empty:
            return 0
        
        wins = (trades['pnl'] > 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for win in wins:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self, trades: pd.DataFrame) -> int:
        """Calcula máximo de pérdidas consecutivas."""
        if trades.empty:
            return 0
        
        losses = (trades['pnl'] < 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_avg_trade_duration(self, trades: pd.DataFrame) -> float:
        """Calcula duración promedio de trades."""
        if trades.empty or 'entry_time' not in trades.columns or 'exit_time' not in trades.columns:
            return 0.0
        
        durations = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.days
        return durations.mean()
    
    def _calculate_volatility_stability(self, returns: pd.Series) -> float:
        """Calcula estabilidad de volatilidad."""
        if returns.empty:
            return 0.0
        
        # Calcular volatilidad rolling
        rolling_vol = returns.rolling(window=min(30, len(returns))).std()
        
        if rolling_vol.empty or rolling_vol.std() == 0:
            return 1.0
        
        return 1 - (rolling_vol.std() / rolling_vol.mean())
    
    def _calculate_positive_periods(self, returns: pd.Series, period: str) -> float:
        """Calcula porcentaje de períodos positivos."""
        if returns.empty:
            return 0.0
        
        period_returns = returns.resample(period).apply(lambda x: (1 + x).prod() - 1)
        positive_periods = len(period_returns[period_returns > 0])
        
        return positive_periods / len(period_returns) if len(period_returns) > 0 else 0.0
    
    def _calculate_outlier_ratio(self, returns: pd.Series) -> float:
        """Calcula ratio de outliers."""
        if returns.empty:
            return 0.0
        
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = returns[(returns < Q1 - 1.5 * IQR) | (returns > Q3 + 1.5 * IQR)]
        
        return len(outliers) / len(returns)
    
    def _calculate_fat_tail_ratio(self, returns: pd.Series) -> float:
        """Calcula ratio de fat tails."""
        if returns.empty:
            return 0.0
        
        kurtosis = returns.kurtosis()
        return max(0, kurtosis - 3) / 3  # Normalizar kurtosis
    
    def _calculate_normality_test(self, returns: pd.Series) -> float:
        """Calcula test de normalidad (simplificado)."""
        if returns.empty:
            return 0.0
        
        # Test de normalidad simplificado basado en skewness y kurtosis
        skewness = abs(returns.skew())
        kurtosis = abs(returns.kurtosis() - 3)
        
        # Score de normalidad (0 = normal, 1 = no normal)
        normality_score = min(1.0, (skewness + kurtosis) / 2)
        
        return 1 - normality_score  # Invertir para que 1 = normal
    
    def _calculate_autocorrelation(self, returns: pd.Series) -> float:
        """Calcula autocorrelación."""
        if returns.empty or len(returns) < 2:
            return 0.0
        
        return returns.autocorr(lag=1)
    
    def _calculate_heteroscedasticity(self, returns: pd.Series) -> float:
        """Calcula heteroscedasticidad."""
        if returns.empty:
            return 0.0
        
        # Calcular volatilidad rolling
        rolling_vol = returns.rolling(window=min(30, len(returns))).std()
        
        if rolling_vol.empty or rolling_vol.std() == 0:
            return 0.0
        
        # Medir variabilidad de volatilidad
        return rolling_vol.std() / rolling_vol.mean()
    
    def _calculate_correlation_stability(self, returns: pd.Series, 
                                       benchmark_returns: pd.Series) -> float:
        """Calcula estabilidad de correlación."""
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        # Alinear series
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 30:
            return 0.0
        
        # Calcular correlación rolling
        rolling_corr = aligned_returns.iloc[:, 0].rolling(window=30).corr(aligned_returns.iloc[:, 1])
        
        if rolling_corr.empty or rolling_corr.std() == 0:
            return 1.0
        
        return 1 - (rolling_corr.std() / (abs(rolling_corr.mean()) + 1e-8))
    
    def _calculate_beta_stability(self, returns: pd.Series, 
                                benchmark_returns: pd.Series) -> float:
        """Calcula estabilidad de beta."""
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        # Alinear series
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 30:
            return 0.0
        
        # Calcular beta rolling
        rolling_beta = []
        window = 30
        
        for i in range(window, len(aligned_returns)):
            window_data = aligned_returns.iloc[i-window:i]
            if len(window_data) > 1:
                covariance = np.cov(window_data.iloc[:, 0], window_data.iloc[:, 1])[0, 1]
                benchmark_variance = np.var(window_data.iloc[:, 1])
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                rolling_beta.append(beta)
        
        if not rolling_beta or np.std(rolling_beta) == 0:
            return 1.0
        
        return 1 - (np.std(rolling_beta) / (abs(np.mean(rolling_beta)) + 1e-8))
    
    def _empty_performance_metrics(self) -> Dict[str, float]:
        """Retorna métricas de performance vacías."""
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
            'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0
        }
    
    def _empty_risk_metrics(self) -> Dict[str, float]:
        """Retorna métricas de riesgo vacías."""
        return {
            'var_95': 0.0, 'var_99': 0.0, 'cvar_95': 0.0, 'cvar_99': 0.0,
            'max_drawdown': 0.0, 'avg_drawdown': 0.0, 'max_drawdown_duration': 0,
            'skewness': 0.0, 'kurtosis': 0.0, 'tail_ratio': 0.0, 'ulcer_index': 0.0
        }
    
    def _empty_trading_metrics(self) -> Dict[str, float]:
        """Retorna métricas de trading vacías."""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'profit_factor': 0.0, 'expectancy': 0.0,
            'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
            'avg_trade_duration': 0.0
        }
    
    def _empty_stability_metrics(self) -> Dict[str, float]:
        """Retorna métricas de estabilidad vacías."""
        return {
            'return_stability': 0.0, 'volatility_stability': 0.0,
            'positive_months': 0.0, 'positive_quarters': 0.0, 'positive_years': 0.0,
            'period_consistency': 0.0, 'period_stability': 0.0
        }
    
    def _empty_robustness_metrics(self) -> Dict[str, float]:
        """Retorna métricas de robustez vacías."""
        return {
            'outlier_ratio': 0.0, 'fat_tail_ratio': 0.0, 'normality_test': 0.0,
            'autocorrelation': 0.0, 'heteroscedasticity': 0.0,
            'correlation_stability': 0.0, 'beta_stability': 0.0
        }

