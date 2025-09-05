"""
Backtesting con Walk-Forward Analysis.
Implementa validación temporal robusta con purga de datos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

from .base import BaseBacktester, BacktestConfig, BacktestResult, BacktestMetrics, Trade, TradeDirection, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuración de walk-forward analysis."""
    train_period: int = 252        # 1 año de entrenamiento
    test_period: int = 63          # 3 meses de prueba
    step_size: int = 21            # 1 mes de paso
    min_train_periods: int = 126   # 6 meses mínimo
    purge_period: int = 1          # 1 día de purga
    gap_period: int = 0            # Sin gap
    max_periods: int = 10          # Máximo 10 períodos
    retrain_frequency: int = 1     # Reentrenar cada período
    validation_method: str = "expanding"  # "expanding" o "rolling"


class WalkForwardBacktester(BaseBacktester):
    """
    Backtester con Walk-Forward Analysis.
    """
    
    def __init__(self, config: BacktestConfig, walk_forward_config: WalkForwardConfig = None):
        """
        Inicializa el walk-forward backtester.
        
        Args:
            config: Configuración de backtesting
            walk_forward_config: Configuración de walk-forward
        """
        super().__init__(config)
        
        self.walk_forward_config = walk_forward_config or WalkForwardConfig()
        self.periods = []
        self.period_results = []
        self.consolidated_results = None
        
        logger.info(f"Walk-Forward Backtester inicializado: {self.walk_forward_config.train_period} días train, {self.walk_forward_config.test_period} días test")
    
    def _validate_config(self) -> None:
        """
        Valida la configuración de walk-forward.
        """
        if self.walk_forward_config.train_period <= 0:
            raise ValueError("train_period debe ser positivo")
        
        if self.walk_forward_config.test_period <= 0:
            raise ValueError("test_period debe ser positivo")
        
        if self.walk_forward_config.step_size <= 0:
            raise ValueError("step_size debe ser positivo")
        
        if self.walk_forward_config.min_train_periods > self.walk_forward_config.train_period:
            raise ValueError("min_train_periods no puede ser mayor que train_period")
        
        if self.walk_forward_config.purge_period < 0:
            raise ValueError("purge_period no puede ser negativo")
        
        if self.walk_forward_config.gap_period < 0:
            raise ValueError("gap_period no puede ser negativo")
    
    def _generate_signals(self, data: pd.DataFrame, 
                         features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Genera señales de trading (implementación base).
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            
        Returns:
            DataFrame con señales
        """
        # Implementación base - señales aleatorias
        # En implementación real, esto vendría del modelo entrenado
        signals = pd.DataFrame(index=data.index, columns=data.columns)
        
        for symbol in data.columns:
            if symbol in ['open', 'high', 'low', 'close', 'volume']:
                continue
            
            # Generar señales aleatorias para demo
            np.random.seed(42)
            signal_values = np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3])
            signals[symbol] = signal_values
        
        return signals
    
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
        if signal == 0:
            return 0.0
        
        # Calcular tamaño basado en volatilidad
        position_value = current_capital * self.config.max_position_size
        
        # Ajustar por señal
        if signal > 0:
            return position_value
        else:
            return -position_value
    
    def run_walk_forward_backtest(self, data: pd.DataFrame, 
                                 features: pd.DataFrame = None,
                                 benchmark_data: pd.DataFrame = None,
                                 model_trainer = None) -> Dict[str, Any]:
        """
        Ejecuta walk-forward backtesting.
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            benchmark_data: Datos del benchmark (opcional)
            model_trainer: Entrenador de modelos (opcional)
            
        Returns:
            Diccionario con resultados de walk-forward
        """
        logger.info("Iniciando walk-forward backtesting...")
        
        # Validar datos
        self._validate_data(data)
        
        # Filtrar datos por rango de fechas
        data = self._filter_data_by_date(data)
        
        # Generar períodos de walk-forward
        self._generate_walk_forward_periods(data)
        
        # Ejecutar backtesting para cada período
        for i, period in enumerate(self.periods):
            logger.info(f"Ejecutando período {i+1}/{len(self.periods)}: {period['train_start']} - {period['test_end']}")
            
            # Entrenar modelo en período de entrenamiento
            if model_trainer is not None:
                model = self._train_model_for_period(
                    data, features, period, model_trainer
                )
            else:
                model = None
            
            # Ejecutar backtesting en período de prueba
            period_result = self._run_period_backtest(
                data, features, period, model, benchmark_data
            )
            
            self.period_results.append(period_result)
        
        # Consolidar resultados
        self.consolidated_results = self._consolidate_results()
        
        logger.info(f"Walk-forward backtesting completado: {len(self.periods)} períodos")
        
        return self.consolidated_results
    
    def _generate_walk_forward_periods(self, data: pd.DataFrame) -> None:
        """
        Genera períodos de walk-forward.
        
        Args:
            data: Datos de precios
        """
        self.periods = []
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        period_count = 0
        
        while (current_date + timedelta(days=self.walk_forward_config.train_period + 
                                      self.walk_forward_config.test_period) <= end_date and
               period_count < self.walk_forward_config.max_periods):
            
            # Período de entrenamiento
            train_start = current_date
            train_end = current_date + timedelta(days=self.walk_forward_config.train_period - 1)
            
            # Período de purga
            purge_start = train_end + timedelta(days=1)
            purge_end = purge_start + timedelta(days=self.walk_forward_config.purge_period - 1)
            
            # Período de gap
            gap_start = purge_end + timedelta(days=1)
            gap_end = gap_start + timedelta(days=self.walk_forward_config.gap_period - 1)
            
            # Período de prueba
            test_start = gap_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.walk_forward_config.test_period - 1)
            
            # Verificar que el período de prueba no exceda los datos disponibles
            if test_end > end_date:
                break
            
            # Verificar que hay suficientes datos de entrenamiento
            train_data = data.loc[train_start:train_end]
            if len(train_data) < self.walk_forward_config.min_train_periods:
                break
            
            period = {
                'period_id': period_count + 1,
                'train_start': train_start,
                'train_end': train_end,
                'purge_start': purge_start,
                'purge_end': purge_end,
                'gap_start': gap_start,
                'gap_end': gap_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_days': len(train_data),
                'test_days': self.walk_forward_config.test_period
            }
            
            self.periods.append(period)
            
            # Mover al siguiente período
            if self.walk_forward_config.validation_method == "expanding":
                # Expanding window: mantener inicio, expandir fin
                current_date = current_date + timedelta(days=self.walk_forward_config.step_size)
            else:
                # Rolling window: mover ventana
                current_date = current_date + timedelta(days=self.walk_forward_config.step_size)
            
            period_count += 1
        
        logger.info(f"Generados {len(self.periods)} períodos de walk-forward")
    
    def _train_model_for_period(self, data: pd.DataFrame, 
                               features: pd.DataFrame, 
                               period: Dict[str, Any],
                               model_trainer) -> Any:
        """
        Entrena modelo para un período específico.
        
        Args:
            data: Datos de precios
            features: Features adicionales
            period: Período de entrenamiento
            model_trainer: Entrenador de modelos
            
        Returns:
            Modelo entrenado
        """
        # Obtener datos de entrenamiento
        train_data = data.loc[period['train_start']:period['train_end']]
        
        if features is not None:
            train_features = features.loc[period['train_start']:period['train_end']]
        else:
            train_features = None
        
        # Entrenar modelo
        try:
            model = model_trainer.train_model(train_data, train_features)
            logger.debug(f"Modelo entrenado para período {period['period_id']}")
            return model
        except Exception as e:
            logger.warning(f"Error entrenando modelo para período {period['period_id']}: {str(e)}")
            return None
    
    def _run_period_backtest(self, data: pd.DataFrame, 
                           features: pd.DataFrame,
                           period: Dict[str, Any],
                           model: Any,
                           benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Ejecuta backtesting para un período específico.
        
        Args:
            data: Datos de precios
            features: Features adicionales
            period: Período de prueba
            model: Modelo entrenado
            benchmark_data: Datos del benchmark
            
        Returns:
            Resultado del período
        """
        # Obtener datos de prueba
        test_data = data.loc[period['test_start']:period['test_end']]
        
        if features is not None:
            test_features = features.loc[period['test_start']:period['test_end']]
        else:
            test_features = None
        
        # Generar señales
        if model is not None:
            signals = self._generate_signals_with_model(test_data, test_features, model)
        else:
            signals = self._generate_signals(test_data, test_features)
        
        # Obtener datos del benchmark
        if benchmark_data is not None:
            benchmark_test_data = benchmark_data.loc[period['test_start']:period['test_end']]
            benchmark_returns = self._calculate_benchmark_returns(benchmark_test_data)
        else:
            benchmark_returns = pd.Series(dtype=float)
        
        # Ejecutar simulación
        period_trades = []
        period_portfolio_values = pd.Series(index=test_data.index, dtype=float)
        period_portfolio_values.iloc[0] = self.config.initial_capital
        
        current_capital = self.config.initial_capital
        positions = {}
        
        # Procesar cada día
        for i, (date, row) in enumerate(test_data.iterrows()):
            if i == 0:
                continue
            
            # Obtener señales del día
            day_signals = signals.loc[date] if date in signals.index else pd.Series(dtype=float)
            
            # Procesar trades
            day_trades = self._process_period_trades(
                date, row, day_signals, current_capital, positions
            )
            period_trades.extend(day_trades)
            
            # Actualizar capital
            for trade in day_trades:
                current_capital -= trade.commission
            
            # Actualizar valor del portafolio
            position_value = sum(quantity * row.get(symbol, 0) for symbol, quantity in positions.items())
            period_portfolio_values.loc[date] = current_capital + position_value
        
        # Calcular retornos
        period_returns = period_portfolio_values.pct_change().dropna()
        
        # Calcular métricas
        period_metrics = self._calculate_period_metrics(
            period_portfolio_values, period_returns, benchmark_returns, period_trades
        )
        
        return {
            'period_id': period['period_id'],
            'train_start': period['train_start'],
            'train_end': period['train_end'],
            'test_start': period['test_start'],
            'test_end': period['test_end'],
            'trades': period_trades,
            'portfolio_values': period_portfolio_values,
            'returns': period_returns,
            'benchmark_returns': benchmark_returns,
            'metrics': period_metrics,
            'model': model
        }
    
    def _generate_signals_with_model(self, data: pd.DataFrame, 
                                   features: pd.DataFrame,
                                   model: Any) -> pd.DataFrame:
        """
        Genera señales usando modelo entrenado.
        
        Args:
            data: Datos de precios
            features: Features adicionales
            model: Modelo entrenado
            
        Returns:
            DataFrame con señales
        """
        # Implementación base - señales aleatorias
        # En implementación real, esto usaría el modelo para generar señales
        signals = pd.DataFrame(index=data.index, columns=data.columns)
        
        for symbol in data.columns:
            if symbol in ['open', 'high', 'low', 'close', 'volume']:
                continue
            
            # Generar señales aleatorias para demo
            np.random.seed(42 + hash(str(model)) % 1000)
            signal_values = np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3])
            signals[symbol] = signal_values
        
        return signals
    
    def _process_period_trades(self, date: datetime, price_data: pd.Series, 
                             signals: pd.Series, current_capital: float,
                             positions: Dict[str, float]) -> List[Trade]:
        """
        Procesa trades para un día específico en un período.
        
        Args:
            date: Fecha actual
            price_data: Datos de precios del día
            signals: Señales del día
            current_capital: Capital actual
            positions: Posiciones actuales
            
        Returns:
            Lista de trades ejecutados
        """
        trades = []
        
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
                signal, price, symbol, current_capital
            )
            
            if abs(position_size) < self.config.min_trade_size:
                continue
            
            # Crear trade
            direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
            quantity = position_size / price
            
            # Aplicar slippage
            slippage = price * self.config.slippage_rate
            if direction == TradeDirection.LONG:
                entry_price = price + slippage
            else:
                entry_price = price - slippage
            
            # Calcular comisión
            commission = abs(position_size) * self.config.commission_rate
            
            trade = Trade(
                symbol=symbol,
                direction=direction,
                entry_time=date,
                entry_price=entry_price,
                quantity=quantity,
                commission=commission,
                slippage=slippage
            )
            
            trades.append(trade)
            
            # Actualizar posición
            if symbol in positions:
                positions[symbol] += quantity
            else:
                positions[symbol] = quantity
        
        return trades
    
    def _calculate_period_metrics(self, portfolio_values: pd.Series, 
                                returns: pd.Series,
                                benchmark_returns: pd.Series,
                                trades: List[Trade]) -> Dict[str, float]:
        """
        Calcula métricas para un período específico.
        
        Args:
            portfolio_values: Valores del portafolio
            returns: Retornos del portafolio
            benchmark_returns: Retornos del benchmark
            trades: Lista de trades
            
        Returns:
            Diccionario con métricas
        """
        if returns.empty:
            return self._create_empty_period_metrics()
        
        # Métricas básicas
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio_for_period(returns)
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown_for_period(portfolio_values)
        
        # Métricas de trading
        trade_metrics = self._calculate_trade_metrics_for_period(trades)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': trade_metrics['total_trades'],
            'win_rate': trade_metrics['win_rate'],
            'profit_factor': trade_metrics['profit_factor']
        }
    
    def _create_empty_period_metrics(self) -> Dict[str, float]:
        """
        Crea métricas vacías para un período.
        
        Returns:
            Diccionario con métricas vacías
        """
        return {
            'total_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    def _calculate_sharpe_ratio_for_period(self, returns: pd.Series) -> float:
        """
        Calcula Sharpe ratio para un período.
        
        Args:
            returns: Retornos del período
            
        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.config.risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown_for_period(self, portfolio_values: pd.Series) -> float:
        """
        Calcula maximum drawdown para un período.
        
        Args:
            portfolio_values: Valores del portafolio
            
        Returns:
            Maximum drawdown
        """
        if portfolio_values.empty:
            return 0.0
        
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_trade_metrics_for_period(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calcula métricas de trading para un período.
        
        Args:
            trades: Lista de trades
            
        Returns:
            Diccionario con métricas de trading
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Calcular PnL para trades cerrados
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Calcular PnL
        for trade in closed_trades:
            if trade.exit_price is not None:
                if trade.direction == TradeDirection.LONG:
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - trade.commission
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity - trade.commission
        
        # Métricas
        pnls = [t.pnl for t in closed_trades if t.pnl is not None]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _consolidate_results(self) -> Dict[str, Any]:
        """
        Consolida resultados de todos los períodos.
        
        Returns:
            Diccionario con resultados consolidados
        """
        if not self.period_results:
            return {}
        
        # Consolidar métricas
        consolidated_metrics = self._consolidate_metrics()
        
        # Consolidar trades
        all_trades = []
        for result in self.period_results:
            all_trades.extend(result['trades'])
        
        # Consolidar valores del portafolio
        all_portfolio_values = []
        for result in self.period_results:
            all_portfolio_values.append(result['portfolio_values'])
        
        if all_portfolio_values:
            consolidated_portfolio_values = pd.concat(all_portfolio_values)
        else:
            consolidated_portfolio_values = pd.Series(dtype=float)
        
        # Consolidar retornos
        all_returns = []
        for result in self.period_results:
            all_returns.append(result['returns'])
        
        if all_returns:
            consolidated_returns = pd.concat(all_returns)
        else:
            consolidated_returns = pd.Series(dtype=float)
        
        # Consolidar retornos del benchmark
        all_benchmark_returns = []
        for result in self.period_results:
            if not result['benchmark_returns'].empty:
                all_benchmark_returns.append(result['benchmark_returns'])
        
        if all_benchmark_returns:
            consolidated_benchmark_returns = pd.concat(all_benchmark_returns)
        else:
            consolidated_benchmark_returns = pd.Series(dtype=float)
        
        # Crear análisis por período
        period_analysis = self._create_period_analysis()
        
        return {
            'walk_forward_config': self.walk_forward_config.__dict__,
            'periods': self.periods,
            'period_results': self.period_results,
            'consolidated_metrics': consolidated_metrics,
            'all_trades': all_trades,
            'consolidated_portfolio_values': consolidated_portfolio_values,
            'consolidated_returns': consolidated_returns,
            'consolidated_benchmark_returns': consolidated_benchmark_returns,
            'period_analysis': period_analysis,
            'summary': self._create_walk_forward_summary()
        }
    
    def _consolidate_metrics(self) -> Dict[str, Any]:
        """
        Consolida métricas de todos los períodos.
        
        Returns:
            Diccionario con métricas consolidadas
        """
        if not self.period_results:
            return {}
        
        # Extraer métricas de cada período
        metrics_list = [result['metrics'] for result in self.period_results]
        
        # Calcular estadísticas consolidadas
        consolidated = {}
        
        for metric in ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']:
            values = [m[metric] for m in metrics_list if metric in m]
            
            if values:
                consolidated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values
                }
            else:
                consolidated[metric] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'median': 0.0, 'values': []
                }
        
        return consolidated
    
    def _create_period_analysis(self) -> pd.DataFrame:
        """
        Crea análisis por período.
        
        Returns:
            DataFrame con análisis por período
        """
        if not self.period_results:
            return pd.DataFrame()
        
        analysis_data = []
        
        for result in self.period_results:
            analysis_data.append({
                'period_id': result['period_id'],
                'train_start': result['train_start'],
                'train_end': result['train_end'],
                'test_start': result['test_start'],
                'test_end': result['test_end'],
                'total_return': result['metrics']['total_return'],
                'volatility': result['metrics']['volatility'],
                'sharpe_ratio': result['metrics']['sharpe_ratio'],
                'max_drawdown': result['metrics']['max_drawdown'],
                'total_trades': result['metrics']['total_trades'],
                'win_rate': result['metrics']['win_rate'],
                'profit_factor': result['metrics']['profit_factor']
            })
        
        return pd.DataFrame(analysis_data)
    
    def _create_walk_forward_summary(self) -> Dict[str, Any]:
        """
        Crea resumen del walk-forward analysis.
        
        Returns:
            Diccionario con resumen
        """
        if not self.period_results:
            return {}
        
        # Calcular métricas consolidadas
        total_trades = sum(result['metrics']['total_trades'] for result in self.period_results)
        avg_return = np.mean([result['metrics']['total_return'] for result in self.period_results])
        avg_sharpe = np.mean([result['metrics']['sharpe_ratio'] for result in self.period_results])
        avg_drawdown = np.mean([result['metrics']['max_drawdown'] for result in self.period_results])
        
        # Calcular estabilidad
        returns = [result['metrics']['total_return'] for result in self.period_results]
        stability = 1 - np.std(returns) / (np.mean(returns) + 1e-8)
        
        return {
            'total_periods': len(self.period_results),
            'total_trades': total_trades,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_drawdown,
            'stability': stability,
            'best_period': max(self.period_results, key=lambda x: x['metrics']['total_return'])['period_id'],
            'worst_period': min(self.period_results, key=lambda x: x['metrics']['total_return'])['period_id']
        }
    
    def get_period_results(self) -> List[Dict[str, Any]]:
        """
        Obtiene resultados por período.
        
        Returns:
            Lista con resultados por período
        """
        return self.period_results.copy()
    
    def get_consolidated_results(self) -> Dict[str, Any]:
        """
        Obtiene resultados consolidados.
        
        Returns:
            Diccionario con resultados consolidados
        """
        return self.consolidated_results or {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del walk-forward backtesting.
        
        Returns:
            Diccionario con resumen
        """
        summary = super().get_summary()
        summary.update({
            'walk_forward_config': self.walk_forward_config.__dict__,
            'periods_count': len(self.periods),
            'period_results_count': len(self.period_results),
            'consolidated_results': self.consolidated_results is not None
        })
        
        if self.consolidated_results:
            summary.update(self.consolidated_results['summary'])
        
        return summary

