"""
Backtesting con Cross-Validation Purgada.
Implementa validación cruzada temporal con purga de datos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseBacktester, BacktestConfig, BacktestResult, BacktestMetrics, Trade, TradeDirection, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class PurgedCVConfig:
    """Configuración de cross-validation purgada."""
    n_splits: int = 5                    # 5 folds
    test_size: float = 0.2               # 20% para prueba
    purge_period: int = 1                # 1 día de purga
    gap_period: int = 0                  # Sin gap
    min_train_size: int = 252            # 1 año mínimo de entrenamiento
    max_train_size: int = 1000           # Máximo 4 años de entrenamiento
    validation_method: str = "purged"    # "purged" o "standard"
    shuffle: bool = False                # No shuffle para series temporales
    random_state: int = 42


class PurgedCVBacktester(BaseBacktester):
    """
    Backtester con Cross-Validation Purgada.
    """
    
    def __init__(self, config: BacktestConfig, purged_cv_config: PurgedCVConfig = None):
        """
        Inicializa el purged CV backtester.
        
        Args:
            config: Configuración de backtesting
            purged_cv_config: Configuración de purged CV
        """
        super().__init__(config)
        
        self.purged_cv_config = purged_cv_config or PurgedCVConfig()
        self.cv_splits = []
        self.cv_results = []
        self.consolidated_results = None
        
        logger.info(f"Purged CV Backtester inicializado: {self.purged_cv_config.n_splits} folds")
    
    def _validate_config(self) -> None:
        """
        Valida la configuración de purged CV.
        """
        if self.purged_cv_config.n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2")
        
        if self.purged_cv_config.test_size <= 0 or self.purged_cv_config.test_size >= 1:
            raise ValueError("test_size debe estar entre 0 y 1")
        
        if self.purged_cv_config.purge_period < 0:
            raise ValueError("purge_period no puede ser negativo")
        
        if self.purged_cv_config.gap_period < 0:
            raise ValueError("gap_period no puede ser negativo")
        
        if self.purged_cv_config.min_train_size <= 0:
            raise ValueError("min_train_size debe ser positivo")
        
        if self.purged_cv_config.max_train_size <= self.purged_cv_config.min_train_size:
            raise ValueError("max_train_size debe ser mayor que min_train_size")
    
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
    
    def run_purged_cv_backtest(self, data: pd.DataFrame, 
                              features: pd.DataFrame = None,
                              benchmark_data: pd.DataFrame = None,
                              model_trainer = None) -> Dict[str, Any]:
        """
        Ejecuta purged cross-validation backtesting.
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            benchmark_data: Datos del benchmark (opcional)
            model_trainer: Entrenador de modelos (opcional)
            
        Returns:
            Diccionario con resultados de purged CV
        """
        logger.info("Iniciando purged cross-validation backtesting...")
        
        # Validar datos
        self._validate_data(data)
        
        # Filtrar datos por rango de fechas
        data = self._filter_data_by_date(data)
        
        # Generar splits de cross-validation
        self._generate_cv_splits(data)
        
        # Ejecutar backtesting para cada fold
        for i, split in enumerate(self.cv_splits):
            logger.info(f"Ejecutando fold {i+1}/{len(self.cv_splits)}: {split['train_start']} - {split['test_end']}")
            
            # Entrenar modelo en fold de entrenamiento
            if model_trainer is not None:
                model = self._train_model_for_fold(
                    data, features, split, model_trainer
                )
            else:
                model = None
            
            # Ejecutar backtesting en fold de prueba
            fold_result = self._run_fold_backtest(
                data, features, split, model, benchmark_data
            )
            
            self.cv_results.append(fold_result)
        
        # Consolidar resultados
        self.consolidated_results = self._consolidate_cv_results()
        
        logger.info(f"Purged CV backtesting completado: {len(self.cv_splits)} folds")
        
        return self.consolidated_results
    
    def _generate_cv_splits(self, data: pd.DataFrame) -> None:
        """
        Genera splits de cross-validation purgada.
        
        Args:
            data: Datos de precios
        """
        self.cv_splits = []
        
        # Crear TimeSeriesSplit
        tscv = TimeSeriesSplit(
            n_splits=self.purged_cv_config.n_splits,
            test_size=int(len(data) * self.purged_cv_config.test_size),
            gap=self.purged_cv_config.gap_period
        )
        
        # Generar splits
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
            # Obtener fechas
            train_start = data.index[train_idx[0]]
            train_end = data.index[train_idx[-1]]
            test_start = data.index[test_idx[0]]
            test_end = data.index[test_idx[-1]]
            
            # Verificar tamaño mínimo de entrenamiento
            train_size = len(train_idx)
            if train_size < self.purged_cv_config.min_train_size:
                logger.warning(f"Fold {fold_idx + 1}: Tamaño de entrenamiento ({train_size}) menor al mínimo ({self.purged_cv_config.min_train_size})")
                continue
            
            # Limitar tamaño máximo de entrenamiento
            if train_size > self.purged_cv_config.max_train_size:
                # Truncar desde el final
                train_start = data.index[train_idx[-self.purged_cv_config.max_train_size]]
                train_size = self.purged_cv_config.max_train_size
            
            # Aplicar purga
            if self.purged_cv_config.purge_period > 0:
                purge_start = train_end + timedelta(days=1)
                purge_end = purge_start + timedelta(days=self.purged_cv_config.purge_period - 1)
                
                # Verificar que la purga no interfiera con el test
                if purge_end >= test_start:
                    logger.warning(f"Fold {fold_idx + 1}: Purga interfiere con test, ajustando...")
                    test_start = purge_end + timedelta(days=1)
            
            # Verificar que el test no exceda los datos disponibles
            if test_start > data.index[-1]:
                logger.warning(f"Fold {fold_idx + 1}: Test excede datos disponibles, saltando...")
                continue
            
            # Ajustar test_end si es necesario
            if test_end > data.index[-1]:
                test_end = data.index[-1]
            
            split = {
                'fold_id': fold_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'purge_start': purge_start if self.purged_cv_config.purge_period > 0 else None,
                'purge_end': purge_end if self.purged_cv_config.purge_period > 0 else None,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': train_size,
                'test_size': len(test_idx)
            }
            
            self.cv_splits.append(split)
        
        logger.info(f"Generados {len(self.cv_splits)} folds de cross-validation")
    
    def _train_model_for_fold(self, data: pd.DataFrame, 
                            features: pd.DataFrame, 
                            split: Dict[str, Any],
                            model_trainer) -> Any:
        """
        Entrena modelo para un fold específico.
        
        Args:
            data: Datos de precios
            features: Features adicionales
            split: Split de cross-validation
            model_trainer: Entrenador de modelos
            
        Returns:
            Modelo entrenado
        """
        # Obtener datos de entrenamiento
        train_data = data.loc[split['train_start']:split['train_end']]
        
        if features is not None:
            train_features = features.loc[split['train_start']:split['train_end']]
        else:
            train_features = None
        
        # Entrenar modelo
        try:
            model = model_trainer.train_model(train_data, train_features)
            logger.debug(f"Modelo entrenado para fold {split['fold_id']}")
            return model
        except Exception as e:
            logger.warning(f"Error entrenando modelo para fold {split['fold_id']}: {str(e)}")
            return None
    
    def _run_fold_backtest(self, data: pd.DataFrame, 
                         features: pd.DataFrame,
                         split: Dict[str, Any],
                         model: Any,
                         benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Ejecuta backtesting para un fold específico.
        
        Args:
            data: Datos de precios
            features: Features adicionales
            split: Split de cross-validation
            model: Modelo entrenado
            benchmark_data: Datos del benchmark
            
        Returns:
            Resultado del fold
        """
        # Obtener datos de prueba
        test_data = data.loc[split['test_start']:split['test_end']]
        
        if features is not None:
            test_features = features.loc[split['test_start']:split['test_end']]
        else:
            test_features = None
        
        # Generar señales
        if model is not None:
            signals = self._generate_signals_with_model(test_data, test_features, model)
        else:
            signals = self._generate_signals(test_data, test_features)
        
        # Obtener datos del benchmark
        if benchmark_data is not None:
            benchmark_test_data = benchmark_data.loc[split['test_start']:split['test_end']]
            benchmark_returns = self._calculate_benchmark_returns(benchmark_test_data)
        else:
            benchmark_returns = pd.Series(dtype=float)
        
        # Ejecutar simulación
        fold_trades = []
        fold_portfolio_values = pd.Series(index=test_data.index, dtype=float)
        fold_portfolio_values.iloc[0] = self.config.initial_capital
        
        current_capital = self.config.initial_capital
        positions = {}
        
        # Procesar cada día
        for i, (date, row) in enumerate(test_data.iterrows()):
            if i == 0:
                continue
            
            # Obtener señales del día
            day_signals = signals.loc[date] if date in signals.index else pd.Series(dtype=float)
            
            # Procesar trades
            day_trades = self._process_fold_trades(
                date, row, day_signals, current_capital, positions
            )
            fold_trades.extend(day_trades)
            
            # Actualizar capital
            for trade in day_trades:
                current_capital -= trade.commission
            
            # Actualizar valor del portafolio
            position_value = sum(quantity * row.get(symbol, 0) for symbol, quantity in positions.items())
            fold_portfolio_values.loc[date] = current_capital + position_value
        
        # Calcular retornos
        fold_returns = fold_portfolio_values.pct_change().dropna()
        
        # Calcular métricas
        fold_metrics = self._calculate_fold_metrics(
            fold_portfolio_values, fold_returns, benchmark_returns, fold_trades
        )
        
        return {
            'fold_id': split['fold_id'],
            'train_start': split['train_start'],
            'train_end': split['train_end'],
            'test_start': split['test_start'],
            'test_end': split['test_end'],
            'trades': fold_trades,
            'portfolio_values': fold_portfolio_values,
            'returns': fold_returns,
            'benchmark_returns': benchmark_returns,
            'metrics': fold_metrics,
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
    
    def _process_fold_trades(self, date: datetime, price_data: pd.Series, 
                           signals: pd.Series, current_capital: float,
                           positions: Dict[str, float]) -> List[Trade]:
        """
        Procesa trades para un día específico en un fold.
        
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
    
    def _calculate_fold_metrics(self, portfolio_values: pd.Series, 
                              returns: pd.Series,
                              benchmark_returns: pd.Series,
                              trades: List[Trade]) -> Dict[str, float]:
        """
        Calcula métricas para un fold específico.
        
        Args:
            portfolio_values: Valores del portafolio
            returns: Retornos del portafolio
            benchmark_returns: Retornos del benchmark
            trades: Lista de trades
            
        Returns:
            Diccionario con métricas
        """
        if returns.empty:
            return self._create_empty_fold_metrics()
        
        # Métricas básicas
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio_for_fold(returns)
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown_for_fold(portfolio_values)
        
        # Métricas de trading
        trade_metrics = self._calculate_trade_metrics_for_fold(trades)
        
        # Métricas de benchmark
        benchmark_metrics = self._calculate_benchmark_metrics_for_fold(returns, benchmark_returns)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': trade_metrics['total_trades'],
            'win_rate': trade_metrics['win_rate'],
            'profit_factor': trade_metrics['profit_factor'],
            'excess_return': benchmark_metrics['excess_return'],
            'information_ratio': benchmark_metrics['information_ratio']
        }
    
    def _create_empty_fold_metrics(self) -> Dict[str, float]:
        """
        Crea métricas vacías para un fold.
        
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
            'profit_factor': 0.0,
            'excess_return': 0.0,
            'information_ratio': 0.0
        }
    
    def _calculate_sharpe_ratio_for_fold(self, returns: pd.Series) -> float:
        """
        Calcula Sharpe ratio para un fold.
        
        Args:
            returns: Retornos del fold
            
        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.config.risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown_for_fold(self, portfolio_values: pd.Series) -> float:
        """
        Calcula maximum drawdown para un fold.
        
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
    
    def _calculate_trade_metrics_for_fold(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calcula métricas de trading para un fold.
        
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
    
    def _calculate_benchmark_metrics_for_fold(self, returns: pd.Series, 
                                            benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calcula métricas de benchmark para un fold.
        
        Args:
            returns: Retornos del portafolio
            benchmark_returns: Retornos del benchmark
            
        Returns:
            Diccionario con métricas de benchmark
        """
        if returns.empty or benchmark_returns.empty:
            return {
                'excess_return': 0.0,
                'information_ratio': 0.0
            }
        
        # Alinear series
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 2:
            return {
                'excess_return': 0.0,
                'information_ratio': 0.0
            }
        
        portfolio_ret = aligned_returns.iloc[:, 0]
        benchmark_ret = aligned_returns.iloc[:, 1]
        
        # Excess return
        excess_return = portfolio_ret.mean() - benchmark_ret.mean()
        
        # Information ratio
        excess_returns = portfolio_ret - benchmark_ret
        tracking_error = excess_returns.std()
        information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'excess_return': excess_return,
            'information_ratio': information_ratio
        }
    
    def _consolidate_cv_results(self) -> Dict[str, Any]:
        """
        Consolida resultados de todos los folds.
        
        Returns:
            Diccionario con resultados consolidados
        """
        if not self.cv_results:
            return {}
        
        # Consolidar métricas
        consolidated_metrics = self._consolidate_cv_metrics()
        
        # Consolidar trades
        all_trades = []
        for result in self.cv_results:
            all_trades.extend(result['trades'])
        
        # Consolidar valores del portafolio
        all_portfolio_values = []
        for result in self.cv_results:
            all_portfolio_values.append(result['portfolio_values'])
        
        if all_portfolio_values:
            consolidated_portfolio_values = pd.concat(all_portfolio_values)
        else:
            consolidated_portfolio_values = pd.Series(dtype=float)
        
        # Consolidar retornos
        all_returns = []
        for result in self.cv_results:
            all_returns.append(result['returns'])
        
        if all_returns:
            consolidated_returns = pd.concat(all_returns)
        else:
            consolidated_returns = pd.Series(dtype=float)
        
        # Consolidar retornos del benchmark
        all_benchmark_returns = []
        for result in self.cv_results:
            if not result['benchmark_returns'].empty:
                all_benchmark_returns.append(result['benchmark_returns'])
        
        if all_benchmark_returns:
            consolidated_benchmark_returns = pd.concat(all_benchmark_returns)
        else:
            consolidated_benchmark_returns = pd.Series(dtype=float)
        
        # Crear análisis por fold
        fold_analysis = self._create_fold_analysis()
        
        return {
            'purged_cv_config': self.purged_cv_config.__dict__,
            'cv_splits': self.cv_splits,
            'cv_results': self.cv_results,
            'consolidated_metrics': consolidated_metrics,
            'all_trades': all_trades,
            'consolidated_portfolio_values': consolidated_portfolio_values,
            'consolidated_returns': consolidated_returns,
            'consolidated_benchmark_returns': consolidated_benchmark_returns,
            'fold_analysis': fold_analysis,
            'summary': self._create_purged_cv_summary()
        }
    
    def _consolidate_cv_metrics(self) -> Dict[str, Any]:
        """
        Consolida métricas de todos los folds.
        
        Returns:
            Diccionario con métricas consolidadas
        """
        if not self.cv_results:
            return {}
        
        # Extraer métricas de cada fold
        metrics_list = [result['metrics'] for result in self.cv_results]
        
        # Calcular estadísticas consolidadas
        consolidated = {}
        
        for metric in ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 
                      'win_rate', 'profit_factor', 'excess_return', 'information_ratio']:
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
    
    def _create_fold_analysis(self) -> pd.DataFrame:
        """
        Crea análisis por fold.
        
        Returns:
            DataFrame con análisis por fold
        """
        if not self.cv_results:
            return pd.DataFrame()
        
        analysis_data = []
        
        for result in self.cv_results:
            analysis_data.append({
                'fold_id': result['fold_id'],
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
                'profit_factor': result['metrics']['profit_factor'],
                'excess_return': result['metrics']['excess_return'],
                'information_ratio': result['metrics']['information_ratio']
            })
        
        return pd.DataFrame(analysis_data)
    
    def _create_purged_cv_summary(self) -> Dict[str, Any]:
        """
        Crea resumen del purged cross-validation.
        
        Returns:
            Diccionario con resumen
        """
        if not self.cv_results:
            return {}
        
        # Calcular métricas consolidadas
        total_trades = sum(result['metrics']['total_trades'] for result in self.cv_results)
        avg_return = np.mean([result['metrics']['total_return'] for result in self.cv_results])
        avg_sharpe = np.mean([result['metrics']['sharpe_ratio'] for result in self.cv_results])
        avg_drawdown = np.mean([result['metrics']['max_drawdown'] for result in self.cv_results])
        
        # Calcular estabilidad
        returns = [result['metrics']['total_return'] for result in self.cv_results]
        stability = 1 - np.std(returns) / (np.mean(returns) + 1e-8)
        
        # Calcular consistencia
        positive_returns = sum(1 for r in returns if r > 0)
        consistency = positive_returns / len(returns) if returns else 0.0
        
        return {
            'total_folds': len(self.cv_results),
            'total_trades': total_trades,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_drawdown,
            'stability': stability,
            'consistency': consistency,
            'best_fold': max(self.cv_results, key=lambda x: x['metrics']['total_return'])['fold_id'],
            'worst_fold': min(self.cv_results, key=lambda x: x['metrics']['total_return'])['fold_id']
        }
    
    def get_cv_results(self) -> List[Dict[str, Any]]:
        """
        Obtiene resultados por fold.
        
        Returns:
            Lista con resultados por fold
        """
        return self.cv_results.copy()
    
    def get_consolidated_results(self) -> Dict[str, Any]:
        """
        Obtiene resultados consolidados.
        
        Returns:
            Diccionario con resultados consolidados
        """
        return self.consolidated_results or {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del purged CV backtesting.
        
        Returns:
            Diccionario con resumen
        """
        summary = super().get_summary()
        summary.update({
            'purged_cv_config': self.purged_cv_config.__dict__,
            'cv_splits_count': len(self.cv_splits),
            'cv_results_count': len(self.cv_results),
            'consolidated_results': self.consolidated_results is not None
        })
        
        if self.consolidated_results:
            summary.update(self.consolidated_results['summary'])
        
        return summary

