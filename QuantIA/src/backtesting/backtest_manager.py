"""
Gestor principal de backtesting que coordina todos los componentes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings

from .base import BaseBacktester, BacktestConfig, BacktestResult, BacktestMetrics, Trade, TradeDirection, TradeStatus, BacktestType
from .walk_forward import WalkForwardBacktester, WalkForwardConfig
from .purged_cv import PurgedCVBacktester, PurgedCVConfig
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BacktestManager:
    """
    Gestor principal de backtesting que coordina todos los componentes.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el gestor de backtesting.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.backtest_config = self.config.get('backtesting', {})
        
        # Configuración por defecto
        self.default_config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=1000000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.1,
            min_trade_size=100.0,
            max_trades_per_day=10,
            risk_free_rate=0.02,
            benchmark_symbol="SPY",
            rebalance_frequency="1D",
            lookback_window=252,
            min_periods=60,
            purge_period=1,
            gap_period=0,
            enable_short_selling=True,
            enable_leverage=False,
            max_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        
        # Configuración de walk-forward
        self.walk_forward_config = WalkForwardConfig(
            train_period=252,
            test_period=63,
            step_size=21,
            min_train_periods=126,
            purge_period=1,
            gap_period=0,
            max_periods=10,
            retrain_frequency=1,
            validation_method="expanding"
        )
        
        # Configuración de purged CV
        self.purged_cv_config = PurgedCVConfig(
            n_splits=5,
            test_size=0.2,
            purge_period=1,
            gap_period=0,
            min_train_size=252,
            max_train_size=1000,
            validation_method="purged",
            shuffle=False,
            random_state=42
        )
        
        # Estado del sistema
        self.results = {}
        self.current_backtest = None
        self.last_run = None
        
        logger.info("Backtest Manager inicializado")
    
    def run_backtest(self, data: pd.DataFrame, 
                    features: pd.DataFrame = None,
                    benchmark_data: pd.DataFrame = None,
                    backtest_type: BacktestType = BacktestType.WALK_FORWARD,
                    model_trainer = None,
                    custom_config: BacktestConfig = None) -> Dict[str, Any]:
        """
        Ejecuta backtesting.
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            benchmark_data: Datos del benchmark (opcional)
            backtest_type: Tipo de backtesting
            model_trainer: Entrenador de modelos (opcional)
            custom_config: Configuración personalizada (opcional)
            
        Returns:
            Diccionario con resultados del backtesting
        """
        logger.info(f"Iniciando backtesting: {backtest_type.value}")
        
        # Usar configuración personalizada o por defecto
        config = custom_config or self.default_config
        
        # Crear backtester según tipo
        if backtest_type == BacktestType.WALK_FORWARD:
            backtester = WalkForwardBacktester(config, self.walk_forward_config)
            results = backtester.run_walk_forward_backtest(
                data, features, benchmark_data, model_trainer
            )
        elif backtest_type == BacktestType.PURGED_CV:
            backtester = PurgedCVBacktester(config, self.purged_cv_config)
            results = backtester.run_purged_cv_backtest(
                data, features, benchmark_data, model_trainer
            )
        else:
            raise ValueError(f"Tipo de backtesting no soportado: {backtest_type}")
        
        # Guardar resultados
        self.results[backtest_type.value] = results
        self.current_backtest = backtester
        self.last_run = datetime.now()
        
        logger.info(f"Backtesting completado: {backtest_type.value}")
        
        return results
    
    def run_comprehensive_backtest(self, data: pd.DataFrame, 
                                 features: pd.DataFrame = None,
                                 benchmark_data: pd.DataFrame = None,
                                 model_trainer = None,
                                 custom_config: BacktestConfig = None) -> Dict[str, Any]:
        """
        Ejecuta backtesting comprensivo con múltiples métodos.
        
        Args:
            data: Datos de precios
            features: Features adicionales (opcional)
            benchmark_data: Datos del benchmark (opcional)
            model_trainer: Entrenador de modelos (opcional)
            custom_config: Configuración personalizada (opcional)
            
        Returns:
            Diccionario con resultados comprensivos
        """
        logger.info("Iniciando backtesting comprensivo...")
        
        comprehensive_results = {}
        
        # Walk-Forward Analysis
        logger.info("Ejecutando Walk-Forward Analysis...")
        wf_results = self.run_backtest(
            data, features, benchmark_data, 
            BacktestType.WALK_FORWARD, model_trainer, custom_config
        )
        comprehensive_results['walk_forward'] = wf_results
        
        # Purged Cross-Validation
        logger.info("Ejecutando Purged Cross-Validation...")
        cv_results = self.run_backtest(
            data, features, benchmark_data, 
            BacktestType.PURGED_CV, model_trainer, custom_config
        )
        comprehensive_results['purged_cv'] = cv_results
        
        # Comparar resultados
        comparison = self._compare_backtest_results(comprehensive_results)
        comprehensive_results['comparison'] = comparison
        
        # Generar resumen
        summary = self._generate_comprehensive_summary(comprehensive_results)
        comprehensive_results['summary'] = summary
        
        logger.info("Backtesting comprensivo completado")
        
        return comprehensive_results
    
    def _compare_backtest_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara resultados de diferentes métodos de backtesting.
        
        Args:
            results: Resultados de backtesting
            
        Returns:
            Diccionario con comparación
        """
        comparison = {}
        
        # Comparar métricas consolidadas
        if 'walk_forward' in results and 'purged_cv' in results:
            wf_metrics = results['walk_forward'].get('consolidated_metrics', {})
            cv_metrics = results['purged_cv'].get('consolidated_metrics', {})
            
            # Comparar métricas clave
            key_metrics = ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            
            for metric in key_metrics:
                if metric in wf_metrics and metric in cv_metrics:
                    wf_value = wf_metrics[metric].get('mean', 0.0)
                    cv_value = cv_metrics[metric].get('mean', 0.0)
                    
                    comparison[metric] = {
                        'walk_forward': wf_value,
                        'purged_cv': cv_value,
                        'difference': wf_value - cv_value,
                        'relative_difference': (wf_value - cv_value) / (cv_value + 1e-8)
                    }
        
        # Comparar estabilidad
        if 'walk_forward' in results and 'purged_cv' in results:
            wf_stability = results['walk_forward'].get('summary', {}).get('stability', 0.0)
            cv_consistency = results['purged_cv'].get('summary', {}).get('consistency', 0.0)
            
            comparison['stability'] = {
                'walk_forward': wf_stability,
                'purged_cv': cv_consistency,
                'difference': wf_stability - cv_consistency
            }
        
        return comparison
    
    def _generate_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera resumen comprensivo de todos los resultados.
        
        Args:
            results: Resultados de backtesting
            
        Returns:
            Diccionario con resumen comprensivo
        """
        summary = {
            'total_methods': len(results),
            'methods_used': list(results.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Resumen por método
        for method, result in results.items():
            if method in ['comparison', 'summary']:
                continue
            
            method_summary = result.get('summary', {})
            summary[f'{method}_summary'] = method_summary
        
        # Recomendaciones
        recommendations = self._generate_recommendations(results)
        summary['recommendations'] = recommendations
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones basadas en los resultados.
        
        Args:
            results: Resultados de backtesting
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar estabilidad
        if 'walk_forward' in results:
            wf_stability = results['walk_forward'].get('summary', {}).get('stability', 0.0)
            if wf_stability < 0.5:
                recommendations.append("La estabilidad del Walk-Forward es baja. Considerar ajustar parámetros del modelo.")
        
        if 'purged_cv' in results:
            cv_consistency = results['purged_cv'].get('summary', {}).get('consistency', 0.0)
            if cv_consistency < 0.6:
                recommendations.append("La consistencia del Purged CV es baja. El modelo puede estar sobreajustado.")
        
        # Analizar drawdown
        if 'walk_forward' in results:
            wf_drawdown = results['walk_forward'].get('consolidated_metrics', {}).get('max_drawdown', {}).get('mean', 0.0)
            if abs(wf_drawdown) > 0.15:
                recommendations.append("El drawdown máximo es alto. Considerar implementar stops más agresivos.")
        
        # Analizar Sharpe ratio
        if 'walk_forward' in results:
            wf_sharpe = results['walk_forward'].get('consolidated_metrics', {}).get('sharpe_ratio', {}).get('mean', 0.0)
            if wf_sharpe < 1.0:
                recommendations.append("El Sharpe ratio es bajo. Considerar optimizar el modelo o ajustar el riesgo.")
        
        # Analizar win rate
        if 'walk_forward' in results:
            wf_win_rate = results['walk_forward'].get('consolidated_metrics', {}).get('win_rate', {}).get('mean', 0.0)
            if wf_win_rate < 0.4:
                recommendations.append("El win rate es bajo. Considerar mejorar la selección de señales.")
        
        if not recommendations:
            recommendations.append("Los resultados del backtesting son satisfactorios. El modelo está listo para paper trading.")
        
        return recommendations
    
    def get_results(self, backtest_type: str = None) -> Dict[str, Any]:
        """
        Obtiene resultados de backtesting.
        
        Args:
            backtest_type: Tipo de backtesting (opcional)
            
        Returns:
            Diccionario con resultados
        """
        if backtest_type:
            return self.results.get(backtest_type, {})
        return self.results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del gestor de backtesting.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.default_config.__dict__,
            'walk_forward_config': self.walk_forward_config.__dict__,
            'purged_cv_config': self.purged_cv_config.__dict__,
            'results_count': len(self.results),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'current_backtest': self.current_backtest is not None
        }
    
    def save_results(self, filepath: str, backtest_type: str = None) -> None:
        """
        Guarda resultados de backtesting.
        
        Args:
            filepath: Ruta del archivo
            backtest_type: Tipo de backtesting (opcional)
        """
        if backtest_type:
            results = self.results.get(backtest_type, {})
        else:
            results = self.results
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(results, filepath)
        logger.info(f"Resultados de backtesting guardados: {filepath}")
    
    def load_results(self, filepath: str, backtest_type: str = None) -> None:
        """
        Carga resultados de backtesting.
        
        Args:
            filepath: Ruta del archivo
            backtest_type: Tipo de backtesting (opcional)
        """
        results = joblib.load(filepath)
        
        if backtest_type:
            self.results[backtest_type] = results
        else:
            self.results = results
        
        logger.info(f"Resultados de backtesting cargados: {filepath}")
    
    def create_report(self, backtest_type: str = None) -> str:
        """
        Crea reporte de backtesting.
        
        Args:
            backtest_type: Tipo de backtesting (opcional)
            
        Returns:
            Reporte en formato texto
        """
        if backtest_type:
            results = self.results.get(backtest_type, {})
            if not results:
                return f"No hay resultados para {backtest_type}"
        else:
            results = self.results
            if not results:
                return "No hay resultados de backtesting"
        
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE BACKTESTING")
        report.append("=" * 80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if backtest_type:
            report.append(f"Tipo: {backtest_type.upper()}")
            report.append("")
            self._add_method_report(report, backtest_type, results)
        else:
            report.append("MÉTODOS EJECUTADOS:")
            for method in results.keys():
                if method not in ['comparison', 'summary']:
                    report.append(f"- {method.upper()}")
            report.append("")
            
            # Reporte por método
            for method, result in results.items():
                if method not in ['comparison', 'summary']:
                    report.append(f"\n{method.upper()}:")
                    report.append("-" * 40)
                    self._add_method_report(report, method, result)
            
            # Comparación
            if 'comparison' in results:
                report.append("\nCOMPARACIÓN:")
                report.append("-" * 40)
                self._add_comparison_report(report, results['comparison'])
            
            # Recomendaciones
            if 'summary' in results and 'recommendations' in results['summary']:
                report.append("\nRECOMENDACIONES:")
                report.append("-" * 40)
                for i, rec in enumerate(results['summary']['recommendations'], 1):
                    report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _add_method_report(self, report: List[str], method: str, results: Dict[str, Any]) -> None:
        """
        Agrega reporte de un método específico.
        
        Args:
            report: Lista de líneas del reporte
            method: Método de backtesting
            results: Resultados del método
        """
        # Resumen
        if 'summary' in results:
            summary = results['summary']
            report.append(f"Períodos/Folds: {summary.get('total_periods', summary.get('total_folds', 0))}")
            report.append(f"Total Trades: {summary.get('total_trades', 0)}")
            report.append(f"Retorno Promedio: {summary.get('avg_return', 0.0):.2%}")
            report.append(f"Sharpe Promedio: {summary.get('avg_sharpe', 0.0):.2f}")
            report.append(f"Drawdown Promedio: {summary.get('avg_drawdown', 0.0):.2%}")
            
            if 'stability' in summary:
                report.append(f"Estabilidad: {summary['stability']:.2%}")
            if 'consistency' in summary:
                report.append(f"Consistencia: {summary['consistency']:.2%}")
        
        # Métricas consolidadas
        if 'consolidated_metrics' in results:
            metrics = results['consolidated_metrics']
            report.append("\nMétricas Consolidadas:")
            
            for metric, stats in metrics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    report.append(f"  {metric}: {stats['mean']:.2%} (±{stats['std']:.2%})")
    
    def _add_comparison_report(self, report: List[str], comparison: Dict[str, Any]) -> None:
        """
        Agrega reporte de comparación.
        
        Args:
            report: Lista de líneas del reporte
            comparison: Resultados de comparación
        """
        for metric, data in comparison.items():
            if isinstance(data, dict) and 'walk_forward' in data and 'purged_cv' in data:
                wf_val = data['walk_forward']
                cv_val = data['purged_cv']
                diff = data.get('difference', 0.0)
                
                report.append(f"{metric}:")
                report.append(f"  Walk-Forward: {wf_val:.2%}")
                report.append(f"  Purged CV: {cv_val:.2%}")
                report.append(f"  Diferencia: {diff:.2%}")
                report.append("")
    
    def export_results(self, filepath: str, format: str = 'csv', 
                      backtest_type: str = None) -> None:
        """
        Exporta resultados a archivo.
        
        Args:
            filepath: Ruta del archivo
            format: Formato de exportación ('csv', 'excel', 'json')
            backtest_type: Tipo de backtesting (opcional)
        """
        if backtest_type:
            results = self.results.get(backtest_type, {})
        else:
            results = self.results
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            self._export_to_csv(results, filepath)
        elif format == 'excel':
            self._export_to_excel(results, filepath)
        elif format == 'json':
            self._export_to_json(results, filepath)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Resultados exportados: {filepath}")
    
    def _export_to_csv(self, results: Dict[str, Any], filepath: Path) -> None:
        """
        Exporta resultados a CSV.
        
        Args:
            results: Resultados de backtesting
            filepath: Ruta del archivo
        """
        # Exportar análisis por período/fold
        for method, result in results.items():
            if method in ['comparison', 'summary']:
                continue
            
            if 'period_analysis' in result:
                analysis_df = result['period_analysis']
                analysis_file = filepath.parent / f"{filepath.stem}_{method}_analysis.csv"
                analysis_df.to_csv(analysis_file, index=False)
            
            if 'fold_analysis' in result:
                analysis_df = result['fold_analysis']
                analysis_file = filepath.parent / f"{filepath.stem}_{method}_analysis.csv"
                analysis_df.to_csv(analysis_file, index=False)
    
    def _export_to_excel(self, results: Dict[str, Any], filepath: Path) -> None:
        """
        Exporta resultados a Excel.
        
        Args:
            results: Resultados de backtesting
            filepath: Ruta del archivo
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Resumen
            summary_data = []
            for method, result in results.items():
                if method in ['comparison', 'summary']:
                    continue
                
                if 'summary' in result:
                    summary = result['summary']
                    summary_data.append({
                        'Método': method,
                        'Períodos/Folds': summary.get('total_periods', summary.get('total_folds', 0)),
                        'Total Trades': summary.get('total_trades', 0),
                        'Retorno Promedio': summary.get('avg_return', 0.0),
                        'Sharpe Promedio': summary.get('avg_sharpe', 0.0),
                        'Drawdown Promedio': summary.get('avg_drawdown', 0.0)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Análisis por método
            for method, result in results.items():
                if method in ['comparison', 'summary']:
                    continue
                
                if 'period_analysis' in result:
                    result['period_analysis'].to_excel(writer, sheet_name=f'{method}_periods', index=False)
                
                if 'fold_analysis' in result:
                    result['fold_analysis'].to_excel(writer, sheet_name=f'{method}_folds', index=False)
    
    def _export_to_json(self, results: Dict[str, Any], filepath: Path) -> None:
        """
        Exporta resultados a JSON.
        
        Args:
            results: Resultados de backtesting
            filepath: Ruta del archivo
        """
        import json
        
        # Convertir objetos no serializables
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convierte objetos no serializables a serializables.
        
        Args:
            obj: Objeto a convertir
            
        Returns:
            Objeto serializable
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

