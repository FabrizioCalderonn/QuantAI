"""
Gestor principal de validación que coordina todos los componentes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings

from .metrics import ValidationMetrics, MetricType, ValidationLevel
from .validator import StrategyValidator, ValidationResult, ValidationStatus
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationManager:
    """
    Gestor principal de validación que coordina todos los componentes.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el gestor de validación.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.validation_config = self.config.get('validation', {})
        
        # Inicializar componentes
        self.metrics_calculator = ValidationMetrics(self.validation_config)
        self.validator = StrategyValidator(self.validation_config)
        
        # Estado del sistema
        self.validation_results = {}
        self.last_validation = None
        
        logger.info("ValidationManager inicializado")
    
    def validate_strategy(self, strategy_name: str, 
                         returns: pd.Series,
                         benchmark_returns: pd.Series = None,
                         portfolio_values: pd.Series = None,
                         trades: List[Dict[str, Any]] = None,
                         validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> ValidationResult:
        """
        Valida una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            returns: Serie de retornos
            benchmark_returns: Retornos del benchmark (opcional)
            portfolio_values: Valores del portafolio (opcional)
            trades: Lista de trades (opcional)
            validation_level: Nivel de validación
            
        Returns:
            Resultado de validación
        """
        logger.info(f"Validando estrategia: {strategy_name}")
        
        result = self.validator.validate_strategy(
            strategy_name=strategy_name,
            returns=returns,
            benchmark_returns=benchmark_returns,
            portfolio_values=portfolio_values,
            trades=trades,
            validation_level=validation_level
        )
        
        # Guardar resultado
        self.validation_results[strategy_name] = result
        self.last_validation = datetime.now()
        
        return result
    
    def validate_multiple_strategies(self, strategies: Dict[str, Dict[str, Any]], 
                                   validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> Dict[str, ValidationResult]:
        """
        Valida múltiples estrategias.
        
        Args:
            strategies: Diccionario con estrategias
            validation_level: Nivel de validación
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info(f"Validando {len(strategies)} estrategias")
        
        results = self.validator.validate_multiple_strategies(strategies, validation_level)
        
        # Guardar resultados
        self.validation_results.update(results)
        self.last_validation = datetime.now()
        
        return results
    
    def validate_backtest_results(self, backtest_results: Dict[str, Any], 
                                validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> Dict[str, ValidationResult]:
        """
        Valida resultados de backtesting.
        
        Args:
            backtest_results: Resultados de backtesting
            validation_level: Nivel de validación
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Validando resultados de backtesting")
        
        strategies = {}
        
        # Procesar resultados de walk-forward
        if 'walk_forward' in backtest_results:
            wf_result = backtest_results['walk_forward']
            strategies['walk_forward'] = {
                'returns': wf_result.get('consolidated_returns'),
                'benchmark_returns': wf_result.get('consolidated_benchmark_returns'),
                'portfolio_values': wf_result.get('consolidated_portfolio_values'),
                'trades': wf_result.get('all_trades')
            }
        
        # Procesar resultados de purged CV
        if 'purged_cv' in backtest_results:
            cv_result = backtest_results['purged_cv']
            strategies['purged_cv'] = {
                'returns': cv_result.get('consolidated_returns'),
                'benchmark_returns': cv_result.get('consolidated_benchmark_returns'),
                'portfolio_values': cv_result.get('consolidated_portfolio_values'),
                'trades': cv_result.get('all_trades')
            }
        
        return self.validate_multiple_strategies(strategies, validation_level)
    
    def validate_model_performance(self, model_results: Dict[str, Any], 
                                 validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> Dict[str, ValidationResult]:
        """
        Valida performance de modelos.
        
        Args:
            model_results: Resultados de modelos
            validation_level: Nivel de validación
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Validando performance de modelos")
        
        strategies = {}
        
        # Procesar cada modelo
        for model_name, model_data in model_results.items():
            if 'returns' in model_data:
                strategies[model_name] = {
                    'returns': model_data['returns'],
                    'benchmark_returns': model_data.get('benchmark_returns'),
                    'portfolio_values': model_data.get('portfolio_values'),
                    'trades': model_data.get('trades')
                }
        
        return self.validate_multiple_strategies(strategies, validation_level)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de validación.
        
        Returns:
            Diccionario con resumen
        """
        if not self.validation_results:
            return {}
        
        return self.validator.get_validation_summary(self.validation_results)
    
    def get_strategy_validation(self, strategy_name: str) -> Optional[ValidationResult]:
        """
        Obtiene validación de una estrategia específica.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Resultado de validación o None
        """
        return self.validation_results.get(strategy_name)
    
    def get_passed_strategies(self) -> List[str]:
        """
        Obtiene estrategias que pasaron la validación.
        
        Returns:
            Lista de nombres de estrategias
        """
        return [name for name, result in self.validation_results.items() if result.passed]
    
    def get_failed_strategies(self) -> List[str]:
        """
        Obtiene estrategias que fallaron la validación.
        
        Returns:
            Lista de nombres de estrategias
        """
        return [name for name, result in self.validation_results.items() if not result.passed]
    
    def get_strategies_by_score(self, min_score: float = 0.0) -> List[str]:
        """
        Obtiene estrategias por score mínimo.
        
        Args:
            min_score: Score mínimo
            
        Returns:
            Lista de nombres de estrategias
        """
        return [name for name, result in self.validation_results.items() if result.overall_score >= min_score]
    
    def get_strategies_by_level(self, level: ValidationLevel) -> List[str]:
        """
        Obtiene estrategias por nivel de validación.
        
        Args:
            level: Nivel de validación
            
        Returns:
            Lista de nombres de estrategias
        """
        return [name for name, result in self.validation_results.items() if result.validation_level == level]
    
    def create_validation_report(self, strategy_name: str = None) -> str:
        """
        Crea reporte de validación.
        
        Args:
            strategy_name: Nombre de estrategia específica (opcional)
            
        Returns:
            Reporte en formato texto
        """
        if strategy_name:
            result = self.get_strategy_validation(strategy_name)
            if not result:
                return f"No se encontró validación para la estrategia '{strategy_name}'"
            
            return self._create_single_strategy_report(result)
        else:
            return self._create_comprehensive_report()
    
    def _create_single_strategy_report(self, result: ValidationResult) -> str:
        """
        Crea reporte para una estrategia.
        
        Args:
            result: Resultado de validación
            
        Returns:
            Reporte en formato texto
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE VALIDACIÓN DE ESTRATEGIA")
        report.append("=" * 80)
        report.append(f"Estrategia: {result.strategy_name}")
        report.append(f"Nivel: {result.validation_level.value.upper()}")
        report.append(f"Score: {result.overall_score:.2f}")
        report.append(f"Estado: {'PASÓ' if result.passed else 'FALLÓ'}")
        report.append(f"Fecha: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Métricas por categoría
        for category, metrics in result.metrics_results.items():
            if metrics:
                report.append(f"{category.upper()}:")
                report.append("-" * 40)
                for metric_name, value in metrics.items():
                    report.append(f"  {metric_name}: {value:.4f}")
                report.append("")
        
        # Métricas fallidas
        if result.failed_metrics:
            report.append("MÉTRICAS FALLIDAS:")
            report.append("-" * 40)
            for metric in result.failed_metrics:
                report.append(f"  - {metric}")
            report.append("")
        
        # Advertencias
        if result.warnings:
            report.append("ADVERTENCIAS:")
            report.append("-" * 40)
            for warning in result.warnings:
                report.append(f"  - {warning}")
            report.append("")
        
        # Recomendaciones
        if result.recommendations:
            report.append("RECOMENDACIONES:")
            report.append("-" * 40)
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _create_comprehensive_report(self) -> str:
        """
        Crea reporte comprensivo.
        
        Returns:
            Reporte en formato texto
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE COMPRENSIVO DE VALIDACIÓN")
        report.append("=" * 80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen general
        summary = self.get_validation_summary()
        if summary:
            report.append("RESUMEN GENERAL:")
            report.append("-" * 40)
            report.append(f"Total de estrategias: {summary['total_strategies']}")
            report.append(f"Estrategias que pasaron: {summary['passed_strategies']}")
            report.append(f"Estrategias que fallaron: {summary['failed_strategies']}")
            report.append(f"Tasa de aprobación: {summary['pass_rate']:.2%}")
            report.append(f"Score promedio: {summary['avg_score']:.2f}")
            report.append(f"Score mínimo: {summary['min_score']:.2f}")
            report.append(f"Score máximo: {summary['max_score']:.2f}")
            report.append("")
        
        # Estrategias por nivel
        if summary and 'levels' in summary:
            report.append("ESTRATEGIAS POR NIVEL:")
            report.append("-" * 40)
            for level, stats in summary['levels'].items():
                pass_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
                report.append(f"{level.upper()}: {stats['passed']}/{stats['total']} ({pass_rate:.2%})")
            report.append("")
        
        # Métricas más problemáticas
        if summary and 'most_problematic_metrics' in summary:
            report.append("MÉTRICAS MÁS PROBLEMÁTICAS:")
            report.append("-" * 40)
            for metric, count in summary['most_problematic_metrics']:
                report.append(f"  {metric}: {count} fallos")
            report.append("")
        
        # Detalle por estrategia
        report.append("DETALLE POR ESTRATEGIA:")
        report.append("-" * 40)
        for strategy_name, result in self.validation_results.items():
            status = "PASÓ" if result.passed else "FALLÓ"
            report.append(f"{strategy_name}: {status} (Score: {result.overall_score:.2f})")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_validation_results(self, filepath: str, format: str = 'csv') -> None:
        """
        Exporta resultados de validación.
        
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
        
        logger.info(f"Resultados de validación exportados: {filepath}")
    
    def _export_to_csv(self, filepath: Path) -> None:
        """
        Exporta resultados a CSV.
        
        Args:
            filepath: Ruta del archivo
        """
        # Crear DataFrame con resultados
        data = []
        for strategy_name, result in self.validation_results.items():
            row = {
                'strategy_name': strategy_name,
                'validation_level': result.validation_level.value,
                'overall_score': result.overall_score,
                'passed': result.passed,
                'failed_metrics_count': len(result.failed_metrics),
                'warnings_count': len(result.warnings),
                'recommendations_count': len(result.recommendations),
                'timestamp': result.timestamp.isoformat()
            }
            
            # Agregar métricas
            for category, metrics in result.metrics_results.items():
                for metric_name, value in metrics.items():
                    row[f'{category}_{metric_name}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _export_to_excel(self, filepath: Path) -> None:
        """
        Exporta resultados a Excel.
        
        Args:
            filepath: Ruta del archivo
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Resumen
            summary = self.get_validation_summary()
            if summary:
                summary_data = []
                summary_data.append(['Total de estrategias', summary['total_strategies']])
                summary_data.append(['Estrategias que pasaron', summary['passed_strategies']])
                summary_data.append(['Estrategias que fallaron', summary['failed_strategies']])
                summary_data.append(['Tasa de aprobación', f"{summary['pass_rate']:.2%}"])
                summary_data.append(['Score promedio', f"{summary['avg_score']:.2f}"])
                summary_data.append(['Score mínimo', f"{summary['min_score']:.2f}"])
                summary_data.append(['Score máximo', f"{summary['max_score']:.2f}"])
                
                summary_df = pd.DataFrame(summary_data, columns=['Métrica', 'Valor'])
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Resultados por estrategia
            data = []
            for strategy_name, result in self.validation_results.items():
                row = {
                    'Estrategia': strategy_name,
                    'Nivel': result.validation_level.value,
                    'Score': result.overall_score,
                    'Pasó': result.passed,
                    'Métricas Fallidas': len(result.failed_metrics),
                    'Advertencias': len(result.warnings),
                    'Recomendaciones': len(result.recommendations)
                }
                data.append(row)
            
            results_df = pd.DataFrame(data)
            results_df.to_excel(writer, sheet_name='Resultados', index=False)
            
            # Métricas detalladas
            metrics_data = []
            for strategy_name, result in self.validation_results.items():
                for category, metrics in result.metrics_results.items():
                    for metric_name, value in metrics.items():
                        metrics_data.append({
                            'Estrategia': strategy_name,
                            'Categoría': category,
                            'Métrica': metric_name,
                            'Valor': value
                        })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
    
    def _export_to_json(self, filepath: Path) -> None:
        """
        Exporta resultados a JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        import json
        
        # Convertir resultados a formato serializable
        serializable_results = {}
        for strategy_name, result in self.validation_results.items():
            serializable_results[strategy_name] = {
                'strategy_name': result.strategy_name,
                'validation_level': result.validation_level.value,
                'overall_score': result.overall_score,
                'passed': result.passed,
                'metrics_results': result.metrics_results,
                'failed_metrics': result.failed_metrics,
                'warnings': result.warnings,
                'recommendations': result.recommendations,
                'timestamp': result.timestamp.isoformat()
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    def save_validation_state(self, filepath: str) -> None:
        """
        Guarda estado de validación.
        
        Args:
            filepath: Ruta del archivo
        """
        state = {
            'validation_results': self.validation_results,
            'last_validation': self.last_validation,
            'config': self.config
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(state, filepath)
        logger.info(f"Estado de validación guardado: {filepath}")
    
    def load_validation_state(self, filepath: str) -> None:
        """
        Carga estado de validación.
        
        Args:
            filepath: Ruta del archivo
        """
        state = joblib.load(filepath)
        
        self.validation_results = state.get('validation_results', {})
        self.last_validation = state.get('last_validation')
        
        logger.info(f"Estado de validación cargado: {filepath}")
    
    def get_validation_criteria(self, level: ValidationLevel = None) -> Union[Dict[str, Any], Dict[ValidationLevel, Dict[str, Any]]]:
        """
        Obtiene criterios de validación.
        
        Args:
            level: Nivel específico (opcional)
            
        Returns:
            Criterios de validación
        """
        return self.validator.get_validation_criteria(level)
    
    def update_validation_criteria(self, level: ValidationLevel, 
                                 criteria: Dict[str, Any]) -> None:
        """
        Actualiza criterios de validación.
        
        Args:
            level: Nivel de validación
            criteria: Nuevos criterios
        """
        self.validator.update_validation_criteria(level, criteria)
        logger.info(f"Criterios de validación actualizados para nivel {level.value}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del gestor de validación.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config,
            'validation_results_count': len(self.validation_results),
            'last_validation': self.last_validation.isoformat() if self.last_validation else None,
            'validation_summary': self.get_validation_summary()
        }

