"""
Validador de estrategias con criterios de aprobación.
Implementa validación robusta con múltiples niveles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

from .metrics import ValidationMetrics, MetricType, ValidationLevel, MetricThreshold, ValidationResult

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Estados de validación."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class ValidationCriteria:
    """Criterios de validación."""
    level: ValidationLevel
    thresholds: List[MetricThreshold]
    min_score: float = 0.7
    required_metrics: List[str] = None
    description: str = ""


class StrategyValidator:
    """
    Validador de estrategias con criterios de aprobación.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el validador.
        
        Args:
            config: Configuración del validador
        """
        self.config = config or {}
        self.metrics_calculator = ValidationMetrics(config)
        self.validation_criteria = self._create_validation_criteria()
        
        logger.info("StrategyValidator inicializado")
    
    def _create_validation_criteria(self) -> Dict[ValidationLevel, ValidationCriteria]:
        """
        Crea criterios de validación por nivel.
        
        Returns:
            Diccionario con criterios por nivel
        """
        criteria = {}
        
        # Criterios básicos
        basic_thresholds = [
            MetricThreshold('sharpe_ratio', 0.5, '>=', 1.0, True, "Sharpe ratio mínimo"),
            MetricThreshold('max_drawdown', -0.15, '>=', 1.0, True, "Drawdown máximo"),
            MetricThreshold('win_rate', 0.4, '>=', 0.8, True, "Win rate mínimo"),
            MetricThreshold('total_trades', 30, '>=', 0.5, True, "Mínimo de trades"),
            MetricThreshold('volatility', 0.5, '<=', 0.8, False, "Volatilidad máxima")
        ]
        
        criteria[ValidationLevel.BASIC] = ValidationCriteria(
            level=ValidationLevel.BASIC,
            thresholds=basic_thresholds,
            min_score=0.6,
            required_metrics=['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'],
            description="Validación básica de estrategia"
        )
        
        # Criterios intermedios
        intermediate_thresholds = [
            MetricThreshold('sharpe_ratio', 0.8, '>=', 1.0, True, "Sharpe ratio mínimo"),
            MetricThreshold('max_drawdown', -0.12, '>=', 1.0, True, "Drawdown máximo"),
            MetricThreshold('win_rate', 0.45, '>=', 0.8, True, "Win rate mínimo"),
            MetricThreshold('profit_factor', 1.2, '>=', 0.8, True, "Profit factor mínimo"),
            MetricThreshold('total_trades', 50, '>=', 0.5, True, "Mínimo de trades"),
            MetricThreshold('volatility', 0.4, '<=', 0.8, False, "Volatilidad máxima"),
            MetricThreshold('return_stability', 0.3, '>=', 0.6, False, "Estabilidad de retornos"),
            MetricThreshold('positive_months', 0.5, '>=', 0.6, False, "Meses positivos")
        ]
        
        criteria[ValidationLevel.INTERMEDIATE] = ValidationCriteria(
            level=ValidationLevel.INTERMEDIATE,
            thresholds=intermediate_thresholds,
            min_score=0.7,
            required_metrics=['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades'],
            description="Validación intermedia de estrategia"
        )
        
        # Criterios avanzados
        advanced_thresholds = [
            MetricThreshold('sharpe_ratio', 1.0, '>=', 1.0, True, "Sharpe ratio mínimo"),
            MetricThreshold('max_drawdown', -0.10, '>=', 1.0, True, "Drawdown máximo"),
            MetricThreshold('win_rate', 0.50, '>=', 0.8, True, "Win rate mínimo"),
            MetricThreshold('profit_factor', 1.3, '>=', 0.8, True, "Profit factor mínimo"),
            MetricThreshold('total_trades', 100, '>=', 0.5, True, "Mínimo de trades"),
            MetricThreshold('volatility', 0.35, '<=', 0.8, False, "Volatilidad máxima"),
            MetricThreshold('return_stability', 0.4, '>=', 0.6, False, "Estabilidad de retornos"),
            MetricThreshold('positive_months', 0.55, '>=', 0.6, False, "Meses positivos"),
            MetricThreshold('calmar_ratio', 1.5, '>=', 0.8, False, "Calmar ratio mínimo"),
            MetricThreshold('information_ratio', 0.5, '>=', 0.6, False, "Information ratio mínimo")
        ]
        
        criteria[ValidationLevel.ADVANCED] = ValidationCriteria(
            level=ValidationLevel.ADVANCED,
            thresholds=advanced_thresholds,
            min_score=0.75,
            required_metrics=['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades', 'calmar_ratio'],
            description="Validación avanzada de estrategia"
        )
        
        # Criterios institucionales
        institutional_thresholds = [
            MetricThreshold('sharpe_ratio', 1.2, '>=', 1.0, True, "Sharpe ratio mínimo"),
            MetricThreshold('max_drawdown', -0.08, '>=', 1.0, True, "Drawdown máximo"),
            MetricThreshold('win_rate', 0.55, '>=', 0.8, True, "Win rate mínimo"),
            MetricThreshold('profit_factor', 1.4, '>=', 0.8, True, "Profit factor mínimo"),
            MetricThreshold('total_trades', 200, '>=', 0.5, True, "Mínimo de trades"),
            MetricThreshold('volatility', 0.30, '<=', 0.8, False, "Volatilidad máxima"),
            MetricThreshold('return_stability', 0.5, '>=', 0.6, False, "Estabilidad de retornos"),
            MetricThreshold('positive_months', 0.60, '>=', 0.6, False, "Meses positivos"),
            MetricThreshold('calmar_ratio', 2.0, '>=', 0.8, False, "Calmar ratio mínimo"),
            MetricThreshold('information_ratio', 0.7, '>=', 0.6, False, "Information ratio mínimo"),
            MetricThreshold('normality_test', 0.6, '>=', 0.4, False, "Test de normalidad"),
            MetricThreshold('correlation_stability', 0.7, '>=', 0.4, False, "Estabilidad de correlación")
        ]
        
        criteria[ValidationLevel.INSTITUTIONAL] = ValidationCriteria(
            level=ValidationLevel.INSTITUTIONAL,
            thresholds=institutional_thresholds,
            min_score=0.8,
            required_metrics=['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades', 'calmar_ratio', 'information_ratio'],
            description="Validación institucional de estrategia"
        )
        
        return criteria
    
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
        logger.info(f"Validando estrategia '{strategy_name}' con nivel {validation_level.value}")
        
        # Obtener criterios de validación
        criteria = self.validation_criteria[validation_level]
        
        # Calcular métricas
        metrics_results = {}
        
        # Métricas de performance
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            returns, benchmark_returns
        )
        metrics_results['performance'] = performance_metrics
        
        # Métricas de riesgo
        risk_metrics = self.metrics_calculator.calculate_risk_metrics(
            returns, portfolio_values
        )
        metrics_results['risk'] = risk_metrics
        
        # Métricas de trading
        if trades:
            trading_metrics = self.metrics_calculator.calculate_trading_metrics(trades)
            metrics_results['trading'] = trading_metrics
        else:
            metrics_results['trading'] = {}
        
        # Métricas de estabilidad
        stability_metrics = self.metrics_calculator.calculate_stability_metrics(returns)
        metrics_results['stability'] = stability_metrics
        
        # Métricas de robustez
        robustness_metrics = self.metrics_calculator.calculate_robustness_metrics(
            returns, benchmark_returns
        )
        metrics_results['robustness'] = robustness_metrics
        
        # Validar métricas
        validation_results = self._validate_metrics(metrics_results, criteria)
        
        # Calcular score general
        overall_score = self._calculate_overall_score(validation_results, criteria)
        
        # Determinar si pasó la validación
        passed = self._determine_validation_status(validation_results, criteria, overall_score)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(validation_results, criteria)
        
        # Crear resultado
        result = ValidationResult(
            strategy_name=strategy_name,
            validation_level=validation_level,
            overall_score=overall_score,
            passed=passed,
            metrics_results=metrics_results,
            failed_metrics=validation_results['failed_metrics'],
            warnings=validation_results['warnings'],
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        logger.info(f"Validación completada: {strategy_name} - Score: {overall_score:.2f} - Passed: {passed}")
        
        return result
    
    def _validate_metrics(self, metrics_results: Dict[str, Dict[str, float]], 
                         criteria: ValidationCriteria) -> Dict[str, Any]:
        """
        Valida métricas contra criterios.
        
        Args:
            metrics_results: Resultados de métricas
            criteria: Criterios de validación
            
        Returns:
            Resultados de validación
        """
        validation_results = {
            'passed_metrics': [],
            'failed_metrics': [],
            'warnings': [],
            'metric_scores': {}
        }
        
        # Aplanar métricas
        all_metrics = {}
        for category, metrics in metrics_results.items():
            for metric_name, value in metrics.items():
                all_metrics[metric_name] = value
        
        # Validar cada umbral
        for threshold in criteria.thresholds:
            metric_name = threshold.metric_name
            
            if metric_name not in all_metrics:
                if threshold.required:
                    validation_results['failed_metrics'].append(metric_name)
                    validation_results['warnings'].append(f"Métrica requerida '{metric_name}' no encontrada")
                continue
            
            metric_value = all_metrics[metric_name]
            
            # Evaluar umbral
            passed = self._evaluate_threshold(metric_value, threshold)
            
            if passed:
                validation_results['passed_metrics'].append(metric_name)
                validation_results['metric_scores'][metric_name] = 1.0
            else:
                if threshold.required:
                    validation_results['failed_metrics'].append(metric_name)
                else:
                    validation_results['warnings'].append(f"Métrica '{metric_name}' no cumple umbral: {metric_value:.3f} {threshold.comparison_operator} {threshold.threshold_value}")
                
                validation_results['metric_scores'][metric_name] = 0.0
        
        # Verificar métricas requeridas
        for required_metric in criteria.required_metrics:
            if required_metric not in all_metrics:
                validation_results['failed_metrics'].append(required_metric)
                validation_results['warnings'].append(f"Métrica requerida '{required_metric}' no encontrada")
        
        return validation_results
    
    def _evaluate_threshold(self, value: float, threshold: MetricThreshold) -> bool:
        """
        Evalúa un valor contra un umbral.
        
        Args:
            value: Valor a evaluar
            threshold: Umbral
            
        Returns:
            True si cumple el umbral
        """
        if threshold.comparison_operator == '>':
            return value > threshold.threshold_value
        elif threshold.comparison_operator == '<':
            return value < threshold.threshold_value
        elif threshold.comparison_operator == '>=':
            return value >= threshold.threshold_value
        elif threshold.comparison_operator == '<=':
            return value <= threshold.threshold_value
        elif threshold.comparison_operator == '==':
            return abs(value - threshold.threshold_value) < 1e-8
        elif threshold.comparison_operator == '!=':
            return abs(value - threshold.threshold_value) >= 1e-8
        else:
            logger.warning(f"Operador de comparación no reconocido: {threshold.comparison_operator}")
            return False
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any], 
                                criteria: ValidationCriteria) -> float:
        """
        Calcula score general de validación.
        
        Args:
            validation_results: Resultados de validación
            criteria: Criterios de validación
            
        Returns:
            Score general
        """
        if not validation_results['metric_scores']:
            return 0.0
        
        # Calcular score ponderado
        total_weight = 0.0
        weighted_score = 0.0
        
        for threshold in criteria.thresholds:
            metric_name = threshold.metric_name
            
            if metric_name in validation_results['metric_scores']:
                score = validation_results['metric_scores'][metric_name]
                weight = threshold.weight
                
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_score / total_weight
    
    def _determine_validation_status(self, validation_results: Dict[str, Any], 
                                   criteria: ValidationCriteria, 
                                   overall_score: float) -> bool:
        """
        Determina si la estrategia pasó la validación.
        
        Args:
            validation_results: Resultados de validación
            criteria: Criterios de validación
            overall_score: Score general
            
        Returns:
            True si pasó la validación
        """
        # Verificar score mínimo
        if overall_score < criteria.min_score:
            return False
        
        # Verificar métricas requeridas
        for required_metric in criteria.required_metrics:
            if required_metric in validation_results['failed_metrics']:
                return False
        
        return True
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], 
                                criteria: ValidationCriteria) -> List[str]:
        """
        Genera recomendaciones basadas en la validación.
        
        Args:
            validation_results: Resultados de validación
            criteria: Criterios de validación
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendaciones para métricas fallidas
        for failed_metric in validation_results['failed_metrics']:
            if failed_metric == 'sharpe_ratio':
                recommendations.append("Mejorar el Sharpe ratio optimizando el balance riesgo-retorno")
            elif failed_metric == 'max_drawdown':
                recommendations.append("Implementar stops más agresivos para reducir el drawdown máximo")
            elif failed_metric == 'win_rate':
                recommendations.append("Mejorar la selección de señales para aumentar el win rate")
            elif failed_metric == 'profit_factor':
                recommendations.append("Optimizar el ratio de ganancias/pérdidas")
            elif failed_metric == 'total_trades':
                recommendations.append("Aumentar la frecuencia de trading o extender el período de prueba")
            elif failed_metric == 'volatility':
                recommendations.append("Reducir la volatilidad del portafolio")
            elif failed_metric == 'return_stability':
                recommendations.append("Mejorar la estabilidad de retornos")
            elif failed_metric == 'positive_months':
                recommendations.append("Aumentar la consistencia de retornos positivos")
            elif failed_metric == 'calmar_ratio':
                recommendations.append("Mejorar el Calmar ratio optimizando retorno vs drawdown")
            elif failed_metric == 'information_ratio':
                recommendations.append("Mejorar el information ratio vs benchmark")
            elif failed_metric == 'normality_test':
                recommendations.append("Mejorar la normalidad de retornos")
            elif failed_metric == 'correlation_stability':
                recommendations.append("Mejorar la estabilidad de correlación con benchmark")
        
        # Recomendaciones generales
        if not recommendations:
            recommendations.append("La estrategia cumple todos los criterios de validación")
        
        return recommendations
    
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
        results = {}
        
        for strategy_name, strategy_data in strategies.items():
            try:
                result = self.validate_strategy(
                    strategy_name=strategy_name,
                    returns=strategy_data.get('returns'),
                    benchmark_returns=strategy_data.get('benchmark_returns'),
                    portfolio_values=strategy_data.get('portfolio_values'),
                    trades=strategy_data.get('trades'),
                    validation_level=validation_level
                )
                results[strategy_name] = result
            except Exception as e:
                logger.error(f"Error validando estrategia {strategy_name}: {str(e)}")
                # Crear resultado de error
                results[strategy_name] = ValidationResult(
                    strategy_name=strategy_name,
                    validation_level=validation_level,
                    overall_score=0.0,
                    passed=False,
                    metrics_results={},
                    failed_metrics=[],
                    warnings=[f"Error en validación: {str(e)}"],
                    recommendations=["Revisar datos de entrada"],
                    timestamp=datetime.now()
                )
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Obtiene resumen de validación de múltiples estrategias.
        
        Args:
            results: Resultados de validación
            
        Returns:
            Resumen de validación
        """
        if not results:
            return {}
        
        total_strategies = len(results)
        passed_strategies = sum(1 for r in results.values() if r.passed)
        failed_strategies = total_strategies - passed_strategies
        
        # Estadísticas de scores
        scores = [r.overall_score for r in results.values()]
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Estrategias por nivel
        levels = {}
        for result in results.values():
            level = result.validation_level.value
            if level not in levels:
                levels[level] = {'total': 0, 'passed': 0}
            levels[level]['total'] += 1
            if result.passed:
                levels[level]['passed'] += 1
        
        # Métricas más problemáticas
        all_failed_metrics = []
        for result in results.values():
            all_failed_metrics.extend(result.failed_metrics)
        
        metric_failures = {}
        for metric in all_failed_metrics:
            metric_failures[metric] = metric_failures.get(metric, 0) + 1
        
        # Ordenar por frecuencia
        most_problematic_metrics = sorted(metric_failures.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_strategies': total_strategies,
            'passed_strategies': passed_strategies,
            'failed_strategies': failed_strategies,
            'pass_rate': passed_strategies / total_strategies,
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'levels': levels,
            'most_problematic_metrics': most_problematic_metrics[:10],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_validation_criteria(self, level: ValidationLevel = None) -> Union[ValidationCriteria, Dict[ValidationLevel, ValidationCriteria]]:
        """
        Obtiene criterios de validación.
        
        Args:
            level: Nivel específico (opcional)
            
        Returns:
            Criterios de validación
        """
        if level:
            return self.validation_criteria.get(level)
        return self.validation_criteria.copy()
    
    def update_validation_criteria(self, level: ValidationLevel, 
                                 criteria: ValidationCriteria) -> None:
        """
        Actualiza criterios de validación.
        
        Args:
            level: Nivel de validación
            criteria: Nuevos criterios
        """
        self.validation_criteria[level] = criteria
        logger.info(f"Criterios de validación actualizados para nivel {level.value}")
    
    def add_custom_threshold(self, level: ValidationLevel, 
                           threshold: MetricThreshold) -> None:
        """
        Agrega umbral personalizado.
        
        Args:
            level: Nivel de validación
            threshold: Umbral personalizado
        """
        if level in self.validation_criteria:
            self.validation_criteria[level].thresholds.append(threshold)
            logger.info(f"Umbral personalizado agregado: {threshold.metric_name}")
        else:
            logger.warning(f"Nivel de validación {level.value} no encontrado")
    
    def remove_threshold(self, level: ValidationLevel, 
                        metric_name: str) -> None:
        """
        Remueve umbral.
        
        Args:
            level: Nivel de validación
            metric_name: Nombre de la métrica
        """
        if level in self.validation_criteria:
            criteria = self.validation_criteria[level]
            criteria.thresholds = [t for t in criteria.thresholds if t.metric_name != metric_name]
            logger.info(f"Umbral removido: {metric_name}")
        else:
            logger.warning(f"Nivel de validación {level.value} no encontrado")

