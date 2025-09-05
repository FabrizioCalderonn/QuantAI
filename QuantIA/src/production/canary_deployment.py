"""
Sistema de despliegue canary para trading en producción.
Implementa despliegue gradual con monitoreo y rollback automático.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import asyncio
import threading
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Etapas de despliegue."""
    PREPARATION = "preparation"
    CANARY = "canary"
    ROLLOUT = "rollout"
    FULL_DEPLOYMENT = "full_deployment"
    ROLLBACK = "rollback"
    COMPLETED = "completed"


class CanaryStatus(Enum):
    """Estados del canary."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"


class AlertLevel(Enum):
    """Niveles de alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CanaryMetrics:
    """Métricas del canary."""
    timestamp: datetime
    stage: DeploymentStage
    canary_percentage: float
    success_rate: float
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    pnl: float
    drawdown: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CanaryConfig:
    """Configuración del canary."""
    # Configuración de despliegue
    initial_canary_percentage: float = 5.0
    max_canary_percentage: float = 50.0
    canary_increment: float = 5.0
    canary_duration_minutes: int = 30
    rollout_duration_minutes: int = 60
    full_deployment_duration_minutes: int = 120
    
    # Criterios de salud
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05
    max_latency_p95: float = 1000.0  # ms
    max_latency_p99: float = 2000.0  # ms
    min_throughput: float = 10.0  # trades/min
    max_drawdown_threshold: float = 0.05  # 5%
    min_sharpe_ratio: float = 1.0
    min_win_rate: float = 0.45
    
    # Criterios de rollback
    rollback_success_rate: float = 0.90
    rollback_error_rate: float = 0.10
    rollback_latency_p95: float = 1500.0  # ms
    rollback_latency_p99: float = 3000.0  # ms
    rollback_drawdown_threshold: float = 0.08  # 8%
    rollback_sharpe_ratio: float = 0.5
    
    # Configuración de monitoreo
    metrics_collection_interval: int = 60  # segundos
    health_check_interval: int = 30  # segundos
    alert_cooldown_minutes: int = 15
    
    # Configuración de notificaciones
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    critical_alert_channels: List[str] = field(default_factory=lambda: ["email", "slack", "sms"])


@dataclass
class DeploymentAlert:
    """Alerta de despliegue."""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    stage: DeploymentStage
    canary_percentage: float
    metrics: Dict[str, float]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class CanaryDeployment:
    """
    Sistema de despliegue canary para trading en producción.
    """
    
    def __init__(self, config: CanaryConfig = None):
        """
        Inicializa el sistema de despliegue canary.
        
        Args:
            config: Configuración del canary
        """
        self.config = config or CanaryConfig()
        
        # Estado del despliegue
        self.current_stage = DeploymentStage.PREPARATION
        self.canary_percentage = 0.0
        self.start_time = None
        self.last_update = None
        
        # Métricas y monitoreo
        self.metrics_history: List[CanaryMetrics] = []
        self.alerts: List[DeploymentAlert] = []
        self.health_status = CanaryStatus.HEALTHY
        
        # Control de despliegue
        self.is_deploying = False
        self.deployment_thread = None
        self.monitoring_thread = None
        
        # Callbacks
        self.on_stage_change = None
        self.on_alert = None
        self.on_rollback = None
        
        # Estadísticas
        self.total_deployments = 0
        self.successful_deployments = 0
        self.failed_deployments = 0
        self.rollbacks = 0
        
        logger.info("CanaryDeployment inicializado")
    
    def start_deployment(self, strategy_config: Dict[str, Any]) -> bool:
        """
        Inicia el despliegue canary.
        
        Args:
            strategy_config: Configuración de la estrategia
            
        Returns:
            True si el despliegue se inició correctamente
        """
        if self.is_deploying:
            logger.warning("Despliegue ya está en progreso")
            return False
        
        try:
            self.is_deploying = True
            self.current_stage = DeploymentStage.PREPARATION
            self.canary_percentage = 0.0
            self.start_time = datetime.now()
            self.last_update = datetime.now()
            
            # Iniciar hilos de despliegue y monitoreo
            self.deployment_thread = threading.Thread(target=self._deployment_loop, daemon=True)
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            
            self.deployment_thread.start()
            self.monitoring_thread.start()
            
            self.total_deployments += 1
            
            logger.info("Despliegue canary iniciado")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando despliegue: {str(e)}")
            self.is_deploying = False
            return False
    
    def stop_deployment(self) -> None:
        """Detiene el despliegue canary."""
        if not self.is_deploying:
            logger.warning("No hay despliegue en progreso")
            return
        
        self.is_deploying = False
        
        # Esperar a que terminen los hilos
        if self.deployment_thread:
            self.deployment_thread.join(timeout=5.0)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Despliegue canary detenido")
    
    def _deployment_loop(self) -> None:
        """Loop principal de despliegue."""
        try:
            while self.is_deploying:
                if self.current_stage == DeploymentStage.PREPARATION:
                    self._handle_preparation_stage()
                elif self.current_stage == DeploymentStage.CANARY:
                    self._handle_canary_stage()
                elif self.current_stage == DeploymentStage.ROLLOUT:
                    self._handle_rollout_stage()
                elif self.current_stage == DeploymentStage.FULL_DEPLOYMENT:
                    self._handle_full_deployment_stage()
                elif self.current_stage == DeploymentStage.ROLLBACK:
                    self._handle_rollback_stage()
                elif self.current_stage == DeploymentStage.COMPLETED:
                    break
                
                time.sleep(10)  # Verificar cada 10 segundos
                
        except Exception as e:
            logger.error(f"Error en loop de despliegue: {str(e)}")
            self._trigger_rollback("Error en loop de despliegue")
    
    def _monitoring_loop(self) -> None:
        """Loop de monitoreo."""
        try:
            while self.is_deploying:
                # Recopilar métricas
                metrics = self._collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # Verificar salud
                    self._check_health(metrics)
                    
                    # Verificar criterios de rollback
                    if self._should_rollback(metrics):
                        self._trigger_rollback("Criterios de rollback cumplidos")
                
                time.sleep(self.config.health_check_interval)
                
        except Exception as e:
            logger.error(f"Error en loop de monitoreo: {str(e)}")
            self._trigger_rollback("Error en loop de monitoreo")
    
    def _handle_preparation_stage(self) -> None:
        """Maneja la etapa de preparación."""
        logger.info("Etapa de preparación")
        
        # Simular preparación
        time.sleep(5)
        
        # Avanzar a canary
        self._advance_stage(DeploymentStage.CANARY)
    
    def _handle_canary_stage(self) -> None:
        """Maneja la etapa de canary."""
        logger.info(f"Etapa de canary - {self.canary_percentage}%")
        
        # Incrementar porcentaje de canary
        if self.canary_percentage < self.config.max_canary_percentage:
            self.canary_percentage = min(
                self.canary_percentage + self.config.canary_increment,
                self.config.max_canary_percentage
            )
        
        # Verificar si es tiempo de avanzar
        if self._should_advance_from_canary():
            self._advance_stage(DeploymentStage.ROLLOUT)
    
    def _handle_rollout_stage(self) -> None:
        """Maneja la etapa de rollout."""
        logger.info(f"Etapa de rollout - {self.canary_percentage}%")
        
        # Incrementar porcentaje gradualmente
        if self.canary_percentage < 100.0:
            self.canary_percentage = min(
                self.canary_percentage + self.config.canary_increment,
                100.0
            )
        
        # Verificar si es tiempo de avanzar
        if self._should_advance_from_rollout():
            self._advance_stage(DeploymentStage.FULL_DEPLOYMENT)
    
    def _handle_full_deployment_stage(self) -> None:
        """Maneja la etapa de despliegue completo."""
        logger.info("Etapa de despliegue completo")
        
        # Mantener en 100% por un tiempo
        time.sleep(self.config.full_deployment_duration_minutes * 60)
        
        # Completar despliegue
        self._advance_stage(DeploymentStage.COMPLETED)
    
    def _handle_rollback_stage(self) -> None:
        """Maneja la etapa de rollback."""
        logger.info("Etapa de rollback")
        
        # Reducir porcentaje gradualmente
        if self.canary_percentage > 0.0:
            self.canary_percentage = max(
                self.canary_percentage - self.config.canary_increment,
                0.0
            )
        else:
            # Rollback completado
            self._advance_stage(DeploymentStage.COMPLETED)
            self.rollbacks += 1
    
    def _advance_stage(self, new_stage: DeploymentStage) -> None:
        """
        Avanza a una nueva etapa.
        
        Args:
            new_stage: Nueva etapa
        """
        old_stage = self.current_stage
        self.current_stage = new_stage
        self.last_update = datetime.now()
        
        logger.info(f"Avanzando de {old_stage.value} a {new_stage.value}")
        
        # Llamar callback si está definido
        if self.on_stage_change:
            self.on_stage_change(old_stage, new_stage)
    
    def _should_advance_from_canary(self) -> bool:
        """
        Determina si se debe avanzar desde la etapa de canary.
        
        Returns:
            True si se debe avanzar
        """
        # Verificar tiempo mínimo
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            if elapsed.total_seconds() < self.config.canary_duration_minutes * 60:
                return False
        
        # Verificar métricas recientes
        recent_metrics = self._get_recent_metrics(5)  # Últimos 5 minutos
        if not recent_metrics:
            return False
        
        # Verificar que todas las métricas estén saludables
        for metrics in recent_metrics:
            if not self._is_metrics_healthy(metrics):
                return False
        
        return True
    
    def _should_advance_from_rollout(self) -> bool:
        """
        Determina si se debe avanzar desde la etapa de rollout.
        
        Returns:
            True si se debe avanzar
        """
        # Verificar tiempo mínimo
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            if elapsed.total_seconds() < self.config.rollout_duration_minutes * 60:
                return False
        
        # Verificar métricas recientes
        recent_metrics = self._get_recent_metrics(10)  # Últimos 10 minutos
        if not recent_metrics:
            return False
        
        # Verificar que todas las métricas estén saludables
        for metrics in recent_metrics:
            if not self._is_metrics_healthy(metrics):
                return False
        
        return True
    
    def _collect_metrics(self) -> Optional[CanaryMetrics]:
        """
        Recopila métricas del sistema.
        
        Returns:
            Métricas recopiladas o None si hay error
        """
        try:
            # En un sistema real, esto vendría de métricas reales
            # Por ahora, simulamos métricas
            
            # Simular métricas basadas en el porcentaje de canary
            base_success_rate = 0.98
            base_error_rate = 0.02
            base_latency = 100.0
            
            # Ajustar métricas basadas en porcentaje de canary
            canary_factor = self.canary_percentage / 100.0
            
            success_rate = base_success_rate - (canary_factor * 0.02)  # Peor con más tráfico
            error_rate = base_error_rate + (canary_factor * 0.01)      # Más errores con más tráfico
            latency = base_latency + (canary_factor * 50.0)            # Más latencia con más tráfico
            
            # Simular métricas de trading
            pnl = np.random.normal(1000 * canary_factor, 500)
            drawdown = np.random.uniform(0.01, 0.03)
            sharpe_ratio = np.random.uniform(1.5, 2.5)
            max_drawdown = np.random.uniform(0.02, 0.05)
            win_rate = np.random.uniform(0.45, 0.55)
            
            total_trades = int(100 * canary_factor)
            winning_trades = int(total_trades * win_rate)
            losing_trades = total_trades - winning_trades
            
            metrics = CanaryMetrics(
                timestamp=datetime.now(),
                stage=self.current_stage,
                canary_percentage=self.canary_percentage,
                success_rate=success_rate,
                error_rate=error_rate,
                latency_p50=latency,
                latency_p95=latency * 1.5,
                latency_p99=latency * 2.0,
                throughput=total_trades / 60.0,  # trades por minuto
                pnl=pnl,
                drawdown=drawdown,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                risk_metrics={
                    'var_95': np.random.uniform(0.01, 0.03),
                    'cvar_95': np.random.uniform(0.02, 0.04),
                    'volatility': np.random.uniform(0.15, 0.25)
                },
                custom_metrics={
                    'model_accuracy': np.random.uniform(0.85, 0.95),
                    'feature_importance_stability': np.random.uniform(0.90, 0.98)
                }
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error recopilando métricas: {str(e)}")
            return None
    
    def _check_health(self, metrics: CanaryMetrics) -> None:
        """
        Verifica la salud del sistema basada en métricas.
        
        Args:
            metrics: Métricas a verificar
        """
        old_status = self.health_status
        
        # Verificar criterios de salud
        if (metrics.success_rate < self.config.min_success_rate or
            metrics.error_rate > self.config.max_error_rate or
            metrics.latency_p95 > self.config.max_latency_p95 or
            metrics.latency_p99 > self.config.max_latency_p99 or
            metrics.throughput < self.config.min_throughput or
            metrics.max_drawdown > self.config.max_drawdown_threshold or
            metrics.sharpe_ratio < self.config.min_sharpe_ratio or
            metrics.win_rate < self.config.min_win_rate):
            
            self.health_status = CanaryStatus.DEGRADED
            
            # Verificar si es crítico
            if (metrics.success_rate < self.config.rollback_success_rate or
                metrics.error_rate > self.config.rollback_error_rate or
                metrics.latency_p95 > self.config.rollback_latency_p95 or
                metrics.latency_p99 > self.config.rollback_latency_p99 or
                metrics.max_drawdown > self.config.rollback_drawdown_threshold or
                metrics.sharpe_ratio < self.config.rollback_sharpe_ratio):
                
                self.health_status = CanaryStatus.CRITICAL
        else:
            self.health_status = CanaryStatus.HEALTHY
        
        # Generar alerta si cambió el estado
        if old_status != self.health_status:
            self._generate_alert(old_status, self.health_status, metrics)
    
    def _is_metrics_healthy(self, metrics: CanaryMetrics) -> bool:
        """
        Verifica si las métricas están saludables.
        
        Args:
            metrics: Métricas a verificar
            
        Returns:
            True si las métricas están saludables
        """
        return (metrics.success_rate >= self.config.min_success_rate and
                metrics.error_rate <= self.config.max_error_rate and
                metrics.latency_p95 <= self.config.max_latency_p95 and
                metrics.latency_p99 <= self.config.max_latency_p99 and
                metrics.throughput >= self.config.min_throughput and
                metrics.max_drawdown <= self.config.max_drawdown_threshold and
                metrics.sharpe_ratio >= self.config.min_sharpe_ratio and
                metrics.win_rate >= self.config.min_win_rate)
    
    def _should_rollback(self, metrics: CanaryMetrics) -> bool:
        """
        Determina si se debe hacer rollback.
        
        Args:
            metrics: Métricas a verificar
            
        Returns:
            True si se debe hacer rollback
        """
        return (metrics.success_rate < self.config.rollback_success_rate or
                metrics.error_rate > self.config.rollback_error_rate or
                metrics.latency_p95 > self.config.rollback_latency_p95 or
                metrics.latency_p99 > self.config.rollback_latency_p99 or
                metrics.max_drawdown > self.config.rollback_drawdown_threshold or
                metrics.sharpe_ratio < self.config.rollback_sharpe_ratio)
    
    def _trigger_rollback(self, reason: str) -> None:
        """
        Dispara un rollback.
        
        Args:
            reason: Razón del rollback
        """
        logger.warning(f"Disparando rollback: {reason}")
        
        # Avanzar a etapa de rollback
        self._advance_stage(DeploymentStage.ROLLBACK)
        
        # Generar alerta crítica
        self._generate_alert(
            self.health_status,
            CanaryStatus.CRITICAL,
            None,
            AlertLevel.CRITICAL,
            f"Rollback disparado: {reason}"
        )
        
        # Llamar callback si está definido
        if self.on_rollback:
            self.on_rollback(reason)
    
    def _generate_alert(self, old_status: CanaryStatus, new_status: CanaryStatus, 
                       metrics: Optional[CanaryMetrics], level: AlertLevel = None,
                       custom_message: str = None) -> None:
        """
        Genera una alerta.
        
        Args:
            old_status: Estado anterior
            new_status: Estado nuevo
            metrics: Métricas actuales
            level: Nivel de alerta
            custom_message: Mensaje personalizado
        """
        if level is None:
            if new_status == CanaryStatus.CRITICAL:
                level = AlertLevel.CRITICAL
            elif new_status == CanaryStatus.DEGRADED:
                level = AlertLevel.WARNING
            else:
                level = AlertLevel.INFO
        
        if custom_message:
            message = custom_message
        else:
            message = f"Estado de salud cambió de {old_status.value} a {new_status.value}"
        
        alert = DeploymentAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts):06d}",
            level=level,
            message=message,
            timestamp=datetime.now(),
            stage=self.current_stage,
            canary_percentage=self.canary_percentage,
            metrics=metrics.__dict__ if metrics else {}
        )
        
        self.alerts.append(alert)
        
        # Llamar callback si está definido
        if self.on_alert:
            self.on_alert(alert)
        
        logger.info(f"Alerta generada: {message}")
    
    def _get_recent_metrics(self, minutes: int) -> List[CanaryMetrics]:
        """
        Obtiene métricas recientes.
        
        Args:
            minutes: Minutos hacia atrás
            
        Returns:
            Lista de métricas recientes
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del despliegue.
        
        Returns:
            Diccionario con estado del despliegue
        """
        return {
            'is_deploying': self.is_deploying,
            'current_stage': self.current_stage.value,
            'canary_percentage': self.canary_percentage,
            'health_status': self.health_status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'metrics_count': len(self.metrics_history),
            'alerts_count': len(self.alerts),
            'total_deployments': self.total_deployments,
            'successful_deployments': self.successful_deployments,
            'failed_deployments': self.failed_deployments,
            'rollbacks': self.rollbacks
        }
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene métricas recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de métricas recientes
        """
        recent_metrics = self.metrics_history[-limit:]
        return [m.__dict__ for m in recent_metrics]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene alertas recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de alertas recientes
        """
        recent_alerts = self.alerts[-limit:]
        return [a.__dict__ for a in recent_alerts]
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del despliegue.
        
        Returns:
            Diccionario con resumen
        """
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            'deployment_status': self.get_deployment_status(),
            'latest_metrics': latest_metrics.__dict__,
            'health_status': self.health_status.value,
            'canary_percentage': self.canary_percentage,
            'current_stage': self.current_stage.value,
            'deployment_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'metrics_summary': {
                'avg_success_rate': np.mean([m.success_rate for m in self.metrics_history]),
                'avg_error_rate': np.mean([m.error_rate for m in self.metrics_history]),
                'avg_latency_p95': np.mean([m.latency_p95 for m in self.metrics_history]),
                'avg_throughput': np.mean([m.throughput for m in self.metrics_history]),
                'avg_pnl': np.mean([m.pnl for m in self.metrics_history]),
                'avg_sharpe_ratio': np.mean([m.sharpe_ratio for m in self.metrics_history]),
                'max_drawdown': max([m.max_drawdown for m in self.metrics_history])
            }
        }
    
    def export_deployment_data(self, filepath: str) -> None:
        """
        Exporta datos del despliegue.
        
        Args:
            filepath: Ruta del archivo
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'deployment_status': self.get_deployment_status(),
            'metrics_history': self.get_recent_metrics(),
            'alerts': self.get_recent_alerts(),
            'config': self.config.__dict__,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Datos de despliegue exportados: {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del sistema de despliegue canary.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config.__dict__,
            'deployment_status': self.get_deployment_status(),
            'health_status': self.health_status.value,
            'metrics_count': len(self.metrics_history),
            'alerts_count': len(self.alerts),
            'total_deployments': self.total_deployments,
            'successful_deployments': self.successful_deployments,
            'failed_deployments': self.failed_deployments,
            'rollbacks': self.rollbacks
        }

