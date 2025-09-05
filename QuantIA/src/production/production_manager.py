"""
Gestor principal de producción que coordina despliegue canary y monitoreo.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings
import json

from .canary_deployment import CanaryDeployment, CanaryConfig, DeploymentStage, CanaryStatus
from .production_monitor import ProductionMonitor, MonitorStatus, AlertSeverity, MetricType, MetricThreshold
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProductionManager:
    """
    Gestor principal de producción que coordina despliegue canary y monitoreo.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el gestor de producción.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.production_config = self.config.get('production', {})
        
        # Configuración de canary
        canary_config = CanaryConfig(
            initial_canary_percentage=self.production_config.get('initial_canary_percentage', 5.0),
            max_canary_percentage=self.production_config.get('max_canary_percentage', 50.0),
            canary_increment=self.production_config.get('canary_increment', 5.0),
            canary_duration_minutes=self.production_config.get('canary_duration_minutes', 30),
            rollout_duration_minutes=self.production_config.get('rollout_duration_minutes', 60),
            full_deployment_duration_minutes=self.production_config.get('full_deployment_duration_minutes', 120),
            min_success_rate=self.production_config.get('min_success_rate', 0.95),
            max_error_rate=self.production_config.get('max_error_rate', 0.05),
            max_latency_p95=self.production_config.get('max_latency_p95', 1000.0),
            max_latency_p99=self.production_config.get('max_latency_p99', 2000.0),
            min_throughput=self.production_config.get('min_throughput', 10.0),
            max_drawdown_threshold=self.production_config.get('max_drawdown_threshold', 0.05),
            min_sharpe_ratio=self.production_config.get('min_sharpe_ratio', 1.0),
            min_win_rate=self.production_config.get('min_win_rate', 0.45),
            rollback_success_rate=self.production_config.get('rollback_success_rate', 0.90),
            rollback_error_rate=self.production_config.get('rollback_error_rate', 0.10),
            rollback_latency_p95=self.production_config.get('rollback_latency_p95', 1500.0),
            rollback_latency_p99=self.production_config.get('rollback_latency_p99', 3000.0),
            rollback_drawdown_threshold=self.production_config.get('rollback_drawdown_threshold', 0.08),
            rollback_sharpe_ratio=self.production_config.get('rollback_sharpe_ratio', 0.5),
            metrics_collection_interval=self.production_config.get('metrics_collection_interval', 60),
            health_check_interval=self.production_config.get('health_check_interval', 30),
            alert_cooldown_minutes=self.production_config.get('alert_cooldown_minutes', 15),
            enable_notifications=self.production_config.get('enable_notifications', True),
            notification_channels=self.production_config.get('notification_channels', ["email", "slack"]),
            critical_alert_channels=self.production_config.get('critical_alert_channels', ["email", "slack", "sms"])
        )
        
        # Configuración de monitoreo
        monitoring_config = {
            'monitoring_interval': self.production_config.get('monitoring_interval', 30),
            'alert_cooldown': self.production_config.get('alert_cooldown', 300),
            'retention_days': self.production_config.get('retention_days', 30),
            'enable_notifications': self.production_config.get('enable_notifications', True)
        }
        
        # Inicializar componentes
        self.canary_deployment = CanaryDeployment(canary_config)
        self.production_monitor = ProductionMonitor(monitoring_config)
        
        # Estado del sistema
        self.is_production_active = False
        self.current_strategy = None
        self.deployment_history = []
        
        # Configurar callbacks
        self._setup_callbacks()
        
        logger.info("ProductionManager inicializado")
    
    def _setup_callbacks(self) -> None:
        """Configura callbacks para los componentes."""
        # Callbacks de canary deployment
        self.canary_deployment.on_stage_change = self._on_stage_change
        self.canary_deployment.on_alert = self._on_canary_alert
        self.canary_deployment.on_rollback = self._on_rollback
        
        # Callbacks de production monitor
        self.production_monitor.on_alert = self._on_monitor_alert
        self.production_monitor.on_status_change = self._on_status_change
        self.production_monitor.on_metric_update = self._on_metric_update
    
    def _on_stage_change(self, old_stage: DeploymentStage, new_stage: DeploymentStage) -> None:
        """
        Callback para cambio de etapa.
        
        Args:
            old_stage: Etapa anterior
            new_stage: Nueva etapa
        """
        logger.info(f"Etapa de despliegue cambió de {old_stage.value} a {new_stage.value}")
        
        # Actualizar historial
        self.deployment_history.append({
            'timestamp': datetime.now(),
            'old_stage': old_stage.value,
            'new_stage': new_stage.value,
            'canary_percentage': self.canary_deployment.canary_percentage
        })
    
    def _on_canary_alert(self, alert) -> None:
        """
        Callback para alertas de canary.
        
        Args:
            alert: Alerta de canary
        """
        logger.warning(f"Alerta de canary: {alert.message}")
        
        # Enviar notificación si está habilitado
        if self.canary_deployment.config.enable_notifications:
            self._send_notification(alert)
    
    def _on_rollback(self, reason: str) -> None:
        """
        Callback para rollback.
        
        Args:
            reason: Razón del rollback
        """
        logger.error(f"Rollback ejecutado: {reason}")
        
        # Actualizar historial
        self.deployment_history.append({
            'timestamp': datetime.now(),
            'event': 'rollback',
            'reason': reason,
            'canary_percentage': self.canary_deployment.canary_percentage
        })
    
    def _on_monitor_alert(self, alert) -> None:
        """
        Callback para alertas de monitoreo.
        
        Args:
            alert: Alerta de monitoreo
        """
        logger.warning(f"Alerta de monitoreo: {alert.message}")
        
        # Enviar notificación si está habilitado
        if self.production_monitor.config.get('enable_notifications', True):
            self._send_notification(alert)
    
    def _on_status_change(self, old_status: MonitorStatus, new_status: MonitorStatus) -> None:
        """
        Callback para cambio de estado.
        
        Args:
            old_status: Estado anterior
            new_status: Nuevo estado
        """
        logger.info(f"Estado de monitoreo cambió de {old_status.value} a {new_status.value}")
    
    def _on_metric_update(self, metric) -> None:
        """
        Callback para actualización de métrica.
        
        Args:
            metric: Métrica actualizada
        """
        # Log solo para métricas críticas
        if metric.metric_type == MetricType.RISK and metric.value > 0.1:
            logger.warning(f"Métrica de riesgo alta: {metric.metric_name} = {metric.value}")
    
    def _send_notification(self, alert) -> None:
        """
        Envía notificación.
        
        Args:
            alert: Alerta a notificar
        """
        try:
            # En un sistema real, esto enviaría notificaciones reales
            # Por ahora, solo log
            logger.info(f"Notificación enviada: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {str(e)}")
    
    def start_production(self, strategy_config: Dict[str, Any]) -> bool:
        """
        Inicia el sistema de producción.
        
        Args:
            strategy_config: Configuración de la estrategia
            
        Returns:
            True si se inició correctamente
        """
        if self.is_production_active:
            logger.warning("Producción ya está activa")
            return False
        
        try:
            # Guardar configuración de estrategia
            self.current_strategy = strategy_config
            
            # Iniciar monitoreo
            self.production_monitor.start_monitoring()
            
            # Iniciar despliegue canary
            success = self.canary_deployment.start_deployment(strategy_config)
            
            if success:
                self.is_production_active = True
                logger.info("Sistema de producción iniciado")
                return True
            else:
                # Detener monitoreo si falla el despliegue
                self.production_monitor.stop_monitoring()
                return False
                
        except Exception as e:
            logger.error(f"Error iniciando producción: {str(e)}")
            return False
    
    def stop_production(self) -> None:
        """Detiene el sistema de producción."""
        if not self.is_production_active:
            logger.warning("Producción no está activa")
            return
        
        try:
            # Detener despliegue canary
            self.canary_deployment.stop_deployment()
            
            # Detener monitoreo
            self.production_monitor.stop_monitoring()
            
            self.is_production_active = False
            self.current_strategy = None
            
            logger.info("Sistema de producción detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo producción: {str(e)}")
    
    def get_production_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de producción.
        
        Returns:
            Diccionario con estado de producción
        """
        return {
            'is_production_active': self.is_production_active,
            'current_strategy': self.current_strategy,
            'canary_status': self.canary_deployment.get_deployment_status(),
            'monitoring_status': self.production_monitor.get_monitoring_status(),
            'deployment_history_count': len(self.deployment_history)
        }
    
    def get_canary_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del canary.
        
        Returns:
            Diccionario con estado del canary
        """
        return self.canary_deployment.get_deployment_status()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del monitoreo.
        
        Returns:
            Diccionario con estado del monitoreo
        """
        return self.production_monitor.get_monitoring_status()
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene métricas recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de métricas recientes
        """
        return self.production_monitor.get_recent_metrics(limit)
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene alertas recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de alertas recientes
        """
        # Combinar alertas de canary y monitoreo
        canary_alerts = self.canary_deployment.get_recent_alerts(limit // 2)
        monitor_alerts = self.production_monitor.get_recent_alerts(limit // 2)
        
        # Combinar y ordenar por timestamp
        all_alerts = canary_alerts + monitor_alerts
        all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_alerts[:limit]
    
    def get_system_health(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene salud del sistema.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de registros de salud
        """
        return self.production_monitor.get_system_health(limit)
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene historial de despliegue.
        
        Returns:
            Lista de eventos de despliegue
        """
        return self.deployment_history.copy()
    
    def add_monitoring_threshold(self, threshold: MetricThreshold) -> None:
        """
        Añade un umbral de monitoreo.
        
        Args:
            threshold: Umbral a añadir
        """
        self.production_monitor.add_threshold(threshold)
    
    def remove_monitoring_threshold(self, metric_name: str) -> None:
        """
        Elimina un umbral de monitoreo.
        
        Args:
            metric_name: Nombre de la métrica
        """
        self.production_monitor.remove_threshold(metric_name)
    
    def update_monitoring_threshold(self, metric_name: str, **kwargs) -> None:
        """
        Actualiza un umbral de monitoreo.
        
        Args:
            metric_name: Nombre de la métrica
            **kwargs: Parámetros a actualizar
        """
        self.production_monitor.update_threshold(metric_name, **kwargs)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Reconoce una alerta.
        
        Args:
            alert_id: ID de la alerta
            acknowledged_by: Usuario que reconoce
            
        Returns:
            True si la alerta fue reconocida
        """
        # Intentar reconocer en monitoreo primero
        if self.production_monitor.acknowledge_alert(alert_id, acknowledged_by):
            return True
        
        # Si no se encuentra, podría ser una alerta de canary
        # (Las alertas de canary no tienen reconocimiento por ahora)
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resuelve una alerta.
        
        Args:
            alert_id: ID de la alerta
            
        Returns:
            True si la alerta fue resuelta
        """
        # Intentar resolver en monitoreo primero
        if self.production_monitor.resolve_alert(alert_id):
            return True
        
        # Si no se encuentra, podría ser una alerta de canary
        # (Las alertas de canary no tienen resolución por ahora)
        return False
    
    def create_production_report(self) -> str:
        """
        Crea reporte de producción.
        
        Returns:
            Reporte en formato texto
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE PRODUCCIÓN - SISTEMA DE TRADING")
        report.append("=" * 80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Estado: {'ACTIVO' if self.is_production_active else 'INACTIVO'}")
        report.append("")
        
        # Estado del canary
        canary_status = self.get_canary_status()
        report.append("ESTADO DEL CANARY:")
        report.append("-" * 40)
        report.append(f"Desplegando: {canary_status['is_deploying']}")
        report.append(f"Etapa actual: {canary_status['current_stage']}")
        report.append(f"Porcentaje canary: {canary_status['canary_percentage']:.1f}%")
        report.append(f"Estado de salud: {canary_status['health_status']}")
        report.append(f"Tiempo de inicio: {canary_status['start_time']}")
        report.append(f"Última actualización: {canary_status['last_update']}")
        report.append(f"Métricas recopiladas: {canary_status['metrics_count']}")
        report.append(f"Alertas generadas: {canary_status['alerts_count']}")
        report.append("")
        
        # Estado del monitoreo
        monitoring_status = self.get_monitoring_status()
        report.append("ESTADO DEL MONITOREO:")
        report.append("-" * 40)
        report.append(f"Monitoreando: {monitoring_status['is_monitoring']}")
        report.append(f"Estado actual: {monitoring_status['current_status']}")
        report.append(f"Tiempo de inicio: {monitoring_status['start_time']}")
        report.append(f"Uptime: {monitoring_status['uptime']:.0f} segundos")
        report.append(f"Métricas recopiladas: {monitoring_status['metrics_collected']}")
        report.append(f"Alertas generadas: {monitoring_status['alerts_generated']}")
        report.append(f"Alertas activas: {monitoring_status['active_alerts']}")
        report.append(f"Alertas críticas: {monitoring_status['critical_alerts']}")
        report.append(f"Umbrales configurados: {monitoring_status['thresholds_count']}")
        report.append("")
        
        # Métricas recientes
        recent_metrics = self.get_recent_metrics(10)
        if recent_metrics:
            report.append("MÉTRICAS RECIENTES:")
            report.append("-" * 40)
            for metric in recent_metrics[-5:]:  # Últimas 5
                report.append(f"{metric['metric_name']}: {metric['value']:.4f} {metric['unit']} ({metric['timestamp']})")
            report.append("")
        
        # Alertas recientes
        recent_alerts = self.get_recent_alerts(10)
        if recent_alerts:
            report.append("ALERTAS RECIENTES:")
            report.append("-" * 40)
            for alert in recent_alerts[-5:]:  # Últimas 5
                report.append(f"{alert['severity']}: {alert['message']} ({alert['timestamp']})")
            report.append("")
        
        # Salud del sistema
        system_health = self.get_system_health(1)
        if system_health:
            health = system_health[0]
            report.append("SALUD DEL SISTEMA:")
            report.append("-" * 40)
            report.append(f"Estado general: {health['overall_status']}")
            report.append(f"Alertas activas: {health['active_alerts']}")
            report.append(f"Alertas críticas: {health['critical_alerts']}")
            report.append("")
        
        # Historial de despliegue
        deployment_history = self.get_deployment_history()
        if deployment_history:
            report.append("HISTORIAL DE DESPLIEGUE:")
            report.append("-" * 40)
            for event in deployment_history[-5:]:  # Últimos 5 eventos
                if 'event' in event:
                    report.append(f"{event['timestamp']}: {event['event']} - {event.get('reason', '')}")
                else:
                    report.append(f"{event['timestamp']}: {event['old_stage']} -> {event['new_stage']} ({event['canary_percentage']:.1f}%)")
            report.append("")
        
        # Configuración
        report.append("CONFIGURACIÓN:")
        report.append("-" * 40)
        report.append(f"Porcentaje inicial canary: {self.canary_deployment.config.initial_canary_percentage}%")
        report.append(f"Porcentaje máximo canary: {self.canary_deployment.config.max_canary_percentage}%")
        report.append(f"Incremento canary: {self.canary_deployment.config.canary_increment}%")
        report.append(f"Duración canary: {self.canary_deployment.config.canary_duration_minutes} minutos")
        report.append(f"Duración rollout: {self.canary_deployment.config.rollout_duration_minutes} minutos")
        report.append(f"Duración despliegue completo: {self.canary_deployment.config.full_deployment_duration_minutes} minutos")
        report.append(f"Tasa de éxito mínima: {self.canary_deployment.config.min_success_rate:.2%}")
        report.append(f"Tasa de error máxima: {self.canary_deployment.config.max_error_rate:.2%}")
        report.append(f"Latencia P95 máxima: {self.canary_deployment.config.max_latency_p95} ms")
        report.append(f"Drawdown máximo: {self.canary_deployment.config.max_drawdown_threshold:.2%}")
        report.append(f"Sharpe ratio mínimo: {self.canary_deployment.config.min_sharpe_ratio}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_production_data(self, filepath: str, format: str = 'json') -> None:
        """
        Exporta datos de producción.
        
        Args:
            filepath: Ruta del archivo
            format: Formato de exportación ('json', 'csv', 'excel')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._export_to_json(filepath)
        elif format == 'csv':
            self._export_to_csv(filepath)
        elif format == 'excel':
            self._export_to_excel(filepath)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Datos de producción exportados: {filepath}")
    
    def _export_to_json(self, filepath: Path) -> None:
        """
        Exporta datos a JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        data = {
            'production_status': self.get_production_status(),
            'canary_status': self.get_canary_status(),
            'monitoring_status': self.get_monitoring_status(),
            'recent_metrics': self.get_recent_metrics(),
            'recent_alerts': self.get_recent_alerts(),
            'system_health': self.get_system_health(),
            'deployment_history': self.get_deployment_history(),
            'config': self.config,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_to_csv(self, filepath: Path) -> None:
        """
        Exporta datos a CSV.
        
        Args:
            filepath: Ruta del archivo
        """
        # Exportar métricas
        metrics = self.get_recent_metrics()
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(filepath.with_suffix('.metrics.csv'), index=False)
        
        # Exportar alertas
        alerts = self.get_recent_alerts()
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df.to_csv(filepath.with_suffix('.alerts.csv'), index=False)
        
        # Exportar salud del sistema
        health = self.get_system_health()
        if health:
            health_df = pd.DataFrame(health)
            health_df.to_csv(filepath.with_suffix('.health.csv'), index=False)
    
    def _export_to_excel(self, filepath: Path) -> None:
        """
        Exporta datos a Excel.
        
        Args:
            filepath: Ruta del archivo
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Métricas
            metrics = self.get_recent_metrics()
            if metrics:
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Alertas
            alerts = self.get_recent_alerts()
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                alerts_df.to_excel(writer, sheet_name='Alerts', index=False)
            
            # Salud del sistema
            health = self.get_system_health()
            if health:
                health_df = pd.DataFrame(health)
                health_df.to_excel(writer, sheet_name='Health', index=False)
            
            # Historial de despliegue
            history = self.get_deployment_history()
            if history:
                history_df = pd.DataFrame(history)
                history_df.to_excel(writer, sheet_name='Deployment History', index=False)
    
    def save_production_state(self, filepath: str) -> None:
        """
        Guarda estado de producción.
        
        Args:
            filepath: Ruta del archivo
        """
        state = {
            'canary_deployment': self.canary_deployment,
            'production_monitor': self.production_monitor,
            'is_production_active': self.is_production_active,
            'current_strategy': self.current_strategy,
            'deployment_history': self.deployment_history,
            'config': self.config
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(state, filepath)
        logger.info(f"Estado de producción guardado: {filepath}")
    
    def load_production_state(self, filepath: str) -> None:
        """
        Carga estado de producción.
        
        Args:
            filepath: Ruta del archivo
        """
        state = joblib.load(filepath)
        
        self.canary_deployment = state.get('canary_deployment', self.canary_deployment)
        self.production_monitor = state.get('production_monitor', self.production_monitor)
        self.is_production_active = state.get('is_production_active', False)
        self.current_strategy = state.get('current_strategy')
        self.deployment_history = state.get('deployment_history', [])
        
        # Reconfigurar callbacks
        self._setup_callbacks()
        
        logger.info(f"Estado de producción cargado: {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del gestor de producción.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config,
            'production_status': self.get_production_status(),
            'canary_summary': self.canary_deployment.get_summary(),
            'monitoring_summary': self.production_monitor.get_summary(),
            'deployment_history_count': len(self.deployment_history)
        }

