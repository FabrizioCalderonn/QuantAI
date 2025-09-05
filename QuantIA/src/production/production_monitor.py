"""
Sistema de monitoreo de producción para trading.
Monitorea métricas críticas, alertas y salud del sistema en tiempo real.
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
import psutil
import requests

logger = logging.getLogger(__name__)


class MonitorStatus(Enum):
    """Estados del monitor."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


class AlertSeverity(Enum):
    """Severidad de alertas."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Tipos de métricas."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class MetricThreshold:
    """Umbral de métrica."""
    metric_name: str
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    description: str
    enabled: bool = True


@dataclass
class ProductionMetric:
    """Métrica de producción."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionAlert:
    """Alerta de producción."""
    alert_id: str
    metric_name: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class SystemHealth:
    """Salud del sistema."""
    timestamp: datetime
    overall_status: MonitorStatus
    system_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    custom_metrics: Dict[str, float]
    active_alerts: int
    critical_alerts: int


class ProductionMonitor:
    """
    Sistema de monitoreo de producción para trading.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el monitor de producción.
        
        Args:
            config: Configuración del monitor
        """
        self.config = config or {}
        
        # Configuración
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # segundos
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # segundos
        self.retention_days = self.config.get('retention_days', 30)
        self.enable_notifications = self.config.get('enable_notifications', True)
        
        # Estado del monitor
        self.is_monitoring = False
        self.monitoring_thread = None
        self.start_time = None
        
        # Métricas y alertas
        self.metrics_history: List[ProductionMetric] = []
        self.alerts: List[ProductionAlert] = []
        self.thresholds: Dict[str, MetricThreshold] = {}
        
        # Salud del sistema
        self.system_health: List[SystemHealth] = []
        self.current_status = MonitorStatus.HEALTHY
        
        # Callbacks
        self.on_alert = None
        self.on_status_change = None
        self.on_metric_update = None
        
        # Estadísticas
        self.total_metrics_collected = 0
        self.total_alerts_generated = 0
        self.uptime_start = datetime.now()
        
        # Inicializar umbrales por defecto
        self._initialize_default_thresholds()
        
        logger.info("ProductionMonitor inicializado")
    
    def _initialize_default_thresholds(self) -> None:
        """Inicializa umbrales por defecto."""
        default_thresholds = [
            # Métricas del sistema
            MetricThreshold("cpu_usage", MetricType.SYSTEM, 80.0, 95.0, ">", "Uso de CPU"),
            MetricThreshold("memory_usage", MetricType.SYSTEM, 85.0, 95.0, ">", "Uso de memoria"),
            MetricThreshold("disk_usage", MetricType.SYSTEM, 80.0, 90.0, ">", "Uso de disco"),
            MetricThreshold("network_latency", MetricType.SYSTEM, 100.0, 500.0, ">", "Latencia de red"),
            
            # Métricas de trading
            MetricThreshold("order_execution_time", MetricType.TRADING, 1000.0, 5000.0, ">", "Tiempo de ejecución de órdenes"),
            MetricThreshold("order_success_rate", MetricType.TRADING, 0.95, 0.90, "<", "Tasa de éxito de órdenes"),
            MetricThreshold("trade_frequency", MetricType.TRADING, 10.0, 5.0, "<", "Frecuencia de trades"),
            MetricThreshold("slippage", MetricType.TRADING, 0.001, 0.005, ">", "Slippage"),
            
            # Métricas de riesgo
            MetricThreshold("portfolio_drawdown", MetricType.RISK, 0.05, 0.10, ">", "Drawdown del portfolio"),
            MetricThreshold("var_95", MetricType.RISK, 0.03, 0.05, ">", "VaR 95%"),
            MetricThreshold("leverage", MetricType.RISK, 2.0, 3.0, ">", "Leverage"),
            MetricThreshold("concentration_risk", MetricType.RISK, 0.20, 0.30, ">", "Riesgo de concentración"),
            
            # Métricas de performance
            MetricThreshold("sharpe_ratio", MetricType.PERFORMANCE, 1.0, 0.5, "<", "Sharpe ratio"),
            MetricThreshold("win_rate", MetricType.PERFORMANCE, 0.45, 0.35, "<", "Win rate"),
            MetricThreshold("profit_factor", MetricType.PERFORMANCE, 1.2, 1.0, "<", "Profit factor"),
            MetricThreshold("max_drawdown", MetricType.PERFORMANCE, 0.08, 0.15, ">", "Maximum drawdown")
        ]
        
        for threshold in default_thresholds:
            self.thresholds[threshold.metric_name] = threshold
    
    def start_monitoring(self) -> None:
        """Inicia el monitoreo."""
        if self.is_monitoring:
            logger.warning("Monitoreo ya está activo")
            return
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        # Iniciar hilo de monitoreo
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoreo de producción iniciado")
    
    def stop_monitoring(self) -> None:
        """Detiene el monitoreo."""
        if not self.is_monitoring:
            logger.warning("Monitoreo no está activo")
            return
        
        self.is_monitoring = False
        
        # Esperar a que termine el hilo
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Monitoreo de producción detenido")
    
    def _monitoring_loop(self) -> None:
        """Loop principal de monitoreo."""
        try:
            while self.is_monitoring:
                # Recopilar métricas
                self._collect_system_metrics()
                self._collect_trading_metrics()
                self._collect_risk_metrics()
                self._collect_performance_metrics()
                
                # Verificar umbrales y generar alertas
                self._check_thresholds()
                
                # Actualizar salud del sistema
                self._update_system_health()
                
                # Limpiar datos antiguos
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            logger.error(f"Error en loop de monitoreo: {str(e)}")
    
    def _collect_system_metrics(self) -> None:
        """Recopila métricas del sistema."""
        try:
            # CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            self._add_metric("cpu_usage", MetricType.SYSTEM, cpu_usage, "%")
            
            # Memoria
            memory = psutil.virtual_memory()
            self._add_metric("memory_usage", MetricType.SYSTEM, memory.percent, "%")
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self._add_metric("disk_usage", MetricType.SYSTEM, disk_usage, "%")
            
            # Red (simulado)
            network_latency = np.random.uniform(10, 100)  # ms
            self._add_metric("network_latency", MetricType.SYSTEM, network_latency, "ms")
            
            # Procesos
            process_count = len(psutil.pids())
            self._add_metric("process_count", MetricType.SYSTEM, process_count, "count")
            
        except Exception as e:
            logger.error(f"Error recopilando métricas del sistema: {str(e)}")
    
    def _collect_trading_metrics(self) -> None:
        """Recopila métricas de trading."""
        try:
            # Simular métricas de trading
            order_execution_time = np.random.uniform(50, 200)  # ms
            self._add_metric("order_execution_time", MetricType.TRADING, order_execution_time, "ms")
            
            order_success_rate = np.random.uniform(0.92, 0.98)
            self._add_metric("order_success_rate", MetricType.TRADING, order_success_rate, "ratio")
            
            trade_frequency = np.random.uniform(5, 20)  # trades/min
            self._add_metric("trade_frequency", MetricType.TRADING, trade_frequency, "trades/min")
            
            slippage = np.random.uniform(0.0001, 0.002)
            self._add_metric("slippage", MetricType.TRADING, slippage, "ratio")
            
            # Métricas adicionales
            active_orders = np.random.randint(0, 50)
            self._add_metric("active_orders", MetricType.TRADING, active_orders, "count")
            
            pending_orders = np.random.randint(0, 20)
            self._add_metric("pending_orders", MetricType.TRADING, pending_orders, "count")
            
        except Exception as e:
            logger.error(f"Error recopilando métricas de trading: {str(e)}")
    
    def _collect_risk_metrics(self) -> None:
        """Recopila métricas de riesgo."""
        try:
            # Simular métricas de riesgo
            portfolio_drawdown = np.random.uniform(0.01, 0.08)
            self._add_metric("portfolio_drawdown", MetricType.RISK, portfolio_drawdown, "ratio")
            
            var_95 = np.random.uniform(0.02, 0.04)
            self._add_metric("var_95", MetricType.RISK, var_95, "ratio")
            
            leverage = np.random.uniform(1.0, 2.5)
            self._add_metric("leverage", MetricType.RISK, leverage, "ratio")
            
            concentration_risk = np.random.uniform(0.10, 0.25)
            self._add_metric("concentration_risk", MetricType.RISK, concentration_risk, "ratio")
            
            # Métricas adicionales
            exposure = np.random.uniform(0.80, 1.20)
            self._add_metric("portfolio_exposure", MetricType.RISK, exposure, "ratio")
            
            correlation_risk = np.random.uniform(0.30, 0.70)
            self._add_metric("correlation_risk", MetricType.RISK, correlation_risk, "ratio")
            
        except Exception as e:
            logger.error(f"Error recopilando métricas de riesgo: {str(e)}")
    
    def _collect_performance_metrics(self) -> None:
        """Recopila métricas de performance."""
        try:
            # Simular métricas de performance
            sharpe_ratio = np.random.uniform(0.8, 2.5)
            self._add_metric("sharpe_ratio", MetricType.PERFORMANCE, sharpe_ratio, "ratio")
            
            win_rate = np.random.uniform(0.40, 0.60)
            self._add_metric("win_rate", MetricType.PERFORMANCE, win_rate, "ratio")
            
            profit_factor = np.random.uniform(1.0, 2.0)
            self._add_metric("profit_factor", MetricType.PERFORMANCE, profit_factor, "ratio")
            
            max_drawdown = np.random.uniform(0.05, 0.15)
            self._add_metric("max_drawdown", MetricType.PERFORMANCE, max_drawdown, "ratio")
            
            # Métricas adicionales
            total_return = np.random.uniform(0.05, 0.25)
            self._add_metric("total_return", MetricType.PERFORMANCE, total_return, "ratio")
            
            volatility = np.random.uniform(0.15, 0.30)
            self._add_metric("volatility", MetricType.PERFORMANCE, volatility, "ratio")
            
        except Exception as e:
            logger.error(f"Error recopilando métricas de performance: {str(e)}")
    
    def _add_metric(self, name: str, metric_type: MetricType, value: float, unit: str, 
                   tags: Dict[str, str] = None, metadata: Dict[str, Any] = None) -> None:
        """
        Añade una métrica.
        
        Args:
            name: Nombre de la métrica
            metric_type: Tipo de métrica
            value: Valor de la métrica
            unit: Unidad de la métrica
            tags: Tags adicionales
            metadata: Metadatos adicionales
        """
        metric = ProductionMetric(
            metric_name=name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metric)
        self.total_metrics_collected += 1
        
        # Llamar callback si está definido
        if self.on_metric_update:
            self.on_metric_update(metric)
    
    def _check_thresholds(self) -> None:
        """Verifica umbrales y genera alertas."""
        for metric_name, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue
            
            # Obtener métrica más reciente
            recent_metric = self._get_recent_metric(metric_name)
            if not recent_metric:
                continue
            
            # Verificar umbral
            if self._check_threshold(recent_metric.value, threshold):
                # Generar alerta
                severity = self._determine_severity(recent_metric.value, threshold)
                self._generate_alert(metric_name, threshold, recent_metric, severity)
    
    def _get_recent_metric(self, metric_name: str) -> Optional[ProductionMetric]:
        """
        Obtiene la métrica más reciente.
        
        Args:
            metric_name: Nombre de la métrica
            
        Returns:
            Métrica más reciente o None
        """
        recent_metrics = [m for m in self.metrics_history if m.metric_name == metric_name]
        if not recent_metrics:
            return None
        
        return max(recent_metrics, key=lambda m: m.timestamp)
    
    def _check_threshold(self, value: float, threshold: MetricThreshold) -> bool:
        """
        Verifica si un valor excede un umbral.
        
        Args:
            value: Valor a verificar
            threshold: Umbral
            
        Returns:
            True si excede el umbral
        """
        if threshold.operator == '>':
            return value > threshold.warning_threshold
        elif threshold.operator == '<':
            return value < threshold.warning_threshold
        elif threshold.operator == '>=':
            return value >= threshold.warning_threshold
        elif threshold.operator == '<=':
            return value <= threshold.warning_threshold
        elif threshold.operator == '==':
            return value == threshold.warning_threshold
        elif threshold.operator == '!=':
            return value != threshold.warning_threshold
        
        return False
    
    def _determine_severity(self, value: float, threshold: MetricThreshold) -> AlertSeverity:
        """
        Determina la severidad de una alerta.
        
        Args:
            value: Valor de la métrica
            threshold: Umbral
            
        Returns:
            Severidad de la alerta
        """
        if threshold.operator in ['>', '>=']:
            if value >= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            else:
                return AlertSeverity.HIGH
        elif threshold.operator in ['<', '<=']:
            if value <= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            else:
                return AlertSeverity.HIGH
        else:
            return AlertSeverity.MEDIUM
    
    def _generate_alert(self, metric_name: str, threshold: MetricThreshold, 
                       metric: ProductionMetric, severity: AlertSeverity) -> None:
        """
        Genera una alerta.
        
        Args:
            metric_name: Nombre de la métrica
            threshold: Umbral
            metric: Métrica
            severity: Severidad
        """
        # Verificar si ya hay una alerta activa para esta métrica
        active_alert = self._get_active_alert(metric_name)
        if active_alert:
            # Verificar cooldown
            if (datetime.now() - active_alert.timestamp).total_seconds() < self.alert_cooldown:
                return
        
        # Crear nueva alerta
        alert = ProductionAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts):06d}",
            metric_name=metric_name,
            metric_type=threshold.metric_type,
            severity=severity,
            message=f"{threshold.description}: {metric.value:.4f} {metric.unit} (umbral: {threshold.warning_threshold:.4f})",
            value=metric.value,
            threshold=threshold.warning_threshold,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.total_alerts_generated += 1
        
        # Llamar callback si está definido
        if self.on_alert:
            self.on_alert(alert)
        
        logger.warning(f"Alerta generada: {alert.message}")
    
    def _get_active_alert(self, metric_name: str) -> Optional[ProductionAlert]:
        """
        Obtiene la alerta activa más reciente para una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            
        Returns:
            Alerta activa o None
        """
        active_alerts = [a for a in self.alerts if a.metric_name == metric_name and not a.resolved]
        if not active_alerts:
            return None
        
        return max(active_alerts, key=lambda a: a.timestamp)
    
    def _update_system_health(self) -> None:
        """Actualiza la salud del sistema."""
        try:
            # Recopilar métricas actuales
            system_metrics = {}
            trading_metrics = {}
            risk_metrics = {}
            performance_metrics = {}
            custom_metrics = {}
            
            for metric in self.metrics_history[-100:]:  # Últimas 100 métricas
                if metric.metric_type == MetricType.SYSTEM:
                    system_metrics[metric.metric_name] = metric.value
                elif metric.metric_type == MetricType.TRADING:
                    trading_metrics[metric.metric_name] = metric.value
                elif metric.metric_type == MetricType.RISK:
                    risk_metrics[metric.metric_name] = metric.value
                elif metric.metric_type == MetricType.PERFORMANCE:
                    performance_metrics[metric.metric_name] = metric.value
                elif metric.metric_type == MetricType.CUSTOM:
                    custom_metrics[metric.metric_name] = metric.value
            
            # Contar alertas activas
            active_alerts = len([a for a in self.alerts if not a.resolved])
            critical_alerts = len([a for a in self.alerts if not a.resolved and a.severity == AlertSeverity.CRITICAL])
            
            # Determinar estado general
            if critical_alerts > 0:
                overall_status = MonitorStatus.CRITICAL
            elif active_alerts > 5:
                overall_status = MonitorStatus.WARNING
            elif active_alerts > 0:
                overall_status = MonitorStatus.WARNING
            else:
                overall_status = MonitorStatus.HEALTHY
            
            # Crear registro de salud
            health = SystemHealth(
                timestamp=datetime.now(),
                overall_status=overall_status,
                system_metrics=system_metrics,
                trading_metrics=trading_metrics,
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics,
                custom_metrics=custom_metrics,
                active_alerts=active_alerts,
                critical_alerts=critical_alerts
            )
            
            self.system_health.append(health)
            
            # Verificar cambio de estado
            if self.current_status != overall_status:
                old_status = self.current_status
                self.current_status = overall_status
                
                # Llamar callback si está definido
                if self.on_status_change:
                    self.on_status_change(old_status, overall_status)
                
                logger.info(f"Estado del sistema cambió de {old_status.value} a {overall_status.value}")
            
        except Exception as e:
            logger.error(f"Error actualizando salud del sistema: {str(e)}")
    
    def _cleanup_old_data(self) -> None:
        """Limpia datos antiguos."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            # Limpiar métricas antiguas
            self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            # Limpiar alertas antiguas
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
            
            # Limpiar registros de salud antiguos
            self.system_health = [h for h in self.system_health if h.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error limpiando datos antiguos: {str(e)}")
    
    def add_threshold(self, threshold: MetricThreshold) -> None:
        """
        Añade un umbral.
        
        Args:
            threshold: Umbral a añadir
        """
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Umbral añadido: {threshold.metric_name}")
    
    def remove_threshold(self, metric_name: str) -> None:
        """
        Elimina un umbral.
        
        Args:
            metric_name: Nombre de la métrica
        """
        if metric_name in self.thresholds:
            del self.thresholds[metric_name]
            logger.info(f"Umbral eliminado: {metric_name}")
    
    def update_threshold(self, metric_name: str, **kwargs) -> None:
        """
        Actualiza un umbral.
        
        Args:
            metric_name: Nombre de la métrica
            **kwargs: Parámetros a actualizar
        """
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            for key, value in kwargs.items():
                if hasattr(threshold, key):
                    setattr(threshold, key, value)
            logger.info(f"Umbral actualizado: {metric_name}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Reconoce una alerta.
        
        Args:
            alert_id: ID de la alerta
            acknowledged_by: Usuario que reconoce
            
        Returns:
            True si la alerta fue reconocida
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alerta reconocida: {alert_id} por {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resuelve una alerta.
        
        Args:
            alert_id: ID de la alerta
            
        Returns:
            True si la alerta fue resuelta
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alerta resuelta: {alert_id}")
                return True
        
        return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del monitoreo.
        
        Returns:
            Diccionario con estado del monitoreo
        """
        return {
            'is_monitoring': self.is_monitoring,
            'current_status': self.current_status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': (datetime.now() - self.uptime_start).total_seconds(),
            'metrics_collected': self.total_metrics_collected,
            'alerts_generated': self.total_alerts_generated,
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'critical_alerts': len([a for a in self.alerts if not a.resolved and a.severity == AlertSeverity.CRITICAL]),
            'thresholds_count': len(self.thresholds)
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
    
    def get_system_health(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene salud del sistema.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de registros de salud
        """
        recent_health = self.system_health[-limit:]
        return [h.__dict__ for h in recent_health]
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del monitoreo.
        
        Returns:
            Diccionario con resumen
        """
        if not self.system_health:
            return {}
        
        latest_health = self.system_health[-1]
        
        return {
            'monitoring_status': self.get_monitoring_status(),
            'latest_health': latest_health.__dict__,
            'current_status': self.current_status.value,
            'uptime': (datetime.now() - self.uptime_start).total_seconds(),
            'metrics_summary': {
                'total_metrics': self.total_metrics_collected,
                'total_alerts': self.total_alerts_generated,
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'critical_alerts': len([a for a in self.alerts if not a.resolved and a.severity == AlertSeverity.CRITICAL])
            }
        }
    
    def export_monitoring_data(self, filepath: str) -> None:
        """
        Exporta datos de monitoreo.
        
        Args:
            filepath: Ruta del archivo
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'monitoring_status': self.get_monitoring_status(),
            'metrics_history': self.get_recent_metrics(),
            'alerts': self.get_recent_alerts(),
            'system_health': self.get_system_health(),
            'thresholds': {name: t.__dict__ for name, t in self.thresholds.items()},
            'config': self.config,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Datos de monitoreo exportados: {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del monitor de producción.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config,
            'monitoring_status': self.get_monitoring_status(),
            'current_status': self.current_status.value,
            'metrics_count': len(self.metrics_history),
            'alerts_count': len(self.alerts),
            'thresholds_count': len(self.thresholds),
            'total_metrics_collected': self.total_metrics_collected,
            'total_alerts_generated': self.total_alerts_generated
        }

