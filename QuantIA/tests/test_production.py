"""
Tests para sistema de producción.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import os
import time
import json

from src.production.canary_deployment import (
    CanaryDeployment, CanaryConfig, DeploymentStage, CanaryStatus, AlertLevel
)
from src.production.production_monitor import (
    ProductionMonitor, MonitorStatus, AlertSeverity, MetricType, MetricThreshold, ProductionMetric, ProductionAlert
)
from src.production.production_manager import ProductionManager


class TestCanaryConfig:
    """Tests para CanaryConfig."""
    
    def test_canary_config_creation(self):
        """Test de creación de CanaryConfig."""
        config = CanaryConfig(
            initial_canary_percentage=10.0,
            max_canary_percentage=60.0,
            canary_increment=10.0
        )
        
        assert config.initial_canary_percentage == 10.0
        assert config.max_canary_percentage == 60.0
        assert config.canary_increment == 10.0
        assert config.min_success_rate == 0.95
        assert config.max_error_rate == 0.05


class TestCanaryDeployment:
    """Tests para CanaryDeployment."""
    
    @pytest.fixture
    def canary_deployment(self):
        """CanaryDeployment de prueba."""
        config = CanaryConfig(
            initial_canary_percentage=5.0,
            max_canary_percentage=50.0,
            canary_increment=5.0,
            canary_duration_minutes=1,  # 1 minuto para tests
            rollout_duration_minutes=1,
            full_deployment_duration_minutes=1
        )
        return CanaryDeployment(config)
    
    def test_canary_deployment_initialization(self):
        """Test de inicialización de CanaryDeployment."""
        config = CanaryConfig()
        deployment = CanaryDeployment(config)
        
        assert deployment.config == config
        assert deployment.current_stage == DeploymentStage.PREPARATION
        assert deployment.canary_percentage == 0.0
        assert deployment.is_deploying == False
        assert len(deployment.metrics_history) == 0
        assert len(deployment.alerts) == 0
        assert deployment.health_status == CanaryStatus.HEALTHY
    
    def test_start_deployment(self, canary_deployment):
        """Test de inicio de despliegue."""
        strategy_config = {"strategy": "test"}
        
        success = canary_deployment.start_deployment(strategy_config)
        
        assert success == True
        assert canary_deployment.is_deploying == True
        assert canary_deployment.current_stage == DeploymentStage.PREPARATION
        assert canary_deployment.start_time is not None
        assert canary_deployment.total_deployments == 1
    
    def test_stop_deployment(self, canary_deployment):
        """Test de parada de despliegue."""
        canary_deployment.start_deployment({"strategy": "test"})
        
        canary_deployment.stop_deployment()
        
        assert canary_deployment.is_deploying == False
    
    def test_collect_metrics(self, canary_deployment):
        """Test de recopilación de métricas."""
        metrics = canary_deployment._collect_metrics()
        
        assert metrics is not None
        assert metrics.timestamp is not None
        assert metrics.stage == canary_deployment.current_stage
        assert metrics.canary_percentage == canary_deployment.canary_percentage
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 0.0 <= metrics.error_rate <= 1.0
        assert metrics.latency_p50 > 0
        assert metrics.latency_p95 > 0
        assert metrics.latency_p99 > 0
        assert metrics.throughput >= 0
        assert metrics.total_trades >= 0
        assert metrics.winning_trades >= 0
        assert metrics.losing_trades >= 0
    
    def test_check_health(self, canary_deployment):
        """Test de verificación de salud."""
        # Crear métricas saludables
        metrics = canary_deployment._collect_metrics()
        canary_deployment._check_health(metrics)
        
        # Verificar que el estado de salud se actualizó
        assert canary_deployment.health_status in [CanaryStatus.HEALTHY, CanaryStatus.DEGRADED]
    
    def test_should_rollback(self, canary_deployment):
        """Test de criterios de rollback."""
        # Crear métricas que no deberían disparar rollback
        metrics = canary_deployment._collect_metrics()
        should_rollback = canary_deployment._should_rollback(metrics)
        
        # En condiciones normales, no debería hacer rollback
        assert isinstance(should_rollback, bool)
    
    def test_trigger_rollback(self, canary_deployment):
        """Test de disparo de rollback."""
        canary_deployment._trigger_rollback("Test rollback")
        
        assert canary_deployment.current_stage == DeploymentStage.ROLLBACK
        assert len(canary_deployment.alerts) > 0
    
    def test_get_deployment_status(self, canary_deployment):
        """Test de obtención de estado de despliegue."""
        status = canary_deployment.get_deployment_status()
        
        assert 'is_deploying' in status
        assert 'current_stage' in status
        assert 'canary_percentage' in status
        assert 'health_status' in status
        assert 'start_time' in status
        assert 'last_update' in status
        assert 'metrics_count' in status
        assert 'alerts_count' in status
        assert 'total_deployments' in status
        assert 'successful_deployments' in status
        assert 'failed_deployments' in status
        assert 'rollbacks' in status
    
    def test_get_recent_metrics(self, canary_deployment):
        """Test de obtención de métricas recientes."""
        # Añadir algunas métricas
        for _ in range(5):
            metrics = canary_deployment._collect_metrics()
            canary_deployment.metrics_history.append(metrics)
        
        recent_metrics = canary_deployment.get_recent_metrics(3)
        
        assert len(recent_metrics) == 3
        assert all('timestamp' in m for m in recent_metrics)
        assert all('stage' in m for m in recent_metrics)
        assert all('canary_percentage' in m for m in recent_metrics)
    
    def test_get_recent_alerts(self, canary_deployment):
        """Test de obtención de alertas recientes."""
        recent_alerts = canary_deployment.get_recent_alerts(10)
        
        assert isinstance(recent_alerts, list)
    
    def test_get_deployment_summary(self, canary_deployment):
        """Test de obtención de resumen de despliegue."""
        summary = canary_deployment.get_deployment_summary()
        
        assert isinstance(summary, dict)
    
    def test_export_deployment_data(self, canary_deployment, temp_dir):
        """Test de exportación de datos de despliegue."""
        filepath = temp_dir + "/test_deployment.json"
        canary_deployment.export_deployment_data(filepath)
        
        assert os.path.exists(filepath)
        
        # Verificar contenido
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert 'deployment_status' in data
        assert 'metrics_history' in data
        assert 'alerts' in data
        assert 'config' in data
        assert 'export_timestamp' in data
    
    def test_get_summary(self, canary_deployment):
        """Test de obtención de resumen."""
        summary = canary_deployment.get_summary()
        
        assert 'config' in summary
        assert 'deployment_status' in summary
        assert 'health_status' in summary
        assert 'metrics_count' in summary
        assert 'alerts_count' in summary
        assert 'total_deployments' in summary
        assert 'successful_deployments' in summary
        assert 'failed_deployments' in summary
        assert 'rollbacks' in summary


class TestMetricThreshold:
    """Tests para MetricThreshold."""
    
    def test_metric_threshold_creation(self):
        """Test de creación de MetricThreshold."""
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.SYSTEM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        assert threshold.metric_name == "test_metric"
        assert threshold.metric_type == MetricType.SYSTEM
        assert threshold.warning_threshold == 80.0
        assert threshold.critical_threshold == 95.0
        assert threshold.operator == ">"
        assert threshold.description == "Test metric"
        assert threshold.enabled == True


class TestProductionMetric:
    """Tests para ProductionMetric."""
    
    def test_production_metric_creation(self):
        """Test de creación de ProductionMetric."""
        metric = ProductionMetric(
            metric_name="test_metric",
            metric_type=MetricType.SYSTEM,
            value=75.0,
            timestamp=datetime.now(),
            unit="%"
        )
        
        assert metric.metric_name == "test_metric"
        assert metric.metric_type == MetricType.SYSTEM
        assert metric.value == 75.0
        assert metric.unit == "%"
        assert isinstance(metric.timestamp, datetime)


class TestProductionAlert:
    """Tests para ProductionAlert."""
    
    def test_production_alert_creation(self):
        """Test de creación de ProductionAlert."""
        alert = ProductionAlert(
            alert_id="test_alert",
            metric_name="test_metric",
            metric_type=MetricType.SYSTEM,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            value=85.0,
            threshold=80.0,
            timestamp=datetime.now()
        )
        
        assert alert.alert_id == "test_alert"
        assert alert.metric_name == "test_metric"
        assert alert.metric_type == MetricType.SYSTEM
        assert alert.severity == AlertSeverity.HIGH
        assert alert.message == "Test alert"
        assert alert.value == 85.0
        assert alert.threshold == 80.0
        assert alert.resolved == False
        assert alert.acknowledged == False


class TestProductionMonitor:
    """Tests para ProductionMonitor."""
    
    @pytest.fixture
    def production_monitor(self):
        """ProductionMonitor de prueba."""
        config = {
            'monitoring_interval': 1,  # 1 segundo para tests
            'alert_cooldown': 5,  # 5 segundos para tests
            'retention_days': 1
        }
        return ProductionMonitor(config)
    
    def test_production_monitor_initialization(self):
        """Test de inicialización de ProductionMonitor."""
        config = {'monitoring_interval': 30}
        monitor = ProductionMonitor(config)
        
        assert monitor.config == config
        assert monitor.is_monitoring == False
        assert monitor.start_time is None
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alerts) == 0
        assert len(monitor.thresholds) > 0  # Debería tener umbrales por defecto
        assert monitor.current_status == MonitorStatus.HEALTHY
        assert monitor.total_metrics_collected == 0
        assert monitor.total_alerts_generated == 0
    
    def test_start_monitoring(self, production_monitor):
        """Test de inicio de monitoreo."""
        production_monitor.start_monitoring()
        
        assert production_monitor.is_monitoring == True
        assert production_monitor.start_time is not None
    
    def test_stop_monitoring(self, production_monitor):
        """Test de parada de monitoreo."""
        production_monitor.start_monitoring()
        production_monitor.stop_monitoring()
        
        assert production_monitor.is_monitoring == False
    
    def test_collect_system_metrics(self, production_monitor):
        """Test de recopilación de métricas del sistema."""
        production_monitor._collect_system_metrics()
        
        # Verificar que se recopilaron métricas del sistema
        system_metrics = [m for m in production_monitor.metrics_history if m.metric_type == MetricType.SYSTEM]
        assert len(system_metrics) > 0
        
        # Verificar métricas específicas
        metric_names = [m.metric_name for m in system_metrics]
        assert "cpu_usage" in metric_names
        assert "memory_usage" in metric_names
        assert "disk_usage" in metric_names
        assert "network_latency" in metric_names
    
    def test_collect_trading_metrics(self, production_monitor):
        """Test de recopilación de métricas de trading."""
        production_monitor._collect_trading_metrics()
        
        # Verificar que se recopilaron métricas de trading
        trading_metrics = [m for m in production_monitor.metrics_history if m.metric_type == MetricType.TRADING]
        assert len(trading_metrics) > 0
        
        # Verificar métricas específicas
        metric_names = [m.metric_name for m in trading_metrics]
        assert "order_execution_time" in metric_names
        assert "order_success_rate" in metric_names
        assert "trade_frequency" in metric_names
        assert "slippage" in metric_names
    
    def test_collect_risk_metrics(self, production_monitor):
        """Test de recopilación de métricas de riesgo."""
        production_monitor._collect_risk_metrics()
        
        # Verificar que se recopilaron métricas de riesgo
        risk_metrics = [m for m in production_monitor.metrics_history if m.metric_type == MetricType.RISK]
        assert len(risk_metrics) > 0
        
        # Verificar métricas específicas
        metric_names = [m.metric_name for m in risk_metrics]
        assert "portfolio_drawdown" in metric_names
        assert "var_95" in metric_names
        assert "leverage" in metric_names
        assert "concentration_risk" in metric_names
    
    def test_collect_performance_metrics(self, production_monitor):
        """Test de recopilación de métricas de performance."""
        production_monitor._collect_performance_metrics()
        
        # Verificar que se recopilaron métricas de performance
        performance_metrics = [m for m in production_monitor.metrics_history if m.metric_type == MetricType.PERFORMANCE]
        assert len(performance_metrics) > 0
        
        # Verificar métricas específicas
        metric_names = [m.metric_name for m in performance_metrics]
        assert "sharpe_ratio" in metric_names
        assert "win_rate" in metric_names
        assert "profit_factor" in metric_names
        assert "max_drawdown" in metric_names
    
    def test_add_metric(self, production_monitor):
        """Test de añadir métrica."""
        production_monitor._add_metric("test_metric", MetricType.CUSTOM, 75.0, "%")
        
        assert len(production_monitor.metrics_history) == 1
        assert production_monitor.total_metrics_collected == 1
        
        metric = production_monitor.metrics_history[0]
        assert metric.metric_name == "test_metric"
        assert metric.metric_type == MetricType.CUSTOM
        assert metric.value == 75.0
        assert metric.unit == "%"
    
    def test_check_thresholds(self, production_monitor):
        """Test de verificación de umbrales."""
        # Añadir métrica que exceda umbral
        production_monitor._add_metric("cpu_usage", MetricType.SYSTEM, 90.0, "%")
        
        # Verificar umbrales
        production_monitor._check_thresholds()
        
        # Debería generar una alerta
        assert len(production_monitor.alerts) > 0
        assert production_monitor.total_alerts_generated > 0
    
    def test_get_recent_metric(self, production_monitor):
        """Test de obtención de métrica reciente."""
        # Añadir métricas
        production_monitor._add_metric("test_metric", MetricType.CUSTOM, 50.0, "%")
        production_monitor._add_metric("test_metric", MetricType.CUSTOM, 75.0, "%")
        
        recent_metric = production_monitor._get_recent_metric("test_metric")
        
        assert recent_metric is not None
        assert recent_metric.value == 75.0
    
    def test_check_threshold(self, production_monitor):
        """Test de verificación de umbral."""
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        # Verificar umbral
        assert production_monitor._check_threshold(90.0, threshold) == True
        assert production_monitor._check_threshold(70.0, threshold) == False
    
    def test_determine_severity(self, production_monitor):
        """Test de determinación de severidad."""
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        # Verificar severidad
        severity = production_monitor._determine_severity(90.0, threshold)
        assert severity == AlertSeverity.HIGH
        
        severity = production_monitor._determine_severity(98.0, threshold)
        assert severity == AlertSeverity.CRITICAL
    
    def test_generate_alert(self, production_monitor):
        """Test de generación de alerta."""
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        metric = ProductionMetric(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            value=90.0,
            timestamp=datetime.now(),
            unit="%"
        )
        
        production_monitor._generate_alert("test_metric", threshold, metric, AlertSeverity.HIGH)
        
        assert len(production_monitor.alerts) == 1
        assert production_monitor.total_alerts_generated == 1
        
        alert = production_monitor.alerts[0]
        assert alert.metric_name == "test_metric"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.value == 90.0
        assert alert.threshold == 80.0
    
    def test_update_system_health(self, production_monitor):
        """Test de actualización de salud del sistema."""
        # Añadir algunas métricas
        production_monitor._add_metric("cpu_usage", MetricType.SYSTEM, 75.0, "%")
        production_monitor._add_metric("memory_usage", MetricType.SYSTEM, 80.0, "%")
        
        production_monitor._update_system_health()
        
        assert len(production_monitor.system_health) == 1
        
        health = production_monitor.system_health[0]
        assert health.overall_status in [MonitorStatus.HEALTHY, MonitorStatus.WARNING, MonitorStatus.CRITICAL]
        assert "cpu_usage" in health.system_metrics
        assert "memory_usage" in health.system_metrics
    
    def test_add_threshold(self, production_monitor):
        """Test de añadir umbral."""
        threshold = MetricThreshold(
            metric_name="custom_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=50.0,
            critical_threshold=75.0,
            operator=">",
            description="Custom metric"
        )
        
        production_monitor.add_threshold(threshold)
        
        assert "custom_metric" in production_monitor.thresholds
        assert production_monitor.thresholds["custom_metric"] == threshold
    
    def test_remove_threshold(self, production_monitor):
        """Test de eliminar umbral."""
        # Añadir umbral
        threshold = MetricThreshold(
            metric_name="custom_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=50.0,
            critical_threshold=75.0,
            operator=">",
            description="Custom metric"
        )
        production_monitor.add_threshold(threshold)
        
        # Eliminar umbral
        production_monitor.remove_threshold("custom_metric")
        
        assert "custom_metric" not in production_monitor.thresholds
    
    def test_update_threshold(self, production_monitor):
        """Test de actualizar umbral."""
        # Actualizar umbral existente
        production_monitor.update_threshold("cpu_usage", warning_threshold=70.0)
        
        assert production_monitor.thresholds["cpu_usage"].warning_threshold == 70.0
    
    def test_acknowledge_alert(self, production_monitor):
        """Test de reconocer alerta."""
        # Generar alerta
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        metric = ProductionMetric(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            value=90.0,
            timestamp=datetime.now(),
            unit="%"
        )
        
        production_monitor._generate_alert("test_metric", threshold, metric, AlertSeverity.HIGH)
        
        alert_id = production_monitor.alerts[0].alert_id
        
        # Reconocer alerta
        success = production_monitor.acknowledge_alert(alert_id, "test_user")
        
        assert success == True
        assert production_monitor.alerts[0].acknowledged == True
        assert production_monitor.alerts[0].acknowledged_by == "test_user"
        assert production_monitor.alerts[0].acknowledged_at is not None
    
    def test_resolve_alert(self, production_monitor):
        """Test de resolver alerta."""
        # Generar alerta
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        metric = ProductionMetric(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            value=90.0,
            timestamp=datetime.now(),
            unit="%"
        )
        
        production_monitor._generate_alert("test_metric", threshold, metric, AlertSeverity.HIGH)
        
        alert_id = production_monitor.alerts[0].alert_id
        
        # Resolver alerta
        success = production_monitor.resolve_alert(alert_id)
        
        assert success == True
        assert production_monitor.alerts[0].resolved == True
        assert production_monitor.alerts[0].resolved_at is not None
    
    def test_get_monitoring_status(self, production_monitor):
        """Test de obtención de estado de monitoreo."""
        status = production_monitor.get_monitoring_status()
        
        assert 'is_monitoring' in status
        assert 'current_status' in status
        assert 'start_time' in status
        assert 'uptime' in status
        assert 'metrics_collected' in status
        assert 'alerts_generated' in status
        assert 'active_alerts' in status
        assert 'critical_alerts' in status
        assert 'thresholds_count' in status
    
    def test_get_recent_metrics(self, production_monitor):
        """Test de obtención de métricas recientes."""
        # Añadir métricas
        for i in range(5):
            production_monitor._add_metric(f"metric_{i}", MetricType.CUSTOM, i * 10, "%")
        
        recent_metrics = production_monitor.get_recent_metrics(3)
        
        assert len(recent_metrics) == 3
        assert all('metric_name' in m for m in recent_metrics)
        assert all('value' in m for m in recent_metrics)
        assert all('timestamp' in m for m in recent_metrics)
    
    def test_get_recent_alerts(self, production_monitor):
        """Test de obtención de alertas recientes."""
        recent_alerts = production_monitor.get_recent_alerts(10)
        
        assert isinstance(recent_alerts, list)
    
    def test_get_system_health(self, production_monitor):
        """Test de obtención de salud del sistema."""
        system_health = production_monitor.get_system_health(5)
        
        assert isinstance(system_health, list)
    
    def test_get_monitoring_summary(self, production_monitor):
        """Test de obtención de resumen de monitoreo."""
        summary = production_monitor.get_monitoring_summary()
        
        assert isinstance(summary, dict)
    
    def test_export_monitoring_data(self, production_monitor, temp_dir):
        """Test de exportación de datos de monitoreo."""
        filepath = temp_dir + "/test_monitoring.json"
        production_monitor.export_monitoring_data(filepath)
        
        assert os.path.exists(filepath)
        
        # Verificar contenido
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert 'monitoring_status' in data
        assert 'metrics_history' in data
        assert 'alerts' in data
        assert 'system_health' in data
        assert 'thresholds' in data
        assert 'config' in data
        assert 'export_timestamp' in data
    
    def test_get_summary(self, production_monitor):
        """Test de obtención de resumen."""
        summary = production_monitor.get_summary()
        
        assert 'config' in summary
        assert 'monitoring_status' in summary
        assert 'current_status' in summary
        assert 'metrics_count' in summary
        assert 'alerts_count' in summary
        assert 'thresholds_count' in summary
        assert 'total_metrics_collected' in summary
        assert 'total_alerts_generated' in summary


class TestProductionManager:
    """Tests para ProductionManager."""
    
    @pytest.fixture
    def production_manager(self):
        """ProductionManager de prueba."""
        return ProductionManager()
    
    def test_production_manager_initialization(self):
        """Test de inicialización de ProductionManager."""
        manager = ProductionManager()
        
        assert manager.canary_deployment is not None
        assert manager.production_monitor is not None
        assert manager.is_production_active == False
        assert manager.current_strategy is None
        assert len(manager.deployment_history) == 0
    
    def test_start_production(self, production_manager):
        """Test de inicio de producción."""
        strategy_config = {"strategy": "test", "model": "baseline"}
        
        success = production_manager.start_production(strategy_config)
        
        assert success == True
        assert production_manager.is_production_active == True
        assert production_manager.current_strategy == strategy_config
    
    def test_stop_production(self, production_manager):
        """Test de parada de producción."""
        production_manager.start_production({"strategy": "test"})
        production_manager.stop_production()
        
        assert production_manager.is_production_active == False
        assert production_manager.current_strategy is None
    
    def test_get_production_status(self, production_manager):
        """Test de obtención de estado de producción."""
        status = production_manager.get_production_status()
        
        assert 'is_production_active' in status
        assert 'current_strategy' in status
        assert 'canary_status' in status
        assert 'monitoring_status' in status
        assert 'deployment_history_count' in status
    
    def test_get_canary_status(self, production_manager):
        """Test de obtención de estado del canary."""
        status = production_manager.get_canary_status()
        
        assert 'is_deploying' in status
        assert 'current_stage' in status
        assert 'canary_percentage' in status
        assert 'health_status' in status
    
    def test_get_monitoring_status(self, production_manager):
        """Test de obtención de estado del monitoreo."""
        status = production_manager.get_monitoring_status()
        
        assert 'is_monitoring' in status
        assert 'current_status' in status
        assert 'start_time' in status
        assert 'uptime' in status
    
    def test_get_recent_metrics(self, production_manager):
        """Test de obtención de métricas recientes."""
        metrics = production_manager.get_recent_metrics(10)
        
        assert isinstance(metrics, list)
    
    def test_get_recent_alerts(self, production_manager):
        """Test de obtención de alertas recientes."""
        alerts = production_manager.get_recent_alerts(10)
        
        assert isinstance(alerts, list)
    
    def test_get_system_health(self, production_manager):
        """Test de obtención de salud del sistema."""
        health = production_manager.get_system_health(5)
        
        assert isinstance(health, list)
    
    def test_get_deployment_history(self, production_manager):
        """Test de obtención de historial de despliegue."""
        history = production_manager.get_deployment_history()
        
        assert isinstance(history, list)
    
    def test_add_monitoring_threshold(self, production_manager):
        """Test de añadir umbral de monitoreo."""
        threshold = MetricThreshold(
            metric_name="custom_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=50.0,
            critical_threshold=75.0,
            operator=">",
            description="Custom metric"
        )
        
        production_manager.add_monitoring_threshold(threshold)
        
        assert "custom_metric" in production_manager.production_monitor.thresholds
    
    def test_remove_monitoring_threshold(self, production_manager):
        """Test de eliminar umbral de monitoreo."""
        # Añadir umbral
        threshold = MetricThreshold(
            metric_name="custom_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=50.0,
            critical_threshold=75.0,
            operator=">",
            description="Custom metric"
        )
        production_manager.add_monitoring_threshold(threshold)
        
        # Eliminar umbral
        production_manager.remove_monitoring_threshold("custom_metric")
        
        assert "custom_metric" not in production_manager.production_monitor.thresholds
    
    def test_update_monitoring_threshold(self, production_manager):
        """Test de actualizar umbral de monitoreo."""
        production_manager.update_monitoring_threshold("cpu_usage", warning_threshold=70.0)
        
        assert production_manager.production_monitor.thresholds["cpu_usage"].warning_threshold == 70.0
    
    def test_acknowledge_alert(self, production_manager):
        """Test de reconocer alerta."""
        # Generar alerta
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        production_manager.add_monitoring_threshold(threshold)
        production_manager.production_monitor._add_metric("test_metric", MetricType.CUSTOM, 90.0, "%")
        production_manager.production_monitor._check_thresholds()
        
        if production_manager.production_monitor.alerts:
            alert_id = production_manager.production_monitor.alerts[0].alert_id
            success = production_manager.acknowledge_alert(alert_id, "test_user")
            assert success == True
    
    def test_resolve_alert(self, production_manager):
        """Test de resolver alerta."""
        # Generar alerta
        threshold = MetricThreshold(
            metric_name="test_metric",
            metric_type=MetricType.CUSTOM,
            warning_threshold=80.0,
            critical_threshold=95.0,
            operator=">",
            description="Test metric"
        )
        
        production_manager.add_monitoring_threshold(threshold)
        production_manager.production_monitor._add_metric("test_metric", MetricType.CUSTOM, 90.0, "%")
        production_manager.production_monitor._check_thresholds()
        
        if production_manager.production_monitor.alerts:
            alert_id = production_manager.production_monitor.alerts[0].alert_id
            success = production_manager.resolve_alert(alert_id)
            assert success == True
    
    def test_create_production_report(self, production_manager):
        """Test de creación de reporte de producción."""
        report = production_manager.create_production_report()
        
        assert isinstance(report, str)
        assert "REPORTE DE PRODUCCIÓN - SISTEMA DE TRADING" in report
        assert "ESTADO DEL CANARY" in report
        assert "ESTADO DEL MONITOREO" in report
        assert "CONFIGURACIÓN" in report
    
    def test_export_production_data(self, production_manager, temp_dir):
        """Test de exportación de datos de producción."""
        # Exportar a JSON
        json_path = temp_dir + "/test_production.json"
        production_manager.export_production_data(json_path, "json")
        assert os.path.exists(json_path)
        
        # Exportar a CSV
        csv_path = temp_dir + "/test_production.csv"
        production_manager.export_production_data(csv_path, "csv")
        assert os.path.exists(csv_path.with_suffix('.metrics.csv'))
        assert os.path.exists(csv_path.with_suffix('.alerts.csv'))
        assert os.path.exists(csv_path.with_suffix('.health.csv'))
        
        # Exportar a Excel
        excel_path = temp_dir + "/test_production.xlsx"
        production_manager.export_production_data(excel_path, "excel")
        assert os.path.exists(excel_path)
    
    def test_save_load_production_state(self, production_manager, temp_dir):
        """Test de guardado y carga de estado de producción."""
        # Guardar estado
        save_path = temp_dir + "/test_state.pkl"
        production_manager.save_production_state(save_path)
        assert os.path.exists(save_path)
        
        # Cargar estado
        new_manager = ProductionManager()
        new_manager.load_production_state(save_path)
        
        # Verificar que se cargó el estado
        assert new_manager.canary_deployment is not None
        assert new_manager.production_monitor is not None
    
    def test_get_summary(self, production_manager):
        """Test de obtención de resumen."""
        summary = production_manager.get_summary()
        
        assert 'config' in summary
        assert 'production_status' in summary
        assert 'canary_summary' in summary
        assert 'monitoring_summary' in summary
        assert 'deployment_history_count' in summary


if __name__ == "__main__":
    pytest.main([__file__])

