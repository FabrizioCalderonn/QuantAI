#!/usr/bin/env python3
"""
Script principal para ejecutar el sistema de trading cuantitativo.
"""

import argparse
import sys
from pathlib import Path
import logging

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.pipeline import DataPipeline
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.ml_trainer import MLModelTrainer
from src.risk.risk_manager import RiskManager
from src.backtesting.backtest_manager import BacktestManager
from src.backtesting.base import BacktestType
from src.validation.validation_manager import ValidationManager
from src.validation.validator import ValidationLevel
from src.explainability.explainability_manager import ExplainabilityManager
from src.paper_trading.paper_trading_manager import PaperTradingManager
from src.paper_trading.portfolio import PositionSide, OrderType
from src.production.production_manager import ProductionManager
from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def run_data_pipeline(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta el pipeline de datos completo.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando pipeline de datos...")
    
    try:
        # Cargar y validar configuración
        config = load_config(config_path)
        validate_config(config)
        logger.info("Configuración cargada y validada exitosamente")
        
        # Crear pipeline
        pipeline = DataPipeline(config_path)
        
        # Ejecutar pipeline completo
        data = pipeline.run_full_pipeline()
        
        logger.info(f"Pipeline completado: {len(data)} instrumentos procesados")
        
        # Crear features
        logger.info("Iniciando feature engineering...")
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.create_features(data)
        
        # Guardar features
        feature_engineer.save_features(features, "data/processed/features")
        
        logger.info(f"Features creados: {len(features)} instrumentos")
        
        # Entrenar modelos baseline
        logger.info("Iniciando entrenamiento de modelos baseline...")
        trainer = ModelTrainer(config_path)
        trained_models = trainer.train_models(features)
        
        # Validar modelos baseline
        validation_results = trainer.validate_models(features)
        
        # Guardar modelos baseline
        trainer.save_models("models/baseline_models")
        
        # Generar reporte baseline
        report = trainer.generate_model_report()
        
        logger.info(f"Modelos baseline entrenados: {len(trained_models)}")
        
        # Entrenar modelos ML
        logger.info("Iniciando entrenamiento de modelos ML...")
        ml_trainer = MLModelTrainer(config_path)
        ml_trained_models = ml_trainer.train_models(features, optimize_hyperparams=True)
        
        # Validar modelos ML
        ml_validation_results = ml_trainer.validate_models(features)
        
        # Guardar modelos ML
        ml_trainer.save_models("models/ml_models")
        
        # Generar reporte ML
        ml_report = ml_trainer.generate_model_report()
        
        # Comparar con baseline
        comparison_report = ml_trainer.compare_with_baseline(validation_results)
        
        logger.info(f"Modelos ML entrenados: {len(ml_trained_models)}")
        
        # Inicializar sistema de gestión de riesgo
        logger.info("Inicializando sistema de gestión de riesgo...")
        risk_manager = RiskManager(config_path)
        
        # Simular posiciones para demo
        from src.risk.base import Position
        demo_positions = [
            Position(
                symbol="SPY",
                quantity=100.0,
                price=400.0,
                timestamp=datetime.now(),
                side="long"
            ),
            Position(
                symbol="QQQ",
                quantity=50.0,
                price=300.0,
                timestamp=datetime.now(),
                side="long"
            )
        ]
        
        # Actualizar portafolio con gestión de riesgo
        current_prices = {"SPY": 410.0, "QQQ": 310.0}
        risk_summary = risk_manager.update_portfolio(demo_positions, current_prices)
        
        logger.info("Sistema de gestión de riesgo inicializado")
        
        # Inicializar sistema de backtesting
        logger.info("Inicializando sistema de backtesting...")
        backtest_manager = BacktestManager(config_path)
        
        # Ejecutar backtesting comprensivo
        backtest_results = backtest_manager.run_comprehensive_backtest(
            data, features, benchmark_data=None, model_trainer=None
        )
        
        logger.info("Sistema de backtesting inicializado")
        
        # Inicializar sistema de validación
        logger.info("Inicializando sistema de validación...")
        validation_manager = ValidationManager(config_path)
        
        # Validar resultados de backtesting
        validation_results = validation_manager.validate_backtest_results(
            backtest_results, ValidationLevel.INTERMEDIATE
        )
        
        logger.info("Sistema de validación inicializado")
        
        # Inicializar sistema de explainability
        logger.info("Inicializando sistema de explainability...")
        explainability_manager = ExplainabilityManager(config_path)
        
        # Explicar modelos (si están disponibles)
        try:
            # Explicar modelos baseline
            if 'baseline_models' in locals():
                for model_name, model in baseline_models.items():
                    if hasattr(model, 'predict'):
                        explanations = explainability_manager.explain_model(
                            model, features, features.head(100), model_name
                        )
                        logger.info(f"Modelo {model_name} explicado")
            
            # Explicar modelos ML
            if 'ml_models' in locals():
                for model_name, model in ml_models.items():
                    if hasattr(model, 'predict'):
                        explanations = explainability_manager.explain_model(
                            model, features, features.head(100), model_name
                        )
                        logger.info(f"Modelo ML {model_name} explicado")
        
        except Exception as e:
            logger.warning(f"Error explicando modelos: {str(e)}")
        
        logger.info("Sistema de explainability inicializado")
        
        # Inicializar sistema de paper trading
        logger.info("Inicializando sistema de paper trading...")
        paper_trading_manager = PaperTradingManager(config_path)
        
        # Iniciar paper trading
        paper_trading_manager.start_trading()
        
        # Simular algunas operaciones de trading
        try:
            # Actualizar datos de mercado
            paper_trading_manager.update_market_data("SPY", 400.0, 1000000.0)
            paper_trading_manager.update_market_data("QQQ", 300.0, 800000.0)
            
            # Colocar algunas órdenes de ejemplo
            order1 = paper_trading_manager.place_market_order("SPY", PositionSide.LONG, 100.0)
            order2 = paper_trading_manager.place_limit_order("QQQ", PositionSide.LONG, 50.0, 295.0)
            
            logger.info(f"Órdenes colocadas: {order1}, {order2}")
            
        except Exception as e:
            logger.warning(f"Error en operaciones de paper trading: {str(e)}")
        
        logger.info("Sistema de paper trading inicializado")
        
        # Inicializar sistema de producción
        logger.info("Inicializando sistema de producción...")
        production_manager = ProductionManager(config_path)
        
        # Simular inicio de producción
        try:
            strategy_config = {
                "strategy": "baseline_momentum",
                "model": "momentum",
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02
                }
            }
            
            # Iniciar producción
            production_success = production_manager.start_production(strategy_config)
            
            if production_success:
                logger.info("Sistema de producción iniciado")
            else:
                logger.warning("Error iniciando sistema de producción")
            
        except Exception as e:
            logger.warning(f"Error en sistema de producción: {str(e)}")
        
        logger.info("Sistema de producción inicializado")
        
        # Mostrar resumen
        print("\n=== RESUMEN DEL PIPELINE ===")
        for symbol, df in data.items():
            print(f"{symbol}: {len(df)} registros desde {df.index.min()} hasta {df.index.max()}")
        
        print("\n=== RESUMEN DE FEATURES ===")
        for symbol, feature_df in features.items():
            print(f"{symbol}: {len(feature_df.columns)} features, {len(feature_df)} registros")
        
        print("\n=== RESUMEN DE MODELOS BASELINE ===")
        if not report.empty:
            print(report.to_string(index=False))
        else:
            print("No se generó reporte de modelos baseline")
        
        print("\n=== RESUMEN DE MODELOS ML ===")
        if not ml_report.empty:
            print(ml_report.to_string(index=False))
        else:
            print("No se generó reporte de modelos ML")
        
        print("\n=== COMPARACIÓN BASELINE vs ML ===")
        if not comparison_report.empty:
            print(comparison_report.to_string(index=False))
        else:
            print("No se generó reporte de comparación")
        
        print("\n=== RESUMEN DE GESTIÓN DE RIESGO ===")
        risk_summary = risk_manager.get_risk_summary()
        print(f"Valor del portafolio: ${risk_summary['portfolio']['value']:,.2f}")
        print(f"Posiciones: {risk_summary['portfolio']['positions_count']}")
        print(f"Alertas activas: {risk_summary['alerts']['total']}")
        print(f"Circuit breakers activos: {risk_summary['circuit_breakers']['active_circuit_breakers']}")
        
        print("\n=== RESUMEN DE BACKTESTING ===")
        backtest_summary = backtest_manager.get_summary()
        print(f"Métodos ejecutados: {backtest_summary['results_count']}")
        print(f"Última ejecución: {backtest_summary['last_run']}")
        
        # Mostrar resumen de walk-forward
        if 'walk_forward' in backtest_results:
            wf_summary = backtest_results['walk_forward']['summary']
            print(f"Walk-Forward - Períodos: {wf_summary['total_periods']}, Trades: {wf_summary['total_trades']}")
            print(f"Walk-Forward - Retorno Promedio: {wf_summary['avg_return']:.2%}, Sharpe: {wf_summary['avg_sharpe']:.2f}")
        
        # Mostrar resumen de purged CV
        if 'purged_cv' in backtest_results:
            cv_summary = backtest_results['purged_cv']['summary']
            print(f"Purged CV - Folds: {cv_summary['total_folds']}, Trades: {cv_summary['total_trades']}")
            print(f"Purged CV - Retorno Promedio: {cv_summary['avg_return']:.2%}, Sharpe: {cv_summary['avg_sharpe']:.2f}")
        
        # Mostrar recomendaciones
        if 'summary' in backtest_results and 'recommendations' in backtest_results['summary']:
            print("\n=== RECOMENDACIONES ===")
            for i, rec in enumerate(backtest_results['summary']['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("\n=== RESUMEN DE VALIDACIÓN ===")
        validation_summary = validation_manager.get_validation_summary()
        print(f"Estrategias validadas: {validation_summary['total_strategies']}")
        print(f"Estrategias que pasaron: {validation_summary['passed_strategies']}")
        print(f"Tasa de aprobación: {validation_summary['pass_rate']:.2%}")
        print(f"Score promedio: {validation_summary['avg_score']:.2f}")
        
        # Mostrar resultados por estrategia
        for strategy_name, result in validation_results.items():
            status = "PASÓ" if result.passed else "FALLÓ"
            print(f"{strategy_name}: {status} (Score: {result.overall_score:.2f})")
        
        # Mostrar recomendaciones de validación
        if validation_results:
            print("\n=== RECOMENDACIONES DE VALIDACIÓN ===")
            for strategy_name, result in validation_results.items():
                if result.recommendations:
                    print(f"\n{strategy_name}:")
                    for i, rec in enumerate(result.recommendations, 1):
                        print(f"  {i}. {rec}")
        
        print("\n=== RESUMEN DE EXPLICABILIDAD ===")
        explainability_summary = explainability_manager.get_summary()
        print(f"Modelos explicados: {explainability_summary['model_explanations_count']}")
        print(f"Modelos: {', '.join(explainability_summary['models_explained'])}")
        
        # Mostrar resumen de auditoría
        audit_summary = explainability_summary['audit_summary']
        print(f"Eventos de auditoría: {audit_summary['events_count']}")
        print(f"Sesión: {audit_summary['session_id']}")
        
        # Crear reporte de explicación
        explanation_report = explainability_manager.create_explanation_report()
        print(f"\n=== REPORTE DE EXPLICABILIDAD ===")
        print(explanation_report)
        
        print("\n=== RESUMEN DE PAPER TRADING ===")
        paper_trading_summary = paper_trading_manager.get_summary()
        print(f"Trading activo: {paper_trading_summary['is_running']}")
        print(f"Tiempo de inicio: {paper_trading_summary['start_time']}")
        
        # Mostrar resumen del portfolio
        portfolio_summary = paper_trading_summary['portfolio_summary']
        print(f"Valor del portfolio: ${portfolio_summary['current_value']:,.2f}")
        print(f"Cash: ${portfolio_summary['cash']:,.2f}")
        print(f"Posiciones: {portfolio_summary['positions_count']}")
        print(f"Órdenes: {portfolio_summary['orders_count']}")
        print(f"Trades: {portfolio_summary['trades_count']}")
        
        # Mostrar estadísticas de ejecución
        execution_stats = paper_trading_summary['execution_stats']
        print(f"Ejecuciones totales: {execution_stats['total_executions']}")
        print(f"Tasa de éxito: {execution_stats['success_rate']:.2%}")
        print(f"Latencia promedio: {execution_stats['avg_latency_ms']:.2f} ms")
        print(f"Slippage promedio: {execution_stats['avg_slippage']:.4f}")
        
        # Crear reporte de paper trading
        paper_trading_report = paper_trading_manager.create_performance_report()
        print(f"\n=== REPORTE DE PAPER TRADING ===")
        print(paper_trading_report)
        
        print("\n=== RESUMEN DE PRODUCCIÓN ===")
        production_summary = production_manager.get_summary()
        print(f"Producción activa: {production_summary['production_status']['is_production_active']}")
        
        # Mostrar estado del canary
        canary_status = production_summary['canary_summary']['deployment_status']
        print(f"Canary desplegando: {canary_status['is_deploying']}")
        print(f"Etapa actual: {canary_status['current_stage']}")
        print(f"Porcentaje canary: {canary_status['canary_percentage']:.1f}%")
        print(f"Estado de salud: {canary_status['health_status']}")
        
        # Mostrar estado del monitoreo
        monitoring_status = production_summary['monitoring_summary']['monitoring_status']
        print(f"Monitoreo activo: {monitoring_status['is_monitoring']}")
        print(f"Estado actual: {monitoring_status['current_status']}")
        print(f"Métricas recopiladas: {monitoring_status['metrics_collected']}")
        print(f"Alertas generadas: {monitoring_status['alerts_generated']}")
        print(f"Alertas activas: {monitoring_status['active_alerts']}")
        print(f"Alertas críticas: {monitoring_status['critical_alerts']}")
        
        # Crear reporte de producción
        production_report = production_manager.create_production_report()
        print(f"\n=== REPORTE DE PRODUCCIÓN ===")
        print(production_report)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en pipeline de datos: {str(e)}")
        return False


def run_production(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el sistema de producción.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando sistema de producción...")
    
    try:
        # Inicializar gestor de producción
        production_manager = ProductionManager(config_path)
        
        # Configurar estrategia de ejemplo
        strategy_config = {
            "strategy": "baseline_momentum",
            "model": "momentum",
            "parameters": {
                "lookback_period": 20,
                "threshold": 0.02,
                "risk_limit": 0.05
            },
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.04
            }
        }
        
        # Iniciar producción
        success = production_manager.start_production(strategy_config)
        
        if not success:
            logger.error("Error iniciando sistema de producción")
            return False
        
        logger.info("Sistema de producción iniciado")
        
        # Simular operación por un tiempo
        import time
        logger.info("Simulando operación de producción...")
        time.sleep(5)  # Simular 5 segundos de operación
        
        # Mostrar resumen
        print("\n=== SISTEMA DE PRODUCCIÓN ===")
        production_summary = production_manager.get_summary()
        print(f"Producción activa: {production_summary['production_status']['is_production_active']}")
        print(f"Estrategia: {production_summary['production_status']['current_strategy']['strategy']}")
        print(f"Modelo: {production_summary['production_status']['current_strategy']['model']}")
        
        # Mostrar estado del canary
        canary_status = production_summary['canary_summary']['deployment_status']
        print(f"\n=== ESTADO DEL CANARY ===")
        print(f"Desplegando: {canary_status['is_deploying']}")
        print(f"Etapa actual: {canary_status['current_stage']}")
        print(f"Porcentaje canary: {canary_status['canary_percentage']:.1f}%")
        print(f"Estado de salud: {canary_status['health_status']}")
        print(f"Tiempo de inicio: {canary_status['start_time']}")
        print(f"Última actualización: {canary_status['last_update']}")
        print(f"Métricas recopiladas: {canary_status['metrics_count']}")
        print(f"Alertas generadas: {canary_status['alerts_count']}")
        print(f"Total de despliegues: {canary_status['total_deployments']}")
        print(f"Despliegues exitosos: {canary_status['successful_deployments']}")
        print(f"Despliegues fallidos: {canary_status['failed_deployments']}")
        print(f"Rollbacks: {canary_status['rollbacks']}")
        
        # Mostrar estado del monitoreo
        monitoring_status = production_summary['monitoring_summary']['monitoring_status']
        print(f"\n=== ESTADO DEL MONITOREO ===")
        print(f"Monitoreando: {monitoring_status['is_monitoring']}")
        print(f"Estado actual: {monitoring_status['current_status']}")
        print(f"Tiempo de inicio: {monitoring_status['start_time']}")
        print(f"Uptime: {monitoring_status['uptime']:.0f} segundos")
        print(f"Métricas recopiladas: {monitoring_status['metrics_collected']}")
        print(f"Alertas generadas: {monitoring_status['alerts_generated']}")
        print(f"Alertas activas: {monitoring_status['active_alerts']}")
        print(f"Alertas críticas: {monitoring_status['critical_alerts']}")
        print(f"Umbrales configurados: {monitoring_status['thresholds_count']}")
        
        # Mostrar métricas recientes
        recent_metrics = production_manager.get_recent_metrics(10)
        if recent_metrics:
            print(f"\n=== MÉTRICAS RECIENTES ===")
            for metric in recent_metrics[-5:]:  # Últimas 5
                print(f"{metric['metric_name']}: {metric['value']:.4f} {metric['unit']} ({metric['timestamp']})")
        
        # Mostrar alertas recientes
        recent_alerts = production_manager.get_recent_alerts(10)
        if recent_alerts:
            print(f"\n=== ALERTAS RECIENTES ===")
            for alert in recent_alerts[-5:]:  # Últimas 5
                print(f"{alert['severity']}: {alert['message']} ({alert['timestamp']})")
        
        # Mostrar salud del sistema
        system_health = production_manager.get_system_health(1)
        if system_health:
            health = system_health[0]
            print(f"\n=== SALUD DEL SISTEMA ===")
            print(f"Estado general: {health['overall_status']}")
            print(f"Alertas activas: {health['active_alerts']}")
            print(f"Alertas críticas: {health['critical_alerts']}")
            
            # Mostrar métricas por tipo
            if health['system_metrics']:
                print(f"\nMétricas del sistema:")
                for name, value in list(health['system_metrics'].items())[:3]:
                    print(f"  {name}: {value:.4f}")
            
            if health['trading_metrics']:
                print(f"\nMétricas de trading:")
                for name, value in list(health['trading_metrics'].items())[:3]:
                    print(f"  {name}: {value:.4f}")
            
            if health['risk_metrics']:
                print(f"\nMétricas de riesgo:")
                for name, value in list(health['risk_metrics'].items())[:3]:
                    print(f"  {name}: {value:.4f}")
            
            if health['performance_metrics']:
                print(f"\nMétricas de performance:")
                for name, value in list(health['performance_metrics'].items())[:3]:
                    print(f"  {name}: {value:.4f}")
        
        # Mostrar historial de despliegue
        deployment_history = production_manager.get_deployment_history()
        if deployment_history:
            print(f"\n=== HISTORIAL DE DESPLIEGUE ===")
            for event in deployment_history[-5:]:  # Últimos 5 eventos
                if 'event' in event:
                    print(f"{event['timestamp']}: {event['event']} - {event.get('reason', '')}")
                else:
                    print(f"{event['timestamp']}: {event['old_stage']} -> {event['new_stage']} ({event['canary_percentage']:.1f}%)")
        
        # Mostrar configuración
        print(f"\n=== CONFIGURACIÓN ===")
        canary_config = production_manager.canary_deployment.config
        print(f"Porcentaje inicial canary: {canary_config.initial_canary_percentage}%")
        print(f"Porcentaje máximo canary: {canary_config.max_canary_percentage}%")
        print(f"Incremento canary: {canary_config.canary_increment}%")
        print(f"Duración canary: {canary_config.canary_duration_minutes} minutos")
        print(f"Duración rollout: {canary_config.rollout_duration_minutes} minutos")
        print(f"Duración despliegue completo: {canary_config.full_deployment_duration_minutes} minutos")
        print(f"Tasa de éxito mínima: {canary_config.min_success_rate:.2%}")
        print(f"Tasa de error máxima: {canary_config.max_error_rate:.2%}")
        print(f"Latencia P95 máxima: {canary_config.max_latency_p95} ms")
        print(f"Drawdown máximo: {canary_config.max_drawdown_threshold:.2%}")
        print(f"Sharpe ratio mínimo: {canary_config.min_sharpe_ratio}")
        
        # Crear reporte completo
        production_report = production_manager.create_production_report()
        print(f"\n=== REPORTE COMPLETO DE PRODUCCIÓN ===")
        print(production_report)
        
        # Detener producción
        production_manager.stop_production()
        
        return True
        
    except Exception as e:
        logger.error(f"Error en sistema de producción: {str(e)}")
        return False


def run_paper_trading(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el sistema de paper trading.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando sistema de paper trading...")
    
    try:
        # Inicializar gestor de paper trading
        paper_trading_manager = PaperTradingManager(config_path)
        
        # Iniciar paper trading
        paper_trading_manager.start_trading()
        
        # Simular datos de mercado
        logger.info("Simulando datos de mercado...")
        
        # Actualizar precios de mercado
        symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        prices = {"SPY": 400.0, "QQQ": 300.0, "IWM": 200.0, "GLD": 180.0, "TLT": 100.0}
        volumes = {"SPY": 1000000.0, "QQQ": 800000.0, "IWM": 500000.0, "GLD": 300000.0, "TLT": 200000.0}
        
        for symbol in symbols:
            paper_trading_manager.update_market_data(symbol, prices[symbol], volumes[symbol])
        
        # Simular estrategia de trading
        logger.info("Simulando estrategia de trading...")
        
        # Estrategia simple: comprar SPY y QQQ
        orders = []
        
        # Orden de mercado para SPY
        order1 = paper_trading_manager.place_market_order("SPY", PositionSide.LONG, 100.0)
        orders.append(order1)
        
        # Orden limit para QQQ
        order2 = paper_trading_manager.place_limit_order("QQQ", PositionSide.LONG, 50.0, 295.0)
        orders.append(order2)
        
        # Orden stop para IWM
        order3 = paper_trading_manager.place_stop_order("IWM", PositionSide.LONG, 75.0, 205.0)
        orders.append(order3)
        
        # Orden stop-limit para GLD
        order4 = paper_trading_manager.place_stop_limit_order("GLD", PositionSide.LONG, 25.0, 185.0, 190.0)
        orders.append(order4)
        
        logger.info(f"Órdenes colocadas: {orders}")
        
        # Simular movimiento de precios
        logger.info("Simulando movimiento de precios...")
        
        # Actualizar precios para ejecutar algunas órdenes
        paper_trading_manager.update_market_data("SPY", 401.0, 1100000.0)  # Ejecutar orden de mercado
        paper_trading_manager.update_market_data("QQQ", 294.0, 850000.0)   # Ejecutar orden limit
        paper_trading_manager.update_market_data("IWM", 206.0, 550000.0)   # Ejecutar orden stop
        paper_trading_manager.update_market_data("GLD", 186.0, 320000.0)   # Ejecutar orden stop-limit
        
        # Esperar un poco para que se procesen las órdenes
        import time
        time.sleep(2.0)
        
        # Mostrar resumen
        print("\n=== SISTEMA DE PAPER TRADING ===")
        paper_trading_summary = paper_trading_manager.get_summary()
        print(f"Trading activo: {paper_trading_summary['is_running']}")
        print(f"Tiempo de inicio: {paper_trading_summary['start_time']}")
        
        # Mostrar resumen del portfolio
        portfolio_summary = paper_trading_summary['portfolio_summary']
        print(f"Valor del portfolio: ${portfolio_summary['current_value']:,.2f}")
        print(f"Cash: ${portfolio_summary['cash']:,.2f}")
        print(f"Posiciones: {portfolio_summary['positions_count']}")
        print(f"Órdenes: {portfolio_summary['orders_count']}")
        print(f"Trades: {portfolio_summary['trades_count']}")
        
        # Mostrar métricas de performance
        metrics = paper_trading_manager.get_performance_metrics()
        print(f"\n=== MÉTRICAS DE PERFORMANCE ===")
        print(f"Retorno total: {metrics.get('total_return', 0):.2%}")
        print(f"PnL total: ${metrics.get('total_pnl', 0):,.2f}")
        print(f"PnL realizado: ${metrics.get('realized_pnl', 0):,.2f}")
        print(f"PnL no realizado: ${metrics.get('unrealized_pnl', 0):,.2f}")
        print(f"Comisiones pagadas: ${metrics.get('commission_paid', 0):,.2f}")
        print(f"Total de trades: {metrics.get('total_trades', 0)}")
        print(f"Win rate: {metrics.get('win_rate', 0):.2%}")
        print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Maximum drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        # Mostrar posiciones actuales
        positions = paper_trading_manager.get_positions()
        if positions:
            print(f"\n=== POSICIONES ACTUALES ===")
            for symbol, pos in positions.items():
                print(f"{symbol}: {pos['quantity']} {pos['side']} @ {pos['entry_price']:.2f} (PnL: ${pos['unrealized_pnl']:,.2f})")
        
        # Mostrar órdenes
        orders = paper_trading_manager.get_orders()
        if orders:
            print(f"\n=== ÓRDENES ===")
            for order in orders:
                print(f"{order['order_id']}: {order['side']} {order['quantity']} {order['symbol']} - {order['status']}")
        
        # Mostrar trades
        trades = paper_trading_manager.get_trades()
        if trades:
            print(f"\n=== TRADES EJECUTADOS ===")
            for trade in trades:
                print(f"{trade['trade_id']}: {trade['side']} {trade['quantity']} {trade['symbol']} @ {trade['price']:.2f}")
        
        # Mostrar estadísticas de ejecución
        execution_stats = paper_trading_summary['execution_stats']
        print(f"\n=== ESTADÍSTICAS DE EJECUCIÓN ===")
        print(f"Ejecuciones totales: {execution_stats['total_executions']}")
        print(f"Ejecuciones exitosas: {execution_stats['successful_executions']}")
        print(f"Ejecuciones fallidas: {execution_stats['failed_executions']}")
        print(f"Tasa de éxito: {execution_stats['success_rate']:.2%}")
        print(f"Latencia promedio: {execution_stats['avg_latency_ms']:.2f} ms")
        print(f"Slippage promedio: {execution_stats['avg_slippage']:.4f}")
        
        # Mostrar configuración
        print(f"\n=== CONFIGURACIÓN ===")
        print(f"Modo de ejecución: {execution_stats['config']['mode']}")
        print(f"Latencia base: {execution_stats['config']['base_latency_ms']} ms")
        print(f"Tasa de slippage: {execution_stats['config']['slippage_rate']:.4f}")
        print(f"Probabilidad de llenado: {execution_stats['config']['fill_probability']:.2%}")
        
        # Crear reporte completo
        paper_trading_report = paper_trading_manager.create_performance_report()
        print(f"\n=== REPORTE COMPLETO DE PAPER TRADING ===")
        print(paper_trading_report)
        
        # Detener paper trading
        paper_trading_manager.stop_trading()
        
        return True
        
    except Exception as e:
        logger.error(f"Error en sistema de paper trading: {str(e)}")
        return False


def run_explainability(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el sistema de explainability.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando sistema de explainability...")
    
    try:
        # Cargar datos de muestra
        from src.data.pipeline import DataPipeline
        from src.features.engineering import FeatureEngineer
        from src.models.trainer import ModelTrainer
        from src.models.ml_trainer import MLModelTrainer
        
        # Crear pipeline de datos
        data_pipeline = DataPipeline(config_path)
        data = data_pipeline.run_pipeline()
        
        if not data:
            logger.error("No se pudieron cargar los datos")
            return False
        
        # Crear features
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.create_features(data)
        
        if not features:
            logger.error("No se pudieron crear las features")
            return False
        
        # Entrenar modelos para explicar
        logger.info("Entrenando modelos para explicar...")
        
        # Modelos baseline
        baseline_trainer = ModelTrainer(config_path)
        baseline_models = baseline_trainer.train_all_models(data, features)
        
        # Modelos ML
        ml_trainer = MLModelTrainer(config_path)
        ml_models = ml_trainer.train_all_models(data, features)
        
        # Inicializar gestor de explainability
        explainability_manager = ExplainabilityManager(config_path)
        
        # Explicar modelos baseline
        logger.info("Explicando modelos baseline...")
        baseline_explanations = {}
        for model_name, model in baseline_models.items():
            if hasattr(model, 'predict'):
                try:
                    explanations = explainability_manager.explain_model(
                        model, features, features.head(100), f"baseline_{model_name}"
                    )
                    baseline_explanations[model_name] = explanations
                    logger.info(f"Modelo baseline {model_name} explicado")
                except Exception as e:
                    logger.warning(f"Error explicando modelo baseline {model_name}: {str(e)}")
        
        # Explicar modelos ML
        logger.info("Explicando modelos ML...")
        ml_explanations = {}
        for model_name, model in ml_models.items():
            if hasattr(model, 'predict'):
                try:
                    explanations = explainability_manager.explain_model(
                        model, features, features.head(100), f"ml_{model_name}"
                    )
                    ml_explanations[model_name] = explanations
                    logger.info(f"Modelo ML {model_name} explicado")
                except Exception as e:
                    logger.warning(f"Error explicando modelo ML {model_name}: {str(e)}")
        
        # Mostrar resumen
        print("\n=== SISTEMA DE EXPLICABILIDAD ===")
        explainability_summary = explainability_manager.get_summary()
        print(f"Modelos explicados: {explainability_summary['model_explanations_count']}")
        print(f"Modelos: {', '.join(explainability_summary['models_explained'])}")
        
        # Mostrar resumen de auditoría
        audit_summary = explainability_summary['audit_summary']
        print(f"Eventos de auditoría: {audit_summary['events_count']}")
        print(f"Sesión: {audit_summary['session_id']}")
        
        # Mostrar explicaciones por modelo
        print(f"\n=== EXPLICACIONES DE MODELOS BASELINE ===")
        for model_name, explanations in baseline_explanations.items():
            print(f"\n{model_name}:")
            if 'summary' in explanations:
                summary = explanations['summary']
                print(f"  Features: {summary['features_count']}")
                print(f"  Muestras: {summary['samples_count']}")
                print(f"  Valor base: {summary['base_value']:.4f}")
            
            if 'feature_importance' in explanations:
                feature_importance = explanations['feature_importance']
                print(f"  Top 5 features:")
                for i, fi in enumerate(feature_importance[:5], 1):
                    print(f"    {i}. {fi.feature_name}: {fi.importance:.4f} ({fi.direction})")
        
        print(f"\n=== EXPLICACIONES DE MODELOS ML ===")
        for model_name, explanations in ml_explanations.items():
            print(f"\n{model_name}:")
            if 'summary' in explanations:
                summary = explanations['summary']
                print(f"  Features: {summary['features_count']}")
                print(f"  Muestras: {summary['samples_count']}")
                print(f"  Valor base: {summary['base_value']:.4f}")
            
            if 'feature_importance' in explanations:
                feature_importance = explanations['feature_importance']
                print(f"  Top 5 features:")
                for i, fi in enumerate(feature_importance[:5], 1):
                    print(f"    {i}. {fi.feature_name}: {fi.importance:.4f} ({fi.direction})")
        
        # Demostrar explicación de predicción
        print(f"\n=== DEMOSTRACIÓN DE EXPLICACIÓN DE PREDICCIÓN ===")
        if baseline_models:
            model_name = list(baseline_models.keys())[0]
            model = baseline_models[model_name]
            
            # Explicar predicción
            prediction_explanation = explainability_manager.explain_prediction(
                model, features, features.head(1), f"baseline_{model_name}"
            )
            
            print(f"Modelo: {model_name}")
            print(f"Predicción: {prediction_explanation['prediction']:.4f}")
            print(f"Confianza: {prediction_explanation['confidence']:.2%}")
            print(f"Razonamiento: {prediction_explanation['reasoning']}")
        
        # Demostrar explicación de decisión de trading
        print(f"\n=== DEMOSTRACIÓN DE EXPLICACIÓN DE TRADING ===")
        trade_explanation = explainability_manager.explain_trade_decision(
            symbol="SPY",
            action="buy",
            quantity=100.0,
            price=400.0,
            model_explanation=prediction_explanation if 'prediction_explanation' in locals() else None,
            risk_assessment={"risk_level": "medium", "volatility": 0.15}
        )
        
        print(f"Trade ID: {trade_explanation['trade_id']}")
        print(f"Acción: {trade_explanation['action']} {trade_explanation['quantity']} {trade_explanation['symbol']} @ {trade_explanation['price']}")
        print(f"Razonamiento: {trade_explanation['reasoning']}")
        
        # Crear reporte completo
        explanation_report = explainability_manager.create_explanation_report()
        print(f"\n=== REPORTE COMPLETO DE EXPLICABILIDAD ===")
        print(explanation_report)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en sistema de explainability: {str(e)}")
        return False


def run_validation(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el sistema de validación.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando sistema de validación...")
    
    try:
        # Cargar datos de muestra
        from src.data.pipeline import DataPipeline
        from src.features.engineering import FeatureEngineer
        from src.backtesting.backtest_manager import BacktestManager
        
        # Crear pipeline de datos
        data_pipeline = DataPipeline(config_path)
        data = data_pipeline.run_pipeline()
        
        if not data:
            logger.error("No se pudieron cargar los datos")
            return False
        
        # Crear features
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.create_features(data)
        
        if not features:
            logger.error("No se pudieron crear las features")
            return False
        
        # Ejecutar backtesting para obtener resultados
        backtest_manager = BacktestManager(config_path)
        backtest_results = backtest_manager.run_comprehensive_backtest(
            data, features, benchmark_data=None, model_trainer=None
        )
        
        # Inicializar gestor de validación
        validation_manager = ValidationManager(config_path)
        
        # Validar resultados de backtesting
        logger.info("Validando resultados de backtesting...")
        validation_results = validation_manager.validate_backtest_results(
            backtest_results, ValidationLevel.INTERMEDIATE
        )
        
        # Mostrar resumen
        print("\n=== SISTEMA DE VALIDACIÓN ===")
        validation_summary = validation_manager.get_validation_summary()
        print(f"Estrategias validadas: {validation_summary['total_strategies']}")
        print(f"Estrategias que pasaron: {validation_summary['passed_strategies']}")
        print(f"Estrategias que fallaron: {validation_summary['failed_strategies']}")
        print(f"Tasa de aprobación: {validation_summary['pass_rate']:.2%}")
        print(f"Score promedio: {validation_summary['avg_score']:.2f}")
        print(f"Score mínimo: {validation_summary['min_score']:.2f}")
        print(f"Score máximo: {validation_summary['max_score']:.2f}")
        
        # Mostrar resultados por estrategia
        print(f"\n=== RESULTADOS POR ESTRATEGIA ===")
        for strategy_name, result in validation_results.items():
            status = "PASÓ" if result.passed else "FALLÓ"
            print(f"{strategy_name}: {status} (Score: {result.overall_score:.2f})")
            print(f"  Nivel: {result.validation_level.value}")
            print(f"  Métricas fallidas: {len(result.failed_metrics)}")
            print(f"  Advertencias: {len(result.warnings)}")
            print(f"  Recomendaciones: {len(result.recommendations)}")
        
        # Mostrar métricas más problemáticas
        if validation_summary['most_problematic_metrics']:
            print(f"\n=== MÉTRICAS MÁS PROBLEMÁTICAS ===")
            for metric, count in validation_summary['most_problematic_metrics']:
                print(f"{metric}: {count} fallos")
        
        # Mostrar recomendaciones detalladas
        print(f"\n=== RECOMENDACIONES DETALLADAS ===")
        for strategy_name, result in validation_results.items():
            if result.recommendations:
                print(f"\n{strategy_name}:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"  {i}. {rec}")
        
        # Crear reporte
        report = validation_manager.create_validation_report()
        print(f"\n=== REPORTE COMPLETO ===")
        print(report)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en sistema de validación: {str(e)}")
        return False


def run_backtesting(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el sistema de backtesting.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando sistema de backtesting...")
    
    try:
        # Cargar datos de muestra
        from src.data.pipeline import DataPipeline
        from src.features.engineering import FeatureEngineer
        
        # Crear pipeline de datos
        data_pipeline = DataPipeline(config_path)
        data = data_pipeline.run_pipeline()
        
        if not data:
            logger.error("No se pudieron cargar los datos")
            return False
        
        # Crear features
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.create_features(data)
        
        if not features:
            logger.error("No se pudieron crear las features")
            return False
        
        # Inicializar gestor de backtesting
        backtest_manager = BacktestManager(config_path)
        
        # Ejecutar backtesting comprensivo
        logger.info("Ejecutando backtesting comprensivo...")
        backtest_results = backtest_manager.run_comprehensive_backtest(
            data, features, benchmark_data=None, model_trainer=None
        )
        
        # Mostrar resumen
        print("\n=== SISTEMA DE BACKTESTING ===")
        backtest_summary = backtest_manager.get_summary()
        print(f"Métodos ejecutados: {backtest_summary['results_count']}")
        print(f"Última ejecución: {backtest_summary['last_run']}")
        
        # Mostrar resumen de walk-forward
        if 'walk_forward' in backtest_results:
            wf_summary = backtest_results['walk_forward']['summary']
            print(f"\n=== WALK-FORWARD ANALYSIS ===")
            print(f"Períodos: {wf_summary['total_periods']}")
            print(f"Total Trades: {wf_summary['total_trades']}")
            print(f"Retorno Promedio: {wf_summary['avg_return']:.2%}")
            print(f"Sharpe Promedio: {wf_summary['avg_sharpe']:.2f}")
            print(f"Drawdown Promedio: {wf_summary['avg_drawdown']:.2%}")
            print(f"Estabilidad: {wf_summary['stability']:.2%}")
            print(f"Mejor Período: {wf_summary['best_period']}")
            print(f"Peor Período: {wf_summary['worst_period']}")
        
        # Mostrar resumen de purged CV
        if 'purged_cv' in backtest_results:
            cv_summary = backtest_results['purged_cv']['summary']
            print(f"\n=== PURGED CROSS-VALIDATION ===")
            print(f"Folds: {cv_summary['total_folds']}")
            print(f"Total Trades: {cv_summary['total_trades']}")
            print(f"Retorno Promedio: {cv_summary['avg_return']:.2%}")
            print(f"Sharpe Promedio: {cv_summary['avg_sharpe']:.2f}")
            print(f"Drawdown Promedio: {cv_summary['avg_drawdown']:.2%}")
            print(f"Consistencia: {cv_summary['consistency']:.2%}")
            print(f"Mejor Fold: {cv_summary['best_fold']}")
            print(f"Peor Fold: {cv_summary['worst_fold']}")
        
        # Mostrar comparación
        if 'comparison' in backtest_results:
            comparison = backtest_results['comparison']
            print(f"\n=== COMPARACIÓN DE MÉTODOS ===")
            for metric, data in comparison.items():
                if isinstance(data, dict) and 'walk_forward' in data and 'purged_cv' in data:
                    wf_val = data['walk_forward']
                    cv_val = data['purged_cv']
                    diff = data.get('difference', 0.0)
                    print(f"{metric}:")
                    print(f"  Walk-Forward: {wf_val:.2%}")
                    print(f"  Purged CV: {cv_val:.2%}")
                    print(f"  Diferencia: {diff:.2%}")
        
        # Mostrar recomendaciones
        if 'summary' in backtest_results and 'recommendations' in backtest_results['summary']:
            print(f"\n=== RECOMENDACIONES ===")
            for i, rec in enumerate(backtest_results['summary']['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Crear reporte
        report = backtest_manager.create_report()
        print(f"\n=== REPORTE COMPLETO ===")
        print(report)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en sistema de backtesting: {str(e)}")
        return False


def run_risk_management(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el sistema de gestión de riesgo.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando sistema de gestión de riesgo...")
    
    try:
        # Inicializar gestor de riesgo
        risk_manager = RiskManager(config_path)
        
        # Simular posiciones para demo
        from src.risk.base import Position
        demo_positions = [
            Position(
                symbol="SPY",
                quantity=100.0,
                price=400.0,
                timestamp=datetime.now(),
                side="long"
            ),
            Position(
                symbol="QQQ",
                quantity=50.0,
                price=300.0,
                timestamp=datetime.now(),
                side="long"
            ),
            Position(
                symbol="IWM",
                quantity=75.0,
                price=200.0,
                timestamp=datetime.now(),
                side="short"
            )
        ]
        
        # Simular precios actuales
        current_prices = {"SPY": 410.0, "QQQ": 310.0, "IWM": 195.0}
        
        # Simular retornos
        import pandas as pd
        import numpy as np
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        returns = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        
        # Actualizar portafolio
        risk_summary = risk_manager.update_portfolio(demo_positions, current_prices, returns)
        
        # Verificar límites de riesgo
        alerts = risk_manager.check_risk_limits(returns)
        
        # Ajustar tamaño de posición
        position_size = risk_manager.adjust_position_size(
            signal=1.0,
            symbol="AAPL",
            current_price=150.0
        )
        
        # Mostrar resumen
        print("\n=== SISTEMA DE GESTIÓN DE RIESGO ===")
        print(f"Valor del portafolio: ${risk_summary['portfolio_value']:,.2f}")
        print(f"Posiciones: {risk_summary['positions_count']}")
        print(f"Alertas: {len(alerts)}")
        
        if alerts:
            print("\n=== ALERTAS DE RIESGO ===")
            for alert in alerts:
                print(f"- {alert.risk_type.value.upper()}: {alert.message}")
        
        print(f"\n=== AJUSTE DE POSICIÓN ===")
        print(f"Tamaño sugerido para AAPL: {position_size:.2f}")
        
        # Mostrar resumen detallado
        detailed_summary = risk_manager.get_risk_summary()
        
        print(f"\n=== RESUMEN DETALLADO ===")
        print(f"Volatilidad objetivo: {detailed_summary['volatility_targeting']['target_volatility']:.3f}")
        print(f"Concentración del portafolio: {detailed_summary['volatility_targeting']['portfolio_concentration']:.3f}")
        print(f"Leverage del portafolio: {detailed_summary['volatility_targeting']['portfolio_leverage']:.3f}")
        
        # Mostrar estado de circuit breakers
        circuit_breakers = detailed_summary['circuit_breakers']['circuit_breakers']
        print(f"\n=== CIRCUIT BREAKERS ===")
        for name, status in circuit_breakers.items():
            print(f"{name}: {status['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en sistema de gestión de riesgo: {str(e)}")
        return False


def run_ml_models(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el entrenamiento de modelos ML.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando entrenamiento de modelos ML...")
    
    try:
        # Cargar features existentes
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.load_features("data/processed/features")
        
        if not features:
            logger.error("No se encontraron features. Ejecute primero el feature engineering.")
            return False
        
        # Entrenar modelos ML
        ml_trainer = MLModelTrainer(config_path)
        ml_trained_models = ml_trainer.train_models(features, optimize_hyperparams=True)
        
        # Validar modelos ML
        ml_validation_results = ml_trainer.validate_models(features)
        
        # Cross-validation
        ml_cv_results = ml_trainer.cross_validate_models(features, n_splits=3)
        
        # Guardar modelos ML
        ml_trainer.save_models("models/ml_models")
        
        # Generar reporte
        ml_report = ml_trainer.generate_model_report()
        
        # Mostrar resumen
        print("\n=== MODELOS ML ENTRENADOS ===")
        for model_name, model in ml_trained_models.items():
            print(f"{model_name}: {model.performance_metrics}")
        
        print("\n=== REPORTE DE PERFORMANCE ML ===")
        if not ml_report.empty:
            print(ml_report.to_string(index=False))
        else:
            print("No se generó reporte de modelos ML")
        
        # Mostrar mejor modelo ML
        try:
            best_model_name, best_model = ml_trainer.get_best_model('sharpe_ratio')
            print(f"\n=== MEJOR MODELO ML ===")
            print(f"Modelo: {best_model_name}")
            print(f"Sharpe Ratio: {best_model.performance_metrics.get('sharpe_ratio', 'N/A')}")
        except Exception as e:
            logger.warning(f"No se pudo determinar el mejor modelo ML: {str(e)}")
        
        # Mostrar hyperparámetros optimizados
        if ml_trainer.hyperparameter_results:
            print(f"\n=== HYPERPARÁMETROS OPTIMIZADOS ===")
            for model_name, params in ml_trainer.hyperparameter_results.items():
                print(f"{model_name}: {params}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en entrenamiento de modelos ML: {str(e)}")
        return False


def run_baseline_models(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el entrenamiento de modelos baseline.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando entrenamiento de modelos baseline...")
    
    try:
        # Cargar features existentes
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.load_features("data/processed/features")
        
        if not features:
            logger.error("No se encontraron features. Ejecute primero el feature engineering.")
            return False
        
        # Entrenar modelos baseline
        trainer = ModelTrainer(config_path)
        trained_models = trainer.train_models(features)
        
        # Validar modelos
        validation_results = trainer.validate_models(features)
        
        # Cross-validation
        cv_results = trainer.cross_validate_models(features, n_splits=3)
        
        # Guardar modelos
        trainer.save_models("models/baseline_models")
        
        # Generar reporte
        report = trainer.generate_model_report()
        
        # Mostrar resumen
        print("\n=== MODELOS BASELINE ENTRENADOS ===")
        for model_name, model in trained_models.items():
            print(f"{model_name}: {model.performance_metrics}")
        
        print("\n=== REPORTE DE PERFORMANCE ===")
        if not report.empty:
            print(report.to_string(index=False))
        else:
            print("No se generó reporte de modelos")
        
        # Mostrar mejor modelo
        try:
            best_model_name, best_model = trainer.get_best_model('sharpe_ratio')
            print(f"\n=== MEJOR MODELO ===")
            print(f"Modelo: {best_model_name}")
            print(f"Sharpe Ratio: {best_model.performance_metrics.get('sharpe_ratio', 'N/A')}")
        except Exception as e:
            logger.warning(f"No se pudo determinar el mejor modelo: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en entrenamiento de modelos baseline: {str(e)}")
        return False


def run_feature_engineering(config_path: str = "configs/default_parameters.yaml"):
    """
    Ejecuta solo el feature engineering.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    logger.info("Iniciando feature engineering...")
    
    try:
        # Cargar datos procesados
        pipeline = DataPipeline(config_path)
        
        # Cargar datos existentes
        all_symbols = []
        for category, symbols in pipeline.config['instrumentos'].items():
            all_symbols.extend(symbols)
        
        data = {}
        for symbol in all_symbols:
            df = pipeline.load_data(symbol, data_type="processed")
            if df is not None and not df.empty:
                data[symbol] = df
        
        if not data:
            logger.error("No se encontraron datos procesados. Ejecute primero el pipeline de datos.")
            return False
        
        # Crear features
        feature_engineer = FeatureEngineer(config_path)
        features = feature_engineer.create_features(data)
        
        # Guardar features
        feature_engineer.save_features(features, "data/processed/features")
        
        # Mostrar resumen
        print("\n=== FEATURE ENGINEERING COMPLETADO ===")
        for symbol, feature_df in features.items():
            print(f"{symbol}: {len(feature_df.columns)} features, {len(feature_df)} registros")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en feature engineering: {str(e)}")
        return False


def run_demo():
    """
    Ejecuta una demo del sistema con datos de muestra.
    """
    logger.info("Iniciando demo del sistema...")
    
    try:
        # Configurar fechas para demo (últimos 3 meses)
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        # Instrumentos para demo
        demo_symbols = ['SPY', 'QQQ', 'GLD']
        
        # Crear pipeline
        pipeline = DataPipeline()
        
        # Descargar datos de demo
        print(f"Descargando datos de demo desde {start_date} hasta {end_date}")
        raw_data = pipeline.download_data(
            symbols=demo_symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1h'
        )
        
        # Procesar datos
        processed_data = pipeline._process_data(raw_data)
        
        # Guardar datos
        pipeline.save_data(raw_data, data_type="raw")
        pipeline.save_data(processed_data, data_type="processed")
        
        # Crear features
        print("Creando features...")
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_features(processed_data)
        
        # Guardar features
        feature_engineer.save_features(features, "data/processed/features")
        
        # Mostrar resumen
        print(f"\nDemo completada exitosamente!")
        print(f"Total de instrumentos: {len(processed_data)}")
        
        for symbol, df in processed_data.items():
            print(f"{symbol}: {len(df)} registros, {len(df.columns)} columnas")
        
        print(f"\nFeatures creados:")
        for symbol, feature_df in features.items():
            print(f"{symbol}: {len(feature_df.columns)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en demo: {str(e)}")
        return False


def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de Trading Cuantitativo Institucional-Grade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --demo                    # Ejecutar demo con datos de muestra
  python main.py --pipeline               # Ejecutar pipeline completo
  python main.py --features               # Ejecutar solo feature engineering
  python main.py --baseline               # Ejecutar solo modelos baseline
  python main.py --ml                     # Ejecutar solo modelos ML
  python main.py --risk                   # Ejecutar solo gestión de riesgo
  python main.py --backtest               # Ejecutar solo backtesting
  python main.py --validation             # Ejecutar solo validación
  python main.py --explainability         # Ejecutar solo explainability
  python main.py --paper-trading          # Ejecutar solo paper trading
  python main.py --production             # Ejecutar solo sistema de producción
  python main.py --pipeline --config custom_config.yaml  # Con configuración personalizada
        """
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Ejecutar demo del sistema con datos de muestra'
    )
    
    parser.add_argument(
        '--pipeline',
        action='store_true',
        help='Ejecutar pipeline completo de datos'
    )
    
    parser.add_argument(
        '--features',
        action='store_true',
        help='Ejecutar solo feature engineering'
    )
    
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Ejecutar solo entrenamiento de modelos baseline'
    )
    
    parser.add_argument(
        '--ml',
        action='store_true',
        help='Ejecutar solo entrenamiento de modelos ML'
    )
    
    parser.add_argument(
        '--risk',
        action='store_true',
        help='Ejecutar solo sistema de gestión de riesgo'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Ejecutar solo sistema de backtesting'
    )
    
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Ejecutar solo sistema de validación'
    )
    
    parser.add_argument(
        '--explainability',
        action='store_true',
        help='Ejecutar solo sistema de explainability'
    )
    
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        help='Ejecutar solo sistema de paper trading'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Ejecutar solo sistema de producción'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_parameters.yaml',
        help='Ruta al archivo de configuración'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Nivel de logging'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(log_level=args.log_level, log_file="trading.log")
    
    logger.info("=== Sistema de Trading Cuantitativo ===")
    logger.info(f"Configuración: {args.config}")
    logger.info(f"Nivel de logging: {args.log_level}")
    
    success = False
    
    if args.demo:
        success = run_demo()
    elif args.pipeline:
        success = run_data_pipeline(args.config)
    elif args.features:
        success = run_feature_engineering(args.config)
    elif args.baseline:
        success = run_baseline_models(args.config)
    elif args.ml:
        success = run_ml_models(args.config)
    elif args.risk:
        success = run_risk_management(args.config)
    elif args.backtest:
        success = run_backtesting(args.config)
    elif args.validation:
        success = run_validation(args.config)
    elif args.explainability:
        success = run_explainability(args.config)
    elif args.paper_trading:
        success = run_paper_trading(args.config)
    elif args.production:
        success = run_production(args.config)
    else:
        # Por defecto, ejecutar demo
        print("No se especificó acción. Ejecutando demo...")
        success = run_demo()
    
    if success:
        logger.info("Ejecución completada exitosamente")
        sys.exit(0)
    else:
        logger.error("Ejecución falló")
        sys.exit(1)


if __name__ == "__main__":
    main()
