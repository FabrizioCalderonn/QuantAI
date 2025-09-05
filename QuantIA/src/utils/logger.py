"""
Módulo de logging para el sistema de trading.
Configura logging estructurado y rotación de archivos.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Archivo de log (opcional)
        log_dir: Directorio para logs
    """
    # Crear directorio de logs
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Limpiar handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Handler para archivo (si se especifica)
    if log_file:
        file_path = log_path / log_file
        
        # Rotating file handler (10MB, 5 archivos)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configurar loggers específicos
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger con el nombre especificado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


class TradingLogger:
    """
    Logger especializado para operaciones de trading.
    """
    
    def __init__(self, name: str = "trading"):
        self.logger = get_logger(name)
        self.trade_logger = get_logger(f"{name}.trades")
        self.risk_logger = get_logger(f"{name}.risk")
        self.performance_logger = get_logger(f"{name}.performance")
    
    def log_trade(self, trade_data: dict):
        """
        Log de operación de trading.
        
        Args:
            trade_data: Diccionario con datos de la operación
        """
        self.trade_logger.info(f"TRADE: {trade_data}")
    
    def log_risk_event(self, event_type: str, details: dict):
        """
        Log de evento de riesgo.
        
        Args:
            event_type: Tipo de evento
            details: Detalles del evento
        """
        self.risk_logger.warning(f"RISK_EVENT: {event_type} - {details}")
    
    def log_performance(self, metrics: dict):
        """
        Log de métricas de performance.
        
        Args:
            metrics: Diccionario con métricas
        """
        self.performance_logger.info(f"PERFORMANCE: {metrics}")
    
    def log_signal(self, signal_data: dict):
        """
        Log de señal generada.
        
        Args:
            signal_data: Datos de la señal
        """
        self.logger.info(f"SIGNAL: {signal_data}")
    
    def log_error(self, error: Exception, context: str = ""):
        """
        Log de error.
        
        Args:
            error: Excepción
            context: Contexto adicional
        """
        self.logger.error(f"ERROR: {context} - {str(error)}", exc_info=True)


# Configurar logging por defecto
setup_logging()

