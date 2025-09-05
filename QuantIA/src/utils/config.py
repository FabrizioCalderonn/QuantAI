"""
Módulo de configuración para el sistema de trading.
Maneja la carga y validación de configuraciones.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay error en el parsing
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuración cargada desde: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valida que la configuración tenga todos los campos requeridos.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        True si la configuración es válida
        
    Raises:
        ValueError: Si faltan campos requeridos
    """
    required_fields = [
        'capital_inicial',
        'apalancamiento_max',
        'instrumentos',
        'timeframes',
        'zona_horaria',
        'ventana_historica_anos',
        'costos',
        'vol_objetivo_anualizada',
        'position_sizing_max_pct',
        'max_daily_loss_pct',
        'max_drawdown_pct'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Campos faltantes en configuración: {missing_fields}")
    
    # Validar valores numéricos
    numeric_fields = [
        'capital_inicial',
        'apalancamiento_max',
        'ventana_historica_anos',
        'vol_objetivo_anualizada',
        'position_sizing_max_pct',
        'max_daily_loss_pct',
        'max_drawdown_pct'
    ]
    
    for field in numeric_fields:
        if not isinstance(config[field], (int, float)) or config[field] <= 0:
            raise ValueError(f"Campo {field} debe ser un número positivo")
    
    # Validar instrumentos
    if not isinstance(config['instrumentos'], dict):
        raise ValueError("instrumentos debe ser un diccionario")
    
    # Validar timeframes
    if not isinstance(config['timeframes'], dict):
        raise ValueError("timeframes debe ser un diccionario")
    
    logger.info("Configuración validada exitosamente")
    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Obtiene un valor de configuración usando notación de punto.
    
    Args:
        config: Diccionario de configuración
        key_path: Ruta al valor (ej: 'costos.comision_bps')
        default: Valor por defecto si no se encuentra
        
    Returns:
        Valor de configuración o default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Guarda configuración en archivo YAML.
    
    Args:
        config: Diccionario de configuración
        config_path: Ruta donde guardar
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuración guardada en: {config_path}")
        
    except Exception as e:
        logger.error(f"Error guardando configuración: {e}")
        raise

