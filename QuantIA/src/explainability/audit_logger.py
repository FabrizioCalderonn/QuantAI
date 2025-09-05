"""
Sistema de auditoría y logging para transparencia total.
Registra todas las decisiones y operaciones del sistema.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Niveles de log."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Tipos de eventos."""
    DATA_LOAD = "data_load"
    FEATURE_CREATION = "feature_creation"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    BACKTESTING = "backtesting"
    VALIDATION = "validation"
    RISK_MANAGEMENT = "risk_management"
    TRADE_EXECUTION = "trade_execution"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Evento de auditoría."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    level: LogLevel
    message: str
    data: Dict[str, Any]
    user_id: str = "system"
    session_id: str = "default"
    metadata: Dict[str, Any] = None


@dataclass
class ModelDecision:
    """Decisión del modelo."""
    decision_id: str
    model_name: str
    timestamp: datetime
    input_data: Dict[str, Any]
    prediction: float
    confidence: float
    features_used: List[str]
    feature_importance: Dict[str, float]
    explanation: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class TradeDecision:
    """Decisión de trading."""
    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # "buy", "sell", "hold"
    quantity: float
    price: float
    reasoning: str
    model_decision: Optional[ModelDecision] = None
    risk_assessment: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class AuditLogger:
    """
    Sistema de auditoría y logging para transparencia total.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el audit logger.
        
        Args:
            config: Configuración del logger
        """
        self.config = config or {}
        
        # Configuración de base de datos
        self.db_path = self.config.get('db_path', 'audit_logs.db')
        self.retention_days = self.config.get('retention_days', 365)
        self.max_log_size = self.config.get('max_log_size', 1000000)  # 1MB
        
        # Configuración de logging
        self.log_level = LogLevel(self.config.get('log_level', 'INFO'))
        self.enable_console = self.config.get('enable_console', True)
        self.enable_file = self.config.get('enable_file', True)
        self.enable_database = self.config.get('enable_database', True)
        
        # Inicializar componentes
        self._setup_logging()
        self._setup_database()
        
        # Estado
        self.session_id = self._generate_session_id()
        self.events_count = 0
        
        logger.info("AuditLogger inicializado")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging."""
        # Crear logger
        self.logger = logging.getLogger('audit_logger')
        self.logger.setLevel(getattr(logging, self.log_level.value))
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Formato de log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler de consola
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Handler de archivo
        if self.enable_file:
            log_file = self.config.get('log_file', 'audit.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_database(self) -> None:
        """Configura la base de datos de auditoría."""
        if not self.enable_database:
            return
        
        # Crear directorio si no existe
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Conectar a base de datos
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('PRAGMA journal_mode=WAL')
        
        # Crear tablas
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Crea tablas de base de datos."""
        # Tabla de eventos de auditoría
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Tabla de decisiones de modelo
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS model_decisions (
                decision_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction REAL NOT NULL,
                confidence REAL NOT NULL,
                features_used TEXT NOT NULL,
                feature_importance TEXT NOT NULL,
                explanation TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Tabla de decisiones de trading
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trade_decisions (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                reasoning TEXT NOT NULL,
                model_decision_id TEXT,
                risk_assessment TEXT,
                metadata TEXT,
                FOREIGN KEY (model_decision_id) REFERENCES model_decisions (decision_id)
            )
        ''')
        
        # Índices para performance
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events (timestamp)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events (event_type)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_model_decisions_timestamp ON model_decisions (timestamp)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_decisions_timestamp ON trade_decisions (timestamp)')
        
        self.conn.commit()
    
    def _generate_session_id(self) -> str:
        """Genera ID de sesión único."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def _generate_event_id(self) -> str:
        """Genera ID de evento único."""
        self.events_count += 1
        return f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.events_count:06d}"
    
    def log_event(self, event_type: EventType, message: str, 
                  data: Dict[str, Any] = None, level: LogLevel = LogLevel.INFO,
                  user_id: str = "system", metadata: Dict[str, Any] = None) -> str:
        """
        Registra un evento de auditoría.
        
        Args:
            event_type: Tipo de evento
            message: Mensaje del evento
            data: Datos del evento
            level: Nivel de log
            user_id: ID del usuario
            metadata: Metadatos adicionales
            
        Returns:
            ID del evento registrado
        """
        event_id = self._generate_event_id()
        
        # Crear evento
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            level=level,
            message=message,
            data=data or {},
            user_id=user_id,
            session_id=self.session_id,
            metadata=metadata or {}
        )
        
        # Log a consola/archivo
        log_message = f"[{event_type.value}] {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
        
        # Guardar en base de datos
        if self.enable_database:
            self._save_event_to_db(event)
        
        return event_id
    
    def log_model_decision(self, model_name: str, input_data: Dict[str, Any],
                          prediction: float, confidence: float,
                          features_used: List[str], feature_importance: Dict[str, float],
                          explanation: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Registra una decisión de modelo.
        
        Args:
            model_name: Nombre del modelo
            input_data: Datos de entrada
            prediction: Predicción del modelo
            confidence: Confianza de la predicción
            features_used: Features utilizadas
            feature_importance: Importancia de features
            explanation: Explicación de la decisión
            metadata: Metadatos adicionales
            
        Returns:
            ID de la decisión registrada
        """
        decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # Crear decisión
        decision = ModelDecision(
            decision_id=decision_id,
            model_name=model_name,
            timestamp=datetime.now(),
            input_data=input_data,
            prediction=prediction,
            confidence=confidence,
            features_used=features_used,
            feature_importance=feature_importance,
            explanation=explanation or {},
            metadata=metadata or {}
        )
        
        # Log evento
        self.log_event(
            event_type=EventType.MODEL_PREDICTION,
            message=f"Model decision: {model_name} predicted {prediction:.4f} with confidence {confidence:.4f}",
            data={
                'decision_id': decision_id,
                'model_name': model_name,
                'prediction': prediction,
                'confidence': confidence,
                'features_count': len(features_used)
            },
            level=LogLevel.INFO
        )
        
        # Guardar en base de datos
        if self.enable_database:
            self._save_model_decision_to_db(decision)
        
        return decision_id
    
    def log_trade_decision(self, symbol: str, action: str, quantity: float,
                          price: float, reasoning: str,
                          model_decision_id: str = None,
                          risk_assessment: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Registra una decisión de trading.
        
        Args:
            symbol: Símbolo del activo
            action: Acción (buy, sell, hold)
            quantity: Cantidad
            price: Precio
            reasoning: Razonamiento de la decisión
            model_decision_id: ID de decisión del modelo
            risk_assessment: Evaluación de riesgo
            metadata: Metadatos adicionales
            
        Returns:
            ID de la decisión de trading registrada
        """
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # Crear decisión de trading
        trade_decision = TradeDecision(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            reasoning=reasoning,
            model_decision_id=model_decision_id,
            risk_assessment=risk_assessment or {},
            metadata=metadata or {}
        )
        
        # Log evento
        self.log_event(
            event_type=EventType.TRADE_EXECUTION,
            message=f"Trade decision: {action.upper()} {quantity} {symbol} @ {price}",
            data={
                'trade_id': trade_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'model_decision_id': model_decision_id
            },
            level=LogLevel.INFO
        )
        
        # Guardar en base de datos
        if self.enable_database:
            self._save_trade_decision_to_db(trade_decision)
        
        return trade_id
    
    def _save_event_to_db(self, event: AuditEvent) -> None:
        """Guarda evento en base de datos."""
        try:
            self.conn.execute('''
                INSERT INTO audit_events 
                (event_id, event_type, timestamp, level, message, data, user_id, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.level.value,
                event.message,
                json.dumps(event.data, default=str),
                event.user_id,
                event.session_id,
                json.dumps(event.metadata, default=str) if event.metadata else None
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error guardando evento en DB: {str(e)}")
    
    def _save_model_decision_to_db(self, decision: ModelDecision) -> None:
        """Guarda decisión de modelo en base de datos."""
        try:
            self.conn.execute('''
                INSERT INTO model_decisions 
                (decision_id, model_name, timestamp, input_data, prediction, confidence, 
                 features_used, feature_importance, explanation, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision.decision_id,
                decision.model_name,
                decision.timestamp.isoformat(),
                json.dumps(decision.input_data, default=str),
                decision.prediction,
                decision.confidence,
                json.dumps(decision.features_used),
                json.dumps(decision.feature_importance),
                json.dumps(decision.explanation, default=str),
                json.dumps(decision.metadata, default=str) if decision.metadata else None
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error guardando decisión de modelo en DB: {str(e)}")
    
    def _save_trade_decision_to_db(self, trade_decision: TradeDecision) -> None:
        """Guarda decisión de trading en base de datos."""
        try:
            self.conn.execute('''
                INSERT INTO trade_decisions 
                (trade_id, timestamp, symbol, action, quantity, price, reasoning, 
                 model_decision_id, risk_assessment, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_decision.trade_id,
                trade_decision.timestamp.isoformat(),
                trade_decision.symbol,
                trade_decision.action,
                trade_decision.quantity,
                trade_decision.price,
                trade_decision.reasoning,
                trade_decision.model_decision_id,
                json.dumps(trade_decision.risk_assessment, default=str),
                json.dumps(trade_decision.metadata, default=str) if trade_decision.metadata else None
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error guardando decisión de trading en DB: {str(e)}")
    
    def get_events(self, event_type: EventType = None, 
                   start_date: datetime = None, end_date: datetime = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Obtiene eventos de auditoría.
        
        Args:
            event_type: Tipo de evento (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        if not self.enable_database:
            return []
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                event = {
                    'event_id': row[0],
                    'event_type': row[1],
                    'timestamp': row[2],
                    'level': row[3],
                    'message': row[4],
                    'data': json.loads(row[5]),
                    'user_id': row[6],
                    'session_id': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {}
                }
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error obteniendo eventos: {str(e)}")
            return []
    
    def get_model_decisions(self, model_name: str = None,
                           start_date: datetime = None, end_date: datetime = None,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Obtiene decisiones de modelo.
        
        Args:
            model_name: Nombre del modelo (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de decisiones de modelo
        """
        if not self.enable_database:
            return []
        
        query = "SELECT * FROM model_decisions WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            decisions = []
            for row in rows:
                decision = {
                    'decision_id': row[0],
                    'model_name': row[1],
                    'timestamp': row[2],
                    'input_data': json.loads(row[3]),
                    'prediction': row[4],
                    'confidence': row[5],
                    'features_used': json.loads(row[6]),
                    'feature_importance': json.loads(row[7]),
                    'explanation': json.loads(row[8]),
                    'metadata': json.loads(row[9]) if row[9] else {}
                }
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error obteniendo decisiones de modelo: {str(e)}")
            return []
    
    def get_trade_decisions(self, symbol: str = None,
                           start_date: datetime = None, end_date: datetime = None,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Obtiene decisiones de trading.
        
        Args:
            symbol: Símbolo (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de decisiones de trading
        """
        if not self.enable_database:
            return []
        
        query = "SELECT * FROM trade_decisions WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            decisions = []
            for row in rows:
                decision = {
                    'trade_id': row[0],
                    'timestamp': row[1],
                    'symbol': row[2],
                    'action': row[3],
                    'quantity': row[4],
                    'price': row[5],
                    'reasoning': row[6],
                    'model_decision_id': row[7],
                    'risk_assessment': json.loads(row[8]) if row[8] else {},
                    'metadata': json.loads(row[9]) if row[9] else {}
                }
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error obteniendo decisiones de trading: {str(e)}")
            return []
    
    def create_audit_report(self, start_date: datetime = None, 
                           end_date: datetime = None) -> Dict[str, Any]:
        """
        Crea reporte de auditoría.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Reporte de auditoría
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Obtener eventos
        events = self.get_events(start_date=start_date, end_date=end_date)
        
        # Obtener decisiones de modelo
        model_decisions = self.get_model_decisions(start_date=start_date, end_date=end_date)
        
        # Obtener decisiones de trading
        trade_decisions = self.get_trade_decisions(start_date=start_date, end_date=end_date)
        
        # Estadísticas
        event_types = {}
        for event in events:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        model_names = {}
        for decision in model_decisions:
            model_name = decision['model_name']
            model_names[model_name] = model_names.get(model_name, 0) + 1
        
        symbols = {}
        for decision in trade_decisions:
            symbol = decision['symbol']
            symbols[symbol] = symbols.get(symbol, 0) + 1
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'total_model_decisions': len(model_decisions),
                'total_trade_decisions': len(trade_decisions)
            },
            'event_types': event_types,
            'model_usage': model_names,
            'symbols_traded': symbols,
            'session_id': self.session_id,
            'generated_at': datetime.now().isoformat()
        }
    
    def cleanup_old_logs(self, days: int = None) -> int:
        """
        Limpia logs antiguos.
        
        Args:
            days: Días de retención (opcional)
            
        Returns:
            Número de registros eliminados
        """
        if not self.enable_database:
            return 0
        
        if days is None:
            days = self.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        try:
            # Eliminar eventos antiguos
            cursor = self.conn.execute('DELETE FROM audit_events WHERE timestamp < ?', (cutoff_str,))
            events_deleted = cursor.rowcount
            
            # Eliminar decisiones de modelo antiguas
            cursor = self.conn.execute('DELETE FROM model_decisions WHERE timestamp < ?', (cutoff_str,))
            model_decisions_deleted = cursor.rowcount
            
            # Eliminar decisiones de trading antiguas
            cursor = self.conn.execute('DELETE FROM trade_decisions WHERE timestamp < ?', (cutoff_str,))
            trade_decisions_deleted = cursor.rowcount
            
            self.conn.commit()
            
            total_deleted = events_deleted + model_decisions_deleted + trade_decisions_deleted
            
            logger.info(f"Limpieza completada: {total_deleted} registros eliminados")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error en limpieza de logs: {str(e)}")
            return 0
    
    def close(self) -> None:
        """Cierra el audit logger."""
        if self.enable_database and hasattr(self, 'conn'):
            self.conn.close()
        
        logger.info("AuditLogger cerrado")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del audit logger.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'session_id': self.session_id,
            'events_count': self.events_count,
            'config': self.config,
            'database_enabled': self.enable_database,
            'database_path': self.db_path if self.enable_database else None
        }

