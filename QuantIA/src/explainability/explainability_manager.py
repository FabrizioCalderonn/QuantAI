"""
Gestor principal de explainability que coordina todos los componentes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings

from .shap_explainer import SHAPExplainer, SHAPExplanation, ExplanationType, FeatureImportance
from .audit_logger import AuditLogger, EventType, LogLevel, AuditEvent, ModelDecision, TradeDecision
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityManager:
    """
    Gestor principal de explainability que coordina todos los componentes.
    """
    
    def __init__(self, config_path: str = "configs/default_parameters.yaml"):
        """
        Inicializa el gestor de explainability.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.explainability_config = self.config.get('explainability', {})
        
        # Inicializar componentes
        self.shap_explainer = SHAPExplainer(self.explainability_config.get('shap', {}))
        self.audit_logger = AuditLogger(self.explainability_config.get('audit', {}))
        
        # Estado del sistema
        self.explanations = {}
        self.model_explanations = {}
        self.last_explanation = None
        
        logger.info("ExplainabilityManager inicializado")
    
    def explain_model(self, model: Any, X_train: pd.DataFrame, 
                     X_test: pd.DataFrame = None, model_name: str = None) -> Dict[str, Any]:
        """
        Explica un modelo completo.
        
        Args:
            model: Modelo entrenado
            X_train: Datos de entrenamiento
            X_test: Datos de prueba (opcional)
            model_name: Nombre del modelo (opcional)
            
        Returns:
            Diccionario con explicaciones del modelo
        """
        if model_name is None:
            model_name = type(model).__name__
        
        logger.info(f"Explicando modelo: {model_name}")
        
        # Log evento
        self.audit_logger.log_event(
            event_type=EventType.MODEL_TRAINING,
            message=f"Starting model explanation for {model_name}",
            data={'model_name': model_name, 'train_samples': len(X_train)},
            level=LogLevel.INFO
        )
        
        # Ajustar explainer SHAP
        self.shap_explainer.fit_explainer(model, X_train)
        
        # Generar explicaciones
        explanations = {}
        
        # Explicación global
        global_explanation = self.shap_explainer.explain_global(X_test)
        explanations['global'] = global_explanation
        
        # Importancia de features
        feature_importance = self.shap_explainer.get_feature_importance(global_explanation)
        explanations['feature_importance'] = feature_importance
        
        # Interacciones de features
        if X_test is not None:
            interactions = self.shap_explainer.get_feature_interactions(X_test)
            explanations['interactions'] = interactions
        
        # Resumen de explicación
        explanation_summary = self.shap_explainer.get_explanation_summary(global_explanation)
        explanations['summary'] = explanation_summary
        
        # Guardar explicaciones
        self.model_explanations[model_name] = explanations
        
        # Log evento
        self.audit_logger.log_event(
            event_type=EventType.MODEL_TRAINING,
            message=f"Model explanation completed for {model_name}",
            data={
                'model_name': model_name,
                'features_count': len(feature_importance),
                'top_features': [fi.feature_name for fi in feature_importance[:5]]
            },
            level=LogLevel.INFO
        )
        
        return explanations
    
    def explain_prediction(self, model: Any, X_train: pd.DataFrame, 
                          instance: pd.DataFrame, model_name: str = None) -> Dict[str, Any]:
        """
        Explica una predicción específica.
        
        Args:
            model: Modelo entrenado
            X_train: Datos de entrenamiento
            instance: Instancia a explicar
            model_name: Nombre del modelo (opcional)
            
        Returns:
            Diccionario con explicación de la predicción
        """
        if model_name is None:
            model_name = type(model).__name__
        
        logger.info(f"Explicando predicción para modelo: {model_name}")
        
        # Ajustar explainer SHAP si no está ajustado
        if self.shap_explainer.model != model:
            self.shap_explainer.fit_explainer(model, X_train)
        
        # Generar explicación local
        local_explanation = self.shap_explainer.explain_local(instance, 0)
        
        # Obtener predicción
        prediction = model.predict(instance)[0]
        
        # Calcular confianza (simplificado)
        confidence = self._calculate_confidence(local_explanation)
        
        # Obtener importancia de features
        feature_importance = self.shap_explainer.get_feature_importance(local_explanation)
        
        # Crear explicación de predicción
        prediction_explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'explanation': local_explanation,
            'feature_importance': feature_importance,
            'top_features': feature_importance[:10],
            'reasoning': self._generate_reasoning(feature_importance, prediction)
        }
        
        # Log decisión del modelo
        decision_id = self.audit_logger.log_model_decision(
            model_name=model_name,
            input_data=instance.iloc[0].to_dict(),
            prediction=prediction,
            confidence=confidence,
            features_used=list(instance.columns),
            feature_importance={fi.feature_name: fi.importance for fi in feature_importance},
            explanation=prediction_explanation
        )
        
        prediction_explanation['decision_id'] = decision_id
        
        return prediction_explanation
    
    def explain_trade_decision(self, symbol: str, action: str, quantity: float,
                              price: float, model_explanation: Dict[str, Any] = None,
                              risk_assessment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Explica una decisión de trading.
        
        Args:
            symbol: Símbolo del activo
            action: Acción (buy, sell, hold)
            quantity: Cantidad
            price: Precio
            model_explanation: Explicación del modelo (opcional)
            risk_assessment: Evaluación de riesgo (opcional)
            
        Returns:
            Diccionario con explicación de la decisión de trading
        """
        logger.info(f"Explicando decisión de trading: {action} {quantity} {symbol} @ {price}")
        
        # Generar razonamiento
        reasoning = self._generate_trade_reasoning(
            symbol, action, quantity, price, model_explanation, risk_assessment
        )
        
        # Log decisión de trading
        trade_id = self.audit_logger.log_trade_decision(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            reasoning=reasoning,
            model_decision_id=model_explanation.get('decision_id') if model_explanation else None,
            risk_assessment=risk_assessment
        )
        
        # Crear explicación de trading
        trade_explanation = {
            'trade_id': trade_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'reasoning': reasoning,
            'model_explanation': model_explanation,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now().isoformat()
        }
        
        return trade_explanation
    
    def create_explanation_report(self, model_name: str = None) -> str:
        """
        Crea reporte de explicación.
        
        Args:
            model_name: Nombre del modelo (opcional)
            
        Returns:
            Reporte en formato texto
        """
        if model_name and model_name in self.model_explanations:
            return self._create_model_report(model_name)
        else:
            return self._create_comprehensive_report()
    
    def _create_model_report(self, model_name: str) -> str:
        """
        Crea reporte para un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Reporte en formato texto
        """
        explanations = self.model_explanations[model_name]
        
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE EXPLICACIÓN DE MODELO")
        report.append("=" * 80)
        report.append(f"Modelo: {model_name}")
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen
        if 'summary' in explanations:
            summary = explanations['summary']
            report.append("RESUMEN:")
            report.append("-" * 40)
            report.append(f"Features: {summary['features_count']}")
            report.append(f"Muestras: {summary['samples_count']}")
            report.append(f"Valor base: {summary['base_value']:.4f}")
            report.append("")
        
        # Features más importantes
        if 'feature_importance' in explanations:
            feature_importance = explanations['feature_importance']
            report.append("FEATURES MÁS IMPORTANTES:")
            report.append("-" * 40)
            for i, fi in enumerate(feature_importance[:10], 1):
                report.append(f"{i:2d}. {fi.feature_name}: {fi.importance:.4f} ({fi.direction})")
            report.append("")
        
        # Interacciones
        if 'interactions' in explanations:
            interactions = explanations['interactions']
            if interactions:
                report.append("INTERACCIONES DE FEATURES:")
                report.append("-" * 40)
                for interaction, data in list(interactions.items())[:5]:
                    report.append(f"{interaction}: {data['correlation']:.4f} ({data['direction']})")
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
        report.append("REPORTE COMPRENSIVO DE EXPLICABILIDAD")
        report.append("=" * 80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Modelos explicados
        report.append("MODELOS EXPLICADOS:")
        report.append("-" * 40)
        for model_name in self.model_explanations.keys():
            report.append(f"- {model_name}")
        report.append("")
        
        # Resumen de auditoría
        audit_summary = self.audit_logger.get_summary()
        report.append("RESUMEN DE AUDITORÍA:")
        report.append("-" * 40)
        report.append(f"Sesión: {audit_summary['session_id']}")
        report.append(f"Eventos: {audit_summary['events_count']}")
        report.append(f"Base de datos: {'Habilitada' if audit_summary['database_enabled'] else 'Deshabilitada'}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _calculate_confidence(self, explanation: SHAPExplanation) -> float:
        """
        Calcula confianza de la explicación.
        
        Args:
            explanation: Explicación SHAP
            
        Returns:
            Confianza calculada
        """
        # Método simplificado: basado en la consistencia de los valores SHAP
        if len(explanation.values.shape) > 1:
            values = explanation.values[0]
        else:
            values = explanation.values
        
        # Calcular confianza basada en la magnitud de los valores
        confidence = min(1.0, np.sum(np.abs(values)) / len(values))
        
        return confidence
    
    def _generate_reasoning(self, feature_importance: List[FeatureImportance], 
                           prediction: float) -> str:
        """
        Genera razonamiento basado en importancia de features.
        
        Args:
            feature_importance: Lista de importancia de features
            prediction: Predicción del modelo
            
        Returns:
            Razonamiento generado
        """
        reasoning_parts = []
        
        # Agregar predicción
        if prediction > 0.5:
            reasoning_parts.append("El modelo predice una señal positiva")
        else:
            reasoning_parts.append("El modelo predice una señal negativa")
        
        # Agregar features más importantes
        top_features = feature_importance[:3]
        if top_features:
            feature_names = [fi.feature_name for fi in top_features]
            reasoning_parts.append(f"Basado principalmente en: {', '.join(feature_names)}")
        
        # Agregar dirección de features
        positive_features = [fi for fi in top_features if fi.direction == "positive"]
        negative_features = [fi for fi in top_features if fi.direction == "negative"]
        
        if positive_features:
            reasoning_parts.append(f"Features positivos: {', '.join([fi.feature_name for fi in positive_features])}")
        
        if negative_features:
            reasoning_parts.append(f"Features negativos: {', '.join([fi.feature_name for fi in negative_features])}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_trade_reasoning(self, symbol: str, action: str, quantity: float,
                                 price: float, model_explanation: Dict[str, Any] = None,
                                 risk_assessment: Dict[str, Any] = None) -> str:
        """
        Genera razonamiento para decisión de trading.
        
        Args:
            symbol: Símbolo del activo
            action: Acción (buy, sell, hold)
            quantity: Cantidad
            price: Precio
            model_explanation: Explicación del modelo (opcional)
            risk_assessment: Evaluación de riesgo (opcional)
            
        Returns:
            Razonamiento generado
        """
        reasoning_parts = []
        
        # Agregar acción
        reasoning_parts.append(f"Decisión: {action.upper()} {quantity} {symbol} @ {price}")
        
        # Agregar explicación del modelo
        if model_explanation:
            model_reasoning = model_explanation.get('reasoning', '')
            if model_reasoning:
                reasoning_parts.append(f"Modelo: {model_reasoning}")
            
            confidence = model_explanation.get('confidence', 0)
            reasoning_parts.append(f"Confianza del modelo: {confidence:.2%}")
        
        # Agregar evaluación de riesgo
        if risk_assessment:
            risk_level = risk_assessment.get('risk_level', 'unknown')
            reasoning_parts.append(f"Nivel de riesgo: {risk_level}")
        
        return ". ".join(reasoning_parts) + "."
    
    def export_explanations(self, filepath: str, format: str = 'json') -> None:
        """
        Exporta explicaciones.
        
        Args:
            filepath: Ruta del archivo
            format: Formato de exportación ('json', 'pickle')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._export_to_json(filepath)
        elif format == 'pickle':
            self._export_to_pickle(filepath)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Explicaciones exportadas: {filepath}")
    
    def _export_to_json(self, filepath: Path) -> None:
        """
        Exporta explicaciones a JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        import json
        
        # Convertir explicaciones a formato serializable
        serializable_explanations = {}
        for model_name, explanations in self.model_explanations.items():
            serializable_explanations[model_name] = {}
            
            for key, value in explanations.items():
                if key == 'feature_importance':
                    serializable_explanations[model_name][key] = [
                        {
                            'feature_name': fi.feature_name,
                            'importance': fi.importance,
                            'rank': fi.rank,
                            'contribution': fi.contribution,
                            'direction': fi.direction
                        }
                        for fi in value
                    ]
                elif key == 'summary':
                    serializable_explanations[model_name][key] = value
                elif key == 'interactions':
                    serializable_explanations[model_name][key] = value
                else:
                    # Para SHAPExplanation, convertir a dict
                    if hasattr(value, '__dict__'):
                        serializable_explanations[model_name][key] = {
                            'explanation_type': value.explanation_type.value,
                            'values': value.values.tolist(),
                            'base_values': value.base_values,
                            'data': value.data.tolist(),
                            'feature_names': value.feature_names,
                            'timestamp': value.timestamp.isoformat(),
                            'model_name': value.model_name,
                            'metadata': value.metadata
                        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_explanations, f, indent=2, default=str)
    
    def _export_to_pickle(self, filepath: Path) -> None:
        """
        Exporta explicaciones a pickle.
        
        Args:
            filepath: Ruta del archivo
        """
        joblib.dump(self.model_explanations, filepath)
    
    def get_audit_report(self, start_date: datetime = None, 
                        end_date: datetime = None) -> Dict[str, Any]:
        """
        Obtiene reporte de auditoría.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Reporte de auditoría
        """
        return self.audit_logger.create_audit_report(start_date, end_date)
    
    def get_model_explanations(self, model_name: str = None) -> Dict[str, Any]:
        """
        Obtiene explicaciones de modelos.
        
        Args:
            model_name: Nombre del modelo (opcional)
            
        Returns:
            Explicaciones de modelos
        """
        if model_name:
            return self.model_explanations.get(model_name, {})
        return self.model_explanations.copy()
    
    def cleanup_old_logs(self, days: int = None) -> int:
        """
        Limpia logs antiguos.
        
        Args:
            days: Días de retención (opcional)
            
        Returns:
            Número de registros eliminados
        """
        return self.audit_logger.cleanup_old_logs(days)
    
    def close(self) -> None:
        """Cierra el gestor de explainability."""
        self.audit_logger.close()
        logger.info("ExplainabilityManager cerrado")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del gestor de explainability.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'config': self.config,
            'model_explanations_count': len(self.model_explanations),
            'models_explained': list(self.model_explanations.keys()),
            'audit_summary': self.audit_logger.get_summary(),
            'shap_summary': self.shap_explainer.get_summary()
        }

