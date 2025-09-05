"""
Explainer SHAP para modelos de trading.
Proporciona explicaciones detalladas de las decisiones del modelo.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP no está disponible. Instalar con: pip install shap")

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Tipos de explicación."""
    GLOBAL = "global"
    LOCAL = "local"
    FEATURE_IMPORTANCE = "feature_importance"
    INTERACTION = "interaction"
    SUMMARY = "summary"


@dataclass
class SHAPExplanation:
    """Explicación SHAP."""
    explanation_type: ExplanationType
    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    feature_names: List[str]
    timestamp: datetime
    model_name: str
    metadata: Dict[str, Any] = None


@dataclass
class FeatureImportance:
    """Importancia de features."""
    feature_name: str
    importance: float
    rank: int
    contribution: float
    direction: str  # "positive", "negative", "neutral"


class SHAPExplainer:
    """
    Explainer SHAP para modelos de trading.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el explainer SHAP.
        
        Args:
            config: Configuración del explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP no está disponible. Instalar con: pip install shap")
        
        self.config = config or {}
        self.explainer = None
        self.model = None
        self.feature_names = []
        self.explanations = {}
        
        # Configuración por defecto
        self.sample_size = self.config.get('sample_size', 1000)
        self.max_features = self.config.get('max_features', 50)
        self.random_state = self.config.get('random_state', 42)
        
        logger.info("SHAPExplainer inicializado")
    
    def fit_explainer(self, model: Any, X_train: pd.DataFrame, 
                     model_type: str = "auto") -> None:
        """
        Ajusta el explainer SHAP al modelo.
        
        Args:
            model: Modelo entrenado
            X_train: Datos de entrenamiento
            model_type: Tipo de modelo ("tree", "linear", "auto")
        """
        logger.info(f"Ajustando explainer SHAP para modelo tipo: {model_type}")
        
        self.model = model
        self.feature_names = list(X_train.columns)
        
        # Determinar tipo de explainer
        if model_type == "auto":
            model_type = self._detect_model_type(model)
        
        # Crear explainer apropiado
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_train)
        else:
            # Usar KernelExplainer como fallback
            sample_data = self._sample_data(X_train)
            self.explainer = shap.KernelExplainer(model.predict, sample_data)
        
        logger.info(f"Explainer SHAP ajustado: {type(self.explainer).__name__}")
    
    def _detect_model_type(self, model: Any) -> str:
        """
        Detecta el tipo de modelo automáticamente.
        
        Args:
            model: Modelo a analizar
            
        Returns:
            Tipo de modelo detectado
        """
        model_name = type(model).__name__.lower()
        
        if any(tree_model in model_name for tree_model in ['tree', 'forest', 'gradient', 'xgboost', 'lightgbm']):
            return "tree"
        elif any(linear_model in model_name for linear_model in ['linear', 'lasso', 'ridge', 'elastic']):
            return "linear"
        else:
            return "kernel"
    
    def _sample_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Muestra datos para KernelExplainer.
        
        Args:
            X: Datos completos
            
        Returns:
            Muestra de datos
        """
        if len(X) > self.sample_size:
            return X.sample(n=self.sample_size, random_state=self.random_state).values
        return X.values
    
    def explain_global(self, X: pd.DataFrame = None) -> SHAPExplanation:
        """
        Genera explicación global del modelo.
        
        Args:
            X: Datos para explicar (opcional)
            
        Returns:
            Explicación global
        """
        if self.explainer is None:
            raise ValueError("Explainer no ha sido ajustado. Llamar fit_explainer primero.")
        
        logger.info("Generando explicación global")
        
        # Usar datos de muestra si no se proporcionan
        if X is None:
            X = self._get_sample_data()
        
        # Calcular valores SHAP
        shap_values = self.explainer.shap_values(X)
        
        # Manejar diferentes tipos de salida
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Usar primera clase para clasificación
        
        # Calcular valores base
        base_values = self.explainer.expected_value
        if isinstance(base_values, list):
            base_values = base_values[0]
        
        explanation = SHAPExplanation(
            explanation_type=ExplanationType.GLOBAL,
            values=shap_values,
            base_values=base_values,
            data=X.values,
            feature_names=self.feature_names,
            timestamp=datetime.now(),
            model_name=type(self.model).__name__,
            metadata={
                'sample_size': len(X),
                'explainer_type': type(self.explainer).__name__
            }
        )
        
        self.explanations['global'] = explanation
        
        return explanation
    
    def explain_local(self, X: pd.DataFrame, instance_idx: int = 0) -> SHAPExplanation:
        """
        Genera explicación local para una instancia específica.
        
        Args:
            X: Datos para explicar
            instance_idx: Índice de la instancia
            
        Returns:
            Explicación local
        """
        if self.explainer is None:
            raise ValueError("Explainer no ha sido ajustado. Llamar fit_explainer primero.")
        
        logger.info(f"Generando explicación local para instancia {instance_idx}")
        
        # Obtener instancia específica
        instance = X.iloc[[instance_idx]]
        
        # Calcular valores SHAP
        shap_values = self.explainer.shap_values(instance)
        
        # Manejar diferentes tipos de salida
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Calcular valores base
        base_values = self.explainer.expected_value
        if isinstance(base_values, list):
            base_values = base_values[0]
        
        explanation = SHAPExplanation(
            explanation_type=ExplanationType.LOCAL,
            values=shap_values,
            base_values=base_values,
            data=instance.values,
            feature_names=self.feature_names,
            timestamp=datetime.now(),
            model_name=type(self.model).__name__,
            metadata={
                'instance_idx': instance_idx,
                'explainer_type': type(self.explainer).__name__
            }
        )
        
        self.explanations[f'local_{instance_idx}'] = explanation
        
        return explanation
    
    def get_feature_importance(self, explanation: SHAPExplanation = None) -> List[FeatureImportance]:
        """
        Obtiene importancia de features desde explicación SHAP.
        
        Args:
            explanation: Explicación SHAP (opcional)
            
        Returns:
            Lista de importancia de features
        """
        if explanation is None:
            if 'global' not in self.explanations:
                explanation = self.explain_global()
            else:
                explanation = self.explanations['global']
        
        # Calcular importancia promedio
        if len(explanation.values.shape) > 1:
            importance_scores = np.mean(np.abs(explanation.values), axis=0)
        else:
            importance_scores = np.abs(explanation.values)
        
        # Crear lista de importancia
        feature_importance = []
        for i, (feature_name, importance) in enumerate(zip(explanation.feature_names, importance_scores)):
            # Determinar dirección
            if len(explanation.values.shape) > 1:
                avg_contribution = np.mean(explanation.values[:, i])
            else:
                avg_contribution = explanation.values[i]
            
            if avg_contribution > 0.01:
                direction = "positive"
            elif avg_contribution < -0.01:
                direction = "negative"
            else:
                direction = "neutral"
            
            feature_importance.append(FeatureImportance(
                feature_name=feature_name,
                importance=importance,
                rank=0,  # Se asignará después
                contribution=avg_contribution,
                direction=direction
            ))
        
        # Ordenar por importancia y asignar ranks
        feature_importance.sort(key=lambda x: x.importance, reverse=True)
        for i, fi in enumerate(feature_importance):
            fi.rank = i + 1
        
        return feature_importance
    
    def get_feature_interactions(self, X: pd.DataFrame, 
                               top_features: int = 10) -> Dict[str, Any]:
        """
        Obtiene interacciones entre features.
        
        Args:
            X: Datos para analizar
            top_features: Número de features top para analizar
            
        Returns:
            Diccionario con interacciones
        """
        if self.explainer is None:
            raise ValueError("Explainer no ha sido ajustado. Llamar fit_explainer primero.")
        
        logger.info(f"Calculando interacciones para top {top_features} features")
        
        # Obtener explicación global
        explanation = self.explain_global(X)
        
        # Obtener features más importantes
        feature_importance = self.get_feature_importance(explanation)
        top_feature_names = [fi.feature_name for fi in feature_importance[:top_features]]
        top_feature_indices = [explanation.feature_names.index(name) for name in top_feature_names]
        
        # Calcular interacciones usando SHAP
        try:
            # Usar sample más pequeño para interacciones
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
            
            # Calcular valores SHAP
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calcular interacciones
            interactions = {}
            for i, feature1 in enumerate(top_feature_names):
                for j, feature2 in enumerate(top_feature_names):
                    if i < j:  # Evitar duplicados
                        idx1 = top_feature_indices[i]
                        idx2 = top_feature_indices[j]
                        
                        # Calcular correlación entre valores SHAP
                        corr = np.corrcoef(shap_values[:, idx1], shap_values[:, idx2])[0, 1]
                        
                        interactions[f"{feature1}_x_{feature2}"] = {
                            'correlation': corr,
                            'strength': abs(corr),
                            'direction': 'positive' if corr > 0 else 'negative'
                        }
            
            return interactions
            
        except Exception as e:
            logger.warning(f"Error calculando interacciones: {str(e)}")
            return {}
    
    def create_summary_plot(self, explanation: SHAPExplanation = None, 
                          save_path: str = None) -> Dict[str, Any]:
        """
        Crea plot de resumen SHAP.
        
        Args:
            explanation: Explicación SHAP (opcional)
            save_path: Ruta para guardar plot (opcional)
            
        Returns:
            Diccionario con información del plot
        """
        if explanation is None:
            if 'global' not in self.explanations:
                explanation = self.explain_global()
            else:
                explanation = self.explanations['global']
        
        try:
            # Crear plot de resumen
            shap.summary_plot(explanation.values, explanation.data, 
                            feature_names=explanation.feature_names, show=False)
            
            plot_info = {
                'type': 'summary_plot',
                'features_count': len(explanation.feature_names),
                'samples_count': len(explanation.data),
                'created_at': datetime.now().isoformat()
            }
            
            if save_path:
                import matplotlib.pyplot as plt
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_info['saved_to'] = save_path
            
            return plot_info
            
        except Exception as e:
            logger.error(f"Error creando plot de resumen: {str(e)}")
            return {'error': str(e)}
    
    def create_waterfall_plot(self, explanation: SHAPExplanation, 
                            save_path: str = None) -> Dict[str, Any]:
        """
        Crea plot de cascada SHAP.
        
        Args:
            explanation: Explicación SHAP local
            save_path: Ruta para guardar plot (opcional)
            
        Returns:
            Diccionario con información del plot
        """
        if explanation.explanation_type != ExplanationType.LOCAL:
            raise ValueError("Waterfall plot requiere explicación local")
        
        try:
            # Crear plot de cascada
            shap.waterfall_plot(explanation.base_values, explanation.values[0], 
                              explanation.data[0], feature_names=explanation.feature_names, show=False)
            
            plot_info = {
                'type': 'waterfall_plot',
                'features_count': len(explanation.feature_names),
                'base_value': explanation.base_values,
                'prediction': explanation.base_values + np.sum(explanation.values[0]),
                'created_at': datetime.now().isoformat()
            }
            
            if save_path:
                import matplotlib.pyplot as plt
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_info['saved_to'] = save_path
            
            return plot_info
            
        except Exception as e:
            logger.error(f"Error creando plot de cascada: {str(e)}")
            return {'error': str(e)}
    
    def create_force_plot(self, explanation: SHAPExplanation, 
                         save_path: str = None) -> Dict[str, Any]:
        """
        Crea plot de fuerza SHAP.
        
        Args:
            explanation: Explicación SHAP local
            save_path: Ruta para guardar plot (opcional)
            
        Returns:
            Diccionario con información del plot
        """
        if explanation.explanation_type != ExplanationType.LOCAL:
            raise ValueError("Force plot requiere explicación local")
        
        try:
            # Crear plot de fuerza
            force_plot = shap.force_plot(explanation.base_values, explanation.values[0], 
                                       explanation.data[0], feature_names=explanation.feature_names)
            
            plot_info = {
                'type': 'force_plot',
                'features_count': len(explanation.feature_names),
                'base_value': explanation.base_values,
                'prediction': explanation.base_values + np.sum(explanation.values[0]),
                'created_at': datetime.now().isoformat()
            }
            
            if save_path:
                shap.save_html(save_path, force_plot)
                plot_info['saved_to'] = save_path
            
            return plot_info
            
        except Exception as e:
            logger.error(f"Error creando plot de fuerza: {str(e)}")
            return {'error': str(e)}
    
    def _get_sample_data(self) -> pd.DataFrame:
        """
        Obtiene datos de muestra para explicación global.
        
        Returns:
            DataFrame con datos de muestra
        """
        # Esto debería ser implementado por el usuario o pasado como parámetro
        raise NotImplementedError("Implementar _get_sample_data o pasar X a explain_global")
    
    def get_explanation_summary(self, explanation: SHAPExplanation = None) -> Dict[str, Any]:
        """
        Obtiene resumen de explicación SHAP.
        
        Args:
            explanation: Explicación SHAP (opcional)
            
        Returns:
            Diccionario con resumen
        """
        if explanation is None:
            if 'global' not in self.explanations:
                explanation = self.explain_global()
            else:
                explanation = self.explanations['global']
        
        # Calcular estadísticas
        if len(explanation.values.shape) > 1:
            mean_abs_values = np.mean(np.abs(explanation.values), axis=0)
            std_values = np.std(explanation.values, axis=0)
        else:
            mean_abs_values = np.abs(explanation.values)
            std_values = np.zeros_like(explanation.values)
        
        # Obtener features más importantes
        feature_importance = self.get_feature_importance(explanation)
        top_features = feature_importance[:10]
        
        return {
            'explanation_type': explanation.explanation_type.value,
            'model_name': explanation.model_name,
            'features_count': len(explanation.feature_names),
            'samples_count': len(explanation.data),
            'base_value': explanation.base_values,
            'top_features': [
                {
                    'name': fi.feature_name,
                    'importance': fi.importance,
                    'rank': fi.rank,
                    'contribution': fi.contribution,
                    'direction': fi.direction
                }
                for fi in top_features
            ],
            'statistics': {
                'mean_abs_importance': np.mean(mean_abs_values),
                'std_importance': np.mean(std_values),
                'max_importance': np.max(mean_abs_values),
                'min_importance': np.min(mean_abs_values)
            },
            'timestamp': explanation.timestamp.isoformat(),
            'metadata': explanation.metadata
        }
    
    def save_explanation(self, explanation: SHAPExplanation, filepath: str) -> None:
        """
        Guarda explicación SHAP.
        
        Args:
            explanation: Explicación SHAP
            filepath: Ruta del archivo
        """
        import joblib
        
        # Convertir a formato serializable
        explanation_data = {
            'explanation_type': explanation.explanation_type.value,
            'values': explanation.values,
            'base_values': explanation.base_values,
            'data': explanation.data,
            'feature_names': explanation.feature_names,
            'timestamp': explanation.timestamp.isoformat(),
            'model_name': explanation.model_name,
            'metadata': explanation.metadata
        }
        
        joblib.dump(explanation_data, filepath)
        logger.info(f"Explicación SHAP guardada: {filepath}")
    
    def load_explanation(self, filepath: str) -> SHAPExplanation:
        """
        Carga explicación SHAP.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Explicación SHAP cargada
        """
        import joblib
        
        explanation_data = joblib.load(filepath)
        
        explanation = SHAPExplanation(
            explanation_type=ExplanationType(explanation_data['explanation_type']),
            values=explanation_data['values'],
            base_values=explanation_data['base_values'],
            data=explanation_data['data'],
            feature_names=explanation_data['feature_names'],
            timestamp=datetime.fromisoformat(explanation_data['timestamp']),
            model_name=explanation_data['model_name'],
            metadata=explanation_data.get('metadata', {})
        )
        
        logger.info(f"Explicación SHAP cargada: {filepath}")
        return explanation
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del explainer SHAP.
        
        Returns:
            Diccionario con resumen
        """
        return {
            'explainer_type': type(self.explainer).__name__ if self.explainer else None,
            'model_name': type(self.model).__name__ if self.model else None,
            'feature_names': self.feature_names,
            'explanations_count': len(self.explanations),
            'config': self.config
        }

