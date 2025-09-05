# Resumen de Explainability

## Descripción General

El sistema de explainability proporciona transparencia total en las decisiones del sistema de trading cuantitativo mediante explicaciones SHAP detalladas y un sistema de auditoría comprehensivo. Implementa explainability tanto a nivel global (modelo completo) como local (predicciones individuales) con logging completo de todas las operaciones.

## Arquitectura del Sistema

### Componentes Principales

#### 1. SHAPExplainer
- **Explainer SHAP**: Proporciona explicaciones detalladas usando SHAP
- **Múltiples Tipos**: Tree, Linear, Kernel explainers
- **Explicaciones**: Global, local, feature importance, interacciones
- **Visualizaciones**: Summary plots, waterfall plots, force plots

#### 2. AuditLogger
- **Sistema de Auditoría**: Registra todas las operaciones del sistema
- **Base de Datos**: SQLite para persistencia de logs
- **Eventos**: Modelos, predicciones, trades, errores
- **Reportes**: Reportes de auditoría comprehensivos

#### 3. ExplainabilityManager (Principal)
- **Coordinador**: Orquesta todos los componentes de explainability
- **Integración**: Se integra con modelos y decisiones de trading
- **Reportes**: Genera reportes detallados de explicaciones
- **Exportación**: Múltiples formatos de exportación

## Componentes Implementados

### 1. SHAPExplainer

#### Tipos de Explainer
- **TreeExplainer**: Para modelos basados en árboles (XGBoost, Random Forest)
- **LinearExplainer**: Para modelos lineales (Lasso, Ridge, Elastic Net)
- **KernelExplainer**: Para cualquier modelo (fallback universal)

#### Tipos de Explicación

##### Global
- **Explicación del Modelo**: Explicación completa del modelo
- **Feature Importance**: Importancia de todas las features
- **Interacciones**: Interacciones entre features
- **Estadísticas**: Estadísticas de explicación

##### Local
- **Predicción Individual**: Explicación de una predicción específica
- **Contribución por Feature**: Contribución de cada feature
- **Valor Base**: Valor base del modelo
- **Razonamiento**: Razonamiento generado automáticamente

#### Características Técnicas
- **Detección Automática**: Detecta tipo de modelo automáticamente
- **Manejo de Errores**: Manejo robusto de errores
- **Configuración**: Configuración flexible por YAML
- **Persistencia**: Guardado/carga de explicaciones

### 2. AuditLogger

#### Tipos de Eventos
- **DATA_LOAD**: Carga de datos
- **FEATURE_CREATION**: Creación de features
- **MODEL_TRAINING**: Entrenamiento de modelos
- **MODEL_PREDICTION**: Predicciones de modelos
- **BACKTESTING**: Ejecución de backtesting
- **VALIDATION**: Validación de estrategias
- **RISK_MANAGEMENT**: Gestión de riesgo
- **TRADE_EXECUTION**: Ejecución de trades
- **SYSTEM_EVENT**: Eventos del sistema
- **ERROR**: Errores del sistema

#### Niveles de Log
- **DEBUG**: Información de depuración
- **INFO**: Información general
- **WARNING**: Advertencias
- **ERROR**: Errores
- **CRITICAL**: Errores críticos

#### Estructura de Base de Datos
```sql
-- Eventos de auditoría
CREATE TABLE audit_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    metadata TEXT
);

-- Decisiones de modelo
CREATE TABLE model_decisions (
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
);

-- Decisiones de trading
CREATE TABLE trade_decisions (
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
);
```

#### Características Técnicas
- **SQLite**: Base de datos ligera y portable
- **Índices**: Índices para performance
- **Retención**: Limpieza automática de logs antiguos
- **Configuración**: Configuración flexible
- **Múltiples Handlers**: Consola, archivo, base de datos

### 3. ExplainabilityManager (Principal)

#### Configuración por Defecto
```yaml
explainability:
  shap:
    sample_size: 1000
    max_features: 50
    random_state: 42
  audit:
    db_path: "audit_logs.db"
    retention_days: 365
    max_log_size: 1000000
    log_level: "INFO"
    enable_console: true
    enable_file: true
    enable_database: true
```

#### Características Técnicas
- **Integración Completa**: Se integra con todos los componentes
- **Explicaciones Múltiples**: Explica múltiples modelos
- **Reportes Detallados**: Genera reportes comprensivos
- **Exportación**: JSON, pickle
- **Persistencia**: Guardado/carga de estado

## Sistema de Explicaciones

### Explicación de Modelos
1. **Ajuste del Explainer**: Ajusta explainer SHAP al modelo
2. **Explicación Global**: Genera explicación global del modelo
3. **Feature Importance**: Calcula importancia de features
4. **Interacciones**: Calcula interacciones entre features
5. **Resumen**: Genera resumen de explicación

### Explicación de Predicciones
1. **Explicación Local**: Genera explicación local para instancia
2. **Cálculo de Confianza**: Calcula confianza de la predicción
3. **Razonamiento**: Genera razonamiento automático
4. **Log de Decisión**: Registra decisión en auditoría

### Explicación de Trading
1. **Razonamiento de Trading**: Genera razonamiento para decisión
2. **Integración con Modelo**: Integra explicación del modelo
3. **Evaluación de Riesgo**: Considera evaluación de riesgo
4. **Log de Trading**: Registra decisión de trading

## Uso del Sistema

### Inicialización
```python
from src.explainability.explainability_manager import ExplainabilityManager

# Crear gestor de explainability
explainability_manager = ExplainabilityManager("configs/default_parameters.yaml")

# Explicar modelo
explanations = explainability_manager.explain_model(
    model, X_train, X_test, "mi_modelo"
)
```

### Explicación de Predicción
```python
# Explicar predicción específica
prediction_explanation = explainability_manager.explain_prediction(
    model, X_train, instance, "mi_modelo"
)

print(f"Predicción: {prediction_explanation['prediction']}")
print(f"Confianza: {prediction_explanation['confidence']}")
print(f"Razonamiento: {prediction_explanation['reasoning']}")
```

### Explicación de Trading
```python
# Explicar decisión de trading
trade_explanation = explainability_manager.explain_trade_decision(
    symbol="SPY",
    action="buy",
    quantity=100.0,
    price=400.0,
    model_explanation=prediction_explanation,
    risk_assessment={"risk_level": "medium"}
)
```

### Análisis de Resultados
```python
# Obtener explicaciones de modelos
model_explanations = explainability_manager.get_model_explanations()

# Crear reporte
report = explainability_manager.create_explanation_report()

# Obtener reporte de auditoría
audit_report = explainability_manager.get_audit_report()
```

## Comandos de Ejecución

```bash
# Solo explainability
python main.py --explainability

# Pipeline completo (incluye explainability)
python main.py --pipeline

# Demo con explainability
python main.py --demo
```

## Almacenamiento

### Estructura de Archivos
```
explainability/
├── shap_explainer.py           # Explainer SHAP
├── audit_logger.py            # Sistema de auditoría
└── explainability_manager.py  # Gestor principal
```

### Persistencia de Explicaciones
```python
# Exportar explicaciones
explainability_manager.export_explanations("explanations.json", "json")
explainability_manager.export_explanations("explanations.pkl", "pickle")

# Limpiar logs antiguos
deleted = explainability_manager.cleanup_old_logs(days=30)
```

## Tests

### Cobertura de Tests
- **SHAPExplainer**: 8 tests
- **AuditLogger**: 12 tests
- **ExplainabilityManager**: 10 tests

### Ejecutar Tests
```bash
pytest tests/test_explainability.py -v
```

## Ventajas del Sistema

### 1. Transparencia Total
- **Explicaciones SHAP**: Explicaciones detalladas de decisiones
- **Auditoría Completa**: Registro de todas las operaciones
- **Trazabilidad**: Trazabilidad completa de decisiones
- **Reportes**: Reportes detallados y comprensivos

### 2. Flexibilidad
- **Múltiples Explainer**: Soporte para diferentes tipos de modelos
- **Configuración**: Configuración flexible por YAML
- **Exportación**: Múltiples formatos de exportación
- **Persistencia**: Guardado/carga de estado

### 3. Integración Completa
- **Modelos**: Se integra con todos los modelos
- **Trading**: Explica decisiones de trading
- **Pipeline**: Parte del pipeline completo
- **Auditoría**: Auditoría de todas las operaciones

### 4. Usabilidad
- **Comandos Simples**: Comandos de línea simples
- **Reportes Claros**: Reportes fáciles de entender
- **Visualizaciones**: Plots SHAP integrados
- **Documentación**: Documentación completa

## Limitaciones

### 1. Dependencias
- **SHAP**: Requiere instalación de SHAP
- **Computación**: Requiere recursos computacionales
- **Datos**: Requiere datos de calidad
- **Modelos**: Requiere modelos entrenados

### 2. Complejidad
- **Configuración**: Requiere configuración cuidadosa
- **Mantenimiento**: Requiere mantenimiento regular
- **Expertise**: Requiere conocimiento de SHAP
- **Recursos**: Requiere recursos computacionales

### 3. Limitaciones Técnicas
- **Modelos Complejos**: Puede ser lento para modelos muy complejos
- **Datos Grandes**: Puede requerir muestreo para datos grandes
- **Interpretabilidad**: Algunos modelos son inherentemente no interpretables
- **Overfitting**: Explicaciones pueden overfittear

## Casos de Uso

### 1. Explicación de Modelos
- **Desarrollo**: Explicar modelos durante desarrollo
- **Validación**: Validar comportamiento de modelos
- **Optimización**: Optimizar modelos basado en explicaciones
- **Documentación**: Documentar comportamiento de modelos

### 2. Auditoría y Cumplimiento
- **Reguladores**: Cumplimiento regulatorio
- **Auditores**: Auditoría de decisiones
- **Compliance**: Cumplimiento de políticas
- **Risk Management**: Gestión de riesgo

### 3. Transparencia
- **Stakeholders**: Transparencia para stakeholders
- **Clientes**: Explicaciones para clientes
- **Inversores**: Transparencia para inversores
- **Público**: Transparencia pública

## Próximos Pasos

1. **Paper Trading**: Integrar con paper trading
2. **Producción**: Plan de paso a producción

---

**Total de Componentes**: 3 (SHAP Explainer, Audit Logger, Gestor)
**Tipos de Explainer**: 3 (Tree, Linear, Kernel)
**Tipos de Explicación**: 2 (Global, Local)
**Tipos de Eventos**: 10 tipos de eventos
**Niveles de Log**: 5 niveles
**Tests**: 30 tests unitarios
**Configuración**: YAML completamente configurable
**Persistencia**: Base de datos SQLite
**Exportación**: Múltiples formatos
**Reportes**: Reportes detallados

