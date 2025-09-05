# Resumen de Modelos ML

## Descripción General

Los modelos ML son algoritmos de machine learning parsimoniosos que complementan los modelos baseline con mayor sofisticación y capacidad de capturar patrones complejos en los datos de trading. Están diseñados con regularización, selección de features y optimización de hyperparámetros para evitar sobreajuste.

## Arquitectura del Sistema

### Componentes Principales

#### 1. LassoModel
- **Algoritmo**: Regresión Lasso con regularización L1
- **Características**: Selección automática de features, escalado, regularización
- **Ventajas**: Interpretable, resistente al sobreajuste, selección de features
- **Uso**: Modelo lineal parsimonioso para señales de trading

#### 2. XGBoostModel
- **Algoritmo**: Gradient Boosting con regularización
- **Características**: Tree-based, manejo de missing values, regularización L1/L2
- **Ventajas**: Alta performance, robusto, maneja no-linealidad
- **Uso**: Modelo no-lineal para capturar interacciones complejas

#### 3. RandomForestModel
- **Algoritmo**: Random Forest con class balancing
- **Características**: Ensemble de árboles, bootstrap, feature bagging
- **Ventajas**: Robusto, no sobreajusta, importancia de features
- **Uso**: Modelo ensemble para estabilidad y robustez

#### 4. MLEnsemble
- **Algoritmo**: Ensemble de múltiples modelos ML
- **Características**: Votación suave/dura, pesos adaptativos, performance weighting
- **Ventajas**: Combina fortalezas de diferentes algoritmos
- **Uso**: Modelo final que combina todos los enfoques

### MLModelTrainer
- Coordina entrenamiento de modelos ML
- Optimización de hyperparámetros con Optuna
- Validación temporal y cross-validation
- Comparación con modelos baseline
- Métricas especializadas de trading

## Modelos Implementados

### 1. LassoModel

#### Configuración por Defecto
```yaml
alpha: 0.01
max_iter: 1000
tol: 1e-4
random_state: 42
selection: 'cyclic'
normalize: False
feature_selection: True
n_features: 50
scaling: True
```

#### Características Técnicas
- **Regularización L1**: Penaliza coeficientes grandes
- **Selección de Features**: SelectKBest con f_classif
- **Escalado**: StandardScaler para normalización
- **Conversión de Señales**: Percentiles para discretización
- **Importancia**: Basada en magnitud de coeficientes

#### Proceso de Entrenamiento
1. **Preparación de Datos**: Remover NaN, seleccionar features
2. **Escalado**: Normalizar features para estabilidad numérica
3. **Selección de Features**: Top K features por importancia
4. **Entrenamiento**: Lasso con regularización L1
5. **Conversión**: Predicciones continuas a señales discretas

#### Features Utilizados
- Selección automática de top 50 features
- Basado en importancia estadística (f_classif)
- Incluye features de momentum, mean reversion, volatilidad
- Escalado para estabilidad numérica

### 2. XGBoostModel

#### Configuración por Defecto
```yaml
n_estimators: 100
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 1.0
random_state: 42
n_jobs: -1
feature_selection: True
n_features: 100
early_stopping_rounds: 10
```

#### Características Técnicas
- **Gradient Boosting**: Ensemble de árboles secuenciales
- **Regularización**: L1 (reg_alpha) y L2 (reg_lambda)
- **Feature Selection**: Basada en importancia de XGBoost
- **Manejo de Missing**: Automático en XGBoost
- **Multi-class**: Objetivo multi:softprob para 3 clases

#### Proceso de Entrenamiento
1. **Preparación de Datos**: Remover NaN, convertir target
2. **Selección de Features**: Basada en importancia de XGBoost
3. **Entrenamiento**: XGBoost con regularización
4. **Predicción**: Probabilidades para cada clase
5. **Conversión**: De [0,1,2] a [-1,0,1]

#### Features Utilizados
- Selección automática de top 100 features
- Basado en importancia de XGBoost (gain, cover, frequency)
- Incluye features de todas las familias
- Manejo automático de missing values

### 3. RandomForestModel

#### Configuración por Defecto
```yaml
n_estimators: 100
max_depth: 10
min_samples_split: 5
min_samples_leaf: 2
max_features: 'sqrt'
bootstrap: True
random_state: 42
n_jobs: -1
feature_selection: True
n_features: 100
```

#### Características Técnicas
- **Random Forest**: Ensemble de árboles con bootstrap
- **Feature Bagging**: max_features='sqrt' para diversidad
- **Class Balancing**: class_weight='balanced'
- **Feature Selection**: RFE con Random Forest
- **Robustez**: Menos propenso a sobreajuste

#### Proceso de Entrenamiento
1. **Preparación de Datos**: Remover NaN
2. **Selección de Features**: RFE con Random Forest
3. **Entrenamiento**: Random Forest con class balancing
4. **Predicción**: Votación mayoritaria de árboles
5. **Importancia**: Promedio de importancia de árboles

#### Features Utilizados
- Selección automática de top 100 features
- Basada en RFE (Recursive Feature Elimination)
- Incluye features de todas las familias
- Bootstrap para robustez

### 4. MLEnsemble

#### Configuración por Defecto
```yaml
models: ['lasso', 'xgboost', 'random_forest']
weights: None  # Se calculan automáticamente
voting_method: 'soft'  # 'hard' o 'soft'
performance_weighting: True
min_models: 2
```

#### Características Técnicas
- **Ensemble**: Combina múltiples modelos ML
- **Votación Suave**: Promedio ponderado de probabilidades
- **Votación Dura**: Votación mayoritaria de predicciones
- **Pesos Adaptativos**: Basados en performance (Sharpe ratio)
- **Robustez**: Múltiples algoritmos para estabilidad

#### Proceso de Entrenamiento
1. **Entrenamiento Individual**: Cada modelo se entrena por separado
2. **Cálculo de Pesos**: Basado en Sharpe ratio de cada modelo
3. **Normalización**: Pesos suman 1.0
4. **Predicción**: Combinación ponderada de predicciones
5. **Validación**: Performance del ensemble vs modelos individuales

#### Métodos de Votación
- **Hard Voting**: Votación mayoritaria de señales discretas
- **Soft Voting**: Promedio ponderado de probabilidades
- **Performance Weighting**: Pesos basados en Sharpe ratio
- **Fallback**: Pesos uniformes si no hay performance positiva

## Optimización de Hyperparámetros

### Framework: Optuna
- **Algoritmo**: Tree-structured Parzen Estimator (TPE)
- **Objetivo**: Maximizar Sharpe ratio
- **Validación**: TimeSeriesSplit con 3 folds
- **Trials**: 50 por modelo (configurable)

### Parámetros Optimizados

#### LassoModel
```python
'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True)
'max_iter': trial.suggest_int('max_iter', 500, 2000)
'n_features': trial.suggest_int('n_features', 20, 100)
```

#### XGBoostModel
```python
'n_estimators': trial.suggest_int('n_estimators', 50, 300)
'max_depth': trial.suggest_int('max_depth', 3, 10)
'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
'subsample': trial.suggest_float('subsample', 0.6, 1.0)
'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
'n_features': trial.suggest_int('n_features', 50, 150)
```

#### RandomForestModel
```python
'n_estimators': trial.suggest_int('n_estimators', 50, 300)
'max_depth': trial.suggest_int('max_depth', 5, 20)
'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8])
'n_features': trial.suggest_int('n_features', 50, 150)
```

### Proceso de Optimización
1. **Definir Espacio**: Rangos de parámetros para cada modelo
2. **Función Objetivo**: Sharpe ratio en validación temporal
3. **Sampling**: TPE para exploración eficiente
4. **Validación**: TimeSeriesSplit para robustez temporal
5. **Selección**: Mejores parámetros por modelo

## Métricas de Evaluación

### Métricas de Clasificación
- **Accuracy**: Precisión general de señales
- **Precision**: Precisión por clase (weighted average)
- **Recall**: Sensibilidad por clase (weighted average)
- **F1-Score**: Media armónica de precision y recall

### Métricas de Trading
- **Hit Ratio**: Porcentaje de trades ganadores
- **Profit Factor**: Gross Profit / Gross Loss
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Maximum Drawdown**: Pérdida máxima desde peak
- **Total Return**: Retorno total de la estrategia

### Métricas de Modelo
- **N Features Used**: Número de features seleccionadas
- **Model Parameters**: Parámetros específicos del modelo
- **Training Time**: Tiempo de entrenamiento
- **Prediction Time**: Tiempo de predicción

## Validación y Testing

### 1. Validación Temporal
- **Split**: 80% entrenamiento, 20% validación
- **Orden**: Cronológico (no aleatorio)
- **Métricas**: Calculadas en datos de validación
- **Robustez**: Evalúa estabilidad temporal

### 2. Cross-Validation Temporal
- **TimeSeriesSplit**: 5 folds con orden temporal
- **Métricas**: Promediadas entre folds
- **Estabilidad**: Evalúa consistencia entre folds
- **Generalización**: Performance en diferentes períodos

### 3. Comparación con Baseline
- **Métricas**: Mismas métricas para comparación justa
- **Significancia**: Test estadístico de diferencias
- **Robustez**: Performance en diferentes regímenes
- **Interpretabilidad**: Balance entre performance y transparencia

## Uso del Sistema

### Entrenamiento
```python
from src.models.ml_trainer import MLModelTrainer

# Crear trainer ML
ml_trainer = MLModelTrainer("configs/default_parameters.yaml")

# Entrenar modelos con optimización
ml_trained_models = ml_trainer.train_models(features, optimize_hyperparams=True)

# Validar modelos
ml_validation_results = ml_trainer.validate_models(features)

# Cross-validation
ml_cv_results = ml_trainer.cross_validate_models(features, n_splits=5)
```

### Predicción
```python
# Cargar modelo entrenado
lasso_model = LassoModel()
lasso_model.load_model("models/ml_models/lasso_model.pkl")

# Generar predicciones
predictions = lasso_model.predict(features)
probabilities = lasso_model.predict_proba(features)
```

### Evaluación
```python
# Obtener mejor modelo
best_model_name, best_model = ml_trainer.get_best_model('sharpe_ratio')

# Generar reporte
ml_report = ml_trainer.generate_model_report()

# Comparar con baseline
comparison_report = ml_trainer.compare_with_baseline(baseline_results)
```

## Comandos de Ejecución

```bash
# Entrenar solo modelos ML
python main.py --ml

# Pipeline completo (datos + features + baseline + ML)
python main.py --pipeline

# Demo con modelos ML
python main.py --demo
```

## Almacenamiento

### Estructura de Archivos
```
models/ml_models/
├── lasso_model.pkl
├── xgboost_model.pkl
├── random_forest_model.pkl
├── ml_ensemble_model.pkl
└── ml_training_results.pkl
```

### Metadatos Guardados
- Configuración del modelo
- Hyperparámetros optimizados
- Métricas de performance
- Resultados de validación
- Importancia de features
- Timestamp de entrenamiento

## Tests

### Cobertura de Tests
- **LassoModel**: 6 tests
- **XGBoostModel**: 5 tests
- **RandomForestModel**: 5 tests
- **MLEnsemble**: 5 tests
- **MLModelTrainer**: 7 tests

### Ejecutar Tests
```bash
pytest tests/test_ml_models.py -v
```

## Ventajas de los Modelos ML

### 1. Capacidad de Aprendizaje
- **Patrones Complejos**: Captura interacciones no lineales
- **Adaptabilidad**: Se ajusta a diferentes regímenes
- **Feature Selection**: Selecciona automáticamente features relevantes
- **Regularización**: Evita sobreajuste

### 2. Performance Superior
- **Métricas Mejoradas**: Generalmente supera modelos baseline
- **Robustez**: Múltiples algoritmos para estabilidad
- **Optimización**: Hyperparámetros optimizados automáticamente
- **Ensemble**: Combina fortalezas de diferentes enfoques

### 3. Escalabilidad
- **Múltiples Features**: Maneja cientos de features
- **Paralelización**: Entrenamiento paralelo
- **Eficiencia**: Algoritmos optimizados
- **Memoria**: Gestión eficiente de memoria

### 4. Flexibilidad
- **Configuración**: Parámetros adaptables
- **Extensibilidad**: Fácil agregar nuevos modelos
- **Modularidad**: Componentes independientes
- **Personalización**: Configuración por YAML

## Limitaciones

### 1. Complejidad
- **Interpretabilidad**: Menos transparente que baseline
- **Debugging**: Más difícil de debuggear
- **Mantenimiento**: Requiere más expertise
- **Dependencias**: Más librerías externas

### 2. Riesgos
- **Sobreajuste**: Riesgo de overfitting
- **Data Leakage**: Riesgo de leakage temporal
- **Estabilidad**: Puede ser menos estable
- **Regulaciones**: Menos compliance automático

### 3. Recursos
- **Computación**: Requiere más recursos
- **Tiempo**: Entrenamiento más lento
- **Memoria**: Mayor uso de memoria
- **Dependencias**: Más librerías externas

## Comparación con Baseline

### Ventajas ML vs Baseline
- **Performance**: Generalmente superior
- **Capacidad**: Maneja patrones complejos
- **Automatización**: Menos intervención manual
- **Escalabilidad**: Maneja más features

### Ventajas Baseline vs ML
- **Transparencia**: Más interpretable
- **Robustez**: Menos propenso a sobreajuste
- **Simplicidad**: Más fácil de entender
- **Compliance**: Mejor para regulaciones

### Casos de Uso
- **ML**: Cuando se necesita máxima performance
- **Baseline**: Cuando se necesita transparencia
- **Híbrido**: Combinar ambos enfoques
- **Contexto**: Depende del caso de uso específico

## Próximos Pasos

1. **Gestión de Riesgo**: Implementar risk management
2. **Backtesting**: Walk-forward analysis
3. **Métricas**: Criterios de aprobación robustos
4. **Explainability**: SHAP y logs de auditoría
5. **Paper Trading**: Módulo de paper trading

---

**Total de Modelos**: 4 (Lasso, XGBoost, Random Forest, ML Ensemble)
**Features Utilizados**: ~100 por modelo (selección automática)
**Métricas**: 15+ métricas de evaluación
**Tests**: 28 tests unitarios
**Validación**: Temporal + Cross-validation + Hyperparameter tuning
**Optimización**: Optuna con 50 trials por modelo

