# Resumen de Modelos Baseline

## Descripción General

Los modelos baseline son estrategias de trading transparentes y interpretables que sirven como punto de referencia para evaluar modelos ML más complejos. Están diseñados con reglas claras basadas en indicadores técnicos tradicionales y principios de trading cuantitativo.

## Arquitectura del Sistema

### Componentes Principales

#### 1. BaseModel
- Clase abstracta base para todos los modelos
- Interfaz común: `fit()`, `predict()`, `predict_proba()`
- Métricas de trading integradas
- Sistema de guardado/carga
- Validación de datos automática

#### 2. MomentumModel
- **Estrategia**: Seguir tendencias de precios
- **Señales**: Long en momentum positivo, Short en momentum negativo
- **Indicadores**: RSI, MACD, Moving Averages, Rate of Change
- **Filtros**: Volatilidad, confianza mínima

#### 3. MeanReversionModel
- **Estrategia**: Revertir a la media cuando hay sobrecompra/sobreventa
- **Señales**: Long en oversold, Short en overbought
- **Indicadores**: Bollinger Bands, RSI, Williams %R, Stochastic
- **Filtros**: Z-score, confianza mínima

#### 4. HybridModel
- **Estrategia**: Combinar momentum y mean reversion según régimen
- **Regímenes**: Trending (más momentum), Ranging (más mean reversion)
- **Detección**: Basada en volatilidad y ATR
- **Pesos**: Adaptativos según régimen detectado

### ModelTrainer
- Coordina entrenamiento de todos los modelos
- Validación temporal y cross-validation
- Métricas de trading especializadas
- Selección del mejor modelo
- Reportes de performance

## Modelos Implementados

### 1. MomentumModel

#### Configuración por Defecto
```yaml
short_period: 5
medium_period: 20
long_period: 50
rsi_period: 14
rsi_oversold: 30
rsi_overbought: 70
volatility_period: 20
volatility_threshold: 0.02
min_confidence: 0.3
```

#### Lógica de Señales
1. **Momentum de Precios**: Retornos positivos → Long, negativos → Short
2. **RSI**: Oversold → Long, Overbought → Short
3. **Moving Averages**: Precio > SMA → Long, Precio < SMA → Short
4. **MACD**: MACD > Signal → Long, MACD < Signal → Short
5. **Filtro de Volatilidad**: Reduce confianza en alta volatilidad

#### Features Utilizados
- `returns_1`, `returns_5`, `returns_10`, `returns_20`
- `rsi_14`, `rsi_21`, `rsi_50`
- `price_sma_ratio_20`, `price_sma_ratio_50`
- `volatility_20`, `volatility_50`
- `macd`, `macd_signal`, `macd_histogram`

### 2. MeanReversionModel

#### Configuración por Defecto
```yaml
bb_period: 20
bb_std: 2
rsi_period: 14
rsi_oversold: 30
rsi_overbought: 70
williams_period: 14
williams_oversold: -80
williams_overbought: -20
stoch_period: 14
stoch_oversold: 20
stoch_overbought: 80
min_confidence: 0.3
```

#### Lógica de Señales
1. **Bollinger Bands**: Near lower band → Long, near upper band → Short
2. **RSI**: Oversold → Long, Overbought → Short
3. **Williams %R**: Oversold → Long, Overbought → Short
4. **Stochastic**: Oversold → Long, Overbought → Short
5. **Z-Score**: 2 std below mean → Long, 2 std above mean → Short

#### Features Utilizados
- `bb_position_20_2`, `bb_position_50_2`
- `rsi_14`, `rsi_21`, `rsi_50`
- `williams_r_14`, `williams_r_21`, `williams_r_50`
- `stoch_k_14_3`, `stoch_k_14_5`, `stoch_d_14_3`, `stoch_d_14_5`
- `zscore_20`, `zscore_50`

### 3. HybridModel

#### Configuración por Defecto
```yaml
momentum_weight: 0.6
mean_reversion_weight: 0.4
volatility_threshold: 0.02
regime_detection_period: 50
min_confidence: 0.3
```

#### Lógica de Regímenes
1. **Trending**: Alta volatilidad (>1.5x threshold) → 80% momentum, 20% mean reversion
2. **Ranging**: Baja volatilidad (<0.5x threshold) → 20% momentum, 80% mean reversion
3. **Mixed**: Volatilidad normal → Pesos configurados (60% momentum, 40% mean reversion)

#### Detección de Regímenes
- **Volatilidad**: Rolling volatility vs threshold
- **ATR**: Average True Range para confirmación
- **Adaptativo**: Pesos cambian según régimen detectado

## Métricas de Evaluación

### Métricas de Clasificación
- **Accuracy**: Precisión general de señales
- **Precision**: Precisión por clase (Long/Short/Neutral)
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precision y recall

### Métricas de Trading
- **Hit Ratio**: Porcentaje de trades ganadores
- **Profit Factor**: Gross Profit / Gross Loss
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Maximum Drawdown**: Pérdida máxima desde peak
- **Total Return**: Retorno total de la estrategia

### Métricas de Señales
- **Total Signals**: Número total de señales generadas
- **Long Signals**: Número de señales long
- **Short Signals**: Número de señales short
- **Neutral Signals**: Número de señales neutral

## Validación y Testing

### 1. Validación Temporal
- **Split**: 80% entrenamiento, 20% validación
- **Orden**: Cronológico (no aleatorio)
- **Métricas**: Calculadas en datos de validación

### 2. Cross-Validation Temporal
- **TimeSeriesSplit**: 5 folds con orden temporal
- **Métricas**: Promediadas entre folds
- **Robustez**: Evalúa estabilidad temporal

### 3. Métricas de Trading
- **Retornos de Estrategia**: Señales × Retornos reales
- **Risk-Adjusted**: Sharpe, Sortino, Calmar
- **Drawdown**: Maximum drawdown y duración
- **Capacity**: Tamaño máximo de capital

## Uso del Sistema

### Entrenamiento
```python
from src.models.trainer import ModelTrainer

# Crear trainer
trainer = ModelTrainer("configs/default_parameters.yaml")

# Entrenar modelos
trained_models = trainer.train_models(features)

# Validar modelos
validation_results = trainer.validate_models(features)

# Cross-validation
cv_results = trainer.cross_validate_models(features, n_splits=5)
```

### Predicción
```python
# Cargar modelo entrenado
momentum_model = MomentumModel()
momentum_model.load_model("models/baseline_models/momentum_model.pkl")

# Generar predicciones
predictions = momentum_model.predict(features)
probabilities = momentum_model.predict_proba(features)
```

### Evaluación
```python
# Obtener mejor modelo
best_model_name, best_model = trainer.get_best_model('sharpe_ratio')

# Generar reporte
report = trainer.generate_model_report()
print(report)
```

## Comandos de Ejecución

```bash
# Entrenar solo modelos baseline
python main.py --baseline

# Pipeline completo (datos + features + modelos)
python main.py --pipeline

# Demo con modelos baseline
python main.py --demo
```

## Almacenamiento

### Estructura de Archivos
```
models/baseline_models/
├── momentum_model.pkl
├── mean_reversion_model.pkl
├── hybrid_model.pkl
└── training_results.pkl
```

### Metadatos Guardados
- Configuración del modelo
- Métricas de performance
- Resultados de validación
- Timestamp de entrenamiento

## Tests

### Cobertura de Tests
- **MomentumModel**: 6 tests
- **MeanReversionModel**: 4 tests
- **HybridModel**: 4 tests
- **ModelTrainer**: 7 tests
- **TradingSignal**: 4 tests

### Ejecutar Tests
```bash
pytest tests/test_baseline_models.py -v
```

## Ventajas de los Modelos Baseline

### 1. Transparencia
- **Reglas Claras**: Lógica explícita y comprensible
- **Interpretabilidad**: Fácil entender por qué se genera una señal
- **Auditabilidad**: Completamente trazable

### 2. Robustez
- **Simplicidad**: Menos propenso a sobreajuste
- **Estabilidad**: Performance consistente en el tiempo
- **Generalización**: Funciona en diferentes regímenes

### 3. Benchmarking
- **Punto de Referencia**: Para comparar modelos ML
- **Baseline Mínimo**: Performance mínima esperada
- **Validación**: Verificar que ML supera reglas simples

### 4. Producción
- **Confiabilidad**: Menos riesgo de fallos
- **Mantenimiento**: Fácil de mantener y actualizar
- **Regulaciones**: Cumple con requerimientos de transparencia

## Limitaciones

### 1. Rigidez
- **Parámetros Fijos**: No se adapta automáticamente
- **Regímenes**: Puede fallar en cambios estructurales
- **Optimización**: Requiere ajuste manual de parámetros

### 2. Complejidad Limitada
- **Features Simples**: No captura relaciones complejas
- **No Linealidad**: No maneja interacciones no lineales
- **Contexto**: No considera información macro

### 3. Performance
- **Subóptimo**: Puede no ser el mejor posible
- **Oportunidades**: Puede perder oportunidades de trading
- **Eficiencia**: No optimiza para métricas específicas

## Próximos Pasos

1. **Modelos ML**: Implementar Lasso, XGBoost, Ensemble
2. **Feature Selection**: Selección automática de features
3. **Hyperparameter Tuning**: Optimización de parámetros
4. **Ensemble Methods**: Combinar baseline con ML
5. **Online Learning**: Adaptación en tiempo real

---

**Total de Modelos**: 3 (Momentum, Mean Reversion, Hybrid)
**Features Utilizados**: ~50 por modelo
**Métricas**: 10+ métricas de evaluación
**Tests**: 25 tests unitarios
**Validación**: Temporal + Cross-validation

