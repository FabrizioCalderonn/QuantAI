# Resumen del Feature Store

## Descripción General

El Feature Store del sistema de trading cuantitativo es un sistema robusto y escalable que genera más de 200 features por instrumento, organizados en tres familias principales:

1. **Features Técnicos** - Indicadores técnicos tradicionales y avanzados
2. **Features Estadísticos** - Análisis estadístico avanzado y detección de regímenes
3. **Features Cross-Asset** - Correlaciones, carry, term structure y factores macro

## Arquitectura del Sistema

### Componentes Principales

#### 1. BaseFeatureExtractor
- Clase abstracta base para todos los extractores
- Validación de datos automática
- Funciones utilitarias para cálculos seguros
- Manejo de errores robusto

#### 2. TechnicalFeatureExtractor
- **Momentum**: RSI, MACD, Rate of Change, Momentum
- **Mean Reversion**: Bollinger Bands, Williams %R, Stochastic Oscillator
- **Volatilidad**: ATR, Volatilidad rolling, Volatility clustering
- **Volumen**: VWAP, OBV, Volume ratios
- **Posición de Precio**: Price position, High-Low ratios

#### 3. StatisticalFeatureExtractor
- **Distribución**: Skewness, Kurtosis, Jarque-Bera, Percentiles
- **Autocorrelación**: Lags múltiples, Ljung-Box test
- **Volatilidad Avanzada**: GARCH-like, Parkinson, Vol of Vol
- **Regímenes**: HMM simplificado, Volatility regimes, Trend regimes
- **Estacionalidad**: Day-of-week, Month, Hour effects
- **Momentos**: 3rd-6th moments, Persistence, Fractals

#### 4. CrossAssetFeatureExtractor
- **Correlaciones**: Rolling correlations, Volatility correlations
- **Carry**: Relative carry, Carry momentum, Term structure
- **Factores Macro**: Risk-on/Risk-off, Market regimes
- **Momentum Cross-Asset**: Relative momentum, Momentum ranking
- **Volatilidad Cross-Asset**: Relative volatility, Vol clustering

### FeatureEngineer
- Coordina todos los extractores
- Post-procesamiento de features
- Validación de calidad
- Guardado y carga de features

## Features Generados

### Por Familia

#### Features Técnicos (~80 features)
```
- returns_1, returns_5, returns_10, returns_20
- log_returns_1, log_returns_5, log_returns_10, log_returns_20
- volatility_5, volatility_10, volatility_20, volatility_50
- atr_14, atr_21, atr_50, atr_pct_14, atr_pct_21, atr_pct_50
- rsi_14, rsi_21, rsi_50, rsi_norm_14, rsi_norm_21, rsi_norm_50
- macd, macd_signal, macd_histogram
- roc_5, roc_10, roc_20, roc_50
- momentum_5, momentum_10, momentum_20, momentum_50
- bb_upper_20_1, bb_upper_20_2, bb_middle_20_1, bb_middle_20_2
- bb_lower_20_1, bb_lower_20_2, bb_position_20_1, bb_position_20_2
- bb_width_20_1, bb_width_20_2
- williams_r_14, williams_r_21, williams_r_50
- stoch_k_14_3, stoch_k_14_5, stoch_d_14_3, stoch_d_14_5
- price_sma_ratio_20, price_sma_ratio_50, price_sma_ratio_100, price_sma_ratio_200
- volume_sma_10, volume_sma_20, volume_sma_50
- volume_ratio_10, volume_ratio_20, volume_ratio_50
- vwap_20, vwap_50, price_vwap_ratio_20, price_vwap_ratio_50
- obv, obv_sma_10, obv_sma_20, obv_sma_50
- price_position_daily, price_position_20, price_position_50
- zscore_20, zscore_50, zscore_100
- percentile_rank_20, percentile_rank_50, percentile_rank_100
```

#### Features Estadísticos (~60 features)
```
- skewness_20, skewness_50, skewness_100
- kurtosis_20, kurtosis_50, kurtosis_100
- jarque_bera_20, jarque_bera_50, jarque_bera_100
- percentile_5_20, percentile_10_20, percentile_25_20, percentile_75_20, percentile_90_20, percentile_95_20
- iqr_20, iqr_50, iqr_100
- autocorr_lag1_20, autocorr_lag1_50, autocorr_lag1_100
- autocorr_lag2_20, autocorr_lag2_50, autocorr_lag2_100
- autocorr_lag3_20, autocorr_lag3_50, autocorr_lag3_100
- autocorr_lag5_20, autocorr_lag5_50, autocorr_lag5_100
- autocorr_lag10_20, autocorr_lag10_50, autocorr_lag10_100
- ljung_box_20, ljung_box_50, ljung_box_100
- vol_ewma_20, vol_ewma_50, vol_ewma_100
- vol_parkinson_20, vol_parkinson_50, vol_parkinson_100
- vol_of_vol_20, vol_of_vol_50, vol_of_vol_100
- vol_clustering_20, vol_clustering_50, vol_clustering_100
- high_vol_regime_50, high_vol_regime_100, high_vol_regime_200
- uptrend_regime_50, uptrend_regime_100, uptrend_regime_200
- positive_momentum_regime_50, positive_momentum_regime_100, positive_momentum_regime_200
- hmm_state_low_vol, hmm_state_medium_vol, hmm_state_high_vol
- day_0_mean, day_1_mean, day_2_mean, day_3_mean, day_4_mean, day_5_mean, day_6_mean
- month_1_mean, month_2_mean, ..., month_12_mean
- hour_0_mean, hour_1_mean, ..., hour_23_mean
- moment3_20, moment3_50, moment3_100
- moment4_20, moment4_50, moment4_100
- moment5_20, moment5_50, moment5_100
- moment6_20, moment6_50, moment6_100
- hurst_20, hurst_50, hurst_100
- variance_ratio_20, variance_ratio_50, variance_ratio_100
- fractal_dimension_20, fractal_dimension_50, fractal_dimension_100
```

#### Features Cross-Asset (~80 features)
```
- corr_SPY_20, corr_SPY_50, corr_SPY_100
- corr_QQQ_20, corr_QQQ_50, corr_QQQ_100
- corr_GLD_20, corr_GLD_50, corr_GLD_100
- vol_corr_SPY_20, vol_corr_SPY_50, vol_corr_SPY_100
- vol_corr_QQQ_20, vol_corr_QQQ_50, vol_corr_QQQ_100
- vol_corr_GLD_20, vol_corr_GLD_50, vol_corr_GLD_100
- carry_20_100, carry_20_200, carry_50_100, carry_50_200
- carry_momentum_20_100, carry_momentum_20_200, carry_momentum_50_100, carry_momentum_50_200
- relative_carry_SPY, relative_carry_QQQ, relative_carry_GLD
- term_slope_5_50, term_slope_5_100, term_slope_5_200
- term_slope_10_50, term_slope_10_100, term_slope_10_200
- term_slope_20_50, term_slope_20_100, term_slope_20_200
- term_curvature_5_50, term_curvature_5_100, term_curvature_5_200
- risk_on_off
- corr_risk_on_off_20, corr_risk_on_off_50
- relative_momentum_SPY_20, relative_momentum_SPY_50, relative_momentum_SPY_100
- relative_momentum_QQQ_20, relative_momentum_QQQ_50, relative_momentum_QQQ_100
- relative_momentum_GLD_20, relative_momentum_GLD_50, relative_momentum_GLD_100
- momentum_rank_20, momentum_rank_50, momentum_rank_100
- relative_volatility
- vol_clustering_cross_20, vol_clustering_cross_50
- high_corr_regime_SPY_50, high_corr_regime_SPY_100
- high_corr_regime_QQQ_50, high_corr_regime_QQQ_100
- high_corr_regime_GLD_50, high_corr_regime_GLD_100
```

## Post-Procesamiento

### 1. Limpieza de Outliers
- **Método IQR**: Remueve valores fuera de Q1 - 3*IQR y Q3 + 3*IQR
- **Método Z-Score**: Remueve valores con |z-score| > 3
- **Método Modified Z-Score**: Usa MAD (Median Absolute Deviation)

### 2. Normalización por Volatilidad
- Normaliza features por volatilidad rolling
- Evita que features de alta volatilidad dominen el modelo

### 3. Winsorización
- Límites: 1% superior e inferior
- Reduce impacto de outliers extremos

### 4. Imputación de Valores Faltantes
- Forward fill → Backward fill → Rolling mean
- Mantiene la integridad temporal de los datos

## Validación de Calidad

### Checks Automáticos
1. **Columnas completamente NaN**: Removidas automáticamente
2. **Filas con >80% NaN**: Removidas automáticamente
3. **Valores infinitos**: Reemplazados con NaN
4. **Validación de rangos**: Features con rangos esperados

### Métricas de Calidad
- **Coverage**: Porcentaje de valores no-NaN
- **Stability**: Consistencia temporal de features
- **Correlation**: Correlación con target (returns)

## Almacenamiento

### Formato
- **Parquet**: Compresión eficiente, tipos de datos preservados
- **Metadatos JSON**: Información de creación, símbolos, conteos
- **Estructura**: `{symbol}_features.parquet`

### Organización
```
data/processed/features/
├── SPY_features.parquet
├── QQQ_features.parquet
├── GLD_features.parquet
└── features_metadata.json
```

## Uso del Sistema

### Creación de Features
```python
from src.features.engineering import FeatureEngineer

# Inicializar
engineer = FeatureEngineer("configs/default_parameters.yaml")

# Crear features
features = engineer.create_features(data)

# Guardar
engineer.save_features(features, "data/processed/features")
```

### Carga de Features
```python
# Cargar features existentes
features = engineer.load_features("data/processed/features")

# Obtener resumen
summary = engineer.get_feature_summary(features)
```

### Análisis de Importancia
```python
# Calcular importancia
importance = engineer.get_feature_importance(features, target='returns_1')

# Mostrar top features
for symbol, imp_df in importance.items():
    print(f"\n{symbol} - Top 10 Features:")
    print(imp_df.head(10)[['feature', 'abs_correlation']])
```

## Comandos de Ejecución

```bash
# Ejecutar solo feature engineering
python main.py --features

# Ejecutar pipeline completo (datos + features)
python main.py --pipeline

# Ejecutar demo con features
python main.py --demo
```

## Tests

### Cobertura de Tests
- **TechnicalFeatureExtractor**: 5 tests
- **StatisticalFeatureExtractor**: 3 tests  
- **CrossAssetFeatureExtractor**: 3 tests
- **FeatureEngineer**: 5 tests
- **BaseFeatureExtractor**: 3 tests

### Ejecutar Tests
```bash
pytest tests/test_feature_engineering.py -v
```

## Próximos Pasos

1. **Modelos Baseline**: Usar features para reglas simples
2. **Modelos ML**: Entrenar con features seleccionados
3. **Feature Selection**: Implementar selección automática
4. **Feature Monitoring**: Detectar drift en features
5. **Feature Versioning**: Control de versiones de features

---

**Total de Features**: ~220 por instrumento
**Familias**: 3 (Technical, Statistical, Cross-Asset)
**Validación**: Automática con múltiples checks
**Almacenamiento**: Parquet con metadatos
**Tests**: 19 tests unitarios

