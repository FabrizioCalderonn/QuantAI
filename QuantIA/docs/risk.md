# Documento de Gestión de Riesgo

## 1. Marco de Gestión de Riesgo

### 1.1 Objetivos de Riesgo
- **Volatilidad Objetivo**: 15% anualizada
- **Drawdown Máximo**: 10% del capital
- **Pérdida Diaria Máxima**: 2% del capital
- **Sharpe Ratio Mínimo**: 1.5 (out-of-sample)

### 1.2 Principios Fundamentales
- **Diversificación**: No más del 20% de exposición por activo
- **Concentración de Factores**: No más del 30% de exposición por factor
- **Liquidez**: Solo instrumentos con volumen diario > $10M
- **Transparencia**: Todas las decisiones de riesgo deben ser auditables

## 2. Identificación y Clasificación de Riesgos

### 2.1 Riesgos de Mercado

#### 2.1.1 Riesgo de Precio
- **Definición**: Pérdidas por movimientos adversos de precios
- **Medición**: VaR 95% y 99%, CVaR, Maximum Drawdown
- **Control**: Stop losses, position sizing, volatility targeting

#### 2.1.2 Riesgo de Volatilidad
- **Definición**: Cambios en la volatilidad de los activos
- **Medición**: VIX, realized volatility, volatility of volatility
- **Control**: Volatility targeting, regime detection

#### 2.1.3 Riesgo de Correlación
- **Definición**: Aumento de correlaciones en crisis
- **Medición**: Rolling correlation, correlation breakdown
- **Control**: Diversificación, correlation limits

### 2.2 Riesgos Operacionales

#### 2.2.1 Riesgo de Ejecución
- **Definición**: Slippage, latencia, fallos de órdenes
- **Medición**: Implementation shortfall, execution quality
- **Control**: Order management, latency monitoring

#### 2.2.2 Riesgo de Modelo
- **Definición**: Errores en modelos, sobreajuste
- **Medición**: Model stability, out-of-sample performance
- **Control**: Walk-forward validation, model monitoring

#### 2.2.3 Riesgo de Datos
- **Definición**: Datos incorrectos, faltantes, tardíos
- **Medición**: Data quality metrics, latency
- **Control**: Data validation, multiple sources

### 2.3 Riesgos de Liquidez

#### 2.3.1 Riesgo de Liquidez de Mercado
- **Definición**: Imposibilidad de cerrar posiciones
- **Medición**: Bid-ask spread, market depth
- **Control**: Liquidity filters, position limits

#### 2.3.2 Riesgo de Liquidez de Financiamiento
- **Definición**: Imposibilidad de obtener financiamiento
- **Medición**: Funding costs, margin requirements
- **Control**: Cash management, margin monitoring

## 3. Medición y Monitoreo de Riesgo

### 3.1 Métricas de Riesgo Principales

#### 3.1.1 Value at Risk (VaR)
```python
# VaR 95% y 99% diario
var_95 = np.percentile(returns, 5)
var_99 = np.percentile(returns, 1)

# VaR paramétrico
var_parametric = -1.645 * volatility * np.sqrt(1/252)  # 95%
```

#### 3.1.2 Conditional Value at Risk (CVaR)
```python
# CVaR 95%
cvar_95 = returns[returns <= var_95].mean()
```

#### 3.1.3 Maximum Drawdown
```python
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

#### 3.1.4 Volatility Targeting
```python
def volatility_targeting(returns, target_vol=0.15):
    realized_vol = returns.std() * np.sqrt(252)
    scaling_factor = target_vol / realized_vol
    return scaling_factor
```

### 3.2 Monitoreo en Tiempo Real

#### 3.2.1 Dashboard de Riesgo
- P&L en tiempo real
- Drawdown actual vs máximo
- VaR actual vs límites
- Correlaciones en tiempo real
- Exposiciones por activo/factor

#### 3.2.2 Alertas Automáticas
- Exceso de drawdown (80% del límite)
- Spikes de volatilidad (>2x promedio)
- Breakdown de correlaciones
- Exceso de exposición por activo
- Errores de datos o sistema

## 4. Controles de Riesgo

### 4.1 Position Sizing

#### 4.1.1 Volatility Targeting
```python
def calculate_position_size(capital, volatility, target_vol=0.15):
    """
    Calcula el tamaño de posición basado en volatility targeting
    """
    position_value = capital * (target_vol / volatility)
    return min(position_value, capital * 0.05)  # Max 5% por trade
```

#### 4.1.2 Kelly Criterion
```python
def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Calcula el tamaño óptimo de posición usando Kelly Criterion
    """
    if avg_loss == 0:
        return 0
    
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return max(0, min(kelly_fraction, 0.25))  # Cap at 25%
```

### 4.2 Stop Losses

#### 4.2.1 ATR-based Stops
```python
def atr_stop_loss(entry_price, atr, multiplier=2.0, direction='long'):
    """
    Calcula stop loss basado en ATR
    """
    if direction == 'long':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)
```

#### 4.2.2 Time-based Stops
```python
def time_based_stop(entry_time, max_holding_days=5):
    """
    Stop loss basado en tiempo
    """
    return entry_time + pd.Timedelta(days=max_holding_days)
```

### 4.3 Portfolio Limits

#### 4.3.1 Diversificación
```python
def check_diversification_limits(positions, max_per_asset=0.20):
    """
    Verifica límites de diversificación
    """
    total_exposure = sum(abs(pos) for pos in positions.values())
    
    for asset, position in positions.items():
        exposure_pct = abs(position) / total_exposure
        if exposure_pct > max_per_asset:
            return False, f"Exposure to {asset} exceeds {max_per_asset*100}%"
    
    return True, "Diversification limits OK"
```

#### 4.3.2 Factor Exposure
```python
def check_factor_exposure(factor_exposures, max_per_factor=0.30):
    """
    Verifica límites de exposición por factor
    """
    for factor, exposure in factor_exposures.items():
        if abs(exposure) > max_per_factor:
            return False, f"Factor {factor} exposure exceeds {max_per_factor*100}%"
    
    return True, "Factor exposure limits OK"
```

## 5. Circuit Breakers y Kill Switch

### 5.1 Circuit Breakers Automáticos

#### 5.1.1 Daily Loss Limit
```python
def check_daily_loss_limit(daily_pnl, capital, max_daily_loss=0.02):
    """
    Verifica límite de pérdida diaria
    """
    loss_pct = abs(daily_pnl) / capital if daily_pnl < 0 else 0
    
    if loss_pct > max_daily_loss:
        return True, f"Daily loss limit exceeded: {loss_pct:.2%}"
    
    return False, "Daily loss limit OK"
```

#### 5.1.2 Maximum Drawdown
```python
def check_max_drawdown(current_drawdown, max_drawdown=0.10):
    """
    Verifica límite de drawdown máximo
    """
    if current_drawdown > max_drawdown:
        return True, f"Maximum drawdown exceeded: {current_drawdown:.2%}"
    
    return False, "Maximum drawdown OK"
```

### 5.2 Kill Switch Protocol

#### 5.2.1 Activación Automática
- Pérdida diaria > 2%
- Drawdown > 10%
- Error crítico del sistema
- Falta de datos > 30 minutos
- Correlación > 0.9 entre posiciones principales

#### 5.2.2 Acciones del Kill Switch
1. **Inmediato**: Cancelar todas las órdenes pendientes
2. **Liquidación**: Cerrar todas las posiciones abiertas
3. **Notificación**: Alertar a todos los stakeholders
4. **Análisis**: Generar reporte de incidente
5. **Revisión**: Requerir aprobación manual para reactivar

## 6. Stress Testing

### 6.1 Escenarios de Stress

#### 6.1.1 Crisis Financiera (2008)
- Caída de mercados: -50%
- Aumento de volatilidad: +300%
- Correlaciones: +0.8

#### 6.1.2 COVID-19 (2020)
- Caída inicial: -35%
- Recuperación rápida: +50%
- Volatilidad extrema: VIX > 80

#### 6.1.3 Flash Crash (2010)
- Caída rápida: -10% en minutos
- Recuperación inmediata
- Liquidez cero temporal

### 6.2 Monte Carlo Simulation
```python
def monte_carlo_stress_test(returns, n_simulations=10000, days=252):
    """
    Simulación de Monte Carlo para stress testing
    """
    results = []
    
    for _ in range(n_simulations):
        # Generar retornos aleatorios basados en distribución histórica
        random_returns = np.random.choice(returns, size=days, replace=True)
        cumulative_return = (1 + random_returns).prod() - 1
        max_dd = calculate_max_drawdown(random_returns)
        
        results.append({
            'cumulative_return': cumulative_return,
            'max_drawdown': max_dd,
            'sharpe': np.mean(random_returns) / np.std(random_returns) * np.sqrt(252)
        })
    
    return pd.DataFrame(results)
```

## 7. Reportes de Riesgo

### 7.1 Reportes Diarios
- P&L y drawdown
- VaR actual vs límites
- Exposiciones por activo/factor
- Alertas activadas
- Calidad de datos

### 7.2 Reportes Semanales
- Performance vs benchmarks
- Análisis de regímenes
- Stress test results
- Model stability
- Risk-adjusted returns

### 7.3 Reportes Mensuales
- Análisis de riesgo completo
- Backtesting results
- Model performance
- Regulatory compliance
- Recommendations

## 8. Cumplimiento Regulatorio

### 8.1 Requerimientos SEC/FINRA
- Audit trail completo
- Risk management procedures
- Regular reporting
- Model validation
- Stress testing

### 8.2 Documentación Requerida
- Risk management manual
- Model documentation
- Backtesting reports
- Stress test results
- Incident reports

## 9. Continuidad del Negocio

### 9.1 Plan de Contingencia
- Backup de datos
- Sistemas redundantes
- Procedimientos de emergencia
- Comunicación de crisis

### 9.2 Recovery Procedures
- Data recovery
- System restoration
- Position reconciliation
- Performance analysis

---

**Fecha de Creación**: $(date)
**Versión**: 1.0
**Autor**: Sistema de Trading Cuantitativo
**Estado**: Aprobado

