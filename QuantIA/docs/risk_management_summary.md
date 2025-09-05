# Resumen de Gestión de Riesgo

## Descripción General

El sistema de gestión de riesgo es un componente crítico que protege el portafolio mediante múltiples capas de control, incluyendo volatility targeting, circuit breakers automáticos, y stops dinámicos. Está diseñado para cumplir con estándares institucionales y regulaciones financieras.

## Arquitectura del Sistema

### Componentes Principales

#### 1. BaseRiskManager
- **Clase Base**: Interfaz común para todos los gestores de riesgo
- **Funcionalidades**: Métricas de riesgo, validación, alertas, posiciones
- **Métricas**: VaR, CVaR, Sharpe, Sortino, Calmar, Beta, Correlación
- **Alertas**: Sistema de alertas con niveles de severidad

#### 2. VolatilityTargetingRiskManager
- **Estrategia**: Mantener volatilidad objetivo del portafolio
- **Características**: Ajuste dinámico de posiciones, rebalance automático
- **Métodos**: EWMA, GARCH, Simple rolling volatility
- **Límites**: Concentración, leverage, volatilidad máxima

#### 3. CircuitBreakerRiskManager
- **Estrategia**: Circuit breakers automáticos para protección
- **Tipos**: Precio, volatilidad, volumen, drawdown, correlación, liquidez
- **Acciones**: Halt, reduce, alert
- **Estados**: Normal, warning, triggered, halted

#### 4. RiskManager (Principal)
- **Coordinador**: Orquesta todos los componentes de riesgo
- **Funcionalidades**: Actualización de portafolio, ajuste de posiciones
- **Integración**: Combina volatility targeting y circuit breakers
- **Monitoreo**: Alertas en tiempo real, métricas consolidadas

## Componentes Implementados

### 1. VolatilityTargetingRiskManager

#### Configuración por Defecto
```yaml
target_volatility: 0.15  # 15% anual
volatility_window: 20    # 20 períodos
rebalance_frequency: 1   # 1 período
max_position_size: 0.1   # 10% del portafolio
max_portfolio_volatility: 0.20  # 20% anual
max_drawdown: 0.10       # 10% máximo drawdown
max_leverage: 2.0        # 2x leverage máximo
max_concentration: 0.3   # 30% máximo concentración
stop_loss_pct: 0.05      # 5% stop loss
take_profit_pct: 0.15    # 15% take profit
volatility_floor: 0.05   # 5% volatilidad mínima
volatility_ceiling: 0.30 # 30% volatilidad máxima
```

#### Características Técnicas
- **Volatility Targeting**: Ajusta posiciones para mantener volatilidad objetivo
- **Múltiples Métodos**: EWMA, GARCH, Simple rolling
- **Rebalance Automático**: Rebalancea portafolio según frecuencia configurada
- **Límites Dinámicos**: Ajusta límites según condiciones de mercado
- **Feature Selection**: Selecciona features relevantes para volatilidad

#### Proceso de Volatility Targeting
1. **Estimación de Volatilidad**: Calcula volatilidad de cada instrumento
2. **Ajuste de Posiciones**: Ajusta tamaño basado en volatilidad objetivo
3. **Verificación de Límites**: Verifica límites de concentración y leverage
4. **Rebalance**: Rebalancea portafolio para mantener objetivo
5. **Monitoreo**: Monitorea desviaciones y ajusta dinámicamente

### 2. CircuitBreakerRiskManager

#### Configuración por Defecto
```yaml
circuit_breakers:
  price_drop:
    threshold: 0.05      # 5% caída de precio
    duration: 15         # 15 minutos
    action: 'halt'       # Detener trading
    severity: 'high'
  volatility_spike:
    threshold: 0.30      # 30% volatilidad
    duration: 30         # 30 minutos
    action: 'reduce'     # Reducir posiciones
    severity: 'medium'
  volume_drop:
    threshold: 0.5       # 50% caída de volumen
    duration: 10         # 10 minutos
    action: 'alert'      # Solo alertar
    severity: 'low'
  drawdown_limit:
    threshold: 0.08      # 8% drawdown
    duration: 60         # 60 minutos
    action: 'halt'       # Detener trading
    severity: 'critical'
  correlation_spike:
    threshold: 0.9       # 90% correlación
    duration: 20         # 20 minutos
    action: 'reduce'     # Reducir posiciones
    severity: 'medium'
  liquidity_crisis:
    threshold: 0.8       # 80% riesgo de liquidez
    duration: 45         # 45 minutos
    action: 'halt'       # Detener trading
    severity: 'high'
```

#### Características Técnicas
- **Múltiples Tipos**: Precio, volatilidad, volumen, drawdown, correlación, liquidez
- **Acciones Automáticas**: Halt, reduce, alert
- **Estados Dinámicos**: Normal, warning, triggered, halted
- **Cooldown**: Período de enfriamiento después de activación
- **Historial**: Mantiene historial de activaciones

#### Proceso de Circuit Breakers
1. **Monitoreo Continuo**: Monitorea condiciones en tiempo real
2. **Detección de Breach**: Detecta cuando se exceden umbrales
3. **Activación**: Activa circuit breaker según configuración
4. **Ejecución de Acción**: Ejecuta acción (halt/reduce/alert)
5. **Cooldown**: Establece período de enfriamiento
6. **Recuperación**: Vuelve a normal después de duración configurada

### 3. RiskManager (Principal)

#### Configuración por Defecto
```yaml
risk_management:
  update_frequency: 60        # segundos
  alert_threshold: 0.8        # 80% de límite
  auto_rebalance: true        # Rebalance automático
  emergency_stop: true        # Parada de emergencia
  volatility_targeting: {...} # Configuración VT
  circuit_breakers: {...}     # Configuración CB
```

#### Características Técnicas
- **Coordinación**: Coordina todos los componentes de riesgo
- **Actualización en Tiempo Real**: Actualiza portafolio continuamente
- **Ajuste de Posiciones**: Ajusta tamaño considerando todos los factores
- **Alertas Consolidadas**: Consolida alertas de todos los componentes
- **Estado Persistente**: Guarda y carga estado del sistema

#### Proceso Principal
1. **Actualización de Portafolio**: Actualiza posiciones y precios
2. **Cálculo de Métricas**: Calcula métricas de riesgo consolidadas
3. **Verificación de Límites**: Verifica límites de todos los componentes
4. **Ajuste de Posiciones**: Ajusta tamaño considerando todos los factores
5. **Generación de Alertas**: Genera alertas consolidadas
6. **Persistencia**: Guarda estado para recuperación

## Métricas de Riesgo

### Métricas Básicas
- **Volatilidad**: Volatilidad anualizada del portafolio
- **VaR (95%, 99%)**: Value at Risk en diferentes niveles de confianza
- **CVaR (95%, 99%)**: Conditional Value at Risk
- **Maximum Drawdown**: Pérdida máxima desde peak
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Sortino Ratio**: Retorno ajustado por downside risk
- **Calmar Ratio**: Retorno anual / Maximum drawdown

### Métricas Avanzadas
- **Beta**: Sensibilidad a movimientos del mercado
- **Correlación**: Correlación entre instrumentos
- **Concentración**: Índice de Herfindahl del portafolio
- **Leverage**: Ratio de exposición / capital
- **Liquidez**: Riesgo de liquidez promedio

### Métricas de Circuit Breakers
- **Trigger Count**: Número de activaciones por breaker
- **Status**: Estado actual de cada breaker
- **Cooldown**: Tiempo restante de enfriamiento
- **Duration**: Duración de activación
- **Action**: Acción tomada cuando se activa

## Sistema de Alertas

### Tipos de Alertas
- **MARKET**: Riesgos de mercado (volatilidad, drawdown)
- **LIQUIDITY**: Riesgos de liquidez
- **CONCENTRATION**: Riesgos de concentración
- **LEVERAGE**: Riesgos de leverage
- **OPERATIONAL**: Riesgos operacionales

### Niveles de Severidad
- **LOW**: Informacional, no requiere acción
- **MEDIUM**: Requiere atención, acción opcional
- **HIGH**: Requiere acción inmediata
- **CRITICAL**: Requiere acción de emergencia

### Estructura de Alerta
```python
@dataclass
class RiskAlert:
    risk_type: RiskType
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    symbol: str = None
    action_required: bool = False
    metadata: Dict[str, Any] = None
```

## Gestión de Posiciones

### Estructura de Posición
```python
@dataclass
class Position:
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
```

### Ajuste de Posiciones
1. **Volatility Targeting**: Ajusta basado en volatilidad objetivo
2. **Circuit Breakers**: Reduce o detiene según activaciones
3. **Límites Globales**: Aplica límites de concentración y leverage
4. **Stops y Takes**: Implementa stops dinámicos y takes
5. **Rebalance**: Rebalancea para mantener pesos objetivo

## Límites de Riesgo

### Límites por Defecto
- **Max Position Size**: 10% del portafolio
- **Max Portfolio Volatility**: 20% anual
- **Max Drawdown**: 10%
- **Max Leverage**: 2.0x
- **Max Concentration**: 30% (índice de Herfindahl)
- **Max Correlation**: 70% entre instrumentos
- **VaR Limit (95%)**: -5%
- **VaR Limit (99%)**: -10%

### Límites Dinámicos
- **Volatility Floor/Ceiling**: 5% - 30%
- **Position Sizing**: 1% - 15% del portafolio
- **Correlation Threshold**: 70% para reducción
- **Liquidity Threshold**: 10% para alertas

## Uso del Sistema

### Inicialización
```python
from src.risk.risk_manager import RiskManager

# Crear gestor de riesgo
risk_manager = RiskManager("configs/default_parameters.yaml")

# Actualizar portafolio
risk_summary = risk_manager.update_portfolio(positions, current_prices, returns)

# Verificar límites
alerts = risk_manager.check_risk_limits(returns)

# Ajustar posición
position_size = risk_manager.adjust_position_size(signal, symbol, current_price)
```

### Monitoreo
```python
# Obtener resumen de riesgo
summary = risk_manager.get_risk_summary()

# Obtener alertas
alerts = risk_manager.get_alerts(risk_type=RiskType.MARKET)

# Obtener métricas
metrics = risk_manager.get_risk_metrics()
```

### Gestión de Posiciones
```python
# Agregar posición
risk_manager.add_position(position)

# Remover posición
risk_manager.remove_position(symbol)

# Rebalancear portafolio
rebalanced_positions = risk_manager.rebalance_portfolio(target_weights, current_prices)
```

## Comandos de Ejecución

```bash
# Solo gestión de riesgo
python main.py --risk

# Pipeline completo (incluye gestión de riesgo)
python main.py --pipeline

# Demo con gestión de riesgo
python main.py --demo
```

## Almacenamiento

### Estructura de Archivos
```
risk/
├── base.py                    # Clase base y estructuras
├── volatility_targeting.py    # Volatility targeting
├── circuit_breakers.py        # Circuit breakers
└── risk_manager.py           # Gestor principal
```

### Persistencia de Estado
```python
# Guardar estado
risk_manager.save_state("risk_state.pkl")

# Cargar estado
risk_manager.load_state("risk_state.pkl")
```

## Tests

### Cobertura de Tests
- **BaseRiskManager**: 5 tests
- **VolatilityTargetingRiskManager**: 8 tests
- **CircuitBreakerRiskManager**: 10 tests
- **RiskManager**: 8 tests
- **RiskMetrics**: 1 test

### Ejecutar Tests
```bash
pytest tests/test_risk_management.py -v
```

## Ventajas del Sistema

### 1. Protección Integral
- **Múltiples Capas**: Volatility targeting + circuit breakers
- **Tiempo Real**: Monitoreo y ajuste continuo
- **Automático**: Acciones automáticas sin intervención manual
- **Robusto**: Maneja múltiples tipos de riesgo

### 2. Cumplimiento Regulatorio
- **Transparencia**: Lógica clara y auditables
- **Trazabilidad**: Historial completo de decisiones
- **Documentación**: Documentación completa de procesos
- **Validación**: Validación automática de límites

### 3. Flexibilidad
- **Configurable**: Parámetros ajustables por YAML
- **Extensible**: Fácil agregar nuevos tipos de riesgo
- **Modular**: Componentes independientes
- **Escalable**: Maneja portafolios de cualquier tamaño

### 4. Performance
- **Eficiente**: Cálculos optimizados
- **Tiempo Real**: Actualización en tiempo real
- **Paralelo**: Procesamiento paralelo cuando es posible
- **Memoria**: Gestión eficiente de memoria

## Limitaciones

### 1. Complejidad
- **Configuración**: Requiere configuración cuidadosa
- **Mantenimiento**: Requiere mantenimiento regular
- **Expertise**: Requiere conocimiento de riesgo
- **Dependencias**: Depende de datos de mercado

### 2. Riesgos
- **Over-Engineering**: Puede ser demasiado complejo
- **False Positives**: Alertas falsas
- **Lag**: Retraso en detección de riesgos
- **Correlación**: Riesgos correlacionados pueden no detectarse

### 3. Recursos
- **Computación**: Requiere recursos computacionales
- **Datos**: Requiere datos de mercado en tiempo real
- **Personal**: Requiere personal especializado
- **Infraestructura**: Requiere infraestructura robusta

## Casos de Uso

### 1. Trading Institucional
- **Hedge Funds**: Gestión de riesgo para fondos
- **Asset Managers**: Gestión de portafolios institucionales
- **Prop Trading**: Trading propietario
- **Family Offices**: Gestión de patrimonio familiar

### 2. Trading Retail
- **Robo-Advisors**: Asesores automatizados
- **Trading Platforms**: Plataformas de trading
- **Investment Apps**: Aplicaciones de inversión
- **Educational**: Herramientas educativas

### 3. Investigación
- **Academia**: Investigación académica
- **Consulting**: Consultoría financiera
- **Regulators**: Reguladores financieros
- **Auditors**: Auditores de riesgo

## Próximos Pasos

1. **Backtesting**: Implementar backtesting con gestión de riesgo
2. **Métricas**: Desarrollar métricas adicionales
3. **Explainability**: Implementar explainability de decisiones
4. **Paper Trading**: Integrar con paper trading
5. **Producción**: Plan de paso a producción

---

**Total de Componentes**: 4 (Base, Volatility Targeting, Circuit Breakers, Principal)
**Métricas de Riesgo**: 15+ métricas
**Circuit Breakers**: 6 tipos diferentes
**Tests**: 32 tests unitarios
**Configuración**: YAML completamente configurable
**Persistencia**: Guardado/carga de estado
**Monitoreo**: Tiempo real con alertas

