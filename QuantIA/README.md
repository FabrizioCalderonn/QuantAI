# Sistema de Trading Cuantitativo Institucional-Grade

## Descripción

Sistema de trading algorítmico diseñado para generar retornos consistentes y superiores al mercado con un perfil de riesgo controlado, evitando sobreajuste y sesgos mediante metodologías robustas de validación.

## Características Principales

- **Pipeline de Datos Robusto**: Ingesta, limpieza y validación de datos de múltiples fuentes
- **Feature Engineering Avanzado**: Indicadores técnicos, estadísticos y cross-asset
- **Modelos ML Parsimoniosos**: Baselines, ML models y ensemble con validación walk-forward
- **Gestión de Riesgo Institucional**: Volatility targeting, position sizing y circuit breakers
- **Backtesting Correcto**: Walk-forward analysis, CV purgada y métricas robustas
- **Monitoreo en Tiempo Real**: Dashboard, alertas y kill-switch automático
- **Paper Trading**: Simulación completa antes de trading en vivo

## Estructura del Proyecto

```
QuantIA/
├── docs/                    # Documentación
│   ├── design.md           # Documento de diseño principal
│   ├── architecture_diagram.md  # Diagramas de arquitectura
│   ├── risk.md             # Documentación de gestión de riesgo
│   └── backtest_report.pdf # Reporte de backtesting
├── src/                    # Código fuente
│   ├── data/              # Pipeline de datos
│   ├── features/          # Feature engineering
│   ├── models/            # Modelos ML y baselines
│   ├── risk/              # Gestión de riesgo
│   ├── execution/         # Motor de ejecución
│   ├── monitoring/        # Sistema de monitoreo
│   └── utils/             # Utilidades comunes
├── configs/               # Configuraciones
│   └── default_parameters.yaml
├── data/                  # Datos
│   ├── raw/              # Datos crudos
│   ├── processed/        # Datos procesados
│   └── external/         # Datos externos
├── notebooks/            # Jupyter notebooks para EDA
├── backtests/            # Resultados de backtesting
├── monitoring/           # Dashboard y monitoreo
├── tests/                # Tests unitarios
└── logs/                 # Logs del sistema
```

## Configuración Inicial

### Requisitos

- Python 3.9+
- pip o conda
- Git

### Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd QuantIA
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus API keys
```

## Uso Rápido

### 1. Pipeline de Datos
```python
from src.data.pipeline import DataPipeline

pipeline = DataPipeline()
data = pipeline.run()
```

### 2. Feature Engineering
```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(data)
```

### 3. Entrenamiento de Modelos
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_models(features)
```

### 4. Backtesting
```python
from src.execution.backtester import Backtester

backtester = Backtester()
results = backtester.run_backtest(models)
```

### 5. Paper Trading
```python
from src.execution.paper_trader import PaperTrader

trader = PaperTrader()
trader.start_trading()
```

## Configuración

Los parámetros principales se configuran en `configs/default_parameters.yaml`:

- **Capital inicial**: $1,000,000 USD
- **Apalancamiento máximo**: 3:1
- **Volatilidad objetivo**: 15% anual
- **Drawdown máximo**: 10%
- **Sharpe mínimo**: 1.5

## Métricas de Evaluación

### Métricas de Retorno
- Sharpe Ratio ≥ 1.5
- Sortino Ratio
- Calmar Ratio
- Profit Factor ≥ 1.3

### Métricas de Riesgo
- Maximum Drawdown ≤ 10%
- VaR 95% y 99%
- CVaR (Conditional VaR)
- Tail Risk

### Métricas Operacionales
- Hit Ratio ≥ 45%
- Turnover
- Capacity
- Stability

## Validación y Backtesting

El sistema utiliza metodologías robustas de validación:

1. **División Temporal**: Train (60%) → Validation (20%) → Test (20%)
2. **Walk-Forward Analysis**: Ventanas deslizantes de 12 meses
3. **Cross-Validation Purgada**: 5 folds con 5 días de purga
4. **Out-of-Sample Testing**: Validación final en datos nunca vistos
5. **Regime Analysis**: Performance en diferentes condiciones de mercado

## Gestión de Riesgo

### Controles Automáticos
- **Position Sizing**: Volatility targeting con Kelly criterion
- **Stop Loss**: ATR-based stops dinámicos
- **Portfolio Limits**: Diversificación y concentración
- **Kill Switch**: Circuit breakers automáticos

### Límites de Riesgo
- Exposición máxima por activo: 20%
- Exposición máxima por factor: 30%
- Pérdida diaria máxima: 2%
- Drawdown máximo: 10%

## Monitoreo

### Dashboard en Tiempo Real
- P&L diario y acumulado
- Drawdown actual
- Volatilidad realizada
- Sharpe ratio rolling
- Hit ratio

### Alertas Automáticas
- Exceso de drawdown máximo
- Spikes de volatilidad
- Breakdown de correlaciones
- Data drift
- Errores del sistema

## Paper Trading

El sistema incluye un módulo completo de paper trading:

- Simulación de órdenes con latencia realista
- Modelado de costos (comisiones, slippage)
- Tracking de performance en tiempo real
- Logs detallados de todas las operaciones

## Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Disclaimer

Este software es para fines educativos y de investigación. El trading con dinero real conlleva riesgos significativos. Los desarrolladores no se hacen responsables de pérdidas financieras.

## Contacto

Para preguntas o soporte, por favor abrir un issue en GitHub.

---

**Versión**: 1.0.0
**Última actualización**: $(date)
**Estado**: En desarrollo activo

