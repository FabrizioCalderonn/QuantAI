# Diagrama de Arquitectura del Sistema de Trading

## Diagrama de Flujo Principal

```mermaid
graph TB
    subgraph "Data Sources"
        A1[Yahoo Finance API]
        A2[Alpha Vantage API]
        A3[IEX Cloud API]
        A4[News/Sentiment APIs]
    end
    
    subgraph "Data Pipeline"
        B1[Data Ingestion]
        B2[Data Cleaning]
        B3[Data Validation]
        B4[Time Zone Sync]
        B5[Data Storage]
    end
    
    subgraph "Feature Engineering"
        C1[Technical Indicators]
        C2[Statistical Features]
        C3[Cross-Asset Features]
        C4[Regime Detection]
        C5[Feature Store]
    end
    
    subgraph "Model Layer"
        D1[Baseline Models]
        D2[ML Models]
        D3[Ensemble Model]
        D4[Risk Adjustment]
    end
    
    subgraph "Risk Management"
        E1[Position Sizing]
        E2[Stop Loss/Take Profit]
        E3[Portfolio Limits]
        E4[Kill Switch]
    end
    
    subgraph "Execution Engine"
        F1[Signal Generation]
        F2[Order Management]
        F3[Cost Modeling]
        F4[Performance Tracking]
    end
    
    subgraph "Monitoring System"
        G1[Real-time Dashboard]
        G2[Alert System]
        G3[Logging & Audit]
        G4[Reporting]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    
    B5 --> C1
    B5 --> C2
    B5 --> C3
    B5 --> C4
    
    C1 --> C5
    C2 --> C5
    C3 --> C5
    C4 --> C5
    
    C5 --> D1
    C5 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> D4
    
    D4 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    
    E4 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    
    F4 --> G1
    F1 --> G2
    F2 --> G3
    G1 --> G4
    G2 --> G4
    G3 --> G4
```

## Diagrama de Flujo de Datos Temporal

```mermaid
sequenceDiagram
    participant DS as Data Sources
    participant DP as Data Pipeline
    participant FS as Feature Store
    participant ML as Model Layer
    participant RM as Risk Management
    participant EE as Execution Engine
    participant MS as Monitoring System
    
    loop Cada Barra de Tiempo
        DS->>DP: Ingesta de Datos
        DP->>DP: Validación y Limpieza
        DP->>FS: Almacenamiento
        FS->>FS: Feature Engineering
        FS->>ML: Features Actualizados
        ML->>ML: Inferencia del Modelo
        ML->>RM: Señales de Trading
        RM->>RM: Evaluación de Riesgo
        alt Señal Aprobada
            RM->>EE: Orden de Trading
            EE->>EE: Ejecución
            EE->>MS: Actualización de Performance
        else Señal Rechazada
            RM->>MS: Log de Rechazo
        end
        MS->>MS: Monitoreo Continuo
    end
```

## Diagrama de Validación y Backtesting

```mermaid
graph LR
    subgraph "Data Split"
        A1[Train Set<br/>60%]
        A2[Validation Set<br/>20%]
        A3[Test Set<br/>20%]
    end
    
    subgraph "Walk-Forward Analysis"
        B1[Window 1]
        B2[Window 2]
        B3[Window N]
    end
    
    subgraph "Cross-Validation"
        C1[Fold 1]
        C2[Fold 2]
        C3[Fold K]
    end
    
    subgraph "Out-of-Sample Testing"
        D1[Final Validation]
        D2[Regime Analysis]
        D3[Stress Testing]
    end
    
    A1 --> B1
    A1 --> B2
    A1 --> B3
    
    B1 --> C1
    B1 --> C2
    B1 --> C3
    
    C1 --> D1
    C2 --> D1
    C3 --> D1
    
    D1 --> D2
    D1 --> D3
```

## Diagrama de Gestión de Riesgo

```mermaid
graph TD
    A[Señal de Trading] --> B{Evaluación de Riesgo}
    
    B --> C[Position Sizing]
    C --> D[Volatility Targeting]
    D --> E[Portfolio Limits]
    
    E --> F{Verificación de Límites}
    F -->|Aprobado| G[Generación de Orden]
    F -->|Rechazado| H[Log de Rechazo]
    
    G --> I[Stop Loss]
    G --> J[Take Profit]
    G --> K[Time-based Exit]
    
    I --> L[Ejecución]
    J --> L
    K --> L
    
    L --> M[Performance Tracking]
    M --> N{Monitoreo Continuo}
    
    N -->|Normal| O[Continuar]
    N -->|Alerta| P[Notificación]
    N -->|Crítico| Q[Kill Switch]
    
    Q --> R[Liquidación de Posiciones]
    P --> S[Revisión Manual]
```

## Diagrama de Monitoreo en Tiempo Real

```mermaid
graph TB
    subgraph "Métricas en Tiempo Real"
        A1[P&L Diario]
        A2[Drawdown Actual]
        A3[Volatilidad Realizada]
        A4[Sharpe Rolling]
        A5[Hit Ratio]
    end
    
    subgraph "Alertas"
        B1[Max Drawdown Exceeded]
        B2[Volatility Spike]
        B3[Correlation Breakdown]
        B4[Data Drift]
        B5[System Error]
    end
    
    subgraph "Dashboard"
        C1[Performance Chart]
        C2[Risk Metrics]
        C3[Position Summary]
        C4[Trade Log]
        C5[System Status]
    end
    
    A1 --> C1
    A2 --> C2
    A3 --> C2
    A4 --> C1
    A5 --> C1
    
    B1 --> C5
    B2 --> C5
    B3 --> C5
    B4 --> C5
    B5 --> C5
    
    C1 --> C4
    C2 --> C4
    C3 --> C4
    C4 --> C5
```

## Diagrama de Arquitectura Técnica

```mermaid
graph TB
    subgraph "Frontend Layer"
        F1[Web Dashboard]
        F2[Mobile App]
        F3[API Gateway]
    end
    
    subgraph "Application Layer"
        A1[Trading Engine]
        A2[Risk Engine]
        A3[Signal Engine]
        A4[Monitoring Service]
    end
    
    subgraph "Data Layer"
        D1[Time Series DB]
        D2[Feature Store]
        D3[Model Registry]
        D4[Configuration Store]
    end
    
    subgraph "Infrastructure Layer"
        I1[Message Queue]
        I2[Cache Layer]
        I3[Log Aggregation]
        I4[Monitoring Stack]
    end
    
    subgraph "External Services"
        E1[Market Data APIs]
        E2[Broker APIs]
        E3[News APIs]
        E4[Cloud Services]
    end
    
    F1 --> F3
    F2 --> F3
    F3 --> A1
    F3 --> A2
    F3 --> A3
    F3 --> A4
    
    A1 --> D1
    A1 --> D2
    A2 --> D1
    A2 --> D4
    A3 --> D2
    A3 --> D3
    A4 --> D1
    A4 --> I3
    
    A1 --> I1
    A2 --> I1
    A3 --> I1
    A4 --> I1
    
    I1 --> I2
    I1 --> I3
    I1 --> I4
    
    A1 --> E2
    A3 --> E1
    A4 --> E3
    I4 --> E4
```

