# Fishing Predictor - Sistema de Prediccion de Pesca desde Orilla

Sistema avanzado de prediccion de puntos optimos de pesca desde orilla para la costa sur de Peru (Tacna - Ilo).

**Fecha de actualizacion:** 2026-01-28

## Caracteristicas Principales

### Datos 100% Reales
- **SST Satelital**: NOAA NCEI OISST / Open-Meteo Marine (datos de satelite reales)
- **Condiciones Marinas**: Open-Meteo ERA5 (reanalisis real)
- **Actividad Pesquera**: Global Fishing Watch (datos AIS reales)
- **Zonas Historicas**: IMARPE (Instituto del Mar del Peru)

### Machine Learning Avanzado
- **32 Features Oceanograficos**: SST, frentes termicos, corrientes, olas, upwelling, etc.
- **PCA**: Reduccion de dimensionalidad para analisis de componentes principales
- **Gradient Boosting**: Regresion para prediccion de zonas de pesca
- **Modo Supervisado**: Entrenamiento con datos reales de pesca de GFW

### Visualizacion
- **Linea costera real**: 3,400+ puntos de OpenStreetMap
- **Mapas interactivos**: Folium con capas de SST, corrientes y zonas de pesca
- **Vectores de corriente**: Visualizacion de flujo oceanico

## Estructura del Proyecto

```
fishing_predictor/
├── domain.py                  # Constantes de dominio centralizadas
├── config.py                  # Configuracion del sistema
├── main.py                    # Punto de entrada principal
├── environment.yml            # Entorno conda
│
├── models/
│   ├── features.py            # Extractor de 32 features oceanograficos
│   └── predictor.py           # ML: PCA + Gradient Boosting
│
├── controllers/
│   └── analysis.py            # Controlador de analisis
│
├── views/
│   └── map_view.py            # Generador de mapas Folium
│
├── core/
│   ├── coastline_real.py      # Procesador de costa OSM
│   ├── weather_solunar.py     # Clima y calculos solunares
│   └── marine_data.py         # Datos marinos y frentes termicos
│
├── data/
│   ├── fetchers/
│   │   ├── real_data_only.py  # Fetcher de datos 100% reales
│   │   └── historical_fetcher.py
│   ├── cache/                 # Cache de datos
│   └── historical/            # Datos historicos reales
│
├── scripts/
│   └── download_100_real.py   # Descarga datos reales (requiere GFW API)
│
├── tests/                     # 18 tests unitarios
│   └── test_models.py
│
├── docs/
│   └── oceanographic_data_sources.md
│
└── output/                    # Mapas y reportes generados
```

## Instalacion

### 1. Crear entorno conda

```bash
conda env create -f environment.yml
conda activate fishing_predictor
```

### 2. Obtener API Key de Global Fishing Watch (Opcional pero recomendado)

Para datos de pesca reales:

1. Ve a: https://globalfishingwatch.org/our-apis/
2. Click en "Request API Access"
3. Completa el formulario (gratis para investigacion)
4. Configura la API key:

```bash
export GFW_API_KEY='tu_api_key_aqui'
```

### 3. Descargar datos reales

```bash
python scripts/download_100_real.py --months 6
```

## Uso

### Analisis Rapido (sin datos historicos)

```bash
python main.py
```

### Analisis con ML Supervisado (requiere datos historicos)

```bash
python main.py --supervised
```

### Notebook Interactivo

```bash
jupyter lab fishing_analysis.ipynb
```

## Fuentes de Datos Reales

| Tipo de Dato | Fuente | Descripcion |
|--------------|--------|-------------|
| SST | NOAA NCEI OISST / Open-Meteo | Temperatura superficial del mar por satelite |
| Olas/Viento | Open-Meteo ERA5 | Reanalisis ECMWF (datos reales + modelo) |
| Corrientes | Open-Meteo Marine / HYCOM | Velocidad y direccion |
| Pesca | Global Fishing Watch | Actividad pesquera AIS (requiere API key) |
| Zonas Historicas | IMARPE | Reportes del Instituto del Mar del Peru |

## Especies Objetivo (5)

| Especie | Temp Optima | Sustrato | Senuelos |
|---------|-------------|----------|----------|
| Cabrilla | 16-19°C | Roca | Grubs 3", jigs 15-25g |
| Corvina | 15-18°C | Arena/Mixto | Jigs metalicos 30-50g |
| Robalo | 17-21°C | Roca/Mixto | Poppers 12cm, minnows |
| Lenguado | 14-17°C | Arena | Vinilos paddle tail |
| Pejerrey | 14-18°C | Arena/Mixto | Cucharillas pequenas |

## Features del Modelo ML (32 total)

### SST (6 features)
- Temperatura, anomalia, score optimo, score por especie, variabilidad, tendencia

### Frentes Termicos (5 features)
- Magnitud del gradiente, direccion (sin/cos), deteccion de frente, intensidad

### Corrientes (6 features)
- Velocidad, componentes U/V, convergencia, cizalladura, direccion a costa

### Olas (3 features)
- Altura, periodo, indice favorable

### Upwelling (3 features)
- Indice de surgencia, transporte Ekman, indice favorable

### Espacial (4 features)
- Distancia a costa, profundidad, zona costera, zona offshore

### Historico (2 features)
- Distancia a hotspots conocidos, similitud con hotspots

### Temporal (3 features)
- Score por hora (alba/ocaso), fase lunar, estacionalidad

## Hotspots Verificados (18 ubicaciones)

Basados en datos de IMARPE y pescadores locales:

| Top 5 | Ubicacion | Bonus | Caracteristica |
|-------|-----------|-------|----------------|
| 1 | Punta Coles | 1.35 | Reserva, mucha vida marina |
| 2 | Punta Blanca | 1.30 | Punta rocosa, robalo |
| 3 | Pozo Redondo | 1.25 | Pozas naturales |
| 4 | Vila Vila | 1.25 | Rocas con estructura |
| 5 | Fundicion | 1.20 | Rocas grandes |

## Ejemplo de Salida

```
============================================================
FISHING PREDICTOR - Analisis Completo
============================================================
Region: Tacna-Ilo, Peru
Fecha: 28/01/2026

[OK] 3,412 puntos de costa cargados
[OK] 288 registros SST reales (Open-Meteo)
[OK] 288 registros marinos reales (ERA5)

TOP 5 MEJORES ZONAS DE PESCA
==============================

#1 - Score: 87.3/100 | Confianza: 92%
    Coordenadas: -17.702, -71.332
    SST: 17.8C | Olas: 1.2m
    Causa: thermal_front + historical_imarpe (Punta Coles)
    Especies: Cabrilla, Robalo

#2 - Score: 82.1/100 | Confianza: 88%
    ...
```

## Tests

```bash
python -m pytest tests/ -v
```

Resultado: 18 tests passed

## Requisitos

- Python 3.11+
- Dependencias principales:
  - numpy, pandas, scipy
  - scikit-learn (ML)
  - folium (mapas)
  - requests (APIs)
  - xarray (datos NetCDF)

## Investigacion Cientifica

El sistema esta basado en investigacion oceanografica del Sistema de la Corriente de Humboldt:

- **IMARPE**: Instituto del Mar del Peru
- **FAO**: Reportes de pesca para Peru
- **Chavez et al. (2008)**: Climate and fisheries in the Humboldt Current
- **Belkin & O'Reilly (2009)**: Thermal front detection algorithm
- **pyBOA**: Algoritmo Belkin-O'Reilly para deteccion de frentes termicos

## Arquitectura

El proyecto sigue el patron MVC:

- **Model** (`models/`): Logica de ML y extraccion de features
- **View** (`views/`): Renderizado de mapas Folium
- **Controller** (`controllers/`): Orquestacion del flujo de datos

Las constantes de dominio estan centralizadas en `domain.py` usando estructuras inmutables (NamedTuple, FrozenSet) para eficiencia y seguridad de tipos.

## Documentacion Adicional

| Documento | Descripcion |
|-----------|-------------|
| [`DEVELOPMENT_GUIDELINES.md`](DEVELOPMENT_GUIDELINES.md) | Lineamientos tecnicos, buenas practicas de codigo, estructuras de datos |
| [`docs/oceanographic_data_sources.md`](docs/oceanographic_data_sources.md) | Fuentes de datos oceanograficos y APIs |
| [`fishing_predictor_spec.md`](../fishing_predictor_spec.md) | Especificacion tecnica completa del sistema |
| [`investigacion.md`](investigacion.md) | Investigacion cientifica y metodologia |

## Licencia

Desarrollado para pesca con spinning desde orilla en la costa sur de Peru.

---

*Ultima actualizacion: 2026-01-28*
