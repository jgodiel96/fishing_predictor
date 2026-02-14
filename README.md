# Fishing Predictor - Sistema de Prediccion de Pesca desde Orilla

Sistema avanzado de prediccion de puntos optimos de pesca desde orilla para la costa sur de Peru (Tacna - Ilo).

**Fecha de actualizacion:** 2026-02-13

## Diagrama de Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FUENTES DE DATOS EXTERNAS                              │
├─────────────────────┬─────────────────────┬─────────────────────┬───────────────┤
│   COPERNICUS        │    OPEN-METEO       │  GLOBAL FISHING     │   IMARPE      │
│   Marine Service    │    ERA5 API         │  WATCH (GFW)        │   Historico   │
├─────────────────────┼─────────────────────┼─────────────────────┼───────────────┤
│ • SST (OSTIA)       │ • Olas (altura,     │ • Actividad AIS     │ • Hotspots    │
│ • Corrientes (uo,vo)│   periodo, dir)     │ • Horas de pesca    │ • Zonas       │
│ • Olas (VHM0,VTPK)  │ • Viento (u,v)      │ • Tipo de pesca     │   verificadas │
│ • Clorofila-a       │ • SST alternativa   │                     │               │
└──────────┬──────────┴──────────┬──────────┴──────────┬──────────┴───────────────┘
           │                     │                     │
           ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CAPA BRONZE (data/raw/) - INMUTABLE                     │
├─────────────────────┬─────────────────────┬─────────────────────────────────────┤
│ sst/copernicus/     │ open_meteo/         │ gfw/                                │
│ currents/           │ YYYY-MM.parquet     │ YYYY-MM.parquet                     │
│ waves/              │ _manifest.json      │ _manifest.json                      │
│ chlorophyll/        │                     │                                     │
│ YYYY-MM.parquet     │                     │                                     │
└──────────┬──────────┴──────────┬──────────┴──────────┬──────────────────────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       CAPA SILVER (data/processed/) - REGENERABLE                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  fishing_consolidated.db    marine_consolidated.db    training_features.parquet │
│        (1,085 reg)              (572,793 reg)              (213,378 reg)        │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 CORE LAYER                                       │
├───────────────────────────────┬─────────────────────────────────────────────────┤
│   copernicus_data_provider.py │   marine_data.py                                │
│   ┌─────────────────────┐     │   ┌─────────────────────────────────────────┐   │
│   │ CopernicusDataProvider│   │   │ MarineDataFetcher                       │   │
│   │ • load_sst()        │     │   │ • fetch_from_copernicus() [PRINCIPAL]   │   │
│   │ • load_currents()   │     │   │ • fetch_from_api()        [FALLBACK]    │   │
│   │ • load_waves()      │     │   └─────────────────────────────────────────┘   │
│   │ • load_chlorophyll()│     │   ┌─────────────────────────────────────────┐   │
│   │ • get_data_for_date()│    │   │ ThermalFrontDetector                    │   │
│   └─────────────────────┘     │   │ • detect_fronts()                       │   │
│                               │   │ • calculate_gradients()                 │   │
├───────────────────────────────┴───┴─────────────────────────────────────────────┤
│   coastline_real.py           │   weather_solunar.py                            │
│   • CoastlineProcessor        │   • SolunarCalculator                           │
│   • 7,741 pts OSM             │   • Fases lunares, alba/ocaso                   │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                MODELS LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│   features.py                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ FeatureExtractor - 32 Features Oceanograficos                             │ │
│   │ ┌──────────────────┬──────────────────┬──────────────────┬──────────────┐ │ │
│   │ │ SST (6)          │ Frentes (5)      │ Corrientes (6)   │ Olas (3)     │ │ │
│   │ │ • temperatura    │ • gradiente      │ • velocidad      │ • altura     │ │ │
│   │ │ • anomalia       │ • direccion      │ • componentes    │ • periodo    │ │ │
│   │ │ • score optimo   │ • is_front       │ • convergencia   │ • favorable  │ │ │
│   │ │ • score especie  │ • intensidad     │ • cizalladura    │              │ │ │
│   │ ├──────────────────┼──────────────────┼──────────────────┼──────────────┤ │ │
│   │ │ Upwelling (3)    │ Espacial (4)     │ Historico (2)    │ Temporal (3) │ │ │
│   │ │ • indice         │ • dist_costa     │ • dist_hotspot   │ • hora       │ │ │
│   │ │ • ekman          │ • profundidad    │ • similitud      │ • luna       │ │ │
│   │ │ • favorable      │ • zona costera   │                  │ • estacion   │ │ │
│   │ └──────────────────┴──────────────────┴──────────────────┴──────────────┘ │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│   predictor.py                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ FishingPredictor - Pipeline ML                                            │ │
│   │                                                                           │ │
│   │   32 Features ──► StandardScaler ──► PCA (8 comp) ──► KMeans (6 clusters) │ │
│   │                                            │                              │ │
│   │                                            ▼                              │ │
│   │                                   GradientBoosting ──► Score (0-100)      │ │
│   │                                                                           │ │
│   │   Modo Supervisado: Entrena con datos GFW historicos                      │ │
│   │   Modo No-Supervisado: Clustering + Domain Knowledge                      │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CONTROLLER LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│   controllers/analysis.py                                                        │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ AnalysisController                                                        │ │
│   │ • run_full_analysis()                                                     │ │
│   │   1. Cargar linea costera (OSM)                                           │ │
│   │   2. Fetch datos marinos (Copernicus/Open-Meteo)                          │ │
│   │   3. Detectar frentes termicos                                            │ │
│   │   4. Extraer 32 features                                                  │ │
│   │   5. Entrenar/Predecir con ML                                             │ │
│   │   6. Filtrar por proximidad (opcional)                                    │ │
│   │   7. Generar mapa interactivo                                             │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 VIEW LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│   views/map_view.py                                                              │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ MapGenerator (Folium)                                                     │ │
│   │ • Capa base: OpenStreetMap / CartoDB                                      │ │
│   │ • Capa SST: Heatmap de temperatura                                        │ │
│   │ • Capa Corrientes: Vectores de flujo                                      │ │
│   │ • Capa Pesca: Marcadores con score/cluster                                │ │
│   │ • Capa Proximidad: Circulo de busqueda + ubicacion usuario                │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   OUTPUT                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│   output/fishing_analysis_ml.html                                                │
│   • Mapa interactivo con todas las capas                                         │
│   • Top 10 zonas de pesca recomendadas                                           │
│   • Estadisticas de scores y clusters                                            │
│   • Feature importance del modelo                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Flujo de Datos

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│   COPERNICUS   │    │   OPEN-METEO   │    │     GFW        │
│   (Principal)  │    │   (Fallback)   │    │  (Training)    │
└───────┬────────┘    └───────┬────────┘    └───────┬────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────────────────────────────────────────────────────┐
│                    Datos Parquet (Bronze)                      │
│            data/raw/{sst,currents,waves,chlorophyll}/          │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              CopernicusDataProvider.get_data_for_date()        │
│                  Fusion: SST + Corrientes + Olas               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│             FeatureExtractor.extract_from_marine_points()      │
│                      32 Features Oceanograficos                │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              FishingPredictor.fit_unsupervised() / predict()   │
│              PCA ──► KMeans ──► GradientBoosting ──► Score     │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  MapGenerator.generate_map()                   │
│                    Mapa HTML Interactivo                       │
└───────────────────────────────────────────────────────────────┘
```

## Caracteristicas Principales

### Datos 100% Reales (6 anos: 2020-2026)
- **SST Satelital**: Copernicus Marine OSTIA (354,362 registros)
- **Condiciones Marinas**: Open-Meteo ERA5 (216,354 registros)
- **Actividad Pesquera**: Global Fishing Watch AIS (1,085 registros)
- **Zonas Historicas**: IMARPE (Instituto del Mar del Peru)

### Arquitectura de Datos Bronze/Silver/Gold
- **Bronze (raw/)**: Datos crudos inmutables, particionados por mes
- **Silver (processed/)**: Datos consolidados y validados
- **Gold (analytics/)**: Datasets listos para ML

### Machine Learning Avanzado
- **32 Features Oceanograficos**: SST, frentes termicos, corrientes, olas, upwelling, etc.
- **PCA**: Reduccion de dimensionalidad para analisis de componentes principales
- **Gradient Boosting**: Regresion para prediccion de zonas de pesca
- **Modo Supervisado**: Entrenamiento con datos reales de pesca de GFW

### Busqueda por Proximidad
- **Buscar cerca de ti**: Especifica tu ubicacion (lat/lon) y radio de busqueda
- **Resultados ordenados**: Por score de pesca y distancia a tu ubicacion
- **Vista dual**: Muestra tanto spots cercanos como mejores spots globales
- **Visualizacion en mapa**: Marcador de ubicacion y circulo de radio de busqueda

### Visualizacion
- **Linea costera real**: 7,741 puntos de OpenStreetMap (coastline v8)
- **Mapas interactivos**: Folium con capas de SST, corrientes y zonas de pesca
- **Vectores de corriente**: Visualizacion de flujo oceanico

## Estructura del Proyecto

```
fishing_predictor/
├── domain.py                  # Constantes de dominio centralizadas
├── config.py                  # Configuracion del sistema
├── main.py                    # Punto de entrada principal (--copernicus flag)
├── environment.yml            # Entorno conda
│
├── data/
│   ├── data_config.py         # Configuracion centralizada de paths
│   ├── manifest.py            # Gestion de manifests con checksums
│   ├── consolidator.py        # Consolidacion Bronze -> Silver
│   │
│   ├── raw/                   # BRONZE LAYER (inmutable)
│   │   ├── gfw/               # Global Fishing Watch (73 archivos)
│   │   ├── open_meteo/        # Condiciones marinas (73 archivos)
│   │   ├── sst/               # SST satelital Copernicus (73 archivos)
│   │   ├── currents/          # Corrientes oceanicas Copernicus (24 archivos)
│   │   ├── waves/             # Olas Copernicus (24 archivos)
│   │   └── chlorophyll/       # Clorofila-a Copernicus
│   │
│   ├── processed/             # SILVER LAYER (regenerable)
│   │   ├── fishing_consolidated.db
│   │   ├── marine_consolidated.db
│   │   └── training_features.parquet
│   │
│   ├── analytics/             # GOLD LAYER (ML-ready)
│   │   ├── current/
│   │   └── versions/
│   │
│   ├── metadata/              # Documentacion de datos
│   │   ├── sources.json
│   │   ├── schema.json
│   │   └── region.json
│   │
│   └── fetchers/
│       ├── real_data_only.py
│       └── historical_fetcher.py
│
├── models/
│   ├── features.py            # Extractor de 32 features oceanograficos
│   ├── predictor.py           # ML: PCA + KMeans + Gradient Boosting
│   ├── timeline.py            # Analisis temporal
│   └── anchovy_migration.py   # Modelo de migracion
│
├── controllers/
│   └── analysis.py            # Controlador principal (Copernicus + fallback)
│
├── views/
│   └── map_view.py            # Generador de mapas Folium
│
├── core/
│   ├── copernicus_data_provider.py  # Proveedor unificado de datos Copernicus
│   ├── coastline_real.py            # Procesador de costa OSM (7,741 pts)
│   ├── weather_solunar.py           # Clima y calculos solunares
│   ├── marine_data.py               # Datos marinos y frentes termicos
│   └── verification_image.py        # Generador de imagenes de verificacion
│
├── scripts/
│   ├── download_incremental.py          # Descarga incremental de datos
│   ├── download_copernicus_data.py      # Descarga SST/corrientes/olas
│   ├── run_prediction_with_copernicus.py # Prediccion con datos Copernicus
│   ├── test_model_with_real_data.py     # Tests con datos reales
│   ├── update_database.py               # Actualizacion completa de BD
│   ├── validate_data.py                 # Validacion de integridad
│   ├── migrate_existing_data.py         # Migracion de datos legacy
│   └── coastline_processing/            # Scripts de costa (one-time)
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

### 2. Configurar credenciales

Crear archivo `.env` en la raiz del proyecto:

```bash
# Copiar plantilla
cp .env.example .env

# Editar con tus credenciales
nano .env
```

Contenido de `.env`:
```env
GFW_API_KEY=tu_api_key_aqui
COPERNICUS_USER=tu_email
COPERNICUS_PASS=tu_password
EARTHDATA_USER=tu_email
EARTHDATA_PASS=tu_password
```

### 3. Descargar datos (2020-2026)

```bash
# Descarga completa (~15 minutos)
python scripts/update_database.py

# O descargar fuentes individuales
python scripts/download_incremental.py --source gfw --start 2020-01 --end 2026-01
python scripts/download_incremental.py --source open_meteo --start 2020-01 --end 2026-01
python scripts/download_incremental.py --source copernicus_sst --start 2020-01 --end 2026-01
```

### 4. Validar datos

```bash
python scripts/validate_data.py --all
```

## Uso

### Analisis Rapido (Open-Meteo - tiempo real)

```bash
python main.py
```

### Analisis con Datos Copernicus (mayor precision)

```bash
# Usar datos de Copernicus (requiere datos descargados)
python main.py --copernicus

# Analisis historico con Copernicus
python main.py --date 2025-01-15 --copernicus
```

### Analisis para fecha especifica

```bash
python main.py --date 2026-02-15
```

### Busqueda por Proximidad

Busca los mejores spots cerca de tu ubicacion:

```bash
# Buscar dentro de 10km de tu ubicacion (default)
python main.py --lat -17.8 --lon -71.2

# Buscar dentro de 5km
python main.py --lat -17.8 --lon -71.2 --radius 5

# Combinar con fecha especifica
python main.py --lat -17.8 --lon -71.2 --radius 15 --date 2026-02-10
```

El sistema mostrara:
1. **Spots cercanos**: Ordenados por score y distancia a tu ubicacion
2. **Mejores spots globales**: Top 10 en toda el area de estudio con distancia a ti
3. **Mapa interactivo**: Con tu ubicacion marcada y circulo de radio de busqueda

### Analisis con ML Supervisado (requiere datos historicos)

```bash
python main.py --supervised
```

### Actualizar Base de Datos

```bash
# Actualiza todos los datos al mes actual
python scripts/update_database.py
```

### Notebook Interactivo

```bash
jupyter lab fishing_analysis.ipynb
```

## Arquitectura de Datos

### Bronze Layer (raw/) - INMUTABLE
- Archivos Parquet particionados por mes (YYYY-MM.parquet)
- Manifests con checksums SHA256 para trazabilidad
- NUNCA se modifican, solo se agregan nuevos meses

### Silver Layer (processed/) - REGENERABLE
- Bases SQLite consolidadas
- Deduplicacion automatica
- Puede regenerarse desde Bronze con `python data/consolidator.py`

### Gold Layer (analytics/) - VERSIONADO
- Datasets listos para ML
- Versionado por fecha para reproducibilidad

### Estadisticas Actuales

| Capa | Fuente | Archivos | Registros |
|------|--------|----------|-----------|
| Bronze | GFW | 73 | 1,305 |
| Bronze | Open-Meteo | 73 | 216,354 |
| Bronze | Copernicus SST | 73 | 354,362 |
| Bronze | Copernicus Corrientes | 24 | ~50,000 |
| Bronze | Copernicus Olas | 24 | ~50,000 |
| Silver | Fishing DB | 1 | 1,085 |
| Silver | Marine DB | 1 | 572,793 |
| Silver | Training Features | 1 | 213,378 |

## Fuentes de Datos Reales

| Tipo de Dato | Fuente | Descripcion | Periodo |
|--------------|--------|-------------|---------|
| SST | Copernicus Marine OSTIA | Temperatura superficial del mar | 2020-2026 |
| Corrientes | Copernicus GLORYS | Velocidad (uo, vo) en m/s | 2024-2025 |
| Olas | Copernicus Wave | Altura (VHM0), Periodo (VTPK), Dir (VMDR) | 2024-2025 |
| Clorofila-a | Copernicus Ocean Colour | Concentracion en mg/m3 | 2024-2025 |
| Olas/Viento (fallback) | Open-Meteo ERA5 | Reanalisis ECMWF | 2020-2026 |
| Pesca | Global Fishing Watch | Actividad pesquera AIS | 2020-2026 |
| Zonas Historicas | IMARPE | Reportes del Instituto del Mar | Climatologia |

### Prioridad de Datos
1. **Copernicus** (Principal): Datos satelitales de alta precision
2. **Open-Meteo** (Fallback): Cuando no hay datos de Copernicus disponibles

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
  - xarray, netCDF4, h5py (datos NetCDF)
  - copernicusmarine (Copernicus API)
  - python-dotenv (credenciales)

## Documentacion Adicional

| Documento | Descripcion |
|-----------|-------------|
| [`DEVELOPMENT_GUIDELINES.md`](DEVELOPMENT_GUIDELINES.md) | Lineamientos tecnicos y arquitectura de datos |
| [`docs/oceanographic_data_sources.md`](docs/oceanographic_data_sources.md) | Fuentes de datos oceanograficos y APIs |
| [`investigacion.md`](investigacion.md) | Investigacion cientifica y metodologia |

## Comandos Utiles

```bash
# Actualizar base de datos
python scripts/update_database.py

# Validar integridad de datos
python scripts/validate_data.py --all

# Descargar fuente especifica
python scripts/download_incremental.py --source copernicus_sst --start 2024-01 --end 2026-01

# Consolidar datos (Bronze -> Silver)
python -c "from data.consolidator import Consolidator; Consolidator().consolidate_all()"

# Ver estado de manifests
python -c "from data.manifest import ManifestManager; m = ManifestManager('gfw'); print(m.get_summary())"
```

## Licencia

Desarrollado para pesca con spinning desde orilla en la costa sur de Peru.

---

*Ultima actualizacion: 2026-02-03*
