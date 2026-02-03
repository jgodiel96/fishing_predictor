# Fishing Predictor - Sistema de Prediccion de Pesca desde Orilla

Sistema avanzado de prediccion de puntos optimos de pesca desde orilla para la costa sur de Peru (Tacna - Ilo).

**Fecha de actualizacion:** 2026-02-03

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
├── main.py                    # Punto de entrada principal
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
│   │   └── sst/
│   │       └── copernicus/    # SST satelital (73 archivos)
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
│   ├── predictor.py           # ML: PCA + Gradient Boosting
│   ├── timeline.py            # Analisis temporal
│   └── anchovy_migration.py   # Modelo de migracion
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
│   ├── marine_data.py         # Datos marinos y frentes termicos
│   └── verification_image.py  # Generador de imagenes de verificacion
│
├── scripts/
│   ├── download_incremental.py    # Descarga incremental de datos
│   ├── update_database.py         # Actualizacion completa de BD
│   ├── validate_data.py           # Validacion de integridad
│   ├── migrate_existing_data.py   # Migracion de datos legacy
│   └── coastline_processing/      # Scripts de procesamiento de costa (one-time)
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

### Analisis Rapido (sin datos historicos)

```bash
python main.py
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
| Silver | Fishing DB | 1 | 1,085 |
| Silver | Marine DB | 1 | 572,793 |
| Silver | Training Features | 1 | 213,378 |

## Fuentes de Datos Reales

| Tipo de Dato | Fuente | Descripcion | Periodo |
|--------------|--------|-------------|---------|
| SST | Copernicus Marine OSTIA | Temperatura superficial del mar | 2020-2026 |
| Olas/Viento | Open-Meteo ERA5 | Reanalisis ECMWF | 2020-2026 |
| Pesca | Global Fishing Watch | Actividad pesquera AIS | 2020-2026 |
| Zonas Historicas | IMARPE | Reportes del Instituto del Mar | Climatologia |

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
