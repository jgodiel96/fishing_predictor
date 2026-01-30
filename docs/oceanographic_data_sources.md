# Oceanographic Data Sources for Fishing Prediction

This document provides a comprehensive list of free/open APIs and data sources for oceanographic data that can be used for fishing prediction, with special focus on South America/Peru coverage.

**Ultima actualizacion:** 2026-01-30

---

## 1. Fuentes Implementadas en el Proyecto

### 1.1 Resumen de Datos Actuales

| Fuente | Tipo | Registros | Periodo | Actualizacion |
|--------|------|-----------|---------|---------------|
| **Copernicus Marine** | SST | 354,362 | 2020-2026 | Mensual |
| **Open-Meteo ERA5** | Olas, Viento | 216,354 | 2020-2026 | Mensual |
| **Global Fishing Watch** | Pesca AIS | 1,085 | 2020-2026 | Mensual |
| **IMARPE** | Climatologia | - | Historico | Estatico |

### 1.2 Copernicus Marine - SST (Implementado)

| Atributo | Detalle |
|----------|---------|
| **Datasets** | METOFFICE-GLO-SST-L4-REP-OBS-SST (historico), METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 (reciente) |
| **Resolucion** | 0.05° (~5km) |
| **Variables** | `analysed_sst` |
| **Formato** | NetCDF -> Parquet |
| **Autenticacion** | COPERNICUS_USER, COPERNICUS_PASS en `.env` |

**Uso en el proyecto:**
```python
# Descarga automatica via download_incremental.py
python scripts/download_incremental.py --source copernicus_sst --start 2020-01 --end 2026-01

# Datos guardados en:
# data/raw/sst/copernicus/YYYY-MM.parquet
```

**Datasets usados:**
- **REP** (1981-2023): `METOFFICE-GLO-SST-L4-REP-OBS-SST`
- **NRT** (2024+): `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2`

### 1.3 Open-Meteo ERA5 - Condiciones Marinas (Implementado)

| Atributo | Detalle |
|----------|---------|
| **API** | https://marine-api.open-meteo.com/v1/marine |
| **Variables** | wave_height_max, wave_period_max, wave_direction_dominant, wind_wave_height_max |
| **Resolucion** | 0.1° grid |
| **Autenticacion** | Ninguna (publico) |

**Uso en el proyecto:**
```python
python scripts/download_incremental.py --source open_meteo --start 2020-01 --end 2026-01

# Datos guardados en:
# data/raw/open_meteo/YYYY-MM.parquet
```

### 1.4 Global Fishing Watch - Actividad Pesquera (Implementado)

| Atributo | Detalle |
|----------|---------|
| **API** | https://gateway.api.globalfishingwatch.org/v3/4wings/report |
| **Variables** | fishing_hours, lat, lon, flag_state, gear_type |
| **Autenticacion** | GFW_API_KEY en `.env` (Bearer token) |

**Uso en el proyecto:**
```python
python scripts/download_incremental.py --source gfw --start 2020-01 --end 2026-01

# Datos guardados en:
# data/raw/gfw/YYYY-MM.parquet
```

**Obtencion de API Key:**
1. Ir a: https://globalfishingwatch.org/our-apis/
2. Click en "Request API Access"
3. Completar formulario (gratis para investigacion)

---

## 2. Fuentes Adicionales Disponibles

### 2.1 NOAA OISST (Alternativa SST)

| Atributo | Detalle |
|----------|---------|
| **Provider** | NOAA NCEI |
| **Dataset** | NOAA 1/4° Daily Optimum Interpolation SST v2.1 |
| **Resolucion** | 0.25° x 0.25° (~25km) |
| **Temporal** | Diario, desde 1981 |
| **Cobertura** | Global |
| **Costo** | Gratis, sin cuenta |

**Acceso via ERDDAP:**
```python
from erddapy import ERDDAP

server = ERDDAP(
    server="https://coastwatch.pfeg.noaa.gov/erddap/",
    protocol="griddap"
)
server.dataset_id = "ncdcOisst21Agg_LonPM180"
```

### 2.2 NASA MUR SST (Alta Resolucion)

| Atributo | Detalle |
|----------|---------|
| **Provider** | NASA JPL PO.DAAC |
| **Resolucion** | 1km (0.01°) |
| **Temporal** | Diario |
| **Costo** | Gratis (requiere Earthdata login) |

**Acceso:**
```python
import earthaccess

earthaccess.login()
results = earthaccess.search_data(
    short_name="MUR-JPL-L4-GLOB-v4.1",
    temporal=("2024-01-01", "2024-01-31"),
    bounding_box=(-71.5, -18.3, -70.8, -17.3)
)
```

### 2.3 HYCOM - Corrientes Oceanicas

| Atributo | Detalle |
|----------|---------|
| **Provider** | HYCOM Consortium / NOAA |
| **Resolucion** | 1/12° (~8km) |
| **Variables** | Velocidad U/V, temperatura, salinidad |
| **Cobertura** | Global |

### 2.4 FAO FishStat - Estadisticas Historicas

| Atributo | Detalle |
|----------|---------|
| **Provider** | FAO |
| **Datos** | Captura, acuicultura, comercio |
| **Cobertura** | Global (200+ paises) |
| **Periodo** | 1976-presente |

---

## 3. Configuracion de Credenciales

### 3.1 Archivo `.env`

Crear en la raiz del proyecto:

```env
# Global Fishing Watch
GFW_API_KEY=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...

# Copernicus Marine
COPERNICUS_USER=tu_email@ejemplo.com
COPERNICUS_PASS=tu_password

# NASA Earthdata (opcional, para MUR SST)
EARTHDATA_USER=tu_email@ejemplo.com
EARTHDATA_PASS=tu_password
```

### 3.2 Registro de APIs

| Servicio | URL de Registro | Tiempo Aprobacion |
|----------|-----------------|-------------------|
| Global Fishing Watch | https://globalfishingwatch.org/our-apis/ | 1-3 dias |
| Copernicus Marine | https://data.marine.copernicus.eu/register | Inmediato |
| NASA Earthdata | https://urs.earthdata.nasa.gov/users/new | Inmediato |

---

## 4. Arquitectura de Datos del Proyecto

### 4.1 Bronze Layer (raw/)

Datos crudos inmutables, particionados por mes:

```
data/raw/
├── gfw/
│   ├── 2020-01.parquet
│   ├── ...
│   ├── 2026-01.parquet
│   └── _manifest.json
├── open_meteo/
│   ├── YYYY-MM.parquet
│   └── _manifest.json
└── sst/
    └── copernicus/
        ├── YYYY-MM.parquet
        └── _manifest.json
```

### 4.2 Formato Parquet

Cada archivo contiene:

**GFW (pesca):**
```
date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source
```

**Open-Meteo (marino):**
```
date, lat, lon, wave_height, wave_period, wave_direction, wind_speed, source
```

**Copernicus (SST):**
```
date, lat, lon, sst, source
```

### 4.3 Manifests

Cada fuente tiene un `_manifest.json` que registra:
- Checksum SHA256 de cada archivo
- Fecha de descarga
- Numero de registros
- Periodo cubierto

---

## 5. Comandos de Descarga

### 5.1 Descarga Completa

```bash
# Actualizar todo (GFW + Open-Meteo + Copernicus)
python scripts/update_database.py
```

### 5.2 Descarga por Fuente

```bash
# Solo Global Fishing Watch
python scripts/download_incremental.py --source gfw --start 2020-01 --end 2026-01

# Solo Open-Meteo
python scripts/download_incremental.py --source open_meteo --start 2020-01 --end 2026-01

# Solo Copernicus SST
python scripts/download_incremental.py --source copernicus_sst --start 2020-01 --end 2026-01
```

### 5.3 Opciones

```bash
# Ver que se descargaria sin hacerlo
python scripts/download_incremental.py --dry-run

# Modo verbose
python scripts/download_incremental.py --source gfw --verbose
```

---

## 6. Region del Proyecto

### 6.1 Tacna-Ilo, Peru

| Parametro | Valor |
|-----------|-------|
| lat_min | -18.3 |
| lat_max | -17.3 |
| lon_min | -71.5 |
| lon_max | -70.8 |
| Grid resolution | 0.1° |

### 6.2 Sistema de la Corriente de Humboldt

El area de estudio esta dentro del Sistema de Corriente de Humboldt Norte (NHCS):
- Ecosistema mas productivo del mundo
- 15% de la captura anual global de peces
- Dominado por anchoveta y especies asociadas

---

## 7. Referencias de APIs

### 7.1 Documentacion Oficial

| API | Documentacion |
|-----|---------------|
| Copernicus Marine | https://help.marine.copernicus.eu/ |
| Open-Meteo Marine | https://open-meteo.com/en/docs/marine-weather-api |
| Global Fishing Watch | https://globalfishingwatch.org/our-apis/documentation |
| NOAA ERDDAP | https://coastwatch.pfeg.noaa.gov/erddap/ |
| NASA Earthdata | https://www.earthdata.nasa.gov/ |

### 7.2 Librerias Python

```python
# Instalacion
pip install copernicusmarine    # Copernicus Marine
pip install erddapy             # ERDDAP servers
pip install earthaccess         # NASA data
pip install xarray netCDF4 h5py # NetCDF handling
pip install python-dotenv       # Credenciales
```

---

## 8. Integracion con el Proyecto

### 8.1 Usar DataConfig

```python
from data.data_config import DataConfig

# Paths
raw_gfw = DataConfig.RAW_GFW
processed_db = DataConfig.FISHING_DB

# Credenciales
gfw_key = DataConfig.get_gfw_api_key()
copernicus_creds = DataConfig.get_copernicus_credentials()
```

### 8.2 Usar ManifestManager

```python
from data.manifest import ManifestManager

# Ver estado de descargas
manager = ManifestManager('gfw')
print(manager.get_summary())

# Ver meses faltantes
missing = manager.get_missing_months(2020, 1, 2026, 1)
```

### 8.3 Usar Consolidator

```python
from data.consolidator import Consolidator

# Consolidar Bronze -> Silver
consolidator = Consolidator()
consolidator.consolidate_all()
```

---

*Documento actualizado: 2026-01-30*
*Proyecto: Fishing Predictor - Tacna/Ilo, Peru*
