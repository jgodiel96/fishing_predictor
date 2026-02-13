# Fuentes de Datos para Fishing Predictor V8

## Resumen de Fuentes Disponibles

| Fuente | Tipo de Datos | Cobertura | Estado |
|--------|---------------|-----------|--------|
| **OSM** | Costas, Agua | Global | ✅ Disponible |
| **Copernicus Marine** | Océano (física, bio, olas) | Global | ⚙️ Requiere registro |
| **GEBCO** | Batimetría | Global | ⬇️ Descarga manual |
| **OpenSeaMap** | Náutico (puertos, boyas) | Global | 🌐 API/WMS |
| **IMARPE** | Pesquería Perú | Perú | 📋 Solicitud |

---

## 1. OpenStreetMap (OSM)

**URL:** https://osmdata.openstreetmap.de/data/coastlines.html

### Datasets Disponibles

| Dataset | Descripción | Tamaño | Estado |
|---------|-------------|--------|--------|
| `coastlines-split-4326` | Líneas de costa | ~500MB | ✅ Descargado |
| `water-polygons-split-4326` | **Polígonos de agua** | ~700MB | ✅ Descargado |
| `land-polygons-split-4326` | Polígonos de tierra | ~600MB | Opcional |

### Uso en el Sistema

```python
from core.cv_analysis.osm_coastline import OSMCoastlineLoader

loader = OSMCoastlineLoader()
result = loader.load_coastline(lat_min, lat_max, lon_min, lon_max)

# Clasificar punto como agua/tierra
is_water = loader.is_in_water(lat, lon, result.water_polygons)
```

---

## 2. Copernicus Marine Service

**URL:** https://marine.copernicus.eu/

**Registro:** Gratuito en https://marine.copernicus.eu/register-copernicus-marine-service

### Variables de Física Oceánica

| Variable | Nombre | Unidad | Uso en Pesca |
|----------|--------|--------|--------------|
| `thetao` | Temperatura agua | °C | Distribución especies |
| `so` | Salinidad | PSU | Masas de agua, frentes |
| `zos` | Altura superficie (SLA) | m | Upwelling, corrientes |
| `uo` | Corriente Este | m/s | Transporte larvas |
| `vo` | Corriente Norte | m/s | Transporte larvas |
| `mlotst` | Capa de mezcla | m | Estratificación |

### Variables de Color del Océano

| Variable | Nombre | Unidad | Uso en Pesca |
|----------|--------|--------|--------------|
| `CHL` | Clorofila-a | mg/m³ | Productividad primaria |
| `KD490` | Coef. atenuación | m⁻¹ | Claridad del agua |

### Variables de Olas

| Variable | Nombre | Unidad | Uso en Pesca |
|----------|--------|--------|--------------|
| `VHM0` | Altura ola significativa | m | Condiciones navegación |
| `VTPK` | Período pico ola | s | Tipo oleaje |
| `VMDR` | Dirección ola | ° | Planificación |

### Variables de Viento

| Variable | Nombre | Unidad | Uso en Pesca |
|----------|--------|--------|--------------|
| `eastward_wind` | Viento Este (U10) | m/s | Afecta oleaje |
| `northward_wind` | Viento Norte (V10) | m/s | Afecta oleaje |

### Comando de Descarga

```bash
# Configurar credenciales en .env
COPERNICUS_USER=tu_email@example.com
COPERNICUS_PASS=tu_password

# Descargar todas las variables
python scripts/download_all_copernicus.py --start 2024-01 --end 2026-01

# Descargar solo física oceánica
python scripts/download_all_copernicus.py --datasets physics_daily

# Listar variables disponibles
python scripts/download_all_copernicus.py --list
```

---

## 3. GEBCO (Batimetría)

**URL:** https://www.gebco.net/data_and_products/gridded_bathymetry_data/

### GEBCO 2025 Grid

| Característica | Valor |
|----------------|-------|
| Resolución | 15 arc-second (~450m) |
| Cobertura | Global |
| Variable | Elevación (m) |
| Formato | NetCDF (recomendado) |
| Tamaño | ~250-300MB (costa Perú) |

### Descarga (Costa completa de Perú)

1. Ir a https://download.gebco.net/
2. Seleccionar región completa de Perú:
   - **North:** -3 (frontera Ecuador)
   - **South:** -20 (frontera Chile)
   - **West:** -82 (zona offshore)
   - **East:** -70 (costa)
3. Seleccionar formatos:
   - **Grid:** 2D netCDF (obligatorio)
   - **GeoTIFF:** opcional para visualización
   - Desmarcar: Esri ASCII, JPEG, PNG (no necesarios)
4. Guardar archivo netCDF en: `data/bathymetry/GEBCO_2025_peru.nc`

### Uso en el Sistema

```python
from data.data_config import DataConfig
from core.cv_analysis.bathymetry import GEBCOBathymetry

# Verificar disponibilidad
if DataConfig.has_gebco_data():
    gebco = GEBCOBathymetry(DataConfig.GEBCO_FILE)
    depth, confidence = gebco.get_depth(lat, lon)
    # depth negativo = bajo agua, positivo = sobre tierra
```

---

## 4. OpenSeaMap (Datos Náuticos)

**URL:** https://map.openseamap.org/

### Datos Disponibles

- Puertos y marinas (5000+ a nivel mundial)
- Boyas y señalización
- Batimetría crowdsourced (0-100m)
- Rutas de navegación

### Acceso

- **Visualización:** https://map.openseamap.org/
- **API WMS:** Para integración en mapas
- **Descarga:** Datos incluidos en OSM

---

## 5. IMARPE (Perú)

**URL:** https://www.imarpe.gob.pe/

### Datos Disponibles

- Estadísticas de desembarque
- Cruceros oceanográficos
- Monitoreo de anchoveta
- Boletines oceanográficos

### Acceso

Requiere solicitud formal a IMARPE para datos históricos.
Algunos datos disponibles en repositorio: https://repositorio.imarpe.gob.pe/

---

## 6. Fuentes Adicionales

### NOAA ERDDAP

**URL:** https://coastwatch.pfeg.noaa.gov/erddap/

- SST alta resolución (GHRSST)
- Clorofila MODIS/VIIRS
- Vientos QuikSCAT/ASCAT

### EMODnet (Europa)

**URL:** https://emodnet.ec.europa.eu/

- Sustrato del fondo marino
- Batimetría alta resolución
- **Nota:** Cobertura principalmente europea

---

## Prioridad de Descarga

### Alta Prioridad (necesarios)
1. ✅ `water-polygons-split-4326` - Clasificación agua/tierra (1.2GB)
2. ⬇️ GEBCO Costa Perú - Profundidad real (~300MB) - **PENDIENTE**
3. ✅ SST (Copernicus) - Temperatura (73 archivos)

### Media Prioridad (mejoran predicciones)
4. ✅ `CHL` - Clorofila (24 archivos)
5. ✅ `SSS` - Salinidad (25 archivos)
6. ✅ `SLA` - Nivel del mar (25 archivos)
7. ⬇️ `uo/vo` - Corrientes (transporte) - Pendiente
8. ⬇️ `VHM0` - Olas (condiciones) - Pendiente

### Baja Prioridad (complementarios)
9. `KD490` - Claridad agua
10. `mlotst` - Capa mezcla
11. Viento (ya disponible en Open-Meteo)

---

## Estructura de Directorios

```
data/
├── coastlines/
│   ├── coastlines-split-4326/     # Líneas de costa
│   └── water-polygons-split-4326/ # Polígonos de agua ✅
├── bathymetry/
│   └── GEBCO_2024_subset.nc       # Batimetría
├── raw/
│   ├── sst/copernicus/            # Temperatura
│   ├── sss/copernicus/            # Salinidad
│   ├── chl/copernicus/            # Clorofila
│   ├── uo/copernicus/             # Corriente E
│   ├── vo/copernicus/             # Corriente N
│   └── waves/copernicus/          # Olas
└── cache/                         # Cache de consultas
```
