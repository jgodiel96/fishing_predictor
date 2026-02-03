# Development Guidelines - Fishing Predictor

**Version**: 3.1
**Fecha**: 2026-02-03

Este documento establece los lineamientos tecnicos y buenas practicas para el desarrollo del sistema de prediccion de pesca.

---

## 1. Arquitectura del Proyecto

### 1.1 Patron MVC + Data Lakehouse

```
fishing_predictor/
├── models/          # Logica de negocio y ML
├── views/           # Renderizado (mapas, reportes)
├── controllers/     # Orquestacion de flujo
├── core/            # Servicios de infraestructura (runtime)
├── scripts/         # Scripts de utilidad y one-time processing
│   └── coastline_processing/  # Scripts de procesamiento de costa
├── data/            # Acceso y gestion de datos
│   ├── raw/         # Bronze Layer (inmutable)
│   ├── processed/   # Silver Layer (regenerable)
│   └── analytics/   # Gold Layer (ML-ready)
├── config.py        # Configuracion centralizada (LEGACY_DB, paths)
└── domain.py        # Constantes de dominio (CENTRAL)
```

**Reglas:**
- **Models**: Solo logica de negocio. Sin I/O directo.
- **Views**: Solo renderizado. Sin logica de negocio.
- **Controllers**: Orquestacion. Minima logica.
- **Core**: Servicios reutilizables (fetchers, parsers).
- **Data**: Gestion de capas Bronze/Silver/Gold.

### 1.2 Modulo Central: `domain.py`

**TODAS las constantes de dominio DEBEN definirse en `domain.py`:**

```python
# CORRECTO - Importar desde domain.py
from domain import HOTSPOTS, SPECIES, THRESHOLDS

# INCORRECTO - Hardcodear constantes
HOTSPOTS = [(-17.70, -71.33, "Punta Coles", 1.3), ...]  # NO
```

### 1.3 Configuracion de Datos: `data/data_config.py`

**TODOS los paths de datos DEBEN usar DataConfig:**

```python
# CORRECTO - Usar DataConfig
from data.data_config import DataConfig

db_path = DataConfig.FISHING_DB
raw_dir = DataConfig.RAW_GFW

# INCORRECTO - Paths hardcodeados
db_path = "data/processed/fishing.db"  # NO
```

### 1.4 Configuracion Global: `config.py`

**Constantes compartidas entre modulos DEBEN definirse en `config.py`:**

```python
# CORRECTO - Importar desde config.py
from config import LEGACY_DB, COASTLINE_FILE

# INCORRECTO - Definir la misma constante en multiples archivos
LEGACY_DB = PROJECT_ROOT / "data" / "real_only" / "real_data_100.db"  # NO - duplicado
```

### 1.5 Sincronizacion de Bounds

**DEFAULT_BBOX y similares DEBEN sincronizarse con STUDY_AREA:**

```python
# CORRECTO - Usar STUDY_AREA como fuente unica
from domain import STUDY_AREA

DEFAULT_BBOX = {
    "north": STUDY_AREA.north,
    "south": STUDY_AREA.south,
    "west": STUDY_AREA.west,
    "east": STUDY_AREA.east
}

# INCORRECTO - Valores hardcodeados que pueden desincronizarse
DEFAULT_BBOX = {"north": -17.50, "south": -18.25, ...}  # NO
```

---

## 2. Arquitectura de Datos Bronze/Silver/Gold

### 2.1 Bronze Layer (raw/) - INMUTABLE

```
data/raw/
├── gfw/                    # Global Fishing Watch
│   ├── 2020-01.parquet
│   ├── 2020-02.parquet
│   ├── ...
│   ├── 2026-01.parquet
│   └── _manifest.json
├── open_meteo/             # Condiciones marinas
│   ├── YYYY-MM.parquet
│   └── _manifest.json
└── sst/
    └── copernicus/         # SST satelital
        ├── YYYY-MM.parquet
        └── _manifest.json
```

**Reglas Bronze:**
1. **NUNCA** modificar archivos existentes
2. Solo **AGREGAR** nuevos archivos (nuevos meses)
3. Cada archivo = 1 mes de datos
4. Naming: `YYYY-MM.parquet`
5. Manifest obligatorio por fuente
6. Checksum SHA256 para verificacion

```python
# CORRECTO - Agregar nuevo mes
output_path = DataConfig.RAW_GFW / "2026-02.parquet"
df.to_parquet(output_path)
manifest.add_download(filename="2026-02.parquet", ...)

# INCORRECTO - Modificar archivo existente
existing_path = DataConfig.RAW_GFW / "2026-01.parquet"
df.to_parquet(existing_path)  # NO - viola inmutabilidad
```

### 2.2 Silver Layer (processed/) - REGENERABLE

```
data/processed/
├── fishing_consolidated.db     # Toda la pesca unificada
├── marine_consolidated.db      # Condiciones + SST
├── training_features.parquet   # Features para ML
└── _consolidation_log.json     # Registro de consolidacion
```

**Reglas Silver:**
1. Puede regenerarse desde Bronze
2. Script de consolidacion idempotente
3. Deduplicacion automatica
4. Log de cada consolidacion

```python
# Regenerar Silver desde Bronze
from data.consolidator import Consolidator
consolidator = Consolidator()
consolidator.consolidate_all()
```

### 2.3 Gold Layer (analytics/) - VERSIONADO

```
data/analytics/
├── current/
│   ├── training_dataset.parquet
│   └── model_metadata.json
└── versions/
    ├── v20260115/
    ├── v20260122/
    └── v20260129/
```

**Reglas Gold:**
1. Versionado por fecha
2. Mantener ultimas 5 versiones
3. Metadata de features usados
4. Reproducibilidad garantizada

### 2.4 Manifests

Cada fuente en Bronze tiene un `_manifest.json`:

```json
{
  "source": "gfw",
  "version": "1.0",
  "downloads": [
    {
      "file": "2024-01.parquet",
      "period": {"start": "2024-01-01", "end": "2024-01-31"},
      "downloaded_at": "2026-01-29T10:30:00Z",
      "records": 1527,
      "checksum": "sha256:abc123...",
      "api_response_code": 200
    }
  ]
}
```

---

## 3. Estructuras de Datos Eficientes

### 3.1 Preferir NamedTuple sobre Dict

```python
# INCORRECTO - Dict mutable, sin tipado
BBOX = {"north": -17.50, "south": -18.25, ...}

# CORRECTO - NamedTuple inmutable, tipado
class BoundingBox(NamedTuple):
    north: float
    south: float
    west: float
    east: float

STUDY_AREA = BoundingBox(-17.50, -18.25, -71.45, -70.55)
```

### 3.2 Preferir Tuple sobre List para datos fijos

```python
# INCORRECTO - Lista mutable
FEATURE_NAMES = ['sst', 'sst_anomaly', ...]

# CORRECTO - Tupla inmutable
FEATURE_NAMES: Tuple[str, ...] = ('sst', 'sst_anomaly', ...)
```

### 3.3 Preferir FrozenSet para colecciones de busqueda

```python
# INCORRECTO - Set mutable
VALID_SPECIES = {'Cabrilla', 'Corvina', 'Robalo'}

# CORRECTO - FrozenSet inmutable
VALID_SPECIES: FrozenSet[str] = frozenset({'Cabrilla', 'Corvina', 'Robalo'})
```

---

## 4. Manejo de Datos

### 4.1 NO usar datos sinteticos

```python
# PROHIBIDO - Generacion de datos sinteticos
def generate_synthetic_sst():
    return np.random.normal(17.5, 2.0, size=1000)  # NO

# CORRECTO - Solo datos reales
def fetch_real_sst():
    return erddap_client.get_sst(STUDY_AREA)
```

### 4.2 Prioridad de fuentes de datos

1. **Global Fishing Watch** - Actividad pesquera real (API)
2. **Copernicus Marine** - SST satelital (API)
3. **Open-Meteo ERA5** - Condiciones marinas (API)
4. **IMARPE** - Datos historicos verificados (climatologia)

### 4.3 Descargas Incrementales

```python
# CORRECTO - Usar download_incremental.py
python scripts/download_incremental.py --source gfw --start 2026-01 --end 2026-02

# El script:
# 1. Verifica que meses ya existen
# 2. Solo descarga meses faltantes
# 3. Actualiza manifest con checksums
# 4. NUNCA modifica datos existentes
```

### 4.4 Credenciales

```python
# CORRECTO - Usar .env y DataConfig
from data.data_config import DataConfig

gfw_key = DataConfig.get_gfw_api_key()
copernicus_user = DataConfig.get_copernicus_credentials()

# INCORRECTO - Hardcodear credenciales
GFW_API_KEY = "abc123..."  # NO - nunca en codigo
```

---

## 5. Convencion de Nombres

### 5.1 Archivos de datos

```
# Bronze layer
YYYY-MM.parquet          # Datos mensuales
_manifest.json           # Manifest de fuente

# Silver layer
*_consolidated.db        # Base consolidada
training_features.parquet

# Gold layer
training_dataset.parquet
model_metadata.json
```

### 5.2 Constantes

```python
# Constantes de modulo - UPPER_SNAKE_CASE
STUDY_AREA = BoundingBox(...)
MAX_WAVE_HEIGHT = 2.0

# Constantes de configuracion
class DataConfig:
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
```

### 5.3 Funciones y Variables

```python
# Funciones - snake_case
def calculate_sst_score(sst: float) -> float:
    pass

# Variables - snake_case
current_temperature = 17.5
```

---

## 6. Type Hints Obligatorios

### 6.1 Funciones publicas

```python
def calculate_score(
    lat: float,
    lon: float,
    sst: float,
    *,
    include_historical: bool = True
) -> float:
    """Calcula score de pesca para una ubicacion."""
    ...
```

### 6.2 Return types explicitos

```python
# CORRECTO
def get_hotspots() -> Tuple[FishingLocation, ...]:
    return HOTSPOTS

# INCORRECTO - Sin return type
def get_hotspots():
    return HOTSPOTS
```

---

## 7. Imports

### 7.1 Orden de imports

```python
# 1. Standard library
import os
from pathlib import Path
from typing import List, Dict, Tuple

# 2. Third party
import numpy as np
import pandas as pd

# 3. Local application
from domain import HOTSPOTS, SPECIES
from data.data_config import DataConfig
from models.features import FeatureExtractor
```

### 7.2 Imports explicitos

```python
# CORRECTO - Imports explicitos
from domain import HOTSPOTS, SPECIES, THRESHOLDS
from data.data_config import DataConfig

# EVITAR - Import de todo
from domain import *
```

---

## 8. Scripts de Datos

### 8.1 Descarga incremental

```bash
# Descargar todo
python scripts/update_database.py

# Descargar fuente especifica
python scripts/download_incremental.py --source copernicus_sst --start 2024-01 --end 2026-01

# Dry run (ver que se descargaria)
python scripts/download_incremental.py --dry-run
```

### 8.2 Validacion

```bash
# Validar todas las capas
python scripts/validate_data.py --all

# Validar capa especifica
python scripts/validate_data.py --bronze
python scripts/validate_data.py --silver
```

### 8.3 Consolidacion

```python
from data.consolidator import Consolidator

# Consolidar todo
consolidator = Consolidator()
consolidator.consolidate_all()

# Consolidar fuente especifica
consolidator.consolidate_fishing()
consolidator.consolidate_marine()
consolidator.consolidate_sst()
```

---

## 9. Testing

### 9.1 Estructura de tests

```
tests/
├── test_domain.py       # Tests de constantes
├── test_features.py     # Tests de extraccion
├── test_predictor.py    # Tests de ML
├── test_data_config.py  # Tests de configuracion
└── conftest.py          # Fixtures compartidos
```

### 9.2 Tests de datos

```python
def test_bronze_layer_integrity():
    """Verifica que Bronze layer tiene manifests validos."""
    from scripts.validate_data import DataValidator
    validator = DataValidator()
    errors = validator.validate_bronze()
    assert len(errors) == 0
```

---

## 10. Checklist Pre-Commit

Antes de hacer commit, verificar:

- [ ] `domain.py` no tiene imports circulares
- [ ] No hay constantes hardcodeadas fuera de `domain.py`
- [ ] No hay paths hardcodeados fuera de `DataConfig`
- [ ] No hay datos sinteticos
- [ ] No hay credenciales en el codigo
- [ ] Todas las funciones publicas tienen type hints
- [ ] Tests pasan: `python -m pytest tests/`
- [ ] Datos validan: `python scripts/validate_data.py --all`
- [ ] `.env` esta en `.gitignore`

---

## 11. Comandos Utiles

```bash
# Actualizar base de datos completa
python scripts/update_database.py

# Validar integridad
python scripts/validate_data.py --all

# Ver estado de manifests
python -c "from data.manifest import ManifestManager; m = ManifestManager('gfw'); print(m.get_summary())"

# Consolidar datos
python -c "from data.consolidator import Consolidator; Consolidator().consolidate_all()"

# Ejecutar tests
python -m pytest tests/ -v
```

---

## 12. Referencias

- [PEP 8 - Style Guide](https://peps.python.org/pep-0008/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Data Lakehouse Architecture](https://www.databricks.com/glossary/data-lakehouse)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)

---

*Lineamientos establecidos: 2026-01-30*
*Ultima actualizacion: 2026-02-03*
