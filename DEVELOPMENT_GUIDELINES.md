# Development Guidelines - Fishing Predictor

**Version**: 2.0
**Fecha**: 2026-01-28

Este documento establece los lineamientos tecnicos y buenas practicas para el desarrollo del sistema de prediccion de pesca.

---

## 1. Arquitectura del Proyecto

### 1.1 Patron MVC

```
fishing_predictor/
├── models/          # Logica de negocio y ML
├── views/           # Renderizado (mapas, reportes)
├── controllers/     # Orquestacion de flujo
├── core/            # Servicios de infraestructura
├── data/            # Acceso a datos
└── domain.py        # Constantes de dominio (CENTRAL)
```

**Reglas:**
- **Models**: Solo logica de negocio. Sin I/O directo.
- **Views**: Solo renderizado. Sin logica de negocio.
- **Controllers**: Orquestacion. Minima logica.
- **Core**: Servicios reutilizables (fetchers, parsers).

### 1.2 Modulo Central: `domain.py`

**TODAS las constantes de dominio DEBEN definirse en `domain.py`:**

```python
# CORRECTO - Importar desde domain.py
from domain import HOTSPOTS, SPECIES, THRESHOLDS

# INCORRECTO - Hardcodear constantes
HOTSPOTS = [(-17.70, -71.33, "Punta Coles", 1.3), ...]  # NO
```

---

## 2. Estructuras de Datos Eficientes

### 2.1 Preferir NamedTuple sobre Dict

```python
# INCORRECTO - Dict mutable, sin tipado
BBOX = {
    "north": -17.50,
    "south": -18.25,
    "west": -71.45,
    "east": -70.55,
}

# CORRECTO - NamedTuple inmutable, tipado
class BoundingBox(NamedTuple):
    north: float
    south: float
    west: float
    east: float

STUDY_AREA = BoundingBox(-17.50, -18.25, -71.45, -70.55)
```

**Beneficios:**
- Inmutabilidad (seguridad)
- Acceso por atributo (`box.north` vs `box["north"]`)
- Menor uso de memoria
- Type hints automaticos

### 2.2 Preferir Tuple sobre List para datos fijos

```python
# INCORRECTO - Lista mutable para datos constantes
FEATURE_NAMES = [
    'sst', 'sst_anomaly', 'sst_optimal_score', ...
]

# CORRECTO - Tupla inmutable
FEATURE_NAMES: Tuple[str, ...] = (
    'sst', 'sst_anomaly', 'sst_optimal_score', ...
)
```

### 2.3 Preferir FrozenSet para colecciones de busqueda

```python
# INCORRECTO - Set mutable
VALID_SPECIES = {'Cabrilla', 'Corvina', 'Robalo'}

# CORRECTO - FrozenSet inmutable y hasheable
VALID_SPECIES: FrozenSet[str] = frozenset({'Cabrilla', 'Corvina', 'Robalo'})
```

### 2.4 Usar Dataclass para estructuras con metodos

```python
# CORRECTO - Dataclass cuando necesitas metodos
@dataclass(frozen=True)
class Species:
    name: str
    temp_min: float
    temp_max: float

    def temp_score(self, sst: float) -> float:
        """Calcula score de temperatura."""
        if self.temp_min <= sst <= self.temp_max:
            return 1.0
        return 0.1
```

---

## 3. Evitar Variables Intermedias Innecesarias

### 3.1 Usar expresiones directas

```python
# INCORRECTO - Variables intermedias innecesarias
temp_list = []
for species in SPECIES:
    score = species.temp_score(sst)
    temp_list.append(score)
max_score = max(temp_list)

# CORRECTO - Expresion directa
max_score = max(sp.temp_score(sst) for sp in SPECIES)
```

### 3.2 Usar operador walrus cuando sea claro

```python
# INCORRECTO
point = self._fetch_point(lat, lon)
if point:
    self.sampled_points.append(point)

# CORRECTO (Python 3.8+)
if point := self._fetch_point(lat, lon):
    self.sampled_points.append(point)
```

### 3.3 Vectorizar con NumPy

```python
# INCORRECTO - Bucle Python
distances = []
for hotspot in HOTSPOTS:
    d = haversine(lat, lon, hotspot.lat, hotspot.lon)
    distances.append(d)
min_dist = min(distances)

# CORRECTO - Operacion vectorizada
hotspot_coords = np.array([(h.lat, h.lon) for h in HOTSPOTS])
distances = haversine_vectorized(lat, lon, hotspot_coords)
min_dist = distances.min()
```

---

## 4. Convencion de Nombres

### 4.1 Constantes

```python
# Constantes de modulo - UPPER_SNAKE_CASE
STUDY_AREA = BoundingBox(...)
MAX_WAVE_HEIGHT = 2.0
N_FEATURES = 32

# Constantes de clase - UPPER_SNAKE_CASE
class Config:
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
```

### 4.2 Clases y Tipos

```python
# Clases - PascalCase
class FishingPredictor:
    pass

class MarineDataFetcher:
    pass

# NamedTuples - PascalCase (son tipos)
class BoundingBox(NamedTuple):
    pass
```

### 4.3 Funciones y Variables

```python
# Funciones - snake_case
def calculate_sst_score(sst: float) -> float:
    pass

# Variables - snake_case
current_temperature = 17.5
hotspot_distance = 500.0

# Variables privadas - _prefijo
_cache_data = {}
```

### 4.4 Evitar nombres genericos

```python
# INCORRECTO
data = fetch_data()
result = process(data)
temp = calculate()

# CORRECTO
marine_points = fetch_marine_data()
scored_zones = calculate_zone_scores(marine_points)
sst_value = extract_sst(point)
```

---

## 5. Type Hints Obligatorios

### 5.1 Funciones publicas

```python
# CORRECTO - Siempre tipar funciones publicas
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

### 5.2 Usar tipos de collections.abc

```python
from typing import Sequence, Mapping, Iterable

# CORRECTO - Tipos abstractos para parametros
def process_points(points: Sequence[MarinePoint]) -> List[float]:
    pass

# INCORRECTO - Tipos concretos para parametros
def process_points(points: List[MarinePoint]) -> List[float]:
    pass
```

### 5.3 Return types explicitos

```python
# CORRECTO
def get_hotspots() -> Tuple[FishingLocation, ...]:
    return HOTSPOTS

# INCORRECTO - Sin return type
def get_hotspots():
    return HOTSPOTS
```

---

## 6. Manejo de Datos

### 6.1 NO usar datos sinteticos

```python
# PROHIBIDO - Generacion de datos sinteticos
def generate_synthetic_sst():
    return np.random.normal(17.5, 2.0, size=1000)  # NO

# CORRECTO - Solo datos reales
def fetch_real_sst():
    return erddap_client.get_sst(STUDY_AREA)
```

### 6.2 Prioridad de fuentes de datos

1. **Global Fishing Watch** - Actividad pesquera real
2. **Open-Meteo ERA5** - Reanalisis (datos reales procesados)
3. **NOAA ERDDAP** - SST satelital
4. **IMARPE** - Datos historicos verificados

### 6.3 Cache con TTL

```python
# CORRECTO - Cache con tiempo de vida
CACHE_TTL_HOURS = 6

def get_cached_data(key: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < CACHE_TTL_HOURS:
            return json.loads(cache_file.read_text())
    return None
```

---

## 7. Imports

### 7.1 Orden de imports

```python
# 1. Standard library
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# 2. Third party
import numpy as np
import pandas as pd
import requests

# 3. Local application
from domain import HOTSPOTS, SPECIES, THRESHOLDS
from models.features import FeatureExtractor
from core.marine_data import MarineDataFetcher
```

### 7.2 Imports explicitos

```python
# CORRECTO - Imports explicitos
from domain import HOTSPOTS, SPECIES, THRESHOLDS

# EVITAR - Import de todo
from domain import *
```

---

## 8. Documentacion

### 8.1 Docstrings para funciones publicas

```python
def calculate_thermal_front(
    sst_grid: np.ndarray,
    threshold: float = 0.45
) -> np.ndarray:
    """
    Detecta frentes termicos usando algoritmo Belkin-O'Reilly.

    Args:
        sst_grid: Matriz 2D de SST en grados Celsius
        threshold: Diferencia minima para detectar frente (default 0.45C)

    Returns:
        Matriz binaria con 1 donde hay frente termico

    References:
        Belkin & O'Reilly (2009) - An algorithm for oceanic front detection
    """
    ...
```

### 8.2 NO documentar codigo obvio

```python
# INCORRECTO - Comentario obvio
# Incrementa el contador
counter += 1

# CORRECTO - Sin comentario innecesario
counter += 1

# CORRECTO - Comentario que explica el "por que"
# Umbral de 0.45C basado en Belkin-O'Reilly 2009
if gradient > 0.45:
    is_front = True
```

---

## 9. Testing

### 9.1 Estructura de tests

```
tests/
├── test_domain.py       # Tests de constantes
├── test_features.py     # Tests de extraccion
├── test_predictor.py    # Tests de ML
└── conftest.py          # Fixtures compartidos
```

### 9.2 Naming convention

```python
def test_species_temp_score_in_range():
    """Score debe ser 1.0 cuando SST esta en rango optimo."""
    ...

def test_species_temp_score_out_of_range():
    """Score debe ser <1.0 cuando SST esta fuera de rango."""
    ...
```

---

## 10. Performance

### 10.1 Evitar recalculos

```python
# INCORRECTO - Recalcula len() en cada iteracion
for i in range(len(points)):
    process(points[i])

# CORRECTO - Calcula una vez o usa enumerate
n_points = len(points)
for i in range(n_points):
    process(points[i])

# MEJOR - Usa enumerate o iteracion directa
for point in points:
    process(point)
```

### 10.2 Lazy evaluation

```python
# CORRECTO - Generador para datos grandes
def iter_marine_points(coords: Iterable[Tuple[float, float]]):
    for lat, lon in coords:
        yield fetch_point(lat, lon)

# USO - No carga todo en memoria
for point in iter_marine_points(coordinates):
    process(point)
```

---

## 11. Checklist Pre-Commit

Antes de hacer commit, verificar:

- [ ] `domain.py` no tiene imports circulares
- [ ] No hay constantes hardcodeadas fuera de `domain.py`
- [ ] No hay datos sinteticos
- [ ] Todas las funciones publicas tienen type hints
- [ ] Tests pasan: `python -m pytest tests/`
- [ ] Imports ordenados correctamente
- [ ] No hay variables intermedias innecesarias
- [ ] Estructuras de datos son inmutables donde sea posible

---

## 12. Referencias

- [PEP 8 - Style Guide](https://peps.python.org/pep-0008/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

---

*Lineamientos establecidos: 2026-01-28*
