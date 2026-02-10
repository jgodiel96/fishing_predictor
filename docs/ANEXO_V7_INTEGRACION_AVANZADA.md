# ANEXO V7: Integración Avanzada de Fuentes de Datos

**Versión:** 7.0
**Fecha:** 2026-02-08
**Estado:** EN IMPLEMENTACIÓN
**Autor:** Sistema de Predicción Pesquera

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura de Alta Eficiencia](#2-arquitectura-de-alta-eficiencia)
3. [Estándares de Programación](#3-estándares-de-programación)
4. [Integración 1: Clorofila-a](#4-integración-1-clorofila-a)
5. [Integración 2: SST Histórico](#5-integración-2-sst-histórico)
6. [Integración 3: Hotspots Dinámicos GFW](#6-integración-3-hotspots-dinámicos-gfw)
7. [Sistema de Scoring Unificado V7](#7-sistema-de-scoring-unificado-v7)
8. [Validación y Testing](#8-validación-y-testing)
9. [Métricas de Rendimiento](#9-métricas-de-rendimiento)

---

## 1. Resumen Ejecutivo

### 1.1 Objetivo

Integrar tres nuevas fuentes de datos al sistema de predicción pesquera:

| Fuente | Propósito | Impacto en Score |
|--------|-----------|------------------|
| **Clorofila-a** | Indicador de productividad primaria | ±8 puntos |
| **SST Histórico** | Tendencias térmicas y anomalías | ±6 puntos |
| **Hotspots GFW** | Zonas de actividad pesquera real | ±10 puntos |

### 1.2 Beneficios Esperados

- **Precisión:** +15-20% en correlación score-CPUE
- **Cobertura temporal:** Análisis de patrones históricos 2020-2026
- **Validación:** Hotspots basados en datos reales de pesca comercial

### 1.3 Prerequisitos

```
✅ V1-V6 completados
✅ Credenciales Copernicus configuradas
✅ Datos GFW descargados (1,085 registros)
✅ SST histórico descargado (354,362 registros)
```

---

## 2. Arquitectura de Alta Eficiencia

### 2.1 Principios de Diseño

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRINCIPIOS DE EFICIENCIA                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. LAZY LOADING                                                    │
│     ├── Cargar datos solo cuando se necesitan                      │
│     ├── Usar generadores para datasets grandes                     │
│     └── Implementar cache con TTL (Time-To-Live)                   │
│                                                                     │
│  2. MEMORY EFFICIENCY                                               │
│     ├── Usar tipos de datos óptimos (float32 vs float64)          │
│     ├── Liberar memoria explícitamente después de uso             │
│     └── Procesar en chunks para archivos > 100MB                   │
│                                                                     │
│  3. COMPUTATIONAL EFFICIENCY                                        │
│     ├── Vectorizar operaciones con NumPy                           │
│     ├── Usar índices espaciales (KD-Tree, R-Tree)                 │
│     ├── Pre-calcular valores frecuentes                            │
│     └── Evitar loops Python en datos > 1000 elementos             │
│                                                                     │
│  4. I/O EFFICIENCY                                                  │
│     ├── Usar Parquet sobre CSV (10x más rápido)                   │
│     ├── Comprimir con snappy (balance velocidad/tamaño)           │
│     ├── Batch writes en lugar de escrituras individuales          │
│     └── Connection pooling para APIs                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Flujo de Datos Optimizado

```
                    ┌──────────────────┐
                    │   BRONZE LAYER   │
                    │   (Inmutable)    │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Clorofila-a  │   │ SST Histórico │   │   GFW Data    │
│   Parquet     │   │   Parquet     │   │   Parquet     │
│  ~50MB/año    │   │  ~200MB/año   │   │   ~5MB/año    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ SILVER LAYER  │
                    │  (Procesado)  │
                    │               │
                    │ ┌───────────┐ │
                    │ │ Spatial   │ │
                    │ │ Index     │ │
                    │ │ (KD-Tree) │ │
                    │ └───────────┘ │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  GOLD LAYER   │
                    │  (Analytics)  │
                    │               │
                    │ Features +    │
                    │ Scores        │
                    └───────────────┘
```

### 2.3 Gestión de Caché

```python
# Configuración de caché multinivel
CACHE_CONFIG = {
    'L1_memory': {
        'max_size_mb': 256,
        'ttl_seconds': 300,      # 5 minutos
        'eviction': 'LRU'
    },
    'L2_disk': {
        'max_size_mb': 1024,
        'ttl_seconds': 3600,     # 1 hora
        'location': 'data/cache/'
    }
}
```

---

## 3. Estándares de Programación

### 3.1 Tipos de Variables y Eficiencia de Memoria

#### 3.1.1 Guía de Tipos de Datos

| Dato | Tipo Recomendado | Memoria | Justificación |
|------|------------------|---------|---------------|
| Latitud/Longitud | `np.float32` | 4 bytes | Precisión suficiente (6 decimales) |
| Temperatura (SST) | `np.float32` | 4 bytes | Rango -5 a 35°C |
| Clorofila-a | `np.float32` | 4 bytes | Rango 0.01 a 50 mg/m³ |
| Score (0-100) | `np.float32` | 4 bytes | Precisión decimal suficiente |
| Fechas | `np.datetime64[D]` | 8 bytes | Resolución diaria |
| Horas | `np.uint8` | 1 byte | Rango 0-23 |
| Flags booleanos | `np.bool_` | 1 byte | True/False |
| Contadores | `np.uint32` | 4 bytes | Hasta 4.3 mil millones |
| Índices espaciales | `np.int32` | 4 bytes | Suficiente para grids |

#### 3.1.2 Conversiones Obligatorias

```python
# ❌ INCORRECTO - Usa float64 por defecto (8 bytes)
df['lat'] = df['lat'].astype(float)

# ✅ CORRECTO - Especifica float32 (4 bytes)
df['lat'] = df['lat'].astype(np.float32)

# ❌ INCORRECTO - String para categorías
df['species'] = 'Cabrilla'

# ✅ CORRECTO - Categorical para valores repetidos
df['species'] = pd.Categorical(df['species'])

# ❌ INCORRECTO - datetime64[ns] completo
df['date'] = pd.to_datetime(df['date'])

# ✅ CORRECTO - datetime64[D] para fechas sin hora
df['date'] = pd.to_datetime(df['date']).dt.floor('D')
```

#### 3.1.3 Optimización de DataFrames

```python
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiza tipos de datos para reducir uso de memoria.

    Reducción típica: 60-80% del tamaño original.
    """
    optimized = df.copy()

    for col in optimized.columns:
        col_type = optimized[col].dtype

        # Floats: usar float32
        if col_type == np.float64:
            optimized[col] = optimized[col].astype(np.float32)

        # Integers: usar el tipo más pequeño posible
        elif col_type == np.int64:
            c_min, c_max = optimized[col].min(), optimized[col].max()
            if c_min >= 0:
                if c_max < 255:
                    optimized[col] = optimized[col].astype(np.uint8)
                elif c_max < 65535:
                    optimized[col] = optimized[col].astype(np.uint16)
                else:
                    optimized[col] = optimized[col].astype(np.uint32)
            else:
                if c_min > -128 and c_max < 127:
                    optimized[col] = optimized[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    optimized[col] = optimized[col].astype(np.int16)
                else:
                    optimized[col] = optimized[col].astype(np.int32)

        # Objects: convertir a categorical si hay pocos valores únicos
        elif col_type == object:
            num_unique = optimized[col].nunique()
            num_total = len(optimized[col])
            if num_unique / num_total < 0.5:  # < 50% valores únicos
                optimized[col] = pd.Categorical(optimized[col])

    return optimized
```

### 3.2 Organización de Código

#### 3.2.1 Estructura de Módulos

```
data/fetchers/
├── __init__.py                          # Exports públicos
├── base_fetcher.py                      # Clase base abstracta
├── copernicus_chlorophyll_fetcher.py    # NUEVO
├── copernicus_physics_fetcher.py        # Existente (SSS, SLA)
├── sst_historical_provider.py           # NUEVO
├── gfw_hotspot_generator.py             # NUEVO
└── _utils.py                            # Funciones privadas compartidas
```

#### 3.2.2 Clase Base para Fetchers

```python
"""
base_fetcher.py - Clase base abstracta para todos los fetchers.

Implementa el patrón Template Method para garantizar consistencia
y eficiencia en todos los fetchers de datos.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Generator
from pathlib import Path
import numpy as np
import pandas as pd
from functools import lru_cache

@dataclass(frozen=True, slots=True)
class FetcherConfig:
    """
    Configuración inmutable para fetchers.

    Usar slots=True reduce memoria ~40% vs dataclass normal.
    Usar frozen=True permite hashear para caché.
    """
    source_name: str
    output_dir: Path
    cache_ttl_seconds: int = 3600
    chunk_size: int = 10000
    compression: str = 'snappy'


class BaseFetcher(ABC):
    """
    Clase base abstracta para fetchers de datos oceanográficos.

    Responsabilidades:
    - Gestión de caché L1 (memoria)
    - Logging estandarizado
    - Manejo de errores consistente
    - Optimización de tipos de datos

    Subclases deben implementar:
    - _download_impl(): Lógica específica de descarga
    - _parse_impl(): Conversión a formato estándar
    - calculate_score(): Normalización a 0-1
    """

    __slots__ = ('config', '_cache', '_manifest')

    def __init__(self, config: FetcherConfig):
        self.config = config
        self._cache: dict = {}
        self._manifest = None  # Lazy load

    @abstractmethod
    def _download_impl(self, year: int, month: int) -> bytes:
        """Implementación específica de descarga."""
        pass

    @abstractmethod
    def _parse_impl(self, raw_data: bytes) -> pd.DataFrame:
        """Parseo específico del formato de datos."""
        pass

    @abstractmethod
    def calculate_score(self, value: float) -> float:
        """Calcula score normalizado 0-1 para el valor."""
        pass

    def download_month(self, year: int, month: int) -> bool:
        """
        Descarga datos de un mes (Template Method).

        Flujo:
        1. Verificar si ya existe (idempotencia)
        2. Descargar datos crudos
        3. Parsear a DataFrame
        4. Optimizar tipos
        5. Guardar como Parquet
        6. Actualizar manifest

        Returns:
            bool: True si exitoso o ya existía
        """
        output_path = self._get_output_path(year, month)

        # Idempotencia
        if output_path.exists():
            return True

        try:
            # Descargar
            raw_data = self._download_impl(year, month)

            # Parsear
            df = self._parse_impl(raw_data)

            # Optimizar tipos
            df = self._optimize_types(df)

            # Guardar
            df.to_parquet(
                output_path,
                compression=self.config.compression,
                index=False
            )

            # Actualizar manifest
            self._update_manifest(output_path, len(df), year, month)

            return True

        except Exception as e:
            self._log_error(f"Error descargando {year}-{month}: {e}")
            return False

    @lru_cache(maxsize=128)
    def get_value_for_location(
        self,
        date: str,
        lat: float,
        lon: float,
        radius_deg: float = 0.1
    ) -> Optional[float]:
        """
        Obtiene valor para ubicación específica con caché LRU.

        Args:
            date: Fecha ISO (YYYY-MM-DD)
            lat: Latitud (-90 a 90)
            lon: Longitud (-180 a 180)
            radius_deg: Radio de búsqueda en grados (~11km por 0.1°)

        Returns:
            Valor promedio en el área o None si no hay datos
        """
        # Convertir a tipos hashables para caché
        date_key = str(date)
        lat_key = round(lat, 4)
        lon_key = round(lon, 4)

        # Cargar datos del mes
        year, month = int(date_key[:4]), int(date_key[5:7])
        df = self._load_month_data(year, month)

        if df is None or df.empty:
            return None

        # Filtrar por bbox (más eficiente que distancia)
        mask = (
            (df['lat'] >= lat_key - radius_deg) &
            (df['lat'] <= lat_key + radius_deg) &
            (df['lon'] >= lon_key - radius_deg) &
            (df['lon'] <= lon_key + radius_deg)
        )

        filtered = df.loc[mask, 'value']

        if filtered.empty:
            return None

        return float(filtered.mean())

    def _optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza tipos de datos del DataFrame."""
        # Coordenadas a float32
        for col in ['lat', 'lon', 'value']:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        # Fechas a datetime64[D]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.floor('D')

        return df

    def _get_output_path(self, year: int, month: int) -> Path:
        """Genera ruta de salida estandarizada."""
        return self.config.output_dir / f"{year}-{month:02d}.parquet"

    def _load_month_data(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de un mes con caché."""
        cache_key = f"{year}-{month:02d}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_output_path(year, month)
        if not path.exists():
            return None

        df = pd.read_parquet(path)

        # Limitar tamaño de caché
        if len(self._cache) > 12:  # Máximo 12 meses en memoria
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = df
        return df
```

### 3.3 Patrones de Eficiencia

#### 3.3.1 Vectorización con NumPy

```python
# ❌ INCORRECTO - Loop Python (lento)
def calculate_distances_slow(points, target):
    distances = []
    for p in points:
        d = haversine(p[0], p[1], target[0], target[1])
        distances.append(d)
    return distances

# ✅ CORRECTO - Vectorizado con NumPy (100x más rápido)
def calculate_distances_fast(
    points: np.ndarray,  # shape (N, 2)
    target: np.ndarray   # shape (2,)
) -> np.ndarray:
    """
    Calcula distancia Haversine vectorizada.

    Para N=10000 puntos:
    - Loop Python: ~500ms
    - Vectorizado: ~5ms
    """
    lat1 = np.radians(points[:, 0])
    lon1 = np.radians(points[:, 1])
    lat2 = np.radians(target[0])
    lon2 = np.radians(target[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371.0 * c  # km
```

#### 3.3.2 Índices Espaciales (KD-Tree)

```python
from scipy.spatial import cKDTree
from typing import List, Tuple
import numpy as np

class SpatialIndex:
    """
    Índice espacial para búsquedas eficientes.

    Complejidad:
    - Construcción: O(N log N)
    - Búsqueda: O(log N) vs O(N) búsqueda lineal

    Para N=100000 puntos, 1000 queries:
    - Lineal: ~10 segundos
    - KD-Tree: ~0.1 segundos
    """

    __slots__ = ('_tree', '_data', '_coords')

    def __init__(self, data: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lon'):
        """
        Construye índice espacial.

        Args:
            data: DataFrame con coordenadas
            lat_col: Nombre de columna de latitud
            lon_col: Nombre de columna de longitud
        """
        self._data = data
        self._coords = data[[lat_col, lon_col]].values.astype(np.float32)
        self._tree = cKDTree(self._coords)

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float
    ) -> pd.DataFrame:
        """
        Busca puntos dentro de radio.

        Args:
            lat: Latitud del centro
            lon: Longitud del centro
            radius_km: Radio en kilómetros

        Returns:
            DataFrame con puntos encontrados
        """
        # Convertir km a grados (aproximación)
        radius_deg = radius_km / 111.0

        indices = self._tree.query_ball_point([lat, lon], radius_deg)

        if not indices:
            return pd.DataFrame()

        return self._data.iloc[indices]

    def query_nearest(
        self,
        lat: float,
        lon: float,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encuentra k vecinos más cercanos.

        Returns:
            (distancias, índices)
        """
        distances, indices = self._tree.query([lat, lon], k=k)
        return distances * 111.0, indices  # Convertir a km
```

#### 3.3.3 Procesamiento por Chunks

```python
def process_large_file(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 50000
) -> None:
    """
    Procesa archivos grandes sin cargar todo en memoria.

    Para archivo de 1GB:
    - Sin chunks: ~8GB RAM, puede fallar
    - Con chunks 50k: ~200MB RAM, estable
    """
    # Usar generador para leer
    chunks = pd.read_parquet(
        input_path,
        columns=['lat', 'lon', 'value', 'date']  # Solo columnas necesarias
    ).pipe(lambda df: np.array_split(df, max(1, len(df) // chunk_size)))

    processed_chunks = []

    for i, chunk in enumerate(chunks):
        # Procesar chunk
        processed = process_chunk(chunk)
        processed_chunks.append(processed)

        # Liberar memoria del chunk original
        del chunk

        # Cada 10 chunks, forzar garbage collection
        if i % 10 == 0:
            import gc
            gc.collect()

    # Concatenar y guardar
    result = pd.concat(processed_chunks, ignore_index=True)
    result.to_parquet(output_path, compression='snappy')
```

### 3.4 Convenciones de Nomenclatura

```python
# Módulos: snake_case
copernicus_chlorophyll_fetcher.py

# Clases: PascalCase
class ChlorophyllFetcher:
    pass

# Funciones/métodos: snake_case
def calculate_chla_score():
    pass

# Constantes: SCREAMING_SNAKE_CASE
CHLA_OPTIMAL_MIN = 2.0
CHLA_OPTIMAL_MAX = 10.0

# Variables privadas: prefijo _
_cache = {}

# Variables de instancia en __slots__: sin prefijo
__slots__ = ('config', 'data', 'index')

# Type hints obligatorios en funciones públicas
def get_value(self, lat: float, lon: float) -> Optional[float]:
    pass
```

---

## 4. Integración 1: Clorofila-a

### 4.1 Fundamento Científico

La **Clorofila-a (Chl-a)** es el principal indicador de productividad primaria marina:

```
Clorofila-a Alta → Fitoplancton Abundante → Zooplancton → Peces Forrajeros → Depredadores
```

#### 4.1.1 Rangos Óptimos (IMARPE/FAO)

| Rango (mg/m³) | Clasificación | Score | Justificación |
|---------------|---------------|-------|---------------|
| 0.0 - 0.5 | Oligotrófico | 0.2 | Baja productividad |
| 0.5 - 1.0 | Bajo | 0.4 | Productividad limitada |
| 1.0 - 2.0 | Moderado | 0.6 | Condiciones aceptables |
| **2.0 - 10.0** | **Óptimo** | **0.9** | **Máxima actividad pesquera** |
| 10.0 - 20.0 | Alto (bloom) | 0.7 | Posible HAB* |
| > 20.0 | Bloom intenso | 0.3 | Riesgo HAB, evitar |

*HAB = Harmful Algal Bloom

### 4.2 Dataset Copernicus

```yaml
Product: cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D
Variable: CHL (mass_concentration_of_chlorophyll_a_in_sea_water)
Unidad: mg/m³ (miligramos por metro cúbico)
Resolución: 4km x 4km
Frecuencia: Diaria
Cobertura: Global
Latencia: ~5 días
```

### 4.3 Implementación del Fetcher

```python
# Archivo: data/fetchers/copernicus_chlorophyll_fetcher.py

"""
Fetcher para datos de Clorofila-a de Copernicus Marine.

Descarga datos diarios de concentración de clorofila-a
y los procesa para integración con el sistema de scoring.

Uso:
    fetcher = ChlorophyllFetcher()
    fetcher.download_range(2024, 1, 2024, 12)
    value = fetcher.get_value_for_location('2024-06-15', -17.8, -71.2)
    score = fetcher.calculate_score(value)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from data.data_config import DataConfig
from data.manifest import ManifestManager

logger = logging.getLogger(__name__)


# Constantes de dominio
CHLA_OPTIMAL_MIN: float = 2.0   # mg/m³
CHLA_OPTIMAL_MAX: float = 10.0  # mg/m³
CHLA_BLOOM_THRESHOLD: float = 20.0
CHLA_OLIGOTROPHIC: float = 0.5

# Configuración del dataset
COPERNICUS_CHLA_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
COPERNICUS_CHLA_VARIABLE = "CHL"


@dataclass(frozen=True, slots=True)
class ChlaConfig:
    """Configuración inmutable para el fetcher de Clorofila-a."""
    output_dir: Path
    region_lat_min: float
    region_lat_max: float
    region_lon_min: float
    region_lon_max: float
    cache_ttl_seconds: int = 3600


class ChlorophyllFetcher:
    """
    Fetcher para datos de Clorofila-a de Copernicus Marine.

    Atributos:
        config: Configuración del fetcher
        _spatial_index: Índice KD-Tree para búsquedas eficientes
        _cache: Caché de datos mensuales

    Ejemplo:
        >>> fetcher = ChlorophyllFetcher()
        >>> fetcher.download_month(2024, 6)
        True
        >>> value = fetcher.get_value_for_location('2024-06-15', -17.8, -71.2)
        >>> print(f"Chl-a: {value:.2f} mg/m³")
        Chl-a: 3.45 mg/m³
        >>> score = fetcher.calculate_score(value)
        >>> print(f"Score: {score:.2f}")
        Score: 0.90
    """

    __slots__ = ('config', '_cache', '_manifest', '_spatial_index')

    def __init__(self, config: Optional[ChlaConfig] = None):
        """
        Inicializa el fetcher.

        Args:
            config: Configuración opcional. Si no se provee,
                   usa valores por defecto de DataConfig.
        """
        if config is None:
            config = ChlaConfig(
                output_dir=DataConfig.RAW_DIR / "chla" / "copernicus",
                region_lat_min=DataConfig.REGION['lat_min'],
                region_lat_max=DataConfig.REGION['lat_max'],
                region_lon_min=DataConfig.REGION['lon_min'],
                region_lon_max=DataConfig.REGION['lon_max']
            )

        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._manifest = ManifestManager('copernicus_chla')
        self._spatial_index = None

        # Crear directorio si no existe
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def download_month(self, year: int, month: int) -> bool:
        """
        Descarga datos de Clorofila-a para un mes.

        Args:
            year: Año (2020-2026)
            month: Mes (1-12)

        Returns:
            True si exitoso o ya existía, False si error

        Raises:
            ValueError: Si año/mes fuera de rango válido
        """
        # Validar entrada
        if not (2020 <= year <= 2030):
            raise ValueError(f"Año {year} fuera de rango válido (2020-2030)")
        if not (1 <= month <= 12):
            raise ValueError(f"Mes {month} fuera de rango válido (1-12)")

        output_path = self._get_output_path(year, month)

        # Idempotencia - no re-descargar
        if output_path.exists():
            logger.debug(f"Archivo ya existe: {output_path}")
            return True

        try:
            import copernicusmarine

            # Definir rango de fechas
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)

            logger.info(f"Descargando Chl-a {year}-{month:02d}...")

            # Descargar a archivo temporal
            temp_nc = self.config.output_dir / f"temp_{year}_{month}.nc"

            copernicusmarine.subset(
                dataset_id=COPERNICUS_CHLA_DATASET,
                variables=[COPERNICUS_CHLA_VARIABLE],
                start_datetime=start_date.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=end_date.strftime("%Y-%m-%dT23:59:59"),
                minimum_latitude=self.config.region_lat_min - 0.5,
                maximum_latitude=self.config.region_lat_max + 0.5,
                minimum_longitude=self.config.region_lon_min - 0.5,
                maximum_longitude=self.config.region_lon_max + 0.5,
                output_filename=str(temp_nc),
                output_directory=str(self.config.output_dir),
                force_download=True
            )

            # Convertir NetCDF a Parquet optimizado
            df = self._netcdf_to_dataframe(temp_nc)

            # Guardar como Parquet
            df.to_parquet(
                output_path,
                compression='snappy',
                index=False
            )

            # Limpiar archivo temporal
            temp_nc.unlink(missing_ok=True)

            # Actualizar manifest
            self._manifest.add_download(
                filename=output_path.name,
                period_start=start_date.strftime("%Y-%m-%d"),
                period_end=end_date.strftime("%Y-%m-%d"),
                records=len(df),
                source_url=f"copernicus:{COPERNICUS_CHLA_DATASET}"
            )
            self._manifest.save()

            logger.info(f"✅ Descargado: {output_path.name} ({len(df)} registros)")
            return True

        except Exception as e:
            logger.error(f"❌ Error descargando Chl-a {year}-{month}: {e}")
            return False

    def _netcdf_to_dataframe(self, nc_path: Path) -> pd.DataFrame:
        """
        Convierte NetCDF a DataFrame optimizado.

        Args:
            nc_path: Ruta al archivo NetCDF

        Returns:
            DataFrame con columnas [date, lat, lon, value]
        """
        ds = xr.open_dataset(nc_path)

        # Extraer variable
        var_name = COPERNICUS_CHLA_VARIABLE
        if var_name not in ds:
            # Buscar nombre alternativo
            var_name = [v for v in ds.data_vars if 'chl' in v.lower()][0]

        # Convertir a DataFrame
        df = ds[var_name].to_dataframe().reset_index()

        # Renombrar columnas estándar
        col_mapping = {
            'time': 'date',
            'latitude': 'lat',
            'longitude': 'lon',
            var_name: 'value'
        }
        df = df.rename(columns=col_mapping)

        # Filtrar NaN y valores inválidos
        df = df.dropna(subset=['value'])
        df = df[df['value'] > 0]  # Chl-a siempre positiva
        df = df[df['value'] < 100]  # Filtrar outliers extremos

        # Optimizar tipos
        df['lat'] = df['lat'].astype(np.float32)
        df['lon'] = df['lon'].astype(np.float32)
        df['value'] = df['value'].astype(np.float32)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')

        ds.close()

        return df

    @lru_cache(maxsize=128)
    def get_value_for_location(
        self,
        date: str,
        lat: float,
        lon: float,
        radius_deg: float = 0.05
    ) -> Optional[float]:
        """
        Obtiene valor de Clorofila-a para ubicación y fecha.

        Usa caché LRU para queries repetidas.

        Args:
            date: Fecha ISO (YYYY-MM-DD)
            lat: Latitud
            lon: Longitud
            radius_deg: Radio de búsqueda (~5.5km por 0.05°)

        Returns:
            Valor promedio de Chl-a en mg/m³ o None
        """
        year = int(date[:4])
        month = int(date[5:7])

        df = self._load_month_data(year, month)

        if df is None or df.empty:
            return None

        # Filtrar por fecha
        target_date = pd.Timestamp(date)
        date_mask = df['date'] == target_date

        # Filtrar por ubicación (bbox)
        spatial_mask = (
            (df['lat'] >= lat - radius_deg) &
            (df['lat'] <= lat + radius_deg) &
            (df['lon'] >= lon - radius_deg) &
            (df['lon'] <= lon + radius_deg)
        )

        values = df.loc[date_mask & spatial_mask, 'value']

        if values.empty:
            # Fallback: usar promedio del mes si no hay dato exacto
            values = df.loc[spatial_mask, 'value']

        if values.empty:
            return None

        return float(values.mean())

    @staticmethod
    def calculate_score(chla: Optional[float]) -> float:
        """
        Calcula score de pesca basado en Clorofila-a.

        Basado en literatura IMARPE y FAO para pesquerías
        del Pacífico Sureste.

        Args:
            chla: Concentración de Clorofila-a en mg/m³

        Returns:
            Score normalizado 0-1

        Ejemplo:
            >>> ChlorophyllFetcher.calculate_score(5.0)
            0.9
            >>> ChlorophyllFetcher.calculate_score(0.3)
            0.2
            >>> ChlorophyllFetcher.calculate_score(25.0)
            0.3
        """
        if chla is None:
            return 0.5  # Neutral si no hay dato

        # Rango óptimo: 2-10 mg/m³
        if CHLA_OPTIMAL_MIN <= chla <= CHLA_OPTIMAL_MAX:
            return 0.9

        # Moderado: 1-2 mg/m³
        if 1.0 <= chla < CHLA_OPTIMAL_MIN:
            return 0.6 + 0.3 * (chla - 1.0) / (CHLA_OPTIMAL_MIN - 1.0)

        # Alto pero no bloom: 10-20 mg/m³
        if CHLA_OPTIMAL_MAX < chla <= CHLA_BLOOM_THRESHOLD:
            return 0.9 - 0.2 * (chla - CHLA_OPTIMAL_MAX) / (CHLA_BLOOM_THRESHOLD - CHLA_OPTIMAL_MAX)

        # Bajo: 0.5-1 mg/m³
        if CHLA_OLIGOTROPHIC <= chla < 1.0:
            return 0.4 + 0.2 * (chla - CHLA_OLIGOTROPHIC) / (1.0 - CHLA_OLIGOTROPHIC)

        # Oligotrófico: < 0.5 mg/m³
        if chla < CHLA_OLIGOTROPHIC:
            return 0.2

        # Bloom intenso: > 20 mg/m³ (posible HAB)
        return 0.3

    def _get_output_path(self, year: int, month: int) -> Path:
        """Genera ruta de salida estandarizada."""
        return self.config.output_dir / f"{year}-{month:02d}.parquet"

    def _load_month_data(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de un mes con caché."""
        cache_key = f"{year}-{month:02d}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_output_path(year, month)
        if not path.exists():
            # Intentar descargar
            if not self.download_month(year, month):
                return None

        df = pd.read_parquet(path)

        # Gestión de caché LRU simple
        if len(self._cache) >= 12:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[cache_key] = df
        return df

    def download_range(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> Tuple[int, int]:
        """
        Descarga rango de meses.

        Args:
            start_year: Año inicial
            start_month: Mes inicial
            end_year: Año final
            end_month: Mes final

        Returns:
            (exitosos, fallidos)
        """
        success = 0
        failed = 0

        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        while current <= end:
            if self.download_month(current.year, current.month):
                success += 1
            else:
                failed += 1

            # Siguiente mes
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        return success, failed


# Función de conveniencia para uso directo
def get_chla_score(date: str, lat: float, lon: float) -> float:
    """
    Función de conveniencia para obtener score de Chl-a.

    Args:
        date: Fecha ISO
        lat: Latitud
        lon: Longitud

    Returns:
        Score 0-1
    """
    fetcher = ChlorophyllFetcher()
    value = fetcher.get_value_for_location(date, lat, lon)
    return fetcher.calculate_score(value)
```

### 4.4 Integración en Score

```python
# En controllers/analysis.py - método analyze_spots()

# Obtener valor de Clorofila-a
chla_value = self.chla_fetcher.get_value_for_location(
    date=self.analysis_date,
    lat=spot['lat'],
    lon=spot['lon']
)
chla_score = ChlorophyllFetcher.calculate_score(chla_value)

# Bonus: ±8 puntos
chla_bonus = (chla_score - 0.5) * 16  # Rango: -8 a +8

spot['chla_value'] = chla_value
spot['chla_score'] = chla_score
spot['chla_bonus'] = chla_bonus
spot['score'] += chla_bonus
```

---

## 5. Integración 2: SST Histórico

### 5.1 Fundamento Científico

El **SST (Sea Surface Temperature)** histórico permite:

1. **Detectar anomalías térmicas** - Desviaciones del promedio histórico
2. **Identificar tendencias** - Calentamiento/enfriamiento estacional
3. **Validar frentes térmicos** - Comparar con promedios mensuales

### 5.2 Datos Disponibles

```yaml
Registros: 354,362
Período: 2020-01 a 2026-02
Fuente: Copernicus Marine (OSTIA)
Resolución: 0.05° (~5km)
Variables: sst, sst_anomaly
```

### 5.3 Implementación del Provider

```python
# Archivo: data/fetchers/sst_historical_provider.py

"""
Provider para SST histórico de Copernicus.

Aprovecha los 354,362 registros ya descargados para:
- Calcular anomalías térmicas vs promedio mensual
- Detectar tendencias de temperatura
- Mejorar la precisión de frentes térmicos

Uso:
    provider = SSTHistoricalProvider()
    sst, anomaly = provider.get_sst_with_anomaly('2024-06-15', -17.8, -71.2)
    trend = provider.get_temperature_trend('2024-06-15', -17.8, -71.2, days=30)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, NamedTuple
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.data_config import DataConfig

logger = logging.getLogger(__name__)


# Constantes de dominio
SST_OPTIMAL_MIN: float = 14.0  # °C
SST_OPTIMAL_MAX: float = 24.0  # °C
SST_OPTIMAL_CENTER: float = 19.0  # °C - óptimo para especies locales
ANOMALY_THRESHOLD_WARM: float = 1.5  # °C - anomalía cálida significativa
ANOMALY_THRESHOLD_COLD: float = -1.5  # °C - anomalía fría significativa


class SSTResult(NamedTuple):
    """Resultado de consulta SST."""
    sst: float
    anomaly: float
    monthly_mean: float
    trend_7d: float
    score: float


@dataclass(frozen=True, slots=True)
class SSTConfig:
    """Configuración inmutable para SST provider."""
    data_dir: Path
    monthly_stats_cache: Path
    grid_resolution_deg: float = 0.05


class SSTHistoricalProvider:
    """
    Provider para datos históricos de SST.

    Características:
    - Índice espacial KD-Tree para búsquedas O(log N)
    - Estadísticas mensuales pre-calculadas
    - Caché multinivel (memoria + disco)

    Performance:
    - Query individual: ~1ms
    - 1000 queries: ~100ms (vs ~10s sin índice)
    """

    __slots__ = (
        'config', '_data', '_spatial_index',
        '_monthly_stats', '_cache'
    )

    def __init__(self, config: Optional[SSTConfig] = None):
        """
        Inicializa el provider.

        Args:
            config: Configuración opcional
        """
        if config is None:
            config = SSTConfig(
                data_dir=DataConfig.RAW_SST_COPERNICUS,
                monthly_stats_cache=DataConfig.PROCESSED_DIR / "sst_monthly_stats.parquet"
            )

        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._spatial_index: Optional[cKDTree] = None
        self._monthly_stats: Optional[pd.DataFrame] = None
        self._cache: dict = {}

    def _ensure_loaded(self) -> None:
        """Carga datos si no están en memoria (lazy loading)."""
        if self._data is not None:
            return

        logger.info("Cargando SST histórico...")

        # Cargar todos los archivos parquet
        parquet_files = list(self.config.data_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No se encontraron archivos SST en {self.config.data_dir}"
            )

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(
                f,
                columns=['date', 'lat', 'lon', 'sst']  # Solo columnas necesarias
            )
            dfs.append(df)

        self._data = pd.concat(dfs, ignore_index=True)

        # Optimizar tipos
        self._data['lat'] = self._data['lat'].astype(np.float32)
        self._data['lon'] = self._data['lon'].astype(np.float32)
        self._data['sst'] = self._data['sst'].astype(np.float32)
        self._data['date'] = pd.to_datetime(self._data['date'])

        # Agregar columnas derivadas
        self._data['month'] = self._data['date'].dt.month.astype(np.uint8)
        self._data['year'] = self._data['date'].dt.year.astype(np.uint16)

        # Construir índice espacial
        coords = self._data[['lat', 'lon']].drop_duplicates().values
        self._spatial_index = cKDTree(coords)

        # Calcular estadísticas mensuales
        self._calculate_monthly_stats()

        logger.info(f"✅ SST cargado: {len(self._data):,} registros")

    def _calculate_monthly_stats(self) -> None:
        """Calcula estadísticas mensuales por ubicación."""
        cache_path = self.config.monthly_stats_cache

        if cache_path.exists():
            self._monthly_stats = pd.read_parquet(cache_path)
            return

        logger.info("Calculando estadísticas mensuales SST...")

        # Agrupar por ubicación y mes
        stats = self._data.groupby(
            ['lat', 'lon', 'month'],
            observed=True
        ).agg(
            sst_mean=('sst', 'mean'),
            sst_std=('sst', 'std'),
            sst_min=('sst', 'min'),
            sst_max=('sst', 'max'),
            count=('sst', 'count')
        ).reset_index()

        # Optimizar tipos
        stats['sst_mean'] = stats['sst_mean'].astype(np.float32)
        stats['sst_std'] = stats['sst_std'].fillna(0).astype(np.float32)
        stats['sst_min'] = stats['sst_min'].astype(np.float32)
        stats['sst_max'] = stats['sst_max'].astype(np.float32)
        stats['count'] = stats['count'].astype(np.uint16)

        # Guardar caché
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_parquet(cache_path, compression='snappy')

        self._monthly_stats = stats

    @lru_cache(maxsize=256)
    def get_sst_with_anomaly(
        self,
        date: str,
        lat: float,
        lon: float
    ) -> SSTResult:
        """
        Obtiene SST y anomalía para ubicación y fecha.

        Args:
            date: Fecha ISO (YYYY-MM-DD)
            lat: Latitud
            lon: Longitud

        Returns:
            SSTResult con sst, anomaly, monthly_mean, trend_7d, score
        """
        self._ensure_loaded()

        target_date = pd.Timestamp(date)
        month = target_date.month

        # Buscar punto más cercano
        _, idx = self._spatial_index.query([lat, lon])
        nearest_coords = self._spatial_index.data[idx]
        nearest_lat, nearest_lon = nearest_coords[0], nearest_coords[1]

        # Obtener SST del día
        mask = (
            (self._data['lat'] == nearest_lat) &
            (self._data['lon'] == nearest_lon) &
            (self._data['date'] == target_date)
        )

        daily_data = self._data.loc[mask, 'sst']

        if daily_data.empty:
            # Fallback: usar promedio del mes
            sst = self._get_monthly_mean(nearest_lat, nearest_lon, month)
            anomaly = 0.0
        else:
            sst = float(daily_data.iloc[0])
            monthly_mean = self._get_monthly_mean(nearest_lat, nearest_lon, month)
            anomaly = sst - monthly_mean

        # Calcular tendencia 7 días
        trend = self._calculate_trend(nearest_lat, nearest_lon, target_date, days=7)

        # Calcular score
        score = self.calculate_score(sst, anomaly)

        return SSTResult(
            sst=sst,
            anomaly=anomaly,
            monthly_mean=self._get_monthly_mean(nearest_lat, nearest_lon, month),
            trend_7d=trend,
            score=score
        )

    def _get_monthly_mean(self, lat: float, lon: float, month: int) -> float:
        """Obtiene promedio mensual histórico."""
        if self._monthly_stats is None:
            return SST_OPTIMAL_CENTER

        mask = (
            (self._monthly_stats['lat'] == lat) &
            (self._monthly_stats['lon'] == lon) &
            (self._monthly_stats['month'] == month)
        )

        result = self._monthly_stats.loc[mask, 'sst_mean']

        if result.empty:
            return SST_OPTIMAL_CENTER

        return float(result.iloc[0])

    def _calculate_trend(
        self,
        lat: float,
        lon: float,
        end_date: pd.Timestamp,
        days: int = 7
    ) -> float:
        """
        Calcula tendencia de temperatura.

        Returns:
            Cambio de temperatura en °C (positivo = calentamiento)
        """
        start_date = end_date - pd.Timedelta(days=days)

        mask = (
            (self._data['lat'] == lat) &
            (self._data['lon'] == lon) &
            (self._data['date'] >= start_date) &
            (self._data['date'] <= end_date)
        )

        period_data = self._data.loc[mask].sort_values('date')

        if len(period_data) < 2:
            return 0.0

        # Regresión lineal simple
        x = np.arange(len(period_data))
        y = period_data['sst'].values

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope * days)  # Cambio total en el período

        return 0.0

    @staticmethod
    def calculate_score(sst: float, anomaly: float) -> float:
        """
        Calcula score combinado de SST y anomalía.

        Args:
            sst: Temperatura en °C
            anomaly: Desviación del promedio mensual en °C

        Returns:
            Score normalizado 0-1
        """
        # Score base por temperatura absoluta
        if SST_OPTIMAL_MIN <= sst <= SST_OPTIMAL_MAX:
            # Dentro del rango óptimo
            dist_from_center = abs(sst - SST_OPTIMAL_CENTER)
            range_half = (SST_OPTIMAL_MAX - SST_OPTIMAL_MIN) / 2
            base_score = 1.0 - (dist_from_center / range_half) * 0.2
        else:
            # Fuera del rango óptimo
            if sst < SST_OPTIMAL_MIN:
                dist = SST_OPTIMAL_MIN - sst
            else:
                dist = sst - SST_OPTIMAL_MAX
            base_score = max(0.3, 0.7 - dist * 0.1)

        # Ajuste por anomalía
        anomaly_factor = 1.0
        if anomaly > ANOMALY_THRESHOLD_WARM:
            # Anomalía cálida - puede ser negativo para pesca
            anomaly_factor = 0.85
        elif anomaly < ANOMALY_THRESHOLD_COLD:
            # Anomalía fría - upwelling, generalmente bueno
            anomaly_factor = 1.1

        return min(1.0, max(0.0, base_score * anomaly_factor))

    def get_temperature_trend(
        self,
        date: str,
        lat: float,
        lon: float,
        days: int = 30
    ) -> dict:
        """
        Obtiene análisis de tendencia extendido.

        Returns:
            Dict con trend, classification, confidence
        """
        self._ensure_loaded()

        target_date = pd.Timestamp(date)

        _, idx = self._spatial_index.query([lat, lon])
        nearest_lat, nearest_lon = self._spatial_index.data[idx]

        trend = self._calculate_trend(nearest_lat, nearest_lon, target_date, days)

        # Clasificar tendencia
        if trend > 1.0:
            classification = "warming_strong"
        elif trend > 0.3:
            classification = "warming_moderate"
        elif trend < -1.0:
            classification = "cooling_strong"
        elif trend < -0.3:
            classification = "cooling_moderate"
        else:
            classification = "stable"

        return {
            'trend_celsius': trend,
            'classification': classification,
            'period_days': days,
            'confidence': min(1.0, days / 30)  # Mayor confianza con más días
        }


# Función de conveniencia
def get_sst_historical_score(date: str, lat: float, lon: float) -> float:
    """Obtiene score SST histórico."""
    provider = SSTHistoricalProvider()
    result = provider.get_sst_with_anomaly(date, lat, lon)
    return result.score
```

### 5.4 Integración en Score

```python
# En controllers/analysis.py

# Obtener SST histórico con anomalía
sst_result = self.sst_provider.get_sst_with_anomaly(
    date=self.analysis_date,
    lat=spot['lat'],
    lon=spot['lon']
)

# Bonus: ±6 puntos
sst_historical_bonus = (sst_result.score - 0.5) * 12

spot['sst_historical'] = sst_result.sst
spot['sst_anomaly'] = sst_result.anomaly
spot['sst_trend'] = sst_result.trend_7d
spot['sst_historical_bonus'] = sst_historical_bonus
spot['score'] += sst_historical_bonus
```

---

## 6. Integración 3: Hotspots Dinámicos GFW

### 6.1 Fundamento

Los datos de **Global Fishing Watch (GFW)** contienen:
- Ubicaciones reales de actividad pesquera
- Horas de pesca por ubicación
- Tipos de embarcación

Usar DBSCAN para identificar clusters de alta actividad.

### 6.2 Algoritmo DBSCAN

```
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Ventajas:
- No requiere especificar número de clusters
- Detecta clusters de forma irregular
- Identifica outliers (ruido)

Parámetros:
- eps: Radio de vecindad (2km = 0.018°)
- min_samples: Mínimo de puntos para formar cluster (5)
```

### 6.3 Implementación

```python
# Archivo: data/fetchers/gfw_hotspot_generator.py

"""
Generador de hotspots dinámicos basado en datos GFW.

Usa clustering DBSCAN para identificar zonas de alta
actividad pesquera real a partir de datos AIS.

Uso:
    generator = GFWHotspotGenerator()
    hotspots = generator.generate_hotspots(min_fishing_hours=10)

    # Por temporada
    hotspots_summer = generator.generate_seasonal_hotspots(
        season='summer',
        min_samples=5
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, NamedTuple
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from data.data_config import DataConfig

logger = logging.getLogger(__name__)


# Constantes
EARTH_RADIUS_KM = 6371.0
DEFAULT_EPS_KM = 2.0  # Radio de vecindad
DEFAULT_MIN_SAMPLES = 5  # Mínimo puntos por cluster


class DynamicHotspot(NamedTuple):
    """Hotspot generado dinámicamente."""
    id: int
    lat: float
    lon: float
    fishing_hours: float
    vessel_count: int
    score: float
    source: str = "GFW_dynamic"
    season: Optional[str] = None


@dataclass(slots=True)
class HotspotConfig:
    """Configuración para generación de hotspots."""
    data_path: Path
    eps_km: float = DEFAULT_EPS_KM
    min_samples: int = DEFAULT_MIN_SAMPLES
    min_fishing_hours: float = 5.0
    output_cache: Path = field(default_factory=lambda: DataConfig.PROCESSED_DIR / "dynamic_hotspots.parquet")


class GFWHotspotGenerator:
    """
    Generador de hotspots pesqueros dinámicos.

    Usa datos reales de Global Fishing Watch y clustering DBSCAN
    para identificar zonas de alta actividad pesquera.

    Características:
    - Clustering espacial DBSCAN
    - Filtrado temporal por temporada
    - Scoring por intensidad de pesca
    - Caché de resultados
    """

    __slots__ = ('config', '_data', '_hotspots_cache')

    def __init__(self, config: Optional[HotspotConfig] = None):
        """
        Inicializa el generador.

        Args:
            config: Configuración opcional
        """
        if config is None:
            config = HotspotConfig(
                data_path=DataConfig.RAW_GFW
            )

        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._hotspots_cache: Dict[str, List[DynamicHotspot]] = {}

    def _ensure_loaded(self) -> None:
        """Carga datos GFW si no están en memoria."""
        if self._data is not None:
            return

        logger.info("Cargando datos GFW...")

        parquet_files = list(self.config.data_path.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No se encontraron archivos GFW en {self.config.data_path}"
            )

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            dfs.append(df)

        self._data = pd.concat(dfs, ignore_index=True)

        # Estandarizar nombres de columnas
        col_mapping = {
            'cell_ll_lat': 'lat',
            'cell_ll_lon': 'lon',
            'fishing_hours': 'fishing_hours',
            'mmsi_present': 'vessel_count'
        }

        for old, new in col_mapping.items():
            if old in self._data.columns and new not in self._data.columns:
                self._data[new] = self._data[old]

        # Asegurar columnas requeridas
        if 'fishing_hours' not in self._data.columns:
            if 'apparent_fishing_hours' in self._data.columns:
                self._data['fishing_hours'] = self._data['apparent_fishing_hours']
            else:
                self._data['fishing_hours'] = 1.0

        if 'vessel_count' not in self._data.columns:
            self._data['vessel_count'] = 1

        # Optimizar tipos
        self._data['lat'] = self._data['lat'].astype(np.float32)
        self._data['lon'] = self._data['lon'].astype(np.float32)
        self._data['fishing_hours'] = self._data['fishing_hours'].astype(np.float32)

        # Agregar columna de mes si hay fecha
        if 'date' in self._data.columns:
            self._data['date'] = pd.to_datetime(self._data['date'])
            self._data['month'] = self._data['date'].dt.month.astype(np.uint8)

        logger.info(f"✅ GFW cargado: {len(self._data):,} registros")

    def generate_hotspots(
        self,
        min_fishing_hours: Optional[float] = None,
        eps_km: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> List[DynamicHotspot]:
        """
        Genera hotspots usando clustering DBSCAN.

        Args:
            min_fishing_hours: Mínimo de horas de pesca para incluir punto
            eps_km: Radio de vecindad en km
            min_samples: Mínimo de puntos por cluster

        Returns:
            Lista de DynamicHotspot ordenados por score descendente
        """
        self._ensure_loaded()

        # Usar valores por defecto si no se especifican
        min_fishing_hours = min_fishing_hours or self.config.min_fishing_hours
        eps_km = eps_km or self.config.eps_km
        min_samples = min_samples or self.config.min_samples

        # Clave de caché
        cache_key = f"all_{min_fishing_hours}_{eps_km}_{min_samples}"
        if cache_key in self._hotspots_cache:
            return self._hotspots_cache[cache_key]

        # Filtrar por mínimo de horas
        filtered = self._data[
            self._data['fishing_hours'] >= min_fishing_hours
        ].copy()

        if filtered.empty:
            logger.warning("No hay datos después de filtrar por fishing_hours")
            return []

        # Preparar coordenadas para DBSCAN
        coords = filtered[['lat', 'lon']].values

        # Convertir eps de km a grados (aproximación)
        eps_deg = eps_km / 111.0

        # Ejecutar DBSCAN
        logger.info(f"Ejecutando DBSCAN (eps={eps_km}km, min_samples={min_samples})...")

        clustering = DBSCAN(
            eps=eps_deg,
            min_samples=min_samples,
            metric='euclidean',  # Aproximación válida para áreas pequeñas
            n_jobs=-1  # Usar todos los cores
        )

        filtered['cluster'] = clustering.fit_predict(coords)

        # Filtrar ruido (cluster = -1)
        clustered = filtered[filtered['cluster'] != -1]

        if clustered.empty:
            logger.warning("No se encontraron clusters")
            return []

        # Calcular centroides y métricas por cluster
        hotspots = []

        for cluster_id in clustered['cluster'].unique():
            cluster_data = clustered[clustered['cluster'] == cluster_id]

            # Centroide ponderado por fishing_hours
            total_hours = cluster_data['fishing_hours'].sum()
            weighted_lat = (
                cluster_data['lat'] * cluster_data['fishing_hours']
            ).sum() / total_hours
            weighted_lon = (
                cluster_data['lon'] * cluster_data['fishing_hours']
            ).sum() / total_hours

            # Métricas
            vessel_count = cluster_data['vessel_count'].sum() if 'vessel_count' in cluster_data else len(cluster_data)

            hotspots.append(DynamicHotspot(
                id=int(cluster_id),
                lat=float(weighted_lat),
                lon=float(weighted_lon),
                fishing_hours=float(total_hours),
                vessel_count=int(vessel_count),
                score=0.0,  # Se calcula después
                source="GFW_dynamic"
            ))

        # Calcular scores normalizados
        hotspots = self._calculate_scores(hotspots)

        # Ordenar por score descendente
        hotspots.sort(key=lambda h: h.score, reverse=True)

        # Guardar en caché
        self._hotspots_cache[cache_key] = hotspots

        logger.info(f"✅ Generados {len(hotspots)} hotspots dinámicos")

        return hotspots

    def generate_seasonal_hotspots(
        self,
        season: str,
        **kwargs
    ) -> List[DynamicHotspot]:
        """
        Genera hotspots filtrados por temporada.

        Args:
            season: 'summer' (Dic-Mar), 'winter' (Jun-Sep),
                   'spring' (Sep-Dic), 'autumn' (Mar-Jun)

        Returns:
            Lista de DynamicHotspot para la temporada
        """
        self._ensure_loaded()

        if 'month' not in self._data.columns:
            logger.warning("No hay datos temporales, usando todos los registros")
            return self.generate_hotspots(**kwargs)

        # Mapear temporada a meses
        season_months = {
            'summer': [12, 1, 2, 3],
            'autumn': [3, 4, 5, 6],
            'winter': [6, 7, 8, 9],
            'spring': [9, 10, 11, 12]
        }

        if season not in season_months:
            raise ValueError(f"Temporada inválida: {season}")

        months = season_months[season]

        # Filtrar datos por temporada
        original_data = self._data.copy()
        self._data = self._data[self._data['month'].isin(months)]

        try:
            hotspots = self.generate_hotspots(**kwargs)

            # Agregar información de temporada
            hotspots = [
                DynamicHotspot(
                    id=h.id,
                    lat=h.lat,
                    lon=h.lon,
                    fishing_hours=h.fishing_hours,
                    vessel_count=h.vessel_count,
                    score=h.score,
                    source=h.source,
                    season=season
                )
                for h in hotspots
            ]

        finally:
            # Restaurar datos originales
            self._data = original_data

        return hotspots

    def _calculate_scores(
        self,
        hotspots: List[DynamicHotspot]
    ) -> List[DynamicHotspot]:
        """
        Calcula scores normalizados para hotspots.

        Score basado en:
        - 70% fishing_hours (normalizado)
        - 30% vessel_count (normalizado)
        """
        if not hotspots:
            return []

        # Extraer métricas
        hours = np.array([h.fishing_hours for h in hotspots])
        vessels = np.array([h.vessel_count for h in hotspots])

        # Normalizar con min-max
        hours_norm = (hours - hours.min()) / (hours.max() - hours.min() + 1e-6)
        vessels_norm = (vessels - vessels.min()) / (vessels.max() - vessels.min() + 1e-6)

        # Score compuesto
        scores = 0.7 * hours_norm + 0.3 * vessels_norm

        # Escalar a 0.5-1.0 (nunca score bajo para un hotspot real)
        scores = 0.5 + scores * 0.5

        # Crear nuevos hotspots con scores
        return [
            DynamicHotspot(
                id=h.id,
                lat=h.lat,
                lon=h.lon,
                fishing_hours=h.fishing_hours,
                vessel_count=h.vessel_count,
                score=float(scores[i]),
                source=h.source,
                season=h.season
            )
            for i, h in enumerate(hotspots)
        ]

    def get_nearest_hotspot(
        self,
        lat: float,
        lon: float,
        hotspots: Optional[List[DynamicHotspot]] = None
    ) -> Tuple[Optional[DynamicHotspot], float]:
        """
        Encuentra el hotspot más cercano a una ubicación.

        Args:
            lat: Latitud del punto
            lon: Longitud del punto
            hotspots: Lista de hotspots (si None, genera nuevos)

        Returns:
            (hotspot_cercano, distancia_km)
        """
        if hotspots is None:
            hotspots = self.generate_hotspots()

        if not hotspots:
            return None, float('inf')

        # Construir KD-Tree
        coords = np.array([[h.lat, h.lon] for h in hotspots])
        tree = cKDTree(coords)

        # Buscar más cercano
        dist_deg, idx = tree.query([lat, lon])
        dist_km = dist_deg * 111.0  # Aproximación

        return hotspots[idx], dist_km

    def calculate_proximity_bonus(
        self,
        lat: float,
        lon: float,
        hotspots: Optional[List[DynamicHotspot]] = None,
        max_distance_km: float = 10.0
    ) -> float:
        """
        Calcula bonus por proximidad a hotspot dinámico.

        Args:
            lat: Latitud del punto
            lon: Longitud del punto
            hotspots: Lista de hotspots
            max_distance_km: Distancia máxima para bonus

        Returns:
            Bonus en puntos (0 a max_bonus)
        """
        nearest, distance = self.get_nearest_hotspot(lat, lon, hotspots)

        if nearest is None or distance > max_distance_km:
            return 0.0

        # Bonus decae linealmente con distancia
        proximity_factor = 1.0 - (distance / max_distance_km)

        # Escalar por score del hotspot
        bonus = proximity_factor * nearest.score * 10.0  # Max 10 puntos

        return round(bonus, 2)

    def export_to_geojson(
        self,
        hotspots: List[DynamicHotspot],
        output_path: Path
    ) -> None:
        """Exporta hotspots a GeoJSON."""
        import json

        features = []
        for h in hotspots:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [h.lon, h.lat]
                },
                "properties": {
                    "id": h.id,
                    "fishing_hours": h.fishing_hours,
                    "vessel_count": h.vessel_count,
                    "score": h.score,
                    "source": h.source,
                    "season": h.season
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Exportado: {output_path}")


# Función de conveniencia
def get_gfw_hotspot_bonus(lat: float, lon: float) -> float:
    """Obtiene bonus por proximidad a hotspot GFW."""
    generator = GFWHotspotGenerator()
    return generator.calculate_proximity_bonus(lat, lon)
```

### 6.4 Integración en Score

```python
# En controllers/analysis.py

# Generar hotspots dinámicos (una vez por análisis)
self.dynamic_hotspots = self.gfw_generator.generate_hotspots()

# Para cada spot
gfw_bonus = self.gfw_generator.calculate_proximity_bonus(
    lat=spot['lat'],
    lon=spot['lon'],
    hotspots=self.dynamic_hotspots,
    max_distance_km=10.0
)

spot['gfw_nearest_hotspot'] = nearest_hotspot
spot['gfw_distance_km'] = distance_km
spot['gfw_bonus'] = gfw_bonus
spot['score'] += gfw_bonus
```

---

## 7. Sistema de Scoring Unificado V7

### 7.1 Fórmula Completa

```
Score Final V7 = Score Base ML
               + Tide Bonus      (±15 pts)
               + Hour Bonus      (±12 pts)
               + SSS Bonus       (±10 pts)
               + SLA Bonus       (±8 pts)
               + Chla Bonus      (±8 pts)   ← NUEVO
               + SST Hist Bonus  (±6 pts)   ← NUEVO
               + GFW Bonus       (±10 pts)  ← NUEVO
               ─────────────────────────────
               = Rango: 0-100 (clamped)
```

### 7.2 Tabla de Bonuses

| Componente | Rango | Cálculo | Justificación |
|------------|-------|---------|---------------|
| Tide | ±15 | `(tide_score - 0.5) * 30` | Impacto directo en actividad de peces |
| Hour | ±12 | `(hour_score - 0.5) * 24` | Alba/ocaso óptimos |
| SSS | ±10 | `(sss_score - 0.5) * 20` | Salinidad afecta distribución |
| SLA | ±8 | `(sla_score - 0.5) * 16` | Upwelling indicator |
| **Chla** | ±8 | `(chla_score - 0.5) * 16` | Productividad primaria |
| **SST Hist** | ±6 | `(sst_score - 0.5) * 12` | Anomalías térmicas |
| **GFW** | ±10 | `proximity_bonus` | Validación real |

### 7.3 Implementación Integrada

```python
def analyze_spots_v7(self, target_hour: int = 6) -> List[dict]:
    """
    Análisis de spots con scoring V7 completo.

    Integra todas las fuentes de datos:
    - ML base score
    - Mareas astronómicas
    - Hora del día
    - Salinidad (SSS)
    - Nivel del mar (SLA)
    - Clorofila-a (NUEVO)
    - SST histórico (NUEVO)
    - Hotspots GFW (NUEVO)
    """
    results = []

    # Pre-generar hotspots dinámicos (una vez)
    dynamic_hotspots = self.gfw_generator.generate_hotspots()

    for spot in self.fishing_spots:
        # Score base ML
        base_score = self._calculate_base_score(spot)

        # === Bonuses existentes ===
        tide_score = self._get_tide_score(target_hour)
        tide_bonus = (tide_score - 0.5) * 30

        hour_score = self._get_hour_score(target_hour)
        hour_bonus = (hour_score - 0.5) * 24

        sss_value = self.physics_fetcher.get_sss_for_location(
            self.analysis_date, spot['lat'], spot['lon']
        )
        sss_score = CopernicusPhysicsFetcher.calculate_sss_score(sss_value)
        sss_bonus = (sss_score - 0.5) * 20

        sla_value = self.physics_fetcher.get_sla_for_location(
            self.analysis_date, spot['lat'], spot['lon']
        )
        sla_score = CopernicusPhysicsFetcher.calculate_sla_score(sla_value)
        sla_bonus = (sla_score - 0.5) * 16

        # === NUEVOS Bonuses V7 ===

        # Clorofila-a
        chla_value = self.chla_fetcher.get_value_for_location(
            self.analysis_date, spot['lat'], spot['lon']
        )
        chla_score = ChlorophyllFetcher.calculate_score(chla_value)
        chla_bonus = (chla_score - 0.5) * 16

        # SST Histórico
        sst_result = self.sst_provider.get_sst_with_anomaly(
            self.analysis_date, spot['lat'], spot['lon']
        )
        sst_hist_bonus = (sst_result.score - 0.5) * 12

        # GFW Hotspots
        gfw_bonus = self.gfw_generator.calculate_proximity_bonus(
            spot['lat'], spot['lon'],
            hotspots=dynamic_hotspots
        )

        # === Score Final ===
        total_bonus = (
            tide_bonus + hour_bonus +
            sss_bonus + sla_bonus +
            chla_bonus + sst_hist_bonus + gfw_bonus
        )

        final_score = np.clip(base_score + total_bonus, 0, 100)

        # Construir resultado
        result = {
            'lat': spot['lat'],
            'lon': spot['lon'],
            'score': round(final_score, 1),
            'base_score': round(base_score, 1),

            # Componentes de scoring
            'tide_score': round(tide_score, 2),
            'tide_bonus': round(tide_bonus, 1),
            'hour_score': round(hour_score, 2),
            'hour_bonus': round(hour_bonus, 1),
            'sss_value': sss_value,
            'sss_bonus': round(sss_bonus, 1),
            'sla_value': sla_value,
            'sla_bonus': round(sla_bonus, 1),

            # NUEVOS campos V7
            'chla_value': chla_value,
            'chla_score': round(chla_score, 2),
            'chla_bonus': round(chla_bonus, 1),
            'sst_historical': sst_result.sst,
            'sst_anomaly': round(sst_result.anomaly, 2),
            'sst_trend': round(sst_result.trend_7d, 2),
            'sst_hist_bonus': round(sst_hist_bonus, 1),
            'gfw_bonus': round(gfw_bonus, 1),

            # Metadata
            'hour': target_hour,
            'version': 'V7'
        }

        results.append(result)

    # Ordenar por score descendente
    results.sort(key=lambda x: x['score'], reverse=True)

    return results
```

---

## 8. Validación y Testing

### 8.1 Tests Unitarios

```python
# tests/test_chlorophyll_fetcher.py

import pytest
import numpy as np
from data.fetchers.copernicus_chlorophyll_fetcher import (
    ChlorophyllFetcher,
    CHLA_OPTIMAL_MIN,
    CHLA_OPTIMAL_MAX
)


class TestChlorophyllScore:
    """Tests para cálculo de score de Clorofila-a."""

    def test_optimal_range_returns_high_score(self):
        """Valores en rango óptimo deben dar score alto."""
        for chla in [2.0, 5.0, 8.0, 10.0]:
            score = ChlorophyllFetcher.calculate_score(chla)
            assert score >= 0.85, f"chla={chla} debería dar score >= 0.85"

    def test_oligotrophic_returns_low_score(self):
        """Valores muy bajos deben dar score bajo."""
        score = ChlorophyllFetcher.calculate_score(0.2)
        assert score <= 0.3

    def test_bloom_returns_moderate_score(self):
        """Bloom alto pero no extremo da score moderado."""
        score = ChlorophyllFetcher.calculate_score(15.0)
        assert 0.5 <= score <= 0.8

    def test_extreme_bloom_returns_low_score(self):
        """Bloom extremo (posible HAB) da score bajo."""
        score = ChlorophyllFetcher.calculate_score(30.0)
        assert score <= 0.4

    def test_none_returns_neutral(self):
        """Valor None debe retornar score neutral."""
        score = ChlorophyllFetcher.calculate_score(None)
        assert score == 0.5

    def test_score_bounds(self):
        """Score siempre entre 0 y 1."""
        for chla in [0.01, 0.5, 1.0, 5.0, 15.0, 50.0, 100.0]:
            score = ChlorophyllFetcher.calculate_score(chla)
            assert 0.0 <= score <= 1.0


class TestSSTHistorical:
    """Tests para SST histórico."""

    def test_optimal_temp_high_score(self):
        from data.fetchers.sst_historical_provider import (
            SSTHistoricalProvider,
            SST_OPTIMAL_CENTER
        )

        score = SSTHistoricalProvider.calculate_score(SST_OPTIMAL_CENTER, 0.0)
        assert score >= 0.9

    def test_cold_anomaly_bonus(self):
        """Anomalía fría (upwelling) debe dar bonus."""
        from data.fetchers.sst_historical_provider import SSTHistoricalProvider

        score_normal = SSTHistoricalProvider.calculate_score(18.0, 0.0)
        score_cold = SSTHistoricalProvider.calculate_score(18.0, -2.0)

        assert score_cold >= score_normal


class TestGFWHotspots:
    """Tests para hotspots dinámicos."""

    def test_generate_returns_list(self):
        from data.fetchers.gfw_hotspot_generator import GFWHotspotGenerator

        generator = GFWHotspotGenerator()
        # Esto puede fallar si no hay datos, lo cual es esperado
        try:
            hotspots = generator.generate_hotspots()
            assert isinstance(hotspots, list)
        except FileNotFoundError:
            pytest.skip("No hay datos GFW disponibles")

    def test_proximity_bonus_decreases_with_distance(self):
        """Bonus debe decrecer con distancia."""
        from data.fetchers.gfw_hotspot_generator import (
            GFWHotspotGenerator,
            DynamicHotspot
        )

        # Crear hotspot de prueba
        test_hotspot = DynamicHotspot(
            id=1,
            lat=-17.8,
            lon=-71.2,
            fishing_hours=100.0,
            vessel_count=10,
            score=0.9,
            source="test"
        )

        generator = GFWHotspotGenerator()

        # Punto cercano
        bonus_near = generator.calculate_proximity_bonus(
            -17.81, -71.21,
            hotspots=[test_hotspot],
            max_distance_km=10.0
        )

        # Punto lejano
        bonus_far = generator.calculate_proximity_bonus(
            -17.9, -71.3,
            hotspots=[test_hotspot],
            max_distance_km=10.0
        )

        assert bonus_near >= bonus_far
```

### 8.2 Tests de Integración

```python
# tests/test_scoring_v7.py

import pytest
from controllers.analysis import AnalysisController


class TestScoringV7Integration:
    """Tests de integración para scoring V7."""

    @pytest.fixture
    def controller(self):
        """Crea controlador con datos de prueba."""
        c = AnalysisController()
        c.analysis_datetime = datetime(2026, 2, 7, 16, 0)
        c.load_coastline('data/gold/coastline/coastline_v8_extended.geojson')
        c.sample_fishing_spots(spacing_m=1000, max_spots=50)  # Pocos para test
        return c

    def test_all_components_contribute(self, controller):
        """Todos los componentes deben contribuir al score."""
        controller.fetch_marine_data()
        controller.get_conditions()

        results = controller.analyze_spots_v7(target_hour=6)

        assert len(results) > 0

        for r in results[:5]:  # Verificar top 5
            # Verificar que todos los campos V7 existen
            assert 'chla_bonus' in r
            assert 'sst_hist_bonus' in r
            assert 'gfw_bonus' in r

            # Verificar que los bonuses están en rango esperado
            assert -8 <= r['chla_bonus'] <= 8
            assert -6 <= r['sst_hist_bonus'] <= 6
            assert 0 <= r['gfw_bonus'] <= 10

    def test_score_changes_with_hour(self, controller):
        """Score debe cambiar significativamente con hora."""
        controller.fetch_marine_data()
        controller.get_conditions()

        results_6am = controller.analyze_spots_v7(target_hour=6)
        results_noon = controller.analyze_spots_v7(target_hour=12)

        # Comparar primer spot
        score_diff = abs(results_6am[0]['score'] - results_noon[0]['score'])

        # Debe haber al menos 5 puntos de diferencia
        assert score_diff >= 5, "Score debe cambiar significativamente con hora"
```

---

## 9. Métricas de Rendimiento

### 9.1 Benchmarks Objetivo

| Operación | Tiempo Máximo | Memoria Máxima |
|-----------|---------------|----------------|
| Cargar Chl-a (1 mes) | 500ms | 50MB |
| Cargar SST histórico (todo) | 5s | 500MB |
| Generar hotspots DBSCAN | 2s | 200MB |
| Query individual (con caché) | 1ms | N/A |
| Análisis completo 600 spots | 30s | 1GB |

### 9.2 Monitoreo

```python
# utils/performance.py

import time
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


def timed(func: Callable) -> Callable:
    """Decorator para medir tiempo de ejecución."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        logger.debug(f"{func.__name__}: {elapsed:.3f}s")

        return result
    return wrapper


def memory_usage() -> float:
    """Retorna uso de memoria actual en MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024
```

---

## Apéndice A: Checklist de Implementación

```markdown
## Pre-Implementación
- [ ] Verificar credenciales Copernicus en .env
- [ ] Verificar datos GFW descargados
- [ ] Verificar SST histórico descargado
- [ ] Ejecutar tests existentes: `pytest tests/ -v`

## Implementación Clorofila-a
- [ ] Crear copernicus_chlorophyll_fetcher.py
- [ ] Agregar RAW_CHLA_COPERNICUS a data_config.py
- [ ] Descargar datos: 2024-01 a 2026-02
- [ ] Tests unitarios pasando
- [ ] Integrar en AnalysisController

## Implementación SST Histórico
- [ ] Crear sst_historical_provider.py
- [ ] Generar caché de estadísticas mensuales
- [ ] Tests unitarios pasando
- [ ] Integrar en AnalysisController

## Implementación GFW Hotspots
- [ ] Crear gfw_hotspot_generator.py
- [ ] Generar hotspots iniciales
- [ ] Exportar a GeoJSON para verificación visual
- [ ] Tests unitarios pasando
- [ ] Integrar en AnalysisController

## Post-Implementación
- [ ] Actualizar analyze_spots() a V7
- [ ] Actualizar views/map_view.py con nuevos campos
- [ ] Tests de integración pasando
- [ ] Documentar en PLAN_V7
- [ ] Benchmark de rendimiento
```

---

## Apéndice B: Comando de Ejecución

```bash
# Descargar Clorofila-a
python -c "
from data.fetchers.copernicus_chlorophyll_fetcher import ChlorophyllFetcher
f = ChlorophyllFetcher()
f.download_range(2024, 1, 2026, 2)
"

# Generar caché SST
python -c "
from data.fetchers.sst_historical_provider import SSTHistoricalProvider
p = SSTHistoricalProvider()
p._ensure_loaded()  # Genera caché
"

# Generar hotspots GFW
python -c "
from data.fetchers.gfw_hotspot_generator import GFWHotspotGenerator
g = GFWHotspotGenerator()
hotspots = g.generate_hotspots()
print(f'Generados {len(hotspots)} hotspots')
g.export_to_geojson(hotspots, Path('output/dynamic_hotspots.geojson'))
"

# Ejecutar análisis completo V7
python main.py --scoring-version v7
```

---

*ANEXO V7 creado: 2026-02-08*
*Proyecto: Fishing Predictor - Tacna/Ilo/Sama, Perú*
