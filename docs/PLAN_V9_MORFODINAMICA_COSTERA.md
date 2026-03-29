# Plan V9: Morfodinámica Costera con Sentinel-2

**Objetivo:** Detectar cambios dinámicos del sustrato costero (arena tapando/destapando rocas y pozas) usando imágenes satelitales Sentinel-2 para reemplazar la heurística estática de sustrato y mejorar la predicción de hábitat accesible para pesca.

**Estado:** Pendiente
**Fecha:** 2026-03-28
**Confianza:** 0.93 (validado con deliberación multi-modelo + investigación profunda)

---

## Problema

El sistema actual asigna sustrato con una heurística basada en profundidad GEBCO:
```python
# controllers/analysis.py _assign_substrate_vectorized()
depth > 0    → rock
-5m to 0m    → mixed
depth < -5m  → sand
```

**Esto es incorrecto** porque:
- La arena se desplaza estacionalmente tapando pozas rocosas donde se concentran peces
- El oleaje y mareas exponen o entierran estructuras de hábitat
- Un spot con score alto pero sustrato enterrado NO tendrá peces
- `rocky_coast_fraction: 0.4` en `real_data_pipeline.py` es un valor arbitrario sin base real
- `SPECIES_BY_SUBSTRATE` en `analysis.py` es redundante con `SPECIES_DATABASE` en `species_zones.py`

---

## Solución Validada

### Stack Tecnológico

| Componente | Tecnología | Justificación |
|---|---|---|
| Acceso datos | CDSE STAC + pystac-client + stackstac | Encaja con patrón CopernicusDataProvider existente |
| Clasificación | Decision tree con 3 índices espectrales | 74.5-93.6% accuracy (papers), simple, debuggeable |
| Resolución | 20m (clasificación), 10m (waterline) | B8A/B11 son nativos 20m, NUNCA upsamplear SWIR |
| Normalización mareal | Composite Waterline Method (CWM) | Más robusto que modelo mareal, ~1m rango micro |

### Alternativas Descartadas

| Opción | Razón de Rechazo | Fuente |
|---|---|---|
| CoastSat | Solo playas arenosas, no clasifica sustrato, requiere GEE, misidentifica rocas | [GitHub README](https://github.com/kvos/CoastSat) |
| openEO | Abstracción innecesaria, STAC encaja mejor con patrón existente | Deliberación ai-counsel (0.87 confianza) |
| Random Forest (v1) | Overkill para 3 clases, decision tree suficiente, más debuggeable | [Lithological mapping paper](https://www.researchgate.net/publication/324700828) |
| Sentinel-1 SAR | Complejidad de preprocesamiento (speckle, terrain correction), postergar a v2 | Deliberación ai-counsel |

### Índices Espectrales Seleccionados

| Índice | Fórmula | Resolución | Función |
|---|---|---|---|
| MNDWI | (B03 - B11) / (B03 + B11) | 20m | Separar agua de tierra |
| B11/B12 ratio | B11 / B12 | 20m | Discriminar roca vs arena (mineralogía SWIR) |
| BSI | ((B11 + B04) - (B8A + B02)) / ((B11 + B04) + (B8A + B02)) | 20m | Suelo desnudo vs otras superficies |

**Nota:** Usar B8A (865nm, 20m nativo) en vez de B08 (842nm, 10m) para todo trabajo a 20m. B8A tiene bandwidth más estrecho que evita absorción de vapor de agua.

### Bandas Sentinel-2 Requeridas

| Banda | Nombre | Resolución | Uso |
|---|---|---|---|
| B02 | Azul (490nm) | 10m | BSI, brillo |
| B03 | Verde (560nm) | 10m | MNDWI numerador, NDWI |
| B04 | Rojo (665nm) | 10m | BSI |
| B08 | NIR (842nm) | 10m | NDWI waterline (solo 10m) |
| B8A | NIR narrow (865nm) | 20m | BSI denominador |
| B11 | SWIR-1 (1610nm) | 20m | MNDWI, B11/B12 ratio |
| B12 | SWIR-2 (2190nm) | 20m | B11/B12 ratio mineralogía |
| SCL | Scene Classification | 20m | Máscara de nubes |

---

## Ground Truth Disponible

### Datos Propios (29 puntos)

| Fuente | Puntos | Tipo | Confianza |
|---|---|---|---|
| Hotspots domain.py | 20 | lat/lon + substrate (ROCK/SAND/MIXED) | Alta |
| Encuestas pescadores | 9 | GPS + species + substrate + catch | Alta |
| **Total** | **29** | Validación cruzada | |

**Distribución de sustrato en hotspots:**
- ROCK: 11 (Punta Coles, Pozo Lizas, Pocoma, Fundicion, Punta Blanca, Gentillar, Ite Sur, Llostay, Punta Mesa, Vila Vila, Pintadilla)
- SAND: 5 (Playa Media Luna, Playa Blanca-Gentillar, Ite Norte, Santa Rosa, Boca del Rio)
- MIXED: 4 (Ilo Puerto, Pozo Redondo, Ite Centro, Carlepe, Los Palos)

### Datos IMARPE (validación adicional)

| Estudio | Zona | Contenido | Referencia |
|---|---|---|---|
| Línea Base Vila Vila | Vila Vila - Ilo | Sustrato rocoso meso/infralitoral, paredes verticales, grietas, hasta 40m | [IMARPE/PRODUCE](https://rnia.produce.gob.pe/wp-content/uploads/2019/09/lbase-vilavila.pdf) |
| Sedimentos superficiales Tacna | Litoral Tacna | Granulometría, distribución sedimentos | [Repositorio IMARPE](https://repositorio.imarpe.gob.pe/handle/20.500.12958/2163) |
| Playas arenosas Moquegua-Tacna | Moquegua-Tacna | Playas intermedias/disipativas, 16 spp, Mesodesma donacium | [Repositorio IMARPE](https://repositorio.imarpe.gob.pe/handle/20.500.12958/3231) |
| Monitoreo bentónico Punta Coles | Punta Coles | Buceo hookah, 4.5km, 15m profundidad, semestral | [IMARPE Ilo](https://pescaymedioambiente.com/moquegua-imarpe-monitorea-comunidad-bentonica-en-la-reserva-de-punta-coles-ilo/) |

---

## Precisión Esperada (Literatura)

| Métrica | Valor | Fuente |
|---|---|---|
| Shoreline RMSE (Sentinel-2) | 5.0 m | [Coastal change Spain 2024](https://www.sciencedirect.com/science/article/pii/S0378383924000656) |
| Shoreline horizontal accuracy (micromareal) | ~10 m | [Nature Comms benchmark 2023](https://www.nature.com/articles/s43247-023-01001-2) |
| Clasificación intermareal supervisada | 92.5-93.6% | [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S1569843225003231) |
| Discriminación litológica SWIR | 74.5% | [Sentinel-2A lithology](https://www.researchgate.net/publication/324700828) |
| Pérdida detectable (cambio temporal) | 12% marsh loss 2017-2023 | [Intertidal monitoring 2025](https://www.sciencedirect.com/science/article/pii/S1569843225003231) |

**Costa Tacna-Ilo:** semiárida (Atacama), ~10-20% nubosidad, rango mareal ~1m (micromareal). Condiciones ideales para teledetección óptica.

---

## Fases de Implementación

### Fase 0: Quick Wins (pre-Sentinel-2)

Correcciones inmediatas al código existente que no requieren datos satelitales.

- [ ] **0.1** Eliminar `SPECIES_BY_SUBSTRATE` hardcodeado en `controllers/analysis.py` — usar `SPECIES_DATABASE` de `species_zones.py` como única fuente
- [ ] **0.2** Eliminar `rocky_coast_fraction: 0.4` de `real_data_pipeline.py`
- [ ] **0.3** Revisar depth default `-10.0` en `analysis.py` — usar promedio local más representativo

---

### Fase 1: Adquisición de Datos Sentinel-2

#### API: CDSE STAC

```python
# Endpoint (nuevo, el legacy se deprecó nov 2025)
BASE_URL = "https://stac.dataspace.copernicus.eu/v1/"

# Autenticación: S3 keys (NO OAuth2)
# Generar en: https://dataspace.copernicus.eu/ → User Settings → S3 Access
AWS_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
AWS_ACCESS_KEY_ID = "<from .env>"
AWS_SECRET_ACCESS_KEY = "<from .env>"
```

#### Tareas

- [ ] **1.1** Registrar cuenta en [dataspace.copernicus.eu](https://dataspace.copernicus.eu) y generar S3 keys
  - **IMPORTANTE:** Las credenciales de marine.copernicus.eu NO funcionan
  - Agregar `CDSE_ACCESS_KEY` y `CDSE_SECRET_KEY` a `.env`

- [ ] **1.2** Instalar dependencias
  ```
  pystac-client>=0.7.0
  stackstac>=0.5.0
  rasterio>=1.3.0
  rioxarray>=0.15.0
  ```

- [ ] **1.3** Agregar paths a `data/data_config.py`
  ```python
  RAW_SENTINEL2 = RAW_DIR / "sentinel2"
  RAW_SENTINEL2_SCENES = RAW_SENTINEL2 / "scenes"
  ```

- [ ] **1.4** Crear `scripts/download_sentinel2.py`
  - Patrón: similar a `scripts/download_all_copernicus.py`
  - Colección: `sentinel-2-l2a`
  - Bandas: B02, B03, B04, B08, B8A, B11, B12, SCL
  - Región: franja costera STUDY_AREA (buffer 2km tierra + 500m mar)
  - Filtro: `eo:cloud_cover < 30`
  - Formato: GeoTIFF multibanda por escena
  - Deduplicar por fecha de adquisición

  ```python
  from pystac_client import Client
  import stackstac

  client = Client.open("https://stac.dataspace.copernicus.eu/v1/")
  search = client.search(
      collections=["sentinel-2-l2a"],
      bbox=[west, south, east, north],
      datetime="2023-01-01/2023-12-31",
      query={"eo:cloud_cover": {"lt": 30}},
  )
  datacube = stackstac.stack(
      items, assets=["B02","B03","B04","B08","B8A","B11","B12","SCL"],
      resolution=20, resampling=Resampling.bilinear
  )
  ```

- [ ] **1.5** Descargar en dos etapas:
  - **Etapa A (PoC):** Año 2023 (~72 escenas, ~2.4 GB) — validar contra 29 puntos ground truth
  - **Etapa B (completo):** 10 años 2015-2025 (~650 escenas, ~21 GB) — tras validar PoC

  Los 10 años cubren:
  - El Niño 2015-16 (fuerte) y 2023-24 (moderado)
  - La Niña 2020-2022
  - Múltiples ciclos estacionales (verano/invierno)
  - 2015-2017: solo Sentinel-2A (~10 días revisita, ~36 escenas/año)
  - 2017-2025: Sentinel-2A + 2B (~5 días revisita, ~72 escenas/año)

#### Estructura Bronze

```
data/raw/sentinel2/
    scenes/
        2015-06/          # Primer mes disponible Sentinel-2
            S2_20150623_bands.tif
        ...
        2023-01/
            S2_20230105_bands.tif
            S2_20230110_bands.tif
        ...
        2025-12/
            S2_20251225_bands.tif
    _manifest.json
```

#### Estimación almacenamiento

| Período | Escenas | Tamaño | Cobertura ENSO |
|---|---|---|---|
| 1 año PoC (2023) | ~72 | ~2.4 GB | El Niño moderado |
| 3 años (2022-2024) | ~216 | ~7 GB | La Niña + El Niño |
| **10 años (2015-2025)** | **~650** | **~21 GB** | **2 El Niño + 1 La Niña + neutral** |

---

### Fase 2: Clasificación de Sustrato (20m)

#### Algoritmo: Decision Tree con 3 Índices

```python
def classify_substrate(b02, b03, b04, b8a, b11, b12, scl):
    """Clasifica cada pixel a 20m en agua/roca/arena/mixto."""
    # Paso 0: Máscara de nubes (SCL)
    valid = np.isin(scl, [4, 5, 6, 7])  # vegetación, suelo, agua, no-clasificado

    # Paso 1: Separar agua de tierra
    mndwi = (b03 - b11) / (b03 + b11 + 1e-10)
    is_water = mndwi > 0.0

    # Paso 2: Discriminar roca vs arena (solo pixels terrestres)
    swir_ratio = b11 / (b12 + 1e-10)
    # Roca ígnea (Peru coast): bajo B11 relativo, ratio < 1.5
    # Arena: alto B11, ratio > 1.5

    # Paso 3: Bare Soil Index
    bsi = ((b11 + b04) - (b8a + b02)) / ((b11 + b04) + (b8a + b02) + 1e-10)

    # Paso 4: Brillo general
    brightness = (b02 + b03 + b04) / 3

    # Clasificación
    result = np.full_like(mndwi, 0, dtype=np.uint8)  # 0=nodata
    result[valid & is_water] = 1                       # agua
    result[valid & ~is_water & (brightness < 800) & (swir_ratio < 1.5)] = 2  # roca
    result[valid & ~is_water & (brightness > 1400) & (bsi > 0.1)] = 3       # arena
    result[valid & ~is_water & (result == 0)] = 4      # mixto (no roca ni arena claros)

    return result  # uint8: 0=nodata, 1=agua, 2=roca, 3=arena, 4=mixto
```

**Nota:** Los umbrales (800, 1400, 1.5, 0.1) se calibrarán con los 29 puntos de ground truth.

#### Tareas

- [ ] **2.1** Crear `core/sentinel2/__init__.py`
- [ ] **2.2** Crear `core/sentinel2/spectral_classifier.py`
  - Implementar decision tree con MNDWI + B11/B12 + BSI
  - Máscara de nubes con SCL
  - Calibrar umbrales con 29 puntos ground truth
- [ ] **2.3** Clasificar todas las escenas del año PoC (2023)
  - Output: GeoTIFF categórico uint8 por escena
  - Almacenar en `data/processed/sentinel2/substrate_maps/`
- [ ] **2.4** Validar: calcular confusion matrix contra 29 puntos
  - Objetivo: >75% overall accuracy (basado en literatura)
  - Si <75%: ajustar umbrales o considerar RF como upgrade
- [ ] **2.5** Tras validar PoC: clasificar las ~650 escenas de 10 años (2015-2025)
  - Procesamiento batch con progreso incremental
  - ~700,000 pixels costeros × 650 escenas = película completa de la costa

---

### Fase 3: Waterline a 10m + Normalización Mareal

#### Método: Composite Waterline Method (CWM)

Para cada mes:
1. Seleccionar 2-3 escenas más cercanas a Mean Low Water (MLW) usando `tide_fetcher.py`
2. Calcular NDWI a 10m: `(B03 - B08) / (B03 + B08)`
3. Extraer contorno waterline con umbral Otsu o fijo (0.0)
4. Tomar mediana de posiciones de waterline → suprime ruido mareal

#### Tareas

- [ ] **3.1** Crear `core/sentinel2/waterline_extractor.py`
  - NDWI a 10m (B03, B08 — ambos nativos 10m)
  - Umbral adaptativo (Otsu) por escena
  - Extracción de contorno con `skimage.measure.find_contours`
  - Conversión pixel→geográfico con `rasterio.transform`
- [ ] **3.2** Integrar con `tide_fetcher.py` existente
  - Para cada escena: obtener nivel mareal al momento de captura (~10:30 local)
  - Seleccionar escenas cercanas a MLW para composite
- [ ] **3.3** Generar waterlines mensuales para 10 años (2015-2025)
  - Almacenar en `data/processed/sentinel2/waterlines.parquet`
  - Schema: `date, lat, lon, ndwi_value, waterline_position_m, tide_height_m`
  - ~120 composites mensuales (10 años × 12 meses)

---

### Fase 4: Análisis Temporal (Mapas de Estabilidad)

#### Métricas por Pixel (10 años)

```python
@dataclass
class StabilityMetrics:
    lat: float
    lon: float
    dominant_substrate: str           # sustrato más frecuente en 10 años
    stability_score: float            # 0-1 (1 = siempre mismo sustrato)
    rock_frequency: float             # fracción del tiempo como roca
    sand_frequency: float             # fracción del tiempo como arena
    water_frequency: float            # fracción del tiempo sumergido
    is_seasonal_burial: bool          # True si roca en verano → arena en invierno
    mean_waterline_distance_m: float  # distancia media de línea de agua
    waterline_std_m: float            # variabilidad
    waterline_trend_m_yr: float       # tendencia acreción/erosión (m/año)
    elnino_substrate: str             # sustrato dominante durante El Niño
    lanina_substrate: str             # sustrato dominante durante La Niña
    n_observations: int               # escenas válidas (sin nubes)
    first_observation: str            # fecha primera escena
    last_observation: str             # fecha última escena
```

#### Tareas Fase 4a: Estabilidad Temporal

- [ ] **4.1** Crear `core/sentinel2/temporal_analysis.py`
- [ ] **4.2** Para cada pixel costero a 20m, calcular sobre 10 años (~650 escenas):
  - Moda de sustrato (dominant_substrate)
  - Frecuencia de cada clase (rock_freq, sand_freq, water_freq)
  - Estabilidad: `n_dominant / n_observations`
  - Detección de entierro estacional: roca en verano (DJF) vs arena en invierno (JJA)
  - Comportamiento ENSO: sustrato durante El Niño (2015-16, 2023-24) vs La Niña (2020-22)
  - Tendencia de waterline: regresión lineal sobre 10 años → m/año
- [ ] **4.3** Generar composites trimestrales (DJF, MAM, JJA, SON) para cada año
  - Moda de sustrato por pixel
  - Almacenar como GeoTIFF categórico
  - 10 años × 4 trimestres = 40 composites

#### Tareas Fase 4b: Análisis de Desplazamiento Vectorial de Arena

Comparación escena-a-escena para trackear cómo se mueve la arena a lo largo de la costa.

- [ ] **4.4** Crear `core/sentinel2/displacement_tracker.py`

  Para cada par de escenas consecutivas (t, t+1):
  ```python
  def compute_displacement(substrate_t0, substrate_t1, transform):
      """Detecta desplazamiento de arena entre dos clasificaciones."""
      # Pixels que cambiaron de roca → arena (enterramiento)
      burial = (substrate_t0 == ROCK) & (substrate_t1 == SAND)
      # Pixels que cambiaron de arena → roca (exposición)
      exposure = (substrate_t0 == SAND) & (substrate_t1 == ROCK)

      # Centroide de cada zona de cambio → vector de desplazamiento
      burial_centroids = get_centroids(burial, transform)
      exposure_centroids = get_centroids(exposure, transform)

      # Dirección predominante del avance de arena
      # (vectores de burial_centroids hacia costa)
      sand_advance_direction = compute_mean_direction(burial_centroids)
      sand_advance_area_m2 = np.sum(burial) * 20 * 20  # pixels × resolución²

      return DisplacementResult(
          date_from=t0, date_to=t1,
          burial_area_m2=sand_advance_area_m2,
          exposure_area_m2=np.sum(exposure) * 400,
          net_change_m2=sand_advance_area_m2 - np.sum(exposure) * 400,
          sand_direction_deg=sand_advance_direction,
          burial_zones=burial_centroids,
          exposure_zones=exposure_centroids
      )
  ```

- [ ] **4.5** Generar serie temporal de desplazamiento para 10 años
  - ~650 escenas → ~649 pares consecutivos (cada 5 días)
  - Para cada par: calcular área enterrada, área expuesta, dirección
  - Resultado: "película" de la dinámica de arena con ~649 fotogramas

- [ ] **4.6** Agregar métricas de desplazamiento al StabilityMetrics
  ```python
  # Métricas adicionales por pixel
  burial_events_count: int            # veces que fue enterrado en 10 años
  exposure_events_count: int          # veces que fue expuesto
  mean_burial_duration_days: float    # duración media de enterramiento
  max_burial_duration_days: float     # enterramiento más largo
  last_transition_date: str           # última vez que cambió sustrato
  days_since_last_change: int         # estabilidad reciente
  sand_advance_direction_deg: float   # dirección predominante del avance
  sand_advance_rate_m2_yr: float      # velocidad de avance anual
  ```

- [ ] **4.7** Generar mapa de "corredores de arena"
  - Zonas donde la arena se mueve consistentemente en una dirección
  - Identifica puntas rocosas que actúan como barreras
  - Identifica bahías donde la arena rota estacionalmente (beach rotation)

- [ ] **4.8** Tracking de waterline por transecto
  - Definir transectos perpendiculares a la costa cada 100m (~2,810 transectos)
  - Calcular posición media, std, tendencia por transecto sobre 10 años

- [ ] **4.9** Almacenar en Gold layer
  ```
  data/analytics/sentinel2/
      stability_map.parquet              # métricas por pixel (10 años)
      displacement_timeseries.parquet    # desplazamiento entre escenas consecutivas
      sand_corridors.geojson             # corredores de movimiento de arena
      waterline_timeseries.parquet       # posición waterline por transecto/mes
      seasonal_composites/               # 40 composites (10 años × 4 trimestres)
          DJF_2015_substrate.tif
          MAM_2015_substrate.tif
          ...
          SON_2025_substrate.tif
  ```

---

### Fase 5: Integración con el Predictor

#### Score de Accesibilidad de Hábitat

```python
def compute_habitat_accessibility(lat, lon, stability_map, current_substrate):
    """Score 0-1: qué tan accesible es el hábitat rocoso en este punto."""

    # Base score por sustrato actual
    base = {'rock': 1.0, 'mixed': 0.6, 'sand': 0.2, 'water': 0.3}[current_substrate]

    # Bonus por estabilidad (roca estable > roca recién expuesta)
    nearest = spatial_lookup(lat, lon, stability_map)
    stability_bonus = nearest.stability_score * 0.2

    # Penalización por riesgo de entierro
    burial_penalty = 0.15 if nearest.is_seasonal_burial else 0.0

    return min(1.0, base + stability_bonus - burial_penalty)
```

#### 6 Features Nuevos (32 → 38)

| Index | Feature | Descripción | Rango | Fuente |
|---|---|---|---|---|
| 32 | `habitat_accessibility` | Score de sustrato actual | 0-1 | Clasificación más reciente |
| 33 | `substrate_stability` | Estabilidad temporal (10 años) | 0-1 | Fase 4a |
| 34 | `waterline_anomaly` | Posición waterline vs media | -3 a +3 std | Fase 3 |
| 35 | `seasonal_burial_risk` | Probabilidad de entierro estacional | 0-1 | Fase 4a |
| 36 | `days_since_substrate_change` | Días desde último cambio de sustrato | 0-3650 (normalizado) | Fase 4b |
| 37 | `sand_advance_rate` | Velocidad de avance de arena (m²/año) | 0-1 (normalizado) | Fase 4b |

#### Tareas

- [ ] **5.1** Crear `core/sentinel2/habitat_accessibility.py`
  - Carga `stability_map.parquet` al iniciar
  - Lookup espacial con `scipy.spatial.cKDTree`
  - Retorna 4 valores por spot
- [ ] **5.2** Actualizar `domain.py`
  - Agregar 6 features a `FEATURE_NAMES` (32 → 38)
  - Actualizar `N_FEATURES = 38`
  - Agregar `habitat_accessibility: float = 0.10` a `ScoringWeights`
- [ ] **5.3** Actualizar `models/features.py`
  - Extender `_to_vector()` para indices 32-37
  - Agregar `_extract_habitat_features()` con lookup a stability_map
- [ ] **5.4** Actualizar `controllers/analysis.py`
  - Importar `HabitatAccessibilityProvider` (try/except, opcional)
  - Integrar en `analyze_spots()` como bonus ambiental
  - **Fase inicial:** peso bajo (0.05), subir gradualmente tras validación
- [ ] **5.5** Refactorizar `_assign_substrate_vectorized()`
  - Reemplazar heurística depth-based con lookup a stability_map
  - Mantener encuestas como override (radio 500m)
- [ ] **5.6** Re-entrenar modelo ML con 36 features
  - PCA componentes se ajustan automáticamente
  - GradientBoosting maneja features nuevos bien
  - Validar que accuracy no degrada vs 32 features

#### Rebalanceo de Pesos Propuesto

```python
# Actual (V8)
ScoringWeights:
    front_proximity: 0.25
    chlorophyll: 0.20
    upwelling: 0.15
    fishing_activity: 0.15
    golden_hour: 0.10
    safety: 0.10
    lunar: 0.05

# Propuesto (V9) — redistribuir de upwelling y fishing_activity
ScoringWeights:
    front_proximity: 0.22
    chlorophyll: 0.18
    upwelling: 0.13
    fishing_activity: 0.13
    habitat_accessibility: 0.10   # NUEVO
    golden_hour: 0.10
    safety: 0.09
    lunar: 0.05
```

---

## Estructura de Archivos Nuevos

```
core/sentinel2/
    __init__.py
    spectral_classifier.py         # Fase 2: MNDWI + B11/B12 + BSI
    waterline_extractor.py         # Fase 3: NDWI 10m + CWM
    temporal_analysis.py           # Fase 4a: estabilidad + composites
    displacement_tracker.py        # Fase 4b: desplazamiento vectorial de arena
    habitat_accessibility.py       # Fase 5: score para ML pipeline

scripts/
    download_sentinel2.py          # Fase 1: CDSE STAC → GeoTIFF

data/raw/sentinel2/                # Bronze (inmutable) ~21 GB
    scenes/
        2015-06/*.tif              # Primer Sentinel-2
        ...
        2025-12/*.tif
    _manifest.json

data/processed/sentinel2/          # Silver (regenerable)
    substrate_maps/*.tif           # 1 clasificación por escena (~650)
    waterlines.parquet             # waterlines mensuales (120 meses)

data/analytics/sentinel2/          # Gold (ML-ready)
    stability_map.parquet          # métricas 10 años por pixel
    displacement_timeseries.parquet # desplazamiento entre escenas (~649 pares)
    sand_corridors.geojson         # corredores de movimiento de arena
    waterline_timeseries.parquet   # posición por transecto/mes
    seasonal_composites/           # 40 composites trimestrales
```

---

## Dependencias Nuevas

```
pystac-client>=0.7.0      # CDSE STAC client
stackstac>=0.5.0           # STAC → xarray datacube
rasterio>=1.3.0            # lectura/escritura GeoTIFF
rioxarray>=0.15.0          # xarray + rasterio integration
scikit-image>=0.21.0       # find_contours para waterline
```

**Ya disponibles:** numpy, scipy, pandas, xarray (en conda env fishing_predictor)

---

## Riesgos y Mitigaciones

| Riesgo | Severidad | Mitigación |
|---|---|---|
| Credenciales CDSE diferentes a Marine | Media | Documentar en .env.example, registro gratuito |
| Nubes en invierno (jun-ago) | Baja | Costa semiárida (~10-20%), composites mensuales con mediana |
| 20m no detecta pozas <20m | Baja | Aceptable: predictor opera a ~100m spacing entre spots |
| Feature expansion 32→36 | Media | Peso inicial bajo (0.05), subir tras validación cruzada |
| Species affinity cascade | Media | `SPECIES_DATABASE` ya soporta sustrato dinámico, no requiere cambios |
| Umbrales de clasificación regionales | Media | Calibrar con 29 ground truth + IMARPE estudios |
| CoastSat no viable | N/A | Implementación propia con MNDWI/NDWI, más flexible |

---

## Orden de Implementación

```
Fase 0 (Quick Wins)      ~30 min     Sin dependencias externas
         ↓
Fase 1 (Datos CDSE)      ~3 días     Registro + download PoC 2023 + expandir 10 años
         ↓
Fase 2 (Clasificación)   ~2 días     Decision tree + validación + batch 650 escenas
         ↓
Fase 3 (Waterline)       ~1 día      NDWI 10m + CWM 120 meses
         ↓
Fase 4a (Estabilidad)    ~1 día      Stability maps 10 años + composites
         ↓
Fase 4b (Desplazamiento) ~2 días     Tracking vectorial + corredores + ENSO
         ↓
Fase 5 (Integración)     ~2 días     Features 32→38 + retraining
```

**Total estimado: ~11 días de desarrollo**

Cada fase es funcional e independiente — se valida antes de pasar a la siguiente.

---

## Criterios de Éxito

| Criterio | Umbral | Medición |
|---|---|---|
| Accuracy clasificación sustrato | ≥75% | Confusion matrix vs 29 puntos |
| Correlación sustrato↔especies capturadas | Significativa (p<0.05) | Chi-squared test encuestas |
| Score ML no degrada | ≤2% pérdida vs baseline 32 features | Cross-validation |
| Detección de entierro estacional | ≥3 zonas identificadas | Inspección visual composites |
| Corredores de arena identificados | ≥2 corredores principales | sand_corridors.geojson |
| Diferencia El Niño vs La Niña detectable | Cambio significativo en rock_freq | t-test entre períodos |
| Cobertura temporal | ≥600 escenas procesadas | _manifest.json |
| Tiempo de predicción | <5s adicionales | Benchmark hot path |

---

## Referencias Científicas

1. Vos, K. et al. (2019). CoastSat: A Google Earth Engine-enabled Python toolkit. *Environmental Modelling & Software*. [DOI](https://www.sciencedirect.com/science/article/pii/S1364815219300490) — **Rechazado** para este proyecto (solo sandy beaches)
2. Sánchez-García, E. et al. (2024). Satellite coastal change detection. *Coastal Engineering*. [DOI](https://www.sciencedirect.com/science/article/pii/S0378383924000656)
3. Vos, K. et al. (2023). Benchmarking satellite-derived shoreline algorithms. *Nature Communications Earth & Environment*. [DOI](https://www.nature.com/articles/s43247-023-01001-2)
4. van der Werff, H. & van der Meer, F. (2018). Sentinel-2A lithological classification. *Remote Sensing*. [DOI](https://www.researchgate.net/publication/324700828)
5. Intertidal monitoring with Sentinel-2 (2025). *Int. J. Applied Earth Observation*. [DOI](https://www.sciencedirect.com/science/article/pii/S1569843225003231)
6. Briceño-Zuluaga, F. et al. (2022). Climate vulnerability Peru fisheries. *Scientific Reports*. [DOI](https://www.nature.com/articles/s41598-022-08818-5)
7. Velazco, F. et al. (2004). Sedimentos marinos Tacna. *IMARPE*. [Repositorio](https://repositorio.imarpe.gob.pe/handle/20.500.12958/2163)
8. IMARPE (2015). Caracterización playas arenosas Moquegua-Tacna. [Repositorio](https://repositorio.imarpe.gob.pe/handle/20.500.12958/3231)
9. IMARPE (2019). Línea Base Vila Vila. [PRODUCE](https://rnia.produce.gob.pe/wp-content/uploads/2019/09/lbase-vilavila.pdf)
10. CDSE STAC API Documentation. [Copernicus](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)

---

*Plan creado: 2026-03-28*
*Investigación: ai-counsel (Claude Opus deliberación), PAL (Gemini análisis arquitectura), Context7 (openEO/pystac docs), WebSearch (papers + IMARPE)*
