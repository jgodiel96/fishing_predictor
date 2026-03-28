# Plan V8: Mapa de Alta Resolución con Sustrato, Batimetría y Scores Geográficos

**Estado:** ✅ IMPLEMENTADO (superó objetivos originales)

## Contexto
El mapa original tenía 606 spots a ~469m de espaciado, sin info de sustrato ni profundidad.

## Resultado Implementado (2026-03-26)
1. ~~Resolución de 25m (~11,200 spots)~~ → **Resolución de 5m con 4 bandas offshore (224,800 spots)**
2. ✅ Tipo de sustrato per-spot (roca, arena, mixto) — basado en GEBCO + encuestas
3. ✅ Profundidad per-spot (batimetría GEBCO via RegularGridInterpolator)
4. ✅ Scores pre-calculados para 24 horas (top-200 por hora)
5. ✅ Visualización migrada a deck.gl (GPU WebGL) para soportar 224k spots

> **Nota:** El plan original especificaba 11,200 spots con 25m spacing. La implementación final usa 5m spacing con 4 bandas offshore (0m, 50m, 150m, 300m), generando ~224,800 spots — 20x más que lo planeado. deck.gl maneja esto sin problemas gracias al rendering GPU.

## Principio de Diseño: Todo Vectorizado con numpy/scipy

**CERO loops de Python sobre 224,800 spots.** Toda operación pesada usa:
- `numpy` broadcasting y operaciones vectorizadas
- `scipy.interpolate.RegularGridInterpolator` para grillas
- `scipy.spatial.cKDTree` para búsquedas de distancia eficientes

## Infraestructura Existente

| Herramienta | Archivo | Uso |
|------------|---------|-----|
| BathymetryFusion (GEBCO+SDB) | `core/cv_analysis/bathymetry.py` | Profundidad per-spot |
| RealDataPipeline | `core/cv_analysis/real_data_pipeline.py` | Modelo de sustrato |
| Species/Substrate matrix | `core/cv_analysis/species_zones.py` | 10 especies con afinidades |
| TideFetcher (harmonics) | `data/fetchers/tide_fetcher.py` | Mareas vectorizables |
| GEBCO NetCDF | `data/bathymetry/GEBCO_2025_peru*.nc` | Grilla de profundidad |
| Copernicus parquets | `data/copernicus/` | SSS, SLA, Chla en grilla |
| scipy (ya en requirements) | `requirements.txt` línea 23 | Interpolación + KDTree |

## Archivo a Modificar
Solo **`controllers/analysis.py`**

---

## FASE A: Mapa de alta resolución con sustrato y profundidad

### A1. Aumentar resolución a 25m
En `sample_fishing_spots()`:
- `spacing_m=750` → `spacing_m=25`, `max_spots=15000`
- Resultado: ~11,200 spots
- El sampling ya es vectorizado con `np.linspace` — sin cambios de lógica

### A2. Asignar profundidad GEBCO vectorizado
Nuevo método `_assign_depths_vectorized()`:
```python
from scipy.interpolate import RegularGridInterpolator
import netCDF4

def _assign_depths_vectorized(self):
    """Asigna profundidad GEBCO a todos los spots con interpolación vectorizada."""
    gebco_path = Path('data/bathymetry/GEBCO_2025_peru.nc')
    ds = netCDF4.Dataset(gebco_path)
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    elevation = ds.variables['elevation'][:]  # shape: (nlat, nlon)
    ds.close()

    interp = RegularGridInterpolator(
        (lats, lons), elevation,
        method='linear', bounds_error=False, fill_value=np.nan
    )

    # Vectorizado: interpolar TODOS los spots de una vez
    coords = np.array([[s['lat'], s['lon']] for s in self.sampled_spots])
    depths = interp(coords)  # shape: (N,) — una sola llamada

    for i, spot in enumerate(self.sampled_spots):
        spot['depth_m'] = float(depths[i]) if not np.isnan(depths[i]) else -10.0
        spot['depth_zone'] = self._classify_depth_zone(spot['depth_m'])
```
**Performance:** 1 lectura NetCDF (~50ms) + 1 interpolación vectorizada de 11,200 puntos (~5ms) = **~55ms total**

### A3. Calcular distancias a costa vectorizado con cKDTree
Nuevo método `_compute_coastal_distances_vectorized()`:
```python
from scipy.spatial import cKDTree

def _compute_coastal_distances_vectorized(self):
    """Distancia a costa para todos los spots con KDTree. O(N log M)."""
    # Convertir coastline a array numpy
    coast = np.array(self.coastline_points)  # shape: (M, 2)

    # Convertir a metros aproximados (lat/lon → plano local)
    R = 6371000
    cos_lat = np.cos(np.radians(coast[:, 0].mean()))
    coast_m = np.column_stack([coast[:, 0] * R * np.pi/180,
                                coast[:, 1] * R * np.pi/180 * cos_lat])

    tree = cKDTree(coast_m)

    # Coordenadas de todos los spots
    spots_arr = np.array([[s['lat'], s['lon']] for s in self.sampled_spots])
    spots_m = np.column_stack([spots_arr[:, 0] * R * np.pi/180,
                                spots_arr[:, 1] * R * np.pi/180 * cos_lat])

    # Query vectorizado — O(N log M)
    distances, _ = tree.query(spots_m)  # shape: (N,) en metros

    self._spot_coastal_distances = distances
    return distances
```
**Performance:** KDTree build de ~2,889 puntos: ~5ms. Query de 11,200 spots: ~2ms. **Total: ~7ms**

### A4. Asignar sustrato vectorizado
Nuevo método `_assign_substrate_vectorized()`:
```python
def _assign_substrate_vectorized(self):
    """Asigna sustrato basado en distancia a costa + ground truth."""
    dists = self._spot_coastal_distances  # ya computado

    # Modelo base por distancia (vectorizado)
    substrate = np.where(dists < 100, 'rock',
                np.where(dists < 500, 'mixed', 'sand'))

    # Override con ground truth de encuestas
    ground_truth = self._load_substrate_ground_truth()  # Dict[zona_nombre -> substrate]
    for zona, sust in ground_truth.items():
        # Encontrar spots cercanos a zonas conocidas con KDTree
        # y override su sustrato

    # Asignar a spots
    for i, spot in enumerate(self.sampled_spots):
        spot['substrate'] = substrate[i]
```

### A5. Actualizar especies por sustrato+profundidad
Usar matrix existente de `species_zones.py` (vectorizado):
```python
# Pseudo-código: vectorizado con numpy
depth_arr = np.array([s['depth_m'] for s in self.sampled_spots])
substrate_arr = np.array([s['substrate'] for s in self.sampled_spots])

# Para cada especie, calcular afinidad vectorizada
for species in SPECIES_DB:
    depth_mask = (depth_arr >= species.depth_min) & (depth_arr <= species.depth_max)
    substrate_mask = np.isin(substrate_arr, species.preferred_substrates)
    affinity = depth_mask.astype(float) * 0.5 + substrate_mask.astype(float) * 0.5
```

---

## FASE B: Zona intermareal y scores geográficos

### B1. Modelo de exposición intermareal vectorizado
Nuevo método `_compute_tidal_exposure_vectorized(hour)`:
```python
def _compute_tidal_exposure_vectorized(self, hour: int) -> np.ndarray:
    """Calcula factor de exposición para todos los spots. Vectorizado."""
    depths = np.array([s['depth_m'] for s in self.sampled_spots])

    # Altura de marea para esta hora (1 sola llamada — misma para todos)
    target_dt = self.analysis_datetime.replace(hour=hour)
    center_lat = np.mean([p[0] for p in self.coastline_points])
    center_lon = np.mean([p[1] for p in self.coastline_points])
    tide_height = self.tide_fetcher._calculate_tide_height(target_dt, center_lat, center_lon)

    # Profundidad efectiva = depth + tide (vectorizado)
    effective_depth = depths + tide_height  # shape: (N,)

    # Factor de exposición:
    # <-2m: completamente sumergido → factor 1.0 (bueno para pescar)
    # -2m a 0m: zona intermareal → factor gradual
    # >0m: expuesto → factor bajo (no hay agua suficiente)
    exposure = np.where(
        effective_depth < -2.0, 1.0,           # sumergido: ideal
        np.where(
            effective_depth < 0.0,
            0.5 + effective_depth / 4.0,        # intermareal: 0.0 a 1.0
            np.maximum(0.0, 0.3 - effective_depth * 0.3)  # expuesto: penalización
        )
    )
    return exposure  # shape: (N,)
```
**Performance:** Pure numpy ops sobre array de 11,200 → **~0.1ms**

### B2. Scores ambientales con interpolación espacial scipy
Nuevo método `_build_spatial_scores_vectorized()`:
```python
def _build_spatial_scores_vectorized(self) -> Dict[str, np.ndarray]:
    """Interpola SSS/SLA/Chla/SST para todos los spots. Vectorizado."""
    coords = np.array([[s['lat'], s['lon']] for s in self.sampled_spots])  # (N, 2)
    result = {}

    for var_name, fetcher, score_fn in [
        ('sss', self.physics_fetcher, 'calculate_sss_score'),
        ('sla', self.physics_fetcher, 'calculate_sla_score'),
        # ('chla', self.chla_fetcher, 'calculate_score'),
    ]:
        interp = self._build_interpolator(var_name)  # RegularGridInterpolator
        if interp is None:
            continue

        raw_values = interp(coords)  # (N,) — una llamada
        # Vectorizar el score function
        scores = np.vectorize(getattr(fetcher, score_fn))(raw_values)
        scores = np.where(np.isnan(scores), 0.5, scores)  # NaN → neutral
        result[var_name] = scores

    return result if result else None
```
**Performance:** 1 interpolación de 11,200 puntos por variable: ~2ms. 4 variables: **~8ms total**

### B3. Modificadores costeros vectorizados
```python
def _get_coastal_modifiers_vectorized(self) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (tide_mods, hour_mods) arrays de shape (N,)."""
    dists = self._spot_coastal_distances
    normalized = np.minimum(1.0, dists / 20000.0)
    tide_mods = 1.0 - 0.6 * normalized   # 1.0 → 0.4
    hour_mods = 1.0 - 0.5 * normalized   # 1.0 → 0.5
    return tide_mods, hour_mods
```
**Performance:** **~0.05ms** (3 numpy ops)

### B4. Modificar `analyze_spots()` — scoring vectorizado completo
```python
def analyze_spots(self, target_hour=None):
    # Pre-compute arrays UNA VEZ (cached)
    if not hasattr(self, '_spot_coastal_distances'):
        self._spot_coastal_distances = self._compute_coastal_distances_vectorized()
        self._tide_mods, self._hour_mods = self._get_coastal_modifiers_vectorized()
        self._per_spot_env = self._build_spatial_scores_vectorized()
        self._assign_depths_vectorized()
        self._assign_substrate_vectorized()

    # Scores que cambian por hora
    tide_score, tide_phase, hour_score = self._get_hourly_scores(target_hour)
    exposure = self._compute_tidal_exposure_vectorized(target_hour)

    # === LOOP VECTORIZADO ===
    N = len(self.sampled_spots)

    # Zone scores (estos sí necesitan loop por la lógica de distancia a zonas de peces)
    zone_scores = np.zeros(N)
    # ... (este loop se mantiene, pero es el único)

    # Environmental bonuses — TODO VECTORIZADO
    tide_bonus = (tide_score - 0.5) * 30 * self._tide_mods
    hour_bonus = (hour_score - 0.5) * 24 * self._hour_mods

    env = self._per_spot_env or {}
    sss_bonus = (env.get('sss', np.full(N, self.sss_score)) - 0.5) * 20
    sla_bonus = (env.get('sla', np.full(N, self.sla_score)) - 0.5) * 16
    chla_bonus = (env.get('chla', np.full(N, self.chla_score)) - 0.5) * 16
    sst_bonus = (env.get('sst', np.full(N, self.sst_historical_score)) - 0.5) * 12

    total_bonus = tide_bonus + hour_bonus + sss_bonus + sla_bonus + chla_bonus + sst_bonus
    total_bonus *= exposure  # penalizar spots expuestos por marea

    scores = np.clip(zone_scores + total_bonus, 0, 100)

    # Asignar de vuelta a spots
    for i, spot in enumerate(self.sampled_spots):
        spot['score'] = float(scores[i])
        spot['tide_phase'] = tide_phase
        # ... etc
```

### B5. Optimización para `analyze_spots_all_hours()`
- Scoring de 11,200 spots para 1 hora: ~50ms (vectorizado)
- 24 horas: ~1.2s
- **Pero el HTML solo embebe top-200 spots por hora** (no 11,200 × 24 = 268,800)
- Estrategia:
  1. Score 11,200 a la hora actual → full ranking
  2. Para las 24 horas: solo guardar top-200 en `hourly_spots_data`
  3. El mapa muestra todos los 11,200 como capa estática + top-200 dinámicos

---

## Budget de Performance

| Operación | Método | Tiempo |
|-----------|--------|--------|
| Muestreo 11,200 spots | `np.linspace` | ~200ms |
| GEBCO interpolación | `RegularGridInterpolator` (1 call) | ~55ms |
| Distancias a costa | `cKDTree` (build + query) | ~7ms |
| Sustrato assignment | `np.where` vectorizado | ~1ms |
| Modifiers costeros | `numpy` broadcast | ~0.05ms |
| SSS/SLA/Chla interp | `RegularGridInterpolator` × 4 | ~8ms |
| Tidal exposure per-hour | `numpy` conditional | ~0.1ms |
| Score per-hour (11,200) | `numpy` vectorizado | ~50ms |
| 24 horas scoring | 24 × score | ~1.2s |
| **Total overhead V8** | | **~1.5s** |

Comparado con V7 actual (~2min para 606 spots con loops Python), V8 con 11,200 spots será **más rápido** gracias a vectorización.

---

## Limitaciones y Mitigaciones

| Limitación | Mitigación |
|-----------|-----------|
| GEBCO ~450m res (pobre para 25m) | Interpolación lineal suaviza; futuro: SDB con Sentinel-2 |
| Sustrato por defecto = modelo distancia | Override con ground truth de encuestas_pesca.json |
| Dinámica sedimentaria (arena cubriendo rocas) | Registrar como observación manual; no modelable aún |
| scipy import | Ya en requirements.txt; fallback a loops si falta |

## Verificación
1. `python -m pytest tests/ -v` — tests pasan
2. `python main.py` — generar mapa (~30s esperado)
3. En HTML verificar:
   - ~11,200 puntos visibles a lo largo de 281km
   - Colores de sustrato diferenciados
   - Top-5 cambian al mover slider de hora
   - Spots intermareales aparecen/desaparecen con marea
   - Playa Meca → arena → lenguado como especie principal
4. Comparar score std dev antes vs después (debe aumentar)
