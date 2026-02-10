# Plan V8: Computer Vision + Batimetría + Zonas de Especies

**Versión:** 8.0
**Fecha:** 2026-02-08
**Estado:** PLANIFICADO
**Prerequisitos:** V1-V7 Completados

---

## Resumen Ejecutivo

Sistema integral de análisis costero que combina:
- **Computer Vision** para detectar borde agua/tierra y clasificar sustrato
- **Batimetría** satelital (aguas poco profundas) + GEBCO (profundas)
- **Zonas de especies** basadas en hábitat (sustrato + profundidad)
- **Visualización** con polígonos coloreados por zona/especie

---

## 1. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE CV + BATIMETRÍA                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   ENTRADA    │    │  PROCESAMIENTO│   │    SALIDA    │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ Imagen      │     │ SAM Model   │     │ Coastline   │               │
│  │ Satelital   │────▶│ Segmentación│────▶│ GeoJSON     │               │
│  │ (ESRI/Google)│    │ Agua/Tierra │     │ Precisión 5m│               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ Bandas RGB  │     │ Clasificador│     │ Sustrato    │               │
│  │ + Textura   │────▶│ Roca/Arena  │────▶│ GeoJSON     │               │
│  │             │     │ (Color+CNN) │     │ Polígonos   │               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ Bandas      │     │ Algoritmo   │     │ Batimetría  │               │
│  │ Blue/Green  │────▶│ SDB Stumpf  │────▶│ Raster      │               │
│  │ (Sentinel-2)│     │ ln(B/G)     │     │ 0-30m       │               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ GEBCO 2023  │     │ Merge +     │     │ Batimetría  │               │
│  │ NetCDF      │────▶│ Interpolate │────▶│ Completa    │               │
│  │ 15 arc-sec  │     │             │     │ 0-500m+     │               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│                             │                                           │
│                             ▼                                           │
│                      ┌─────────────┐                                    │
│                      │ GENERADOR   │                                    │
│                      │ DE ZONAS    │                                    │
│                      │             │                                    │
│                      │ Sustrato +  │                                    │
│                      │ Profundidad │                                    │
│                      │ = Especies  │                                    │
│                      └──────┬──────┘                                    │
│                             │                                           │
│                             ▼                                           │
│                      ┌─────────────┐                                    │
│                      │ MAPA FINAL  │                                    │
│                      │             │                                    │
│                      │ • Zonas     │                                    │
│                      │ • Batimetría│                                    │
│                      │ • Spots     │                                    │
│                      └─────────────┘                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes del Sistema

### 2.1 Detección de Línea Costera (SAM)

**Modelo:** Segment Anything Model (Meta AI)
**Precisión objetivo:** 2-5 metros
**Hardware:** MacBook M3 Pro con MPS

```python
class CoastlineDetectorCV:
    """
    Detector de línea costera usando Computer Vision.

    Flujo:
    1. Descargar tiles satelitales de alta resolución
    2. Segmentar agua/tierra con SAM
    3. Extraer contorno como línea costera
    4. Refinar y suavizar
    5. Exportar como GeoJSON
    """

    def __init__(self):
        self.sam_model = load_sam_model('vit_h')
        self.tile_zoom = 17  # ~1.2m/pixel

    def detect_coastline(self, bounds: BoundingBox) -> LineString:
        """Detecta línea costera en el área especificada."""
        pass
```

**Configuración SAM:**
```python
SAM_CONFIG = {
    'model_type': 'vit_h',
    'checkpoint': 'sam_vit_h_4b8939.pth',
    'device': 'mps',  # Metal Performance Shaders
    'points_per_side': 32,
    'pred_iou_thresh': 0.88,
    'stability_score_thresh': 0.95,
}
```

### 2.2 Clasificación de Sustrato

**Método:** Análisis de color + textura
**Clases:** Roca, Arena, Mixto

```python
class SubstrateClassifier:
    """
    Clasifica el sustrato costero visible.

    Características analizadas:
    - Color HSV (roca=gris oscuro, arena=beige)
    - Textura (roca=irregular, arena=suave)
    - Patrón de oleaje (roca=espuma irregular)
    """

    SUBSTRATE_COLORS = {
        'roca': {
            'h_range': (0, 30),      # Grises
            'v_range': (20, 80),     # Oscuro a medio
            'texture': 'high_variance'
        },
        'arena': {
            'h_range': (15, 35),     # Beige/dorado
            'v_range': (60, 95),     # Claro
            'texture': 'low_variance'
        }
    }

    def classify(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Polygon]:
        """Clasifica zonas de sustrato y retorna polígonos."""
        pass
```

**Firmas espectrales típicas:**

| Sustrato | Color Visual | H (HSV) | S (HSV) | V (HSV) | Textura |
|----------|--------------|---------|---------|---------|---------|
| Roca seca | Gris oscuro | 0-20 | 10-30 | 20-50 | Alta varianza |
| Roca húmeda | Gris-verde | 80-120 | 20-50 | 30-60 | Alta varianza |
| Arena seca | Beige claro | 20-35 | 20-50 | 70-95 | Baja varianza |
| Arena húmeda | Beige oscuro | 20-35 | 30-60 | 40-70 | Baja varianza |

### 2.3 Batimetría Derivada de Satélite (SDB)

**Método:** Algoritmo de Stumpf (ratio de bandas)
**Rango efectivo:** 0-30 metros
**Fuente:** Sentinel-2 (10m resolución)

```python
class SatelliteBathymetry:
    """
    Estima profundidad en aguas poco profundas usando
    el ratio de bandas azul/verde (algoritmo Stumpf).

    Principio físico:
    - Agua absorbe luz roja rápidamente
    - Luz azul penetra más profundo
    - Ratio B/G correlaciona con profundidad

    Limitaciones:
    - Solo funciona en agua clara (turbidez baja)
    - Rango efectivo: 0-30m
    - Requiere calibración con datos reales
    """

    # Coeficientes calibrados para Pacífico Sur
    STUMPF_COEFFICIENTS = {
        'm0': 52.073,    # Offset
        'm1': 50.156,    # Pendiente
        'n': 1000,       # Factor de escala
    }

    def estimate_depth(self, blue: np.ndarray, green: np.ndarray) -> np.ndarray:
        """
        Estima profundidad usando algoritmo Stumpf.

        depth = m0 + m1 * ln(n * blue / green)

        Args:
            blue: Banda azul (B2 Sentinel-2)
            green: Banda verde (B3 Sentinel-2)

        Returns:
            Array de profundidad en metros
        """
        ratio = np.log(self.STUMPF_COEFFICIENTS['n'] * blue / green)
        depth = self.STUMPF_COEFFICIENTS['m0'] + \
                self.STUMPF_COEFFICIENTS['m1'] * ratio

        # Limitar a rango válido
        depth = np.clip(depth, 0, 30)

        return depth
```

**Indicadores visuales de profundidad:**

| Profundidad | Color del agua | Características |
|-------------|----------------|-----------------|
| 0-3m | Turquesa brillante | Fondo visible, oleaje |
| 3-8m | Turquesa/Azul claro | Fondo parcialmente visible |
| 8-15m | Azul medio | Fondo no visible |
| 15-30m | Azul oscuro | Agua profunda costera |
| >30m | Azul muy oscuro | Usar GEBCO |

### 2.4 Batimetría GEBCO

**Dataset:** GEBCO 2023 Grid
**Resolución:** 15 arc-second (~450m)
**Formato:** NetCDF

```python
class GEBCOBathymetry:
    """
    Proporciona batimetría de GEBCO para aguas profundas.

    GEBCO (General Bathymetric Chart of the Oceans) es el
    dataset batimétrico global más completo y preciso.
    """

    GEBCO_URL = "https://www.gebco.net/data_and_products/gridded_bathymetry_data/"

    def __init__(self, netcdf_path: Path):
        self.dataset = xr.open_dataset(netcdf_path)

    def get_depth(self, lat: float, lon: float) -> float:
        """Obtiene profundidad en un punto."""
        return float(self.dataset.elevation.sel(
            lat=lat, lon=lon, method='nearest'
        ).values)

    def get_depth_contours(
        self,
        bounds: BoundingBox,
        levels: List[float] = [-5, -10, -15, -20, -30, -50, -100]
    ) -> List[LineString]:
        """Genera contornos de profundidad."""
        pass
```

### 2.5 Fusión de Batimetría

```python
class BathymetryFusion:
    """
    Fusiona batimetría satelital (SDB) con GEBCO.

    Estrategia:
    - 0-25m: Usar SDB (mayor resolución)
    - 25-30m: Transición suave (weighted average)
    - >30m: Usar GEBCO
    """

    def fuse(
        self,
        sdb: np.ndarray,          # Satellite-derived (0-30m)
        gebco: np.ndarray,        # GEBCO (all depths)
        transition_start: float = 25,
        transition_end: float = 30
    ) -> np.ndarray:
        """
        Fusiona las dos fuentes de batimetría.

        En la zona de transición (25-30m), usa weighted average
        para evitar discontinuidades.
        """
        # Crear máscara de transición
        alpha = np.clip(
            (sdb - transition_start) / (transition_end - transition_start),
            0, 1
        )

        # Weighted average en zona de transición
        fused = (1 - alpha) * sdb + alpha * gebco

        # Fuera de transición: usar fuente apropiada
        fused = np.where(sdb < transition_start, sdb, fused)
        fused = np.where(sdb >= transition_end, gebco, fused)

        return fused
```

---

## 3. Generación de Zonas de Especies

### 3.1 Matriz de Hábitat

| Especie | Sustrato | Prof. Mín | Prof. Máx | SST Óptima | Notas |
|---------|----------|-----------|-----------|------------|-------|
| **Cabrilla** | Roca | 2m | 25m | 16-19°C | Fondos rocosos |
| **Pintadilla** | Roca | 1m | 15m | 15-20°C | Muy pegada a roca |
| **Chita** | Roca | 3m | 20m | 16-20°C | Zonas de corriente |
| **Corvina** | Arena/Mixto | 5m | 40m | 15-18°C | Pozos y canales |
| **Lenguado** | Arena | 8m | 50m | 14-18°C | Fondo arenoso |
| **Robalo** | Mixto | 2m | 30m | 16-22°C | Desembocaduras |
| **Pejerrey** | Arena | 0m | 15m | 14-20°C | Cardúmenes costeros |
| **Fortuno** | Roca | 5m | 30m | 15-19°C | Pozos rocosos |

### 3.2 Generador de Zonas

```python
class SpeciesZoneGenerator:
    """
    Genera polígonos de zonas de pesca por especie.

    Cruza información de:
    - Sustrato (roca/arena/mixto)
    - Profundidad (batimetría)
    - Condiciones (SST, corrientes)

    Produce zonas coloreadas con especies probables.
    """

    HABITAT_RULES = {
        'Cabrilla': {
            'substrate': ['roca', 'mixto'],
            'depth_min': 2,
            'depth_max': 25,
            'sst_optimal': (16, 19),
            'color': '#FF6B35',  # Naranja
            'priority': 1
        },
        'Corvina': {
            'substrate': ['arena', 'mixto'],
            'depth_min': 5,
            'depth_max': 40,
            'sst_optimal': (15, 18),
            'color': '#004E89',  # Azul
            'priority': 2
        },
        'Lenguado': {
            'substrate': ['arena'],
            'depth_min': 8,
            'depth_max': 50,
            'sst_optimal': (14, 18),
            'color': '#7B2D26',  # Marrón
            'priority': 3
        },
        'Robalo': {
            'substrate': ['mixto', 'roca'],
            'depth_min': 2,
            'depth_max': 30,
            'sst_optimal': (16, 22),
            'color': '#1A936F',  # Verde
            'priority': 4
        },
        'Pejerrey': {
            'substrate': ['arena'],
            'depth_min': 0,
            'depth_max': 15,
            'sst_optimal': (14, 20),
            'color': '#88D498',  # Verde claro
            'priority': 5
        }
    }

    def generate_zones(
        self,
        substrate: Dict[str, Polygon],
        bathymetry: np.ndarray,
        sst: float
    ) -> Dict[str, List[SpeciesZone]]:
        """
        Genera zonas de especies.

        Returns:
            Dict con especies como keys y lista de SpeciesZone como values
        """
        zones = {}

        for species, rules in self.HABITAT_RULES.items():
            # Filtrar por sustrato
            suitable_substrate = self._filter_substrate(
                substrate, rules['substrate']
            )

            # Filtrar por profundidad
            suitable_depth = self._filter_depth(
                bathymetry,
                rules['depth_min'],
                rules['depth_max']
            )

            # Intersección
            species_zone = suitable_substrate.intersection(suitable_depth)

            # Calcular score basado en SST
            sst_score = self._calculate_sst_score(sst, rules['sst_optimal'])

            zones[species] = SpeciesZone(
                species=species,
                polygon=species_zone,
                color=rules['color'],
                score=sst_score,
                depth_range=(rules['depth_min'], rules['depth_max']),
                substrate=rules['substrate']
            )

        return zones
```

### 3.3 Estructura de Zona

```python
@dataclass
class SpeciesZone:
    """Zona de hábitat para una especie."""
    species: str
    polygon: Polygon
    color: str
    score: float  # 0-1 basado en condiciones actuales
    depth_range: Tuple[float, float]
    substrate: List[str]

    def to_geojson_feature(self) -> Dict:
        """Convierte a feature GeoJSON."""
        return {
            "type": "Feature",
            "geometry": mapping(self.polygon),
            "properties": {
                "species": self.species,
                "color": self.color,
                "score": self.score,
                "depth_min": self.depth_range[0],
                "depth_max": self.depth_range[1],
                "substrate": self.substrate
            }
        }
```

---

## 4. Visualización en Mapa

### 4.1 Capas del Mapa

```python
class EnhancedMapView:
    """
    Mapa mejorado con zonas de especies y batimetría.

    Capas (de abajo hacia arriba):
    1. Imagen satelital (base)
    2. Batimetría (contornos)
    3. Zonas de especies (polígonos semi-transparentes)
    4. Línea costera precisa
    5. Spots de pesca (puntos)
    6. Leyenda interactiva
    """

    def add_bathymetry_layer(self, bathymetry: np.ndarray, bounds: BoundingBox):
        """Agrega capa de batimetría como contornos."""
        contour_levels = [-5, -10, -15, -20, -30, -50]
        colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']

        for level, color in zip(contour_levels, colors):
            contour = self._extract_contour(bathymetry, level)
            self.map.add_child(folium.PolyLine(
                contour,
                color=color,
                weight=1,
                opacity=0.7,
                popup=f"{abs(level)}m depth"
            ))

    def add_species_zones(self, zones: Dict[str, SpeciesZone]):
        """Agrega zonas de especies como polígonos."""
        for species, zone in zones.items():
            self.map.add_child(folium.Polygon(
                locations=list(zone.polygon.exterior.coords),
                color=zone.color,
                fill=True,
                fill_color=zone.color,
                fill_opacity=0.3,
                popup=self._create_zone_popup(zone)
            ))

    def _create_zone_popup(self, zone: SpeciesZone) -> str:
        """Crea popup informativo para zona."""
        return f"""
        <div style="font-family: Arial; padding: 10px;">
            <h4 style="color: {zone.color}; margin: 0;">🐟 {zone.species}</h4>
            <hr style="margin: 5px 0;">
            <b>Profundidad:</b> {zone.depth_range[0]}-{zone.depth_range[1]}m<br>
            <b>Sustrato:</b> {', '.join(zone.substrate)}<br>
            <b>Score actual:</b> {zone.score:.0%}
        </div>
        """
```

### 4.2 Leyenda Interactiva

```javascript
// Leyenda con toggle de capas
const legend = L.control({position: 'bottomright'});

legend.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'legend');
    div.innerHTML = `
        <h4>🗺️ Capas</h4>

        <div class="legend-section">
            <b>Zonas de Especies</b>
            <label><input type="checkbox" checked data-layer="cabrilla">
                <span style="color:#FF6B35">■</span> Cabrilla (roca 2-25m)</label>
            <label><input type="checkbox" checked data-layer="corvina">
                <span style="color:#004E89">■</span> Corvina (arena 5-40m)</label>
            <label><input type="checkbox" checked data-layer="robalo">
                <span style="color:#1A936F">■</span> Robalo (mixto 2-30m)</label>
        </div>

        <div class="legend-section">
            <b>Batimetría</b>
            <div class="depth-gradient"></div>
            <div class="depth-labels">
                <span>0m</span>
                <span>15m</span>
                <span>30m+</span>
            </div>
        </div>

        <div class="legend-section">
            <b>Sustrato</b>
            <span style="color:#8B4513">▨</span> Roca
            <span style="color:#DEB887">▨</span> Arena
            <span style="color:#9ACD32">▨</span> Mixto
        </div>
    `;
    return div;
};
```

### 4.3 Ejemplo Visual

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     🟠🟠🟠                      Leyenda:                    │
│   🟠🟠🟠🟠🟠    ←Zona rocosa    ────────────────           │
│     🟠🟠🟠      (Cabrilla)      🟠 Cabrilla (roca)         │
│       ║                         🔵 Corvina (arena)         │
│       ║ ←Línea costera         🟢 Robalo (mixto)          │
│    🔵🔵║🔵🔵                                               │
│  🔵🔵🔵║🔵🔵🔵  ←Zona arenosa   ─── Contorno 10m          │
│    🔵🔵║🔵🔵     (Corvina)      ─── Contorno 20m          │
│       ║                         ─── Contorno 30m          │
│  🟢🟢🟢🟢🟢🟢  ←Zona mixta                                 │
│    🟢🟢🟢🟢      (Robalo)       ● Spot de pesca           │
│       │                                                     │
│      ─┴── Contorno 30m                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Flujo de Datos

### 5.1 Pipeline Completo

```python
class CVBathymetryPipeline:
    """
    Pipeline completo de análisis CV + Batimetría.

    Ejecuta todo el proceso desde imágenes satelitales
    hasta mapa final con zonas de especies.
    """

    def __init__(self):
        self.coastline_detector = CoastlineDetectorCV()
        self.substrate_classifier = SubstrateClassifier()
        self.sdb_estimator = SatelliteBathymetry()
        self.gebco = GEBCOBathymetry()
        self.zone_generator = SpeciesZoneGenerator()
        self.map_view = EnhancedMapView()

    def run(self, bounds: BoundingBox) -> Path:
        """
        Ejecuta pipeline completo.

        Args:
            bounds: Área a analizar

        Returns:
            Path al mapa HTML generado
        """
        print("=" * 60)
        print("PIPELINE CV + BATIMETRÍA")
        print("=" * 60)

        # 1. Descargar imágenes satelitales
        print("\n[1/7] Descargando imágenes satelitales...")
        rgb_image = self._download_satellite_image(bounds, 'rgb')
        sentinel_image = self._download_satellite_image(bounds, 'sentinel')

        # 2. Detectar línea costera
        print("[2/7] Detectando línea costera con SAM...")
        coastline = self.coastline_detector.detect_coastline(rgb_image)
        print(f"      {len(coastline.coords)} puntos, precisión ~5m")

        # 3. Clasificar sustrato
        print("[3/7] Clasificando sustrato (roca/arena)...")
        substrate_zones = self.substrate_classifier.classify(rgb_image)
        print(f"      {len(substrate_zones)} zonas identificadas")

        # 4. Estimar batimetría costera (SDB)
        print("[4/7] Estimando batimetría costera (<30m)...")
        sdb = self.sdb_estimator.estimate_depth(
            sentinel_image['blue'],
            sentinel_image['green']
        )

        # 5. Obtener batimetría GEBCO (profunda)
        print("[5/7] Cargando batimetría GEBCO...")
        gebco = self.gebco.get_depth_grid(bounds)

        # 6. Fusionar batimetrías
        print("[6/7] Fusionando batimetrías...")
        bathymetry = BathymetryFusion().fuse(sdb, gebco)

        # 7. Generar zonas de especies
        print("[7/7] Generando zonas de especies...")
        sst = self._get_current_sst(bounds)
        species_zones = self.zone_generator.generate_zones(
            substrate_zones, bathymetry, sst
        )

        # Crear mapa
        print("\nGenerando mapa...")
        output_path = self._create_map(
            coastline, substrate_zones, bathymetry, species_zones
        )

        print(f"\n{'=' * 60}")
        print(f"MAPA GENERADO: {output_path}")
        print(f"{'=' * 60}")

        return output_path
```

### 5.2 Estructura de Archivos de Salida

```
output/
├── cv_analysis/
│   ├── coastline_cv.geojson        # Línea costera precisa
│   ├── substrate_zones.geojson     # Polígonos de sustrato
│   ├── bathymetry_sdb.tif          # Batimetría satelital
│   ├── bathymetry_fused.tif        # Batimetría fusionada
│   ├── species_zones.geojson       # Zonas de especies
│   └── analysis_report.json        # Métricas y estadísticas
│
├── maps/
│   ├── fishing_zones_map.html      # Mapa principal
│   └── bathymetry_map.html         # Mapa solo batimetría
│
└── cache/
    ├── satellite_tiles/            # Tiles descargados
    └── gebco/                      # Datos GEBCO
```

---

## 6. Requerimientos Técnicos

### 6.1 Dependencias

```python
# requirements_cv.txt

# Computer Vision
segment-anything>=1.0
torch>=2.0
torchvision>=0.15
opencv-python>=4.8

# Geoespacial
rasterio>=1.3
shapely>=2.0
geopandas>=0.14
pyproj>=3.6

# Datos
xarray>=2023.1
netCDF4>=1.6
sentinelsat>=1.2  # Para Sentinel-2

# Visualización
folium>=0.14
matplotlib>=3.7

# Científico
numpy>=1.24
scipy>=1.11
scikit-image>=0.21
```

### 6.2 Hardware Recomendado

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| RAM | 16GB | 32GB |
| GPU | Metal (M1/M2/M3) | Metal + 16GB VRAM |
| Almacenamiento | 10GB libres | 50GB libres |
| CPU | 4 cores | 8+ cores |

### 6.3 APIs y Datos Externos

| Servicio | Uso | Autenticación |
|----------|-----|---------------|
| ESRI World Imagery | Tiles satelitales RGB | API Key (gratuito) |
| Copernicus Sentinel Hub | Sentinel-2 multiespectral | OAuth2 |
| GEBCO | Batimetría global | Ninguna (descarga directa) |

---

## 7. Cronograma de Implementación

| Fase | Tareas | Duración Est. |
|------|--------|---------------|
| **1** | Setup SAM + descarga GEBCO | 1 día |
| **2** | Detector de línea costera | 2 días |
| **3** | Clasificador de sustrato | 2 días |
| **4** | Batimetría SDB | 2 días |
| **5** | Fusión + zonas de especies | 1 día |
| **6** | Visualización mejorada | 1 día |
| **7** | Testing + ajustes | 1 día |

**Total estimado:** 10 días

---

## 8. Validación

### 8.1 Métricas de Calidad

| Métrica | Objetivo | Método |
|---------|----------|--------|
| Precisión línea costera | <10m error | Comparar con GPS |
| Accuracy sustrato | >80% | Validación visual |
| Error batimetría SDB | <3m (0-15m) | Comparar con cartas náuticas |
| Cobertura zonas | >90% del área | Inspección visual |

### 8.2 Zona de Prueba

**Área sugerida:** La zona de tu imagen
- Coordenadas: ~(-17.7, -71.4) a (-17.6, -71.3)
- Características: Tiene roca, arena, y variación de profundidad
- Tamaño: ~10km de costa

---

## 9. Comandos de Ejecución

```bash
# Descargar GEBCO para la región
python scripts/download_gebco.py --region tacna-ilo

# Ejecutar análisis CV en zona de prueba
python scripts/run_cv_analysis.py \
    --lat-min -17.7 \
    --lat-max -17.6 \
    --lon-min -71.4 \
    --lon-max -71.3 \
    --output output/cv_analysis/

# Generar mapa con zonas
python scripts/generate_zones_map.py \
    --cv-dir output/cv_analysis/ \
    --output output/maps/fishing_zones_map.html

# Pipeline completo
python main.py --mode cv-full --region test
```

---

## 10. Integración con Sistema Existente (V7)

El sistema CV+Batimetría se integra con V7:

```python
# En AnalysisController

def analyze_spots_v8(self, target_hour: int = 6) -> List[dict]:
    """
    Análisis V8 con zonas de especies.

    Además del scoring V7, agrega:
    - Zona de especie donde está el spot
    - Profundidad estimada
    - Sustrato
    """
    # Scoring V7 existente
    results = self.analyze_spots_v7(target_hour)

    # Enriquecer con datos CV
    for spot in results:
        # Obtener zona de especie
        zone = self.zone_generator.get_zone_at(spot['lat'], spot['lon'])
        spot['species_zone'] = zone.species if zone else None
        spot['zone_color'] = zone.color if zone else '#888888'

        # Obtener profundidad
        depth = self.bathymetry.get_depth(spot['lat'], spot['lon'])
        spot['depth_m'] = depth

        # Obtener sustrato
        substrate = self.substrate.get_type_at(spot['lat'], spot['lon'])
        spot['substrate'] = substrate

        # Bonus por coincidencia zona/especie
        if zone and spot.get('target_species') == zone.species:
            spot['score'] += 5  # Bonus por estar en zona correcta

    return results
```

---

*Plan V8 creado: 2026-02-08*
*Proyecto: Fishing Predictor - Tacna/Ilo/Sama, Perú*
*Autor: Sistema de Predicción Pesquera*
