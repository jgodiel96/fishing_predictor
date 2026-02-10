# Plan V8 Actualizado: Vision Computacional para Deteccion Costera

## Revision Critica del Enfoque Anterior

### Problema Identificado

El enfoque inicial tenia una **incompatibilidad fundamental**:

| Lo que usabamos | Lo que requieren los modelos |
|-----------------|------------------------------|
| ESRI World Imagery (RGB, 3 bandas) | DeepWaterMap: 6 bandas (RGB+NIR+SWIR) |
| Tiles visuales procesados | WatNet: 6 bandas Sentinel-2 |
| Sin informacion espectral | Modelos entrenados en datos multiespectrales |

**Resultado**: Los algoritmos HSV/color fallaban porque RGB no tiene la discriminacion espectral necesaria para separar agua de sombras, rocas mojadas, etc.

---

## Nuevo Plan de Implementacion

### Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA V8 ACTUALIZADO                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CAPA 1: DATOS BASE (Sin CV)                                   │
│  ├── Linea costera: OpenStreetMap Coastlines                   │
│  │   └── Descarga: osmdata.openstreetmap.de/data/coastlines    │
│  │   └── Formato: Shapefile, actualizacion diaria              │
│  │   └── Precision: Alta, verificada por humanos               │
│  │                                                              │
│  ├── Batimetria: GEBCO 2023                                    │
│  │   └── Resolucion: ~450m global                              │
│  │   └── Formato: NetCDF                                       │
│  │   └── Cobertura: Global, incluye Peru                       │
│  │                                                              │
│  └── Tiles visuales: ESRI World Imagery                        │
│      └── Solo para visualizacion en mapa                       │
│      └── NO para deteccion automatica                          │
│                                                                 │
│  CAPA 2: DATOS OPCIONALES (Con CV avanzado)                    │
│  ├── Sentinel-2 L2A (si se requiere deteccion CV)              │
│  │   └── API: Copernicus Data Space Ecosystem                  │
│  │   └── Bandas: B02,B03,B04,B08,B11,B12                       │
│  │   └── Resolucion: 10-20m                                    │
│  │                                                              │
│  └── SAM/SamGeo (alternativa para RGB)                         │
│      └── Funciona con tiles ESRI                               │
│      └── Requiere GPU (8GB+ VRAM)                              │
│      └── Precision variable segun prompts                      │
│                                                                 │
│  CAPA 3: GENERACION DE ZONAS                                   │
│  ├── Zonas por distancia a costa (desde OSM)                   │
│  ├── Zonas por profundidad (desde GEBCO)                       │
│  └── Combinacion: especies segun habitat                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fase 1: Implementacion Base (Sin dependencia de CV)

### 1.1 Descarga de Linea Costera OSM

```python
# Fuente: https://osmdata.openstreetmap.de/data/coastlines.html
# Archivo: water-polygons-split-4326.zip (WGS84)

class OSMCoastlineLoader:
    """Carga linea costera desde OpenStreetMap."""

    def load_coastline(self, bounds: Tuple[float, float, float, float]) -> List[Polygon]:
        """
        Carga poligonos de agua para el area especificada.

        Args:
            bounds: (lat_min, lat_max, lon_min, lon_max)

        Returns:
            Lista de poligonos Shapely
        """
        pass

    def get_coastline_points(self, bounds, resolution_m=50) -> List[Tuple[float, float]]:
        """Extrae puntos de la linea costera con resolucion especificada."""
        pass
```

### 1.2 Integracion GEBCO

```python
# Fuente: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
# Archivo: GEBCO_2023 sub-ice topo/bathy

class GEBCOBathymetry:
    """Batimetria real desde GEBCO."""

    def load_region(self, bounds: Tuple[float, float, float, float]) -> xr.DataArray:
        """Carga datos GEBCO para la region."""
        pass

    def get_depth_at_point(self, lat: float, lon: float) -> float:
        """Retorna profundidad en metros (negativo = bajo agua)."""
        pass

    def get_depth_contours(self, depths: List[float]) -> Dict[float, List[Polygon]]:
        """Genera contornos de isoprofundidad."""
        pass
```

### 1.3 Generador de Zonas de Especies

```python
class SpeciesZoneGenerator:
    """
    Genera zonas de pesca basadas en:
    - Distancia a la costa (desde OSM)
    - Profundidad (desde GEBCO)
    - Preferencias de habitat por especie
    """

    def generate_zones(
        self,
        coastline: List[Polygon],
        bathymetry: xr.DataArray,
        species_config: Dict
    ) -> List[SpeciesZone]:
        """
        Genera zonas coloreadas por especie.

        Logica:
        1. Buffer desde costa: 0-100m, 100-500m, 500-2000m
        2. Interseccion con contornos de profundidad
        3. Asignacion de especies segun matriz de habitat
        """
        pass
```

---

## Fase 2: CV Opcional con Sentinel-2

### 2.1 Descarga de Sentinel-2

```python
# Requiere cuenta gratuita en: https://dataspace.copernicus.eu

class Sentinel2Downloader:
    """Descarga imagenes Sentinel-2 L2A."""

    def __init__(self, credentials_file: str):
        self.api = CopernicusAPI(credentials_file)

    def download_scene(
        self,
        bounds: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        max_cloud_cover: float = 20.0
    ) -> Path:
        """
        Descarga escena Sentinel-2 con menor nubosidad.

        Returns:
            Path al archivo .SAFE descargado
        """
        pass

    def extract_bands(self, safe_path: Path) -> Dict[str, np.ndarray]:
        """Extrae bandas necesarias: B02, B03, B04, B08, B11, B12."""
        pass
```

### 2.2 Modelos Compatibles con Sentinel-2

```python
# Opcion A: DeepWaterMap (pre-entrenado)
class DeepWaterMapSegmenter:
    """
    Usa DeepWaterMap v2 para segmentacion de agua.

    Input requerido:
    - 6 bandas: Blue, Green, Red, NIR, SWIR1, SWIR2
    - Normalizadas segun especificacion del modelo

    Fuente: https://github.com/isikdogan/deepwatermap
    """
    pass

# Opcion B: WatNet (pre-entrenado)
class WatNetSegmenter:
    """
    Usa WatNet para segmentacion de agua en Sentinel-2.

    Input requerido:
    - Bandas Sentinel-2: B2, B3, B4, B8, B11, B12

    Fuente: https://github.com/xinluo2018/WatNet
    """
    pass
```

---

## Fase 3: CV Alternativo con SAM (RGB)

### 3.1 SamGeo para Tiles ESRI

```python
# Requiere: pip install segment-geospatial
# Requiere: GPU con 8GB+ VRAM

class SamGeoSegmenter:
    """
    Usa Segment Anything Model para segmentacion.

    Ventajas:
    - Funciona con RGB puro
    - No requiere bandas espectrales
    - Flexible con prompts

    Desventajas:
    - Requiere GPU
    - Precision depende de prompts
    """

    def __init__(self, model_type: str = "vit_h"):
        from samgeo import SamGeo
        self.sam = SamGeo(model_type=model_type)

    def segment_water(
        self,
        image_path: str,
        prompt_points: List[Tuple[int, int]] = None,
        text_prompt: str = "ocean water"
    ) -> np.ndarray:
        """
        Segmenta agua usando SAM.

        Args:
            image_path: Ruta a imagen RGB
            prompt_points: Puntos de ejemplo en agua [(x,y), ...]
            text_prompt: Descripcion textual (requiere Grounding DINO)

        Returns:
            Mascara binaria de agua
        """
        pass
```

---

## Matriz de Habitat por Especie

```python
SPECIES_HABITAT_MATRIX = {
    'corvina': {
        'depth_range': (5, 40),      # metros
        'optimal_depth': 15,
        'distance_from_shore': (50, 2000),  # metros
        'substrate_preference': ['rock', 'mixed'],
        'color': (255, 165, 0),      # Naranja
    },
    'lenguado': {
        'depth_range': (2, 50),
        'optimal_depth': 20,
        'distance_from_shore': (100, 3000),
        'substrate_preference': ['sand'],
        'color': (255, 215, 0),      # Dorado
    },
    'cabrilla': {
        'depth_range': (2, 30),
        'optimal_depth': 10,
        'distance_from_shore': (10, 500),
        'substrate_preference': ['rock'],
        'color': (255, 69, 0),       # Rojo-naranja
    },
    'chita': {
        'depth_range': (1, 20),
        'optimal_depth': 8,
        'distance_from_shore': (5, 300),
        'substrate_preference': ['rock', 'mixed'],
        'color': (220, 20, 60),      # Carmesi
    },
    'pejerrey': {
        'depth_range': (0, 15),
        'optimal_depth': 5,
        'distance_from_shore': (0, 500),
        'substrate_preference': ['sand', 'mixed'],
        'color': (70, 130, 180),     # Azul acero
    },
    'lorna': {
        'depth_range': (3, 35),
        'optimal_depth': 15,
        'distance_from_shore': (50, 1500),
        'substrate_preference': ['sand', 'mixed'],
        'color': (147, 112, 219),    # Purpura
    },
}
```

---

## Estructura de Archivos Actualizada

```
fishing_predictor/
├── core/
│   └── cv_analysis/
│       ├── __init__.py
│       ├── osm_coastline.py      # NUEVO: Carga OSM
│       ├── gebco_bathymetry.py   # ACTUALIZADO: Solo GEBCO
│       ├── species_zones.py      # ACTUALIZADO: Basado en datos reales
│       ├── sentinel2_loader.py   # NUEVO: Descarga S2 (opcional)
│       ├── samgeo_segmenter.py   # NUEVO: SAM para RGB (opcional)
│       └── pipeline.py           # ACTUALIZADO: Usa datos reales
│
├── data/
│   ├── coastlines/               # NUEVO: Datos OSM
│   │   └── water-polygons-peru.shp
│   ├── bathymetry/               # NUEVO: Datos GEBCO
│   │   └── gebco_2023_peru.nc
│   └── sentinel2/                # NUEVO: Escenas S2 (opcional)
│
└── docs/
    ├── PLAN_V8_CV_ACTUALIZADO.md           # Este archivo
    └── INVESTIGACION_CV_ESTADO_DEL_ARTE.md # Investigacion completa
```

---

## Dependencias Actualizadas

```
# requirements_cv.txt

# Base (siempre necesario)
numpy>=1.24.0
opencv-python>=4.8.0
shapely>=2.0.0
geopandas>=0.14.0
xarray>=2023.1.0
netCDF4>=1.6.0
requests>=2.31.0

# Opcional: Sentinel-2
openeo>=0.23.0
rasterio>=1.3.0

# Opcional: SAM (requiere GPU)
segment-geospatial>=0.10.0
torch>=2.0.0
```

---

## Plan de Ejecucion

### Semana 1: Implementacion Base
1. [ ] Descargar coastlines OSM para Peru
2. [ ] Descargar GEBCO para region costera Peru
3. [ ] Implementar `osm_coastline.py`
4. [ ] Actualizar `gebco_bathymetry.py`
5. [ ] Crear generador de zonas basado en datos reales

### Semana 2: Integracion y Pruebas
1. [ ] Integrar con pipeline existente
2. [ ] Generar zonas para area de prueba (Ilo)
3. [ ] Crear visualizador actualizado
4. [ ] Validar precision con conocimiento local

### Semana 3 (Opcional): CV Avanzado
1. [ ] Configurar cuenta Copernicus
2. [ ] Descargar escena Sentinel-2 de prueba
3. [ ] Probar DeepWaterMap/WatNet
4. [ ] Comparar con datos OSM

---

## Metricas de Exito

| Metrica | Objetivo | Metodo de Validacion |
|---------|----------|---------------------|
| Precision costa | <50m error | Comparar con imagen satelital |
| Cobertura batimetria | 100% area | Verificar datos GEBCO |
| Zonas generadas | >5 por especie | Contar zonas output |
| Tiempo procesamiento | <30s | Medir pipeline |

---

## Conclusion

El nuevo enfoque:

1. **Usa datos reales verificados** (OSM, GEBCO) en lugar de deteccion CV imprecisa
2. **Elimina dependencia de bandas espectrales** que no tenemos
3. **Mantiene opcion CV** para usuarios con acceso a Sentinel-2 o GPU
4. **Es mas confiable y reproducible** que algoritmos HSV en RGB

La linea costera de OSM y la batimetria de GEBCO son **datos cientificos reales**, no estimaciones de un algoritmo que puede fallar.
