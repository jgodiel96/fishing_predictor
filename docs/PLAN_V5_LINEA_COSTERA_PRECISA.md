# Plan V5: Línea Costera Precisa con Verificación por Computer Vision

**Fecha:** 2026-02-01
**Versión:** 5.0
**Estado:** PENDIENTE
**Prerequisitos:**
- [Plan V1 - Arquitectura de Datos](PLAN_V1_ARQUITECTURA_DATOS.md) ✅ Completado
- [Plan V2 - Validación Científica](PLAN_V2_VALIDACION_CIENTIFICA.md) ✅ Completado
- [Plan V3 - Implementación Mejoras](PLAN_V3_IMPLEMENTACION_MEJORAS.md) ✅ Completado
- [Plan V4 - Integración de Datos](PLAN_V4_INTEGRACION_DATOS.md) ✅ Completado

---

## Evolución del Proyecto

| Plan | Enfoque | Estado |
|------|---------|--------|
| **V1** | Arquitectura Bronze/Silver/Gold, datos reales 2020-2026 | ✅ Completado |
| **V2** | Estado del Arte, validación científica | ✅ Completado |
| **V3** | Implementación de mejoras técnicas | ✅ Completado |
| **V4** | Integración completa de datos al análisis principal | ✅ Completado |
| **V5** | Línea costera precisa con verificación CV | ⏳ En progreso |

---

## Objetivo

Crear una **línea costera definitiva y permanente** que:
1. Tenga puntos separados por **máximo 50 metros**
2. Sea **verificada** mediante algoritmo de Computer Vision
3. Compare **imagen satelital** vs **mapa simplificado** para validación
4. Una vez validada, **no se modifique** (inmutable)

---

## Problema Actual

| Problema | Impacto |
|----------|---------|
| Línea costera de OSM tiene gaps | Zonas sin cobertura |
| Detección satelital tiene ruido | Puntos incorrectos en tierra |
| No hay verificación cruzada | Errores no detectados |
| Espaciado irregular | Análisis inconsistente |

---

## Arquitectura de la Solución (SAM - Segment Anything Model)

### Hardware Disponible
- **MacBook Pro M3 Pro**
- **18GB RAM**
- **Metal Performance Shaders (MPS)** para aceleración GPU

### Modelo Principal: SAM (Segment Anything Model)
- **Desarrollado por:** Meta AI Research
- **Precisión:** Estado del arte en segmentación
- **Ventaja:** Zero-shot, no requiere entrenamiento
- **RAM requerida:** ~10GB (cabe en 18GB)

```
┌─────────────────────────────────────────────────────────────┐
│                    FASE 1: SEGMENTACIÓN SAM                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                        │
│  │ Imagen Satelital│                                        │
│  │    (ESRI)       │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │         SAM (Segment Anything)          │               │
│  │  ┌─────────────────────────────────┐    │               │
│  │  │   Image Encoder (ViT-H/16)      │    │               │
│  │  │   - 632M parámetros             │    │               │
│  │  │   - Procesa imagen completa     │    │               │
│  │  └─────────────┬───────────────────┘    │               │
│  │                │                        │               │
│  │                ▼                        │               │
│  │  ┌─────────────────────────────────┐    │               │
│  │  │   Prompt: "water body edge"     │    │               │
│  │  │   + puntos semilla en el mar    │    │               │
│  │  └─────────────┬───────────────────┘    │               │
│  │                │                        │               │
│  │                ▼                        │               │
│  │  ┌─────────────────────────────────┐    │               │
│  │  │   Mask Decoder                  │    │               │
│  │  │   - Genera máscara agua/tierra  │    │               │
│  │  └─────────────────────────────────┘    │               │
│  └─────────────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  Máscara Binaria│  (agua=1, tierra=0)                   │
│  │  Alta Precisión │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Extracción de   │                                        │
│  │ Contorno (borde)│  → Puntos candidatos (~10,000)        │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              FASE 2: VERIFICACIÓN DUAL SAM                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ SAM + Satelital  │      │  SAM + Mapa OSM  │            │
│  │                  │      │                  │            │
│  │  Máscara agua A  │      │  Máscara agua B  │            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │                         │                       │
│           └────────────┬────────────┘                       │
│                        ▼                                    │
│           ┌────────────────────────┐                        │
│           │   IoU (Intersection    │                        │
│           │   over Union)          │                        │
│           │                        │                        │
│           │   IoU > 0.85 = VÁLIDO  │                        │
│           └────────────────────────┘                        │
│                        │                                    │
│                        ▼                                    │
│           ┌────────────────────────┐                        │
│           │  Contorno Consensuado  │                        │
│           │  (donde ambos coinciden)│                       │
│           └────────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                FASE 3: REFINAMIENTO GEOMÉTRICO              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Douglas-Peucker simplificación (preservar forma)        │
│                                                             │
│  2. Garantizar espaciado ≤ 50m:                           │
│     ┌─────────────────────────────────────────┐            │
│     │ Para cada par de puntos consecutivos:   │            │
│     │   Si distancia > 50m:                  │            │
│     │     Insertar puntos intermedios         │            │
│     │     usando interpolación cúbica         │            │
│     └─────────────────────────────────────────┘            │
│                                                             │
│  3. Suavizado con Savitzky-Golay filter                    │
│     (elimina ruido, preserva bordes)                       │
│                                                             │
│  4. Ordenar geográficamente (sur → norte)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                FASE 4: VALIDACIÓN Y GUARDADO                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │ VALIDACIONES AUTOMÁTICAS:               │               │
│  │                                         │               │
│  │ ✓ Espaciado máximo ≤ 50m              │               │
│  │ ✓ Espaciado mínimo ≥ 20m               │               │
│  │ ✓ IoU promedio ≥ 85%                   │               │
│  │ ✓ Sin puntos aislados                  │               │
│  │ ✓ Continuidad de la línea              │               │
│  │ ✓ Todos los puntos en zona costera     │               │
│  └─────────────────────────────────────────┘               │
│                        │                                    │
│                        ▼                                    │
│  ┌─────────────────────────────────────────┐               │
│  │ GUARDADO INMUTABLE:                     │               │
│  │                                         │               │
│  │ data/gold/coastline/                    │               │
│  │ ├── coastline_v1.geojson   (datos)     │               │
│  │ ├── coastline_v1.sha256    (checksum)  │               │
│  │ └── coastline_v1_report.json (métricas)│               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Fases de Implementación

### Fase 1: Detección Inicial de Candidatos

| # | Tarea | Archivo | Descripción |
|---|-------|---------|-------------|
| 1.1 | Descargar tiles satelitales | `coastline_detector.py` | ESRI World Imagery, zoom 16 |
| 1.2 | Descargar tiles mapa calle | `coastline_detector.py` | OSM tiles para comparación |
| 1.3 | Detectar agua en satelital | `coastline_detector.py` | HSV segmentation |
| 1.4 | Extraer contornos | `coastline_detector.py` | OpenCV findContours |

### Fase 2: Verificación Cruzada (Computer Vision)

| # | Tarea | Archivo | Descripción |
|---|-------|---------|-------------|
| 2.1 | Crear clase `CoastlineVerifier` | `coastline_verifier.py` | Verificación CV |
| 2.2 | Implementar verificación satelital | `coastline_verifier.py` | Agua a oeste del punto |
| 2.3 | Implementar verificación mapa | `coastline_verifier.py` | Confirmar con mapa calle |
| 2.4 | Scoring de confianza | `coastline_verifier.py` | 0-100% por punto |

### Fase 3: Refinamiento de Línea

| # | Tarea | Archivo | Descripción |
|---|-------|---------|-------------|
| 3.1 | Filtrar puntos baja confianza | `coastline_refiner.py` | Umbral: 80% |
| 3.2 | Interpolar gaps > 50m | `coastline_refiner.py` | Asegurar espaciado máximo |
| 3.3 | Suavizar curva | `coastline_refiner.py` | Eliminar ruido |
| 3.4 | Ordenar geográficamente | `coastline_refiner.py` | Sur a norte |

### Fase 4: Validación y Guardado Permanente

| # | Tarea | Archivo | Descripción |
|---|-------|---------|-------------|
| 4.1 | Validar espaciado ≤ 50m | `coastline_validator.py` | Error si no cumple |
| 4.2 | Generar reporte de calidad | `coastline_validator.py` | Métricas y estadísticas |
| 4.3 | Guardar en Gold layer | `coastline_validator.py` | Inmutable |
| 4.4 | Crear checksum | `coastline_validator.py` | SHA256 para integridad |

---

## Especificaciones Técnicas

### Requisitos de Hardware

```yaml
Hardware Mínimo:
  RAM: 16GB
  GPU: Metal compatible (M1/M2/M3)
  Almacenamiento: 5GB libres

Hardware Usado:
  Modelo: MacBook Pro M3 Pro
  RAM: 18GB
  GPU: Metal Performance Shaders (MPS)
  Aceleración: PyTorch MPS backend
```

### Configuración SAM

```python
SAM_CONFIG = {
    # Modelo
    'model_type': 'vit_h',           # ViT-Huge (mejor precisión)
    'checkpoint': 'sam_vit_h_4b8939.pth',
    'model_size_gb': 2.4,

    # Dispositivo
    'device': 'mps',                  # Metal Performance Shaders
    'dtype': 'float32',               # Precisión completa

    # Imagen
    'image_size': 1024,               # Resolución de entrada SAM
    'tile_size': 512,                 # Tamaño de tile para procesar

    # Prompts para segmentación
    'prompt_points': [                # Puntos semilla en el mar
        {'type': 'point', 'coords': 'auto_detect_water'},
    ],
    'multimask_output': False,        # Una máscara por imagen
}
```

### Parámetros de Segmentación

```python
SEGMENTATION_CONFIG = {
    # Tiles satelitales
    'zoom_level': 16,                 # ~2.4m/pixel
    'tile_source': 'esri_worldimagery',

    # Post-procesamiento de máscara
    'mask_threshold': 0.5,            # Umbral de confianza SAM
    'min_area_pixels': 500,           # Área mínima para considerar
    'morphology_kernel': 5,           # Kernel para limpieza

    # Extracción de contorno
    'contour_method': 'CHAIN_APPROX_NONE',  # Máxima precisión
    'simplify_epsilon': 0.5,          # Douglas-Peucker epsilon
}
```

### Parámetros de Verificación Dual

```python
VERIFICATION_CONFIG = {
    # Comparación de máscaras
    'iou_threshold': 0.85,            # 85% IoU mínimo
    'pixel_agreement_threshold': 0.90, # 90% acuerdo

    # Fuentes de verificación
    'sources': ['esri_satellite', 'osm_standard'],

    # Validación de dirección
    'sea_direction': 'west',          # Océano al oeste
    'direction_check_distance_m': 100, # Verificar 100m hacia el mar
}
```

### Parámetros de Refinamiento

```python
REFINEMENT_CONFIG = {
    # Espaciado
    'max_spacing_m': 50,              # MÁXIMO 50 metros
    'min_spacing_m': 50,              # Mínimo 50 metros
    'target_spacing_m': 150,          # Objetivo ideal

    # Interpolación
    'interpolation_method': 'cubic',  # Spline cúbico
    'preserve_corners': True,         # Mantener esquinas naturales

    # Suavizado
    'smoothing_method': 'savgol',     # Savitzky-Golay
    'smoothing_window': 7,            # Ventana de 7 puntos
    'smoothing_order': 3,             # Orden polinomial
}
```

### Dependencias Python

```python
DEPENDENCIES = {
    # SAM
    'segment-anything': '1.0',
    'torch': '>=2.0',                 # Con soporte MPS

    # Procesamiento de imágenes
    'opencv-python': '>=4.8',
    'pillow': '>=10.0',
    'numpy': '>=1.24',

    # Geoespacial
    'shapely': '>=2.0',
    'geojson': '>=3.0',

    # Científico
    'scipy': '>=1.11',                # Para Savitzky-Golay
}
```

---

## Estructura de Archivos

```
core/
├── coastline_detector.py      # Fase 1: Detección HSV (existe)
├── coastline_sam.py           # Fase 1: Detección SAM (nuevo)
├── coastline_verifier.py      # Fase 2: Verificación dual (nuevo)
├── coastline_refiner.py       # Fase 3: Refinamiento 50m (nuevo)
├── coastline_validator.py     # Fase 4: Validación (nuevo)
└── coastline_pipeline.py      # Orquestador completo (nuevo)

models/
└── sam/
    ├── sam_vit_h_4b8939.pth   # Checkpoint SAM ViT-H (~2.4GB)
    └── README.md              # Instrucciones de descarga

data/
├── cache/
│   └── tiles/                 # Tiles descargados
│       ├── satellite/         # ESRI World Imagery
│       └── street/            # OSM Standard
└── gold/
    └── coastline/
        ├── coastline_v1.geojson      # Versión inmutable
        ├── coastline_v1.sha256       # Checksum
        └── coastline_v1_report.json  # Métricas de calidad
```

---

## Algoritmo Principal: SAM Segmentation

```python
class SAMCoastlineDetector:
    """
    Detector de línea costera usando Segment Anything Model (SAM).

    SAM es un modelo de segmentación zero-shot desarrollado por Meta AI
    que puede segmentar cualquier objeto con alta precisión.
    """

    def __init__(self):
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        # Usar MPS (Metal) en Mac M3
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Cargar modelo SAM ViT-H (el más preciso)
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

    def segment_water(self, image: np.ndarray) -> np.ndarray:
        """
        Segmenta el agua en una imagen satelital.

        Args:
            image: Imagen RGB (H, W, 3)

        Returns:
            Máscara binaria (H, W) donde 1=agua, 0=tierra
        """
        # 1. Preparar imagen para SAM
        self.predictor.set_image(image)

        # 2. Detectar puntos semilla automáticamente
        #    (buscar regiones azules para identificar el mar)
        seed_points = self._detect_water_seeds(image)

        # 3. Generar máscara usando SAM
        masks, scores, _ = self.predictor.predict(
            point_coords=seed_points,
            point_labels=np.ones(len(seed_points)),  # Todos positivos (agua)
            multimask_output=False
        )

        # 4. Post-procesar máscara
        water_mask = masks[0].astype(np.uint8)
        water_mask = self._clean_mask(water_mask)

        return water_mask

    def _detect_water_seeds(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta puntos semilla en el agua usando análisis de color.
        Estos puntos ayudan a SAM a entender qué segmentar.
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Detectar azul (agua)
        water_mask = cv2.inRange(hsv, (90, 30, 30), (130, 255, 255))

        # Encontrar centroide de la región de agua más grande
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: usar esquina oeste (generalmente mar en Perú)
            h, w = image.shape[:2]
            return np.array([[w // 4, h // 2]])

        # Usar centroide del contorno más grande
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return np.array([[cx, cy]])

        return np.array([[image.shape[1] // 4, image.shape[0] // 2]])

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Limpia la máscara con operaciones morfológicas."""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def extract_coastline(self, water_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extrae la línea costera como lista de puntos (x, y).

        La línea costera es el borde entre agua (1) y tierra (0).
        """
        # Encontrar contornos
        contours, _ = cv2.findContours(
            water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return []

        # Tomar el contorno más largo (la costa principal)
        coastline_contour = max(contours, key=len)

        # Convertir a lista de puntos
        points = [(int(p[0][0]), int(p[0][1])) for p in coastline_contour]

        return points


def verify_coastline_dual_source(
    coastline_points: List[Tuple[float, float]],
    detector: SAMCoastlineDetector
) -> List[Dict]:
    """
    Verifica cada punto de la línea costera usando dos fuentes:
    1. Imagen satelital (ESRI)
    2. Mapa de calles (OSM)

    Calcula IoU (Intersection over Union) entre las dos máscaras.
    """
    verified_points = []

    for lat, lon in coastline_points:
        # Obtener tile satelital
        sat_image = fetch_satellite_tile(lat, lon, zoom=16)
        sat_mask = detector.segment_water(sat_image)

        # Obtener tile mapa calle
        street_image = fetch_street_tile(lat, lon, zoom=16)
        street_mask = detector.segment_water(street_image)

        # Calcular IoU
        intersection = np.logical_and(sat_mask, street_mask).sum()
        union = np.logical_or(sat_mask, street_mask).sum()
        iou = intersection / union if union > 0 else 0

        # Verificar que el punto está en el borde
        pixel_x, pixel_y = latlon_to_pixel(lat, lon, tile_bounds)
        is_on_edge = is_point_on_mask_edge(sat_mask, pixel_x, pixel_y, radius=5)

        verified_points.append({
            'lat': lat,
            'lon': lon,
            'iou': iou,
            'is_valid': iou >= 0.85 and is_on_edge,
            'confidence': iou * (0.9 if is_on_edge else 0.5)
        })

    return verified_points
```

### Algoritmo de Refinamiento con Espaciado ≤ 50m

```python
def ensure_max_spacing(
    coastline: List[Tuple[float, float]],
    max_spacing_m: float = 50
) -> List[Tuple[float, float]]:
    """
    Garantiza que ningún par de puntos consecutivos tenga más de max_spacing_m.

    Usa interpolación cúbica para insertar puntos intermedios naturales.
    """
    from scipy.interpolate import CubicSpline

    refined = [coastline[0]]

    for i in range(1, len(coastline)):
        p1 = refined[-1]
        p2 = coastline[i]

        # Calcular distancia
        dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])

        if dist <= max_spacing_m:
            refined.append(p2)
        else:
            # Necesitamos interpolar
            num_segments = int(np.ceil(dist / max_spacing_m))

            # Crear spline entre los dos puntos
            t = np.array([0, 1])
            lats = np.array([p1[0], p2[0]])
            lons = np.array([p1[1], p2[1]])

            # Interpolar
            t_new = np.linspace(0, 1, num_segments + 1)[1:]  # Excluir p1

            for t_val in t_new:
                new_lat = p1[0] + t_val * (p2[0] - p1[0])
                new_lon = p1[1] + t_val * (p2[1] - p1[1])
                refined.append((new_lat, new_lon))

    return refined
```

---

## Validación Final

### Criterios de Aceptación

| Criterio | Valor | Obligatorio |
|----------|-------|-------------|
| Espaciado máximo entre puntos | ≤ 50m | ✅ Sí |
| Espaciado mínimo entre puntos | ≥ 20m | ✅ Sí |
| Confianza promedio de puntos | ≥ 85% | ✅ Sí |
| Puntos con confianza < 70% | ≤ 5% | ✅ Sí |
| Puntos en tierra (falsos) | 0% | ✅ Sí |
| Cobertura de región objetivo | ≥ 95% | ✅ Sí |

### Reporte de Calidad

```json
{
  "version": "v1",
  "created_at": "2026-02-01T12:00:00Z",
  "checksum": "sha256:abc123...",
  "statistics": {
    "total_points": 1800,
    "total_length_km": 90.5,
    "avg_spacing_m": 45,
    "max_spacing_m": 49,
    "min_spacing_m": 22,
    "avg_confidence": 0.91,
    "points_below_80_confidence": 42,
    "coverage_percent": 98.5
  },
  "validation": {
    "spacing_check": "PASS",
    "confidence_check": "PASS",
    "coverage_check": "PASS",
    "overall": "PASS"
  },
  "region": {
    "lat_min": -18.35,
    "lat_max": -17.30,
    "lon_min": -71.50,
    "lon_max": -70.10
  }
}
```

---

## Comandos de Ejecución

```bash
# Ejecutar pipeline completo
python core/coastline_pipeline.py --region full --output data/gold/coastline/

# Solo detección (Fase 1)
python core/coastline_detector.py --zoom 16 --spacing 100

# Solo verificación (Fase 2)
python core/coastline_verifier.py --input detected.geojson

# Solo refinamiento (Fase 3)
python core/coastline_refiner.py --max-spacing 50

# Solo validación (Fase 4)
python core/coastline_validator.py --input refined.geojson
```

---

## Cronograma

| Fase | Tareas | Estimación |
|------|--------|------------|
| 1 | Detección inicial | ✅ Completado |
| 2 | Verificación CV | Pendiente |
| 3 | Refinamiento | Pendiente |
| 4 | Validación y guardado | Pendiente |

---

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Tiles satelitales no disponibles | Cache local + fallback a OSM |
| Nubes en imagen satelital | Usar múltiples fechas |
| Mapa calle incompleto | Priorizar verificación satelital |
| Espaciado > 50m en algunos tramos | Interpolación automática |

---

## Resultado Esperado

```
data/gold/coastline/coastline_v1.geojson
├── ~1800 puntos
├── Espaciado máximo: 50m
├── Espaciado promedio: ~45m
├── Cobertura: Tacna-Ilo-Sama completo
├── Verificado por CV: Sí
└── Inmutable: Sí (con checksum)
```

---

*Plan V5 creado: 2026-02-01*
*Proyecto: Fishing Predictor - Tacna/Ilo/Sama, Peru*
