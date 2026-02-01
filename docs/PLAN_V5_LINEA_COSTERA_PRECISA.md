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
1. Tenga puntos separados por **máximo 200 metros**
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

## Arquitectura de la Solución

```
┌─────────────────────────────────────────────────────────────┐
│                    FASE 1: DETECCIÓN                        │
├─────────────────────────────────────────────────────────────┤
│  Imagen Satelital (ESRI)  ──►  Detección de Agua (HSV)     │
│                                      │                      │
│                                      ▼                      │
│                              Contornos OpenCV               │
│                                      │                      │
│                                      ▼                      │
│                         Puntos Candidatos (~5000)           │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE 2: VERIFICACIÓN                     │
├─────────────────────────────────────────────────────────────┤
│  Para cada punto candidato:                                 │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Satelital  │      │  Mapa Calle  │                    │
│  │   (pixel)    │      │   (pixel)    │                    │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                     │                             │
│         ▼                     ▼                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ ¿Es agua a   │      │ ¿Es agua a   │                    │
│  │ la izquierda?│      │ la izquierda?│                    │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                     │                             │
│         └─────────┬───────────┘                             │
│                   ▼                                         │
│         ┌─────────────────┐                                 │
│         │ Ambos coinciden │──► PUNTO VÁLIDO                │
│         │ (agua oeste,    │                                 │
│         │  tierra este)   │                                 │
│         └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE 3: REFINAMIENTO                     │
├─────────────────────────────────────────────────────────────┤
│  1. Filtrar puntos que no pasaron verificación              │
│  2. Interpolar para garantizar espaciado ≤ 200m            │
│  3. Suavizar curva (Savitzky-Golay o spline)               │
│  4. Ordenar de sur a norte                                  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE 4: VALIDACIÓN FINAL                 │
├─────────────────────────────────────────────────────────────┤
│  1. Verificar espaciado máximo 200m ✓                       │
│  2. Verificar que todos los puntos están en la costa ✓      │
│  3. Generar reporte de calidad                              │
│  4. Guardar como INMUTABLE en data/gold/                    │
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
| 3.2 | Interpolar gaps > 200m | `coastline_refiner.py` | Asegurar espaciado máximo |
| 3.3 | Suavizar curva | `coastline_refiner.py` | Eliminar ruido |
| 3.4 | Ordenar geográficamente | `coastline_refiner.py` | Sur a norte |

### Fase 4: Validación y Guardado Permanente

| # | Tarea | Archivo | Descripción |
|---|-------|---------|-------------|
| 4.1 | Validar espaciado ≤ 200m | `coastline_validator.py` | Error si no cumple |
| 4.2 | Generar reporte de calidad | `coastline_validator.py` | Métricas y estadísticas |
| 4.3 | Guardar en Gold layer | `coastline_validator.py` | Inmutable |
| 4.4 | Crear checksum | `coastline_validator.py` | SHA256 para integridad |

---

## Especificaciones Técnicas

### Parámetros de Detección

```python
DETECTION_CONFIG = {
    'zoom_level': 16,           # Alta resolución
    'tile_size': 256,           # Pixeles por tile
    'meters_per_pixel': 2.4,    # A zoom 16, ~2.4m/pixel

    # HSV ranges para agua
    'water_hsv_low': [90, 30, 30],
    'water_hsv_high': [130, 255, 255],

    # Parámetros de contorno
    'contour_method': cv2.CHAIN_APPROX_NONE,
    'min_contour_area': 1000,   # Pixeles mínimos
}
```

### Parámetros de Verificación

```python
VERIFICATION_CONFIG = {
    'check_radius_pixels': 10,   # Radio de verificación
    'water_threshold': 0.6,      # 60% pixels deben ser agua
    'confidence_threshold': 0.8, # 80% mínimo para aceptar

    # Dirección al mar (Perú costa oeste)
    'sea_direction': 'west',     # El mar está al oeste
    'direction_tolerance': 45,   # Grados de tolerancia
}
```

### Parámetros de Refinamiento

```python
REFINEMENT_CONFIG = {
    'max_spacing_m': 200,        # Espaciado máximo
    'min_spacing_m': 50,         # Espaciado mínimo
    'smoothing_window': 5,       # Ventana de suavizado
    'interpolation_method': 'linear',
}
```

---

## Estructura de Archivos

```
core/
├── coastline_detector.py      # Fase 1: Detección (existe)
├── coastline_verifier.py      # Fase 2: Verificación CV (nuevo)
├── coastline_refiner.py       # Fase 3: Refinamiento (nuevo)
├── coastline_validator.py     # Fase 4: Validación (nuevo)
└── coastline_pipeline.py      # Orquestador completo (nuevo)

data/
├── cache/
│   └── tiles/                 # Tiles descargados
│       ├── satellite/
│       └── street/
└── gold/
    └── coastline/
        ├── coastline_v1.geojson      # Versión inmutable
        ├── coastline_v1.checksum     # SHA256
        └── coastline_v1_report.json  # Reporte de calidad
```

---

## Algoritmo de Verificación CV

```python
def verify_point(lat, lon, satellite_img, street_img):
    """
    Verifica si un punto está en la línea costera.

    Criterio:
    - A la izquierda (oeste) del punto debe haber AGUA
    - A la derecha (este) del punto debe haber TIERRA
    - Esto debe cumplirse en AMBAS imágenes
    """

    # 1. Obtener pixel del punto en ambas imágenes
    px_sat, py_sat = latlon_to_pixel(lat, lon, satellite_img)
    px_str, py_str = latlon_to_pixel(lat, lon, street_img)

    # 2. Analizar región oeste (hacia el mar)
    west_region_sat = satellite_img[py_sat-5:py_sat+5, px_sat-20:px_sat]
    west_region_str = street_img[py_str-5:py_str+5, px_str-20:px_str]

    # 3. Analizar región este (hacia tierra)
    east_region_sat = satellite_img[py_sat-5:py_sat+5, px_sat:px_sat+20]
    east_region_str = street_img[py_str-5:py_str+5, px_str:px_str+20]

    # 4. Calcular porcentaje de agua en cada región
    water_west_sat = calculate_water_percentage(west_region_sat)
    water_west_str = calculate_water_percentage(west_region_str)
    water_east_sat = calculate_water_percentage(east_region_sat)
    water_east_str = calculate_water_percentage(east_region_str)

    # 5. Criterio de validación
    # Oeste debe ser >60% agua, Este debe ser <40% agua
    valid_sat = (water_west_sat > 0.6) and (water_east_sat < 0.4)
    valid_str = (water_west_str > 0.6) and (water_east_str < 0.4)

    # 6. Score de confianza
    confidence = (
        (water_west_sat + water_west_str) / 2 * 0.5 +
        (1 - water_east_sat + 1 - water_east_str) / 2 * 0.5
    )

    return {
        'valid': valid_sat and valid_str,
        'confidence': confidence,
        'details': {
            'water_west_satellite': water_west_sat,
            'water_west_street': water_west_str,
            'water_east_satellite': water_east_sat,
            'water_east_street': water_east_str,
        }
    }
```

---

## Validación Final

### Criterios de Aceptación

| Criterio | Valor | Obligatorio |
|----------|-------|-------------|
| Espaciado máximo entre puntos | ≤ 200m | ✅ Sí |
| Espaciado mínimo entre puntos | ≥ 50m | ✅ Sí |
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
    "total_points": 4500,
    "total_length_km": 90.5,
    "avg_spacing_m": 180,
    "max_spacing_m": 198,
    "min_spacing_m": 52,
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
python core/coastline_refiner.py --max-spacing 200

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
| Espaciado > 200m en algunos tramos | Interpolación automática |

---

## Resultado Esperado

```
data/gold/coastline/coastline_v1.geojson
├── ~4500 puntos
├── Espaciado máximo: 200m
├── Espaciado promedio: ~180m
├── Cobertura: Tacna-Ilo-Sama completo
├── Verificado por CV: Sí
└── Inmutable: Sí (con checksum)
```

---

*Plan V5 creado: 2026-02-01*
*Proyecto: Fishing Predictor - Tacna/Ilo/Sama, Peru*
