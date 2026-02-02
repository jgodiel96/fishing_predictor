# Progreso: Corrección de Visualización de Costa

**Fecha**: 2026-02-01
**Estado**: EN PROGRESO

---

## Resumen del Problema

El mapa muestra errores visuales:
1. Líneas de costa que cruzan tierra (conexiones incorrectas)
2. Flechas de corriente apuntando hacia tierra en vez del océano
3. Segmento de Canepa aparece como línea vertical en vez de seguir la costa

---

## Diagnóstico Realizado

### Problema Original (coastline_v1.geojson)
- **Línea 1 (OSM)**: 6777 puntos con 16 saltos > 1km (máximo 138.5 km)
- **Línea 2 (Canepa)**: 1543 puntos con 122 saltos > 1km

### Corrección Parcial (coastline_v2.geojson)
Se dividió en 47 segmentos continuos (máx 500m entre puntos):
- 21 segmentos de OSM (costa principal)
- 26 segmentos de Canepa

**Resultado**: La costa OSM se ve correcta, pero los segmentos de Canepa son pequeños clusters de puntos (~50m cada uno) dispersos verticalmente, NO una línea costera continua.

### Problema Identificado: Datos de Canepa Corruptos

Los 26 segmentos de Canepa (segmentos 22-47) son:
- Clusters muy pequeños (18-145 puntos cada uno)
- Rango de longitud: ~0.001° (100m)
- Rango de latitud: ~0.001° (100m)
- Dispersos en una línea vertical de -18.09 a -17.96 lat

**Conclusión**: Los datos de Canepa no representan una línea costera real. Son puntos aislados que forman una línea vertical, probablemente generados incorrectamente.

---

## Cambios Realizados

### Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `config.py:105` | Cambiado a `coastline_v2.geojson` |
| `controllers/analysis.py:157-237` | `sample_fishing_spots()` - usa segmentos |
| `controllers/analysis.py:239-310` | `_add_focus_zone_spots()` - usa segmentos |
| `controllers/analysis.py:312-369` | `fetch_marine_data()` - usa segmentos |
| `controllers/analysis.py:871-912` | Agregado `_perpendicular_to_sea_segment()` |
| `views/map_view.py:90-119` | `add_coastline()` - dibuja segmentos separados |
| `models/features.py:450-456` | Fallback para wind_speed None |

### Archivos Creados

| Archivo | Descripción |
|---------|-------------|
| `data/gold/coastline/coastline_v2.geojson` | Costa limpia (47 segmentos) |
| `data/gold/coastline/coastline_v2.sha256` | Checksum |

---

## Próximos Pasos (PENDIENTE)

### Opción A: Eliminar Segmentos de Canepa
Usar solo la costa OSM (segmentos 1-21) que está correcta.

```python
# Filtrar solo segmentos OSM (lon < -70.35)
segments = [s for s in segments if avg_lon(s) < -70.35]
```

### Opción B: Regenerar Costa de Canepa
Si Playa Canepa es importante, regenerar los datos desde:
1. OSM Overpass API para esa zona específica
2. O usar interpolación desde imágenes satelitales

### Opción C: Descargar costa completa desde OSM
Descargar toda la costa Tacna-Ilo-Canepa desde OSM en una sola consulta.

---

## Comandos Útiles

```bash
# Ejecutar análisis
python main.py

# Abrir mapa
open output/fishing_analysis_ml.html

# Analizar estructura del GeoJSON
python3 -c "
import json
with open('data/gold/coastline/coastline_v2.geojson') as f:
    data = json.load(f)
print(f'Segmentos: {len(data[\"features\"][0][\"geometry\"][\"coordinates\"])}')
"
```

---

## Archivos Clave

```
fishing_predictor/
├── config.py                          # COASTLINE_FILE = coastline_v2
├── controllers/analysis.py            # Métodos actualizados
├── views/map_view.py                  # add_coastline con segmentos
├── data/gold/coastline/
│   ├── coastline_v1.geojson          # Original (corrupto)
│   ├── coastline_v2.geojson          # Limpio (Canepa aún malo)
│   └── coastline_v2.sha256
└── docs/
    └── PROGRESO_COASTLINE_FIX.md     # Este archivo
```

---

## Decisión Pendiente

**Pregunta**: ¿Eliminar los datos de Canepa (opción A) o regenerarlos correctamente (opción B/C)?

La costa OSM (Ilo a Tacna) funciona correctamente. El problema está solo en los datos de Playa Canepa.
