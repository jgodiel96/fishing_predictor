# Progreso: Corrección de Visualización de Costa

**Fecha**: 2026-02-02
**Estado**: ✅ COMPLETADO (v5.1)

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

## Solución Implementada (v5.1) - 2026-02-02

### Problema Real Identificado

El problema **no era solo Canepa**, sino el **algoritmo de conexión de puntos**. El código anterior usaba:
```python
points = sorted(set(points), key=lambda p: (p[0], p[1]))  # MALO
```

Esto creaba conexiones falsas entre puntos lejanos porque ordenaba por latitud sin considerar la proximidad geográfica real.

### Solución: Algoritmo de Conexión Inteligente

Se implementó `CoastlineConnector` en `core/coastline_connector.py` con:

| Componente | Descripción |
|------------|-------------|
| `build_nearest_neighbor_chain()` | Construye cadena por vecino más cercano |
| `detect_segments()` | Detecta gaps > 500m para separar segmentos |
| `merge_segments()` | Une segmentos por extremos cercanos (< 1km) |
| `remove_isolated_points()` | Elimina puntos de ruido |

### Resultados

| Métrica | Antes (v2) | Después (v3) |
|---------|------------|--------------|
| Puntos | 8320 | 8300 |
| Segmentos | 47 (muchos corruptos) | 34 (todos válidos) |
| Conexiones falsas | Sí | No |
| Longitud total | N/A | 248.48 km |

### Archivos Modificados/Creados

| Archivo | Acción |
|---------|--------|
| `core/coastline_connector.py` | **NUEVO** - Algoritmo de conexión |
| `core/coastline_sam.py` | Integración del conector |
| `core/coastline_pipeline.py` | Fase 1.5 de conexión |
| `core/coastline_validator.py` | Soporte para segmentos |
| `config.py` | Actualizado a coastline_v3 |
| `docs/PLAN_V5_LINEA_COSTERA_PRECISA.md` | Documentación v5.1 |

### Archivos de Coastline

```
data/gold/coastline/
├── coastline_v1.geojson     # Original (conexiones falsas)
├── coastline_v2.geojson     # Intento manual (aún problemas)
├── coastline_v3.geojson     # ✅ ACTUAL - Algoritmo inteligente
└── coastline_v3.sha256      # Checksum
```

### Verificación

El análisis se ejecuta correctamente:
- 2974 puntos únicos cargados (después de deduplicación)
- 34 segmentos continuos
- Mapa generado sin líneas cruzando tierra
