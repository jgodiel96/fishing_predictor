# Plan V6: Próximos Pasos y Lineamientos

**Fecha:** 2026-02-05
**Versión:** 6.1
**Estado:** V6 IMPLEMENTADO
**Prerequisitos:** V1-V5 Completados

---

## Resumen del Estado Actual

| Componente | Estado | Completitud |
|------------|--------|-------------|
| Arquitectura Bronze/Silver/Gold | ✅ | 100% |
| Datos reales 2020-2026 | ✅ | 100% |
| 32 Features ML | ✅ | 100% |
| Mareas astronómicas | ✅ | 100% |
| SSS/SLA de Copernicus | ✅ | 100% |
| Línea costera (v8) | ✅ | 100% |
| 18 Hotspots verificados | ✅ | 100% |
| Búsqueda por proximidad | ✅ | 100% |
| **Selector de fecha dinámico (7 días)** | ✅ | 100% |
| **Score Unificado V6 (ML + Mareas + Hora)** | ✅ | 100% |
| **Sistema de validación CPUE** | ❌ | 0% |
| **API REST** | ❌ | 0% |
| **Clorofila-a** | ❌ | 0% |

**Completitud general: ~91%**

---

## Problema Identificado: Selector de Fecha Estático

### Descripción
Cuando el usuario cambia la fecha en el mapa HTML, aparece un mensaje pidiendo re-ejecutar el programa. Esto ocurre porque el mapa es HTML estático sin backend.

### Ubicación del Código
```
views/map_view.py:597-602
```

```javascript
function onDateChange(date) {
    console.log('Selected date:', date);
    // Map is static HTML - need to regenerate for new date
    alert('📅 Fecha seleccionada: ' + date + '\n\n' +
          'Para ver predicciones de esta fecha, ejecuta:\n\n' +
          'python main.py --date ' + date);
}
```

### Soluciones Propuestas

| Opción | Complejidad | Descripción |
|--------|-------------|-------------|
| **A: Pre-generar multi-día** | Media | Generar predicciones para 7 días y embeber en HTML |
| **B: API REST + Fetch** | Alta | Backend Flask/FastAPI que responde a peticiones |
| **C: PWA con Service Worker** | Alta | App progresiva con cache offline |

**Recomendación:** Opción A para MVP, migrar a B para producción.

### Solución Implementada (2026-02-05)

Se implementó la **Opción A: Pre-generar 7 días** con los siguientes cambios:

| Archivo | Cambio |
|---------|--------|
| `scripts/generate_hourly_predictions.py` | Agregado método `generate_multiday()` |
| `views/map_view.py` | Agregado `add_multiday_hourly_data()` y JS dinámico |
| `controllers/analysis.py` | Integración de predicciones horarias 7 días |

**Cómo funciona:**
1. Al ejecutar `python main.py`, se generan predicciones horarias para 7 días
2. Los datos se embeben en el HTML como JSON
3. El selector de fecha lee los datos embebidos sin necesidad de servidor
4. Los gráficos y tablas se actualizan dinámicamente en el navegador

**Limitación:** Solo funciona para las fechas pre-generadas (7 días desde la fecha de ejecución).

---

## Problema Identificado: Inconsistencia de Scores

### Descripción del Problema

El sistema tiene **dos sistemas de scoring separados** que no están integrados:

| Sistema | Qué calcula | Features usados | Score ejemplo |
|---------|-------------|-----------------|---------------|
| **Score ML (mapa)** | Calidad del PUNTO | 32 features (frentes térmicos, corrientes, upwelling, SST, distancia costa, hotspots) | 31/100 |
| **Score Horario (panel)** | Mejor HORA para una ubicación de referencia | Mareas + hora del día + SST + SSS + SLA | 65/100 |

**Problema real:** Un pescador ve un punto con score 80 en el mapa, pero si va a mediodía con marea muerta, no pescará nada. El score ML no considera la hora ni las mareas.

### Análisis de Opciones

#### Opción A: Predicción horaria POR PUNTO
- **Qué hace:** Muestra dos scores SEPARADOS para cada punto
- **Problema:** ¿Cuál sigues? Si score ML = 80 pero score horario = 40, ¿es bueno o malo?
- **Veredicto:** Parche visual, no solución real

#### Opción B: Unificar el Scoring ✅ SELECCIONADA
- **Qué hace:** UN SOLO score que considera TODO
- **Fórmula:**
  ```
  Score Final = ML(frentes, corrientes, upwelling, SST, hotspots...)
              + Mareas(fase, altura, horas a pleamar/bajamar)
              + Hora(alba/ocaso bonus)
  ```
- **Ventaja:** El número que ves ES la realidad completa
- **Resultado:** "Punta Coles a las 6am = 78/100" significa TODO alineado

### Implementación de Opción B (2026-02-05) ✅ COMPLETADA

**Objetivo:** Integrar mareas y hora del día en el scoring ML para que cada punto tenga un score que refleje TODAS las condiciones.

**Archivos modificados:**

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `controllers/analysis.py` | `_get_hourly_scores()` - calcula tide_score y hour_score | ✅ |
| `controllers/analysis.py` | `analyze_spots(target_hour)` - scoring unificado | ✅ |
| `controllers/analysis.py` | `analyze_spots_all_hours()` - pre-calcula 24 horas | ✅ |
| `views/map_view.py` | `add_hourly_spots_data()` - panel de scoring unificado | ✅ |

**Fórmula del Score Unificado V6:**

```python
# Mareas: ±10 puntos según fase
tide_bonus = (tide_score - 0.5) * 20  # -10 a +10

# Hora: ±12 puntos (alba/ocaso = excelente)
hour_bonus = (hour_score - 0.5) * 24  # -12 a +12

# SSS: ±5 puntos por salinidad
sss_bonus = (sss_score - 0.5) * 10  # -5 a +5

# SLA: ±3 puntos por anomalía nivel mar
sla_bonus = (sla_score - 0.5) * 6   # -3 a +3

# Score final
total_bonus = tide_bonus + hour_bonus + sss_bonus + sla_bonus
unified_score = base_ml_score + total_bonus
```

**Resultado obtenido:**
- ✅ El mapa muestra scores que CAMBIAN según la hora seleccionada
- ✅ Un punto puede tener score 45 a mediodía pero 78 a las 6am
- ✅ El usuario ve la realidad: "Ve a este punto A ESTA HORA"
- ✅ Panel "Score Unificado V6" en esquina inferior izquierda
- ✅ Slider de hora (00:00 - 23:00) actualiza ranking de spots
- ✅ Botones rápidos para alba (06:00, 07:00) y ocaso (17:00, 18:00)

**Cómo probar:**
```bash
# Ejecutar análisis
python main.py

# Abrir mapa
open output/fishing_analysis_ml.html

# En el navegador:
# - Ver panel verde "Score Unificado V6" (esquina inferior izquierda)
# - Mover slider de hora para ver cómo cambian los scores
# - Hacer clic en un spot de la tabla para centrar el mapa
```

---

## Tareas Priorizadas

### Prioridad ALTA (Críticas para validación)

#### Tarea 1: Sistema de Validación CPUE
**Objetivo:** Medir accuracy real del sistema comparando predicciones vs capturas.

**Lineamientos:**
```bash
# Estructura de datos de captura
{
    "date": "2026-02-03",
    "hour": 17,
    "lat": -17.702,
    "lon": -71.332,
    "species": "Cabrilla",
    "weight_kg": 2.5,
    "effort_hours": 3,
    "fisher_name": "opcional"
}
```

**Archivos a crear:**
| Archivo | Propósito |
|---------|-----------|
| `data/captures/captures.json` | Almacén de capturas reportadas |
| `models/validator.py` | Clase `CPUEValidator` |
| `scripts/validate_predictions.py` | Script de validación |

**Métricas objetivo:**
- **Hit Rate:** % donde CPUE_PFZ > CPUE_noPFZ → Objetivo: >60%
- **CPUE Ratio:** Mean(CPUE_PFZ) / Mean(CPUE_noPFZ) → Objetivo: >1.5x
- **Correlación:** Pearson(score_predicho, CPUE_real) → Objetivo: >0.5

**Comando esperado:**
```bash
# Agregar captura
python scripts/add_capture.py --date 2026-02-03 --hour 17 --lat -17.702 --lon -71.332 --species Cabrilla --kg 2.5 --hours 3

# Validar predicciones
python scripts/validate_predictions.py --period 30d
```

**Dependencia:** Requiere recolectar datos de pescadores (mínimo 100 registros).

---

#### Tarea 2: Selector de Fecha Dinámico (Multi-día) ✅ COMPLETADA
**Objetivo:** El HTML muestra predicciones para 7 días sin re-ejecutar.

**Estado:** Implementado el 2026-02-05

**Archivos modificados:**
| Archivo | Cambio |
|---------|--------|
| `scripts/generate_hourly_predictions.py` | Agregado `generate_multiday()` y `get_multiday_summary()` |
| `views/map_view.py` | Agregado `add_multiday_hourly_data()`, JS `updateHourlyPanelForDate()` |
| `controllers/analysis.py` | Import y llamada a `HourlyPredictionGenerator.generate_multiday()` |

**Cómo probar:**
```bash
# Ejecutar análisis
python main.py

# Abrir mapa
open output/fishing_analysis_ml.html

# En el navegador:
# - Usar la barra de 7 días en la parte superior
# - O usar el selector de fecha en el panel derecho
# - Los datos se actualizan dinámicamente
```

---

#### Tarea 3: Recolectar Datos de Pescadores
**Objetivo:** Obtener mínimo 100 registros de capturas reales.

**Lineamientos:**

1. **Crear formulario simple** (Google Forms o similar):
   - Fecha y hora
   - Ubicación (seleccionar de hotspots o GPS)
   - Especie capturada
   - Peso total (kg)
   - Horas de esfuerzo
   - Condiciones observadas (opcional)

2. **Contactar pescadores locales:**
   - Asociaciones de pescadores Tacna/Ilo
   - Grupos de Facebook/WhatsApp de pesca
   - Tiendas de artículos de pesca

3. **Formato de exportación:**
   ```csv
   date,hour,lat,lon,species,weight_kg,effort_hours
   2026-02-03,17,-17.702,-71.332,Cabrilla,2.5,3
   ```

**Script de importación:**
```bash
python scripts/import_captures.py --file captures.csv
```

---

### Prioridad MEDIA

#### Tarea 4: Agregar Clorofila-a de Copernicus
**Objetivo:** Indicador de productividad primaria para mejorar predicciones.

**Lineamientos:**

**Dataset:** `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D`
**Variable:** `CHL` (mg/m³)

**Archivos a crear/modificar:**
| Archivo | Cambio |
|---------|--------|
| `data/fetchers/copernicus_chlorophyll_fetcher.py` | **NUEVO** - Descarga Chl-a |
| `data/data_config.py` | Agregar `RAW_CHLA_COPERNICUS` |
| `controllers/analysis.py` | Integrar `chla_score` |
| `models/features.py` | Feature `chlorophyll_a` |

**Scoring propuesto:**
```python
def calculate_chla_score(chla: float) -> float:
    """
    Clorofila-a scoring para pesca.
    Rango óptimo: 2-10 mg/m³ (alta productividad)
    """
    if chla is None:
        return 0.5  # Neutral fallback

    if 2.0 <= chla <= 10.0:
        return 0.9  # Óptimo
    elif 1.0 <= chla < 2.0 or 10.0 < chla <= 15.0:
        return 0.7  # Bueno
    elif 0.5 <= chla < 1.0:
        return 0.5  # Moderado
    else:
        return 0.3  # Bajo
```

**Comando:**
```bash
# Descargar Chl-a
python data/fetchers/copernicus_chlorophyll_fetcher.py --start 2024-01 --end 2026-02 -v
```

---

#### Tarea 5: API REST para Predicciones
**Objetivo:** Servicio web consumible desde frontend o app móvil.

**Lineamientos:**

**Framework:** FastAPI (ligero, async, auto-documentación)

**Endpoints propuestos:**
```
GET  /api/v1/prediction?lat=-17.8&lon=-71.2&date=2026-02-03
GET  /api/v1/prediction/hourly?lat=-17.8&lon=-71.2&date=2026-02-03
GET  /api/v1/hotspots
GET  /api/v1/tides?date=2026-02-03
POST /api/v1/capture  # Reportar captura
GET  /api/v1/validation/metrics  # Métricas de accuracy
```

**Archivos a crear:**
| Archivo | Propósito |
|---------|-----------|
| `api/__init__.py` | Módulo API |
| `api/main.py` | FastAPI app |
| `api/routes/predictions.py` | Endpoints de predicción |
| `api/routes/captures.py` | Endpoints de capturas |
| `api/schemas.py` | Pydantic schemas |

**Comando:**
```bash
# Iniciar servidor
uvicorn api.main:app --reload --port 8000

# Documentación automática
open http://localhost:8000/docs
```

---

#### Tarea 6: Usar SST Histórico en Análisis Principal
**Objetivo:** Aprovechar los 354,362 registros de SST descargados.

**Lineamientos:**

Actualmente el análisis usa API Open-Meteo en tiempo real. Los datos históricos de Copernicus (`data/raw/sst/copernicus/`) no se usan en el análisis principal.

**Cambios requeridos:**
1. Crear `SSTHistoricalProvider` que lea desde Silver layer
2. Fallback a API si no hay datos históricos para la fecha
3. Mejorar `analyze_spots()` para usar SST de mayor resolución

**Archivos a modificar:**
| Archivo | Cambio |
|---------|--------|
| `controllers/analysis.py` | Agregar `_get_sst_from_historical()` |
| `data/data_manager.py` | Método para consultar SST por fecha/ubicación |

---

#### Tarea 7: Hotspots Dinámicos desde GFW
**Objetivo:** Generar hotspots automáticamente basados en datos reales de pesca.

**Lineamientos:**

Los 1,085 registros de GFW contienen ubicaciones reales donde pescan los barcos. Usar estos datos para:
1. Identificar clusters de actividad pesquera
2. Generar hotspots dinámicos por temporada
3. Complementar los 18 hotspots verificados

**Algoritmo propuesto:**
```python
from sklearn.cluster import DBSCAN

def generate_dynamic_hotspots(gfw_data, eps_km=2, min_samples=5):
    """
    Identifica hotspots usando clustering DBSCAN.

    Args:
        gfw_data: DataFrame con lat, lon, fishing_hours
        eps_km: Radio de vecindad en km
        min_samples: Mínimo de puntos para formar cluster

    Returns:
        Lista de hotspots dinámicos con centroide y score
    """
    coords = gfw_data[['lat', 'lon']].values

    # DBSCAN con distancia Haversine
    clustering = DBSCAN(eps=eps_km/111, min_samples=min_samples)
    gfw_data['cluster'] = clustering.fit_predict(coords)

    # Calcular centroides y scores
    hotspots = []
    for cluster_id in gfw_data['cluster'].unique():
        if cluster_id == -1:
            continue  # Ruido

        cluster_data = gfw_data[gfw_data['cluster'] == cluster_id]
        centroid = cluster_data[['lat', 'lon']].mean()
        score = cluster_data['fishing_hours'].sum()

        hotspots.append({
            'lat': centroid['lat'],
            'lon': centroid['lon'],
            'score': score,
            'source': 'GFW_dynamic'
        })

    return hotspots
```

---

### Prioridad BAJA (Futuro)

#### Tarea 8: Migrar a Deep Learning (U-Net)
**Cuándo:** Si los datos crecen a >1M registros.

**Lineamientos:**
- Usar arquitectura U-Net para segmentación de zonas
- Entrada: Grilla de SST + Chl-a + corrientes
- Salida: Máscara de probabilidad de pesca
- Framework: PyTorch con MPS (Mac M3)

---

#### Tarea 9: App Móvil para Pescadores
**Cuándo:** Después de implementar API REST.

**Lineamientos:**
- Framework: React Native o Flutter
- Features: Ver predicciones, reportar capturas, modo offline
- Integración con GPS del teléfono

---

#### Tarea 10: Batimetría Real (GEBCO)
**Cuándo:** Si se necesita mayor precisión en profundidad.

**Lineamientos:**
- Dataset: GEBCO 2023 (15 arc-second)
- Formato: NetCDF
- Reemplazar proxy de distancia por profundidad real

---

## Cronograma Sugerido

| Semana | Tareas | Entregable |
|--------|--------|------------|
| 1 | Tarea 2 (Multi-día) | Selector de fecha funcional |
| 2 | Tarea 3 (Recolectar datos) | Formulario + 20 registros |
| 3-4 | Tarea 1 (Validación CPUE) | Sistema de validación |
| 5 | Tarea 4 (Clorofila-a) | Feature Chl-a integrado |
| 6-7 | Tarea 5 (API REST) | Servidor FastAPI |
| 8+ | Tareas 6-7 | Mejoras incrementales |

---

## Comandos de Referencia

```bash
# Análisis básico
python main.py

# Análisis para fecha específica
python main.py --date 2026-02-15

# Búsqueda por proximidad
python main.py --lat -17.8 --lon -71.2 --radius 10

# Predicciones horarias
python scripts/generate_hourly_predictions.py --date 2026-02-15 --all-hotspots

# Actualizar base de datos
python scripts/update_database.py

# Validar integridad
python scripts/validate_data.py --all

# Ejecutar tests
python -m pytest tests/ -v
```

---

## Checklist Pre-Implementación

Antes de comenzar cualquier tarea:

- [ ] Leer este documento completo
- [ ] Verificar prerequisitos (V1-V5 completados)
- [ ] Ejecutar tests existentes: `python -m pytest tests/ -v`
- [ ] Verificar que `python main.py` funciona sin errores
- [ ] Crear rama de feature si aplica

Después de completar cada tarea:

- [ ] Ejecutar tests
- [ ] Actualizar `DEVELOPMENT_GUIDELINES.md` si hay nuevos patrones
- [ ] Actualizar este documento marcando tarea como completada
- [ ] Documentar cualquier deuda técnica identificada

---

## Notas Importantes

1. **No usar datos sintéticos** - Solo datos reales de APIs verificadas
2. **Fallbacks siempre** - Si un dato falla, usar valor neutral (0.5)
3. **Inmutabilidad Bronze** - NUNCA modificar archivos en `data/raw/`
4. **Constantes en `domain.py`** - No hardcodear valores en otros archivos
5. **Paths en `DataConfig`** - Centralizar rutas de datos

---

*Documento creado: 2026-02-03*
*Proyecto: Fishing Predictor - Tacna/Ilo, Perú*
