# Plan V3: Implementacion de Mejoras

**Fecha:** 2026-01-30
**Version:** 3.0
**Estado:** En progreso
**Prerequisitos:**
- [Plan V1 - Arquitectura de Datos](PLAN_V1_ARQUITECTURA_DATOS.md) ✅ Completado
- [Plan V2 - Validacion Cientifica](PLAN_V2_VALIDACION_CIENTIFICA.md) ✅ Completado

---

## Evolucion del Proyecto

| Plan | Enfoque | Estado |
|------|---------|--------|
| **V1** | Arquitectura Bronze/Silver/Gold, datos reales 2020-2026 | ✅ Completado |
| **V2** | Estado del Arte, validacion cientifica | ✅ Completado |
| **V3** | Implementacion de mejoras tecnicas | 📋 En progreso |

---

## 1. Resumen de Mejoras Implementadas

### 1.1 Integracion de Mareas ✅

**Archivo:** `data/fetchers/tide_fetcher.py`

| Componente | Estado | Descripcion |
|------------|--------|-------------|
| Calculo astronomico | ✅ | Armonicos M2, S2, K1, O1 |
| Fase lunar | ✅ | Mareas vivas/muertas |
| Estados de marea | ✅ | flooding, ebbing, slack_high, slack_low |
| Scores de pesca | ✅ | Basado en literatura V2 |
| Extremos del dia | ✅ | Pleamares y bajamares |

**Scores de pesca por fase:**
```python
FISHING_SCORES = {
    'flooding': 0.85,      # Marea entrante - muy buena
    'ebbing': 0.75,        # Marea saliente - buena
    'slack_high': 0.40,    # Reposo en pleamar - moderada
    'slack_low': 0.35,     # Reposo en bajamar - baja
    'peak_change': 1.0,    # Cambio de marea - excelente
}
```

### 1.2 Predicciones Horarias ✅

**Archivo:** `scripts/generate_hourly_predictions.py`

| Componente | Estado | Descripcion |
|------------|--------|-------------|
| Generador horario | ✅ | 24 predicciones por dia |
| Integracion mareas | ✅ | TideFetcher integrado |
| Scores por hora | ✅ | Alba/ocaso bonificados |
| Hotspot scoring | ✅ | Proximidad a zonas conocidas |
| SST integration | ✅ | Datos de Copernicus |
| Base de datos | ✅ | hourly_predictions.db |

**Pesos del score total:**
```python
total_score = (
    tide_score * 0.35 +     # 35% - Mareas (muy importante costa)
    hour_score * 0.25 +     # 25% - Hora del dia
    sst_score * 0.20 +      # 20% - Temperatura
    hotspot_score * 0.20    # 20% - Zona historica
) * 100
```

**Uso:**
```bash
# Prediccion para una ubicacion
python scripts/generate_hourly_predictions.py --date 2026-01-30

# Todos los hotspots
python scripts/generate_hourly_predictions.py --date 2026-01-30 --all-hotspots --save

# Salida JSON
python scripts/generate_hourly_predictions.py --date 2026-01-30 --json
```

---

## 2. Mejoras Pendientes

### 2.1 Mejorar Timeline con Visualizacion Horaria (Prioridad ALTA)

**Objetivo:** Actualizar `views/map_view.py` para mostrar predicciones horarias

**Componentes:**
- [ ] Grafico de mareas del dia
- [ ] Slider de horas (0-23)
- [ ] Tabla de mejores horas
- [ ] Indicador de marea actual

**Mockup:**
```
+------------------------------------------+
|  PREDICCION HORARIA - 30/01/2026         |
+------------------------------------------+
|  [Slider: 0h ----[06h]---- 23h]          |
|                                          |
|  Hora seleccionada: 06:00                |
|  Score: 81/100                           |
|  Marea: Saliente (0.24m)                 |
|  Condicion: Excelente para pesca         |
+------------------------------------------+
```

### 2.2 Agregar SSS (Salinidad) de Copernicus (Prioridad MEDIA)

**Dataset:** `GLOBAL_ANALYSISFORECAST_PHY_001_024`
**Variable:** `so` (salinity)

**Implementacion:**
```python
# En download_incremental.py agregar:
def download_sss(start_date, end_date, region):
    dataset_id = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
    # ...
```

### 2.3 Agregar SLA (Nivel del Mar) de Copernicus (Prioridad MEDIA)

**Dataset:** `GLOBAL_ANALYSISFORECAST_PHY_001_024`
**Variable:** `zos` (sea surface height)

**Relevancia:** Variable #2 en papers segun estado del arte V2

### 2.4 Sistema de Validacion CPUE (Prioridad MEDIA)

**Objetivo:** Comparar predicciones con capturas reales

**Componentes:**
- [ ] API/formulario para reportar capturas
- [ ] Calculo de metricas (Hit Rate, CPUE Ratio)
- [ ] Dashboard de accuracy

---

## 3. Archivos Creados en V3

| Archivo | Proposito | Estado |
|---------|-----------|--------|
| `data/fetchers/tide_fetcher.py` | Calculo de mareas | ✅ |
| `scripts/generate_hourly_predictions.py` | Predicciones horarias | ✅ |
| `data/processed/hourly_predictions.db` | Base de datos horaria | ✅ |
| `docs/PLAN_V3_IMPLEMENTACION_MEJORAS.md` | Este documento | ✅ |

---

## 4. Comandos V3

```bash
# Prediccion horaria para hoy
python scripts/generate_hourly_predictions.py

# Prediccion para fecha especifica
python scripts/generate_hourly_predictions.py --date 2026-02-15

# Ubicacion especifica
python scripts/generate_hourly_predictions.py --lat -17.812 --lon -71.082

# Todos los hotspots y guardar
python scripts/generate_hourly_predictions.py --all-hotspots --save

# Salida JSON para integracion
python scripts/generate_hourly_predictions.py --json
```

---

## 5. Ejemplo de Salida

```
============================================================
PREDICCION HORARIA - 2026-01-30
Ubicacion: (-17.702, -71.332) - Punta Coles
============================================================

--- MAREAS DEL DIA ---
  00:20 - Bajamar: -0.23m
  05:10 - Pleamar: 0.28m
  09:50 - Bajamar: -0.24m
  14:30 - Pleamar: 0.36m
  19:50 - Bajamar: -0.58m

--- MEJORES 5 HORAS ---
  1. 06:00 - Score: 81/100 (Marea saliente, Alba)
  2. 19:00 - Score: 80/100 (Marea saliente, Atardecer)
  3. 07:00 - Score: 79/100 (Marea saliente)
  4. 18:00 - Score: 79/100 (Marea saliente)
  5. 08:00 - Score: 75/100 (Marea saliente)
```

---

## 6. Validacion Cientifica

Las mejoras implementadas estan basadas en literatura revisada en V2:

| Mejora | Justificacion | Paper |
|--------|---------------|-------|
| Mareas | +200-300% exito en cambios de marea | Estudios costeros 2019 |
| Hora alba/ocaso | Picos de actividad alimenticia | Consenso cientifico |
| SST scoring | Variable #3 en prediccion | Papers 2024 |
| Hotspot proximity | Zonas historicas productivas | IMARPE, GFW |

---

## 7. Proximos Pasos

1. **Inmediato:** Actualizar timeline en map_view.py
2. **Corto plazo:** Agregar SSS y SLA
3. **Mediano plazo:** Sistema de validacion CPUE
4. **Largo plazo:** Migrar a deep learning (U-Net) si datos crecen

---

*Plan V3 iniciado: 2026-01-30*
*Proyecto: Fishing Predictor - Tacna/Ilo, Peru*
