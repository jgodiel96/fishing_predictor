# Plan V3: Implementacion de Mejoras

**Fecha:** 2026-01-30
**Version:** 3.0
**Estado:** ✅ COMPLETADO
**Prerequisitos:**
- [Plan V1 - Arquitectura de Datos](PLAN_V1_ARQUITECTURA_DATOS.md) ✅ Completado
- [Plan V2 - Validacion Cientifica](PLAN_V2_VALIDACION_CIENTIFICA.md) ✅ Completado

---

## Evolucion del Proyecto

| Plan | Enfoque | Estado |
|------|---------|--------|
| **V1** | Arquitectura Bronze/Silver/Gold, datos reales 2020-2026 | ✅ Completado |
| **V2** | Estado del Arte, validacion cientifica | ✅ Completado |
| **V3** | Implementacion de mejoras tecnicas | ✅ Completado |
| **V4** | Integración completa de datos al análisis principal | ✅ Completado |

**Siguiente Plan:** [Plan V4 - Integración de Datos](PLAN_V4_INTEGRACION_DATOS.md)

---

## 1. Resumen de Mejoras Implementadas

### 1.1 Integracion de Mareas ✅

**Archivo:** `data/fetchers/tide_fetcher.py` (505 lineas)

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

**Archivo:** `scripts/generate_hourly_predictions.py` (600+ lineas)

| Componente | Estado | Descripcion |
|------------|--------|-------------|
| Generador horario | ✅ | 24 predicciones por dia |
| Integracion mareas | ✅ | TideFetcher integrado |
| Scores por hora | ✅ | Alba/ocaso bonificados |
| Hotspot scoring | ✅ | Proximidad a zonas conocidas |
| SST integration | ✅ | Datos de Copernicus |
| SSS integration | ✅ | Salinidad de Copernicus |
| SLA integration | ✅ | Nivel del mar de Copernicus |
| Base de datos | ✅ | hourly_predictions.db |

**Pesos del score total (con fisica oceanica):**
```python
# Con SSS/SLA disponibles:
total_score = (
    tide_score * 0.30 +     # 30% - Mareas
    hour_score * 0.20 +     # 20% - Hora del dia
    sst_score * 0.15 +      # 15% - Temperatura
    sss_score * 0.15 +      # 15% - Salinidad (#1 en papers)
    sla_score * 0.10 +      # 10% - Nivel del mar (#2 en papers)
    hotspot_score * 0.10    # 10% - Zona historica
) * 100

# Sin SSS/SLA (fallback):
total_score = (
    tide_score * 0.35 +     # 35% - Mareas
    hour_score * 0.25 +     # 25% - Hora
    sst_score * 0.20 +      # 20% - SST
    hotspot_score * 0.20    # 20% - Hotspot
) * 100
```

### 1.3 Visualizacion Horaria en Mapa ✅

**Archivo:** `views/map_view.py` (metodo `add_hourly_panel`)

| Componente | Estado | Descripcion |
|------------|--------|-------------|
| Panel de prediccion | ✅ | Widget interactivo |
| Slider de horas | ✅ | 0-23h con actualizacion en vivo |
| Grafico de scores | ✅ | Barras por hora con colores |
| Grafico de mareas | ✅ | Curva de altura + score |
| Tabla mejores horas | ✅ | Top 5 con medallas |
| Info de mareas | ✅ | Pleamares/bajamares del dia |

### 1.4 SSS (Salinidad) de Copernicus ✅

**Archivo:** `data/fetchers/copernicus_physics_fetcher.py`

| Componente | Estado | Descripcion |
|------------|--------|-------------|
| Fetcher SSS | ✅ | Dataset cmems_mod_glo_phy |
| Score salinidad | ✅ | Rango optimo 34.4-35.3 PSU |
| Integracion | ✅ | En generate_hourly_predictions.py |
| DataConfig | ✅ | RAW_SSS_COPERNICUS path |

**Rango optimo (IMARPE):**
- Minimo: 34.4 PSU
- Maximo: 35.3 PSU
- Fuente: Estudios de anchoveta

### 1.5 SLA (Nivel del Mar) de Copernicus ✅

**Archivo:** `data/fetchers/copernicus_physics_fetcher.py`

| Componente | Estado | Descripcion |
|------------|--------|-------------|
| Fetcher SLA | ✅ | Variable `zos` |
| Score SLA | ✅ | Negativo = upwelling (bueno) |
| Integracion | ✅ | En generate_hourly_predictions.py |
| DataConfig | ✅ | RAW_SLA_COPERNICUS path |

**Logica de scoring:**
- SLA < -0.05m: Upwelling fuerte (score 0.9-1.0)
- SLA < 0: Upwelling leve (score 0.7-0.9)
- SLA ~ 0: Neutro (score 0.5)
- SLA > 0.05m: Hundimiento (score 0.2-0.5)

---

## 2. Archivos Creados/Modificados en V3

### Archivos Nuevos

| Archivo | Lineas | Proposito |
|---------|--------|-----------|
| `data/fetchers/tide_fetcher.py` | 505 | Calculo astronomico de mareas |
| `data/fetchers/copernicus_physics_fetcher.py` | 350 | SSS y SLA de Copernicus |
| `scripts/generate_hourly_predictions.py` | 600+ | Predicciones horarias |
| `data/processed/hourly_predictions.db` | - | Base de datos SQLite |

### Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `views/map_view.py` | +200 lineas: metodo `add_hourly_panel` |
| `data/data_config.py` | +6 lineas: RAW_SSS_*, RAW_SLA_* paths |

---

## 3. Comandos V3

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

# Descargar SSS y SLA (requiere credenciales Copernicus)
python data/fetchers/copernicus_physics_fetcher.py --start 2024-01 --end 2026-01 -v
```

---

## 4. Ejemplo de Salida

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

--- TODAS LAS HORAS ---
Hora   Score    Marea              Altura
---------------------------------------------
00:00     65  ██████     Marea saliente     -0.22m
06:00     81  ████████   Marea saliente      0.24m  ← MEJOR
07:00     79  ███████    Marea saliente      0.11m
...
18:00     79  ███████    Marea saliente     -0.33m
19:00     80  ████████   Marea saliente     -0.53m  ← 2do MEJOR
```

---

## 5. Validacion Cientifica

Las mejoras implementadas estan basadas en literatura revisada en V2:

| Mejora | Justificacion | Paper |
|--------|---------------|-------|
| Mareas | +200-300% exito en cambios de marea | Estudios costeros 2019 |
| Hora alba/ocaso | Picos de actividad alimenticia | Consenso cientifico |
| SST scoring | Variable #3 en prediccion | Papers 2024 |
| SSS scoring | Variable #1 en prediccion | Papers 2024 (Frontiers) |
| SLA scoring | Variable #2, indica upwelling | Chavez et al. 2008 |
| Hotspot proximity | Zonas historicas productivas | IMARPE, GFW |

---

## 6. Siguiente Plan: V4 (COMPLETADO)

**Ver:** [PLAN_V4_INTEGRACION_DATOS.md](PLAN_V4_INTEGRACION_DATOS.md)

V4 integró todos los datos disponibles (mareas, SSS, SLA) al análisis principal:
- Score promedio mejoró de ~25 a ~48 (+92%)
- 6 fuentes de datos ahora integradas
- 3 nuevas variables ambientales con bonuses

### Mejoras Futuras (V5+)

| Prioridad | Mejora | Descripcion |
|-----------|--------|-------------|
| ALTA | Sistema validacion CPUE | Comparar predicciones vs capturas reales |
| MEDIA | Clorofila-a | Agregar Chl-a de Copernicus |
| MEDIA | API REST | Servicio web para predicciones |
| BAJA | Deep Learning | Migrar a U-Net si datos crecen |
| BAJA | App movil | Interfaz para pescadores |

---

## 7. Resumen de Tareas V3

| # | Tarea | Estado |
|---|-------|--------|
| 11 | Integrar datos de mareas | ✅ Completado |
| 12 | Implementar predicciones horarias | ✅ Completado |
| 13 | Mejorar timeline con visualizacion | ✅ Completado |
| 14 | Agregar SSS de Copernicus | ✅ Completado |
| 15 | Agregar SLA de Copernicus | ✅ Completado |
| 16 | Crear documento PLAN_V3 | ✅ Completado |

---

*Plan V3 completado: 2026-01-30*
*Actualizado: 2026-01-31 (referencia a V4)*
*Proyecto: Fishing Predictor - Tacna/Ilo, Peru*
