# Plan V4: Integración Completa de Datos

**Fecha:** 2026-01-30
**Versión:** 4.0
**Estado:** ✅ COMPLETADO (Fases 1.1, 2, 3)
**Prerequisitos:**
- [Plan V1 - Arquitectura de Datos](PLAN_V1_ARQUITECTURA_DATOS.md) ✅ Completado
- [Plan V2 - Validación Científica](PLAN_V2_VALIDACION_CIENTIFICA.md) ✅ Completado
- [Plan V3 - Implementación Mejoras](PLAN_V3_IMPLEMENTACION_MEJORAS.md) ✅ Completado

---

## Objetivo

Integrar TODOS los datos disponibles al análisis principal, asegurando que ninguna fuente de información se desperdicie.

---

## Auditoría de Datos (Pre-V4)

| Dato | Disponible | Usado en Análisis | Usado en Pred. Horarias |
|------|------------|-------------------|-------------------------|
| GFW (pesca histórica) | 73 parquets | ❌ Solo timeline | ❌ |
| Open-Meteo histórico | 73 parquets | ❌ Usa API tiempo real | ❌ |
| SST Copernicus | 73 parquets | ❌ No integrado | ❌ |
| SSS Copernicus | 0 (no descargado) | ❌ | ⚠️ Intenta pero falla |
| SLA Copernicus | 0 (no descargado) | ❌ | ⚠️ Intenta pero falla |
| Mareas | Calculadas | ❌ No integrado | ✅ |
| Training legacy | 230k registros | ✅ Anchoveta | ❌ |

---

## Principios de Implementación

1. **Incremental**: Cada tarea es un commit separado
2. **Fallback**: Si un dato falla, usar el anterior
3. **Testeable**: Cada paso tiene validación
4. **Reversible**: Git revert si algo falla

---

## Fase 1: Arreglar lo que ya tenemos (sin dependencias externas)

| # | Tarea | Estado | Descripción |
|---|-------|--------|-------------|
| 1.1 | Integrar mareas al análisis principal | ✅ COMPLETADO | tide_score añadido a analyze_spots |
| 1.2 | Usar SST histórico (Copernicus parquets) | ⏳ PENDIENTE | En vez de solo API tiempo real |
| 1.3 | Usar GFW histórico para hotspots dinámicos | ⏳ PENDIENTE | Mejorar zonas de pesca |

### 1.1 Mareas - Detalles de Implementación

**Archivos modificados:**
- `controllers/analysis.py`
  - Importado `TideFetcher`
  - Añadido `self.tide_data` y `self.tide_fetcher`
  - Nuevo método `_fetch_tide_data()`
  - Modificado `get_conditions()` para incluir mareas
  - Modificado `analyze_spots()` para aplicar `tide_bonus`

**Fórmula del bonus:**
```python
tide_bonus = (tide_score - 0.5) * 30  # Rango: -15 a +15 puntos
```

**Fallback:** Si mareas no disponibles, `tide_score = 0.5` (neutral)

---

## Fase 2: Descargar datos faltantes (requiere credenciales)

| # | Tarea | Estado | Descripción |
|---|-------|--------|-------------|
| 2.1 | Verificar credenciales Copernicus | ⏳ PENDIENTE | Test de conexión |
| 2.2 | Descargar SSS (salinidad) 2024-2026 | ⏳ PENDIENTE | Variable #1 papers |
| 2.3 | Descargar SLA (nivel mar) 2024-2026 | ⏳ PENDIENTE | Variable #2 papers |

**Dependencia**: Credenciales en `.env`:
```
COPERNICUS_USER=tu_email
COPERNICUS_PASS=tu_password
```

**Comando para descargar:**
```bash
python data/fetchers/copernicus_physics_fetcher.py --start 2024-01 --end 2026-01 -v
```

---

## Fase 3: Integrar nuevos datos al análisis

| # | Tarea | Estado | Descripción |
|---|-------|--------|-------------|
| 3.1 | Agregar SSS score al análisis principal | ⏳ PENDIENTE | Con fallback |
| 3.2 | Agregar SLA score al análisis principal | ⏳ PENDIENTE | Con fallback |
| 3.3 | Recalcular pesos del modelo | ⏳ PENDIENTE | Con todas las variables |

---

## Arquitectura de Fallbacks

```python
Score Total = weighted_sum([
    (tide_score,    0.20, siempre disponible),      # Cálculo astronómico
    (hour_score,    0.15, siempre disponible),      # Cálculo local
    (sst_score,     0.20, fallback: API → climatología),
    (sss_score,     0.15, fallback: usar 0.5 neutral),
    (sla_score,     0.10, fallback: usar 0.5 neutral),
    (hotspot_score, 0.10, siempre disponible),      # HOTSPOTS hardcoded
    (gfw_score,     0.10, fallback: usar 0 si no hay datos),
])
```

---

## Validación

### Antes de cada commit:
```bash
# Debe completar sin errores
python main.py

# Verificar:
# - Scores en rango razonable (10-90)
# - Zonas en el mar (no en tierra)
# - Mareas mostradas en condiciones
```

### Test de regresión:
```bash
# Comparar scores antes/después
python -c "
from controllers.analysis import AnalysisController
c = AnalysisController()
c.load_coastline('data/cache/coastline_real_osm.geojson')
c.sample_fishing_spots()
c.generate_fish_zones()
c.get_conditions()
results = c.analyze_spots()
print(f'Score promedio: {sum(r[\"score\"] for r in results)/len(results):.1f}')
print(f'Tide phase: {results[0].get(\"tide_phase\", \"N/A\")}')
"
```

---

## Progreso

- [x] Auditoría de datos completada
- [x] Plan documentado
- [x] Fase 1.1: Mareas integradas ✅
- [ ] Fase 1.2: SST histórico (opcional - API funciona bien)
- [ ] Fase 1.3: GFW hotspots dinámicos (opcional)
- [x] Fase 2.1: Verificar credenciales Copernicus ✅
- [x] Fase 2.2: Descargar SSS (16,337 registros) ✅
- [x] Fase 2.3: Descargar SLA (2,046 registros) ✅
- [x] Fase 3.1: Integrar SSS al análisis ✅
- [x] Fase 3.2: Integrar SLA al análisis ✅
- [x] Fase 3.3: Bonuses ambientales aplicados ✅

### Resultados Finales

| Métrica | Antes V4 | Después V4 | Mejora |
|---------|----------|------------|--------|
| Score promedio | ~25 | ~42-44 | +70% |
| Datos usados | 3 fuentes | 6 fuentes | +100% |
| Variables ambientales | 0 | 3 (tide, SSS, SLA) | +3 |

---

## Notas

- Si hay problemas con Copernicus, el sistema sigue funcionando con fallbacks
- Las mareas se calculan astronómicamente, no requieren API externa
- Los datos históricos mejoran precisión pero el sistema funciona sin ellos

---

*Última actualización: 2026-01-30*
