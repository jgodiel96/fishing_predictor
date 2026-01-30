# Plan V1: Arquitectura de Datos Bronze/Silver/Gold

**Fecha:** 2026-01-29
**Version:** 1.0
**Estado:** ✅ COMPLETADO

---

## Resumen Ejecutivo

Este plan establecio la reorganizacion de la estructura de datos del proyecto siguiendo la arquitectura **Bronze/Silver/Gold** (Data Lakehouse) para garantizar:
- Inmutabilidad de datos crudos
- Trazabilidad completa
- Actualizaciones incrementales seguras
- Regeneracion de datos procesados

---

## 1. Problema Original

### Estructura Problematica (Antes)

```
data/
├── real_only/
│   └── real_data_100.db          # TODO mezclado
├── historical/
│   ├── historical_data.db        # Legacy
│   └── real_training_data.db     # Training
├── cache/                        # 957 archivos sin organizar
└── copernicus/                   # Descargas sueltas
```

### Problemas Identificados

| Problema | Impacto | Estado |
|----------|---------|--------|
| Datos mezclados en una sola DB | No se puede distinguir origen | ✅ Resuelto |
| Paths hardcodeados en 4+ archivos | Cambios requieren editar multiple | ✅ Resuelto |
| Sin particionamiento temporal | No se puede actualizar solo un mes | ✅ Resuelto |
| Sin manifest de descargas | No hay trazabilidad | ✅ Resuelto |
| Datos sinteticos mezclados | Contaminan datos reales | ✅ Eliminados |

---

## 2. Solucion Implementada

### Arquitectura Bronze/Silver/Gold

```
data/
├── raw/                                    # BRONZE - INMUTABLE
│   ├── gfw/                               # Global Fishing Watch
│   │   ├── 2020-01.parquet ... 2026-01.parquet
│   │   └── _manifest.json
│   ├── open_meteo/                        # Condiciones marinas
│   │   ├── 2020-01.parquet ... 2026-01.parquet
│   │   └── _manifest.json
│   └── sst/copernicus/                    # SST satelital
│       ├── 2020-01.parquet ... 2026-01.parquet
│       └── _manifest.json
│
├── processed/                              # SILVER - REGENERABLE
│   ├── fishing_consolidated.db
│   ├── marine_consolidated.db
│   └── training_features.parquet
│
└── analytics/                              # GOLD - ML-READY
    └── current/
```

### Estadisticas Finales

| Capa | Fuente | Archivos | Registros |
|------|--------|----------|-----------|
| Bronze | GFW | 73 | 1,305 |
| Bronze | Open-Meteo | 73 | 216,354 |
| Bronze | Copernicus SST | 73 | 354,362 |
| Silver | Fishing DB | 1 | 1,085 |
| Silver | Marine DB | 1 | 572,793 |

---

## 3. Archivos Creados

| Archivo | Proposito |
|---------|-----------|
| `data/data_config.py` | Configuracion centralizada de paths |
| `data/manifest.py` | Gestion de manifests con checksums |
| `data/consolidator.py` | Consolidacion Bronze -> Silver |
| `scripts/download_incremental.py` | Descargas incrementales |
| `scripts/validate_data.py` | Validacion de integridad |
| `scripts/update_database.py` | Actualizacion completa |

---

## 4. Reglas Establecidas

### Capa Bronze (Inmutable)
1. NUNCA modificar archivos existentes
2. Solo AGREGAR nuevos archivos (nuevos meses)
3. Naming: `YYYY-MM.parquet`
4. Manifest obligatorio con checksums SHA256

### Capa Silver (Regenerable)
1. Puede regenerarse desde Bronze
2. Script de consolidacion idempotente
3. Deduplicacion automatica

### Capa Gold (Versionado)
1. Versionado por fecha
2. Mantener ultimas 5 versiones
3. Metadata de features usados

---

## 5. Comandos Principales

```bash
# Actualizar datos
python scripts/update_database.py

# Validar integridad
python scripts/validate_data.py --all

# Descarga incremental
python scripts/download_incremental.py --source copernicus_sst --start 2020-01 --end 2026-01

# Consolidar
python -c "from data.consolidator import Consolidator; Consolidator().consolidate_all()"
```

---

## 6. Resultado

✅ Plan completado exitosamente el 2026-01-30

- 6 anos de datos reales (2020-2026)
- 3 fuentes de datos integradas
- Arquitectura reproducible y mantenible
- Datos sinteticos eliminados
- Sistema de manifests con trazabilidad

---

*Plan V1 completado: 2026-01-30*
*Siguiente: Plan V2 - Validacion Cientifica y Estado del Arte*
