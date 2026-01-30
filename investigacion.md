# Investigacion: Prediccion de Zonas de Pesca

**Ultima actualizacion:** 2026-01-30

## Resumen

Este documento recopila la investigacion sobre metodos, APIs y proyectos existentes para la prediccion de zonas de pesca basada en datos oceanograficos reales.

---

## 1. Datos del Proyecto (Estado Actual)

### 1.1 Resumen de Datos Reales

| Fuente | Tipo | Registros | Periodo | Estado |
|--------|------|-----------|---------|--------|
| Copernicus Marine | SST satelital | 354,362 | 2020-2026 | ✅ Implementado |
| Open-Meteo ERA5 | Olas, viento | 216,354 | 2020-2026 | ✅ Implementado |
| Global Fishing Watch | Pesca AIS | 1,085 | 2020-2026 | ✅ Implementado |
| IMARPE | Climatologia SST | - | Historico | ✅ Fallback |

### 1.2 Arquitectura de Datos

```
data/
├── raw/                    # BRONZE - Inmutable
│   ├── gfw/               # 73 archivos mensuales
│   ├── open_meteo/        # 73 archivos mensuales
│   └── sst/copernicus/    # 73 archivos mensuales
│
├── processed/              # SILVER - Regenerable
│   ├── fishing_consolidated.db
│   ├── marine_consolidated.db
│   └── training_features.parquet
│
└── analytics/              # GOLD - ML-ready
    └── current/
```

---

## 2. Proyectos GitHub Relevantes

### 2.1 Prediccion de Zonas de Pesca (PFZ)

| Proyecto | Descripcion | Tecnicas |
|----------|-------------|----------|
| [Potential-Fishing-Zone-Analysis](https://github.com/Sutirtha2304/Potential-Fishing-Zone-Analysis-Using-Machine-Learning) | Metodologia INCOIS para PFZ | Random Forest, PCA, RFE |
| [tuna-prediction](https://github.com/stevenalbert/tuna-prediction) | Prediccion de atun en Indonesia | Naive Bayes, NOAA SST, Global Fishing Watch |
| [PFZ-Prediction-ML](https://github.com/erayon/PFZ-Prediction-Using-Machine-Learning) | PFZ con propiedades fisico-quimicas | Machine Learning |

### 2.2 Deteccion de Frentes Termicos

| Proyecto | Descripcion | Algoritmo |
|----------|-------------|-----------|
| [pyBOA](https://github.com/AlxLhrNc/pyBOA) | Algoritmo Belkin-O'Reilly | Filtro contextual, gradientes |
| [JUNO](https://github.com/CoLAB-ATLANTIC/JUNO) | Multiples algoritmos de frentes | Probabilidades frontales |
| [boaR](https://github.com/galuardi/boaR) | Implementacion R de BOA | Deteccion de bordes |

### 2.3 Global Fishing Watch

| Proyecto | Descripcion |
|----------|-------------|
| [vessel-classification](https://github.com/GlobalFishingWatch/vessel-classification) | CNN para clasificar embarcaciones desde AIS |
| [training-data](https://github.com/GlobalFishingWatch/training-data) | Datos etiquetados de eventos de pesca |
| [gfwr](https://github.com/GlobalFishingWatch/gfwr) | Paquete R para API de GFW |

---

## 3. APIs de Datos Oceanograficos

### 3.1 Fuentes Implementadas

| Fuente | Resolucion | Cobertura | Acceso | Estado |
|--------|------------|-----------|--------|--------|
| **Copernicus Marine** | 0.05° (~5km) | Global | copernicusmarine | ✅ Activo |
| **Open-Meteo ERA5** | Variable | Global | REST API | ✅ Activo |
| **Global Fishing Watch** | Mensual | Global | REST API | ✅ Activo |

### 3.2 Fuentes Alternativas

| Fuente | Resolucion | Uso |
|--------|------------|-----|
| NOAA OISST | 0.25° (~25km) | Backup SST |
| NASA MUR SST | 1km | Alta resolucion |
| HYCOM | 1/12° (~8km) | Corrientes |

---

## 4. Hallazgos Cientificos Clave

### 4.1 SST para Prediccion de Peces

- **Precision**: SST + Clorofila-a logran **75-83%** de precision en prediccion
- **Rango India**: 24-28°C optimo con Chl-a >6.53 mg/m³
- **Modelos GAM**: >83% precision para caballa

### 4.2 Frentes Termicos

**Metodos de deteccion:**
1. **Gradiente** (Belkin-O'Reilly): Detecta regiones con grandes gradientes de temperatura
2. **Histograma** (SIED Cayula-Cornillon): Detecta limites entre poblaciones de pixeles

**Parametros:**
- Ventana: 32x32 pixeles
- Umbral diferencia: **0.45°C** para frentes fuertes
- Ratio varianza: 0.76

**Importancia:**
- Los frentes concentran nutrientes y presas
- Los peces se agregan en limites termicos
- Mayor biodiversidad en zonas frontales

### 4.3 Corrientes Oceanicas y Migracion

- **70%** de cambios en profundidad correlacionan con fluctuaciones de temperatura
- **74%** de cambios en latitud correlacionan con temperatura regional
- Modelos LSTM predicen trayectorias de migracion basadas en temperatura

### 4.4 Upwelling (Surgencia)

**Estadisticas criticas:**
- **25%** de capturas marinas globales provienen de 5 zonas de upwelling
- Estas zonas ocupan solo **5%** del area oceanica total
- Proporcionan **7%** de produccion primaria marina global
- **20%** de pesquerias de captura mundial

---

## 5. Sistema de Corriente de Humboldt (Peru)

### 5.1 Importancia Global

El Sistema de Corriente de Humboldt Norte (NHCS) es el **ecosistema mas productivo del mundo** en terminos de pesca:
- Contribuye a mas del **15%** de la captura anual global de peces
- Dominado por anchoveta (Engraulis ringens)

### 5.2 Parametros Optimos para Anchoveta

| Parametro | Rango Optimo | Fuente |
|-----------|--------------|--------|
| Temperatura | 14-24°C | Encuestas IMARPE |
| Salinidad | 34.40-35.30 | Observaciones de campo |
| Masa de Agua | Agua Costera Fria (CCW) | Estudios |
| Profundidad | Superficie a 50m | Encuestas hidroacusticas |

### 5.3 Variaciones Estacionales

| Estacion | Preferencia SST |
|----------|-----------------|
| Primavera | <22.1°C |
| Verano | <23.1°C |
| Invierno | <17.2°C |

### 5.4 Efectos El Nino

| Evento | Impacto en Biomasa |
|--------|-------------------|
| 1982-83 | Biomasa mas baja registrada |
| 1997-98 | De 5.8M a 1.2M toneladas |
| 2023-24 | 50% reduccion vs 2022 |

**Mecanismos:**
1. Upwelling debilitado reduce nutrientes
2. Menos fitoplancton = menos alimento para anchoveta
3. Anchoveta migra al sur y mas cerca de la costa

---

## 6. Especies Objetivo (Tacna-Ilo)

### 6.1 Rangos de Temperatura Optima

| Especie | Temp Optima | Sustrato | Senuelos |
|---------|-------------|----------|----------|
| **Cabrilla** | 16-19°C | Roca | Grubs 3", jigs 15-25g |
| **Corvina** | 15-18°C | Arena, Mixto | Jigs metalicos 30-50g |
| **Robalo** | 17-21°C | Roca, Mixto | Poppers 12cm, minnows |
| **Lenguado** | 14-17°C | Arena | Vinilos paddle tail |
| **Pejerrey** | 14-18°C | Arena, Mixto | Cucharillas pequenas |

### 6.2 Hotspots Verificados (18 Ubicaciones)

Basados en datos de IMARPE y conocimiento de pescadores locales:

| Ubicacion | Lat | Lon | Sustrato | Bonus |
|-----------|-----|-----|----------|-------|
| **Punta Coles** | -17.702 | -71.332 | Roca | 1.35 |
| **Punta Blanca** | -17.812 | -71.082 | Roca | 1.30 |
| **Pozo Redondo** | -17.782 | -71.122 | Mixto | 1.25 |
| **Vila Vila** | -18.018 | -70.912 | Roca | 1.25 |
| **Fundicion** | -17.757 | -71.172 | Roca | 1.20 |
| **Pozo Lizas** | -17.642 | -71.340 | Roca | 1.20 |
| **Ite Sur** | -17.872 | -71.018 | Roca | 1.20 |
| **Ite Norte** | -17.932 | -70.968 | Arena | 1.20 |
| **Boca del Rio** | -18.1205 | -70.843 | Arena | 1.20 |

---

## 7. Algoritmo PFZ Implementado

### 7.1 Pipeline de Datos

```
1. BRONZE LAYER (raw/):
   - Copernicus SST: YYYY-MM.parquet
   - Open-Meteo: YYYY-MM.parquet
   - GFW: YYYY-MM.parquet

2. SILVER LAYER (processed/):
   - Consolidacion con deduplicacion
   - Validacion de integridad
   - Join de fuentes por fecha/ubicacion

3. GOLD LAYER (analytics/):
   - 32 features extraidos
   - Training dataset listo para ML
```

### 7.2 Features del Modelo (32 total)

| Categoria | Features | Descripcion |
|-----------|----------|-------------|
| SST | 6 | Temperatura, anomalia, scores |
| Frentes | 5 | Gradiente, direccion, intensidad |
| Corrientes | 6 | Velocidad, convergencia, cizalladura |
| Olas | 3 | Altura, periodo, indice |
| Upwelling | 3 | Indice, Ekman, favorable |
| Espacial | 4 | Distancia costa, profundidad |
| Historico | 2 | Distancia/similitud hotspots |
| Temporal | 3 | Hora, luna, estacion |

### 7.3 Clasificacion PFZ

```python
SI (14°C < SST < 24°C) Y
   (wave_height < 2.0m) Y
   (frente_termico_presente O hotspot_cercano) Y
   (condiciones_seguras):
   ENTONCES: zona de pesca ALTA probabilidad
```

---

## 8. Visualizacion

### 8.1 Colormaps Recomendados (cmocean)

| Tipo de Dato | Colormap | Descripcion |
|--------------|----------|-------------|
| SST | `thermal` | Secuencial, frio a calido |
| Clorofila-a | `algae` | Secuencial, verde |
| Anomalias | `balance` | Divergente |
| Velocidad | `speed` | Secuencial |

### 8.2 Accesibilidad

- **7%** de hombres tienen daltonismo rojo-verde
- Evitar rainbow/jet colormaps
- Usar colormaps perceptualmente uniformes

---

## 9. Fuentes Principales

### Papers y Articulos
- Belkin, I.M., O'Reilly, J.E. (2009) - Algoritmo BOA para deteccion de frentes
- Thyng et al. (2016) - "True Colors of Oceanography" - Colormaps cientificos
- Chavez et al. (2008) - Climate and fisheries in the Humboldt Current

### APIs y Herramientas
- [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api)
- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [Global Fishing Watch APIs](https://globalfishingwatch.org/our-apis/)
- [NOAA CoastWatch ERDDAP](https://coastwatch.noaa.gov/)

### Instituciones
- **IMARPE** (Peru): Instituto del Mar del Peru
- **NOAA** (USA): Datos SST, corrientes
- **ESA/Copernicus** (EU): Sentinel-3, datos marinos

---

## 10. Comandos del Proyecto

### 10.1 Actualizacion de Datos

```bash
# Actualizar todo
python scripts/update_database.py

# Validar integridad
python scripts/validate_data.py --all
```

### 10.2 Descarga Incremental

```bash
# Por fuente
python scripts/download_incremental.py --source copernicus_sst --start 2024-01 --end 2026-01
python scripts/download_incremental.py --source gfw --start 2024-01 --end 2026-01
python scripts/download_incremental.py --source open_meteo --start 2024-01 --end 2026-01
```

### 10.3 Consolidacion

```python
from data.consolidator import Consolidator
Consolidator().consolidate_all()
```

---

*Documento actualizado: 2026-01-30*
*Proyecto: Fishing Predictor - Tacna/Ilo, Peru*
