# Investigacion: Prediccion de Zonas de Pesca

## Resumen

Este documento recopila la investigacion sobre metodos, APIs y proyectos existentes para la prediccion de zonas de pesca basada en datos oceanograficos reales.

---

## 1. Proyectos GitHub Relevantes

### 1.1 Prediccion de Zonas de Pesca (PFZ)

| Proyecto | Descripcion | Tecnicas |
|----------|-------------|----------|
| [Potential-Fishing-Zone-Analysis](https://github.com/Sutirtha2304/Potential-Fishing-Zone-Analysis-Using-Machine-Learning) | Metodologia INCOIS para PFZ | Random Forest, PCA, RFE |
| [tuna-prediction](https://github.com/stevenalbert/tuna-prediction) | Prediccion de atun en Indonesia | Naive Bayes, NOAA SST, Global Fishing Watch |
| [PFZ-Prediction-ML](https://github.com/erayon/PFZ-Prediction-Using-Machine-Learning) | PFZ con propiedades fisico-quimicas | Machine Learning |

### 1.2 Deteccion de Frentes Termicos

| Proyecto | Descripcion | Algoritmo |
|----------|-------------|-----------|
| [pyBOA](https://github.com/AlxLhrNc/pyBOA) | Algoritmo Belkin-O'Reilly | Filtro contextual, gradientes |
| [JUNO](https://github.com/CoLAB-ATLANTIC/JUNO) | Multiples algoritmos de frentes | Probabilidades frontales |
| [boaR](https://github.com/galuardi/boaR) | Implementacion R de BOA | Deteccion de bordes |

### 1.3 Acceso a Datos Satelitales

| Proyecto | Descripcion | Uso |
|----------|-------------|-----|
| [erddapy](https://github.com/ioos/erddapy) | Cliente Python para ERDDAP | Acceso a NOAA, NCEI, etc. |
| [copernicus-marine-toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox) | API oficial Copernicus | SST, corrientes, clorofila |
| [cmocean](https://github.com/matplotlib/cmocean) | Colormaps oceanograficos | Visualizacion cientifica |

### 1.4 Global Fishing Watch

| Proyecto | Descripcion |
|----------|-------------|
| [vessel-classification](https://github.com/GlobalFishingWatch/vessel-classification) | CNN para clasificar embarcaciones desde AIS |
| [training-data](https://github.com/GlobalFishingWatch/training-data) | Datos etiquetados de eventos de pesca |
| [gfwr](https://github.com/GlobalFishingWatch/gfwr) | Paquete R para API de GFW |

---

## 2. APIs de Datos Oceanograficos

### 2.1 SST (Temperatura Superficial del Mar)

| Fuente | Resolucion | Cobertura | Acceso | Registro |
|--------|------------|-----------|--------|----------|
| **NOAA OISST** | 0.25° (~25km) | Global, 1981+ | ERDDAP, HTTPS | Gratis |
| **GHRSST MUR** | 1km | Global | NASA PO.DAAC | Earthdata (gratis) |
| **Open-Meteo Marine** | Variable | Global | REST API | Gratis |
| **Copernicus Marine** | Varios | Global | Python lib | Registro gratis |

### 2.2 Corrientes Oceanicas

| Fuente | Resolucion | Variables | Acceso |
|--------|------------|-----------|--------|
| **HYCOM** | 1/12° (~8km) | U/V velocidad, temp, salinidad | ERDDAP, NOMADS |
| **CMEMS MERCATOR** | 1/12° | Corrientes 3D, 50 niveles | copernicusmarine |
| **OSCAR** | 1/3° | Corrientes superficiales | NASA PO.DAAC |
| **Open-Meteo Marine** | Variable | Velocidad, direccion | REST API |

### 2.3 Clorofila-a / Productividad

| Fuente | Resolucion | Temporal | Acceso |
|--------|------------|----------|--------|
| **NASA MODIS** | 4km | Diario/8-dias, 2002+ | earthaccess Python |
| **Sentinel-3 OLCI** | 300m | ~2 dias revisita | Copernicus |
| **VIIRS** | 750m-4km | Diario | NOAA CoastWatch |

### 2.4 Datos de Pesca/Embarcaciones

| Fuente | Tipo | Acceso | Costo |
|--------|------|--------|-------|
| **Global Fishing Watch** | Esfuerzo pesquero, tracks | REST API | Gratis (no comercial) |
| **AISHub** | Posiciones AIS tiempo real | JSON/XML/CSV | Gratis (intercambio) |
| **FAO FishStat** | Estadisticas historicas | FishStatJ, R | Gratis |

---

## 3. Hallazgos Cientificos Clave

### 3.1 SST para Prediccion de Peces

- **Precision**: SST + Clorofila-a logran **75-83%** de precision en prediccion
- **Rango India**: 24-28°C optimo con Chl-a >6.53 mg/m³
- **Modelos GAM**: >83% precision para caballa

### 3.2 Frentes Termicos

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

### 3.3 Corrientes Oceanicas y Migracion

- **70%** de cambios en profundidad correlacionan con fluctuaciones de temperatura
- **74%** de cambios en latitud correlacionan con temperatura regional
- Modelos LSTM predicen trayectorias de migracion basadas en temperatura

### 3.4 Upwelling (Surgencia)

**Estadisticas criticas:**
- **25%** de capturas marinas globales provienen de 5 zonas de upwelling
- Estas zonas ocupan solo **5%** del area oceanica total
- Proporcionan **7%** de produccion primaria marina global
- **20%** de pesquerias de captura mundial

---

## 4. Sistema de Corriente de Humboldt (Peru)

### 4.1 Importancia Global

El Sistema de Corriente de Humboldt Norte (NHCS) es el **ecosistema mas productivo del mundo** en terminos de pesca:
- Contribuye a mas del **15%** de la captura anual global de peces
- Dominado por anchoveta (Engraulis ringens)

### 4.2 Parametros Optimos para Anchoveta

| Parametro | Rango Optimo | Fuente |
|-----------|--------------|--------|
| Temperatura | 14-24°C | Encuestas IMARPE |
| Salinidad | 34.40-35.30 | Observaciones de campo |
| Masa de Agua | Agua Costera Fria (CCW) | Estudios |
| Profundidad | Superficie a 50m | Encuestas hidroacusticas |

### 4.3 Variaciones Estacionales

| Estacion | Preferencia SST |
|----------|-----------------|
| Primavera | <22.1°C |
| Verano | <23.1°C |
| Invierno | <17.2°C |

### 4.4 Efectos El Nino

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

## 5. Especies Objetivo (Tacna-Ilo)

### 5.1 Rangos de Temperatura Optima

| Especie | Temp Optima | Sustrato | Senuelos |
|---------|-------------|----------|----------|
| **Cabrilla** | 16-19°C | Roca | Grubs 3", jigs 15-25g |
| **Corvina** | 15-18°C | Arena, Mixto | Jigs metalicos 30-50g |
| **Robalo** | 17-21°C | Roca, Mixto | Poppers 12cm, minnows |
| **Lenguado** | 14-17°C | Arena | Vinilos paddle tail |
| **Pejerrey** | 14-18°C | Arena, Mixto | Cucharillas pequenas |

### 5.2 Hotspots Verificados (18 Ubicaciones)

Basados en datos de IMARPE y conocimiento de pescadores locales:

| Ubicacion | Lat | Lon | Sustrato | Bonus | Caracteristica |
|-----------|-----|-----|----------|-------|----------------|
| **Punta Coles** | -17.702 | -71.332 | Roca | 1.35 | Reserva, mucha vida |
| **Punta Blanca** | -17.812 | -71.082 | Roca | 1.30 | Punta rocosa, robalo |
| **Pozo Redondo** | -17.782 | -71.122 | Mixto | 1.25 | Pozas naturales |
| **Vila Vila** | -18.018 | -70.912 | Roca | 1.25 | Rocas con estructura |
| **Fundicion** | -17.757 | -71.172 | Roca | 1.20 | Rocas grandes |
| **Pozo Lizas** | -17.642 | -71.340 | Roca | 1.20 | Ideal spinning |
| **Ite Sur** | -17.872 | -71.018 | Roca | 1.20 | Cabrilla |
| **Ite Norte** | -17.932 | -70.968 | Arena | 1.20 | Surgencia activa |
| **Boca del Rio** | -18.1205 | -70.843 | Arena | 1.20 | Desembocadura |
| **Pocoma** | -17.682 | -71.295 | Roca | 1.15 | Acantilados |
| **Ite Centro** | -17.902 | -70.992 | Mixto | 1.15 | Productivo |
| **Punta Mesa** | -17.988 | -70.935 | Roca | 1.15 | Punta con pozas |
| **Gentillar** | -17.842 | -71.048 | Roca | 1.15 | Costa escarpada |
| **Santa Rosa** | -18.087 | -70.868 | Arena | 1.15 | Corvina |
| **Ilo Puerto** | -17.632 | -71.345 | Mixto | 1.10 | Muelle |
| **Media Luna** | -17.732 | -71.222 | Arena | 1.10 | Bahia |
| **Carlepe** | -17.962 | -70.948 | Mixto | 1.10 | Alternado |
| **Los Palos** | -18.052 | -70.883 | Mixto | 1.10 | Disperso |

---

## 6. Mejores Practicas de Visualizacion

### 6.1 Colormaps Recomendados (cmocean)

| Tipo de Dato | Colormap | Descripcion |
|--------------|----------|-------------|
| SST | `thermal` | Secuencial, frio a calido |
| Clorofila-a | `algae` | Secuencial, verde |
| Anomalias | `balance` | Divergente |
| Salinidad | `haline` | Secuencial |
| Velocidad | `speed` | Secuencial |
| Batimetria | `deep` | Secuencial |

### 6.2 Visualizacion de Flujo

**Tecnicas de NASA:**
1. **Particle Trails**: Particulas virtuales con estelas
2. **Color por Temperatura**: Lineas coloreadas por SST
3. **Arrow Fields**: Flechas tradicionales
4. **Streamlines**: Lineas continuas siguiendo flujo

### 6.3 Accesibilidad

- **7%** de hombres tienen daltonismo rojo-verde
- Evitar rainbow/jet colormaps
- Usar colormaps perceptualmente uniformes

---

## 7. Algoritmo Recomendado para PFZ

```
1. RECOLECCION DE DATOS:
   - SST: Open-Meteo Marine API (tiempo real) o MODIS-Aqua (diario, 1km)
   - Chl-a: MODIS-Aqua (diario, 1km) o VIIRS
   - Corrientes: Open-Meteo Marine o HYCOM
   - SSH: Datos altimetria

2. DETECCION DE FRENTES (Cayula-Cornillon SIED):
   - Aplicar ventanas 32x32 pixeles
   - Umbral: diferencia temperatura > 0.45°C
   - Ratio varianza > 0.76

3. CLASIFICACION PFZ:
   SI (14°C < SST < 24°C) Y
      (Chl-a > 2.0 mg/m³) Y
      (frente_termico_presente) Y
      (condiciones_seguras):
      ENTONCES: zona de pesca ALTA probabilidad

4. FACTORES TEMPORALES:
   - Ajustar umbrales estacionalmente
   - Aplicar correccion El Nino cuando SOI indica evento
   - Ponderar datos recientes mas que historicos

5. VISUALIZACION:
   - Usar cmocean 'thermal' para SST
   - Usar cmocean 'algae' para Chl-a
   - Superponer contornos de frentes
   - Mostrar probabilidad PFZ como overlay
```

---

## 8. Fuentes Principales

### Papers y Articulos
- Belkin, I.M., O'Reilly, J.E. (2009) - Algoritmo BOA para deteccion de frentes
- Thyng et al. (2016) - "True Colors of Oceanography" - Colormaps cientificos
- Nature Communications - Upwelling y produccion biologica
- Royal Society - Patrones espaciales de anchoveta
- Chavez et al. (2008) - Climate and fisheries in the Humboldt Current

### APIs y Herramientas
- [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api)
- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [NOAA CoastWatch ERDDAP](https://coastwatch.noaa.gov/)
- [Global Fishing Watch APIs](https://globalfishingwatch.org/our-apis/)
- [NASA PO.DAAC](https://podaac.jpl.nasa.gov/)

### Instituciones
- **IMARPE** (Peru): Instituto del Mar del Peru - encuestas hidroacusticas
- **INCOIS** (India): Sistema operacional de PFZ
- **NOAA** (USA): Datos SST, corrientes, clorofila
- **ESA/Copernicus** (EU): Sentinel-3, datos marinos

---

## 9. Librerias Python Recomendadas

```python
# Acceso a datos
pip install erddapy          # Servidores ERDDAP
pip install copernicusmarine # Copernicus Marine
pip install earthaccess      # NASA data
pip install xarray netCDF4   # Manejo NetCDF

# Visualizacion
pip install folium           # Mapas interactivos
pip install cmocean          # Colormaps oceanograficos
pip install cartopy          # Proyecciones cartograficas
pip install pyvista          # Visualizacion 3D

# Machine Learning
pip install scikit-learn     # ML clasico
pip install tensorflow       # Deep learning

# Procesamiento
pip install numpy pandas scipy
```

---

*Documento actualizado: 2026-01-28*
*Proyecto: Fishing Predictor - Tacna/Ilo, Peru*
