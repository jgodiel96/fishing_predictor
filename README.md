# Fishing Predictor - Sistema de Prediccion de Pesca desde Orilla

Sistema avanzado de prediccion de puntos optimos de pesca desde orilla para la costa sur de Peru (Tacna - Ilo).

## Caracteristicas Principales

### Datos 100% Reales
- **SST Satelital**: NOAA NCEI OISST (datos de satelite reales)
- **Condiciones Marinas**: Open-Meteo ERA5 (reanálisis real)
- **Actividad Pesquera**: Global Fishing Watch (datos AIS reales)
- **Zonas Historicas**: IMARPE (Instituto del Mar del Peru)

### Machine Learning Avanzado
- **32 Features Oceanograficos**: SST, frentes termicos, corrientes, olas, upwelling, etc.
- **PCA**: Reduccion de dimensionalidad para analisis de componentes principales
- **Gradient Boosting**: Regresion para prediccion de zonas de pesca
- **Modo Supervisado**: Entrenamiento con datos reales de pesca de GFW

### Visualizacion
- **Linea costera real**: 3,400+ puntos de OpenStreetMap
- **Mapas interactivos**: Folium con capas de SST, corrientes y zonas de pesca
- **Vectores de corriente**: Visualizacion de flujo oceanico

## Estructura del Proyecto

```
fishing_predictor/
├── main.py                    # Punto de entrada principal
├── analisis_real.py           # Script de analisis interactivo
├── config.py                  # Configuracion y constantes
├── environment.yml            # Entorno conda
│
├── models/
│   ├── features.py            # Extractor de 32 features oceanograficos
│   └── predictor.py           # ML: PCA + Gradient Boosting
│
├── controllers/
│   └── analysis.py            # Controlador de analisis
│
├── views/
│   └── map_generator.py       # Generador de mapas Folium
│
├── core/
│   ├── coastline_real.py      # Procesador de costa OSM
│   ├── weather_solunar.py     # Clima y calculos solunares
│   └── marine_data.py         # Datos marinos y frentes termicos
│
├── data/
│   ├── fetchers/
│   │   ├── real_data_only.py  # Fetcher de datos 100% reales
│   │   └── historical_fetcher.py
│   └── cache/                 # Cache de datos
│
├── scripts/
│   └── download_100_real.py   # Descarga datos reales (requiere GFW API)
│
├── tests/                     # 18 tests unitarios
│   └── test_models.py
│
└── output/                    # Mapas y reportes generados
```

## Instalacion

### 1. Crear entorno conda

```bash
conda env create -f environment.yml
conda activate fishing_predictor
```

### 2. Obtener API Key de Global Fishing Watch (Opcional pero recomendado)

Para datos de pesca reales:

1. Ve a: https://globalfishingwatch.org/our-apis/
2. Click en "Request API Access"
3. Completa el formulario (gratis para investigacion)
4. Configura la API key:

```bash
export GFW_API_KEY='tu_api_key_aqui'
```

### 3. Descargar datos reales

```bash
python scripts/download_100_real.py --months 6
```

## Uso

### Analisis Rapido (sin datos historicos)

```bash
python main.py
```

### Analisis con ML Supervisado (requiere datos historicos)

```bash
python main.py --supervised
```

### Notebook Interactivo

```bash
jupyter lab fishing_analysis.ipynb
```

## Fuentes de Datos Reales

| Tipo de Dato | Fuente | Descripcion |
|--------------|--------|-------------|
| SST | NOAA NCEI OISST | Temperatura superficial del mar por satelite |
| Olas/Viento | Open-Meteo ERA5 | Reanalisis ECMWF (datos reales + modelo) |
| Pesca | Global Fishing Watch | Actividad pesquera AIS (requiere API key) |
| Zonas Historicas | IMARPE | Reportes del Instituto del Mar del Peru |

## Features del Modelo ML (32 total)

### SST (6 features)
- Temperatura, anomalia, score optimo, score por especie, variabilidad, tendencia

### Frentes Termicos (5 features)
- Magnitud del gradiente, direccion, deteccion de frente, intensidad

### Corrientes (6 features)
- Velocidad, componentes U/V, convergencia, cizalladura, direccion a costa

### Olas (3 features)
- Altura, periodo, indice favorable

### Upwelling (3 features)
- Indice de surgencia, transporte Ekman, indice favorable

### Espacial (4 features)
- Distancia a costa, profundidad, zona costera, zona offshore

### Historico (2 features)
- Distancia a hotspots conocidos, similitud con hotspots

### Temporal (3 features)
- Score por hora, fase lunar, estacionalidad

## Ejemplo de Salida

```
============================================================
FISHING PREDICTOR - Analisis Completo
============================================================
Region: Tacna-Ilo, Peru
Fecha: 2026-01-18

[OK] 3,412 puntos de costa cargados
[OK] 288 registros SST reales (NOAA NCEI)
[OK] 288 registros marinos reales (ERA5)

TOP 5 MEJORES ZONAS DE PESCA
==============================

#1 - Score: 87.3/100 | Confianza: 92%
    Coordenadas: -17.700, -71.350
    SST: 17.8C | Olas: 1.2m
    Causa: thermal_front + historical_imarpe (Punta Coles)
    Especies: Cabrilla, Robalo, Corvina

#2 - Score: 82.1/100 | Confianza: 88%
    ...
```

## Tests

```bash
python -m pytest tests/ -v
```

Resultado: 18 tests passed

## Requisitos

- Python 3.10+
- Dependencias principales:
  - numpy, pandas, scipy
  - scikit-learn (ML)
  - folium (mapas)
  - requests (APIs)
  - xarray (datos NetCDF)

## Investigacion Cientifica

El sistema esta basado en investigacion oceanografica del Sistema de la Corriente de Humboldt:

- **IMARPE**: Instituto del Mar del Peru
- **FAO**: Reportes de pesca para Peru
- **Chavez et al. (2008)**: Climate and fisheries in the Humboldt Current
- **pyBOA**: Algoritmo Belkin-O'Reilly para deteccion de frentes termicos

## Licencia

Desarrollado para pesca con spinning desde orilla en la costa sur de Peru.
