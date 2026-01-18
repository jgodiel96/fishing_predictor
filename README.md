# Fishing Predictor - Sistema de Prediccion de Pesca desde Orilla

Sistema de prediccion de puntos optimos de pesca desde orilla para la costa sur de Peru (Tacna - Ilo).

## Caracteristicas

- **Linea costera real**: 3,400+ puntos de OpenStreetMap
- **Clima en tiempo real**: Temperatura, viento, condiciones via Open-Meteo
- **Datos solunares**: Fase lunar, mejores horarios de pesca
- **Recomendaciones de especies**: Segun tipo de sustrato (roca/arena/mixto)
- **Mapas interactivos**: Visualizacion con Folium

## Estructura del Proyecto

```
fishing_predictor/
├── analisis_real.py           # Script principal de analisis
├── config.py                  # Configuracion y constantes
├── environment.yml            # Entorno conda
├── fishing_analysis.ipynb     # Notebook interactivo
├── README.md
├── core/
│   ├── coastline_real.py      # Procesador de costa OSM
│   └── weather_solunar.py     # Clima y calculos solunares
├── data/
│   ├── data_manager.py        # Gestion de datos
│   └── cache/
│       └── coastline_real_osm.geojson  # Costa real (3,400+ puntos)
└── output/
    └── analisis_costa_real.html  # Mapa generado
```

## Instalacion

1. Crear entorno conda:
```bash
conda env create -f environment.yml
conda activate fishing_predictor
```

2. Instalar JupyterLab (opcional):
```bash
pip install jupyterlab
python -m ipykernel install --user --name fishing_predictor --display-name "Python (fishing_predictor)"
```

## Uso

### Script de Analisis

```bash
python analisis_real.py
```

Genera:
- Analisis de mejores puntos de pesca
- Condiciones climaticas actuales
- Fase lunar y mejores horarios
- Mapa interactivo en `output/analisis_costa_real.html`

### Notebook Interactivo

```bash
jupyter lab fishing_analysis.ipynb
```

Permite:
- Visualizar la linea costera
- Analizar puntos especificos
- Ver transectos perpendiculares al mar
- Obtener condiciones en tiempo real

## Ejemplo de Salida

```
TOP 5 MEJORES PUNTOS DE PESCA
==============================

⭐ #1 - Score: 82.4/100
   Coordenadas: -17.663543, -71.357815
   Distancia a peces: 173m
   Especies: Cabrilla, Pintadilla, Robalo

🌡️ Temperatura: 23.3°C
💨 Viento: 7.9 km/h
🌙 Luna: Luna Nueva (0%)
⭐ Rating del dia: 100/100

MEJORES HORARIOS:
   Amanecer: 06:14 | Atardecer: 17:45
```

## APIs Utilizadas

| API | Uso |
|-----|-----|
| OpenStreetMap Overpass | Linea costera real |
| Open-Meteo | Clima en tiempo real |
| Calculos locales | Datos solunares |

## Especies por Tipo de Sustrato

| Sustrato | Especies | Senuelos |
|----------|----------|----------|
| Roca | Cabrilla, Pintadilla, Robalo, Cherlo | Grubs, jigs 15-25g, poppers |
| Arena | Corvina, Lenguado, Pejerrey, Chita | Jigs metalicos, vinilos paddle |
| Mixto | Corvina, Cabrilla, Robalo, Pejerrey | Variado |

## Requisitos

- Python 3.11+
- Dependencias en `environment.yml`:
  - numpy, pandas
  - folium (mapas)
  - requests (APIs)
  - scipy (interpolacion)

## Autor

Desarrollado para pesca con spinning desde orilla en la costa sur de Peru.
