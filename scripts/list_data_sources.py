#!/usr/bin/env python3
"""
Lista todas las fuentes de datos disponibles para el sistema de predicción de pesca.

Muestra:
- Fuentes configuradas
- Variables disponibles
- Estado de descarga
- Recomendaciones
"""

from pathlib import Path
import os

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

print("=" * 70)
print("FUENTES DE DATOS PARA FISHING PREDICTOR V8")
print("=" * 70)

# =============================================================================
# 1. OSM DATA (osmdata.openstreetmap.de)
# =============================================================================
print("\n" + "=" * 70)
print("1. OPENSTREETMAP DATA")
print("   URL: https://osmdata.openstreetmap.de/data/coastlines.html")
print("=" * 70)

osm_dir = DATA_DIR / "coastlines"

osm_datasets = {
    "coastlines-split-4326": {
        "description": "Líneas de costa (LineString)",
        "file": "coastlines-split-4326/lines.shp",
        "size": "~500MB",
        "use": "Definición de línea costera"
    },
    "water-polygons-split-4326": {
        "description": "Polígonos de agua (Polygon)",
        "file": "water-polygons-split-4326/water_polygons.shp",
        "size": "~700MB",
        "use": "Clasificación agua/tierra SIN suposiciones geográficas"
    },
    "land-polygons-split-4326": {
        "description": "Polígonos de tierra (Polygon)",
        "file": "land-polygons-split-4326/land_polygons.shp",
        "size": "~600MB",
        "use": "Alternativa a water-polygons"
    }
}

for name, info in osm_datasets.items():
    filepath = osm_dir / info["file"]
    status = "✅ DESCARGADO" if filepath.exists() else "❌ NO DESCARGADO"
    zippath = osm_dir / f"{name}.zip"
    if zippath.exists() and not filepath.exists():
        status = "📦 ZIP presente (extraer)"

    print(f"\n  {name}")
    print(f"    Descripción: {info['description']}")
    print(f"    Tamaño: {info['size']}")
    print(f"    Uso: {info['use']}")
    print(f"    Estado: {status}")

# =============================================================================
# 2. COPERNICUS MARINE
# =============================================================================
print("\n" + "=" * 70)
print("2. COPERNICUS MARINE SERVICE")
print("   URL: https://marine.copernicus.eu/")
print("=" * 70)

copernicus_datasets = {
    "GLOBAL_ANALYSISFORECAST_PHY": {
        "id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        "variables": {
            "thetao": ("Temperatura del agua", "°C", "Distribución de especies"),
            "so": ("Salinidad", "PSU", "Masas de agua, frentes"),
            "zos": ("Altura superficie mar (SLA)", "m", "Upwelling, corrientes"),
            "uo": ("Corriente Este", "m/s", "Transporte de larvas"),
            "vo": ("Corriente Norte", "m/s", "Transporte de larvas"),
            "mlotst": ("Profundidad capa mezcla", "m", "Estratificación"),
        },
        "resolution": "0.083° (~9km)",
        "temporal": "Diario"
    },
    "GLOBAL_OCEAN_COLOUR": {
        "id": "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        "variables": {
            "CHL": ("Clorofila-a", "mg/m³", "Productividad primaria"),
            "KD490": ("Coef. atenuación", "m⁻¹", "Claridad del agua"),
            "SPM": ("Material suspendido", "g/m³", "Turbidez"),
        },
        "resolution": "4km",
        "temporal": "Diario"
    },
    "GLOBAL_WAVES": {
        "id": "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        "variables": {
            "VHM0": ("Altura ola significativa", "m", "Condiciones navegación"),
            "VTPK": ("Período pico ola", "s", "Tipo de oleaje"),
            "VMDR": ("Dirección ola", "°", "Planificación salida"),
        },
        "resolution": "0.083°",
        "temporal": "3 horas"
    },
    "GLOBAL_WIND": {
        "id": "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        "variables": {
            "eastward_wind": ("Viento Este", "m/s", "Afecta oleaje y pesca"),
            "northward_wind": ("Viento Norte", "m/s", "Afecta oleaje y pesca"),
        },
        "resolution": "0.125°",
        "temporal": "Horario"
    }
}

raw_dir = DATA_DIR / "raw"

for name, info in copernicus_datasets.items():
    print(f"\n  {name}")
    print(f"    Dataset ID: {info['id']}")
    print(f"    Resolución: {info['resolution']} | {info['temporal']}")
    print(f"    Variables:")

    for var, (desc, unit, use) in info["variables"].items():
        # Check if data exists
        var_dir = raw_dir / var.lower() / "copernicus"
        if var_dir.exists() and list(var_dir.glob("*.parquet")):
            status = f"✅ {len(list(var_dir.glob('*.parquet')))} archivos"
        else:
            status = "❌ No descargado"

        print(f"      {var:12} | {desc:25} | {unit:8} | {use}")
        print(f"                  Estado: {status}")

# =============================================================================
# 3. GEBCO BATHYMETRY
# =============================================================================
print("\n" + "=" * 70)
print("3. GEBCO - General Bathymetric Chart of the Oceans")
print("   URL: https://www.gebco.net/data_and_products/gridded_bathymetry_data/")
print("=" * 70)

gebco_info = {
    "GEBCO_2023": {
        "description": "Batimetría global 15 arc-second (~450m)",
        "file": "GEBCO_2023.nc",
        "size": "~7.5GB (global) o subset regional",
        "variables": {
            "elevation": ("Elevación/Profundidad", "m", "Profundidad del fondo marino")
        },
        "use": "Profundidad real para clasificación de hábitat"
    }
}

for name, info in gebco_info.items():
    print(f"\n  {name}")
    print(f"    Descripción: {info['description']}")
    print(f"    Tamaño: {info['size']}")
    print(f"    Uso: {info['use']}")

    # Check if file exists
    gebco_file = DATA_DIR / "bathymetry" / info["file"]
    if gebco_file.exists():
        print(f"    Estado: ✅ DESCARGADO")
    else:
        print(f"    Estado: ❌ NO DESCARGADO")
        print(f"    Descarga: Manual desde GEBCO website o usar subset regional")

# =============================================================================
# 4. ADDITIONAL DATA SOURCES
# =============================================================================
print("\n" + "=" * 70)
print("4. FUENTES ADICIONALES RECOMENDADAS")
print("=" * 70)

additional = {
    "IMARPE": {
        "url": "https://www.imarpe.gob.pe/",
        "data": ["Datos de desembarque", "Cruceros oceanográficos", "Estadísticas pesqueras"],
        "status": "Requiere solicitud"
    },
    "NOAA ERDDAP": {
        "url": "https://coastwatch.pfeg.noaa.gov/erddap/",
        "data": ["SST alta resolución", "Clorofila MODIS/VIIRS", "Vientos"],
        "status": "API gratuita"
    },
    "Copernicus Climate (ERA5)": {
        "url": "https://cds.climate.copernicus.eu/",
        "data": ["Viento 10m", "Presión atmosférica", "Precipitación"],
        "status": "Requiere registro"
    }
}

for name, info in additional.items():
    print(f"\n  {name}")
    print(f"    URL: {info['url']}")
    print(f"    Datos: {', '.join(info['data'])}")
    print(f"    Estado: {info['status']}")

# =============================================================================
# RESUMEN Y RECOMENDACIONES
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN Y RECOMENDACIONES")
print("=" * 70)

print("""
PRIORIDAD ALTA (descargar ahora):
  1. ✅ water-polygons-split-4326 - Clasificación agua/tierra correcta
  2. 🔄 GEBCO subset regional - Profundidad real del fondo

PRIORIDAD MEDIA (útiles para mejorar predicciones):
  3. Corrientes (uo/vo) - Transporte de larvas y nutrientes
  4. Olas (VHM0) - Condiciones de navegación
  5. KD490 - Claridad del agua

PRIORIDAD BAJA (nice-to-have):
  6. Viento - Ya disponible en otras fuentes
  7. SPM - Correlacionado con otros indicadores

COMANDOS DE DESCARGA:

# Water polygons OSM (ejecutando ahora):
curl -L -o water-polygons-split-4326.zip \\
    "https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip"

# GEBCO subset (manual, requiere seleccionar región):
# Ir a: https://download.gebco.net/
# Seleccionar región: Perú (-20, -15, -82, -70)
# Descargar NetCDF

# Copernicus (requiere credenciales en .env):
python -c "from data.fetchers.copernicus_physics_fetcher import download_all_physics; download_all_physics()"
""")
