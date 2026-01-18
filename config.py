"""
Configuracion y constantes del sistema Fishing Spot Predictor.
"""

from dataclasses import dataclass
from typing import Dict
import os

# =============================================================================
# GEOGRAPHIC CONFIGURATION
# =============================================================================

# BBOX ajustado para pesca desde orilla (franja costera)
BBOX = {
    "north": -17.50,
    "south": -18.25,
    "west": -71.40,
    "east": -70.55  # Extendido para incluir zona sur de Tacna
}

# Puntos de referencia para pesca - EN LA LINEA DE COSTA
# Coordenadas ajustadas a la orilla/roca donde se pesca con spinning
# Tipos de sustrato: "roca", "arena", "mixto"
LOCATIONS: Dict[str, Dict] = {
    # Punto de analisis especifico del usuario
    "Playa Usuario": {
        "lat": -18.21437, "lon": -70.57990,
        "tipo": "arena",
        "descripcion": "Punto de pesca personalizado"
    },
    # Tacna - Sur (de sur a norte)
    "Boca del Rio": {
        "lat": -18.1205, "lon": -70.8430,
        "tipo": "arena",
        "descripcion": "Desembocadura, playa arenosa"
    },
    "Playa Santa Rosa": {
        "lat": -18.0870, "lon": -70.8680,
        "tipo": "arena",
        "descripcion": "Playa extensa, buena para corvina"
    },
    "Los Palos": {
        "lat": -18.0520, "lon": -70.8830,
        "tipo": "mixto",
        "descripcion": "Arena con rocas dispersas"
    },
    "Vila Vila": {
        "lat": -18.0180, "lon": -70.9120,
        "tipo": "roca",
        "descripcion": "Zona rocosa, buena estructura"
    },
    "Punta Mesa": {
        "lat": -17.9880, "lon": -70.9350,
        "tipo": "roca",
        "descripcion": "Punta rocosa con pozas"
    },
    "Carlepe": {
        "lat": -17.9620, "lon": -70.9480,
        "tipo": "mixto",
        "descripcion": "Rocas y arena alternadas"
    },
    # Ite
    "Playa Ite Norte": {
        "lat": -17.9320, "lon": -70.9680,
        "tipo": "arena",
        "descripcion": "Playa amplia, surgencia activa"
    },
    "Ite Centro": {
        "lat": -17.9020, "lon": -70.9920,
        "tipo": "mixto",
        "descripcion": "Zona mixta productiva"
    },
    "Ite Sur": {
        "lat": -17.8720, "lon": -71.0180,
        "tipo": "roca",
        "descripcion": "Formaciones rocosas, cabrilla"
    },
    # Zona Intermedia (Gentillar - Punta Blanca)
    "Gentillar": {
        "lat": -17.8420, "lon": -71.0480,
        "tipo": "roca",
        "descripcion": "Costa rocosa escarpada"
    },
    "Punta Blanca": {
        "lat": -17.8120, "lon": -71.0820,
        "tipo": "roca",
        "descripcion": "Punta rocosa, buena para robalo"
    },
    "Pozo Redondo": {
        "lat": -17.7820, "lon": -71.1220,
        "tipo": "mixto",
        "descripcion": "Pozas naturales entre rocas"
    },
    "Fundicion": {
        "lat": -17.7570, "lon": -71.1720,
        "tipo": "roca",
        "descripcion": "Rocas grandes, estructura compleja"
    },
    "Playa Media Luna": {
        "lat": -17.7320, "lon": -71.2220,
        "tipo": "arena",
        "descripcion": "Bahia arenosa en forma de media luna"
    },
    # Moquegua - Ilo
    "Punta Coles": {
        "lat": -17.7020, "lon": -71.3320,
        "tipo": "roca",
        "descripcion": "Reserva, rocas con mucha vida marina"
    },
    "Pocoma": {
        "lat": -17.6820, "lon": -71.2950,
        "tipo": "roca",
        "descripcion": "Acantilados rocosos"
    },
    "Ilo - Pozo Lizas": {
        "lat": -17.6420, "lon": -71.3400,
        "tipo": "roca",
        "descripcion": "Pozas entre rocas, ideal spinning"
    },
    "Ilo - Puerto": {
        "lat": -17.6320, "lon": -71.3450,
        "tipo": "mixto",
        "descripcion": "Muelle y rocas adyacentes"
    },
}

# Centro del mapa para visualizacion
MAP_CENTER = {"lat": -17.85, "lon": -71.15}

# =============================================================================
# GRID CONFIGURATION
# =============================================================================

GRID_RESOLUTION_M = 500  # metros
GRID_RESOLUTION_DEG = GRID_RESOLUTION_M / 111000  # aproximacion en grados

# =============================================================================
# THRESHOLDS
# =============================================================================

@dataclass
class Thresholds:
    """Umbrales para deteccion y clasificacion."""
    # Frentes termicos
    SST_GRADIENT_THRESHOLD: float = 0.5  # C/km para deteccion de frente

    # Productividad
    CHL_HIGH_THRESHOLD: float = 2.0  # mg/m3 para alta productividad
    CHL_LOW_THRESHOLD: float = 0.5  # mg/m3 para baja productividad

    # Seguridad
    WAVE_SAFETY_THRESHOLD: float = 2.0  # metros max para pesca desde orilla
    WIND_SAFETY_THRESHOLD: float = 25.0  # km/h max

    # Distancia a costa
    MAX_DISTANCE_FROM_COAST_KM: float = 10.0  # km maximo de interes

THRESHOLDS = Thresholds()

# =============================================================================
# SCORING WEIGHTS
# =============================================================================

SCORING_WEIGHTS = {
    "front_proximity": 0.25,
    "chlorophyll_score": 0.20,
    "upwelling_index": 0.15,
    "fishing_vessel_proxy": 0.15,
    "golden_hour": 0.10,
    "safety_score": 0.10,
    "lunar_score": 0.05,
}

# Penalizacion por condiciones inseguras
SAFETY_PENALTY_MULTIPLIER = 0.3
SAFETY_THRESHOLD = 0.5

# =============================================================================
# API ENDPOINTS
# =============================================================================

class APIEndpoints:
    """URLs de APIs externas."""
    # ERDDAP
    ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap"

    # Open-Meteo
    OPENMETEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
    OPENMETEO_WEATHER = "https://api.open-meteo.com/v1/forecast"

    # Global Fishing Watch (requiere API key)
    GFW_API = "https://gateway.api.globalfishingwatch.org"

ENDPOINTS = APIEndpoints()

# =============================================================================
# ERDDAP DATASETS
# =============================================================================

ERDDAP_DATASETS = {
    "sst_daily": "jplMURSST41",
    "sst_8day": "erdMH1sstd8day",
    "chlorophyll_daily": "nesdisVHNSQchlaDaily",
    "chlorophyll_8day": "erdMH1chla8day",
}

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 6

# =============================================================================
# VISUALIZATION
# =============================================================================

# Escala de colores para scoring
SCORE_COLORS = {
    "poor": "#ff4444",        # 0-20: Rojo
    "below_avg": "#ff8c00",   # 20-40: Naranja
    "average": "#ffff00",     # 40-60: Amarillo
    "good": "#90ee90",        # 60-80: Verde claro
    "excellent": "#228b22",   # 80-100: Verde oscuro
}

def get_score_color(score: float) -> str:
    """Retorna el color correspondiente al score."""
    if score < 20:
        return SCORE_COLORS["poor"]
    elif score < 40:
        return SCORE_COLORS["below_avg"]
    elif score < 60:
        return SCORE_COLORS["average"]
    elif score < 80:
        return SCORE_COLORS["good"]
    else:
        return SCORE_COLORS["excellent"]

def get_score_category(score: float) -> str:
    """Retorna la categoria del score."""
    if score < 20:
        return "Pobre"
    elif score < 40:
        return "Bajo promedio"
    elif score < 60:
        return "Promedio"
    elif score < 80:
        return "Bueno"
    else:
        return "Excelente"

# =============================================================================
# TIME CONFIGURATION
# =============================================================================

PREDICTION_HORIZONS = [0, 24, 48, 72]  # horas
TIMEZONE = "America/Lima"

# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_OUTPUT_FILE = "fishing_map.html"
