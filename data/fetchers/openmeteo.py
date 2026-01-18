"""
Fetcher para datos meteorologicos y maritimos de Open-Meteo.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json
import os
import hashlib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ENDPOINTS, LOCATIONS, CACHE_DIR, CACHE_TTL_HOURS


class OpenMeteoFetcher:
    """Obtiene datos meteorologicos y maritimos de Open-Meteo API."""

    def __init__(self, use_cache: bool = True):
        self.marine_url = ENDPOINTS.OPENMETEO_MARINE
        self.weather_url = ENDPOINTS.OPENMETEO_WEATHER
        self.use_cache = use_cache
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Crea el directorio de cache si no existe."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def _get_cache_path(self, cache_key: str) -> str:
        """Genera path de archivo cache."""
        hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return os.path.join(CACHE_DIR, f"meteo_{hash_key}.json")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Carga datos del cache si existen y no han expirado."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=CACHE_TTL_HOURS):
                return None

            return cached["data"]
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Guarda datos en cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        cached = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with open(cache_path, "w") as f:
            json.dump(cached, f)

    def fetch_marine_data(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 3
    ) -> pd.DataFrame:
        """
        Obtiene datos maritimos (olas) para una ubicacion.

        Args:
            lat: Latitud
            lon: Longitud
            forecast_days: Dias de pronostico (1-7)

        Returns:
            DataFrame con datos horarios de olas
        """
        cache_key = f"marine_{lat:.3f}_{lon:.3f}_{forecast_days}"
        cached = self._load_from_cache(cache_key)

        if cached is not None:
            df = pd.DataFrame(cached)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            return df

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "wave_height",
                "wave_period",
                "wave_direction",
                "wind_wave_height",
                "swell_wave_height",
                "swell_wave_period",
            ],
            "forecast_days": forecast_days
        }

        try:
            response = requests.get(self.marine_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                return self._get_empty_marine_df()

            hourly = data["hourly"]
            df = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"]),
                "wave_height": hourly.get("wave_height", []),
                "wave_period": hourly.get("wave_period", []),
                "wave_direction": hourly.get("wave_direction", []),
                "wind_wave_height": hourly.get("wind_wave_height", []),
                "swell_wave_height": hourly.get("swell_wave_height", []),
                "swell_wave_period": hourly.get("swell_wave_period", []),
            })

            df["latitude"] = lat
            df["longitude"] = lon

            # Convertir time a string para serializar a JSON
            cache_data = df.copy()
            cache_data["time"] = cache_data["time"].astype(str)
            self._save_to_cache(cache_key, cache_data.to_dict(orient="records"))
            return df

        except requests.RequestException as e:
            print(f"Error fetching marine data: {e}")
            return self._get_empty_marine_df()

    def fetch_weather_data(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 3
    ) -> pd.DataFrame:
        """
        Obtiene datos meteorologicos para una ubicacion.

        Args:
            lat: Latitud
            lon: Longitud
            forecast_days: Dias de pronostico (1-7)

        Returns:
            DataFrame con datos horarios meteorologicos
        """
        cache_key = f"weather_{lat:.3f}_{lon:.3f}_{forecast_days}"
        cached = self._load_from_cache(cache_key)

        if cached is not None:
            df = pd.DataFrame(cached)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            return df

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "pressure_msl",
                "precipitation",
                "cloud_cover",
            ],
            "forecast_days": forecast_days
        }

        try:
            response = requests.get(self.weather_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                return self._get_empty_weather_df()

            hourly = data["hourly"]
            df = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"]),
                "temperature": hourly.get("temperature_2m", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
                "wind_direction": hourly.get("wind_direction_10m", []),
                "pressure": hourly.get("pressure_msl", []),
                "precipitation": hourly.get("precipitation", []),
                "cloud_cover": hourly.get("cloud_cover", []),
            })

            df["latitude"] = lat
            df["longitude"] = lon

            # Convertir time a string para serializar a JSON
            cache_data = df.copy()
            cache_data["time"] = cache_data["time"].astype(str)
            self._save_to_cache(cache_key, cache_data.to_dict(orient="records"))
            return df

        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return self._get_empty_weather_df()

    def fetch_for_locations(
        self,
        locations: Optional[Dict[str, Dict]] = None,
        forecast_days: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Obtiene datos para multiples ubicaciones.

        Args:
            locations: Dict de ubicaciones {nombre: {lat, lon}}
            forecast_days: Dias de pronostico

        Returns:
            Tupla de (marine_df, weather_df) con datos combinados
        """
        if locations is None:
            locations = LOCATIONS

        marine_dfs = []
        weather_dfs = []

        for name, coords in locations.items():
            lat = coords["lat"]
            lon = coords["lon"]

            marine_df = self.fetch_marine_data(lat, lon, forecast_days)
            marine_df["location"] = name
            marine_dfs.append(marine_df)

            weather_df = self.fetch_weather_data(lat, lon, forecast_days)
            weather_df["location"] = name
            weather_dfs.append(weather_df)

        combined_marine = pd.concat(marine_dfs, ignore_index=True) if marine_dfs else self._get_empty_marine_df()
        combined_weather = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else self._get_empty_weather_df()

        return combined_marine, combined_weather

    def get_current_conditions(
        self,
        lat: float,
        lon: float
    ) -> Dict:
        """
        Obtiene condiciones actuales para una ubicacion.

        Returns:
            Dict con condiciones actuales
        """
        marine_df = self.fetch_marine_data(lat, lon, forecast_days=1)
        weather_df = self.fetch_weather_data(lat, lon, forecast_days=1)

        now = datetime.now()

        # Encontrar el registro mas cercano a ahora
        conditions = {
            "timestamp": now.isoformat(),
            "latitude": lat,
            "longitude": lon,
        }

        if not marine_df.empty:
            marine_df["time_diff"] = abs(marine_df["time"] - pd.Timestamp(now))
            closest_marine = marine_df.loc[marine_df["time_diff"].idxmin()]
            conditions.update({
                "wave_height": closest_marine.get("wave_height"),
                "wave_period": closest_marine.get("wave_period"),
                "wave_direction": closest_marine.get("wave_direction"),
            })

        if not weather_df.empty:
            weather_df["time_diff"] = abs(weather_df["time"] - pd.Timestamp(now))
            closest_weather = weather_df.loc[weather_df["time_diff"].idxmin()]
            conditions.update({
                "temperature": closest_weather.get("temperature"),
                "wind_speed": closest_weather.get("wind_speed"),
                "wind_direction": closest_weather.get("wind_direction"),
                "pressure": closest_weather.get("pressure"),
            })

        return conditions

    def _get_empty_marine_df(self) -> pd.DataFrame:
        """Retorna DataFrame vacio con estructura correcta para datos marinos."""
        return pd.DataFrame(columns=[
            "time", "wave_height", "wave_period", "wave_direction",
            "wind_wave_height", "swell_wave_height", "swell_wave_period",
            "latitude", "longitude"
        ])

    def _get_empty_weather_df(self) -> pd.DataFrame:
        """Retorna DataFrame vacio con estructura correcta para datos meteorologicos."""
        return pd.DataFrame(columns=[
            "time", "temperature", "wind_speed", "wind_direction",
            "pressure", "precipitation", "cloud_cover",
            "latitude", "longitude"
        ])
