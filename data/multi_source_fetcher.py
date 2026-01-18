"""
Fetcher multi-fuente para datos oceanográficos.
Intenta múltiples APIs y usa la primera que funcione.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from io import StringIO
import json
import time


class MultiSourceFetcher:
    """
    Obtiene datos oceanográficos de múltiples fuentes.
    Orden de prioridad:
    1. ERDDAP CoastWatch (NOAA)
    2. Open-Meteo Marine API
    3. Copernicus Marine Service
    4. Datos simulados realistas (último recurso)
    """

    # BBOX para costa sur de Perú
    BBOX = {
        "north": -17.50,
        "south": -18.25,
        "west": -71.45,
        "east": -70.50
    }

    # URLs de APIs
    ERDDAP_SERVERS = [
        "https://coastwatch.pfeg.noaa.gov/erddap",
        "https://upwell.pfeg.noaa.gov/erddap",
        "https://polarwatch.noaa.gov/erddap",
    ]

    OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.last_source = None

    def fetch_sst(
        self,
        lat: float = None,
        lon: float = None,
        date: datetime = None
    ) -> Dict:
        """
        Obtiene SST de la mejor fuente disponible.

        Returns:
            Dict con 'sst', 'source', 'timestamp'
        """
        lat = lat or (self.BBOX['north'] + self.BBOX['south']) / 2
        lon = lon or (self.BBOX['west'] + self.BBOX['east']) / 2
        date = date or datetime.now()

        # Intentar Open-Meteo primero (más confiable)
        result = self._fetch_open_meteo_sst(lat, lon)
        if result:
            return result

        # Intentar ERDDAP
        result = self._fetch_erddap_sst(lat, lon, date)
        if result:
            return result

        # Último recurso: estimación basada en climatología
        return self._estimate_sst(lat, lon, date)

    def fetch_marine_conditions(
        self,
        lat: float,
        lon: float
    ) -> Dict:
        """
        Obtiene condiciones marinas completas (olas, corrientes, SST).
        """
        url = f"{self.OPEN_METEO_MARINE}"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "wave_height,wave_direction,wave_period,ocean_current_velocity,ocean_current_direction",
            "hourly": "wave_height,wave_direction,wave_period,sea_water_temperature",
            "timezone": "America/Lima"
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                hourly = data.get("hourly", {})

                # Obtener SST de hourly si existe
                sst = None
                if "sea_water_temperature" in hourly:
                    temps = [t for t in hourly["sea_water_temperature"] if t is not None]
                    if temps:
                        sst = np.mean(temps[-24:])  # Promedio últimas 24h

                result = {
                    "wave_height": current.get("wave_height"),
                    "wave_direction": current.get("wave_direction"),
                    "wave_period": current.get("wave_period"),
                    "current_velocity": current.get("ocean_current_velocity"),
                    "current_direction": current.get("ocean_current_direction"),
                    "sst": sst,
                    "source": "open-meteo",
                    "timestamp": datetime.now().isoformat()
                }
                self.last_source = "open-meteo"
                return result
        except Exception as e:
            print(f"[WARN] Open-Meteo error: {e}")

        return None

    def _fetch_open_meteo_sst(self, lat: float, lon: float) -> Optional[Dict]:
        """Intenta obtener SST de Open-Meteo."""
        try:
            url = f"{self.OPEN_METEO_MARINE}"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "sea_water_temperature",
                "past_days": 7,
                "timezone": "America/Lima"
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})

                if "sea_water_temperature" in hourly:
                    temps = [t for t in hourly["sea_water_temperature"] if t is not None]
                    if temps:
                        # Promedio de los últimos valores disponibles
                        recent_temps = temps[-48:]  # últimas 48 horas
                        sst = np.mean(recent_temps)

                        self.last_source = "open-meteo"
                        return {
                            "sst": round(sst, 2),
                            "sst_min": round(min(recent_temps), 2),
                            "sst_max": round(max(recent_temps), 2),
                            "source": "open-meteo",
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"[WARN] Open-Meteo SST error: {e}")

        return None

    def _fetch_erddap_sst(
        self,
        lat: float,
        lon: float,
        date: datetime
    ) -> Optional[Dict]:
        """Intenta obtener SST de ERDDAP."""
        for server in self.ERDDAP_SERVERS:
            try:
                # MUR SST dataset
                dataset = "jplMURSST41"
                date_str = date.strftime("%Y-%m-%dT12:00:00Z")

                url = (
                    f"{server}/griddap/{dataset}.json?"
                    f"analysed_sst[({date_str})][({lat-0.1}):1:({lat+0.1})][({lon-0.1}):1:({lon+0.1})]"
                )

                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    rows = data.get("table", {}).get("rows", [])
                    if rows:
                        sst_values = [r[3] for r in rows if r[3] is not None]
                        if sst_values:
                            # Convertir de Kelvin si es necesario
                            sst = np.mean(sst_values)
                            if sst > 100:
                                sst -= 273.15

                            self.last_source = f"erddap-{server.split('//')[1].split('/')[0]}"
                            return {
                                "sst": round(sst, 2),
                                "source": self.last_source,
                                "timestamp": datetime.now().isoformat()
                            }
            except Exception as e:
                continue  # Intentar siguiente servidor

        return None

    def _estimate_sst(self, lat: float, lon: float, date: datetime) -> Dict:
        """
        Estima SST basado en climatología de la región.
        Costa sur de Perú: 14-18°C típicamente.
        """
        # Mes del año afecta la temperatura
        month = date.month

        # Temperatura base según mes (hemisferio sur)
        # Verano (Dic-Mar): más cálido, Invierno (Jun-Sep): más frío
        monthly_base = {
            1: 17.5, 2: 18.0, 3: 17.5, 4: 17.0,
            5: 16.0, 6: 15.5, 7: 15.0, 8: 15.0,
            9: 15.5, 10: 16.0, 11: 16.5, 12: 17.0
        }

        base_sst = monthly_base.get(month, 16.0)

        # Ajuste por latitud (más al sur = más frío)
        lat_adjustment = (lat + 18.0) * 0.3  # Aprox 0.3°C por 0.1° latitud

        # Ajuste por cercanía a la costa (surgencia = agua fría)
        # Longitudes más al este = más cerca de la costa
        coast_adjustment = (lon + 71.0) * 0.5

        sst = base_sst + lat_adjustment + coast_adjustment

        # Agregar variabilidad realista
        sst += np.random.normal(0, 0.3)
        sst = np.clip(sst, 14.0, 19.0)

        self.last_source = "climatology-estimate"
        return {
            "sst": round(sst, 2),
            "source": "climatology-estimate",
            "note": "Estimado basado en climatología regional",
            "timestamp": datetime.now().isoformat()
        }

    def fetch_chlorophyll(
        self,
        lat: float = None,
        lon: float = None,
        date: datetime = None
    ) -> Dict:
        """
        Obtiene clorofila-a.
        ERDDAP es la principal fuente, con fallback a estimación.
        """
        lat = lat or (self.BBOX['north'] + self.BBOX['south']) / 2
        lon = lon or (self.BBOX['west'] + self.BBOX['east']) / 2
        date = date or datetime.now()

        # Intentar ERDDAP
        result = self._fetch_erddap_chlorophyll(lat, lon, date)
        if result:
            return result

        # Fallback: estimación basada en región
        return self._estimate_chlorophyll(lat, lon, date)

    def _fetch_erddap_chlorophyll(
        self,
        lat: float,
        lon: float,
        date: datetime
    ) -> Optional[Dict]:
        """Intenta obtener clorofila de ERDDAP."""
        for server in self.ERDDAP_SERVERS:
            try:
                dataset = "erdMH1chla8day"
                date_str = date.strftime("%Y-%m-%dT00:00:00Z")

                url = (
                    f"{server}/griddap/{dataset}.json?"
                    f"chlorophyll[({date_str})][({lat-0.2}):1:({lat+0.2})][({lon-0.2}):1:({lon+0.2})]"
                )

                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    rows = data.get("table", {}).get("rows", [])
                    if rows:
                        chl_values = [r[3] for r in rows if r[3] is not None and r[3] > 0]
                        if chl_values:
                            self.last_source = f"erddap-{server.split('//')[1].split('/')[0]}"
                            return {
                                "chlorophyll": round(np.mean(chl_values), 2),
                                "source": self.last_source,
                                "timestamp": datetime.now().isoformat()
                            }
            except Exception:
                continue

        return None

    def _estimate_chlorophyll(self, lat: float, lon: float, date: datetime) -> Dict:
        """
        Estima clorofila basado en características de la región.
        Costa peruana: alta productividad por surgencia (2-15 mg/m³).
        """
        # Base de clorofila para costa peruana (alta productividad)
        base_chl = 4.0

        # Ajuste por cercanía a la costa (más cerca = más productividad)
        coast_factor = (lon + 71.0) * 2.0

        # Ajuste estacional
        month = date.month
        # Más productividad en invierno-primavera (surgencia más fuerte)
        seasonal = {
            1: 1.0, 2: 0.9, 3: 0.8, 4: 1.0,
            5: 1.2, 6: 1.3, 7: 1.4, 8: 1.5,
            9: 1.4, 10: 1.2, 11: 1.0, 12: 1.0
        }

        chl = base_chl * seasonal.get(month, 1.0) + coast_factor
        chl += np.random.normal(0, 0.5)
        chl = np.clip(chl, 0.5, 15.0)

        self.last_source = "productivity-estimate"
        return {
            "chlorophyll": round(chl, 2),
            "source": "productivity-estimate",
            "note": "Estimado basado en productividad regional",
            "timestamp": datetime.now().isoformat()
        }

    def fetch_complete_analysis(
        self,
        lat: float,
        lon: float
    ) -> Dict:
        """
        Obtiene análisis completo para un punto.
        """
        print(f"[INFO] Obteniendo datos para ({lat:.4f}, {lon:.4f})...")

        # Condiciones marinas (olas, corrientes)
        marine = self.fetch_marine_conditions(lat, lon) or {}

        # SST
        sst_data = self.fetch_sst(lat, lon)

        # Clorofila
        chl_data = self.fetch_chlorophyll(lat, lon)

        result = {
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.now().isoformat(),
            "sst": sst_data.get("sst"),
            "sst_source": sst_data.get("source"),
            "chlorophyll": chl_data.get("chlorophyll"),
            "chlorophyll_source": chl_data.get("source"),
            "wave_height": marine.get("wave_height"),
            "wave_direction": marine.get("wave_direction"),
            "wave_period": marine.get("wave_period"),
            "current_velocity": marine.get("current_velocity"),
            "current_direction": marine.get("current_direction"),
        }

        return result

    def fetch_transect_data(
        self,
        shore_point: Tuple[float, float],
        bearing: float,
        distance_m: float = 500,
        num_points: int = 5
    ) -> List[Dict]:
        """
        Obtiene datos a lo largo de un transecto desde la orilla hacia el mar.

        Args:
            shore_point: (lat, lon) del punto en la orilla
            bearing: dirección hacia el mar en grados
            distance_m: distancia total del transecto
            num_points: número de puntos a muestrear

        Returns:
            Lista de datos por punto del transecto
        """
        lat, lon = shore_point
        bearing_rad = np.radians(bearing)

        results = []

        for i in range(num_points):
            dist = (i / (num_points - 1)) * distance_m if num_points > 1 else 0

            # Calcular posición
            dlat = dist / 111000 * np.cos(bearing_rad)
            dlon = dist / (111000 * np.cos(np.radians(lat))) * np.sin(bearing_rad)

            point_lat = lat + dlat
            point_lon = lon + dlon

            # Obtener datos
            data = self.fetch_complete_analysis(point_lat, point_lon)
            data["distance_from_shore_m"] = dist
            data["transect_point"] = i + 1

            results.append(data)

            # Pequeña pausa para no sobrecargar APIs
            if i < num_points - 1:
                time.sleep(0.5)

        return results


def get_fetcher() -> MultiSourceFetcher:
    """Obtiene instancia del fetcher."""
    return MultiSourceFetcher()
