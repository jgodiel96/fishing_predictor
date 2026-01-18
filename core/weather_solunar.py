"""
Integración de datos meteorológicos y cálculos solunares.
Proporciona información de clima real y mejores horarios de pesca.
"""

import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class WeatherConditions:
    """Condiciones meteorológicas actuales."""
    temperature: float  # °C
    wind_speed: float   # km/h
    wind_direction: float  # grados
    wind_gusts: float   # km/h
    humidity: float     # %
    pressure: float     # hPa
    cloud_cover: float  # %
    precipitation: float  # mm

    # Evaluación para pesca
    is_safe: bool = True
    fishing_rating: str = "bueno"
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        self._evaluate()

    def _evaluate(self):
        """Evalúa condiciones para pesca."""
        self.warnings = []

        # Viento
        if self.wind_speed > 30:
            self.warnings.append(f"Viento muy fuerte: {self.wind_speed:.0f} km/h")
            self.is_safe = False
        elif self.wind_speed > 20:
            self.warnings.append(f"Viento fuerte: {self.wind_speed:.0f} km/h")

        # Rafagas
        if self.wind_gusts > 40:
            self.warnings.append(f"Ráfagas peligrosas: {self.wind_gusts:.0f} km/h")
            self.is_safe = False

        # Lluvia
        if self.precipitation > 5:
            self.warnings.append(f"Lluvia: {self.precipitation:.1f} mm")

        # Rating
        if not self.is_safe:
            self.fishing_rating = "peligroso"
        elif self.wind_speed > 20 or self.precipitation > 2:
            self.fishing_rating = "regular"
        elif self.wind_speed < 15 and self.precipitation < 1:
            self.fishing_rating = "excelente"
        else:
            self.fishing_rating = "bueno"


@dataclass
class SolunarData:
    """Datos solunares para pesca."""
    date: datetime

    # Sol
    sunrise: datetime
    sunset: datetime

    # Luna
    moonrise: Optional[datetime]
    moonset: Optional[datetime]
    moon_phase: float  # 0-1 (0=nueva, 0.5=llena)
    moon_phase_name: str
    moon_illumination: float  # %

    # Períodos de pesca
    major_periods: List[Tuple[datetime, datetime]]  # Mejores (2h)
    minor_periods: List[Tuple[datetime, datetime]]  # Buenos (1h)

    # Score general del día
    day_rating: float  # 0-100
    best_time: str


class WeatherFetcher:
    """
    Obtiene datos meteorológicos de Open-Meteo.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def get_current(self, lat: float, lon: float) -> Optional[WeatherConditions]:
        """
        Obtiene condiciones actuales.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "pressure_msl",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m"
            ],
            "timezone": "America/Lima"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})

                return WeatherConditions(
                    temperature=current.get("temperature_2m", 20),
                    wind_speed=current.get("wind_speed_10m", 0),
                    wind_direction=current.get("wind_direction_10m", 0),
                    wind_gusts=current.get("wind_gusts_10m", 0),
                    humidity=current.get("relative_humidity_2m", 50),
                    pressure=current.get("pressure_msl", 1013),
                    cloud_cover=current.get("cloud_cover", 0),
                    precipitation=current.get("precipitation", 0)
                )
        except Exception as e:
            print(f"[WARN] Error obteniendo clima: {e}")

        return None

    def get_forecast(self, lat: float, lon: float, days: int = 3) -> List[Dict]:
        """
        Obtiene pronóstico horario.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_gusts_10m"
            ],
            "forecast_days": days,
            "timezone": "America/Lima"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})

                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                precip = hourly.get("precipitation", [])
                wind = hourly.get("wind_speed_10m", [])

                forecast = []
                for i, time in enumerate(times):
                    forecast.append({
                        "time": time,
                        "temperature": temps[i] if i < len(temps) else None,
                        "precipitation": precip[i] if i < len(precip) else None,
                        "wind_speed": wind[i] if i < len(wind) else None
                    })

                return forecast
        except Exception as e:
            print(f"[WARN] Error obteniendo pronóstico: {e}")

        return []


class SolunarCalculator:
    """
    Calcula datos solunares y mejores horarios de pesca.

    Basado en la teoría solunar de John Alden Knight:
    - Períodos mayores: cuando la luna está sobre la cabeza o bajo los pies (2h)
    - Períodos menores: salida y puesta de luna (1h)
    - Luna nueva y llena = mejor pesca
    """

    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def calculate(self, date: datetime = None) -> SolunarData:
        """
        Calcula datos solunares para una fecha.
        """
        if date is None:
            date = datetime.now()

        # Calcular fase lunar
        moon_phase = self._moon_phase(date)
        moon_phase_name = self._phase_name(moon_phase)
        moon_illumination = self._illumination(moon_phase)

        # Calcular horas de sol (aproximación)
        sunrise, sunset = self._sun_times(date)

        # Calcular horas de luna (aproximación)
        moonrise, moonset = self._moon_times(date, moon_phase)

        # Calcular períodos de pesca
        major_periods, minor_periods = self._fishing_periods(date, moonrise, moonset)

        # Rating del día
        day_rating = self._day_rating(moon_phase, date)

        # Mejor horario
        best_time = self._best_time(sunrise, sunset, major_periods, minor_periods)

        return SolunarData(
            date=date,
            sunrise=sunrise,
            sunset=sunset,
            moonrise=moonrise,
            moonset=moonset,
            moon_phase=moon_phase,
            moon_phase_name=moon_phase_name,
            moon_illumination=moon_illumination,
            major_periods=major_periods,
            minor_periods=minor_periods,
            day_rating=day_rating,
            best_time=best_time
        )

    def _moon_phase(self, date: datetime) -> float:
        """
        Calcula fase lunar (0-1).
        0 = Luna nueva, 0.5 = Luna llena
        """
        # Fecha de referencia de luna nueva conocida
        ref_new_moon = datetime(2024, 1, 11, 11, 57)  # Luna nueva conocida

        # Período sinódico de la luna (días)
        synodic_month = 29.530588853

        # Días desde la referencia
        days_since = (date - ref_new_moon).total_seconds() / 86400

        # Fase actual (0-1)
        phase = (days_since % synodic_month) / synodic_month

        return phase

    def _phase_name(self, phase: float) -> str:
        """Nombre de la fase lunar."""
        if phase < 0.03 or phase > 0.97:
            return "Luna Nueva"
        elif phase < 0.22:
            return "Creciente"
        elif phase < 0.28:
            return "Cuarto Creciente"
        elif phase < 0.47:
            return "Gibosa Creciente"
        elif phase < 0.53:
            return "Luna Llena"
        elif phase < 0.72:
            return "Gibosa Menguante"
        elif phase < 0.78:
            return "Cuarto Menguante"
        else:
            return "Menguante"

    def _illumination(self, phase: float) -> float:
        """Porcentaje de iluminación lunar."""
        # La iluminación sigue una curva senoidal
        return (1 - math.cos(phase * 2 * math.pi)) / 2 * 100

    def _sun_times(self, date: datetime) -> Tuple[datetime, datetime]:
        """Calcula amanecer y atardecer (aproximación)."""
        # Para la costa peruana (aprox -18°, -71°)
        # Amanecer ~5:30-6:30, Atardecer ~17:30-18:30

        day_of_year = date.timetuple().tm_yday

        # Variación estacional (hemisferio sur)
        # Verano: días más largos, Invierno: días más cortos
        seasonal_offset = math.sin((day_of_year - 172) * 2 * math.pi / 365) * 0.5

        sunrise_hour = 6.0 - seasonal_offset
        sunset_hour = 18.0 + seasonal_offset

        sunrise = date.replace(
            hour=int(sunrise_hour),
            minute=int((sunrise_hour % 1) * 60),
            second=0, microsecond=0
        )
        sunset = date.replace(
            hour=int(sunset_hour),
            minute=int((sunset_hour % 1) * 60),
            second=0, microsecond=0
        )

        return sunrise, sunset

    def _moon_times(self, date: datetime, phase: float) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Calcula salida y puesta de luna (aproximación)."""
        # La luna sale ~50 minutos más tarde cada día
        # En luna nueva: sale con el sol
        # En luna llena: sale cuando se pone el sol

        base_hour = 6 + phase * 12  # 6am en nueva, 18pm en llena

        if base_hour > 24:
            return None, None  # Luna no visible este día

        moonrise_hour = base_hour % 24
        moonset_hour = (base_hour + 12) % 24

        moonrise = date.replace(
            hour=int(moonrise_hour),
            minute=int((moonrise_hour % 1) * 60),
            second=0, microsecond=0
        )

        moonset = date.replace(
            hour=int(moonset_hour),
            minute=int((moonset_hour % 1) * 60),
            second=0, microsecond=0
        )

        return moonrise, moonset

    def _fishing_periods(
        self,
        date: datetime,
        moonrise: Optional[datetime],
        moonset: Optional[datetime]
    ) -> Tuple[List[Tuple[datetime, datetime]], List[Tuple[datetime, datetime]]]:
        """
        Calcula períodos mayores y menores de pesca.

        Períodos mayores (2h): Luna sobre la cabeza o bajo los pies
        Períodos menores (1h): Salida y puesta de luna
        """
        major = []
        minor = []

        if moonrise:
            # Período menor: salida de luna
            minor.append((
                moonrise - timedelta(minutes=30),
                moonrise + timedelta(minutes=30)
            ))

            # Período mayor: luna sobre la cabeza (~6h después de salir)
            overhead = moonrise + timedelta(hours=6)
            major.append((
                overhead - timedelta(hours=1),
                overhead + timedelta(hours=1)
            ))

        if moonset:
            # Período menor: puesta de luna
            minor.append((
                moonset - timedelta(minutes=30),
                moonset + timedelta(minutes=30)
            ))

            # Período mayor: luna bajo los pies (~6h antes de ponerse)
            underfoot = moonset - timedelta(hours=6)
            if underfoot.date() == date.date():
                major.append((
                    underfoot - timedelta(hours=1),
                    underfoot + timedelta(hours=1)
                ))

        return major, minor

    def _day_rating(self, phase: float, date: datetime) -> float:
        """
        Calcula rating del día para pesca (0-100).

        Factores:
        - Fase lunar (luna nueva/llena = mejor)
        - Día de la semana (fines de semana pueden estar más concurridos)
        """
        # Factor lunar: máximo en luna nueva (0) y llena (0.5)
        lunar_factor = 1 - abs(2 * phase - 0.5) * 2 if phase <= 0.5 else 1 - abs(2 * (phase - 0.5) - 0.5) * 2
        lunar_factor = max(0, min(1, lunar_factor))

        # Luna nueva y llena son las mejores
        if phase < 0.05 or phase > 0.95 or (0.45 < phase < 0.55):
            lunar_factor = 1.0
        elif 0.2 < phase < 0.3 or 0.7 < phase < 0.8:  # Cuartos
            lunar_factor = 0.7
        else:
            lunar_factor = 0.5

        rating = lunar_factor * 100

        return round(rating, 1)

    def _best_time(
        self,
        sunrise: datetime,
        sunset: datetime,
        major_periods: List,
        minor_periods: List
    ) -> str:
        """Determina el mejor horario de pesca."""
        times = []

        # Horas doradas siempre son buenas
        times.append(f"Amanecer: {sunrise.strftime('%H:%M')}")
        times.append(f"Atardecer: {sunset.strftime('%H:%M')}")

        # Períodos solunares
        for start, end in major_periods:
            times.append(f"Mayor: {start.strftime('%H:%M')}-{end.strftime('%H:%M')}")

        return " | ".join(times[:3])


def get_fishing_conditions(lat: float, lon: float) -> Dict:
    """
    Obtiene todas las condiciones para pesca en un punto.
    """
    # Clima
    weather_fetcher = WeatherFetcher()
    weather = weather_fetcher.get_current(lat, lon)

    # Solunar
    solunar_calc = SolunarCalculator(lat, lon)
    solunar = solunar_calc.calculate()

    return {
        "weather": {
            "temperature": weather.temperature if weather else None,
            "wind_speed": weather.wind_speed if weather else None,
            "wind_direction": weather.wind_direction if weather else None,
            "is_safe": weather.is_safe if weather else True,
            "rating": weather.fishing_rating if weather else "desconocido",
            "warnings": weather.warnings if weather else []
        },
        "solunar": {
            "moon_phase": solunar.moon_phase_name,
            "moon_illumination": f"{solunar.moon_illumination:.0f}%",
            "day_rating": solunar.day_rating,
            "sunrise": solunar.sunrise.strftime("%H:%M"),
            "sunset": solunar.sunset.strftime("%H:%M"),
            "best_times": solunar.best_time
        },
        "timestamp": datetime.now().isoformat()
    }
