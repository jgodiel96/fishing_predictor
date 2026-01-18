"""
Calculador de datos astronomicos: fases lunares, amanecer/atardecer, mareas.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import ephem
    HAS_EPHEM = True
except ImportError:
    HAS_EPHEM = False

try:
    from astral import LocationInfo
    from astral.sun import sun
    HAS_ASTRAL = True
except ImportError:
    HAS_ASTRAL = False


@dataclass
class LunarData:
    """Datos lunares."""
    phase: float  # 0-1 (0=nueva, 0.5=llena)
    illumination: float  # 0-100%
    phase_name: str  # Nombre de la fase
    is_rising: bool  # Si la luna esta creciendo


@dataclass
class SolarData:
    """Datos solares."""
    sunrise: datetime
    sunset: datetime
    solar_noon: datetime
    day_length_hours: float


@dataclass
class TideData:
    """Datos de marea (simplificado)."""
    height: float  # metros
    phase: str  # "rising", "falling", "high", "low"
    next_high: Optional[datetime]
    next_low: Optional[datetime]


class AstronomicalCalculator:
    """Calcula datos astronomicos para pesca."""

    # Coeficientes simplificados para marea (modelo armonico basico)
    # Basados en componente M2 (marea lunar semidiurna)
    TIDE_M2_PERIOD_HOURS = 12.42  # Periodo de marea lunar
    TIDE_AMPLITUDE_M = 0.8  # Amplitud tipica para costa sur Peru

    def __init__(self, lat: float = -17.85, lon: float = -71.15):
        """
        Inicializa el calculador.

        Args:
            lat: Latitud por defecto
            lon: Longitud por defecto
        """
        self.default_lat = lat
        self.default_lon = lon

    def get_lunar_data(self, date: Optional[datetime] = None) -> LunarData:
        """
        Calcula la fase lunar.

        Args:
            date: Fecha para el calculo (default: ahora)

        Returns:
            LunarData con fase, iluminacion, etc.
        """
        if date is None:
            date = datetime.now()

        if HAS_EPHEM:
            return self._get_lunar_ephem(date)
        else:
            return self._get_lunar_fallback(date)

    def _get_lunar_ephem(self, date: datetime) -> LunarData:
        """Calcula datos lunares usando ephem."""
        moon = ephem.Moon(date)

        # Fase: 0 = nueva, 0.5 = llena, 1 = nueva siguiente
        phase = moon.phase / 100.0  # ephem da 0-100

        # Iluminacion
        illumination = moon.phase

        # Determinar nombre de fase
        if phase < 0.125:
            phase_name = "Luna Nueva"
        elif phase < 0.25:
            phase_name = "Cuarto Creciente"
        elif phase < 0.375:
            phase_name = "Gibosa Creciente"
        elif phase < 0.625:
            phase_name = "Luna Llena"
        elif phase < 0.75:
            phase_name = "Gibosa Menguante"
        elif phase < 0.875:
            phase_name = "Cuarto Menguante"
        else:
            phase_name = "Luna Nueva"

        # Determinar si esta creciendo
        tomorrow = ephem.Moon(date + timedelta(days=1))
        is_rising = tomorrow.phase > moon.phase

        return LunarData(
            phase=phase,
            illumination=illumination,
            phase_name=phase_name,
            is_rising=is_rising
        )

    def _get_lunar_fallback(self, date: datetime) -> LunarData:
        """Calcula datos lunares sin ephem (aproximacion)."""
        # Ciclo lunar: ~29.53 dias
        # Nueva luna de referencia: 6 enero 2000
        ref_new_moon = datetime(2000, 1, 6, 18, 14)
        lunar_cycle = 29.530588853

        days_since_ref = (date - ref_new_moon).total_seconds() / 86400.0
        phase = (days_since_ref % lunar_cycle) / lunar_cycle

        # Iluminacion aproximada
        # Maxima en luna llena (phase=0.5), minima en nueva (phase=0 o 1)
        illumination = (1 - math.cos(2 * math.pi * phase)) / 2 * 100

        # Nombre de fase
        if phase < 0.125:
            phase_name = "Luna Nueva"
        elif phase < 0.25:
            phase_name = "Cuarto Creciente"
        elif phase < 0.375:
            phase_name = "Gibosa Creciente"
        elif phase < 0.625:
            phase_name = "Luna Llena"
        elif phase < 0.75:
            phase_name = "Gibosa Menguante"
        elif phase < 0.875:
            phase_name = "Cuarto Menguante"
        else:
            phase_name = "Luna Nueva"

        is_rising = phase < 0.5

        return LunarData(
            phase=phase,
            illumination=illumination,
            phase_name=phase_name,
            is_rising=is_rising
        )

    def get_solar_data(
        self,
        date: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> SolarData:
        """
        Calcula amanecer, atardecer y datos solares.

        Args:
            date: Fecha para el calculo
            lat: Latitud
            lon: Longitud

        Returns:
            SolarData con tiempos solares
        """
        if date is None:
            date = datetime.now()
        if lat is None:
            lat = self.default_lat
        if lon is None:
            lon = self.default_lon

        if HAS_ASTRAL:
            return self._get_solar_astral(date, lat, lon)
        else:
            return self._get_solar_fallback(date, lat, lon)

    def _get_solar_astral(self, date: datetime, lat: float, lon: float) -> SolarData:
        """Calcula datos solares usando astral."""
        location = LocationInfo(
            name="FishingSpot",
            region="Peru",
            timezone="America/Lima",
            latitude=lat,
            longitude=lon
        )

        s = sun(location.observer, date=date.date())

        sunrise = s["sunrise"].replace(tzinfo=None)
        sunset = s["sunset"].replace(tzinfo=None)
        noon = s["noon"].replace(tzinfo=None)

        day_length = (sunset - sunrise).total_seconds() / 3600.0

        return SolarData(
            sunrise=sunrise,
            sunset=sunset,
            solar_noon=noon,
            day_length_hours=day_length
        )

    def _get_solar_fallback(self, date: datetime, lat: float, lon: float) -> SolarData:
        """Calcula datos solares sin astral (aproximacion)."""
        # Dia del ano
        day_of_year = date.timetuple().tm_yday

        # Declinacion solar aproximada
        declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

        # Angulo horario al amanecer
        lat_rad = math.radians(lat)
        decl_rad = math.radians(declination)

        cos_hour_angle = -math.tan(lat_rad) * math.tan(decl_rad)
        cos_hour_angle = max(-1, min(1, cos_hour_angle))

        hour_angle = math.degrees(math.acos(cos_hour_angle))

        # Duracion del dia en horas
        day_length = 2 * hour_angle / 15.0

        # Mediadia solar (aproximado para zona horaria Peru GMT-5)
        solar_noon = date.replace(hour=12, minute=0, second=0, microsecond=0)
        solar_noon += timedelta(minutes=int(-4 * (lon + 75)))  # Correccion por longitud

        half_day = timedelta(hours=day_length / 2)
        sunrise = solar_noon - half_day
        sunset = solar_noon + half_day

        return SolarData(
            sunrise=sunrise,
            sunset=sunset,
            solar_noon=solar_noon,
            day_length_hours=day_length
        )

    def get_tide_data(
        self,
        date: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> TideData:
        """
        Estima datos de marea usando modelo armonico simplificado.

        Nota: Para predicciones precisas, usar datos de estacion mareal local.

        Args:
            date: Fecha/hora para el calculo
            lat: Latitud
            lon: Longitud

        Returns:
            TideData con altura y fase de marea
        """
        if date is None:
            date = datetime.now()

        # Modelo simplificado basado en componente M2
        # Referencia: alta marea aproximada
        ref_high_tide = datetime(2025, 1, 1, 6, 0)

        hours_since_ref = (date - ref_high_tide).total_seconds() / 3600.0
        tide_phase = (hours_since_ref % self.TIDE_M2_PERIOD_HOURS) / self.TIDE_M2_PERIOD_HOURS

        # Altura de marea (seno para ciclo)
        height = self.TIDE_AMPLITUDE_M * math.cos(2 * math.pi * tide_phase)

        # Determinar fase
        if tide_phase < 0.125 or tide_phase > 0.875:
            phase = "high"
        elif 0.375 < tide_phase < 0.625:
            phase = "low"
        elif tide_phase < 0.5:
            phase = "falling"
        else:
            phase = "rising"

        # Calcular proxima alta y baja
        hours_to_next_high = (1 - tide_phase) * self.TIDE_M2_PERIOD_HOURS
        hours_to_next_low = ((0.5 - tide_phase) % 1) * self.TIDE_M2_PERIOD_HOURS

        next_high = date + timedelta(hours=hours_to_next_high)
        next_low = date + timedelta(hours=hours_to_next_low)

        return TideData(
            height=height,
            phase=phase,
            next_high=next_high,
            next_low=next_low
        )

    def get_golden_hour_score(
        self,
        date: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> float:
        """
        Calcula score de hora dorada (mejor momento para pescar).

        La hora dorada es aproximadamente 1 hora antes y despues del
        amanecer y atardecer.

        Args:
            date: Fecha/hora para el calculo
            lat: Latitud
            lon: Longitud

        Returns:
            Score de 0-100 (100 = hora optima)
        """
        if date is None:
            date = datetime.now()

        solar = self.get_solar_data(date, lat, lon)

        # Horas desde amanecer y hasta atardecer
        hours_from_sunrise = (date - solar.sunrise).total_seconds() / 3600.0
        hours_to_sunset = (solar.sunset - date).total_seconds() / 3600.0

        # Ventana dorada: 1 hora antes/despues de amanecer/atardecer
        golden_window = 1.0

        # Score basado en proximidad a ventanas doradas
        sunrise_score = max(0, 100 - abs(hours_from_sunrise) * 100 / golden_window)
        sunset_score = max(0, 100 - abs(hours_to_sunset) * 100 / golden_window)

        # También considerar la hora previa al amanecer
        pre_sunrise_score = max(0, 100 - abs(hours_from_sunrise + 1) * 100 / golden_window) if hours_from_sunrise < 0 else 0

        return max(sunrise_score, sunset_score, pre_sunrise_score)

    def get_lunar_score(self, date: Optional[datetime] = None) -> float:
        """
        Calcula score lunar para pesca basado en teoria solunar.

        Las fases de luna nueva y llena suelen ser mejores para pesca.

        Args:
            date: Fecha para el calculo

        Returns:
            Score de 0-100
        """
        lunar = self.get_lunar_data(date)

        # Mejor pesca en luna nueva y llena
        # phase cerca de 0 o 0.5 = mejor
        distance_to_new = min(lunar.phase, 1 - lunar.phase)
        distance_to_full = abs(lunar.phase - 0.5)

        min_distance = min(distance_to_new, distance_to_full)

        # Convertir distancia a score (0 distancia = 100 score)
        # Escala: 0.25 es la maxima distancia posible
        score = 100 * (1 - min_distance / 0.25)

        return max(0, min(100, score))

    def get_all_astronomical_data(
        self,
        date: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> Dict:
        """
        Obtiene todos los datos astronomicos.

        Returns:
            Dict con todos los datos astronomicos calculados
        """
        if date is None:
            date = datetime.now()
        if lat is None:
            lat = self.default_lat
        if lon is None:
            lon = self.default_lon

        lunar = self.get_lunar_data(date)
        solar = self.get_solar_data(date, lat, lon)
        tide = self.get_tide_data(date, lat, lon)

        return {
            "timestamp": date.isoformat(),
            "latitude": lat,
            "longitude": lon,
            # Lunar
            "lunar_phase": lunar.phase,
            "lunar_illumination": lunar.illumination,
            "lunar_phase_name": lunar.phase_name,
            "lunar_is_rising": lunar.is_rising,
            "lunar_score": self.get_lunar_score(date),
            # Solar
            "sunrise": solar.sunrise.isoformat(),
            "sunset": solar.sunset.isoformat(),
            "solar_noon": solar.solar_noon.isoformat(),
            "day_length_hours": solar.day_length_hours,
            "golden_hour_score": self.get_golden_hour_score(date, lat, lon),
            # Tide
            "tide_height": tide.height,
            "tide_phase": tide.phase,
            "next_high_tide": tide.next_high.isoformat() if tide.next_high else None,
            "next_low_tide": tide.next_low.isoformat() if tide.next_low else None,
        }
