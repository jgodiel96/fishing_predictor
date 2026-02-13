"""
Proveedor de datos oceanográficos de Copernicus Marine Service.

Integra datos reales de:
- SST (Sea Surface Temperature)
- Corrientes oceánicas (uo, vo)
- Olas (VHM0, VTPK, VMDR)
- Clorofila-a (productividad)

para alimentar el modelo de predicción de pesca.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from data.data_config import DataConfig


@dataclass
class OceanPoint:
    """Punto oceanográfico con datos completos de Copernicus."""
    lat: float
    lon: float
    date: str
    # SST
    sst: Optional[float] = None
    # Corrientes
    uo: Optional[float] = None  # Velocidad Este (m/s)
    vo: Optional[float] = None  # Velocidad Norte (m/s)
    current_speed: Optional[float] = None
    current_direction: Optional[float] = None
    # Olas
    wave_height: Optional[float] = None
    wave_period: Optional[float] = None
    wave_direction: Optional[float] = None
    # Productividad
    chlorophyll: Optional[float] = None

    def is_complete(self) -> bool:
        """Verifica si tiene datos mínimos para predicción."""
        return self.sst is not None


class CopernicusDataProvider:
    """
    Proveedor unificado de datos de Copernicus Marine Service.

    Carga y fusiona datos de múltiples fuentes:
    - data/raw/sst/copernicus/
    - data/raw/currents/copernicus/
    - data/raw/waves/copernicus/
    - data/raw/chla/copernicus/
    """

    def __init__(self):
        self.data_dir = DataConfig.RAW_DIR
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load_parquet(self, subdir: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga archivo parquet con cache."""
        cache_key = f"{subdir}_{year}_{month}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.data_dir / subdir / "copernicus" / f"{year}-{month:02d}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._cache[cache_key] = df
            return df
        return None

    def get_data_for_date(self, date: str) -> List[OceanPoint]:
        """
        Obtiene todos los datos oceanográficos para una fecha.

        Args:
            date: Fecha en formato YYYY-MM-DD

        Returns:
            Lista de OceanPoint con datos fusionados
        """
        year, month, day = int(date[:4]), int(date[5:7]), int(date[8:10])

        # Cargar datasets
        sst_df = self._load_parquet("sst", year, month)
        currents_df = self._load_parquet("currents", year, month)
        waves_df = self._load_parquet("waves", year, month)
        chla_df = self._load_parquet("chla", year, month)

        if sst_df is None:
            return []

        # Filtrar por fecha
        sst_day = sst_df[sst_df['date'] == date] if 'date' in sst_df.columns else sst_df

        points = []
        for _, row in sst_day.iterrows():
            lat, lon = row['lat'], row['lon']

            point = OceanPoint(
                lat=lat,
                lon=lon,
                date=date,
                sst=row.get('sst', row.get('analysed_sst'))
            )

            # Agregar corrientes
            if currents_df is not None:
                point = self._add_currents(point, currents_df, date, lat, lon)

            # Agregar olas
            if waves_df is not None:
                point = self._add_waves(point, waves_df, date, lat, lon)

            # Agregar clorofila
            if chla_df is not None:
                point = self._add_chlorophyll(point, chla_df, date, lat, lon)

            points.append(point)

        return points

    def _add_currents(self, point: OceanPoint, df: pd.DataFrame,
                      date: str, lat: float, lon: float) -> OceanPoint:
        """Agrega datos de corrientes al punto."""
        day_df = df[df['date'] == date] if 'date' in df.columns else df
        nearby = day_df[
            (abs(day_df['lat'] - lat) < 0.1) &
            (abs(day_df['lon'] - lon) < 0.1)
        ]
        if not nearby.empty:
            row = nearby.iloc[0]
            point.uo = row.get('uo')
            point.vo = row.get('vo')
            point.current_speed = row.get('speed')
            point.current_direction = row.get('direction')

            # Calcular si no existen
            if point.current_speed is None and point.uo is not None:
                point.current_speed = np.sqrt(point.uo**2 + point.vo**2)
            if point.current_direction is None and point.uo is not None:
                point.current_direction = np.degrees(np.arctan2(point.vo, point.uo))
        return point

    def _add_waves(self, point: OceanPoint, df: pd.DataFrame,
                   date: str, lat: float, lon: float) -> OceanPoint:
        """Agrega datos de olas al punto."""
        day_df = df[df['date'] == date] if 'date' in df.columns else df
        nearby = day_df[
            (abs(day_df['lat'] - lat) < 0.1) &
            (abs(day_df['lon'] - lon) < 0.1)
        ]
        if not nearby.empty:
            row = nearby.iloc[0]
            point.wave_height = row.get('wave_height', row.get('VHM0'))
            point.wave_period = row.get('wave_period', row.get('VTPK'))
            point.wave_direction = row.get('wave_direction', row.get('VMDR'))
        return point

    def _add_chlorophyll(self, point: OceanPoint, df: pd.DataFrame,
                         date: str, lat: float, lon: float) -> OceanPoint:
        """Agrega datos de clorofila al punto."""
        # Manejar formato de fecha datetime
        if 'date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                day_df = df[df['date'].dt.strftime('%Y-%m-%d') == date]
            else:
                day_df = df[df['date'] == date]
        else:
            day_df = df

        nearby = day_df[
            (abs(day_df['lat'] - lat) < 0.1) &
            (abs(day_df['lon'] - lon) < 0.1)
        ]
        if not nearby.empty:
            point.chlorophyll = nearby.iloc[0].get('value', nearby.iloc[0].get('CHL'))
        return point

    def get_available_dates(self) -> List[str]:
        """Retorna lista de fechas con datos disponibles."""
        dates = set()
        sst_dir = self.data_dir / "sst" / "copernicus"

        if sst_dir.exists():
            for f in sst_dir.glob("*.parquet"):
                if f.name.startswith('_'):
                    continue
                year, month = f.stem.split('-')
                # Agregar el día 15 de cada mes como representativo
                dates.add(f"{year}-{month}-15")

        return sorted(dates)

    def get_statistics(self, date: str) -> Dict:
        """Obtiene estadísticas de los datos para una fecha."""
        points = self.get_data_for_date(date)

        if not points:
            return {'error': 'No data available'}

        stats = {
            'date': date,
            'total_points': len(points),
            'sst': {
                'count': sum(1 for p in points if p.sst is not None),
                'min': min((p.sst for p in points if p.sst), default=None),
                'max': max((p.sst for p in points if p.sst), default=None),
                'mean': np.mean([p.sst for p in points if p.sst]) if any(p.sst for p in points) else None
            },
            'currents': {
                'count': sum(1 for p in points if p.current_speed is not None),
                'mean_speed': np.mean([p.current_speed for p in points if p.current_speed]) if any(p.current_speed for p in points) else None
            },
            'waves': {
                'count': sum(1 for p in points if p.wave_height is not None),
                'mean_height': np.mean([p.wave_height for p in points if p.wave_height]) if any(p.wave_height for p in points) else None
            },
            'chlorophyll': {
                'count': sum(1 for p in points if p.chlorophyll is not None)
            }
        }
        return stats


def convert_to_marine_points(ocean_points: List[OceanPoint]) -> List:
    """
    Convierte OceanPoint a formato compatible con FeatureExtractor.

    Returns:
        Lista de objetos compatibles con extract_from_marine_points
    """
    from dataclasses import dataclass

    @dataclass
    class MarinePointCompat:
        lat: float
        lon: float
        sst: Optional[float]
        wave_height: Optional[float]
        wave_period: Optional[float]
        current_speed: Optional[float]
        current_direction: Optional[float]

    return [
        MarinePointCompat(
            lat=p.lat,
            lon=p.lon,
            sst=p.sst,
            wave_height=p.wave_height if p.wave_height else 1.0,
            wave_period=p.wave_period if p.wave_period else 8.0,
            current_speed=p.current_speed if p.current_speed else 0.1,
            current_direction=p.current_direction if p.current_direction else 180.0
        )
        for p in ocean_points if p.sst is not None
    ]
