"""
Fetcher para datos de Corrientes Oceánicas de Copernicus Marine.

Descarga datos diarios de velocidad de corrientes (uo, vo)
para análisis de transporte de larvas, nutrientes y patrones de pesca.

Variables:
    - uo: Velocidad hacia el Este (m/s)
    - vo: Velocidad hacia el Norte (m/s)

Uso en pesca:
    - Transporte de larvas y huevos
    - Movimiento de nutrientes y fitoplancton
    - Predicción de agregación de cardúmenes
    - Deriva de objetos flotantes (FADs naturales)

Uso:
    fetcher = CurrentsFetcher()
    fetcher.download_month(2024, 6)
    uo, vo = fetcher.get_currents_for_location('2024-06-15', -17.8, -71.2)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from calendar import monthrange

import numpy as np
import pandas as pd

from data.data_config import DataConfig
from data.manifest import ManifestManager

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES DE DOMINIO
# =============================================================================

# Dataset de Copernicus para corrientes
# NRT: Forecast cada 6 horas con uo/vo
COPERNICUS_CURRENTS_DATASET_NRT = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"
# Reanalysis: Daily con uo/vo
COPERNICUS_CURRENTS_DATASET_MY = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# Variables
VARIABLES = ['uo', 'vo']  # Eastward, Northward velocity

# Profundidad para corrientes superficiales
DEPTH_MIN = 0.0
DEPTH_MAX = 15.0  # Capa superficial (0-15m)


@dataclass(frozen=True)
class CurrentsConfig:
    """Configuración para el fetcher de corrientes."""
    output_dir: Path
    region_lat_min: float
    region_lat_max: float
    region_lon_min: float
    region_lon_max: float


class CurrentsFetcher:
    """
    Fetcher para datos de corrientes oceánicas de Copernicus Marine.

    Atributos:
        config: Configuración del fetcher
        _cache: Cache de datos mensuales
        _manifest: Gestor de manifest para tracking

    Ejemplo:
        >>> fetcher = CurrentsFetcher()
        >>> fetcher.download_month(2024, 6)
        True
        >>> uo, vo = fetcher.get_currents_for_location('2024-06-15', -17.8, -71.2)
        >>> speed = np.sqrt(uo**2 + vo**2)
        >>> print(f"Velocidad: {speed:.3f} m/s")
    """

    __slots__ = ('config', '_cache', '_manifest', 'copernicus_user', 'copernicus_pass')

    def __init__(self, config: Optional[CurrentsConfig] = None):
        """Inicializa el fetcher."""
        if config is None:
            # Crear directorio si no existe
            output_dir = DataConfig.RAW_DIR / "currents" / "copernicus"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = CurrentsConfig(
                output_dir=output_dir,
                region_lat_min=DataConfig.REGION['lat_min'],
                region_lat_max=DataConfig.REGION['lat_max'],
                region_lon_min=DataConfig.REGION['lon_min'],
                region_lon_max=DataConfig.REGION['lon_max']
            )

        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._manifest = ManifestManager('copernicus_currents')

        # Credenciales
        import os
        self.copernicus_user = os.environ.get('COPERNICUS_USER', '')
        self.copernicus_pass = os.environ.get('COPERNICUS_PASS', '')

    def _get_output_path(self, year: int, month: int) -> Path:
        """Obtiene la ruta del archivo de salida."""
        return self.config.output_dir / f"{year}-{month:02d}.parquet"

    def download_month(self, year: int, month: int) -> bool:
        """
        Descarga datos de corrientes para un mes.

        Args:
            year: Año (2020-2026)
            month: Mes (1-12)

        Returns:
            True si exitoso o ya existía, False si error
        """
        if not (2020 <= year <= 2030):
            raise ValueError(f"Año {year} fuera de rango válido (2020-2030)")
        if not (1 <= month <= 12):
            raise ValueError(f"Mes {month} fuera de rango válido (1-12)")

        output_path = self._get_output_path(year, month)

        # Idempotencia
        if output_path.exists():
            logger.debug(f"Archivo ya existe: {output_path}")
            return True

        try:
            import copernicusmarine
            import xarray as xr
        except ImportError:
            logger.error("copernicusmarine no instalado: pip install copernicusmarine")
            return False

        # Elegir dataset según año
        dataset_id = COPERNICUS_CURRENTS_DATASET_MY if year <= 2023 else COPERNICUS_CURRENTS_DATASET_NRT

        # Rango de fechas
        start_date = datetime(year, month, 1)
        last_day = monthrange(year, month)[1]
        end_date = datetime(year, month, last_day)

        logger.info(f"Descargando corrientes {year}-{month:02d}...")

        try:
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_path = tmp.name

            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=VARIABLES,
                start_datetime=start_date.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=end_date.strftime("%Y-%m-%dT23:59:59"),
                minimum_latitude=self.config.region_lat_min - 0.5,
                maximum_latitude=self.config.region_lat_max + 0.5,
                minimum_longitude=self.config.region_lon_min - 0.5,
                maximum_longitude=self.config.region_lon_max + 0.5,
                minimum_depth=DEPTH_MIN,
                maximum_depth=DEPTH_MAX,
                output_filename=tmp_path,
                username=self.copernicus_user,
                password=self.copernicus_pass,
                overwrite=True
            )

            # Convertir a DataFrame
            ds = xr.open_dataset(tmp_path)

            # Procesar cada variable
            records = []
            for time_val in ds.time.values:
                date_str = str(time_val)[:10]

                for lat in ds.latitude.values:
                    for lon in ds.longitude.values:
                        # Promediar en profundidad si hay múltiples niveles
                        uo_val = float(ds['uo'].sel(time=time_val, latitude=lat, longitude=lon).mean('depth').values)
                        vo_val = float(ds['vo'].sel(time=time_val, latitude=lat, longitude=lon).mean('depth').values)

                        if not (np.isnan(uo_val) and np.isnan(vo_val)):
                            records.append({
                                'date': date_str,
                                'lat': float(lat),
                                'lon': float(lon),
                                'uo': uo_val,
                                'vo': vo_val,
                                'speed': np.sqrt(uo_val**2 + vo_val**2) if not np.isnan(uo_val) else np.nan,
                                'direction': np.degrees(np.arctan2(vo_val, uo_val)) if not np.isnan(uo_val) else np.nan
                            })

            ds.close()

            # Limpiar temporal
            import os
            os.unlink(tmp_path)

            if not records:
                logger.warning(f"No hay datos para {year}-{month:02d}")
                return False

            df = pd.DataFrame(records)
            df.to_parquet(output_path, compression='snappy', index=False)

            # Actualizar manifest
            self._manifest.add_download(
                filename=output_path.name,
                period_start=start_date.strftime("%Y-%m-%d"),
                period_end=end_date.strftime("%Y-%m-%d"),
                records=len(df),
                source_url=f"copernicus:{dataset_id}"
            )
            self._manifest.save()

            logger.info(f"Descargado: {output_path.name} ({len(df)} registros)")
            return True

        except Exception as e:
            logger.error(f"Error descargando corrientes {year}-{month:02d}: {e}")
            return False

    def get_currents_for_location(
        self,
        date: str,
        lat: float,
        lon: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Obtiene corrientes para una ubicación y fecha.

        Args:
            date: Fecha en formato YYYY-MM-DD
            lat: Latitud
            lon: Longitud

        Returns:
            (uo, vo) tupla de velocidades en m/s, o (None, None) si no hay datos
        """
        year, month = int(date[:4]), int(date[5:7])
        cache_key = f"{year}-{month:02d}"

        # Cargar datos si no están en cache
        if cache_key not in self._cache:
            file_path = self._get_output_path(year, month)
            if not file_path.exists():
                return None, None
            self._cache[cache_key] = pd.read_parquet(file_path)

        df = self._cache[cache_key]

        # Filtrar por fecha
        day_data = df[df['date'] == date]
        if day_data.empty:
            return None, None

        # Encontrar punto más cercano
        distances = np.sqrt((day_data['lat'] - lat)**2 + (day_data['lon'] - lon)**2)
        nearest_idx = distances.idxmin()
        nearest = day_data.loc[nearest_idx]

        return nearest['uo'], nearest['vo']

    def calculate_transport_score(self, uo: float, vo: float) -> float:
        """
        Calcula score basado en condiciones de transporte.

        Corrientes moderadas son mejores para agregación de peces.
        Corrientes muy fuertes dispersan los cardúmenes.

        Args:
            uo: Velocidad Este (m/s)
            vo: Velocidad Norte (m/s)

        Returns:
            Score 0-1
        """
        if uo is None or vo is None or np.isnan(uo) or np.isnan(vo):
            return 0.5  # Neutral si no hay datos

        speed = np.sqrt(uo**2 + vo**2)

        # Corrientes óptimas: 0.1 - 0.3 m/s
        if 0.1 <= speed <= 0.3:
            return 1.0
        elif speed < 0.1:
            # Muy calmado - menor circulación de nutrientes
            return 0.7 + (speed / 0.1) * 0.3
        elif speed <= 0.5:
            # Moderado-fuerte
            return 1.0 - ((speed - 0.3) / 0.2) * 0.3
        else:
            # Muy fuerte - dispersión
            return max(0.3, 0.7 - (speed - 0.5) * 0.5)
