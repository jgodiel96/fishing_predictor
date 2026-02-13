"""
Fetcher para datos de Olas de Copernicus Marine.

Descarga datos de oleaje para análisis de condiciones de navegación
y predicción de actividad pesquera.

Variables:
    - VHM0: Altura significativa de ola (m)
    - VTPK: Período pico de ola (s)
    - VMDR: Dirección media de ola (°)

Uso en pesca:
    - Condiciones de navegación y seguridad
    - Planificación de salidas/entradas a puerto
    - Correlación con actividad pesquera (menos pesca en mar picado)
    - Identificación de ventanas de buen tiempo

Uso:
    fetcher = WavesFetcher()
    fetcher.download_month(2024, 6)
    height, period, direction = fetcher.get_waves_for_location('2024-06-15', -17.8, -71.2)
    score = fetcher.calculate_navigation_score(height, period)
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

# Dataset de Copernicus para olas
COPERNICUS_WAVES_DATASET_NRT = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
COPERNICUS_WAVES_DATASET_MY = "cmems_mod_glo_wav_my_0.083deg_PT3H"  # Reanalysis

# Variables
VARIABLES = ['VHM0', 'VTPK', 'VMDR']  # Altura, período, dirección

# Umbrales de navegación para pesca artesanal
WAVE_HEIGHT_OPTIMAL_MAX = 1.5  # m - Condiciones óptimas
WAVE_HEIGHT_MARGINAL_MAX = 2.5  # m - Condiciones marginales
WAVE_HEIGHT_DANGEROUS = 3.5  # m - Peligroso para embarcaciones pequeñas


@dataclass(frozen=True)
class WavesConfig:
    """Configuración para el fetcher de olas."""
    output_dir: Path
    region_lat_min: float
    region_lat_max: float
    region_lon_min: float
    region_lon_max: float


class WavesFetcher:
    """
    Fetcher para datos de olas de Copernicus Marine.

    Atributos:
        config: Configuración del fetcher
        _cache: Cache de datos mensuales
        _manifest: Gestor de manifest para tracking

    Ejemplo:
        >>> fetcher = WavesFetcher()
        >>> fetcher.download_month(2024, 6)
        True
        >>> h, p, d = fetcher.get_waves_for_location('2024-06-15', -17.8, -71.2)
        >>> print(f"Olas: {h:.1f}m, período {p:.1f}s, dirección {d:.0f}°")
    """

    __slots__ = ('config', '_cache', '_manifest', 'copernicus_user', 'copernicus_pass')

    def __init__(self, config: Optional[WavesConfig] = None):
        """Inicializa el fetcher."""
        if config is None:
            # Crear directorio si no existe
            output_dir = DataConfig.RAW_DIR / "waves" / "copernicus"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = WavesConfig(
                output_dir=output_dir,
                region_lat_min=DataConfig.REGION['lat_min'],
                region_lat_max=DataConfig.REGION['lat_max'],
                region_lon_min=DataConfig.REGION['lon_min'],
                region_lon_max=DataConfig.REGION['lon_max']
            )

        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._manifest = ManifestManager('copernicus_waves')

        # Credenciales
        import os
        self.copernicus_user = os.environ.get('COPERNICUS_USER', '')
        self.copernicus_pass = os.environ.get('COPERNICUS_PASS', '')

    def _get_output_path(self, year: int, month: int) -> Path:
        """Obtiene la ruta del archivo de salida."""
        return self.config.output_dir / f"{year}-{month:02d}.parquet"

    def download_month(self, year: int, month: int) -> bool:
        """
        Descarga datos de olas para un mes.

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
        dataset_id = COPERNICUS_WAVES_DATASET_MY if year <= 2023 else COPERNICUS_WAVES_DATASET_NRT

        # Rango de fechas
        start_date = datetime(year, month, 1)
        last_day = monthrange(year, month)[1]
        end_date = datetime(year, month, last_day)

        logger.info(f"Descargando olas {year}-{month:02d}...")

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
                output_filename=tmp_path,
                username=self.copernicus_user,
                password=self.copernicus_pass,
                overwrite=True
            )

            # Convertir a DataFrame
            ds = xr.open_dataset(tmp_path)

            # Procesar datos - agregar por día (dataset es cada 3 horas)
            records = []

            # Agrupar por día
            for date_str in pd.date_range(start_date, end_date).strftime('%Y-%m-%d'):
                day_data = ds.sel(time=date_str)

                if day_data.time.size == 0:
                    continue

                for lat in ds.latitude.values:
                    for lon in ds.longitude.values:
                        try:
                            # Promediar el día
                            vhm0 = float(day_data['VHM0'].sel(latitude=lat, longitude=lon).mean().values)
                            vtpk = float(day_data['VTPK'].sel(latitude=lat, longitude=lon).mean().values)
                            vmdr = float(day_data['VMDR'].sel(latitude=lat, longitude=lon).mean().values)

                            # También obtener máximo del día (para seguridad)
                            vhm0_max = float(day_data['VHM0'].sel(latitude=lat, longitude=lon).max().values)

                            if not np.isnan(vhm0):
                                records.append({
                                    'date': date_str,
                                    'lat': float(lat),
                                    'lon': float(lon),
                                    'wave_height': vhm0,
                                    'wave_height_max': vhm0_max,
                                    'wave_period': vtpk,
                                    'wave_direction': vmdr
                                })
                        except Exception:
                            continue

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
            logger.error(f"Error descargando olas {year}-{month:02d}: {e}")
            return False

    def get_waves_for_location(
        self,
        date: str,
        lat: float,
        lon: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Obtiene datos de olas para una ubicación y fecha.

        Args:
            date: Fecha en formato YYYY-MM-DD
            lat: Latitud
            lon: Longitud

        Returns:
            (wave_height, wave_period, wave_direction) o (None, None, None)
        """
        year, month = int(date[:4]), int(date[5:7])
        cache_key = f"{year}-{month:02d}"

        # Cargar datos si no están en cache
        if cache_key not in self._cache:
            file_path = self._get_output_path(year, month)
            if not file_path.exists():
                return None, None, None
            self._cache[cache_key] = pd.read_parquet(file_path)

        df = self._cache[cache_key]

        # Filtrar por fecha
        day_data = df[df['date'] == date]
        if day_data.empty:
            return None, None, None

        # Encontrar punto más cercano
        distances = np.sqrt((day_data['lat'] - lat)**2 + (day_data['lon'] - lon)**2)
        nearest_idx = distances.idxmin()
        nearest = day_data.loc[nearest_idx]

        return nearest['wave_height'], nearest['wave_period'], nearest['wave_direction']

    def calculate_navigation_score(
        self,
        wave_height: Optional[float],
        wave_period: Optional[float] = None
    ) -> float:
        """
        Calcula score de condiciones de navegación.

        Basado en seguridad para embarcaciones pesqueras artesanales.

        Args:
            wave_height: Altura significativa (m)
            wave_period: Período de ola (s) - opcional

        Returns:
            Score 0-1 (1 = condiciones óptimas, 0 = peligroso)
        """
        if wave_height is None or np.isnan(wave_height):
            return 0.5  # Neutral si no hay datos

        # Score basado en altura de ola
        if wave_height <= WAVE_HEIGHT_OPTIMAL_MAX:
            # Condiciones óptimas
            height_score = 1.0
        elif wave_height <= WAVE_HEIGHT_MARGINAL_MAX:
            # Condiciones marginales - linear decay
            height_score = 1.0 - (wave_height - WAVE_HEIGHT_OPTIMAL_MAX) / (WAVE_HEIGHT_MARGINAL_MAX - WAVE_HEIGHT_OPTIMAL_MAX) * 0.4
        elif wave_height <= WAVE_HEIGHT_DANGEROUS:
            # Condiciones difíciles
            height_score = 0.6 - (wave_height - WAVE_HEIGHT_MARGINAL_MAX) / (WAVE_HEIGHT_DANGEROUS - WAVE_HEIGHT_MARGINAL_MAX) * 0.4
        else:
            # Peligroso
            height_score = max(0.1, 0.2 - (wave_height - WAVE_HEIGHT_DANGEROUS) * 0.1)

        # Ajuste por período (olas largas son más manejables)
        if wave_period is not None and not np.isnan(wave_period):
            if wave_period > 10:
                # Swell largo - más predecible
                period_bonus = 0.1
            elif wave_period < 5:
                # Olas cortas y choppy - más difícil
                period_bonus = -0.1
            else:
                period_bonus = 0.0

            height_score = min(1.0, max(0.0, height_score + period_bonus))

        return height_score

    def get_sea_state(self, wave_height: Optional[float]) -> str:
        """
        Clasifica el estado del mar según escala Douglas.

        Args:
            wave_height: Altura significativa (m)

        Returns:
            Descripción del estado del mar
        """
        if wave_height is None or np.isnan(wave_height):
            return "Desconocido"

        if wave_height < 0.1:
            return "Calma (0)"
        elif wave_height < 0.5:
            return "Rizada (1-2)"
        elif wave_height < 1.25:
            return "Marejadilla (3)"
        elif wave_height < 2.5:
            return "Marejada (4)"
        elif wave_height < 4.0:
            return "Fuerte marejada (5)"
        elif wave_height < 6.0:
            return "Mar gruesa (6)"
        elif wave_height < 9.0:
            return "Mar muy gruesa (7)"
        else:
            return "Mar arbolada (8-9)"
