"""
Fetcher para datos de Clorofila-a de Copernicus Marine.

Descarga datos diarios de concentracion de clorofila-a
y los procesa para integracion con el sistema de scoring V7.

La Clorofila-a es indicador de productividad primaria:
    Chl-a Alta -> Fitoplancton -> Zooplancton -> Peces Forrajeros -> Depredadores

Rangos optimos (IMARPE/FAO):
    - 2-10 mg/m3: Optimo para pesca
    - <0.5 mg/m3: Oligotrofico (baja productividad)
    - >20 mg/m3: Posible HAB (Harmful Algal Bloom)

Uso:
    fetcher = ChlorophyllFetcher()
    fetcher.download_month(2024, 6)
    value = fetcher.get_value_for_location('2024-06-15', -17.8, -71.2)
    score = fetcher.calculate_score(value)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from functools import lru_cache
from calendar import monthrange

import numpy as np
import pandas as pd

from data.data_config import DataConfig
from data.manifest import ManifestManager

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES DE DOMINIO
# =============================================================================

CHLA_OPTIMAL_MIN: float = 2.0      # mg/m3 - inicio rango optimo
CHLA_OPTIMAL_MAX: float = 10.0     # mg/m3 - fin rango optimo
CHLA_BLOOM_THRESHOLD: float = 20.0 # mg/m3 - umbral de bloom
CHLA_OLIGOTROPHIC: float = 0.5     # mg/m3 - umbral oligotrofico

# Dataset de Copernicus
COPERNICUS_CHLA_DATASET = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
COPERNICUS_CHLA_VARIABLE = "CHL"


@dataclass(frozen=True, slots=True)
class ChlaConfig:
    """Configuracion inmutable para el fetcher de Clorofila-a."""
    output_dir: Path
    region_lat_min: float
    region_lat_max: float
    region_lon_min: float
    region_lon_max: float
    cache_ttl_seconds: int = 3600


class ChlorophyllFetcher:
    """
    Fetcher para datos de Clorofila-a de Copernicus Marine.

    Implementa patron de alta eficiencia:
    - Cache LRU para queries repetidas
    - Tipos de datos optimizados (float32)
    - Lazy loading de datos mensuales

    Atributos:
        config: Configuracion del fetcher
        _cache: Cache de datos mensuales
        _manifest: Gestor de manifest para tracking

    Ejemplo:
        >>> fetcher = ChlorophyllFetcher()
        >>> fetcher.download_month(2024, 6)
        True
        >>> value = fetcher.get_value_for_location('2024-06-15', -17.8, -71.2)
        >>> print(f"Chl-a: {value:.2f} mg/m3")
        Chl-a: 3.45 mg/m3
        >>> score = fetcher.calculate_score(value)
        >>> print(f"Score: {score:.2f}")
        Score: 0.90
    """

    __slots__ = ('config', '_cache', '_manifest')

    def __init__(self, config: Optional[ChlaConfig] = None):
        """
        Inicializa el fetcher.

        Args:
            config: Configuracion opcional. Si no se provee,
                   usa valores por defecto de DataConfig.
        """
        if config is None:
            config = ChlaConfig(
                output_dir=DataConfig.RAW_CHLA_COPERNICUS,
                region_lat_min=DataConfig.REGION['lat_min'],
                region_lat_max=DataConfig.REGION['lat_max'],
                region_lon_min=DataConfig.REGION['lon_min'],
                region_lon_max=DataConfig.REGION['lon_max']
            )

        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._manifest = ManifestManager('copernicus_chla')

        # Crear directorio si no existe
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def download_month(self, year: int, month: int) -> bool:
        """
        Descarga datos de Clorofila-a para un mes.

        Args:
            year: Ano (2020-2026)
            month: Mes (1-12)

        Returns:
            True si exitoso o ya existia, False si error

        Raises:
            ValueError: Si ano/mes fuera de rango valido
        """
        # Validar entrada
        if not (2020 <= year <= 2030):
            raise ValueError(f"Ano {year} fuera de rango valido (2020-2030)")
        if not (1 <= month <= 12):
            raise ValueError(f"Mes {month} fuera de rango valido (1-12)")

        output_path = self._get_output_path(year, month)

        # Idempotencia - no re-descargar
        if output_path.exists():
            logger.debug(f"Archivo ya existe: {output_path}")
            return True

        try:
            import copernicusmarine

            # Definir rango de fechas
            start_date = datetime(year, month, 1)
            last_day = monthrange(year, month)[1]
            end_date = datetime(year, month, last_day)

            logger.info(f"Descargando Chl-a {year}-{month:02d}...")

            # Descargar a archivo temporal
            temp_nc = self.config.output_dir / f"temp_{year}_{month}.nc"

            copernicusmarine.subset(
                dataset_id=COPERNICUS_CHLA_DATASET,
                variables=[COPERNICUS_CHLA_VARIABLE],
                start_datetime=start_date.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=end_date.strftime("%Y-%m-%dT23:59:59"),
                minimum_latitude=self.config.region_lat_min - 0.5,
                maximum_latitude=self.config.region_lat_max + 0.5,
                minimum_longitude=self.config.region_lon_min - 0.5,
                maximum_longitude=self.config.region_lon_max + 0.5,
                output_filename=str(temp_nc),
                output_directory=str(self.config.output_dir),
                force_download=True
            )

            # Convertir NetCDF a Parquet optimizado
            df = self._netcdf_to_dataframe(temp_nc)

            if df.empty:
                logger.warning(f"No hay datos para {year}-{month:02d}")
                return False

            # Guardar como Parquet
            df.to_parquet(
                output_path,
                compression='snappy',
                index=False
            )

            # Limpiar archivo temporal
            temp_nc.unlink(missing_ok=True)

            # Actualizar manifest
            self._manifest.add_download(
                filename=output_path.name,
                period_start=start_date.strftime("%Y-%m-%d"),
                period_end=end_date.strftime("%Y-%m-%d"),
                records=len(df),
                source_url=f"copernicus:{COPERNICUS_CHLA_DATASET}"
            )
            self._manifest.save()

            logger.info(f"Descargado: {output_path.name} ({len(df)} registros)")
            return True

        except ImportError:
            logger.error("copernicusmarine no instalado. Ejecuta: pip install copernicusmarine")
            return False
        except Exception as e:
            logger.error(f"Error descargando Chl-a {year}-{month}: {e}")
            return False

    def _netcdf_to_dataframe(self, nc_path: Path) -> pd.DataFrame:
        """
        Convierte NetCDF a DataFrame optimizado.

        Args:
            nc_path: Ruta al archivo NetCDF

        Returns:
            DataFrame con columnas [date, lat, lon, value]
        """
        try:
            import xarray as xr
        except ImportError:
            logger.error("xarray no instalado. Ejecuta: pip install xarray netcdf4")
            return pd.DataFrame()

        if not nc_path.exists():
            return pd.DataFrame()

        ds = xr.open_dataset(nc_path)

        # Encontrar nombre de variable
        var_name = COPERNICUS_CHLA_VARIABLE
        if var_name not in ds:
            # Buscar nombre alternativo
            var_candidates = [v for v in ds.data_vars if 'chl' in v.lower()]
            if not var_candidates:
                logger.error(f"Variable CHL no encontrada en {nc_path}")
                ds.close()
                return pd.DataFrame()
            var_name = var_candidates[0]

        # Convertir a DataFrame
        df = ds[var_name].to_dataframe().reset_index()

        # Renombrar columnas estandar
        col_mapping = {
            'time': 'date',
            'latitude': 'lat',
            'longitude': 'lon',
            var_name: 'value'
        }

        for old, new in col_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        # Asegurar columnas minimas
        required_cols = ['date', 'lat', 'lon', 'value']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Columna requerida '{col}' no encontrada")
                ds.close()
                return pd.DataFrame()

        # Filtrar NaN y valores invalidos
        df = df.dropna(subset=['value'])
        df = df[df['value'] > 0]       # Chl-a siempre positiva
        df = df[df['value'] < 100]     # Filtrar outliers extremos

        # Optimizar tipos (memoria eficiente)
        df['lat'] = df['lat'].astype(np.float32)
        df['lon'] = df['lon'].astype(np.float32)
        df['value'] = df['value'].astype(np.float32)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')

        ds.close()

        return df[['date', 'lat', 'lon', 'value']]

    @lru_cache(maxsize=256)
    def get_value_for_location(
        self,
        date: str,
        lat: float,
        lon: float,
        radius_deg: float = 0.05
    ) -> Optional[float]:
        """
        Obtiene valor de Clorofila-a para ubicacion y fecha.

        Usa cache LRU para queries repetidas.

        Args:
            date: Fecha ISO (YYYY-MM-DD)
            lat: Latitud
            lon: Longitud
            radius_deg: Radio de busqueda (~5.5km por 0.05 grados)

        Returns:
            Valor promedio de Chl-a en mg/m3 o None
        """
        # Redondear para mejorar cache hits
        lat = round(lat, 4)
        lon = round(lon, 4)

        year = int(date[:4])
        month = int(date[5:7])

        df = self._load_month_data(year, month)

        if df is None or df.empty:
            return None

        # Filtrar por fecha
        target_date = pd.Timestamp(date)
        date_mask = df['date'] == target_date

        # Filtrar por ubicacion (bbox mas eficiente que distancia)
        spatial_mask = (
            (df['lat'] >= lat - radius_deg) &
            (df['lat'] <= lat + radius_deg) &
            (df['lon'] >= lon - radius_deg) &
            (df['lon'] <= lon + radius_deg)
        )

        values = df.loc[date_mask & spatial_mask, 'value']

        if values.empty:
            # Fallback: usar promedio del mes si no hay dato exacto
            values = df.loc[spatial_mask, 'value']

        if values.empty:
            return None

        return float(values.mean())

    @staticmethod
    def calculate_score(chla: Optional[float]) -> float:
        """
        Calcula score de pesca basado en Clorofila-a.

        Basado en literatura IMARPE y FAO para pesquerias
        del Pacifico Sureste.

        Args:
            chla: Concentracion de Clorofila-a en mg/m3

        Returns:
            Score normalizado 0-1

        Rangos:
            0.0 - 0.5 mg/m3: Oligotrofico -> 0.2
            0.5 - 1.0 mg/m3: Bajo -> 0.4
            1.0 - 2.0 mg/m3: Moderado -> 0.6
            2.0 - 10.0 mg/m3: Optimo -> 0.9
            10.0 - 20.0 mg/m3: Alto -> 0.7
            > 20.0 mg/m3: Bloom (posible HAB) -> 0.3
        """
        if chla is None:
            return 0.5  # Neutral si no hay dato

        # Rango optimo: 2-10 mg/m3
        if CHLA_OPTIMAL_MIN <= chla <= CHLA_OPTIMAL_MAX:
            return 0.9

        # Moderado: 1-2 mg/m3
        if 1.0 <= chla < CHLA_OPTIMAL_MIN:
            # Interpolacion lineal 0.6 -> 0.9
            t = (chla - 1.0) / (CHLA_OPTIMAL_MIN - 1.0)
            return 0.6 + t * 0.3

        # Alto pero no bloom: 10-20 mg/m3
        if CHLA_OPTIMAL_MAX < chla <= CHLA_BLOOM_THRESHOLD:
            # Interpolacion lineal 0.9 -> 0.7
            t = (chla - CHLA_OPTIMAL_MAX) / (CHLA_BLOOM_THRESHOLD - CHLA_OPTIMAL_MAX)
            return 0.9 - t * 0.2

        # Bajo: 0.5-1 mg/m3
        if CHLA_OLIGOTROPHIC <= chla < 1.0:
            # Interpolacion lineal 0.4 -> 0.6
            t = (chla - CHLA_OLIGOTROPHIC) / (1.0 - CHLA_OLIGOTROPHIC)
            return 0.4 + t * 0.2

        # Oligotrofico: < 0.5 mg/m3
        if chla < CHLA_OLIGOTROPHIC:
            return 0.2

        # Bloom intenso: > 20 mg/m3 (posible HAB)
        return 0.3

    def _get_output_path(self, year: int, month: int) -> Path:
        """Genera ruta de salida estandarizada."""
        return self.config.output_dir / f"{year}-{month:02d}.parquet"

    def _load_month_data(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de un mes con cache."""
        cache_key = f"{year}-{month:02d}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_output_path(year, month)
        if not path.exists():
            # Intentar descargar si no existe
            logger.info(f"Datos no encontrados, intentando descargar {year}-{month:02d}...")
            if not self.download_month(year, month):
                return None

        if not path.exists():
            return None

        df = pd.read_parquet(path)

        # Gestion de cache LRU simple (max 12 meses en memoria)
        if len(self._cache) >= 12:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[cache_key] = df
        return df

    def download_range(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> Tuple[int, int]:
        """
        Descarga rango de meses.

        Args:
            start_year: Ano inicial
            start_month: Mes inicial
            end_year: Ano final
            end_month: Mes final

        Returns:
            (exitosos, fallidos)
        """
        success = 0
        failed = 0

        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        while current <= end:
            if self.download_month(current.year, current.month):
                success += 1
            else:
                failed += 1

            # Siguiente mes
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        logger.info(f"Descarga completada: {success} exitosos, {failed} fallidos")
        return success, failed

    def get_monthly_average(self, year: int, month: int) -> Optional[float]:
        """
        Obtiene promedio mensual de Chl-a para la region.

        Args:
            year: Ano
            month: Mes

        Returns:
            Promedio mensual o None
        """
        df = self._load_month_data(year, month)
        if df is None or df.empty:
            return None
        return float(df['value'].mean())

    def clear_cache(self) -> None:
        """Limpia la cache de datos mensuales."""
        self._cache.clear()
        # Tambien limpiar cache LRU
        self.get_value_for_location.cache_clear()


# =============================================================================
# FUNCION DE CONVENIENCIA
# =============================================================================

def get_chla_score(date: str, lat: float, lon: float) -> float:
    """
    Funcion de conveniencia para obtener score de Chl-a.

    Args:
        date: Fecha ISO (YYYY-MM-DD)
        lat: Latitud
        lon: Longitud

    Returns:
        Score 0-1

    Ejemplo:
        >>> score = get_chla_score('2024-06-15', -17.8, -71.2)
        >>> print(f"Score: {score:.2f}")
    """
    fetcher = ChlorophyllFetcher()
    value = fetcher.get_value_for_location(date, lat, lon)
    return fetcher.calculate_score(value)
