#!/usr/bin/env python3
"""
Copernicus Marine Physics Fetcher - SSS (Salinity) and SLA (Sea Level Anomaly)

Basado en el estado del arte (Plan V2):
- SSS (Salinidad): Variable #1 en importancia según papers 2024
- SLA (Anomalía Nivel del Mar): Variable #2 en importancia

Dataset: GLOBAL_ANALYSISFORECAST_PHY_001_024
Variables:
- so: Sea water salinity (PSU)
- zos: Sea surface height above geoid (m)

Requiere credenciales de Copernicus Marine en .env:
    COPERNICUS_USER=tu_email
    COPERNICUS_PASS=tu_password

Referencias científicas:
- Paper V2 #4: Optimal variables include SSS, SLA, SST, CV
- INCOIS India: Uses SST + Chl-a + SSH for PFZ validation
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import tempfile

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

from data.data_config import DataConfig
from data.manifest import ManifestManager


class CopernicusPhysicsFetcher:
    """
    Fetcher for Copernicus Marine Physics variables (SSS, SLA).

    Estos datos complementan SST y son críticos para predicción de zonas de pesca:
    - SSS (Salinidad): Indica masas de agua y frentes
    - SLA (Nivel del mar): Indica upwelling y corrientes
    """

    # Dataset para datos de física oceánica
    DATASET_ID = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"  # Daily salinity
    DATASET_SLA = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"  # Daily physics

    # Alternativa: Análisis (mejor para histórico)
    DATASET_REANALYSIS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

    # Rangos óptimos para pesca en Humboldt (basado en IMARPE)
    OPTIMAL_RANGES = {
        'salinity': {
            'min': 34.4,  # PSU
            'max': 35.3,  # PSU
            'source': 'IMARPE anchoveta studies'
        },
        'sla': {
            'upwelling_threshold': -0.05,  # metros (negativo = upwelling)
            'source': 'Chavez et al. 2008'
        }
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.copernicus_user = os.environ.get('COPERNICUS_USER', '')
        self.copernicus_pass = os.environ.get('COPERNICUS_PASS', '')
        self.region = DataConfig.REGION

        # Crear directorios si no existen
        self.sss_dir = DataConfig.RAW_DIR / 'sss' / 'copernicus'
        self.sla_dir = DataConfig.RAW_DIR / 'sla' / 'copernicus'
        self.sss_dir.mkdir(parents=True, exist_ok=True)
        self.sla_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def has_credentials(self) -> bool:
        """Verifica si las credenciales están configuradas."""
        return bool(self.copernicus_user and self.copernicus_pass)

    def download_sss_month(
        self,
        year: int,
        month: int,
        manager: Optional[ManifestManager] = None
    ) -> bool:
        """
        Descarga datos de salinidad (SSS) para un mes.

        Args:
            year: Año
            month: Mes (1-12)
            manager: ManifestManager opcional para tracking

        Returns:
            True si exitoso
        """
        if not self.has_credentials():
            print("ERROR: Credenciales de Copernicus no configuradas")
            return False

        filename = f"{year}-{month:02d}.parquet"
        output_path = self.sss_dir / filename

        if output_path.exists():
            self.log(f"Skipping SSS {filename} (already exists)")
            return True

        start_date = f"{year}-{month:02d}-01"
        last_day = monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day:02d}"

        self.log(f"Downloading Copernicus SSS: {start_date} to {end_date}")

        try:
            import copernicusmarine
            import xarray as xr

            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_path = tmp.name

            # Elegir dataset según año
            dataset_id = self.DATASET_REANALYSIS if year <= 2023 else self.DATASET_ID

            self.log(f"  Using dataset: {dataset_id}")

            # Descargar salinidad
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["so"],  # Salinity
                minimum_longitude=self.region['lon_min'],
                maximum_longitude=self.region['lon_max'],
                minimum_latitude=self.region['lat_min'],
                maximum_latitude=self.region['lat_max'],
                minimum_depth=0,
                maximum_depth=10,  # Superficie
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T23:59:59",
                output_filename=tmp_path,
                username=self.copernicus_user,
                password=self.copernicus_pass,
                overwrite=True
            )

            # Convertir a DataFrame
            ds = xr.open_dataset(tmp_path)
            df = self._netcdf_to_dataframe(ds, 'so', 'salinity')

            if df.empty:
                self.log(f"  WARNING: No SSS data for {year}-{month:02d}")
                return False

            # Guardar como Parquet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            # Actualizar manifest
            if manager:
                manager.add_download(
                    filename=filename,
                    period_start=start_date,
                    period_end=end_date,
                    records=len(df)
                )

            self.log(f"  Saved: {output_path} ({len(df)} records)")

            # Limpiar
            ds.close()
            os.unlink(tmp_path)

            return True

        except Exception as e:
            self.log(f"  ERROR downloading SSS: {e}")
            return False

    def download_sla_month(
        self,
        year: int,
        month: int,
        manager: Optional[ManifestManager] = None
    ) -> bool:
        """
        Descarga datos de anomalía de nivel del mar (SLA) para un mes.

        Args:
            year: Año
            month: Mes (1-12)
            manager: ManifestManager opcional para tracking

        Returns:
            True si exitoso
        """
        if not self.has_credentials():
            print("ERROR: Credenciales de Copernicus no configuradas")
            return False

        filename = f"{year}-{month:02d}.parquet"
        output_path = self.sla_dir / filename

        if output_path.exists():
            self.log(f"Skipping SLA {filename} (already exists)")
            return True

        start_date = f"{year}-{month:02d}-01"
        last_day = monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day:02d}"

        self.log(f"Downloading Copernicus SLA: {start_date} to {end_date}")

        try:
            import copernicusmarine
            import xarray as xr

            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_path = tmp.name

            # Elegir dataset según año
            dataset_id = self.DATASET_REANALYSIS if year <= 2023 else self.DATASET_SLA

            self.log(f"  Using dataset: {dataset_id}")

            # Descargar SLA
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["zos"],  # Sea surface height
                minimum_longitude=self.region['lon_min'],
                maximum_longitude=self.region['lon_max'],
                minimum_latitude=self.region['lat_min'],
                maximum_latitude=self.region['lat_max'],
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T23:59:59",
                output_filename=tmp_path,
                username=self.copernicus_user,
                password=self.copernicus_pass,
                overwrite=True
            )

            # Convertir a DataFrame
            ds = xr.open_dataset(tmp_path)
            df = self._netcdf_to_dataframe(ds, 'zos', 'sla')

            if df.empty:
                self.log(f"  WARNING: No SLA data for {year}-{month:02d}")
                return False

            # Guardar como Parquet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            # Actualizar manifest
            if manager:
                manager.add_download(
                    filename=filename,
                    period_start=start_date,
                    period_end=end_date,
                    records=len(df)
                )

            self.log(f"  Saved: {output_path} ({len(df)} records)")

            # Limpiar
            ds.close()
            os.unlink(tmp_path)

            return True

        except Exception as e:
            self.log(f"  ERROR downloading SLA: {e}")
            return False

    def _netcdf_to_dataframe(
        self,
        ds,
        var_name: str,
        output_name: str
    ) -> pd.DataFrame:
        """Convierte dataset NetCDF a DataFrame."""
        try:
            # Convertir a DataFrame
            df = ds[var_name].to_dataframe().reset_index()

            # Renombrar columnas estándar
            rename_map = {
                'latitude': 'lat',
                'longitude': 'lon',
                'time': 'date',
                var_name: output_name
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # Eliminar columnas no necesarias
            keep_cols = ['date', 'lat', 'lon', output_name]
            df = df[[c for c in keep_cols if c in df.columns]]

            # Convertir fecha
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Eliminar NaN
            df = df.dropna(subset=[output_name])

            # Agregar source
            df['source'] = 'copernicus_physics'

            return df

        except Exception as e:
            self.log(f"  Error converting NetCDF: {e}")
            return pd.DataFrame()

    def calculate_sss_score(self, salinity: float) -> float:
        """
        Calcula score de pesca basado en salinidad.

        Basado en estudios de IMARPE para anchoveta:
        - Rango óptimo: 34.4-35.3 PSU
        """
        opt = self.OPTIMAL_RANGES['salinity']

        if opt['min'] <= salinity <= opt['max']:
            # En rango óptimo: score alto
            center = (opt['min'] + opt['max']) / 2
            dist = abs(salinity - center) / ((opt['max'] - opt['min']) / 2)
            return 1.0 - (dist * 0.2)  # 0.8-1.0
        elif salinity < opt['min']:
            # Muy fresca
            diff = opt['min'] - salinity
            return max(0.2, 0.7 - diff * 0.1)
        else:
            # Muy salina
            diff = salinity - opt['max']
            return max(0.2, 0.7 - diff * 0.1)

    def calculate_sla_score(self, sla: float) -> float:
        """
        Calcula score de pesca basado en anomalía de nivel del mar.

        SLA negativo indica upwelling (bueno para pesca).
        SLA positivo indica hundimiento de agua (menos favorable).
        """
        threshold = self.OPTIMAL_RANGES['sla']['upwelling_threshold']

        if sla < threshold:
            # Upwelling fuerte - excelente
            return min(1.0, 0.9 + abs(sla) * 2)
        elif sla < 0:
            # Upwelling leve - bueno
            return 0.7 + (abs(sla) / abs(threshold)) * 0.2
        elif sla < 0.05:
            # Neutro
            return 0.5
        else:
            # Hundimiento - menos favorable
            return max(0.2, 0.5 - sla * 2)

    def get_sss_for_location(
        self,
        date: str,
        lat: float,
        lon: float
    ) -> Optional[float]:
        """
        Obtiene salinidad para una ubicación y fecha específica.
        """
        try:
            year, month = int(date[:4]), int(date[5:7])
            filename = f"{year}-{month:02d}.parquet"
            filepath = self.sss_dir / filename

            if not filepath.exists():
                return None

            df = pd.read_parquet(filepath)

            # Filtrar por fecha y ubicación cercana
            df = df[df['date'] == date]
            df = df[
                (df['lat'] >= lat - 0.1) & (df['lat'] <= lat + 0.1) &
                (df['lon'] >= lon - 0.1) & (df['lon'] <= lon + 0.1)
            ]

            if df.empty:
                return None

            return df['salinity'].mean()

        except Exception:
            return None

    def get_sla_for_location(
        self,
        date: str,
        lat: float,
        lon: float
    ) -> Optional[float]:
        """
        Obtiene SLA para una ubicación y fecha específica.
        """
        try:
            year, month = int(date[:4]), int(date[5:7])
            filename = f"{year}-{month:02d}.parquet"
            filepath = self.sla_dir / filename

            if not filepath.exists():
                return None

            df = pd.read_parquet(filepath)

            # Filtrar por fecha y ubicación cercana
            df = df[df['date'] == date]
            df = df[
                (df['lat'] >= lat - 0.1) & (df['lat'] <= lat + 0.1) &
                (df['lon'] >= lon - 0.1) & (df['lon'] <= lon + 0.1)
            ]

            if df.empty:
                return None

            return df['sla'].mean()

        except Exception:
            return None


def download_all_physics(
    start_year: int = 2020,
    start_month: int = 1,
    end_year: int = 2026,
    end_month: int = 1,
    verbose: bool = True
):
    """
    Descarga todos los datos de física oceánica (SSS + SLA).

    Args:
        start_year, start_month: Fecha inicio
        end_year, end_month: Fecha fin
        verbose: Mostrar progreso
    """
    fetcher = CopernicusPhysicsFetcher(verbose=verbose)

    if not fetcher.has_credentials():
        print("ERROR: Configure COPERNICUS_USER y COPERNICUS_PASS en .env")
        return

    # Crear managers con nombres de fuente correctos
    sss_manager = ManifestManager('copernicus_sss')
    sla_manager = ManifestManager('copernicus_sla')

    # Generar lista de meses
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    print(f"\nDescargando SSS y SLA: {len(months)} meses")
    print("=" * 50)

    sss_success = 0
    sla_success = 0

    for year, month in months:
        print(f"\n{year}-{month:02d}:")

        # SSS
        if fetcher.download_sss_month(year, month, sss_manager):
            sss_success += 1

        # SLA
        if fetcher.download_sla_month(year, month, sla_manager):
            sla_success += 1

    print("\n" + "=" * 50)
    print(f"SSS: {sss_success}/{len(months)} meses descargados")
    print(f"SLA: {sla_success}/{len(months)} meses descargados")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Descarga SSS y SLA de Copernicus Marine'
    )
    parser.add_argument('--start', type=str, default='2024-01',
                        help='Mes inicio (YYYY-MM)')
    parser.add_argument('--end', type=str, default='2026-01',
                        help='Mes fin (YYYY-MM)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar detalles')

    args = parser.parse_args()

    start_parts = args.start.split('-')
    end_parts = args.end.split('-')

    download_all_physics(
        start_year=int(start_parts[0]),
        start_month=int(start_parts[1]),
        end_year=int(end_parts[0]),
        end_month=int(end_parts[1]),
        verbose=args.verbose
    )
