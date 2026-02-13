#!/usr/bin/env python3
"""
Descarga TODAS las variables disponibles de Copernicus Marine para pesca.

Variables incluidas:
- Física: SST, SSS, SLA, Corrientes (uo, vo), Capa de mezcla
- Biogeoquímica: Clorofila-a, KD490, Oxígeno disuelto
- Olas: Altura significativa, Período, Dirección
- Viento: Componentes E/N

Requiere:
- pip install copernicusmarine
- Credenciales en .env: COPERNICUS_USER, COPERNICUS_PASS
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from calendar import monthrange
import tempfile

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

# =============================================================================
# CONFIGURACIÓN DE DATASETS Y VARIABLES
# =============================================================================

REGION = {
    'lat_min': -20.0,
    'lat_max': -15.0,
    'lon_min': -82.0,
    'lon_max': -70.0,
    'name': 'Peru_Sur'
}

# Datasets de Copernicus y sus variables
DATASETS = {
    # =========================================================================
    # FÍSICA OCEÁNICA
    # =========================================================================
    'physics_daily': {
        'dataset_id': 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m',
        'reanalysis_id': 'cmems_mod_glo_phy_my_0.083deg_P1D-m',
        'variables': {
            'thetao': {
                'name': 'Temperatura',
                'unit': '°C',
                'depth': (0, 10),
                'use': 'Distribución de especies, metabolismo'
            },
            'so': {
                'name': 'Salinidad',
                'unit': 'PSU',
                'depth': (0, 10),
                'use': 'Masas de agua, frentes oceánicos'
            },
            'zos': {
                'name': 'Altura superficie (SLA)',
                'unit': 'm',
                'depth': None,
                'use': 'Upwelling, corrientes geostróficas'
            },
            'uo': {
                'name': 'Corriente Este',
                'unit': 'm/s',
                'depth': (0, 50),
                'use': 'Transporte larvas, nutrientes'
            },
            'vo': {
                'name': 'Corriente Norte',
                'unit': 'm/s',
                'depth': (0, 50),
                'use': 'Transporte larvas, nutrientes'
            },
            'mlotst': {
                'name': 'Capa de mezcla',
                'unit': 'm',
                'depth': None,
                'use': 'Estratificación, productividad'
            }
        }
    },

    # =========================================================================
    # BIOGEOQUÍMICA / COLOR DEL OCÉANO
    # =========================================================================
    'ocean_color': {
        'dataset_id': 'cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D',
        'variables': {
            'CHL': {
                'name': 'Clorofila-a',
                'unit': 'mg/m³',
                'depth': None,
                'use': 'Productividad primaria, cadena trófica'
            }
        }
    },

    'ocean_optics': {
        'dataset_id': 'cmems_obs-oc_glo_bgc-optics_my_l4-gapfree-multi-4km_P1D',
        'variables': {
            'KD490': {
                'name': 'Coef. atenuación luz',
                'unit': 'm⁻¹',
                'depth': None,
                'use': 'Claridad agua, penetración luz'
            }
        }
    },

    # =========================================================================
    # OLAS
    # =========================================================================
    'waves': {
        'dataset_id': 'cmems_mod_glo_wav_anfc_0.083deg_PT3H-i',
        'reanalysis_id': 'cmems_mod_glo_wav_my_0.083deg_PT3H',
        'variables': {
            'VHM0': {
                'name': 'Altura ola significativa',
                'unit': 'm',
                'depth': None,
                'use': 'Condiciones navegación, seguridad'
            },
            'VTPK': {
                'name': 'Período pico ola',
                'unit': 's',
                'depth': None,
                'use': 'Tipo oleaje (swell vs wind)'
            },
            'VMDR': {
                'name': 'Dirección ola',
                'unit': '°',
                'depth': None,
                'use': 'Planificación salida/entrada puerto'
            }
        }
    },

    # =========================================================================
    # VIENTO
    # =========================================================================
    'wind': {
        'dataset_id': 'cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H',
        'reanalysis_id': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',
        'variables': {
            'eastward_wind': {
                'name': 'Viento Este (U10)',
                'unit': 'm/s',
                'depth': None,
                'use': 'Afecta oleaje, upwelling'
            },
            'northward_wind': {
                'name': 'Viento Norte (V10)',
                'unit': 'm/s',
                'depth': None,
                'use': 'Afecta oleaje, upwelling'
            }
        }
    }
}


class CopernicusDownloader:
    """Descargador unificado para todos los datasets de Copernicus."""

    def __init__(self, output_dir: Path = None, verbose: bool = True):
        self.output_dir = output_dir or ROOT_DIR / 'data' / 'raw'
        self.verbose = verbose
        self.user = os.environ.get('COPERNICUS_USER', '')
        self.password = os.environ.get('COPERNICUS_PASS', '')

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def has_credentials(self) -> bool:
        return bool(self.user and self.password)

    def download_variable(
        self,
        dataset_config: dict,
        variable: str,
        var_info: dict,
        year: int,
        month: int
    ) -> bool:
        """Descarga una variable específica para un mes."""
        try:
            import copernicusmarine
            import xarray as xr
            import pandas as pd
        except ImportError:
            self.log("ERROR: pip install copernicusmarine xarray pandas")
            return False

        # Determinar dataset (reanalysis para datos históricos)
        if year <= 2023 and 'reanalysis_id' in dataset_config:
            dataset_id = dataset_config['reanalysis_id']
        else:
            dataset_id = dataset_config['dataset_id']

        # Crear directorio de salida
        var_dir = self.output_dir / variable.lower() / 'copernicus'
        var_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{year}-{month:02d}.parquet"
        output_path = var_dir / filename

        if output_path.exists():
            self.log(f"  ✓ {variable} {year}-{month:02d} ya existe")
            return True

        # Fechas
        start_date = f"{year}-{month:02d}-01"
        last_day = monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day:02d}"

        self.log(f"  Descargando {variable} {year}-{month:02d}...")

        try:
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_path = tmp.name

            # Parámetros de descarga
            params = {
                'dataset_id': dataset_id,
                'variables': [variable],
                'minimum_longitude': REGION['lon_min'],
                'maximum_longitude': REGION['lon_max'],
                'minimum_latitude': REGION['lat_min'],
                'maximum_latitude': REGION['lat_max'],
                'start_datetime': f"{start_date}T00:00:00",
                'end_datetime': f"{end_date}T23:59:59",
                'output_filename': tmp_path,
                'username': self.user,
                'password': self.password,
                'overwrite': True
            }

            # Agregar profundidad si aplica
            depth = var_info.get('depth')
            if depth:
                params['minimum_depth'] = depth[0]
                params['maximum_depth'] = depth[1]

            copernicusmarine.subset(**params)

            # Convertir a DataFrame
            ds = xr.open_dataset(tmp_path)
            df = ds[variable].to_dataframe().reset_index()

            # Renombrar columnas estándar
            rename_map = {
                'latitude': 'lat',
                'longitude': 'lon',
                'time': 'date',
                variable: 'value'
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # Limpiar
            df = df.dropna(subset=['value'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Agregar metadatos
            df['variable'] = variable
            df['source'] = 'copernicus'

            # Guardar
            df.to_parquet(output_path, index=False)

            ds.close()
            os.unlink(tmp_path)

            self.log(f"    ✓ Guardado: {len(df)} registros")
            return True

        except Exception as e:
            self.log(f"    ✗ Error: {e}")
            return False

    def download_all(
        self,
        start_year: int = 2024,
        start_month: int = 1,
        end_year: int = 2026,
        end_month: int = 1,
        datasets: list = None
    ):
        """Descarga todas las variables de todos los datasets."""
        if not self.has_credentials():
            print("=" * 60)
            print("ERROR: Credenciales de Copernicus no configuradas")
            print("Agregar a .env:")
            print("  COPERNICUS_USER=tu_email")
            print("  COPERNICUS_PASS=tu_password")
            print("Registro gratuito en: https://marine.copernicus.eu/")
            print("=" * 60)
            return

        # Filtrar datasets si se especifican
        if datasets:
            selected = {k: v for k, v in DATASETS.items() if k in datasets}
        else:
            selected = DATASETS

        # Generar lista de meses
        months = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            months.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1

        print("=" * 60)
        print("DESCARGA DE DATOS COPERNICUS MARINE")
        print("=" * 60)
        print(f"Región: {REGION['name']}")
        print(f"Período: {start_year}-{start_month:02d} a {end_year}-{end_month:02d}")
        print(f"Meses: {len(months)}")
        print(f"Datasets: {list(selected.keys())}")

        total_vars = sum(len(d['variables']) for d in selected.values())
        print(f"Variables: {total_vars}")
        print("=" * 60)

        results = {}

        for dataset_name, config in selected.items():
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"ID: {config['dataset_id']}")
            print("=" * 60)

            for var_name, var_info in config['variables'].items():
                print(f"\n{var_name}: {var_info['name']} ({var_info['unit']})")
                print(f"  Uso: {var_info['use']}")

                success = 0
                for year, month in months:
                    if self.download_variable(config, var_name, var_info, year, month):
                        success += 1

                results[var_name] = f"{success}/{len(months)}"
                print(f"  Resultado: {success}/{len(months)} meses")

        print("\n" + "=" * 60)
        print("RESUMEN DE DESCARGA")
        print("=" * 60)
        for var, result in results.items():
            print(f"  {var:20s}: {result}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Descarga variables de Copernicus Marine'
    )
    parser.add_argument('--start', type=str, default='2024-01',
                        help='Mes inicio (YYYY-MM)')
    parser.add_argument('--end', type=str, default='2026-01',
                        help='Mes fin (YYYY-MM)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        choices=list(DATASETS.keys()),
                        help='Datasets específicos a descargar')
    parser.add_argument('--list', action='store_true',
                        help='Solo listar variables disponibles')

    args = parser.parse_args()

    if args.list:
        print("=" * 60)
        print("VARIABLES DISPONIBLES EN COPERNICUS MARINE")
        print("=" * 60)
        for dataset_name, config in DATASETS.items():
            print(f"\n{dataset_name}:")
            print(f"  Dataset: {config['dataset_id']}")
            for var, info in config['variables'].items():
                print(f"    {var:15s} | {info['name']:25s} | {info['unit']:8s}")
        return

    start_parts = args.start.split('-')
    end_parts = args.end.split('-')

    downloader = CopernicusDownloader()
    downloader.download_all(
        start_year=int(start_parts[0]),
        start_month=int(start_parts[1]),
        end_year=int(end_parts[0]),
        end_month=int(end_parts[1]),
        datasets=args.datasets
    )


if __name__ == '__main__':
    main()
