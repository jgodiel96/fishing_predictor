#!/usr/bin/env python3
"""
Actualización Completa de Base de Datos - Fishing Predictor

Este script descarga todos los datos nuevos y actualiza la base de datos.
Ejecutar periódicamente (semanal/mensual) para mantener datos actualizados.

Uso:
    # Actualización completa (todas las fuentes)
    python scripts/update_database.py

    # Solo descargar sin consolidar
    python scripts/update_database.py --download-only

    # Solo consolidar (sin descargar)
    python scripts/update_database.py --consolidate-only

    # Ver qué se descargaría sin hacerlo
    python scripts/update_database.py --dry-run

    # Descargar rango específico
    python scripts/update_database.py --start 2020-01 --end 2026-01

Fuentes de Datos:
    - GFW (Global Fishing Watch): Actividad pesquera AIS
    - Open-Meteo: Condiciones marinas (olas, viento)
    - NOAA SST: Temperatura superficial del mar

Credenciales:
    Las credenciales se cargan automáticamente desde .env
    - GFW_API_KEY: Global Fishing Watch
    - EARTHDATA_USER/PASS: NASA Earthdata (MUR SST)
    - COPERNICUS_USER/PASS: Copernicus Marine
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from data.data_config import DataConfig


def print_header(title: str):
    """Print formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(icon: str, message: str):
    """Print status message."""
    print(f"{icon}  {message}")


def get_current_month() -> str:
    """Get current month as YYYY-MM string."""
    now = datetime.now()
    return f"{now.year}-{now.month:02d}"


def download_all_sources(start: str, end: str, dry_run: bool = False) -> dict:
    """
    Download data from all sources.

    Args:
        start: Start month (YYYY-MM)
        end: End month (YYYY-MM)
        dry_run: If True, only show what would be downloaded

    Returns:
        Dictionary with download stats per source
    """
    from scripts.download_incremental import IncrementalDownloader, parse_month

    start_year, start_month = parse_month(start)
    end_year, end_month = parse_month(end)

    downloader = IncrementalDownloader(verbose=True, dry_run=dry_run)

    # Sources to download (in order of priority)
    sources = [
        ('gfw', 'Global Fishing Watch (pesca)'),
        ('open_meteo', 'Open-Meteo ERA5 (olas/viento)'),
        ('noaa_sst', 'NOAA OISST (temperatura)'),
    ]

    all_stats = {}

    for source_id, source_name in sources:
        print_header(f"DESCARGANDO: {source_name}")

        # Check credentials for GFW
        if source_id == 'gfw' and not DataConfig.get_gfw_api_key():
            print_status("⚠️", "GFW_API_KEY no configurada, saltando...")
            all_stats[source_id] = {'skipped': True, 'reason': 'no_api_key'}
            continue

        stats = downloader.download_source(
            source_id,
            start_year, start_month,
            end_year, end_month
        )
        all_stats[source_id] = stats

    return all_stats


def consolidate_data(verbose: bool = True) -> dict:
    """
    Consolidate Bronze layer to Silver layer.

    Returns:
        Dictionary with consolidation stats
    """
    from data.consolidator import Consolidator

    print_header("CONSOLIDANDO DATOS: Bronze → Silver")

    consolidator = Consolidator(verbose=verbose)
    stats = consolidator.consolidate_all()

    return stats


def validate_data() -> bool:
    """
    Run data validation.

    Returns:
        True if all validations pass
    """
    from scripts.validate_data import DataValidator, print_final_summary

    print_header("VALIDANDO INTEGRIDAD DE DATOS")

    validator = DataValidator(verbose=False)
    results = validator.validate_all(quick=True)

    all_passed = print_final_summary(results)
    return all_passed


def print_summary(download_stats: dict, consolidate_stats: dict):
    """Print final summary."""
    print_header("RESUMEN DE ACTUALIZACIÓN")

    print("\nDescargas:")
    total_downloaded = 0
    for source, stats in download_stats.items():
        if stats.get('skipped'):
            print(f"  {source}: Saltado ({stats.get('reason', 'unknown')})")
        else:
            downloaded = stats.get('downloaded', 0)
            failed = stats.get('failed', 0)
            total_downloaded += downloaded
            status = "✅" if failed == 0 else "⚠️"
            print(f"  {status} {source}: {downloaded} meses descargados, {failed} fallidos")

    print(f"\n  Total: {total_downloaded} archivos nuevos")

    if consolidate_stats:
        print("\nConsolidación:")
        for layer, stats in consolidate_stats.items():
            if isinstance(stats, dict):
                records = stats.get('records_output', stats.get('records', 0))
                if records > 0:
                    print(f"  ✅ {layer}: {records:,} registros")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Actualizar base de datos del Fishing Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/update_database.py                    # Actualización completa
  python scripts/update_database.py --dry-run          # Ver qué se descargaría
  python scripts/update_database.py --start 2020-01    # Desde 2020
  python scripts/update_database.py --consolidate-only # Solo consolidar
        """
    )

    parser.add_argument('--start', type=str, default='2020-01',
                       help='Mes inicial (YYYY-MM), default: 2020-01')
    parser.add_argument('--end', type=str, default=None,
                       help='Mes final (YYYY-MM), default: mes actual')
    parser.add_argument('--dry-run', action='store_true',
                       help='Mostrar qué se descargaría sin hacerlo')
    parser.add_argument('--download-only', action='store_true',
                       help='Solo descargar, no consolidar')
    parser.add_argument('--consolidate-only', action='store_true',
                       help='Solo consolidar, no descargar')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Saltar validación final')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mostrar información detallada')

    args = parser.parse_args()

    # Default end to current month
    if args.end is None:
        args.end = get_current_month()

    print_header("ACTUALIZACIÓN DE BASE DE DATOS - FISHING PREDICTOR")
    print(f"  Rango: {args.start} a {args.end}")
    print(f"  Región: Tacna-Ilo, Perú")

    if args.dry_run:
        print("\n  *** MODO DRY-RUN - No se harán cambios ***")

    # Check credentials
    print("\nCredenciales:")
    gfw_key = DataConfig.get_gfw_api_key()
    print(f"  GFW API Key: {'✅ Configurada' if gfw_key else '❌ No configurada'}")

    earthdata = DataConfig.get_earthdata_credentials()
    print(f"  Earthdata: {'✅ Configurada' if earthdata[0] else '❌ No configurada'}")

    copernicus = DataConfig.get_copernicus_credentials()
    print(f"  Copernicus: {'✅ Configurada' if copernicus[0] else '❌ No configurada'}")

    download_stats = {}
    consolidate_stats = {}

    # Download phase
    if not args.consolidate_only:
        download_stats = download_all_sources(args.start, args.end, args.dry_run)

    # Consolidate phase
    if not args.download_only and not args.dry_run:
        consolidate_stats = consolidate_data(args.verbose)

    # Validation phase
    if not args.skip_validation and not args.dry_run:
        validate_data()

    # Print summary
    if not args.dry_run:
        print_summary(download_stats, consolidate_stats)

    print_status("✅", "Actualización completada")
    print()


if __name__ == '__main__':
    main()
