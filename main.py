#!/usr/bin/env python3
"""
Fishing Predictor - Main Entry Point (MVC Architecture)

Usage:
    python main.py                              # Run full analysis for today
    python main.py --date 2026-02-15            # Run analysis for specific date
    python main.py --lat -17.8 --lon -71.2      # Search near location (default 10km)
    python main.py --lat -17.8 --lon -71.2 --radius 5  # Search within 5km
    python main.py --test                       # Run tests
    python main.py --help                       # Show help
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_analysis(target_date: str = None, user_lat: float = None, user_lon: float = None, radius_km: float = 10.0):
    """Run full fishing spot analysis.

    Args:
        target_date: Optional date string (YYYY-MM-DD) for analysis.
        user_lat: User's latitude for proximity search.
        user_lon: User's longitude for proximity search.
        radius_km: Search radius in kilometers (default 10km).
    """
    from controllers.analysis import AnalysisController
    from datetime import datetime
    from config import COASTLINE_FILE

    controller = AnalysisController()

    coastline_path = str(COASTLINE_FILE)
    output_path = "output/fishing_analysis_ml.html"

    if not os.path.exists(coastline_path):
        print(f"ERROR: Coastline file not found: {coastline_path}")
        print("Please ensure the coastline data is downloaded.")
        return 1

    # Parse target date if provided
    analysis_date = None
    if target_date:
        try:
            analysis_date = datetime.strptime(target_date, "%Y-%m-%d")
            print(f"Ejecutando analisis para fecha: {target_date}")
        except ValueError:
            print(f"ERROR: Formato de fecha invalido: {target_date}")
            print("Use formato YYYY-MM-DD (ejemplo: 2026-02-15)")
            return 1

    # Set user location for proximity search
    user_location = None
    if user_lat is not None and user_lon is not None:
        user_location = {'lat': user_lat, 'lon': user_lon, 'radius_km': radius_km}
        print(f"Busqueda por proximidad: ({user_lat:.4f}, {user_lon:.4f}) radio {radius_km}km")

    results = controller.run_full_analysis(
        coastline_path, output_path,
        target_date=analysis_date,
        user_location=user_location
    )

    # Open map in browser
    if results.get('map_path'):
        import webbrowser
        webbrowser.open(f"file://{Path(results['map_path']).absolute()}")

    return 0


def run_tests():
    """Run unit tests."""
    import pytest

    test_dir = Path(__file__).parent / "tests"
    return pytest.main([str(test_dir), '-v', '--tb=short'])


def run_supervised_analysis():
    """Run analysis with supervised ML (trained on historical data)."""
    from controllers.analysis import AnalysisController
    from config import COASTLINE_FILE

    controller = AnalysisController()

    coastline_path = str(COASTLINE_FILE)
    output_path = "output/fishing_analysis_supervised.html"

    if not os.path.exists(coastline_path):
        print(f"ERROR: Coastline file not found: {coastline_path}")
        return 1

    results = controller.run_full_analysis_supervised(
        coastline_path, output_path, train_years=4
    )

    if results.get('map_path'):
        import webbrowser
        webbrowser.open(f"file://{Path(results['map_path']).absolute()}")

    return 0


def download_historical():
    """Download historical data for training."""
    from controllers.analysis import AnalysisController

    controller = AnalysisController()
    results = controller.download_historical_data(years=4, sources=['noaa', 'gfw'])

    print("\nDownload Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fishing Predictor - ML-based fishing spot analysis"
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run unit tests'
    )
    parser.add_argument(
        '--supervised', '-s',
        action='store_true',
        help='Run with supervised ML (uses historical data)'
    )
    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Download historical data (4 years)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/fishing_analysis_ml.html',
        help='Output map path'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date for analysis (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--lat',
        type=float,
        default=None,
        help='Your latitude for proximity search (e.g., -17.8)'
    )
    parser.add_argument(
        '--lon',
        type=float,
        default=None,
        help='Your longitude for proximity search (e.g., -71.2)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=10.0,
        help='Search radius in km (default: 10)'
    )

    args = parser.parse_args()

    if args.test:
        return run_tests()
    elif args.download:
        return download_historical()
    elif args.supervised:
        return run_supervised_analysis()
    else:
        return run_analysis(
            target_date=args.date,
            user_lat=args.lat,
            user_lon=args.lon,
            radius_km=args.radius
        )


if __name__ == "__main__":
    sys.exit(main())
