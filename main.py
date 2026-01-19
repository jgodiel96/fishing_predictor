#!/usr/bin/env python3
"""
Fishing Predictor - Main Entry Point (MVC Architecture)

Usage:
    python main.py                    # Run full analysis
    python main.py --test             # Run tests
    python main.py --help             # Show help
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_analysis():
    """Run full fishing spot analysis."""
    from controllers.analysis import AnalysisController

    controller = AnalysisController()

    coastline_path = "data/cache/coastline_real_osm.geojson"
    output_path = "output/fishing_analysis_ml.html"

    if not os.path.exists(coastline_path):
        print(f"ERROR: Coastline file not found: {coastline_path}")
        print("Please ensure the coastline data is downloaded.")
        return 1

    results = controller.run_full_analysis(coastline_path, output_path)

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

    controller = AnalysisController()

    coastline_path = "data/cache/coastline_real_osm.geojson"
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

    args = parser.parse_args()

    if args.test:
        return run_tests()
    elif args.download:
        return download_historical()
    elif args.supervised:
        return run_supervised_analysis()
    else:
        return run_analysis()


if __name__ == "__main__":
    sys.exit(main())
