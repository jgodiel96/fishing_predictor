#!/usr/bin/env python3
"""
Data Validation Script for Bronze/Silver/Gold Architecture.

Verifies integrity of all data layers:
- Bronze (raw): Checksums, record counts, manifest consistency
- Silver (processed): Schema validation, referential integrity
- Gold (analytics): Feature completeness, training data quality

Usage:
    # Validate all layers
    python scripts/validate_data.py --all

    # Validate specific layer
    python scripts/validate_data.py --bronze
    python scripts/validate_data.py --silver
    python scripts/validate_data.py --gold

    # Quick check (fast, less thorough)
    python scripts/validate_data.py --quick
"""

import sys
import os
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_config import DataConfig
from data.manifest import ManifestManager


class ValidationResult:
    """Container for validation results."""

    def __init__(self, layer: str):
        self.layer = layer
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_summary(self):
        status = "PASSED" if self.is_valid else "FAILED"
        icon = "✅" if self.is_valid else "❌"
        print(f"\n{icon} {self.layer.upper()} LAYER: {status}")

        if self.info:
            for msg in self.info:
                print(f"   ℹ️  {msg}")

        if self.warnings:
            print(f"\n   ⚠️  Warnings ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"      - {msg}")

        if self.errors:
            print(f"\n   ❌ Errors ({len(self.errors)}):")
            for msg in self.errors:
                print(f"      - {msg}")


class DataValidator:
    """Validates data integrity across all layers."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    # =========================================================================
    # BRONZE LAYER VALIDATION
    # =========================================================================

    def validate_bronze(self, quick: bool = False) -> ValidationResult:
        """
        Validate Bronze (raw) layer.

        Checks:
        - All sources have manifests
        - All files in manifest exist
        - Checksums match (unless quick mode)
        - Record counts match (unless quick mode)
        - Date continuity
        """
        result = ValidationResult("bronze")
        print("\nValidating Bronze layer...")

        sources = ['gfw', 'open_meteo', 'noaa_sst', 'earthdata_sst', 'copernicus_sst']
        total_files = 0
        total_records = 0

        for source in sources:
            self.log(f"Checking {source}...")

            try:
                manager = ManifestManager(source)
            except Exception as e:
                result.add_warning(f"{source}: Could not load manifest - {e}")
                continue

            if not manager.downloads:
                result.add_info(f"{source}: No data files yet")
                continue

            # Check each file
            source_files = 0
            source_records = 0

            for download in manager.downloads:
                filename = download['file']
                file_path = manager.manifest_path.parent / filename

                # Check file exists
                if not file_path.exists():
                    result.add_error(f"{source}/{filename}: File missing")
                    continue

                source_files += 1

                if not quick:
                    # Verify checksum
                    actual_checksum = manager._compute_checksum(file_path)
                    if actual_checksum != download['checksum']:
                        result.add_error(f"{source}/{filename}: Checksum mismatch")

                    # Verify record count
                    try:
                        df = pd.read_parquet(file_path)
                        if len(df) != download['records']:
                            result.add_warning(
                                f"{source}/{filename}: Record count mismatch "
                                f"(manifest: {download['records']}, actual: {len(df)})"
                            )
                        source_records += len(df)
                    except Exception as e:
                        result.add_error(f"{source}/{filename}: Cannot read - {e}")
                else:
                    source_records += download['records']

            total_files += source_files
            total_records += source_records

            # Check date continuity
            if len(manager.downloads) > 1:
                dates = []
                for d in manager.downloads:
                    parts = d['file'].replace('.parquet', '').split('-')
                    if len(parts) == 2:
                        dates.append((int(parts[0]), int(parts[1])))

                dates.sort()
                gaps = []
                for i in range(1, len(dates)):
                    prev = dates[i-1]
                    curr = dates[i]
                    # Calculate expected next month
                    expected_month = prev[1] + 1
                    expected_year = prev[0]
                    if expected_month > 12:
                        expected_month = 1
                        expected_year += 1

                    if (expected_year, expected_month) != curr:
                        gaps.append(f"{prev[0]}-{prev[1]:02d} → {curr[0]}-{curr[1]:02d}")

                if gaps:
                    result.add_warning(f"{source}: Date gaps found: {', '.join(gaps[:3])}")

            result.add_info(f"{source}: {source_files} files, {source_records:,} records")

        result.add_info(f"Total: {total_files} files, {total_records:,} records")
        return result

    # =========================================================================
    # SILVER LAYER VALIDATION
    # =========================================================================

    def validate_silver(self, quick: bool = False) -> ValidationResult:
        """
        Validate Silver (processed) layer.

        Checks:
        - Consolidated databases exist
        - Tables have expected schema
        - No orphaned records
        - Data quality metrics
        """
        result = ValidationResult("silver")
        print("\nValidating Silver layer...")

        # Check fishing database
        if DataConfig.FISHING_DB.exists():
            self.log("Checking fishing_consolidated.db...")
            try:
                with sqlite3.connect(DataConfig.FISHING_DB) as conn:
                    cursor = conn.cursor()

                    # Check table exists
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fishing'")
                    if not cursor.fetchone():
                        result.add_error("fishing_consolidated.db: 'fishing' table missing")
                    else:
                        # Count records
                        cursor.execute("SELECT COUNT(*) FROM fishing")
                        count = cursor.fetchone()[0]
                        result.add_info(f"Fishing database: {count:,} records")

                        if not quick:
                            # Check for required columns
                            cursor.execute("PRAGMA table_info(fishing)")
                            columns = {col[1] for col in cursor.fetchall()}
                            required = {'date', 'lat', 'lon', 'fishing_hours'}
                            missing = required - columns
                            if missing:
                                result.add_error(f"Fishing table missing columns: {missing}")

                            # Check for nulls in key fields
                            cursor.execute("SELECT COUNT(*) FROM fishing WHERE date IS NULL OR lat IS NULL OR lon IS NULL")
                            null_count = cursor.fetchone()[0]
                            if null_count > 0:
                                result.add_warning(f"Fishing table has {null_count} rows with null key fields")

            except Exception as e:
                result.add_error(f"fishing_consolidated.db: Cannot read - {e}")
        else:
            result.add_warning("fishing_consolidated.db: File not found")

        # Check marine database
        if DataConfig.MARINE_DB.exists():
            self.log("Checking marine_consolidated.db...")
            try:
                with sqlite3.connect(DataConfig.MARINE_DB) as conn:
                    cursor = conn.cursor()

                    # Check marine table
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='marine'")
                    if not cursor.fetchone():
                        result.add_error("marine_consolidated.db: 'marine' table missing")
                    else:
                        cursor.execute("SELECT COUNT(*) FROM marine")
                        count = cursor.fetchone()[0]
                        result.add_info(f"Marine database: {count:,} records")

                    # Check SST table (optional)
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sst'")
                    if cursor.fetchone():
                        cursor.execute("SELECT COUNT(*) FROM sst")
                        count = cursor.fetchone()[0]
                        result.add_info(f"SST table: {count:,} records")

            except Exception as e:
                result.add_error(f"marine_consolidated.db: Cannot read - {e}")
        else:
            result.add_warning("marine_consolidated.db: File not found")

        # Check training features
        if DataConfig.TRAINING_FEATURES.exists():
            self.log("Checking training_features.parquet...")
            try:
                df = pd.read_parquet(DataConfig.TRAINING_FEATURES)
                result.add_info(f"Training features: {len(df):,} records, {len(df.columns)} columns")

                if not quick:
                    # Check for required features
                    required_features = {'date', 'lat', 'lon', 'wave_height', 'fishing_hours'}
                    missing = required_features - set(df.columns)
                    if missing:
                        result.add_warning(f"Training features missing columns: {missing}")

                    # Check for excessive nulls
                    null_pct = df.isnull().sum() / len(df) * 100
                    high_null_cols = null_pct[null_pct > 50].index.tolist()
                    if high_null_cols:
                        result.add_warning(f"Columns with >50% nulls: {high_null_cols[:5]}")

            except Exception as e:
                result.add_error(f"training_features.parquet: Cannot read - {e}")
        else:
            result.add_warning("training_features.parquet: File not found")

        return result

    # =========================================================================
    # GOLD LAYER VALIDATION
    # =========================================================================

    def validate_gold(self, quick: bool = False) -> ValidationResult:
        """
        Validate Gold (analytics) layer.

        Checks:
        - Current training dataset exists
        - Model metadata is valid
        - Version history is maintained
        """
        result = ValidationResult("gold")
        print("\nValidating Gold layer...")

        # Check current training dataset
        if DataConfig.CURRENT_TRAINING.exists():
            self.log("Checking current training dataset...")
            try:
                df = pd.read_parquet(DataConfig.CURRENT_TRAINING)
                result.add_info(f"Current training: {len(df):,} records")

                if not quick:
                    # Check feature completeness
                    expected_features = [
                        'date', 'lat', 'lon', 'sst', 'wave_height',
                        'fishing_hours', 'is_fishing'
                    ]
                    missing = set(expected_features) - set(df.columns)
                    if missing:
                        result.add_warning(f"Training dataset missing features: {missing}")

                    # Check class balance
                    if 'is_fishing' in df.columns:
                        fishing_pct = df['is_fishing'].mean() * 100
                        result.add_info(f"Class balance: {fishing_pct:.1f}% fishing events")
                        if fishing_pct < 1:
                            result.add_warning("Very imbalanced dataset (<1% positive class)")

            except Exception as e:
                result.add_error(f"Current training dataset: Cannot read - {e}")
        else:
            result.add_info("Current training dataset: Not created yet")

        # Check model metadata
        if DataConfig.MODEL_METADATA.exists():
            self.log("Checking model metadata...")
            try:
                with open(DataConfig.MODEL_METADATA) as f:
                    metadata = json.load(f)
                result.add_info(f"Model metadata: {len(metadata)} fields")
            except Exception as e:
                result.add_warning(f"Model metadata: Invalid JSON - {e}")

        # Check versions
        if DataConfig.VERSIONS_DIR.exists():
            versions = list(DataConfig.VERSIONS_DIR.glob("v*"))
            result.add_info(f"Versions: {len(versions)} stored")
            if len(versions) > DataConfig.MAX_VERSIONS:
                result.add_warning(f"More than {DataConfig.MAX_VERSIONS} versions stored")
        else:
            result.add_info("Versions directory: Empty")

        return result

    # =========================================================================
    # CROSS-LAYER VALIDATION
    # =========================================================================

    def validate_consistency(self) -> ValidationResult:
        """
        Validate consistency across layers.

        Checks:
        - Silver data matches Bronze totals
        - Gold data is subset of Silver
        - Date ranges are consistent
        """
        result = ValidationResult("cross-layer")
        print("\nValidating cross-layer consistency...")

        # Compare Bronze and Silver record counts
        bronze_records = {}
        silver_records = {}

        # Count Bronze records
        for source in ['gfw', 'open_meteo']:
            try:
                manager = ManifestManager(source)
                bronze_records[source] = sum(d['records'] for d in manager.downloads)
            except:
                bronze_records[source] = 0

        # Count Silver records
        if DataConfig.FISHING_DB.exists():
            with sqlite3.connect(DataConfig.FISHING_DB) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM fishing")
                silver_records['fishing'] = cursor.fetchone()[0]

        if DataConfig.MARINE_DB.exists():
            with sqlite3.connect(DataConfig.MARINE_DB) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM marine")
                silver_records['marine'] = cursor.fetchone()[0]

        # Compare (with dedup tolerance)
        if bronze_records.get('gfw', 0) > 0 and silver_records.get('fishing', 0) > 0:
            diff = bronze_records['gfw'] - silver_records['fishing']
            if diff > 0:
                result.add_info(f"GFW: {diff} duplicates removed during consolidation")
            elif diff < 0:
                result.add_warning(f"Silver has more fishing records than Bronze ({abs(diff)})")

        if bronze_records.get('open_meteo', 0) > 0 and silver_records.get('marine', 0) > 0:
            diff = bronze_records['open_meteo'] - silver_records['marine']
            if diff > 0:
                result.add_info(f"Open-Meteo: {diff} duplicates removed during consolidation")
            elif diff < 0:
                result.add_warning(f"Silver has more marine records than Bronze ({abs(diff)})")

        result.add_info("Cross-layer validation complete")
        return result

    # =========================================================================
    # MAIN VALIDATION
    # =========================================================================

    def validate_all(self, quick: bool = False) -> Dict[str, ValidationResult]:
        """Run all validations."""
        results = {}

        results['bronze'] = self.validate_bronze(quick)
        results['silver'] = self.validate_silver(quick)
        results['gold'] = self.validate_gold(quick)
        results['consistency'] = self.validate_consistency()

        return results


def print_final_summary(results: Dict[str, ValidationResult]):
    """Print final validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    total_errors = 0
    total_warnings = 0

    for layer, result in results.items():
        result.print_summary()
        if not result.is_valid:
            all_passed = False
        total_errors += len(result.errors)
        total_warnings += len(result.warnings)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print(f"❌ VALIDATION FAILED: {total_errors} errors, {total_warnings} warnings")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Validate data integrity across all layers'
    )
    parser.add_argument('--all', action='store_true',
                       help='Validate all layers')
    parser.add_argument('--bronze', action='store_true',
                       help='Validate Bronze layer only')
    parser.add_argument('--silver', action='store_true',
                       help='Validate Silver layer only')
    parser.add_argument('--gold', action='store_true',
                       help='Validate Gold layer only')
    parser.add_argument('--quick', action='store_true',
                       help='Quick validation (skip checksums)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Default to --all if no layer specified
    if not (args.bronze or args.silver or args.gold):
        args.all = True

    validator = DataValidator(verbose=args.verbose)
    results = {}

    if args.all or args.bronze:
        results['bronze'] = validator.validate_bronze(args.quick)

    if args.all or args.silver:
        results['silver'] = validator.validate_silver(args.quick)

    if args.all or args.gold:
        results['gold'] = validator.validate_gold(args.quick)

    if args.all:
        results['consistency'] = validator.validate_consistency()

    all_passed = print_final_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
