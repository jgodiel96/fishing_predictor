#!/usr/bin/env python3
"""
Migration script: Extract data from real_data_100.db and partition into Bronze layer.

This script performs a ONE-TIME migration from the legacy database structure
to the new Bronze/Silver/Gold architecture.

Operations:
1. Extract marine data from real_data_100.db → data/raw/open_meteo/YYYY-MM.parquet
2. Extract fishing data (GFW only) → data/raw/gfw/YYYY-MM.parquet
3. Extract user sightings → data/raw/user_sightings/sightings.jsonl
4. Generate manifests for each source
5. Optionally remove synthetic historical_* data (if found)

Usage:
    python scripts/migrate_existing_data.py [--dry-run] [--verbose]

Flags:
    --dry-run   Show what would be done without making changes
    --verbose   Print detailed progress information
"""

import sys
import os
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone
from calendar import monthrange

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from data.data_config import DataConfig
from data.manifest import ManifestManager


def get_month_dates(year: int, month: int) -> tuple:
    """Get first and last day of a month."""
    first_day = f"{year}-{month:02d}-01"
    last_day_num = monthrange(year, month)[1]
    last_day = f"{year}-{month:02d}-{last_day_num:02d}"
    return first_day, last_day


def extract_marine_data(conn: sqlite3.Connection, dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Extract marine data and partition by month.

    Returns:
        Dictionary with migration stats
    """
    print("\n" + "=" * 60)
    print("MIGRATING MARINE DATA (Open-Meteo ERA5)")
    print("=" * 60)

    # Read all marine data
    df = pd.read_sql_query(
        "SELECT date, lat, lon, wave_height, wave_period, wave_direction, "
        "wind_speed, wind_direction, source FROM marine WHERE source = 'open_meteo_era5'",
        conn
    )

    if df.empty:
        print("No marine data found!")
        return {'files_created': 0, 'records_migrated': 0}

    # Convert date to datetime for grouping
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')

    stats = {'files_created': 0, 'records_migrated': 0}
    manager = ManifestManager('open_meteo')

    # Group by month and save as parquet
    for period, group_df in df.groupby('year_month'):
        year = period.year
        month = period.month

        filename = f"{year}-{month:02d}.parquet"
        output_path = DataConfig.RAW_OPEN_METEO / filename

        # Prepare data for parquet (remove year_month column)
        export_df = group_df.drop(columns=['year_month']).copy()
        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')

        record_count = len(export_df)
        first_day, last_day = get_month_dates(year, month)

        if verbose:
            print(f"  {filename}: {record_count} records ({first_day} to {last_day})")

        if not dry_run:
            # Save parquet
            export_df.to_parquet(output_path, index=False)

            # Update manifest
            manager.add_download(
                filename=filename,
                period_start=first_day,
                period_end=last_day,
                records=record_count,
                source_url='migration_from_real_data_100.db',
                api_response_code=200
            )

            stats['files_created'] += 1
            stats['records_migrated'] += record_count

    if not dry_run:
        manager.save()
        print(f"\nMarine data migration complete:")
        print(f"  Files created: {stats['files_created']}")
        print(f"  Records migrated: {stats['records_migrated']:,}")
    else:
        print(f"\n[DRY RUN] Would create {len(df.groupby('year_month'))} files")
        print(f"[DRY RUN] Would migrate {len(df):,} records")

    return stats


def extract_fishing_data(conn: sqlite3.Connection, dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Extract GFW fishing data and partition by month.
    Excludes synthetic historical_* data.

    Returns:
        Dictionary with migration stats
    """
    print("\n" + "=" * 60)
    print("MIGRATING FISHING DATA (Global Fishing Watch)")
    print("=" * 60)

    # Read only GFW data (exclude synthetic historical_*)
    df = pd.read_sql_query(
        "SELECT date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source "
        "FROM fishing WHERE source = 'gfw_ais'",
        conn
    )

    # Check for synthetic data that will be excluded
    cursor = conn.cursor()
    cursor.execute("SELECT source, COUNT(*) FROM fishing WHERE source LIKE 'historical_%' GROUP BY source")
    synthetic = cursor.fetchall()
    if synthetic:
        print("\nExcluding synthetic data:")
        for source, count in synthetic:
            print(f"  {source}: {count} records (SYNTHETIC - NOT MIGRATED)")

    if df.empty:
        print("No GFW fishing data found!")
        return {'files_created': 0, 'records_migrated': 0, 'synthetic_excluded': sum(s[1] for s in synthetic)}

    # Convert date to datetime for grouping
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')

    stats = {'files_created': 0, 'records_migrated': 0, 'synthetic_excluded': sum(s[1] for s in synthetic) if synthetic else 0}
    manager = ManifestManager('gfw')

    # Group by month and save as parquet
    for period, group_df in df.groupby('year_month'):
        year = period.year
        month = period.month

        filename = f"{year}-{month:02d}.parquet"
        output_path = DataConfig.RAW_GFW / filename

        # Prepare data for parquet
        export_df = group_df.drop(columns=['year_month']).copy()
        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')

        record_count = len(export_df)
        first_day, last_day = get_month_dates(year, month)

        if verbose:
            print(f"  {filename}: {record_count} records ({first_day} to {last_day})")

        if not dry_run:
            # Save parquet
            export_df.to_parquet(output_path, index=False)

            # Update manifest
            manager.add_download(
                filename=filename,
                period_start=first_day,
                period_end=last_day,
                records=record_count,
                source_url='migration_from_real_data_100.db',
                api_response_code=200
            )

            stats['files_created'] += 1
            stats['records_migrated'] += record_count

    if not dry_run:
        manager.save()
        print(f"\nFishing data migration complete:")
        print(f"  Files created: {stats['files_created']}")
        print(f"  Records migrated: {stats['records_migrated']:,}")
        if stats['synthetic_excluded'] > 0:
            print(f"  Synthetic records excluded: {stats['synthetic_excluded']:,}")
    else:
        print(f"\n[DRY RUN] Would create {len(df.groupby('year_month'))} files")
        print(f"[DRY RUN] Would migrate {len(df):,} records")

    return stats


def extract_user_sightings(conn: sqlite3.Connection, dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Extract user-reported sightings to JSONL file.

    Returns:
        Dictionary with migration stats
    """
    print("\n" + "=" * 60)
    print("MIGRATING USER SIGHTINGS")
    print("=" * 60)

    # Read user-reported data
    df = pd.read_sql_query(
        "SELECT date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source "
        "FROM fishing WHERE source = 'user_report'",
        conn
    )

    if df.empty:
        print("No user sightings found.")
        return {'records_migrated': 0}

    output_path = DataConfig.RAW_USER_SIGHTINGS

    if verbose:
        print(f"Found {len(df)} user sightings")

    if not dry_run:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write as JSONL (append-only log format)
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                entry = {
                    'date': row['date'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'fishing_hours': row['fishing_hours'],
                    'vessel_id': row['vessel_id'],
                    'flag_state': row['flag_state'],
                    'gear_type': row['gear_type'],
                    'source': 'user_report',
                    'migrated_at': datetime.now(timezone.utc).isoformat()
                }
                f.write(json.dumps(entry) + '\n')

        print(f"\nUser sightings migration complete:")
        print(f"  Records migrated: {len(df)}")
        print(f"  Output file: {output_path}")
    else:
        print(f"\n[DRY RUN] Would migrate {len(df)} user sightings")

    return {'records_migrated': len(df)}


def create_migration_report(stats: dict, output_path: Path) -> None:
    """Create a JSON report of the migration."""
    report = {
        'migration_timestamp': datetime.now(timezone.utc).isoformat(),
        'source_database': str(DataConfig.LEGACY_DB),
        'stats': stats,
        'architecture': 'Bronze/Silver/Gold',
        'notes': [
            'Synthetic historical_* data was excluded from migration',
            'All GFW fishing data migrated to data/raw/gfw/',
            'All Open-Meteo marine data migrated to data/raw/open_meteo/',
            'User sightings migrated to data/raw/user_sightings/sightings.jsonl',
            'Original database preserved at data/real_only/real_data_100.db'
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nMigration report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate existing data to Bronze/Silver/Gold architecture'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress information')
    args = parser.parse_args()

    print("=" * 60)
    print("DATA MIGRATION: Legacy DB → Bronze Layer")
    print("=" * 60)
    print(f"Source: {DataConfig.LEGACY_DB}")
    print(f"Target: {DataConfig.RAW_DIR}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")

    # Check source database exists
    if not DataConfig.LEGACY_DB.exists():
        print(f"\nERROR: Source database not found: {DataConfig.LEGACY_DB}")
        sys.exit(1)

    # Ensure target directories exist
    if not args.dry_run:
        DataConfig.ensure_directories()

    # Connect to source database
    conn = sqlite3.connect(DataConfig.LEGACY_DB)

    try:
        # Migrate each data type
        stats = {
            'marine': extract_marine_data(conn, args.dry_run, args.verbose),
            'fishing': extract_fishing_data(conn, args.dry_run, args.verbose),
            'user_sightings': extract_user_sightings(conn, args.dry_run, args.verbose)
        }

        # Create migration report
        if not args.dry_run:
            report_path = DataConfig.METADATA_DIR / 'migration_report.json'
            create_migration_report(stats, report_path)

        # Summary
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)

        total_files = sum(s.get('files_created', 0) for s in stats.values())
        total_records = sum(s.get('records_migrated', 0) for s in stats.values())
        synthetic_excluded = stats['fishing'].get('synthetic_excluded', 0)

        if args.dry_run:
            print("[DRY RUN] No changes were made")
        else:
            print(f"Total files created: {total_files}")
            print(f"Total records migrated: {total_records:,}")
            if synthetic_excluded > 0:
                print(f"Synthetic records excluded: {synthetic_excluded:,}")

        print("\nNext steps:")
        print("1. Run 'python scripts/validate_data.py' to verify migration")
        print("2. Run 'python -m data.consolidator' to generate Silver layer")
        print("3. Update code to use new paths from data.data_config")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
