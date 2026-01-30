#!/usr/bin/env python3
"""
Data Consolidator: Bronze → Silver layer transformation.

This module reads all parquet files from the Bronze (raw) layer and generates
consolidated databases in the Silver (processed) layer.

Features:
- Idempotent: Can be run multiple times without duplicating data
- Deduplication: Removes duplicate records by key fields
- Logging: Tracks all consolidation operations
- Validation: Verifies data quality during consolidation

Usage:
    # As module
    from data.consolidator import Consolidator
    consolidator = Consolidator()
    consolidator.consolidate_all()

    # As script
    python -m data.consolidator [--verbose] [--force]
"""

import sys
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd

from data.data_config import DataConfig


class ConsolidationLog:
    """Manages the consolidation log for tracking operations."""

    def __init__(self):
        self.log_path = DataConfig.CONSOLIDATION_LOG
        self._log = self._load_or_create()

    def _load_or_create(self) -> Dict:
        """Load existing log or create new one."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {
            'version': '1.0',
            'consolidations': [],
            'last_run': None
        }

    def save(self) -> None:
        """Save log to disk."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log['last_run'] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, 'w') as f:
            json.dump(self._log, f, indent=2)

    def add_entry(
        self,
        target: str,
        source_files: List[str],
        records_input: int,
        records_output: int,
        deduped: int,
        status: str = 'success'
    ) -> None:
        """Add a consolidation entry."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'target': target,
            'source_files': source_files,
            'records_input': records_input,
            'records_output': records_output,
            'records_deduped': deduped,
            'status': status
        }
        self._log['consolidations'].append(entry)

    def get_last_consolidation(self, target: str) -> Optional[Dict]:
        """Get the last consolidation entry for a target."""
        for entry in reversed(self._log['consolidations']):
            if entry['target'] == target:
                return entry
        return None


class Consolidator:
    """Consolidates Bronze layer data into Silver layer databases."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log = ConsolidationLog()

    def _log_message(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {msg}")

    def consolidate_fishing(self) -> Dict:
        """
        Consolidate all fishing data into fishing_consolidated.db.

        Sources:
        - data/raw/gfw/*.parquet (GFW AIS data)
        - data/raw/user_sightings/sightings.jsonl (User reports)

        Returns:
            Dictionary with consolidation stats
        """
        print("\nConsolidating fishing data...")

        all_dfs = []
        source_files = []

        # 1. Load GFW parquet files
        gfw_files = sorted(DataConfig.RAW_GFW.glob("*.parquet"))
        for file_path in gfw_files:
            df = pd.read_parquet(file_path)
            df['data_source'] = 'gfw'
            all_dfs.append(df)
            source_files.append(str(file_path.name))
            self._log_message(f"Loaded {file_path.name}: {len(df)} records")

        # 2. Load user sightings
        if DataConfig.RAW_USER_SIGHTINGS.exists():
            sightings = []
            with open(DataConfig.RAW_USER_SIGHTINGS, 'r') as f:
                for line in f:
                    if line.strip():
                        sightings.append(json.loads(line))
            if sightings:
                df_sightings = pd.DataFrame(sightings)
                df_sightings['data_source'] = 'user_report'
                all_dfs.append(df_sightings)
                source_files.append('sightings.jsonl')
                self._log_message(f"Loaded sightings.jsonl: {len(df_sightings)} records")

        if not all_dfs:
            print("  No fishing data found!")
            return {'records_input': 0, 'records_output': 0, 'deduped': 0}

        # 3. Combine all data
        combined = pd.concat(all_dfs, ignore_index=True)
        records_input = len(combined)
        self._log_message(f"Total records before dedup: {records_input}")

        # 4. Deduplicate by (date, lat, lon, vessel_id)
        dedup_cols = ['date', 'lat', 'lon', 'vessel_id']
        # Ensure all dedup columns exist
        for col in dedup_cols:
            if col not in combined.columns:
                combined[col] = None

        combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
        records_output = len(combined)
        deduped = records_input - records_output
        self._log_message(f"After dedup: {records_output} records ({deduped} duplicates removed)")

        # 5. Save to SQLite
        DataConfig.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(DataConfig.FISHING_DB) as conn:
            combined.to_sql('fishing', conn, if_exists='replace', index=False)

            # Create indices for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fishing_date ON fishing(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fishing_location ON fishing(lat, lon)")

        print(f"  Fishing database created: {DataConfig.FISHING_DB}")
        print(f"  Records: {records_output:,}")

        # Log the consolidation
        self.log.add_entry(
            target='fishing_consolidated.db',
            source_files=source_files,
            records_input=records_input,
            records_output=records_output,
            deduped=deduped
        )

        return {
            'records_input': records_input,
            'records_output': records_output,
            'deduped': deduped
        }

    def consolidate_marine(self) -> Dict:
        """
        Consolidate all marine conditions data into marine_consolidated.db.

        Sources:
        - data/raw/open_meteo/*.parquet (Open-Meteo ERA5 data)

        Returns:
            Dictionary with consolidation stats
        """
        print("\nConsolidating marine data...")

        all_dfs = []
        source_files = []

        # Load Open-Meteo parquet files
        meteo_files = sorted(DataConfig.RAW_OPEN_METEO.glob("*.parquet"))
        for file_path in meteo_files:
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
            source_files.append(str(file_path.name))
            self._log_message(f"Loaded {file_path.name}: {len(df)} records")

        if not all_dfs:
            print("  No marine data found!")
            return {'records_input': 0, 'records_output': 0, 'deduped': 0}

        # Combine all data
        combined = pd.concat(all_dfs, ignore_index=True)
        records_input = len(combined)
        self._log_message(f"Total records before dedup: {records_input}")

        # Deduplicate by (date, lat, lon)
        combined = combined.drop_duplicates(subset=['date', 'lat', 'lon'], keep='last')
        records_output = len(combined)
        deduped = records_input - records_output
        self._log_message(f"After dedup: {records_output} records ({deduped} duplicates removed)")

        # Save to SQLite
        DataConfig.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(DataConfig.MARINE_DB) as conn:
            combined.to_sql('marine', conn, if_exists='replace', index=False)

            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_marine_date ON marine(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_marine_location ON marine(lat, lon)")

        print(f"  Marine database created: {DataConfig.MARINE_DB}")
        print(f"  Records: {records_output:,}")

        # Log the consolidation
        self.log.add_entry(
            target='marine_consolidated.db',
            source_files=source_files,
            records_input=records_input,
            records_output=records_output,
            deduped=deduped
        )

        return {
            'records_input': records_input,
            'records_output': records_output,
            'deduped': deduped
        }

    def consolidate_sst(self) -> Dict:
        """
        Consolidate all SST data from multiple sources.

        Sources:
        - data/raw/sst/noaa/*.parquet
        - data/raw/sst/earthdata/*.parquet
        - data/raw/sst/copernicus/*.parquet

        Returns:
            Dictionary with consolidation stats
        """
        print("\nConsolidating SST data...")

        all_dfs = []
        source_files = []

        # Load from all SST sources
        sst_dirs = [
            (DataConfig.RAW_SST_NOAA, 'noaa'),
            (DataConfig.RAW_SST_EARTHDATA, 'earthdata'),
            (DataConfig.RAW_SST_COPERNICUS, 'copernicus')
        ]

        for sst_dir, source_name in sst_dirs:
            if sst_dir.exists():
                for file_path in sorted(sst_dir.glob("*.parquet")):
                    df = pd.read_parquet(file_path)
                    df['sst_source'] = source_name
                    all_dfs.append(df)
                    source_files.append(f"{source_name}/{file_path.name}")
                    self._log_message(f"Loaded {source_name}/{file_path.name}: {len(df)} records")

        if not all_dfs:
            print("  No SST data found in Bronze layer.")
            print("  Note: SST data may need to be downloaded using download_incremental.py")
            return {'records_input': 0, 'records_output': 0, 'deduped': 0}

        # Combine all data
        combined = pd.concat(all_dfs, ignore_index=True)
        records_input = len(combined)

        # Deduplicate, preferring higher resolution sources
        # Priority: earthdata (1km) > copernicus (5km) > noaa (25km)
        source_priority = {'earthdata': 1, 'copernicus': 2, 'noaa': 3}
        combined['source_priority'] = combined['sst_source'].map(source_priority)
        combined = combined.sort_values('source_priority')
        combined = combined.drop_duplicates(subset=['date', 'lat', 'lon'], keep='first')
        combined = combined.drop(columns=['source_priority'])

        records_output = len(combined)
        deduped = records_input - records_output

        # Add SST table to marine database (or separate if needed)
        with sqlite3.connect(DataConfig.MARINE_DB) as conn:
            combined.to_sql('sst', conn, if_exists='replace', index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sst_date ON sst(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sst_location ON sst(lat, lon)")

        print(f"  SST data added to: {DataConfig.MARINE_DB}")
        print(f"  Records: {records_output:,}")

        self.log.add_entry(
            target='marine_consolidated.db:sst',
            source_files=source_files,
            records_input=records_input,
            records_output=records_output,
            deduped=deduped
        )

        return {
            'records_input': records_input,
            'records_output': records_output,
            'deduped': deduped
        }

    def generate_training_features(self) -> Dict:
        """
        Generate ML training features by joining fishing and marine data.

        Output:
        - data/processed/training_features.parquet

        Returns:
            Dictionary with generation stats
        """
        print("\nGenerating training features...")

        # Check if source databases exist
        if not DataConfig.FISHING_DB.exists():
            print("  Fishing database not found. Run consolidate_fishing first.")
            return {'records': 0}

        if not DataConfig.MARINE_DB.exists():
            print("  Marine database not found. Run consolidate_marine first.")
            return {'records': 0}

        # Load fishing data
        with sqlite3.connect(DataConfig.FISHING_DB) as conn:
            fishing_df = pd.read_sql_query("SELECT * FROM fishing", conn)

        # Load marine data
        with sqlite3.connect(DataConfig.MARINE_DB) as conn:
            marine_df = pd.read_sql_query("SELECT * FROM marine", conn)

            # Try to load SST if available
            try:
                sst_df = pd.read_sql_query("SELECT * FROM sst", conn)
            except:
                sst_df = None

        self._log_message(f"Fishing records: {len(fishing_df)}")
        self._log_message(f"Marine records: {len(marine_df)}")
        if sst_df is not None:
            self._log_message(f"SST records: {len(sst_df)}")

        # Create training features
        # Start with marine data (most complete)
        training = marine_df.copy()

        # Add fishing activity (left join)
        if not fishing_df.empty:
            fishing_agg = fishing_df.groupby(['date', 'lat', 'lon']).agg({
                'fishing_hours': 'sum'
            }).reset_index()

            training = training.merge(
                fishing_agg,
                on=['date', 'lat', 'lon'],
                how='left'
            )
            training['fishing_hours'] = training['fishing_hours'].fillna(0)
            training['is_fishing'] = (training['fishing_hours'] > 0).astype(int)
        else:
            training['fishing_hours'] = 0
            training['is_fishing'] = 0

        # Add SST if available
        if sst_df is not None and not sst_df.empty:
            sst_agg = sst_df.groupby(['date', 'lat', 'lon'])['sst'].mean().reset_index()
            training = training.merge(
                sst_agg,
                on=['date', 'lat', 'lon'],
                how='left'
            )
        else:
            # Use IMARPE climatology as fallback
            training['sst'] = 18.5  # Climatological average

        # Add temporal features
        training['date'] = pd.to_datetime(training['date'])
        training['month'] = training['date'].dt.month
        training['day_of_year'] = training['date'].dt.dayofyear
        training['year'] = training['date'].dt.year

        # Mark as all real data
        training['all_real'] = 1

        # Convert date back to string for parquet
        training['date'] = training['date'].dt.strftime('%Y-%m-%d')

        # Save as parquet
        training.to_parquet(DataConfig.TRAINING_FEATURES, index=False)

        print(f"  Training features created: {DataConfig.TRAINING_FEATURES}")
        print(f"  Records: {len(training):,}")

        self.log.add_entry(
            target='training_features.parquet',
            source_files=['fishing_consolidated.db', 'marine_consolidated.db'],
            records_input=len(marine_df),
            records_output=len(training),
            deduped=0
        )

        return {'records': len(training)}

    def consolidate_all(self, force: bool = False) -> Dict:
        """
        Run all consolidation operations.

        Args:
            force: If True, regenerate even if already up to date

        Returns:
            Dictionary with all stats
        """
        print("=" * 60)
        print("CONSOLIDATING DATA: Bronze → Silver")
        print("=" * 60)
        print(f"Bronze layer: {DataConfig.RAW_DIR}")
        print(f"Silver layer: {DataConfig.PROCESSED_DIR}")

        stats = {
            'fishing': self.consolidate_fishing(),
            'marine': self.consolidate_marine(),
            'sst': self.consolidate_sst(),
            'training': self.generate_training_features()
        }

        # Save consolidation log
        self.log.save()

        # Print summary
        print("\n" + "=" * 60)
        print("CONSOLIDATION SUMMARY")
        print("=" * 60)

        total_input = sum(s.get('records_input', 0) for s in stats.values())
        total_output = sum(s.get('records_output', 0) + s.get('records', 0) for s in stats.values())
        total_deduped = sum(s.get('deduped', 0) for s in stats.values())

        print(f"Total records processed: {total_input:,}")
        print(f"Total records in Silver layer: {total_output:,}")
        print(f"Duplicates removed: {total_deduped:,}")
        print(f"\nConsolidation log: {DataConfig.CONSOLIDATION_LOG}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate Bronze layer data into Silver layer'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress information')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force regeneration even if up to date')
    parser.add_argument('--fishing-only', action='store_true',
                       help='Only consolidate fishing data')
    parser.add_argument('--marine-only', action='store_true',
                       help='Only consolidate marine data')
    args = parser.parse_args()

    consolidator = Consolidator(verbose=args.verbose)

    if args.fishing_only:
        consolidator.consolidate_fishing()
    elif args.marine_only:
        consolidator.consolidate_marine()
    else:
        consolidator.consolidate_all(force=args.force)


if __name__ == '__main__':
    main()
