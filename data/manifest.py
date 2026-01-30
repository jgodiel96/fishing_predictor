"""
Manifest management for Bronze layer data tracking.

Each data source in the Bronze layer has a _manifest.json file that tracks:
- All downloaded files with their metadata
- Checksums for integrity verification
- Download timestamps and API response codes
- Record counts for validation

Usage:
    from data.manifest import ManifestManager

    manager = ManifestManager('gfw')
    manager.add_download('2024-01.parquet', records=1527, source_url='...')
    manager.save()

    # Verify integrity
    errors = manager.verify_all()
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from data.data_config import DataConfig


@dataclass
class DownloadEntry:
    """Metadata for a single downloaded file."""
    file: str
    period_start: str  # ISO date: YYYY-MM-DD
    period_end: str    # ISO date: YYYY-MM-DD
    downloaded_at: str  # ISO datetime with timezone
    records: int
    checksum: str      # sha256:hexdigest
    api_response_code: int
    source_url: str
    file_size_bytes: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> 'DownloadEntry':
        """Create DownloadEntry from dictionary."""
        return cls(
            file=data['file'],
            period_start=data['period']['start'],
            period_end=data['period']['end'],
            downloaded_at=data['downloaded_at'],
            records=data['records'],
            checksum=data['checksum'],
            api_response_code=data.get('api_response_code', 200),
            source_url=data.get('source_url', ''),
            file_size_bytes=data.get('file_size_bytes', 0)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'file': self.file,
            'period': {
                'start': self.period_start,
                'end': self.period_end
            },
            'downloaded_at': self.downloaded_at,
            'records': self.records,
            'checksum': self.checksum,
            'api_response_code': self.api_response_code,
            'source_url': self.source_url,
            'file_size_bytes': self.file_size_bytes
        }


class ManifestManager:
    """Manages manifest files for data source tracking."""

    def __init__(self, source: str):
        """
        Initialize manifest manager for a data source.

        Args:
            source: Data source name (gfw, open_meteo, noaa_sst, etc.)
        """
        self.source = source
        self.manifest_path = DataConfig.get_manifest_path(source)
        self.source_config = DataConfig.SOURCES.get(source, {})
        self._manifest = self._load_or_create()

    def _load_or_create(self) -> Dict:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            'source': self.source,
            'source_name': self.source_config.get('name', self.source),
            'version': '1.0',
            'downloads': [],
            'last_updated': None,
            'total_records': 0,
            'date_range': {
                'earliest': None,
                'latest': None
            }
        }

    def save(self) -> None:
        """Save manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._update_stats()
        with open(self.manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)

    def _update_stats(self) -> None:
        """Update aggregate statistics in manifest."""
        downloads = self._manifest['downloads']
        if not downloads:
            return

        self._manifest['total_records'] = sum(d['records'] for d in downloads)
        self._manifest['last_updated'] = datetime.now(timezone.utc).isoformat()

        # Find date range
        periods = [(d['period']['start'], d['period']['end']) for d in downloads]
        if periods:
            self._manifest['date_range']['earliest'] = min(p[0] for p in periods)
            self._manifest['date_range']['latest'] = max(p[1] for p in periods)

    def add_download(
        self,
        filename: str,
        period_start: str,
        period_end: str,
        records: int,
        source_url: str = '',
        api_response_code: int = 200
    ) -> None:
        """
        Add or update a download entry in the manifest.

        Args:
            filename: Name of the parquet file (e.g., '2024-01.parquet')
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            records: Number of records in the file
            source_url: URL used to fetch the data
            api_response_code: HTTP response code from API
        """
        # Get full path and compute checksum
        file_path = self.manifest_path.parent / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        checksum = self._compute_checksum(file_path)
        file_size = file_path.stat().st_size

        entry = DownloadEntry(
            file=filename,
            period_start=period_start,
            period_end=period_end,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            records=records,
            checksum=checksum,
            api_response_code=api_response_code,
            source_url=source_url,
            file_size_bytes=file_size
        )

        # Remove existing entry for same file if present
        self._manifest['downloads'] = [
            d for d in self._manifest['downloads']
            if d['file'] != filename
        ]

        # Add new entry
        self._manifest['downloads'].append(entry.to_dict())

        # Sort by filename (chronological order)
        self._manifest['downloads'].sort(key=lambda x: x['file'])

    def get_download(self, filename: str) -> Optional[DownloadEntry]:
        """Get download entry by filename."""
        for d in self._manifest['downloads']:
            if d['file'] == filename:
                return DownloadEntry.from_dict(d)
        return None

    def file_exists(self, year: int, month: int) -> bool:
        """Check if a file for a specific month already exists in manifest."""
        filename = f"{year}-{month:02d}.parquet"
        return any(d['file'] == filename for d in self._manifest['downloads'])

    def get_missing_months(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> List[tuple]:
        """
        Get list of (year, month) tuples that are missing from manifest.

        Args:
            start_year: Start year
            start_month: Start month (1-12)
            end_year: End year
            end_month: End month (1-12)

        Returns:
            List of (year, month) tuples for missing data
        """
        existing = set()
        for d in self._manifest['downloads']:
            # Parse filename like '2024-01.parquet'
            parts = d['file'].replace('.parquet', '').split('-')
            if len(parts) == 2:
                existing.add((int(parts[0]), int(parts[1])))

        missing = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            if (year, month) not in existing:
                missing.append((year, month))
            # Increment month
            month += 1
            if month > 12:
                month = 1
                year += 1

        return missing

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def verify_file(self, filename: str) -> List[str]:
        """
        Verify integrity of a single file.

        Args:
            filename: Name of the file to verify

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        entry = self.get_download(filename)
        if not entry:
            errors.append(f"File not in manifest: {filename}")
            return errors

        file_path = self.manifest_path.parent / filename

        # Check file exists
        if not file_path.exists():
            errors.append(f"File missing: {filename}")
            return errors

        # Verify checksum
        actual_checksum = self._compute_checksum(file_path)
        if actual_checksum != entry.checksum:
            errors.append(f"Checksum mismatch: {filename}")

        # Verify record count (requires pandas)
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            if len(df) != entry.records:
                errors.append(
                    f"Record count mismatch in {filename}: "
                    f"expected {entry.records}, got {len(df)}"
                )
        except ImportError:
            pass  # Skip record count verification if pandas not available
        except Exception as e:
            errors.append(f"Error reading {filename}: {e}")

        return errors

    def verify_all(self) -> List[str]:
        """
        Verify integrity of all files in manifest.

        Returns:
            List of all error messages
        """
        all_errors = []
        for download in self._manifest['downloads']:
            errors = self.verify_file(download['file'])
            all_errors.extend(errors)
        return all_errors

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'source': self.source,
            'source_name': self._manifest.get('source_name', self.source),
            'file_count': len(self._manifest['downloads']),
            'total_records': self._manifest.get('total_records', 0),
            'date_range': self._manifest.get('date_range', {}),
            'last_updated': self._manifest.get('last_updated')
        }

    @property
    def downloads(self) -> List[Dict]:
        """Get all download entries."""
        return self._manifest['downloads']


def create_all_manifests() -> Dict[str, ManifestManager]:
    """
    Create manifest managers for all data sources.

    Returns:
        Dictionary of source name to ManifestManager
    """
    managers = {}
    for source in ['gfw', 'open_meteo', 'noaa_sst', 'earthdata_sst', 'copernicus_sst']:
        managers[source] = ManifestManager(source)
    return managers


def verify_all_sources() -> Dict[str, List[str]]:
    """
    Verify integrity of all data sources.

    Returns:
        Dictionary of source name to list of errors
    """
    results = {}
    for source in ['gfw', 'open_meteo', 'noaa_sst', 'earthdata_sst', 'copernicus_sst']:
        try:
            manager = ManifestManager(source)
            results[source] = manager.verify_all()
        except Exception as e:
            results[source] = [f"Error loading manifest: {e}"]
    return results


def print_manifest_summary() -> None:
    """Print summary of all manifests."""
    print("\n" + "=" * 60)
    print("DATA MANIFEST SUMMARY")
    print("=" * 60)

    for source in ['gfw', 'open_meteo', 'noaa_sst', 'earthdata_sst', 'copernicus_sst']:
        try:
            manager = ManifestManager(source)
            stats = manager.get_stats()
            print(f"\n{stats['source_name']} ({source})")
            print(f"  Files: {stats['file_count']}")
            print(f"  Records: {stats['total_records']:,}")
            if stats['date_range'].get('earliest'):
                print(f"  Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            if stats['last_updated']:
                print(f"  Last updated: {stats['last_updated']}")
        except Exception as e:
            print(f"\n{source}: Error - {e}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_manifest_summary()
