#!/usr/bin/env python3
"""
Coastline Validator - Final Validation and Storage

Performs final validation of the coastline and saves it immutably
to the Gold layer with checksums for integrity verification.
"""

import sys
import json
import hashlib
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


@dataclass
class ValidationConfig:
    """Configuration for coastline validation."""
    # Spacing requirements
    max_spacing_m: float = 50  # 50m for high precision
    min_spacing_m: float = 20

    # Confidence requirements
    min_avg_confidence: float = 0.85
    max_low_confidence_percent: float = 5.0
    low_confidence_threshold: float = 0.70

    # Coverage requirements
    min_coverage_percent: float = 95.0

    # Geographic requirements
    expected_lat_range: Tuple[float, float] = (-18.35, -17.30)
    expected_lon_range: Tuple[float, float] = (-71.50, -70.10)


@dataclass
class ValidationResult:
    """Result of coastline validation."""
    is_valid: bool
    checks: Dict[str, bool]
    statistics: Dict
    errors: List[str]
    warnings: List[str]


class CoastlineValidator:
    """
    Validates coastline data and saves to immutable storage.

    Performs checks:
    - Spacing requirements (max 200m, min 50m)
    - Confidence thresholds
    - Coverage of target region
    - Geographic validity
    """

    def __init__(self, config: ValidationConfig = None):
        """Initialize validator."""
        self.config = config or ValidationConfig()

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def check_spacing(
        self,
        points: List[Tuple[float, float]],
        segments: List[List[Tuple[float, float]]] = None
    ) -> Tuple[bool, Dict]:
        """
        Check spacing requirements.

        If segments are provided, validates spacing within each segment
        without counting inter-segment gaps as violations.

        Returns:
            Tuple of (passed, statistics)
        """
        if segments and len(segments) > 0:
            # Validate each segment separately
            all_distances = []
            all_violations = []
            total_points = 0

            for seg_idx, seg in enumerate(segments):
                total_points += len(seg)
                for i in range(1, len(seg)):
                    dist = self.haversine_distance(
                        seg[i-1][0], seg[i-1][1],
                        seg[i][0], seg[i][1]
                    )
                    all_distances.append(dist)

                    if dist > self.config.max_spacing_m:
                        all_violations.append({
                            "segment": seg_idx,
                            "index": i,
                            "distance_m": dist,
                            "type": "too_far"
                        })

            if not all_distances:
                return False, {"error": "Not enough points in segments"}

            stats = {
                "total_points": total_points,
                "num_segments": len(segments),
                "avg_spacing_m": sum(all_distances) / len(all_distances),
                "max_spacing_m": max(all_distances),
                "min_spacing_m": min(all_distances),
                "violations": len(all_violations),
                "violation_details": all_violations[:10]
            }

            passed = len(all_violations) == 0
            return passed, stats

        # Legacy single-line validation
        if len(points) < 2:
            return False, {"error": "Not enough points"}

        distances = []
        violations = []

        for i in range(1, len(points)):
            dist = self.haversine_distance(
                points[i-1][0], points[i-1][1],
                points[i][0], points[i][1]
            )
            distances.append(dist)

            if dist > self.config.max_spacing_m:
                violations.append({
                    "index": i,
                    "distance_m": dist,
                    "type": "too_far"
                })

        stats = {
            "total_points": len(points),
            "avg_spacing_m": sum(distances) / len(distances),
            "max_spacing_m": max(distances),
            "min_spacing_m": min(distances),
            "violations": len(violations),
            "violation_details": violations[:10]  # First 10 only
        }

        passed = len(violations) == 0
        return passed, stats

    def check_confidence(
        self,
        confidence_scores: List[float]
    ) -> Tuple[bool, Dict]:
        """
        Check confidence requirements.

        Args:
            confidence_scores: List of confidence scores (0.0 to 1.0)

        Returns:
            Tuple of (passed, statistics)
        """
        if not confidence_scores:
            return True, {"note": "No confidence data provided"}

        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        low_confidence = sum(
            1 for c in confidence_scores
            if c < self.config.low_confidence_threshold
        )
        low_percent = low_confidence / len(confidence_scores) * 100

        stats = {
            "avg_confidence": avg_confidence,
            "low_confidence_count": low_confidence,
            "low_confidence_percent": low_percent
        }

        passed = (
            avg_confidence >= self.config.min_avg_confidence and
            low_percent <= self.config.max_low_confidence_percent
        )

        return passed, stats

    def check_coverage(
        self,
        points: List[Tuple[float, float]]
    ) -> Tuple[bool, Dict]:
        """
        Check that the coastline covers the expected region.

        Returns:
            Tuple of (passed, statistics)
        """
        if not points:
            return False, {"error": "No points"}

        lats = [p[0] for p in points]
        lons = [p[1] for p in points]

        actual_lat_range = (min(lats), max(lats))
        actual_lon_range = (min(lons), max(lons))

        expected_lat_span = self.config.expected_lat_range[1] - self.config.expected_lat_range[0]
        expected_lon_span = self.config.expected_lon_range[1] - self.config.expected_lon_range[0]

        actual_lat_span = actual_lat_range[1] - actual_lat_range[0]
        actual_lon_span = actual_lon_range[1] - actual_lon_range[0]

        lat_coverage = (actual_lat_span / expected_lat_span) * 100
        lon_coverage = (actual_lon_span / expected_lon_span) * 100

        # For a coastline, we care about latitude coverage (north-south extent)
        # Longitude will naturally be limited to the coast

        stats = {
            "actual_lat_range": actual_lat_range,
            "actual_lon_range": actual_lon_range,
            "lat_coverage_percent": lat_coverage,
            "lon_coverage_percent": lon_coverage,
            "north_extent": actual_lat_range[1],
            "south_extent": actual_lat_range[0]
        }

        passed = lat_coverage >= self.config.min_coverage_percent

        return passed, stats

    def check_geography(
        self,
        points: List[Tuple[float, float]]
    ) -> Tuple[bool, Dict]:
        """
        Check geographic validity of points.

        Ensures all points are within expected bounds.
        """
        if not points:
            return False, {"error": "No points"}

        out_of_bounds = []
        for i, (lat, lon) in enumerate(points):
            if not (self.config.expected_lat_range[0] <= lat <= self.config.expected_lat_range[1]):
                out_of_bounds.append({"index": i, "lat": lat, "lon": lon, "issue": "lat"})
            if not (self.config.expected_lon_range[0] <= lon <= self.config.expected_lon_range[1]):
                out_of_bounds.append({"index": i, "lat": lat, "lon": lon, "issue": "lon"})

        stats = {
            "total_points": len(points),
            "out_of_bounds": len(out_of_bounds),
            "out_of_bounds_details": out_of_bounds[:10]
        }

        passed = len(out_of_bounds) == 0
        return passed, stats

    def check_continuity(
        self,
        points: List[Tuple[float, float]]
    ) -> Tuple[bool, Dict]:
        """
        Check that the coastline is continuous (no isolated clusters).
        """
        if len(points) < 2:
            return True, {"note": "Too few points to check continuity"}

        # Check for large jumps that might indicate discontinuity
        max_allowed_jump = 5000  # 5km threshold for detecting breaks
        breaks = []

        for i in range(1, len(points)):
            dist = self.haversine_distance(
                points[i-1][0], points[i-1][1],
                points[i][0], points[i][1]
            )
            if dist > max_allowed_jump:
                breaks.append({
                    "index": i,
                    "distance_m": dist
                })

        stats = {
            "total_segments": len(points) - 1,
            "breaks_detected": len(breaks),
            "break_details": breaks[:10]
        }

        # Allow some breaks (islands, bays, etc.)
        passed = len(breaks) <= 3

        return passed, stats

    def validate(
        self,
        points: List[Tuple[float, float]],
        confidence_scores: List[float] = None,
        segments: List[List[Tuple[float, float]]] = None
    ) -> ValidationResult:
        """
        Run full validation on coastline.

        Args:
            points: List of (lat, lon) tuples
            confidence_scores: Optional list of confidence scores
            segments: Optional list of segments (for proper spacing validation)

        Returns:
            ValidationResult with all check results
        """
        errors = []
        warnings = []
        checks = {}
        all_stats = {}

        # Check spacing (use segments if provided for accurate gap handling)
        passed, stats = self.check_spacing(points, segments)
        checks["spacing"] = passed
        all_stats["spacing"] = stats
        if not passed:
            errors.append(f"Spacing check failed: {stats.get('violations', 0)} violations")

        # Check confidence (if provided)
        if confidence_scores:
            passed, stats = self.check_confidence(confidence_scores)
            checks["confidence"] = passed
            all_stats["confidence"] = stats
            if not passed:
                warnings.append(f"Confidence check failed: avg={stats.get('avg_confidence', 0):.2f}")
        else:
            checks["confidence"] = True  # Skip if not provided

        # Check coverage
        passed, stats = self.check_coverage(points)
        checks["coverage"] = passed
        all_stats["coverage"] = stats
        if not passed:
            warnings.append(f"Coverage check: {stats.get('lat_coverage_percent', 0):.1f}%")

        # Check geography
        passed, stats = self.check_geography(points)
        checks["geography"] = passed
        all_stats["geography"] = stats
        if not passed:
            errors.append(f"Geography check failed: {stats.get('out_of_bounds', 0)} points out of bounds")

        # Check continuity
        passed, stats = self.check_continuity(points)
        checks["continuity"] = passed
        all_stats["continuity"] = stats
        if not passed:
            warnings.append(f"Continuity check: {stats.get('breaks_detected', 0)} breaks detected")

        # Overall validity
        is_valid = all(checks.values())

        return ValidationResult(
            is_valid=is_valid,
            checks=checks,
            statistics=all_stats,
            errors=errors,
            warnings=warnings
        )

    def compute_checksum(self, data: str) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data.encode()).hexdigest()

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def save_to_gold(
        self,
        points: List[Tuple[float, float]],
        validation_result: ValidationResult,
        output_dir: str = None,
        version: str = None,
        segments: List[List[Tuple[float, float]]] = None
    ) -> Dict[str, str]:
        """
        Save validated coastline to Gold layer with checksums.

        Args:
            points: Validated coastline points (flattened)
            validation_result: Result of validation
            output_dir: Output directory (default: data/gold/coastline)
            version: Version string (default: v1)
            segments: Optional list of segments for MultiLineString output

        Returns:
            Dict with paths to created files
        """
        if output_dir is None:
            output_dir = ROOT_DIR / "data" / "gold" / "coastline"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        if version is None:
            version = "v1"

        # Create GeoJSON - use MultiLineString if segments provided
        if segments and len(segments) > 0:
            # MultiLineString format - preserves segment boundaries
            coordinates = []
            for seg in segments:
                seg_coords = [[lon, lat] for lat, lon in seg]
                coordinates.append(seg_coords)

            total_points = sum(len(seg) for seg in segments)
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "name": "Verified Coastline (Segmented)",
                        "version": version,
                        "num_segments": len(segments),
                        "total_points": total_points,
                        "points_per_segment": [len(seg) for seg in segments],
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "connector_version": "v5.1"
                    }
                }]
            }
        else:
            # Single LineString format (legacy)
            coordinates = [[lon, lat] for lat, lon in points]
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "name": "Verified Coastline",
                        "version": version,
                        "points": len(coordinates),
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    }
                }]
            }

        # Save GeoJSON
        geojson_path = output_dir / f"coastline_{version}.geojson"
        geojson_str = json.dumps(geojson, indent=2)
        with open(geojson_path, 'w') as f:
            f.write(geojson_str)

        # Compute and save checksum
        checksum = self.compute_checksum(geojson_str)
        checksum_path = output_dir / f"coastline_{version}.sha256"
        with open(checksum_path, 'w') as f:
            f.write(f"{checksum}  coastline_{version}.geojson\n")

        # Create validation report
        report = {
            "version": version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "checksum": f"sha256:{checksum}",
            "validation": {
                "is_valid": validation_result.is_valid,
                "checks": validation_result.checks,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings
            },
            "statistics": validation_result.statistics,
            "region": {
                "lat_min": self.config.expected_lat_range[0],
                "lat_max": self.config.expected_lat_range[1],
                "lon_min": self.config.expected_lon_range[0],
                "lon_max": self.config.expected_lon_range[1]
            }
        }

        report_path = output_dir / f"coastline_{version}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)

        print(f"[OK] Guardado en Gold layer:")
        print(f"     {geojson_path}")
        print(f"     {checksum_path}")
        print(f"     {report_path}")

        return {
            "geojson": str(geojson_path),
            "checksum": str(checksum_path),
            "report": str(report_path)
        }

    def verify_checksum(self, geojson_path: str) -> bool:
        """
        Verify integrity of a saved coastline file.

        Args:
            geojson_path: Path to the GeoJSON file

        Returns:
            True if checksum matches
        """
        geojson_path = Path(geojson_path)
        checksum_path = geojson_path.with_suffix('.sha256')

        if not checksum_path.exists():
            print(f"[WARN] No checksum file: {checksum_path}")
            return False

        with open(geojson_path) as f:
            content = f.read()

        with open(checksum_path) as f:
            expected = f.read().split()[0]

        actual = self.compute_checksum(content)

        if actual == expected:
            print(f"[OK] Checksum verified: {geojson_path.name}")
            return True
        else:
            print(f"[ERROR] Checksum mismatch!")
            print(f"        Expected: {expected}")
            print(f"        Actual:   {actual}")
            return False


if __name__ == "__main__":
    print("Coastline Validator - Use via coastline_pipeline.py")
    print("This module validates and saves coastline to Gold layer.")
