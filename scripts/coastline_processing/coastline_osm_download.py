#!/usr/bin/env python3
"""
OSM Coastline Downloader

Downloads fresh coastline data directly from OpenStreetMap using Overpass API.
This ensures we get accurate, up-to-date coastline geometry.

Author: Fishing Predictor Project
Date: 2026-02-02
Version: 5.3
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import requests

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


@dataclass
class OSMConfig:
    """Configuration for OSM download."""
    # Region bounds (Tacna-Ilo, Peru) - Extended to include Canepa/Sama
    lat_min: float = -18.50
    lat_max: float = -17.25
    lon_min: float = -71.55
    lon_max: float = -70.00

    # API settings
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    timeout: int = 180

    # Output
    output_dir: str = "data/gold/coastline"


class OSMCoastlineDownloader:
    """
    Downloads coastline data from OpenStreetMap.

    Uses Overpass API to query for coastline ways in the specified region.
    """

    def __init__(self, config: Optional[OSMConfig] = None):
        self.config = config or OSMConfig()

    def build_query(self) -> str:
        """
        Build Overpass QL query for coastline.

        Returns coastline ways with their geometry.
        """
        bbox = f"{self.config.lat_min},{self.config.lon_min},{self.config.lat_max},{self.config.lon_max}"

        query = f"""
        [out:json][timeout:{self.config.timeout}];
        (
          way["natural"="coastline"]({bbox});
        );
        out body geom;
        """
        return query

    def download(self) -> dict:
        """
        Download coastline data from Overpass API.

        Returns raw JSON response.
        """
        query = self.build_query()

        print(f"[INFO] Downloading coastline from OSM...")
        print(f"       Region: {self.config.lat_min} to {self.config.lat_max} lat")
        print(f"               {self.config.lon_min} to {self.config.lon_max} lon")

        try:
            response = requests.post(
                self.config.overpass_url,
                data={"data": query},
                timeout=self.config.timeout
            )
            response.raise_for_status()

            data = response.json()
            n_ways = len(data.get('elements', []))
            print(f"[OK] Downloaded {n_ways} coastline ways")

            return data

        except requests.Timeout:
            print("[ERROR] Request timed out. Try a smaller region or later.")
            return {}
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return {}

    def extract_points(self, data: dict) -> List[List[Tuple[float, float]]]:
        """
        Extract points from Overpass response.

        Returns list of ways, each way is a list of (lat, lon) points.
        """
        ways = []

        for element in data.get('elements', []):
            if element.get('type') != 'way':
                continue

            geometry = element.get('geometry', [])
            if not geometry:
                continue

            way_points = [(node['lat'], node['lon']) for node in geometry]
            if len(way_points) >= 2:
                ways.append(way_points)

        return ways

    def merge_ways(
        self,
        ways: List[List[Tuple[float, float]]],
        max_gap_m: float = 100
    ) -> List[List[Tuple[float, float]]]:
        """
        Merge ways that connect at endpoints.

        Returns merged segments.
        """
        import math

        def haversine(p1, p2):
            lat1, lon1 = p1
            lat2, lon2 = p2
            R = 6371000
            lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        if not ways:
            return []

        # Make copies to avoid modifying originals
        remaining = [list(w) for w in ways]
        merged = []

        while remaining:
            current = remaining.pop(0)
            changed = True

            while changed:
                changed = False
                for i, other in enumerate(remaining):
                    # Check if endpoints connect
                    dist_end_start = haversine(current[-1], other[0])
                    dist_end_end = haversine(current[-1], other[-1])
                    dist_start_start = haversine(current[0], other[0])
                    dist_start_end = haversine(current[0], other[-1])

                    if dist_end_start < max_gap_m:
                        current = current + other
                        remaining.pop(i)
                        changed = True
                        break
                    elif dist_end_end < max_gap_m:
                        current = current + other[::-1]
                        remaining.pop(i)
                        changed = True
                        break
                    elif dist_start_start < max_gap_m:
                        current = other[::-1] + current
                        remaining.pop(i)
                        changed = True
                        break
                    elif dist_start_end < max_gap_m:
                        current = other + current
                        remaining.pop(i)
                        changed = True
                        break

            merged.append(current)

        return merged

    def order_south_to_north(
        self,
        segments: List[List[Tuple[float, float]]]
    ) -> List[List[Tuple[float, float]]]:
        """Order each segment from south to north."""
        ordered = []
        for seg in segments:
            if seg[0][0] > seg[-1][0]:  # First point more north than last
                ordered.append(seg[::-1])
            else:
                ordered.append(seg)

        # Sort segments by starting latitude
        ordered.sort(key=lambda s: s[0][0])
        return ordered

    def to_geojson(
        self,
        segments: List[List[Tuple[float, float]]],
        metadata: dict = None
    ) -> dict:
        """Convert segments to GeoJSON."""
        total_points = sum(len(seg) for seg in segments)

        # Use MultiLineString for multiple segments
        if len(segments) > 1:
            coordinates = [[[lon, lat] for lat, lon in seg] for seg in segments]
            geom_type = "MultiLineString"
        else:
            coordinates = [[lon, lat] for lat, lon in segments[0]] if segments else []
            geom_type = "LineString"

        properties = {
            "name": "OSM Coastline",
            "source": "openstreetmap_overpass",
            "num_segments": len(segments),
            "total_points": total_points,
        }
        if metadata:
            properties.update(metadata)

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": geom_type,
                    "coordinates": coordinates
                },
                "properties": properties
            }]
        }

    def download_and_save(
        self,
        output_filename: str = "coastline_osm_fresh.geojson"
    ) -> Tuple[str, dict]:
        """
        Download coastline and save to file.

        Returns (output_path, statistics)
        """
        # Download
        data = self.download()
        if not data:
            return "", {}

        # Extract points
        ways = self.extract_points(data)
        print(f"[INFO] Extracted {len(ways)} ways with {sum(len(w) for w in ways)} total points")

        # Merge connected ways
        print("[INFO] Merging connected ways...")
        merged = self.merge_ways(ways)
        print(f"[OK] Merged into {len(merged)} segments")

        # Order
        ordered = self.order_south_to_north(merged)

        # Calculate total length
        import math
        total_length = 0
        for seg in ordered:
            for i in range(1, len(seg)):
                lat1, lon1 = seg[i-1]
                lat2, lon2 = seg[i]
                dlat = (lat2 - lat1) * 111000
                dlon = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))
                total_length += math.sqrt(dlat**2 + dlon**2)

        # Convert to GeoJSON
        geojson = self.to_geojson(ordered, {
            "download_date": time.strftime("%Y-%m-%d"),
            "region": f"{self.config.lat_min},{self.config.lon_min} to {self.config.lat_max},{self.config.lon_max}"
        })

        # Save
        output_dir = Path(ROOT_DIR / self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        # Checksum
        with open(output_path) as f:
            content = f.read()
            checksum = hashlib.sha256(content.encode()).hexdigest()

        checksum_path = output_path.with_suffix('.sha256')
        with open(checksum_path, 'w') as f:
            f.write(f"{checksum}  {output_filename}\n")

        stats = {
            "segments": len(ordered),
            "total_points": sum(len(s) for s in ordered),
            "total_length_km": total_length / 1000,
            "output_path": str(output_path)
        }

        print(f"\n[OK] Saved: {output_path}")
        print(f"     Segments: {stats['segments']}")
        print(f"     Points: {stats['total_points']}")
        print(f"     Length: {stats['total_length_km']:.1f} km")

        return str(output_path), stats


def download_fresh_coastline() -> Tuple[str, dict]:
    """Convenience function to download fresh coastline."""
    downloader = OSMCoastlineDownloader()
    return downloader.download_and_save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download coastline from OSM")
    parser.add_argument("-o", "--output", default="coastline_osm_fresh.geojson",
                       help="Output filename")
    parser.add_argument("--lat-min", type=float, default=-18.40)
    parser.add_argument("--lat-max", type=float, default=-17.25)
    parser.add_argument("--lon-min", type=float, default=-71.55)
    parser.add_argument("--lon-max", type=float, default=-70.05)

    args = parser.parse_args()

    config = OSMConfig(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max
    )

    downloader = OSMCoastlineDownloader(config)
    downloader.download_and_save(args.output)
