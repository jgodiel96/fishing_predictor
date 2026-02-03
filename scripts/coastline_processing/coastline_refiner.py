#!/usr/bin/env python3
"""
Coastline Refiner - Spacing and Smoothing

Refines the coastline to ensure:
1. Maximum spacing of 200 meters between consecutive points
2. Minimum spacing of 50 meters (no redundant points)
3. Smooth curve using Savitzky-Golay filter
4. Proper geographic ordering (south to north)
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class RefinementConfig:
    """Configuration for coastline refinement."""
    max_spacing_m: float = 50   # Maximum allowed spacing (50m for precision)
    min_spacing_m: float = 20   # Minimum spacing (remove closer points)
    target_spacing_m: float = 40  # Ideal spacing for interpolation

    # Interpolation
    interpolation_method: str = "cubic"  # "linear" or "cubic"
    preserve_corners: bool = True
    corner_angle_threshold: float = 30  # Degrees

    # Smoothing
    enable_smoothing: bool = True
    smoothing_window: int = 7
    smoothing_order: int = 3


class CoastlineRefiner:
    """
    Refines coastline points to ensure proper spacing and smoothness.

    Key features:
    - Guarantees max spacing <= 200m
    - Removes redundant points (< 50m apart)
    - Applies Savitzky-Golay smoothing
    - Preserves natural corner points
    """

    def __init__(self, config: RefinementConfig = None):
        """Initialize refiner."""
        self.config = config or RefinementConfig()

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two points in meters using Haversine formula.
        """
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def calculate_angle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        """
        Calculate the angle at p2 formed by p1-p2-p3.

        Returns angle in degrees (0-180).
        """
        # Vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Dot product
        dot = v1[0] * v2[0] + v1[1] * v2[1]

        # Magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            return 180.0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_angle))

        return angle

    def is_corner_point(
        self,
        points: List[Tuple[float, float]],
        index: int
    ) -> bool:
        """
        Check if point at index is a corner point.

        A corner point has a significant angle change.
        """
        if index == 0 or index == len(points) - 1:
            return True  # Endpoints are always preserved

        angle = self.calculate_angle(
            points[index - 1],
            points[index],
            points[index + 1]
        )

        # Sharp angle = corner
        return angle < (180 - self.config.corner_angle_threshold)

    def remove_close_points(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Remove points that are too close together.

        Preserves first/last points and corner points.
        """
        if len(points) <= 2:
            return points

        # First pass: identify corner points
        corners = set()
        if self.config.preserve_corners:
            for i in range(len(points)):
                if self.is_corner_point(points, i):
                    corners.add(i)

        # Second pass: filter close points
        result = [points[0]]
        last_kept_idx = 0

        for i in range(1, len(points)):
            dist = self.haversine_distance(
                points[last_kept_idx][0], points[last_kept_idx][1],
                points[i][0], points[i][1]
            )

            # Keep if: far enough, corner point, or last point
            is_corner = i in corners
            is_last = (i == len(points) - 1)
            far_enough = dist >= self.config.min_spacing_m

            if far_enough or is_corner or is_last:
                result.append(points[i])
                last_kept_idx = i

        return result

    def interpolate_segment(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        num_points: int
    ) -> List[Tuple[float, float]]:
        """
        Interpolate points between p1 and p2.

        Args:
            p1, p2: Start and end points (lat, lon)
            num_points: Number of intermediate points to create

        Returns:
            List of interpolated points (excluding p1, including p2)
        """
        if num_points <= 1:
            return [p2]

        result = []
        for i in range(1, num_points + 1):
            t = i / num_points
            lat = p1[0] + t * (p2[0] - p1[0])
            lon = p1[1] + t * (p2[1] - p1[1])
            result.append((lat, lon))

        return result

    def ensure_max_spacing(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Ensure no gap exceeds max_spacing_m by inserting intermediate points.

        Uses linear or cubic interpolation based on config.
        """
        if len(points) < 2:
            return points

        result = [points[0]]

        for i in range(1, len(points)):
            p1 = result[-1]
            p2 = points[i]

            dist = self.haversine_distance(p1[0], p1[1], p2[0], p2[1])

            if dist <= self.config.max_spacing_m:
                result.append(p2)
            else:
                # Need to interpolate
                num_segments = int(math.ceil(dist / self.config.target_spacing_m))
                interpolated = self.interpolate_segment(p1, p2, num_segments)
                result.extend(interpolated)

        return result

    def smooth_curve(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Apply Savitzky-Golay smoothing to reduce noise while preserving shape.
        """
        if not self.config.enable_smoothing:
            return points

        if not SCIPY_AVAILABLE:
            print("[WARN] scipy not available, skipping smoothing")
            return points

        if len(points) < self.config.smoothing_window:
            return points  # Not enough points to smooth

        # Extract lat/lon arrays
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])

        # Apply Savitzky-Golay filter
        try:
            smoothed_lats = savgol_filter(
                lats,
                self.config.smoothing_window,
                self.config.smoothing_order
            )
            smoothed_lons = savgol_filter(
                lons,
                self.config.smoothing_window,
                self.config.smoothing_order
            )

            # Preserve endpoints exactly
            smoothed_lats[0] = lats[0]
            smoothed_lats[-1] = lats[-1]
            smoothed_lons[0] = lons[0]
            smoothed_lons[-1] = lons[-1]

            return list(zip(smoothed_lats, smoothed_lons))

        except Exception as e:
            print(f"[WARN] Smoothing failed: {e}")
            return points

    def sort_geographically(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Sort points geographically (south to north along the coast).

        For the Peru coast (running roughly north-south), this means
        sorting primarily by latitude.
        """
        if not points:
            return points

        # Sort by latitude (south to north)
        return sorted(points, key=lambda p: (p[0], p[1]))

    def refine(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Apply full refinement pipeline to coastline points.

        Steps:
        1. Sort geographically
        2. Remove points too close together
        3. Insert points where gaps are too large
        4. Apply smoothing

        Args:
            points: Raw coastline points

        Returns:
            Refined coastline points with guaranteed spacing
        """
        if len(points) < 2:
            return points

        print(f"[INFO] Refinando {len(points)} puntos...")

        # Step 1: Sort
        sorted_points = self.sort_geographically(points)
        print(f"       Ordenados geograficamente")

        # Step 2: Remove close points
        spaced_points = self.remove_close_points(sorted_points)
        print(f"       {len(spaced_points)} puntos despues de remover cercanos")

        # Step 3: Ensure max spacing
        interpolated = self.ensure_max_spacing(spaced_points)
        print(f"       {len(interpolated)} puntos despues de interpolar gaps")

        # Step 4: Smooth
        smoothed = self.smooth_curve(interpolated)
        print(f"       Suavizado aplicado")

        # Final sort to ensure order
        final = self.sort_geographically(smoothed)

        print(f"[OK] {len(final)} puntos finales")
        return final

    def get_statistics(
        self,
        points: List[Tuple[float, float]]
    ) -> dict:
        """
        Calculate spacing statistics for the coastline.
        """
        if len(points) < 2:
            return {
                "total_points": len(points),
                "total_length_km": 0,
                "avg_spacing_m": 0,
                "max_spacing_m": 0,
                "min_spacing_m": 0,
                "spacing_violations": 0
            }

        distances = []
        for i in range(1, len(points)):
            dist = self.haversine_distance(
                points[i-1][0], points[i-1][1],
                points[i][0], points[i][1]
            )
            distances.append(dist)

        total_length = sum(distances)
        violations = sum(1 for d in distances if d > self.config.max_spacing_m)

        return {
            "total_points": len(points),
            "total_length_km": total_length / 1000,
            "avg_spacing_m": total_length / len(distances),
            "max_spacing_m": max(distances),
            "min_spacing_m": min(distances),
            "spacing_violations": violations,
            "meets_requirements": violations == 0
        }

    def validate_spacing(
        self,
        points: List[Tuple[float, float]]
    ) -> Tuple[bool, List[int]]:
        """
        Validate that all spacing meets requirements.

        Returns:
            Tuple of (is_valid, list of indices with violations)
        """
        violations = []

        for i in range(1, len(points)):
            dist = self.haversine_distance(
                points[i-1][0], points[i-1][1],
                points[i][0], points[i][1]
            )

            if dist > self.config.max_spacing_m:
                violations.append(i)
            elif dist < self.config.min_spacing_m and i > 1:
                # Only warn, don't fail for close points
                pass

        return len(violations) == 0, violations


if __name__ == "__main__":
    print("Coastline Refiner - Use via coastline_pipeline.py")
    print("This module ensures proper spacing (max 200m) between coastline points.")
