#!/usr/bin/env python3
"""
Coastline Verifier - Dual Source Verification

Verifies coastline points by comparing segmentation from two sources:
1. Satellite imagery (ESRI World Imagery)
2. Street map (OpenStreetMap)

Points are validated when both sources agree on the water/land boundary.
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class VerificationConfig:
    """Configuration for coastline verification."""
    iou_threshold: float = 0.85  # Minimum IoU for valid point
    pixel_agreement_threshold: float = 0.90
    sea_direction: str = "west"  # Ocean is to the west in Peru
    direction_check_distance_px: int = 20  # Pixels to check for direction
    min_confidence: float = 0.70


@dataclass
class VerifiedPoint:
    """A verified coastline point with confidence metrics."""
    lat: float
    lon: float
    iou: float
    confidence: float
    is_valid: bool
    sat_water_west: bool
    street_water_west: bool


class CoastlineVerifier:
    """
    Verifies coastline points using dual-source comparison.

    Uses IoU (Intersection over Union) between water masks from:
    - Satellite imagery (ESRI)
    - Street map (OSM)

    Points where both sources agree are considered valid.
    """

    def __init__(self, config: VerificationConfig = None):
        """Initialize verifier."""
        self.config = config or VerificationConfig()

    def compute_iou(
        self,
        mask_a: np.ndarray,
        mask_b: np.ndarray
    ) -> float:
        """
        Compute Intersection over Union between two masks.

        Args:
            mask_a: Binary mask (0 or 1/255)
            mask_b: Binary mask (0 or 1/255)

        Returns:
            IoU score (0.0 to 1.0)
        """
        # Normalize masks to 0/1
        a = (mask_a > 0).astype(np.uint8)
        b = (mask_b > 0).astype(np.uint8)

        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def check_water_west(
        self,
        water_mask: np.ndarray,
        pixel_x: int,
        pixel_y: int,
        check_distance: int = 20
    ) -> bool:
        """
        Check if there is water to the west of a point.

        In Peru, the ocean is always to the west, so valid coastline
        points should have water on their west side.

        Args:
            water_mask: Binary water mask
            pixel_x, pixel_y: Point coordinates
            check_distance: How far west to check

        Returns:
            True if water is detected to the west
        """
        h, w = water_mask.shape

        # Check pixels to the west
        x_start = max(0, pixel_x - check_distance)
        x_end = pixel_x

        if x_end <= x_start:
            return False

        # Sample the region to the west
        west_region = water_mask[
            max(0, pixel_y - 2):min(h, pixel_y + 3),
            x_start:x_end
        ]

        if west_region.size == 0:
            return False

        # Water should be present to the west
        water_ratio = (west_region > 0).sum() / west_region.size
        return water_ratio > 0.5

    def check_land_east(
        self,
        water_mask: np.ndarray,
        pixel_x: int,
        pixel_y: int,
        check_distance: int = 20
    ) -> bool:
        """
        Check if there is land to the east of a point.

        Args:
            water_mask: Binary water mask
            pixel_x, pixel_y: Point coordinates
            check_distance: How far east to check

        Returns:
            True if land is detected to the east
        """
        h, w = water_mask.shape

        # Check pixels to the east
        x_start = pixel_x + 1
        x_end = min(w, pixel_x + check_distance + 1)

        if x_end <= x_start:
            return False

        # Sample the region to the east
        east_region = water_mask[
            max(0, pixel_y - 2):min(h, pixel_y + 3),
            x_start:x_end
        ]

        if east_region.size == 0:
            return False

        # Land should be present to the east (low water ratio)
        water_ratio = (east_region > 0).sum() / east_region.size
        return water_ratio < 0.5

    def is_on_mask_edge(
        self,
        mask: np.ndarray,
        pixel_x: int,
        pixel_y: int,
        radius: int = 3
    ) -> bool:
        """
        Check if a pixel is on the edge of a mask.

        Args:
            mask: Binary mask
            pixel_x, pixel_y: Point coordinates
            radius: Search radius

        Returns:
            True if the point is on an edge (boundary between 0 and 1)
        """
        h, w = mask.shape

        y_min = max(0, pixel_y - radius)
        y_max = min(h, pixel_y + radius + 1)
        x_min = max(0, pixel_x - radius)
        x_max = min(w, pixel_x + radius + 1)

        region = mask[y_min:y_max, x_min:x_max]

        if region.size == 0:
            return False

        # An edge point should have both 0s and non-0s in its neighborhood
        has_water = (region > 0).any()
        has_land = (region == 0).any()

        return has_water and has_land

    def verify_point(
        self,
        lat: float,
        lon: float,
        sat_mask: np.ndarray,
        street_mask: np.ndarray,
        pixel_x: int,
        pixel_y: int
    ) -> VerifiedPoint:
        """
        Verify a single coastline point.

        Args:
            lat, lon: Point coordinates
            sat_mask: Water mask from satellite image
            street_mask: Water mask from street map
            pixel_x, pixel_y: Point location in pixel coords

        Returns:
            VerifiedPoint with verification results
        """
        # Compute IoU between masks
        iou = self.compute_iou(sat_mask, street_mask)

        # Check water direction
        sat_water_west = self.check_water_west(sat_mask, pixel_x, pixel_y)
        street_water_west = self.check_water_west(street_mask, pixel_x, pixel_y)

        # Check land direction
        sat_land_east = self.check_land_east(sat_mask, pixel_x, pixel_y)
        street_land_east = self.check_land_east(street_mask, pixel_x, pixel_y)

        # Check if on edge
        is_on_sat_edge = self.is_on_mask_edge(sat_mask, pixel_x, pixel_y)
        is_on_street_edge = self.is_on_mask_edge(street_mask, pixel_x, pixel_y)

        # Calculate confidence
        direction_score = 0.0
        if sat_water_west and street_water_west:
            direction_score += 0.5
        if sat_land_east and street_land_east:
            direction_score += 0.5

        edge_score = 0.0
        if is_on_sat_edge:
            edge_score += 0.5
        if is_on_street_edge:
            edge_score += 0.5

        confidence = (iou * 0.4 + direction_score * 0.4 + edge_score * 0.2)

        # Determine validity
        is_valid = (
            iou >= self.config.iou_threshold and
            sat_water_west and
            direction_score >= 0.5
        )

        return VerifiedPoint(
            lat=lat,
            lon=lon,
            iou=iou,
            confidence=confidence,
            is_valid=is_valid,
            sat_water_west=sat_water_west,
            street_water_west=street_water_west
        )

    def verify_coastline(
        self,
        points: List[Tuple[float, float]],
        sat_masks: Dict[Tuple[int, int], np.ndarray],
        street_masks: Dict[Tuple[int, int], np.ndarray],
        pixel_coords: Dict[Tuple[float, float], Tuple[int, int, int, int]]
    ) -> List[VerifiedPoint]:
        """
        Verify a list of coastline points.

        Args:
            points: List of (lat, lon) tuples
            sat_masks: Dict of tile coords -> satellite water mask
            street_masks: Dict of tile coords -> street water mask
            pixel_coords: Dict of (lat, lon) -> (tile_x, tile_y, px, py)

        Returns:
            List of VerifiedPoint objects
        """
        verified = []

        for lat, lon in points:
            if (lat, lon) not in pixel_coords:
                continue

            tile_x, tile_y, px, py = pixel_coords[(lat, lon)]
            tile_key = (tile_x, tile_y)

            if tile_key not in sat_masks or tile_key not in street_masks:
                continue

            verified_point = self.verify_point(
                lat=lat,
                lon=lon,
                sat_mask=sat_masks[tile_key],
                street_mask=street_masks[tile_key],
                pixel_x=px,
                pixel_y=py
            )

            verified.append(verified_point)

        return verified

    def filter_valid_points(
        self,
        verified: List[VerifiedPoint]
    ) -> List[Tuple[float, float]]:
        """
        Filter to only valid points.

        Args:
            verified: List of VerifiedPoint objects

        Returns:
            List of (lat, lon) for valid points only
        """
        return [
            (p.lat, p.lon)
            for p in verified
            if p.is_valid
        ]

    def get_statistics(self, verified: List[VerifiedPoint]) -> Dict:
        """
        Get verification statistics.

        Args:
            verified: List of VerifiedPoint objects

        Returns:
            Dictionary of statistics
        """
        if not verified:
            return {
                "total_points": 0,
                "valid_points": 0,
                "valid_percent": 0,
                "avg_iou": 0,
                "avg_confidence": 0
            }

        valid = [p for p in verified if p.is_valid]

        return {
            "total_points": len(verified),
            "valid_points": len(valid),
            "valid_percent": len(valid) / len(verified) * 100,
            "avg_iou": sum(p.iou for p in verified) / len(verified),
            "avg_confidence": sum(p.confidence for p in verified) / len(verified),
            "points_with_water_west": sum(1 for p in verified if p.sat_water_west),
            "points_with_dual_agreement": sum(
                1 for p in verified
                if p.sat_water_west and p.street_water_west
            )
        }


def create_osm_water_mask(street_image: np.ndarray) -> np.ndarray:
    """
    Create water mask from OSM street map.

    OSM typically shows water in blue tones. This function
    extracts those blue regions.

    Args:
        street_image: RGB image from OSM

    Returns:
        Binary mask where 255=water
    """
    if not CV2_AVAILABLE:
        return np.zeros(street_image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(street_image, cv2.COLOR_RGB2HSV)

    # OSM water is typically a specific shade of blue
    # Range tuned for OSM standard tile set
    water_low = np.array([100, 30, 150])
    water_high = np.array([120, 100, 255])

    mask = cv2.inRange(hsv, water_low, water_high)

    # Also detect lighter water
    water_low2 = np.array([95, 10, 180])
    water_high2 = np.array([115, 50, 255])
    mask2 = cv2.inRange(hsv, water_low2, water_high2)

    water_mask = cv2.bitwise_or(mask, mask2)

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    return water_mask


if __name__ == "__main__":
    print("Coastline Verifier - Use via coastline_pipeline.py")
    print("This module provides dual-source verification for coastline points.")
