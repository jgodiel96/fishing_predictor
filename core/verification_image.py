#!/usr/bin/env python3
"""
Verification Image Generator

Generates a static image showing coastline, hotspots, and fishing zones
for quick visual verification during iteration.

Author: Fishing Predictor Project
Date: 2026-02-02
"""

from pathlib import Path
import json
from typing import List, Tuple, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_coastline(coastline_path: Path) -> List[List[Tuple[float, float]]]:
    """Load coastline segments from GeoJSON."""
    with open(coastline_path) as f:
        data = json.load(f)

    segments = []
    geom = data['features'][0]['geometry']

    if geom['type'] == 'LineString':
        coords = [(lat, lon) for lon, lat in geom['coordinates']]
        segments.append(coords)
    elif geom['type'] == 'MultiLineString':
        for line in geom['coordinates']:
            coords = [(lat, lon) for lon, lat in line]
            segments.append(coords)

    return segments


def generate_verification_image(
    coastline_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    title: str = "Fishing Predictor Verification"
) -> str:
    """
    Generate a verification image showing coastline and hotspots.

    Returns path to generated image.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[ERROR] matplotlib not available")
        return ""

    from config import COASTLINE_FILE, OUTPUT_DIR
    from domain import HOTSPOTS, STUDY_AREA

    coastline_path = coastline_path or COASTLINE_FILE
    output_path = output_path or (OUTPUT_DIR / "verification_map.png")

    # Load coastline
    segments = load_coastline(coastline_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot study area bounds
    ax.axhline(y=STUDY_AREA.north, color='gray', linestyle='--', alpha=0.5, label='Study Area')
    ax.axhline(y=STUDY_AREA.south, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=STUDY_AREA.west, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=STUDY_AREA.east, color='gray', linestyle='--', alpha=0.5)

    # Plot coastline segments
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(segments), 1)))
    for i, segment in enumerate(segments):
        lats = [p[0] for p in segment]
        lons = [p[1] for p in segment]
        ax.plot(lons, lats, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)

    # Plot hotspots
    for hotspot in HOTSPOTS:
        ax.scatter(hotspot.lon, hotspot.lat, s=80, c='red', marker='o',
                   edgecolor='white', linewidth=1, zorder=5)
        ax.annotate(hotspot.name, (hotspot.lon, hotspot.lat),
                   fontsize=6, ha='left', va='bottom',
                   xytext=(3, 3), textcoords='offset points')

    # Highlight Canepa area
    canepa_hotspots = [h for h in HOTSPOTS if 'Canepa' in h.name or 'Sama' in h.name]
    if canepa_hotspots:
        for h in canepa_hotspots:
            circle = Circle((h.lon, h.lat), 0.02, fill=False,
                           color='orange', linewidth=2, linestyle='--')
            ax.add_patch(circle)

    # Labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{title}\nCoastline: {coastline_path.name}, Hotspots: {len(HOTSPOTS)}')

    # Stats text
    total_points = sum(len(s) for s in segments)
    stats_text = f"Segments: {len(segments)}\nPoints: {total_points}\nHotspots: {len(HOTSPOTS)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Set aspect ratio
    ax.set_aspect('equal')

    # Grid
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Verification image saved: {output_path}")
    return str(output_path)


def generate_canepa_detail(output_path: Optional[Path] = None) -> str:
    """Generate a detailed view of the Canepa area."""
    if not MATPLOTLIB_AVAILABLE:
        return ""

    from config import COASTLINE_FILE, OUTPUT_DIR
    from domain import HOTSPOTS

    output_path = output_path or (OUTPUT_DIR / "verification_canepa.png")

    segments = load_coastline(COASTLINE_FILE)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot coastline (only Canepa area)
    for segment in segments:
        lats = [p[0] for p in segment if -18.50 <= p[0] <= -18.30]
        lons = [p[1] for p in segment if -18.50 <= p[0] <= -18.30]
        if lats:
            ax.plot(lons, lats, 'b-', linewidth=2, alpha=0.8)

    # Plot Canepa hotspots with larger markers
    canepa_hotspots = [h for h in HOTSPOTS if -18.50 <= h.lat <= -18.30]
    for h in canepa_hotspots:
        ax.scatter(h.lon, h.lat, s=200, c='red', marker='o',
                   edgecolor='white', linewidth=2, zorder=5)
        ax.annotate(h.name, (h.lon, h.lat), fontsize=10, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Set bounds to Canepa area
    ax.set_xlim(-70.40, -70.28)
    ax.set_ylim(-18.50, -18.30)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Playa Canepa / Morro Sama Detail View')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Canepa detail image saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    print("Generating verification images...")
    generate_verification_image()
    generate_canepa_detail()
