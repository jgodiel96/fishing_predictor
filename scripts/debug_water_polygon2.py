#!/usr/bin/env python3
"""
Debug script to trace water polygon generation step by step.
"""

import sys
sys.path.insert(0, '/Users/jorgegodiel/Documents/codes/Proyecto_pesca/fishing_predictor')

from shapely.geometry import LineString, Polygon, box, MultiPolygon
from core.cv_analysis.osm_coastline import OSMCoastlineLoader

# Test area - Ilo
LAT_MIN, LAT_MAX = -17.68, -17.65
LON_MIN, LON_MAX = -71.36, -71.33

print("="*60)
print("DEBUG: GENERACIÓN DEL POLÍGONO DE AGUA")
print("="*60)

# Load coastline lines directly
loader = OSMCoastlineLoader()
bounds = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
lines = loader._load_coastline_lines(bounds)

print(f"\nLíneas costeras encontradas: {len(lines)}")

# Collect all coordinates
all_coords = []
for line in lines:
    all_coords.extend(list(line.coords))
    print(f"  Línea con {len(line.coords)} puntos")

print(f"\nTotal coordenadas: {len(all_coords)}")

# Sort by latitude
all_coords_sorted = sorted(all_coords, key=lambda c: c[1])

southmost = all_coords_sorted[0]
northmost = all_coords_sorted[-1]

print(f"\nPunto más al sur: ({southmost[0]:.6f}, {southmost[1]:.6f})")
print(f"Punto más al norte: ({northmost[0]:.6f}, {northmost[1]:.6f})")
print(f"\nBbox: lat({LAT_MIN}, {LAT_MAX}), lon({LON_MIN}, {LON_MAX})")

# Check extension conditions
print(f"\n¿Necesita extensión sur? southmost[1] > lat_min: {southmost[1]} > {LAT_MIN} = {southmost[1] > LAT_MIN}")
print(f"¿Necesita extensión norte? northmost[1] < lat_max: {northmost[1]} < {LAT_MAX} = {northmost[1] < LAT_MAX}")

# Create extended coords
extended_coords = []

if southmost[1] > LAT_MIN:
    print(f"\nExtendiendo al sur: ({southmost[0]:.6f}, {LAT_MIN})")
    extended_coords.append((southmost[0], LAT_MIN))

extended_coords.extend(all_coords_sorted)

if northmost[1] < LAT_MAX:
    print(f"Extendiendo al norte: ({northmost[0]:.6f}, {LAT_MAX})")
    extended_coords.append((northmost[0], LAT_MAX))

print(f"\nCoords extendidas: {len(extended_coords)} puntos")
print(f"  Primera: {extended_coords[0]}")
print(f"  Última: {extended_coords[-1]}")

# Create land polygon
land_coords = []

# SE corner
land_coords.append((LON_MAX, LAT_MIN))
print(f"\n1. SE corner: ({LON_MAX}, {LAT_MIN})")

# NE corner
land_coords.append((LON_MAX, LAT_MAX))
print(f"2. NE corner: ({LON_MAX}, {LAT_MAX})")

# West to northernmost extended point
northmost_ext = extended_coords[-1]
land_coords.append((northmost_ext[0], LAT_MAX))
print(f"3. Norte costa ext: ({northmost_ext[0]:.6f}, {LAT_MAX})")

# Follow coastline from north to south
print(f"4. Seguir costa ({len(extended_coords)} puntos)...")
for lon, lat in reversed(extended_coords):
    land_coords.append((lon, lat))

# South extension
southmost_ext = extended_coords[0]
print(f"5. Punto sur ext: ({southmost_ext[0]:.6f}, {southmost_ext[1]:.6f})")

if southmost_ext[1] > LAT_MIN:
    print(f"   -> Agregar: ({southmost_ext[0]:.6f}, {LAT_MIN})")
    land_coords.append((southmost_ext[0], LAT_MIN))

land_coords.append((LON_MAX, LAT_MIN))
print(f"6. Cerrar en SE: ({LON_MAX}, {LAT_MIN})")

print(f"\nPolígono tierra tiene {len(land_coords)} puntos")

# Create polygon
land_poly = Polygon(land_coords)
print(f"Polígono válido: {land_poly.is_valid}")
print(f"Bounds tierra: {land_poly.bounds}")

if not land_poly.is_valid:
    land_poly = land_poly.buffer(0)
    print(f"Corregido: {land_poly.is_valid}")

# Water = bbox - land
bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
print(f"\nBbox bounds: {bbox.bounds}")

water_poly = bbox.difference(land_poly)
print(f"Water empty: {water_poly.is_empty}")
print(f"Water type: {type(water_poly).__name__}")

if isinstance(water_poly, Polygon):
    print(f"Water bounds: {water_poly.bounds}")
elif isinstance(water_poly, MultiPolygon):
    for i, p in enumerate(water_poly.geoms):
        print(f"  Water {i} bounds: {p.bounds}")

# Test points
print("\n" + "="*60)
print("TEST DE PUNTOS")
print("="*60)

test_points = [
    (-17.670, -71.350, "Sur-oeste"),
    (-17.675, -71.355, "Más sur-oeste"),
]

from shapely.geometry import Point
for lat, lon, desc in test_points:
    pt = Point(lon, lat)
    in_land = land_poly.contains(pt)
    in_water = not in_land and bbox.contains(pt)
    print(f"  ({lat}, {lon}) {desc}: tierra={in_land}, agua={in_water}")
