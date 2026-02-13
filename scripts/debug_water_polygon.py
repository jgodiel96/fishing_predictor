#!/usr/bin/env python3
"""
Debug script to visualize the water polygon generation problem.
"""

import sys
sys.path.insert(0, '/Users/jorgegodiel/Documents/codes/Proyecto_pesca/fishing_predictor')

from shapely.geometry import Point, box
from core.cv_analysis.osm_coastline import OSMCoastlineLoader

# Test area - Ilo
LAT_MIN, LAT_MAX = -17.68, -17.65
LON_MIN, LON_MAX = -71.36, -71.33

print("="*60)
print("DIAGNÓSTICO DEL POLÍGONO DE AGUA")
print("="*60)

# Load coastline
loader = OSMCoastlineLoader()
result = loader.load_coastline(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

print(f"\nFuente: {result.source}")
print(f"Puntos costa: {len(result.coastline_points)}")
print(f"Polígonos agua: {len(result.water_polygons)}")

# Show coastline extent
if result.coastline_points:
    lats = [p[0] for p in result.coastline_points]
    lons = [p[1] for p in result.coastline_points]
    print(f"\nExtensión costa:")
    print(f"  Lat: {min(lats):.6f} a {max(lats):.6f}")
    print(f"  Lon: {min(lons):.6f} a {max(lons):.6f}")
    print(f"\nBbox solicitado:")
    print(f"  Lat: {LAT_MIN:.6f} a {LAT_MAX:.6f}")
    print(f"  Lon: {LON_MIN:.6f} a {LON_MAX:.6f}")

    gap_south = min(lats) - LAT_MIN
    gap_north = LAT_MAX - max(lats)
    print(f"\nBrechas:")
    print(f"  Sur: {gap_south:.6f} grados ({gap_south * 111:.0f} km)")
    print(f"  Norte: {gap_north:.6f} grados ({gap_north * 111:.0f} km)")

# Test key points
print("\n" + "="*60)
print("VERIFICACIÓN DE PUNTOS")
print("="*60)

test_points = [
    # (lat, lon, expected, description)
    # Points WITHIN the coastline data extent (lat -17.65 to -17.6648)
    # Coast is at approximately lon -71.355 (north) to -71.36 (south)
    (-17.655, -71.356, "AGUA", "Norte-oeste (debería ser agua)"),
    (-17.655, -71.335, "TIERRA", "Norte-este (debería ser tierra)"),
    (-17.660, -71.359, "AGUA", "Centro-oeste - al oeste de costa (debería ser agua)"),
    (-17.660, -71.340, "TIERRA", "Centro-este (debería ser tierra)"),
    (-17.663, -71.3595, "AGUA", "Sur-oeste (ligeramente dentro del agua)"),
    (-17.663, -71.345, "TIERRA", "Sur-este (debería ser tierra)"),
]

errors = 0
for lat, lon, expected, desc in test_points:
    is_water = loader.is_in_water(lat, lon, result.water_polygons)
    actual = "AGUA" if is_water else "TIERRA"
    status = "✓" if actual == expected else "✗ ERROR"
    if actual != expected:
        errors += 1
    print(f"  ({lat:.3f}, {lon:.3f}): {actual:6s} - {desc} [{status}]")

print(f"\nErrores: {errors}/{len(test_points)}")

# Show water polygon details
if result.water_polygons:
    print("\n" + "="*60)
    print("DETALLE DEL POLÍGONO DE AGUA")
    print("="*60)

    for i, poly in enumerate(result.water_polygons):
        bounds = poly.bounds
        print(f"\nPolígono {i}:")
        print(f"  Bounds: lon({bounds[0]:.6f} a {bounds[2]:.6f}), lat({bounds[1]:.6f} a {bounds[3]:.6f})")
        print(f"  Área: {poly.area * 111 * 111:.4f} km²")

        # Check if polygon covers full width (problem indicator)
        width = bounds[2] - bounds[0]
        bbox_width = LON_MAX - LON_MIN
        if width > bbox_width * 0.9:
            print(f"  ⚠️ PROBLEMA: Polígono cubre casi todo el ancho ({width/bbox_width*100:.0f}%)")

# Generate diagnostic HTML
html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Diagnóstico Agua/Tierra</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        #map {{ height: 100vh; width: 100%; }}
        .legend {{ background: white; padding: 10px; position: absolute; bottom: 20px; left: 20px; z-index: 1000; border-radius: 5px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="legend">
        <b>Diagnóstico</b><br>
        🟢 Verde = Costa OSM<br>
        🔵 Azul = Polígono agua calculado<br>
        ⬜ Puntos = Test agua/tierra
    </div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{(LAT_MIN+LAT_MAX)/2}, {(LON_MIN+LON_MAX)/2}], 15);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}').addTo(map);

        // Bbox
        L.rectangle([[{LAT_MIN}, {LON_MIN}], [{LAT_MAX}, {LON_MAX}]], {{color: 'yellow', fill: false, weight: 2}}).addTo(map);

        // Coastline
        var coastCoords = {[[p[1], p[0]] for p in result.coastline_points]};
        L.polyline(coastCoords.map(c => [c[1], c[0]]), {{color: 'lime', weight: 4}}).addTo(map);

        // Water polygons
'''

for poly in result.water_polygons:
    coords = list(poly.exterior.coords)
    html += f'''
        L.polygon({[[c[1], c[0]] for c in coords]}, {{color: 'blue', fillColor: 'blue', fillOpacity: 0.3, weight: 2}}).addTo(map);
'''

html += '''
        // Test points
'''
for lat, lon, expected, desc in test_points:
    is_water = loader.is_in_water(lat, lon, result.water_polygons)
    color = 'cyan' if is_water else 'red'
    icon_color = 'blue' if expected == "AGUA" else 'orange'
    status = '✓' if (is_water and expected == "AGUA") or (not is_water and expected == "TIERRA") else '✗'
    html += f'''
        L.circleMarker([{lat}, {lon}], {{radius: 8, color: '{color}', fillColor: '{color}', fillOpacity: 0.8}})
            .bindPopup("{desc}<br>Resultado: {'AGUA' if is_water else 'TIERRA'}<br>Esperado: {expected} {status}").addTo(map);
'''

html += '''
    </script>
</body>
</html>
'''

output_path = '/Users/jorgegodiel/Documents/codes/Proyecto_pesca/fishing_predictor/output/debug_water.html'
with open(output_path, 'w') as f:
    f.write(html)

print(f"\n✓ Diagnóstico guardado en: {output_path}")
