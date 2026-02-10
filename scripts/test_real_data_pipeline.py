#!/usr/bin/env python3
"""
Test script for Real Data Pipeline (V8 Phase 1).

Tests the OSM coastline loader and zone generation without CV detection.
Uses verified data sources: OSM coastlines + synthetic/GEBCO bathymetry.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.cv_analysis import (
    RealDataPipeline,
    RealDataConfig,
    OSMCoastlineLoader,
    load_coastline,
    export_result_to_file,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test area: Ilo, Peru (rocky coast)
ILO_BOUNDS = {
    'lat_min': -17.68,
    'lat_max': -17.65,
    'lon_min': -71.36,
    'lon_max': -71.33
}


def test_coastline_loader():
    """Test OSM coastline loading."""
    print("\n" + "="*60)
    print("TEST 1: OSM Coastline Loader")
    print("="*60)

    loader = OSMCoastlineLoader()
    result = loader.load_coastline(
        ILO_BOUNDS['lat_min'],
        ILO_BOUNDS['lat_max'],
        ILO_BOUNDS['lon_min'],
        ILO_BOUNDS['lon_max'],
        resolution_m=50.0
    )

    print(f"\nCoastline source: {result.source}")
    print(f"Coastline points: {len(result.coastline_points)}")
    print(f"Water polygons: {len(result.water_polygons)}")
    print(f"Land polygons: {len(result.land_polygons)}")
    print(f"Coastline length: {result.coastline_length_km:.2f} km")

    if result.coastline_points:
        print(f"\nFirst 5 coastline points:")
        for i, (lat, lon) in enumerate(result.coastline_points[:5]):
            print(f"  {i+1}. ({lat:.6f}, {lon:.6f})")

    return result


def test_real_data_pipeline():
    """Test the full real data pipeline."""
    print("\n" + "="*60)
    print("TEST 2: Real Data Pipeline")
    print("="*60)

    # Create pipeline with default config (no GEBCO file = synthetic depth)
    pipeline = RealDataPipeline()

    # Analyze the Ilo area
    result = pipeline.analyze_area(
        ILO_BOUNDS['lat_min'],
        ILO_BOUNDS['lat_max'],
        ILO_BOUNDS['lon_min'],
        ILO_BOUNDS['lon_max'],
        use_gebco=False,  # Use synthetic depth
        substrate_hint='rock'  # Ilo has rocky coast
    )

    print(f"\nProcessing Info:")
    for key, value in result.processing_info.items():
        print(f"  {key}: {value}")

    print(f"\nGenerated {len(result.zones)} fishing zones")

    if result.zones:
        print("\nTop zones by species:")
        species_summary = result.get_species_summary()
        for species, info in sorted(
            species_summary.items(),
            key=lambda x: x[1]['total_zones'],
            reverse=True
        )[:5]:
            print(f"  {info['name']}: {info['total_zones']} zones, "
                  f"{info['total_area_km2']:.4f} km²")

        print("\nFirst 3 zones:")
        for zone in result.zones[:3]:
            print(f"\n  Zone: {zone.zone_id}")
            print(f"    Distance: {zone.distance_from_coast_m[0]}-{zone.distance_from_coast_m[1]}m")
            print(f"    Avg depth: {zone.avg_depth_m:.1f}m")
            print(f"    Substrate: {zone.substrate.value}")
            print(f"    Primary species: {zone.primary_species}")
            print(f"    Secondary: {', '.join(zone.secondary_species)}")

    return result


def test_export_geojson(result):
    """Test exporting to GeoJSON."""
    print("\n" + "="*60)
    print("TEST 3: Export to GeoJSON")
    print("="*60)

    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'real_data_zones_ilo.geojson'

    success = export_result_to_file(result, output_path)

    if success:
        print(f"\nExported to: {output_path}")

        # Show file size
        size = output_path.stat().st_size
        print(f"File size: {size:,} bytes")

        # Count features
        with open(output_path) as f:
            data = json.load(f)
            print(f"Total features: {len(data['features'])}")

    return success


def create_viewer_html(result):
    """Create an HTML viewer for the results."""
    print("\n" + "="*60)
    print("TEST 4: Create HTML Viewer")
    print("="*60)

    output_dir = project_root / 'output'
    geojson = result.to_geojson()

    # Generate species legend
    species_legend = []
    for zone in result.zones[:10]:  # Top zones
        if zone.primary_species not in [s['species'] for s in species_legend]:
            from core.cv_analysis import SPECIES_DATABASE
            habitat = SPECIES_DATABASE.get(zone.primary_species)
            if habitat:
                score = zone.species_scores.get(zone.primary_species, 0)
                species_legend.append({
                    'species': zone.primary_species,
                    'name': habitat.name_es,
                    'color': f'rgb({habitat.color[0]},{habitat.color[1]},{habitat.color[2]})',
                    'score': score
                })

    html_content = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Data Pipeline V8 - Zonas de Pesca Ilo</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 60px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        }}
        .info-panel h2 {{ font-size: 16px; margin-bottom: 10px; color: #1a73e8; }}
        .info-panel h3 {{ font-size: 13px; margin: 10px 0 5px; color: #333; }}
        .stat {{ display: flex; justify-content: space-between; margin: 5px 0; font-size: 13px; }}
        .stat-label {{ color: #666; }}
        .stat-value {{ font-weight: 600; color: #333; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            color: white;
            background: #4CAF50;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            max-height: 300px;
            overflow-y: auto;
        }}
        .legend h3 {{ margin-bottom: 10px; font-size: 14px; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
        .legend-color {{ width: 18px; height: 18px; border-radius: 3px; margin-right: 8px; border: 1px solid #999; }}
        .layer-control {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .layer-control label {{ display: flex; align-items: center; font-size: 13px; margin: 5px 0; cursor: pointer; }}
        .layer-control input {{ margin-right: 8px; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="info-panel">
        <h2>Real Data Pipeline V8</h2>
        <span class="badge">Datos Verificados</span>

        <h3>Fuentes de Datos</h3>
        <div class="stat">
            <span class="stat-label">Linea costera:</span>
            <span class="stat-value">{result.coastline.source.upper()}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Batimetria:</span>
            <span class="stat-value">{result.processing_info.get('depth_source', 'N/A').upper()}</span>
        </div>

        <h3>Resultados</h3>
        <div class="stat">
            <span class="stat-label">Pts costa:</span>
            <span class="stat-value">{len(result.coastline.coastline_points):,}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Zonas generadas:</span>
            <span class="stat-value">{len(result.zones)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Long. costa:</span>
            <span class="stat-value">{result.coastline.coastline_length_km:.2f} km</span>
        </div>

        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #eee;font-size:11px;color:#666;">
            Click en una zona para ver especies recomendadas
        </div>
    </div>

    <div class="layer-control">
        <label><input type="checkbox" id="showCoastline" checked> Linea costera</label>
        <label><input type="checkbox" id="showZones" checked> Zonas de pesca</label>
        <label><input type="checkbox" id="showSatellite" checked> Satelite</label>
    </div>

    <div class="legend">
        <h3>Especies por Zona</h3>
        {"".join(f'<div class="legend-item"><div class="legend-color" style="background:{s["color"]};"></div>{s["name"]} ({int(s["score"]*100)}%)</div>' for s in species_legend)}
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const geojsonData = {json.dumps(geojson)};

        const map = L.map('map').setView([{(ILO_BOUNDS['lat_min'] + ILO_BOUNDS['lat_max'])/2}, {(ILO_BOUNDS['lon_min'] + ILO_BOUNDS['lon_max'])/2}], 14);

        const satellite = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{ maxZoom: 19, attribution: 'ESRI' }}
        ).addTo(map);

        // Separate coastline and zone features
        const coastlineFeatures = geojsonData.features.filter(f => f.properties.type === 'coastline');
        const zoneFeatures = geojsonData.features.filter(f => f.properties.zone_id);

        // Coastline layer (green)
        const coastlineLayer = L.geoJSON({{type: 'FeatureCollection', features: coastlineFeatures}}, {{
            style: {{ color: '#00FF00', weight: 3, opacity: 0.9 }}
        }}).addTo(map);

        // Zones layer
        const zonesLayer = L.geoJSON({{type: 'FeatureCollection', features: zoneFeatures}}, {{
            style: function(feature) {{
                const color = feature.properties.color || 'rgb(70,130,180)';
                return {{
                    fillColor: color,
                    fillOpacity: 0.5,
                    color: color,
                    weight: 2,
                    opacity: 0.8
                }};
            }},
            onEachFeature: function(feature, layer) {{
                const p = feature.properties;
                const scores = Object.entries(p.species_scores || {{}})
                    .sort((a,b) => b[1]-a[1])
                    .slice(0, 6)
                    .map(s => `<tr><td>${{s[0]}}</td><td><b>${{(s[1]*100).toFixed(0)}}%</b></td></tr>`)
                    .join('');

                layer.bindPopup(`
                    <div style="min-width:220px">
                        <h3 style="color:#1a73e8;margin:0 0 10px 0">${{p.primary_species.toUpperCase()}}</h3>
                        <div><b>Distancia costa:</b> ${{p.distance_from_coast_min}}-${{p.distance_from_coast_max}}m</div>
                        <div><b>Profundidad:</b> ${{p.avg_depth.toFixed(1)}}m</div>
                        <div><b>Sustrato:</b> ${{p.substrate}}</div>
                        <div><b>Zona:</b> ${{p.depth_zone}}</div>
                        <hr style="margin:10px 0">
                        <div><b>Afinidad por especie:</b></div>
                        <table style="width:100%;font-size:12px;margin-top:5px">${{scores}}</table>
                    </div>
                `);
            }}
        }}).addTo(map);

        // Fit bounds
        const bounds = L.latLngBounds([
            [{ILO_BOUNDS['lat_min']}, {ILO_BOUNDS['lon_min']}],
            [{ILO_BOUNDS['lat_max']}, {ILO_BOUNDS['lon_max']}]
        ]);
        map.fitBounds(bounds, {{ padding: [30, 30] }});

        // Layer controls
        document.getElementById('showCoastline').onchange = e =>
            e.target.checked ? map.addLayer(coastlineLayer) : map.removeLayer(coastlineLayer);
        document.getElementById('showZones').onchange = e =>
            e.target.checked ? map.addLayer(zonesLayer) : map.removeLayer(zonesLayer);
        document.getElementById('showSatellite').onchange = e =>
            e.target.checked ? map.addLayer(satellite) : map.removeLayer(satellite);
    </script>
</body>
</html>'''

    output_path = output_dir / 'real_data_viewer_v8.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nViewer created: {output_path}")
    return output_path


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REAL DATA PIPELINE V8 - TEST SUITE")
    print("="*60)
    print("\nArea de prueba: Ilo, Peru (costa rocosa)")
    print(f"Bounds: {ILO_BOUNDS}")

    # Test 1: Coastline loader
    coastline_result = test_coastline_loader()

    # Test 2: Full pipeline
    pipeline_result = test_real_data_pipeline()

    # Test 3: Export to GeoJSON
    test_export_geojson(pipeline_result)

    # Test 4: Create viewer
    viewer_path = create_viewer_html(pipeline_result)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print(f"\nViewer disponible en: {viewer_path}")
    print("Abre el archivo HTML en un navegador para ver los resultados.")


if __name__ == '__main__':
    main()
