#!/usr/bin/env python
"""Generate standalone HTML PWA for iPhone."""

import json
from datetime import datetime
import sys
sys.path.insert(0, '..')

from controllers.analysis import AnalysisController
from domain import PERU_SOUTH_LIMIT

def generate_standalone_html():
    # Run analysis
    print("Ejecutando analisis...")
    c = AnalysisController()
    c.analysis_datetime = datetime(2026, 2, 7, 16, 27)

    c.load_coastline('data/gold/coastline/coastline_v8_extended.geojson')
    c.sample_fishing_spots(spacing_m=300, max_spots=600)
    c.generate_fish_zones()
    c.predict_anchovy_migration()
    c.fetch_marine_data()
    c.get_conditions()
    c.run_ml_prediction()
    hourly_data = c.analyze_spots_all_hours()

    # Filter Peru spots for 6PM
    spots_6pm = hourly_data[18]
    peru_spots = [s for s in spots_6pm if s['lat'] >= PERU_SOUTH_LIMIT]
    peru_spots.sort(key=lambda x: x['score'], reverse=True)

    # Prepare data
    scores = [s['score'] for s in peru_spots]
    min_score, max_score = min(scores), max(scores)

    spots_data = [{
        'lat': s['lat'],
        'lon': s['lon'],
        'score': round(s['score'], 1),
        'tide': s.get('tide_phase', 'N/A'),
        'species': [sp['name'] if isinstance(sp, dict) else sp for sp in s.get('species', [])[:2]]
    } for s in peru_spots]

    spots_json = json.dumps(spots_data)
    top5 = peru_spots[:5]

    # Generate top5 list HTML
    top5_html = ''
    for i, s in enumerate(top5):
        top5_html += f'<li onclick="map.setView([{s["lat"]}, {s["lon"]}], 14)"><span class="rank">{i+1}</span><b>{s["score"]:.1f}</b> <small style="color:#666;">{s["lat"]:.4f}, {s["lon"]:.4f}</small></li>'

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="PescaApp">
    <title>Pesca Predictor - 7 Feb 2026</title>
    <link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%231a5f7a' width='100' height='100' rx='20'/><text x='50' y='65' text-anchor='middle' font-size='50'>🐟</text></svg>">

    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>

    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ height: 100%; font-family: -apple-system, system-ui, sans-serif; }}

        #map {{ height: 100%; width: 100%; }}

        .legend {{
            background: rgba(255,255,255,0.95);
            padding: 12px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-size: 12px;
            max-width: 160px;
        }}
        .legend h4 {{ margin: 0 0 8px 0; color: #1a5f7a; }}
        .legend .gradient {{
            height: 12px;
            border-radius: 6px;
            background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff8000, #ff0000);
            margin: 6px 0;
        }}
        .legend .labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #666;
        }}

        .info-panel {{
            background: rgba(255,255,255,0.97);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.2);
            font-size: 13px;
            max-width: 280px;
            max-height: 70vh;
            overflow-y: auto;
        }}
        .info-panel h3 {{ color: #1a5f7a; margin: 0 0 10px 0; }}
        .info-panel .best {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
            color: white;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 12px;
        }}
        .info-panel .best .score {{ font-size: 28px; font-weight: bold; }}
        .info-panel .spot-list {{ list-style: none; }}
        .info-panel .spot-list li {{
            padding: 10px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
        }}
        .info-panel .spot-list li:active {{ background: #f0f0f0; }}
        .info-panel .spot-list .rank {{
            display: inline-block;
            width: 24px;
            height: 24px;
            background: #1a5f7a;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            margin-right: 8px;
            font-size: 11px;
        }}

        .popup-content {{ font-size: 13px; }}
        .popup-content h4 {{ margin: 0 0 8px 0; color: #1a5f7a; }}
        .popup-content .score-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: bold;
            color: white;
        }}

        @media (max-width: 500px) {{
            .info-panel {{
                max-width: calc(100vw - 20px);
                font-size: 12px;
            }}
        }}

        /* iOS safe areas */
        @supports (padding: env(safe-area-inset-top)) {{
            .info-panel {{ margin-top: env(safe-area-inset-top); }}
            .legend {{ margin-bottom: env(safe-area-inset-bottom); }}
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
        // Spot data
        const spots = {spots_json};
        const minScore = {min_score};
        const maxScore = {max_score};

        // Initialize map
        const map = L.map('map').setView([{top5[0]['lat']}, {top5[0]['lon']}], 11);

        // Tile layers
        const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri'
        }});

        const streets = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: 'OSM'
        }});

        satellite.addTo(map);

        L.control.layers({{'Satelite': satellite, 'Calles': streets}}).addTo(map);

        // Heatmap color function
        function getColor(score) {{
            const norm = (score - minScore) / (maxScore - minScore);
            let r, g, b;

            if (norm < 0.2) {{
                r = 0; g = Math.round(255 * norm * 5); b = 255;
            }} else if (norm < 0.4) {{
                const t = (norm - 0.2) * 5;
                r = 0; g = 255; b = Math.round(255 * (1 - t));
            }} else if (norm < 0.6) {{
                const t = (norm - 0.4) * 5;
                r = Math.round(255 * t); g = 255; b = 0;
            }} else if (norm < 0.8) {{
                const t = (norm - 0.6) * 5;
                r = 255; g = Math.round(255 * (1 - t * 0.5)); b = 0;
            }} else {{
                const t = (norm - 0.8) * 5;
                r = 255; g = Math.round(128 * (1 - t)); b = 0;
            }}

            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }}

        // Add spots
        spots.forEach((spot, i) => {{
            const isTop5 = i < 5;
            const color = i === 0 ? '#FF0000' : getColor(spot.score);

            const marker = L.circleMarker([spot.lat, spot.lon], {{
                radius: isTop5 ? 12 : 6,
                fillColor: color,
                fillOpacity: 0.85,
                color: '#000',
                weight: isTop5 ? 2 : 1
            }}).addTo(map);

            const title = i === 0 ? '🏆 #1 MEJOR' : '#' + (i+1);
            const speciesText = spot.species.join(', ') || 'N/A';

            const popupHtml = '<div class="popup-content">' +
                '<h4>' + title + '</h4>' +
                '<span class="score-badge" style="background:' + color + '">' + spot.score + '/100</span>' +
                '<p style="margin:8px 0 4px;">' +
                '<b>Hora:</b> 18:00<br>' +
                '<b>Marea:</b> ' + spot.tide + '<br>' +
                '<b>Especies:</b> ' + speciesText +
                '</p>' +
                '<small style="color:#666;">' + spot.lat.toFixed(5) + ', ' + spot.lon.toFixed(5) + '</small>' +
                '</div>';

            marker.bindPopup(popupHtml);
        }});

        // Legend
        const legend = L.control({{position: 'bottomleft'}});
        legend.onAdd = function() {{
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4>🎣 Pesca Predictor</h4>' +
                '<small style="color:#666;">7 Feb 2026 - 18:00</small>' +
                '<hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">' +
                '<b>Score:</b>' +
                '<div class="gradient"></div>' +
                '<div class="labels"><span>{int(min_score)}</span><span>{int(max_score)}</span></div>' +
                '<div style="margin-top:6px;"><span style="color:#FF0000;">●</span> #1 Mejor</div>';
            return div;
        }};
        legend.addTo(map);

        // Info panel
        const info = L.control({{position: 'topright'}});
        info.onAdd = function() {{
            const div = L.DomUtil.create('div', 'info-panel');
            div.innerHTML = '<h3>Top 5 - Peru 🇵🇪</h3>' +
                '<div class="best">' +
                '<div>🏆 Mejor Spot</div>' +
                '<div class="score">{top5[0]["score"]:.1f}</div>' +
                '<small>{top5[0]["lat"]:.4f}, {top5[0]["lon"]:.4f}</small>' +
                '</div>' +
                '<ul class="spot-list">' +
                '{top5_html}' +
                '</ul>' +
                '<div style="margin-top:12px;padding:10px;background:#e3f2fd;border-radius:8px;font-size:11px;">' +
                '<b>⏰ Hora optima:</b> 18:00<br>' +
                '<b>🌊 Marea:</b> Bajando<br>' +
                '<b>🐟 Especies:</b> Corvina, Cabrilla' +
                '</div>';
            return div;
        }};
        info.addTo(map);
    </script>
</body>
</html>'''

    # Save
    output_path = 'output/pesca_app.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'\n✅ Archivo creado: {output_path}')
    print(f'📦 Tamaño: {len(html) / 1024:.1f} KB')
    print('')
    print('📱 Para enviar a tu iPhone:')
    print('   1. AirDrop: Click derecho > Compartir > AirDrop')
    print('   2. Email: Adjunta el archivo')
    print('   3. iCloud: Sube a iCloud Drive')
    print('')
    print('📲 En tu iPhone:')
    print('   1. Abre el archivo en Safari')
    print('   2. Toca Compartir > "Añadir a pantalla de inicio"')

    return output_path

if __name__ == '__main__':
    generate_standalone_html()
