"""
Map component using deck.gl + MapLibre GL JS.
Replaces Folium for GPU-accelerated rendering of 200k+ spots.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from views.styles.map_styles import (
    COLORS, get_sst_color, get_flow_color, get_zone_colors,
    get_anchovy_colors, get_heatmap_color
)


@dataclass
class MapConfig:
    """Configuration for map rendering."""
    center: Tuple[float, float] = (-17.9, -71.0)
    zoom: int = 10
    tile_layer: str = "satellite"


class MapComponent:
    """
    Core map component using deck.gl + MapLibre GL JS.
    Accumulates layer data, then renders a self-contained HTML file.
    """

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self._spot_score_range = (0, 100)

        # Data accumulators (populated by add_* methods)
        self._coastline_segments: List[List[Tuple[float, float]]] = []
        self._fish_zones: List[Dict] = []
        self._anchovy_zones: List[Dict] = []
        self._flow_lines_data: List[Dict] = []
        self._marine_points: List[Dict] = []
        self._spots_data: List[Dict] = []
        self._top_spots: List[Dict] = []
        self._heatmap_data: List[Dict] = []
        self._user_location: Optional[Dict] = None
        self._extra_html: List[str] = []

    @property
    def map(self):
        """Backward compat — returns self so .get_root().html.add_child() calls work."""
        return self

    def get_root(self):
        """Backward compat shim for folium.Element injection."""
        return self

    @property
    def html(self):
        return self

    def add_child(self, element):
        """Capture HTML/JS strings that panels inject."""
        if hasattr(element, '_html'):
            self._extra_html.append(element._html)
        elif isinstance(element, str):
            self._extra_html.append(element)

    def create(self, center: Tuple[float, float] = None, zoom: int = None):
        """Initialize map config."""
        if center:
            self.config.center = center
        if zoom:
            self.config.zoom = zoom
        return self

    def add_coastline(self, points: List[Tuple[float, float]],
                      segments: List[List[Tuple[float, float]]] = None):
        if segments and len(segments) > 0:
            self._coastline_segments = [s for s in segments if len(s) >= 2]
        elif points:
            self._coastline_segments = [points]

    def add_fish_zones(self, zones: List[Dict]):
        if not zones:
            return
        for z in zones:
            if z.get('is_anchovy'):
                self._anchovy_zones.append(z)
            else:
                self._fish_zones.append(z)

    def add_flow_lines(self, flow_lines: List[List[Tuple[float, float]]], vectors: List = None):
        if not flow_lines:
            return
        for i, line in enumerate(flow_lines):
            if len(line) < 2:
                continue
            speed = vectors[i].speed if vectors and i < len(vectors) else 0.1
            color = get_flow_color(speed)
            self._flow_lines_data.append({
                'path': [[lon, lat] for lat, lon in line],
                'color': self._hex_to_rgb(color),
                'speed': speed
            })

    def add_marine_points(self, points: List):
        if not points:
            return
        for p in points:
            sst = p.sst if p.sst is not None else 17.0
            color = get_sst_color(sst)
            wave = p.wave_height if p.wave_height else 0
            spd = p.current_speed if p.current_speed else 0
            self._marine_points.append({
                'position': [p.lon, p.lat],
                'sst': round(sst, 1),
                'wave': round(wave, 1),
                'speed': round(spd, 2),
                'color': self._hex_to_rgb(color)
            })

    def add_fishing_spots(self, spots: List[Dict], top_n: int = 5):
        if not spots:
            return
        scores = [s['score'] for s in spots]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 100
        self._spot_score_range = (min_score, max_score)

        # Subsample for heatmap (max 20k points)
        N = len(spots)
        MAX_HEAT = 20000
        stride = max(1, N // MAX_HEAT)
        score_range = max(max_score - min_score, 1)

        self._spots_data = [
            {
                'position': [spots[i]['lon'], spots[i]['lat']],
                'weight': max(0.01, (spots[i]['score'] - min_score) / score_range)
            }
            for i in range(0, N, stride)
            if spots[i]['score'] > 0
        ]

        # Top spots for markers
        top_count = min(top_n * 4, 20)
        for i, spot in enumerate(spots[:top_count]):
            is_best = (i == 0)
            is_top = (i < top_n)
            color = '#FF0000' if is_best else get_heatmap_color(spot['score'], min_score, max_score)

            species = spot.get('species', [])
            species_names = [s['name'] for s in species[:3]] if species else []

            self._top_spots.append({
                'position': [spot['lon'], spot['lat']],
                'score': round(spot['score'], 1),
                'color': self._hex_to_rgb(color),
                'radius': 14 if is_best else (10 if is_top else 7),
                'rank': i + 1,
                'is_top': is_top,
                'species': species_names,
                'distance_to_fish': spot.get('distance_to_fish', 0),
                'lat': spot['lat'],
                'lon': spot['lon']
            })

    def add_heatmap(self, heatmap_data: List[Dict], show: bool = False):
        self._heatmap_data = [
            {'position': [h['lon'], h['lat']], 'weight': h['intensity']}
            for h in heatmap_data if h.get('intensity', 0) > 0
        ] if heatmap_data else []

    def add_user_location(self, lat: float, lon: float, radius_km: float = 5):
        self._user_location = {
            'position': [lon, lat],
            'lat': lat, 'lon': lon,
            'radius': radius_km * 1000
        }

    def finalize(self):
        return self

    def save(self, filepath: str):
        html = self.render_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    def get_score_range(self) -> Tuple[float, float]:
        return self._spot_score_range

    def render_html(self) -> str:
        """Generate the complete self-contained HTML with deck.gl + MapLibre."""
        center_lon, center_lat = self.config.center[1], self.config.center[0]

        # Serialize all layer data
        data_js = self._build_data_js()

        return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictor de Pesca - Mapa Interactivo</title>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/deck.gl@9.1.4/dist.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
#map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}

/* Tooltip */
#deck-tooltip {{
    position: absolute; z-index: 2000; pointer-events: none;
    background: rgba(0,0,0,0.85); color: white; padding: 8px 12px;
    border-radius: 6px; font-size: 12px; max-width: 250px;
    display: none; white-space: nowrap;
}}

/* Layer control */
#layer-control {{
    position: fixed; top: 60px; left: 10px; z-index: 1001;
    background: rgba(255,255,255,0.95); padding: 10px 14px;
    border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    font-size: 12px; max-height: 80vh; overflow-y: auto;
}}
#layer-control label {{ display: block; margin: 4px 0; cursor: pointer; }}
#layer-control input {{ margin-right: 6px; }}
#layer-toggle-btn {{
    position: fixed; top: 60px; left: 10px; z-index: 1001;
    background: rgba(255,255,255,0.95); padding: 8px 12px;
    border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    cursor: pointer; font-size: 12px; display: none; border: none;
}}
</style>
</head>
<body>
<div id="map"></div>
<div id="deck-tooltip"></div>

<div id="layer-control">
    <b>Capas</b>
    <button onclick="toggleLayerControl()" style="float:right;border:none;background:none;cursor:pointer;">X</button>
    <hr style="margin:5px 0;">
    <label><input type="checkbox" checked onchange="toggleLayer('heatmap', this.checked)"> Mapa de Scores</label>
    <label><input type="checkbox" checked onchange="toggleLayer('coastline', this.checked)"> Costa</label>
    <label><input type="checkbox" checked onchange="toggleLayer('topSpots', this.checked)"> Top Spots</label>
    <label><input type="checkbox" checked onchange="toggleLayer('fishZones', this.checked)"> Zonas de Peces</label>
    <label><input type="checkbox" checked onchange="toggleLayer('anchovyZones', this.checked)"> Anchoveta</label>
    <label><input type="checkbox" checked onchange="toggleLayer('flowLines', this.checked)"> Corrientes</label>
    <label><input type="checkbox" checked onchange="toggleLayer('marine', this.checked)"> Datos Marinos</label>
    <label><input type="checkbox" onchange="toggleLayer('historical', this.checked)"> Heatmap Historico</label>
</div>
<button id="layer-toggle-btn" onclick="toggleLayerControl()">Capas</button>

{data_js}

<script>
// === MapLibre base map ===
const map = new maplibregl.Map({{
    container: 'map',
    style: {{
        version: 8,
        sources: {{
            'satellite': {{
                type: 'raster',
                tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'],
                tileSize: 256,
                attribution: 'Esri'
            }},
            'osm': {{
                type: 'raster',
                tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
                tileSize: 256,
                attribution: 'OSM'
            }}
        }},
        layers: [
            {{ id: 'satellite', type: 'raster', source: 'satellite', layout: {{ visibility: 'visible' }} }},
            {{ id: 'osm', type: 'raster', source: 'osm', layout: {{ visibility: 'none' }} }}
        ]
    }},
    center: [{center_lon}, {center_lat}],
    zoom: {self.config.zoom},
    antialias: true
}});

map.addControl(new maplibregl.NavigationControl(), 'top-left');

// Tile switcher
const tileBtn = document.createElement('button');
tileBtn.textContent = 'Calles';
tileBtn.style.cssText = 'position:fixed;top:10px;left:50px;z-index:1001;background:white;border:1px solid #ccc;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:12px;';
document.body.appendChild(tileBtn);
let showSatellite = true;
tileBtn.onclick = function() {{
    showSatellite = !showSatellite;
    map.setLayoutProperty('satellite', 'visibility', showSatellite ? 'visible' : 'none');
    map.setLayoutProperty('osm', 'visibility', showSatellite ? 'none' : 'visible');
    tileBtn.textContent = showSatellite ? 'Calles' : 'Satelite';
}};

// === Layer visibility state ===
const layerVis = {{
    heatmap: true, coastline: true, topSpots: true, fishZones: true,
    anchovyZones: true, flowLines: true, marine: true, historical: false
}};

function toggleLayer(name, show) {{
    layerVis[name] = show;
    updateDeckLayers();
}}

function toggleLayerControl() {{
    const ctrl = document.getElementById('layer-control');
    const btn = document.getElementById('layer-toggle-btn');
    if (ctrl.style.display === 'none') {{
        ctrl.style.display = 'block'; btn.style.display = 'none';
    }} else {{
        ctrl.style.display = 'none'; btn.style.display = 'block';
    }}
}}

// === Tooltip ===
const tooltip = document.getElementById('deck-tooltip');

function showTooltip(info) {{
    if (info.object) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (info.x + 10) + 'px';
        tooltip.style.top = (info.y + 10) + 'px';
        tooltip.innerHTML = info.object._tooltip || '';
    }} else {{
        tooltip.style.display = 'none';
    }}
}}

// === Score color helper ===
function getScoreColor(score) {{
    if (score >= 80) return [34,139,34];
    if (score >= 60) return [50,205,50];
    if (score >= 40) return [255,215,0];
    return [220,20,60];
}}

// === Build deck.gl layers ===
// Dynamic hourly data (set by hourly panel)
let currentHourlySpots = null;

function buildLayers() {{
    const layers = [];

    // Score heatmap
    if (layerVis.heatmap && SPOTS_DATA.length > 0) {{
        const heatSrc = currentHourlySpots || SPOTS_DATA;
        layers.push(new deck.HeatmapLayer({{
            id: 'score-heatmap',
            data: heatSrc,
            getPosition: d => d.position,
            getWeight: d => d.weight,
            radiusPixels: 25,
            intensity: 1.2,
            threshold: 0.05,
            colorRange: [
                [26,5,48], [59,7,100], [220,38,38],
                [249,115,22], [234,179,8], [132,204,22], [34,197,94]
            ]
        }}));
    }}

    // Coastline
    if (layerVis.coastline && COASTLINE_DATA.length > 0) {{
        layers.push(new deck.PathLayer({{
            id: 'coastline',
            data: COASTLINE_DATA,
            getPath: d => d.path,
            getColor: [204, 204, 0, 230],
            widthMinPixels: 3,
            widthMaxPixels: 4
        }}));
    }}

    // Fish zones (circles)
    if (layerVis.fishZones && FISH_ZONES.length > 0) {{
        layers.push(new deck.ScatterplotLayer({{
            id: 'fish-zones-fill',
            data: FISH_ZONES,
            getPosition: d => d.position,
            getRadius: d => d.radius,
            getFillColor: d => [...d.fillColor, 90],
            getLineColor: d => [...d.borderColor, 200],
            lineWidthMinPixels: 3,
            stroked: true,
            filled: true,
            pickable: true,
            onHover: showTooltip
        }}));
        layers.push(new deck.ScatterplotLayer({{
            id: 'fish-zones-center',
            data: FISH_ZONES,
            getPosition: d => d.position,
            getRadius: 120,
            getFillColor: d => [...d.borderColor, 255],
            getLineColor: [0, 0, 0, 200],
            lineWidthMinPixels: 2,
            stroked: true,
            filled: true
        }}));
    }}

    // Anchovy zones
    if (layerVis.anchovyZones && ANCHOVY_ZONES.length > 0) {{
        layers.push(new deck.ScatterplotLayer({{
            id: 'anchovy-zones',
            data: ANCHOVY_ZONES,
            getPosition: d => d.position,
            getRadius: d => d.radius,
            getFillColor: d => [...d.fillColor, 100],
            getLineColor: d => [...d.borderColor, 200],
            lineWidthMinPixels: 4,
            stroked: true,
            filled: true,
            pickable: true,
            onHover: showTooltip
        }}));
        layers.push(new deck.TextLayer({{
            id: 'anchovy-labels',
            data: ANCHOVY_ZONES,
            getPosition: d => d.position,
            getText: () => '🐟',
            getSize: 20,
            getTextAnchor: 'middle',
            getAlignmentBaseline: 'center'
        }}));
    }}

    // Flow lines
    if (layerVis.flowLines && FLOW_LINES.length > 0) {{
        layers.push(new deck.PathLayer({{
            id: 'flow-lines',
            data: FLOW_LINES,
            getPath: d => d.path,
            getColor: d => [...d.color, 130],
            widthMinPixels: 2,
            widthMaxPixels: 3
        }}));
    }}

    // Marine SST points
    if (layerVis.marine && MARINE_POINTS.length > 0) {{
        layers.push(new deck.ScatterplotLayer({{
            id: 'marine-points',
            data: MARINE_POINTS,
            getPosition: d => d.position,
            getRadius: 500,
            getFillColor: d => [...d.color, 200],
            getLineColor: [0, 0, 0, 180],
            lineWidthMinPixels: 1,
            stroked: true,
            filled: true,
            pickable: true,
            onHover: showTooltip
        }}));
    }}

    // Historical heatmap
    if (layerVis.historical && HISTORICAL_HEATMAP.length > 0) {{
        layers.push(new deck.HeatmapLayer({{
            id: 'historical-heatmap',
            data: HISTORICAL_HEATMAP,
            getPosition: d => d.position,
            getWeight: d => d.weight,
            radiusPixels: 30,
            intensity: 1.0,
            threshold: 0.1,
            colorRange: [
                [0,0,255], [0,255,255], [0,255,0],
                [255,255,0], [255,128,0], [255,0,0]
            ]
        }}));
    }}

    // Top spots (markers)
    if (layerVis.topSpots && TOP_SPOTS.length > 0) {{
        // Use hourly top spots if available
        const topSrc = currentHourlyTop || TOP_SPOTS;
        layers.push(new deck.ScatterplotLayer({{
            id: 'top-spots',
            data: topSrc,
            getPosition: d => d.position,
            getRadius: d => d.radius * 30,
            getFillColor: d => [...d.color, 230],
            getLineColor: [0, 0, 0, 220],
            lineWidthMinPixels: 2,
            stroked: true,
            filled: true,
            radiusMinPixels: 6,
            radiusMaxPixels: 16,
            pickable: true,
            onHover: showTooltip
        }}));
        layers.push(new deck.TextLayer({{
            id: 'top-labels',
            data: topSrc.filter(d => d.is_top),
            getPosition: d => d.position,
            getText: d => '#' + d.rank + (d.species.length ? ' ' + d.species[0] : ''),
            getSize: 13,
            getColor: [255, 255, 255, 255],
            fontWeight: 'bold',
            outlineWidth: 3,
            outlineColor: [0, 0, 0, 200],
            getTextAnchor: 'start',
            getAlignmentBaseline: 'center',
            getPixelOffset: [14, 0]
        }}));
    }}

    // User location
    if (USER_LOCATION) {{
        layers.push(new deck.ScatterplotLayer({{
            id: 'user-location',
            data: [USER_LOCATION],
            getPosition: d => d.position,
            getRadius: d => d.radius,
            getFillColor: [33, 150, 243, 25],
            getLineColor: [33, 150, 243, 180],
            lineWidthMinPixels: 2,
            stroked: true,
            filled: true
        }}));
        layers.push(new deck.ScatterplotLayer({{
            id: 'user-dot',
            data: [USER_LOCATION],
            getPosition: d => d.position,
            getRadius: 200,
            getFillColor: [255, 0, 0, 220],
            radiusMinPixels: 8,
            radiusMaxPixels: 12
        }}));
    }}

    return layers;
}}

// === deck.gl overlay ===
let currentHourlyTop = null;
const deckOverlay = new deck.MapboxOverlay({{
    interleaved: true,
    layers: []
}});

map.addControl(deckOverlay);

function updateDeckLayers() {{
    deckOverlay.setProps({{ layers: buildLayers() }});
}}

// Initialize layers after map loads
map.on('load', function() {{
    updateDeckLayers();
}});

// === Global update functions for hourly panel ===
function updateSpotsForHour(hour) {{
    if (typeof hourlySpotsData === 'undefined') return;
    const spots = hourlySpotsData[hour.toString()];
    if (!spots || spots.length === 0) return;

    const sorted = spots.slice().sort((a, b) => b.score - a.score);
    const maxS = sorted[0].score || 1;
    const minS = sorted[sorted.length - 1].score || 0;
    const range = Math.max(maxS - minS, 1);

    // Update heatmap data
    currentHourlySpots = sorted.map(s => ({{
        position: [s.lon, s.lat],
        weight: Math.max(0.01, (s.score - minS) / range)
    }}));

    // Update top-5 markers
    currentHourlyTop = sorted.slice(0, 5).map((s, idx) => ({{
        position: [s.lon, s.lat],
        score: s.score,
        color: getScoreColor(s.score),
        radius: idx === 0 ? 14 : 10,
        rank: idx + 1,
        is_top: true,
        species: [],
        distance_to_fish: 0,
        lat: s.lat, lon: s.lon,
        _tooltip: '#' + (idx+1) + ' Score: ' + s.score.toFixed(0) + (s.tide_phase ? ' | ' + s.tide_phase : '')
    }}));

    updateDeckLayers();
}}

function updateMarkersForDay(date) {{
    if (typeof multidaySpotsData === 'undefined') return;
    const spots = multidaySpotsData[date];
    if (!spots || spots.length === 0) return;

    currentHourlyTop = spots.slice(0, 10).map((s, idx) => ({{
        position: [s.lon, s.lat],
        score: s.score,
        color: getScoreColor(s.score),
        radius: idx < 5 ? 12 : 8,
        rank: idx + 1,
        is_top: idx < 5,
        species: (s.species || []).map(sp => sp.name || sp),
        distance_to_fish: s.distance_to_fish || 0,
        lat: s.lat, lon: s.lon,
        _tooltip: '#' + (idx+1) + ' Score: ' + s.score.toFixed(0)
    }}));

    updateDeckLayers();
}}
</script>

<!--PANELS_PLACEHOLDER-->

</body>
</html>'''

    def _build_data_js(self) -> str:
        """Serialize all layer data as JavaScript constants."""
        # Coastline
        coastline = [
            {'path': [[lon, lat] for lat, lon in seg]}
            for seg in self._coastline_segments
        ]

        # Fish zones
        fish_zones = []
        for z in self._fish_zones:
            intensity = z.get('intensity', 0.5)
            colors = get_zone_colors(intensity)
            fish_zones.append({
                'position': [z['lon'], z['lat']],
                'radius': z.get('radius', 250) * 1.5,
                'borderColor': self._hex_to_rgb(colors['border']),
                'fillColor': self._hex_to_rgb(colors['fill']),
                '_tooltip': f"Zona #{z.get('id','?')} | Intensidad: {intensity:.0%} | SST: {z.get('sst','N/A')}C"
            })

        # Anchovy zones
        anchovy_zones = []
        for z in self._anchovy_zones:
            intensity = z.get('intensity', 0.5)
            colors = get_anchovy_colors(intensity)
            score = intensity * 100
            rating = "Alta" if score >= 70 else "Media" if score >= 40 else "Baja"
            anchovy_zones.append({
                'position': [z['lon'], z['lat']],
                'radius': z.get('radius', 400) * 1.8,
                'borderColor': self._hex_to_rgb(colors['border']),
                'fillColor': self._hex_to_rgb(colors['fill']),
                '_tooltip': f"🐟 Anchoveta | {rating} ({score:.0f}%) | SST: {z.get('sst','N/A')}C"
            })

        # Add tooltips to top spots and marine points
        for s in self._top_spots:
            species_str = ', '.join(s['species']) if s['species'] else ''
            s['_tooltip'] = (
                f"#{s['rank']} Score: {s['score']}/100 | "
                f"Dist. peces: {s['distance_to_fish']:.0f}m"
                + (f" | {species_str}" if species_str else '')
            )

        for m in self._marine_points:
            m['_tooltip'] = f"SST: {m['sst']}C | Olas: {m['wave']}m | Corriente: {m['speed']} m/s"

        user_loc = json.dumps(self._user_location) if self._user_location else 'null'

        return f'''<script>
const COASTLINE_DATA = {json.dumps(coastline)};
const FISH_ZONES = {json.dumps(fish_zones)};
const ANCHOVY_ZONES = {json.dumps(anchovy_zones)};
const FLOW_LINES = {json.dumps(self._flow_lines_data)};
const MARINE_POINTS = {json.dumps(self._marine_points)};
const SPOTS_DATA = {json.dumps(self._spots_data)};
const TOP_SPOTS = {json.dumps(self._top_spots)};
const HISTORICAL_HEATMAP = {json.dumps(self._heatmap_data)};
const USER_LOCATION = {user_loc};
</script>'''

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> List[int]:
        """Convert hex color to [r, g, b] list."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return [128, 128, 128]
        return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
