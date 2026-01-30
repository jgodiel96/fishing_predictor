"""
Map visualization using Folium.
Separates rendering logic from data processing.
Includes timeline controls, trend charts, and heatmap.
"""

import folium
import json
import numpy as np
from folium.plugins import HeatMap
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MapConfig:
    """Configuration for map rendering."""
    center: Tuple[float, float] = (-17.9, -71.0)
    zoom: int = 10
    tile_layer: str = "satellite"
    show_timeline: bool = True


class MapView:
    """
    View layer for map visualization.
    Renders data provided by Controller without processing logic.
    """

    # Color palettes (exclusive per layer)
    COLORS = {
        'spots': {
            'best': '#FF0000',
            'excellent': '#228B22',
            'good': '#32CD32',
            'regular': '#FFD700',
            'poor': '#DC143C'
        },
        'sst': {
            'cold': '#0000CD',
            'cool': '#1E90FF',
            'fresh': '#00BFFF',
            'optimal_low': '#00FA9A',
            'optimal': '#00FF00',
            'optimal_high': '#ADFF2F',
            'warm': '#FFD700',
            'hot': '#FF8C00',
            'very_hot': '#FF4500'
        },
        'flow': {
            'slow': '#DA70D6',
            'moderate': '#9932CC',
            'fast': '#8B008B',
            'very_fast': '#4B0082'
        },
        'zones': {
            'high': '#00CED1',
            'medium': '#20B2AA',
            'low': '#48D1CC',
            'fill_high': '#00FFFF',
            'fill_medium': '#40E0D0',
            'fill_low': '#7FFFD4'
        },
        'movement': '#FFD700',
        'coast': '#CCCC00'
    }

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self.map: Optional[folium.Map] = None

    def create_map(self, center: Tuple[float, float] = None, zoom: int = None) -> folium.Map:
        """Initialize the map."""
        center = center or self.config.center
        zoom = zoom or self.config.zoom

        self.map = folium.Map(location=list(center), zoom_start=zoom, tiles=None)

        # Add tile layers
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satelite"
        ).add_to(self.map)

        folium.TileLayer(tiles="OpenStreetMap", name="Calles").add_to(self.map)

        return self.map

    def add_coastline(self, points: List[Tuple[float, float]]):
        """Add coastline polyline."""
        if not self.map or not points:
            return

        folium.PolyLine(
            locations=points,
            color=self.COLORS['coast'],
            weight=3,
            opacity=0.9,
            popup=f"Costa OSM ({len(points)} puntos)"
        ).add_to(self.map)

    def add_fish_zones(self, zones: List[Dict]):
        """Add fish zone circles with movement arrows."""
        if not self.map or not zones:
            return

        # Separate regular zones from anchovy zones
        regular_zones = [z for z in zones if not z.get('is_anchovy')]
        anchovy_zones = [z for z in zones if z.get('is_anchovy')]

        # Regular fish zones
        fg = folium.FeatureGroup(name="Zonas de Peces")
        for zone in regular_zones:
            intensity = zone.get('intensity', 0.5)
            colors = self._get_zone_colors(intensity)

            folium.Circle(
                location=[zone['lat'], zone['lon']],
                radius=zone.get('radius', 250) * 1.5,
                color=colors['border'],
                weight=3,
                fill=True,
                fillColor=colors['fill'],
                fillOpacity=0.35,
                popup=self._zone_popup(zone)
            ).add_to(fg)

            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=8,
                color="#000",
                weight=2,
                fill=True,
                fillColor=colors['border'],
                fillOpacity=1.0
            ).add_to(fg)

            self._add_movement_arrow(fg, zone)

        fg.add_to(self.map)

        # Anchovy zones (distinct orange/red style)
        if anchovy_zones:
            fg_anchovy = folium.FeatureGroup(name="Zonas Anchoveta")
            for zone in anchovy_zones:
                intensity = zone.get('intensity', 0.5)

                # Orange-red gradient for anchovy
                if intensity >= 0.7:
                    border_color = '#FF4500'  # OrangeRed
                    fill_color = '#FF6347'    # Tomato
                elif intensity >= 0.4:
                    border_color = '#FF8C00'  # DarkOrange
                    fill_color = '#FFA500'    # Orange
                else:
                    border_color = '#FFD700'  # Gold
                    fill_color = '#FFEC8B'    # LightGoldenrod

                # Larger circle for anchovy schools
                folium.Circle(
                    location=[zone['lat'], zone['lon']],
                    radius=zone.get('radius', 400) * 1.8,
                    color=border_color,
                    weight=4,
                    fill=True,
                    fillColor=fill_color,
                    fillOpacity=0.4,
                    popup=self._anchovy_popup(zone)
                ).add_to(fg_anchovy)

                # Fish icon marker
                folium.Marker(
                    location=[zone['lat'], zone['lon']],
                    icon=folium.DivIcon(
                        html=f'''<div style="
                            font-size:20px;
                            text-shadow: 1px 1px 2px black;
                            transform: translate(-10px, -10px);
                        ">🐟</div>''',
                        icon_size=(20, 20)
                    ),
                    popup=self._anchovy_popup(zone)
                ).add_to(fg_anchovy)

            fg_anchovy.add_to(self.map)

    def add_flow_lines(self, flow_lines: List[List[Tuple[float, float]]], vectors: List = None):
        """Add current flow lines with direction arrows."""
        if not self.map or not flow_lines:
            return

        fg = folium.FeatureGroup(name="Flujo de Corrientes")

        for i, line in enumerate(flow_lines):
            if len(line) < 2:
                continue

            # Get speed for color
            speed = vectors[i].speed if vectors and i < len(vectors) else 0.1
            color = self._get_flow_color(speed)

            # Flow line
            folium.PolyLine(
                locations=line,
                color=color,
                weight=3,
                opacity=0.8
            ).add_to(fg)

            # Arrow at end
            self._add_flow_arrow(fg, line, color)

        fg.add_to(self.map)

    def add_marine_points(self, points: List):
        """Add SST sampling points as diamonds."""
        if not self.map or not points:
            return

        fg = folium.FeatureGroup(name="Datos Marinos (SST)")

        for point in points:
            sst = point.sst if point.sst is not None else 17.0
            color = self._get_sst_color(sst)

            popup = self._marine_popup(point, sst)

            folium.RegularPolygonMarker(
                location=[point.lat, point.lon],
                number_of_sides=4,
                radius=7,
                rotation=45,
                color="#000",
                weight=1,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                popup=folium.Popup(popup, max_width=220),
                tooltip=f"SST: {sst:.1f}C"
            ).add_to(fg)

        fg.add_to(self.map)

    def add_fishing_spots(self, spots: List[Dict], top_n: int = 5):
        """Add fishing spot markers with scores."""
        if not self.map or not spots:
            return

        fg = folium.FeatureGroup(name="Puntos de Pesca")

        for i, spot in enumerate(spots):
            is_best = (i == 0)
            is_top = (i < top_n)
            color = self._get_spot_color(spot['score'], is_best)

            popup = self._spot_popup(spot, i, is_best)

            folium.CircleMarker(
                location=[spot['lat'], spot['lon']],
                radius=12 if is_best else (9 if is_top else 6),
                color="#000" if is_top else "#333",
                weight=2 if is_top else 1,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=folium.Popup(popup, max_width=250),
                tooltip=self._spot_tooltip(spot, is_best)
            ).add_to(fg)

            # Top labels
            if is_top:
                self._add_spot_label(fg, spot, i)

        fg.add_to(self.map)

    def add_legend(self):
        """Add map legend."""
        if not self.map:
            return

        legend_html = self._build_legend_html()
        self.map.get_root().html.add_child(folium.Element(legend_html))

    def add_timeline(self, timeline_data: Dict):
        """Add timeline controls, charts, and forecast panel."""
        if not self.map or not timeline_data:
            return

        # Add heatmap layer
        self._add_heatmap_layer(timeline_data.get('heatmap', []))

        # Add timeline UI
        timeline_html = self._build_timeline_html(timeline_data)
        self.map.get_root().html.add_child(folium.Element(timeline_html))

    def _add_heatmap_layer(self, heatmap_data: List[Dict]):
        """Add historical fishing activity heatmap."""
        if not heatmap_data:
            return

        # Prepare data for heatmap: [[lat, lon, intensity], ...]
        heat_data = [
            [h['lat'], h['lon'], h['intensity']]
            for h in heatmap_data
            if h['intensity'] > 0
        ]

        if heat_data:
            fg = folium.FeatureGroup(name="Heatmap Historico", show=False)
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_zoom=13,
                radius=20,
                blur=15,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}
            ).add_to(fg)
            fg.add_to(self.map)

    def _build_timeline_html(self, data: Dict) -> str:
        """Build the complete timeline UI with charts and controls."""
        monthly = data.get('monthly_stats', [])
        forecast = data.get('weekly_forecast', [])
        today_stats = data.get('today_stats', {})
        date_range = data.get('date_range', {})
        yearly = data.get('yearly_trend', [])

        # Prepare chart data
        months_labels = json.dumps([m['month_name'][:3] for m in monthly])
        fishing_rates = json.dumps([round(m['fishing_rate'], 1) for m in monthly])
        sst_values = json.dumps([round(m['avg_sst'], 1) for m in monthly])

        forecast_labels = json.dumps([f['day_name'] for f in forecast])
        forecast_probs = json.dumps([round(f['fishing_probability'], 1) for f in forecast])
        forecast_sst = json.dumps([round(f['predicted_sst'], 1) for f in forecast])

        # Today's conditions
        today_sst = today_stats.get('avg_sst', 'N/A') if today_stats else 'N/A'
        today_wave = today_stats.get('avg_wave', 'N/A') if today_stats else 'N/A'
        today_rate = today_stats.get('fishing_rate', 0) if today_stats else 0

        return f'''
        <!-- Chart.js CDN -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <!-- Timeline Panel -->
        <div id="timeline-panel" style="
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.97);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            width: 320px;
            max-height: 90vh;
            overflow-y: auto;
        ">
            <!-- Header -->
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <h3 style="margin:0;color:#1a5f7a;">Linea de Tiempo</h3>
                <button onclick="toggleTimeline()" style="border:none;background:none;cursor:pointer;font-size:18px;">_</button>
            </div>

            <!-- Today's Conditions -->
            <div style="background:#e3f2fd;padding:10px;border-radius:8px;margin-bottom:12px;">
                <b style="color:#1565c0;">Hoy - {data.get('today', 'N/A')}</b>
                <div style="display:flex;justify-content:space-between;margin-top:8px;">
                    <span>SST: <b>{today_sst if isinstance(today_sst, str) else f"{today_sst:.1f}"}C</b></span>
                    <span>Olas: <b>{today_wave if isinstance(today_wave, str) else f"{today_wave:.1f}"}m</b></span>
                    <span>Pesca: <b>{today_rate:.1f}%</b></span>
                </div>
            </div>

            <!-- Date Selector -->
            <div style="margin-bottom:12px;">
                <label style="font-size:12px;color:#666;">Seleccionar fecha:</label>
                <input type="date" id="date-picker"
                    min="{date_range.get('min', '2020-01-01')}"
                    max="{date_range.get('max', '2026-01-31')}"
                    value="{data.get('today', '')}"
                    style="width:100%;padding:8px;border:1px solid #ddd;border-radius:5px;margin-top:4px;"
                    onchange="onDateChange(this.value)">
            </div>

            <!-- Tabs -->
            <div style="display:flex;margin-bottom:10px;border-bottom:2px solid #eee;">
                <button class="tab-btn active" onclick="showTab('monthly')" style="flex:1;padding:8px;border:none;background:none;cursor:pointer;font-weight:bold;">Mensual</button>
                <button class="tab-btn" onclick="showTab('forecast')" style="flex:1;padding:8px;border:none;background:none;cursor:pointer;">7 Dias</button>
                <button class="tab-btn" onclick="showTab('yearly')" style="flex:1;padding:8px;border:none;background:none;cursor:pointer;">Anual</button>
            </div>

            <!-- Monthly Chart -->
            <div id="tab-monthly" class="tab-content">
                <canvas id="monthlyChart" height="180"></canvas>
                <div style="margin-top:10px;font-size:11px;color:#666;">
                    <b>Mejor mes:</b> {self._get_best_month(monthly)}<br>
                    <b>Peor mes:</b> {self._get_worst_month(monthly)}
                </div>
            </div>

            <!-- Forecast Tab -->
            <div id="tab-forecast" class="tab-content" style="display:none;">
                <canvas id="forecastChart" height="180"></canvas>
                <div style="margin-top:10px;">
                    <b style="font-size:12px;">Pronostico 7 dias:</b>
                    <table style="width:100%;font-size:11px;margin-top:5px;">
                        <tr style="background:#f5f5f5;">
                            <th>Dia</th><th>Prob.</th><th>SST</th><th>Conf.</th>
                        </tr>
                        {self._build_forecast_rows(forecast)}
                    </table>
                </div>
            </div>

            <!-- Yearly Tab -->
            <div id="tab-yearly" class="tab-content" style="display:none;">
                <table style="width:100%;font-size:11px;">
                    <tr style="background:#f5f5f5;">
                        <th>Ano</th><th>Eventos</th><th>Horas</th><th>Tasa</th>
                    </tr>
                    {self._build_yearly_rows(yearly)}
                </table>
            </div>

            <!-- Heatmap Toggle -->
            <div style="margin-top:12px;padding-top:10px;border-top:1px solid #eee;">
                <label style="display:flex;align-items:center;cursor:pointer;">
                    <input type="checkbox" id="heatmap-toggle" onchange="toggleHeatmap(this.checked)" style="margin-right:8px;">
                    <span style="font-size:12px;">Mostrar heatmap historico</span>
                </label>
            </div>
        </div>

        <!-- Minimized Button -->
        <div id="timeline-btn" style="
            display: none;
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: #1a5f7a;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        " onclick="toggleTimeline()">
            Linea de Tiempo
        </div>

        <style>
            .tab-btn.active {{
                color: #1a5f7a;
                border-bottom: 2px solid #1a5f7a;
            }}
            #timeline-panel::-webkit-scrollbar {{
                width: 6px;
            }}
            #timeline-panel::-webkit-scrollbar-thumb {{
                background: #ccc;
                border-radius: 3px;
            }}
        </style>

        <script>
            // Chart data
            const monthsLabels = {months_labels};
            const fishingRates = {fishing_rates};
            const sstValues = {sst_values};
            const forecastLabels = {forecast_labels};
            const forecastProbs = {forecast_probs};
            const forecastSst = {forecast_sst};

            // Initialize charts when DOM is ready
            document.addEventListener('DOMContentLoaded', function() {{
                initCharts();
            }});

            // Also try immediate init (for Folium)
            setTimeout(initCharts, 500);

            function initCharts() {{
                // Monthly chart
                const monthlyCtx = document.getElementById('monthlyChart');
                if (monthlyCtx && !monthlyCtx.chart) {{
                    monthlyCtx.chart = new Chart(monthlyCtx, {{
                        type: 'bar',
                        data: {{
                            labels: monthsLabels,
                            datasets: [{{
                                label: 'Tasa Pesca (%)',
                                data: fishingRates,
                                backgroundColor: 'rgba(26, 95, 122, 0.7)',
                                borderColor: 'rgba(26, 95, 122, 1)',
                                borderWidth: 1,
                                yAxisID: 'y'
                            }}, {{
                                label: 'SST (C)',
                                data: sstValues,
                                type: 'line',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 2,
                                fill: false,
                                yAxisID: 'y1'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, position: 'left', title: {{ display: true, text: 'Pesca %' }} }},
                                y1: {{ beginAtZero: false, position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'SST' }} }}
                            }}
                        }}
                    }});
                }}

                // Forecast chart
                const forecastCtx = document.getElementById('forecastChart');
                if (forecastCtx && !forecastCtx.chart) {{
                    forecastCtx.chart = new Chart(forecastCtx, {{
                        type: 'line',
                        data: {{
                            labels: forecastLabels,
                            datasets: [{{
                                label: 'Prob. Pesca (%)',
                                data: forecastProbs,
                                borderColor: 'rgba(76, 175, 80, 1)',
                                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                                fill: true,
                                tension: 0.3
                            }}, {{
                                label: 'SST (C)',
                                data: forecastSst,
                                borderColor: 'rgba(255, 152, 0, 1)',
                                borderWidth: 2,
                                fill: false,
                                tension: 0.3
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }}
                            }}
                        }}
                    }});
                }}
            }}

            function showTab(tabName) {{
                document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById('tab-' + tabName).style.display = 'block';
                event.target.classList.add('active');
            }}

            function toggleTimeline() {{
                const panel = document.getElementById('timeline-panel');
                const btn = document.getElementById('timeline-btn');
                if (panel.style.display === 'none') {{
                    panel.style.display = 'block';
                    btn.style.display = 'none';
                }} else {{
                    panel.style.display = 'none';
                    btn.style.display = 'block';
                }}
            }}

            function onDateChange(date) {{
                console.log('Selected date:', date);
                // Could trigger AJAX to fetch data for this date
                alert('Datos para ' + date + '\\n\\nEsta funcion requiere recarga del mapa con los datos del dia seleccionado.');
            }}

            function toggleHeatmap(show) {{
                // Find heatmap layer and toggle
                const layers = document.querySelectorAll('.leaflet-overlay-pane');
                // This is a simplified toggle - actual implementation depends on layer structure
                console.log('Heatmap toggle:', show);
            }}
        </script>
        '''

    def _get_best_month(self, monthly: List[Dict]) -> str:
        if not monthly:
            return "N/A"
        best = max(monthly, key=lambda m: m.get('fishing_rate', 0))
        return f"{best['month_name']} ({best['fishing_rate']:.1f}%)"

    def _get_worst_month(self, monthly: List[Dict]) -> str:
        if not monthly:
            return "N/A"
        worst = min(monthly, key=lambda m: m.get('fishing_rate', 0))
        return f"{worst['month_name']} ({worst['fishing_rate']:.1f}%)"

    def _build_forecast_rows(self, forecast: List[Dict]) -> str:
        rows = []
        for f in forecast:
            color = '#4caf50' if f['fishing_probability'] > 2.5 else '#ff9800' if f['fishing_probability'] > 1.5 else '#f44336'
            rows.append(f'''
                <tr>
                    <td>{f['day_name']} {f['date'][5:]}</td>
                    <td style="color:{color};font-weight:bold;">{f['fishing_probability']:.1f}%</td>
                    <td>{f['predicted_sst']:.1f}C</td>
                    <td>{f['confidence']:.0f}%</td>
                </tr>
            ''')
        return ''.join(rows)

    def _build_yearly_rows(self, yearly: List[Dict]) -> str:
        rows = []
        for y in yearly:
            rows.append(f'''
                <tr>
                    <td><b>{y['year']}</b></td>
                    <td>{y['fishing_events']:,}</td>
                    <td>{y['total_hours']:,.0f}h</td>
                    <td>{y['fishing_rate']:.1f}%</td>
                </tr>
            ''')
        return ''.join(rows)

    def add_hourly_panel(self, hourly_data: Dict):
        """
        Add hourly predictions panel with tide chart and best hours.

        Args:
            hourly_data: Dict with keys:
                - date: str (YYYY-MM-DD)
                - location_name: str
                - predictions: List[Dict] (24 hourly predictions)
                - tide_extremes: List[Dict] (high/low tides)
                - best_hours: List[Dict] (top 5 hours)
        """
        if not self.map or not hourly_data:
            return

        predictions = hourly_data.get('predictions', [])
        tide_extremes = hourly_data.get('tide_extremes', [])
        best_hours = hourly_data.get('best_hours', [])
        date = hourly_data.get('date', 'N/A')
        location = hourly_data.get('location_name', 'Ubicacion')

        # Prepare data for charts
        hours_labels = json.dumps([f"{h:02d}:00" for h in range(24)])
        total_scores = json.dumps([p.get('total_score', 0) for p in predictions])
        tide_heights = json.dumps([p.get('tide_height', 0) for p in predictions])
        tide_scores = json.dumps([p.get('tide_score', 0) for p in predictions])
        hour_scores = json.dumps([p.get('hour_score', 0) for p in predictions])

        # Best hours table
        best_hours_rows = self._build_best_hours_rows(best_hours)
        tide_extremes_html = self._build_tide_extremes_html(tide_extremes)

        hourly_html = f'''
        <!-- Hourly Predictions Panel -->
        <div id="hourly-panel" style="
            position: fixed;
            bottom: 10px;
            right: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.97);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            width: 380px;
            max-height: 85vh;
            overflow-y: auto;
        ">
            <!-- Header -->
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div>
                    <h3 style="margin:0;color:#0d47a1;">Prediccion Horaria</h3>
                    <small style="color:#666;">{date} - {location}</small>
                </div>
                <button onclick="toggleHourlyPanel()" style="border:none;background:none;cursor:pointer;font-size:18px;">_</button>
            </div>

            <!-- Tide Info -->
            <div style="background:#e3f2fd;padding:10px;border-radius:8px;margin-bottom:12px;">
                <b style="color:#1565c0;">Mareas del Dia</b>
                <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;">
                    {tide_extremes_html}
                </div>
            </div>

            <!-- Hour Slider -->
            <div style="margin-bottom:12px;">
                <label style="font-size:12px;color:#666;">Seleccionar hora:</label>
                <div style="display:flex;align-items:center;gap:10px;">
                    <input type="range" id="hour-slider" min="0" max="23" value="6"
                        style="flex:1;" oninput="updateHourDisplay(this.value)">
                    <span id="hour-display" style="font-weight:bold;min-width:50px;">06:00</span>
                </div>
                <div id="hour-details" style="margin-top:8px;padding:10px;background:#f5f5f5;border-radius:5px;">
                    <div style="display:flex;justify-content:space-between;">
                        <span>Score Total:</span>
                        <b id="detail-score" style="color:#0d47a1;">--</b>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span>Marea:</span>
                        <span id="detail-tide">--</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span>Altura:</span>
                        <span id="detail-height">--</span>
                    </div>
                </div>
            </div>

            <!-- Charts Tabs -->
            <div style="display:flex;margin-bottom:10px;border-bottom:2px solid #eee;">
                <button class="hourly-tab-btn active" onclick="showHourlyTab('scores')" style="flex:1;padding:8px;border:none;background:none;cursor:pointer;font-weight:bold;">Scores</button>
                <button class="hourly-tab-btn" onclick="showHourlyTab('tides')" style="flex:1;padding:8px;border:none;background:none;cursor:pointer;">Mareas</button>
                <button class="hourly-tab-btn" onclick="showHourlyTab('best')" style="flex:1;padding:8px;border:none;background:none;cursor:pointer;">Mejores</button>
            </div>

            <!-- Scores Chart -->
            <div id="hourly-tab-scores" class="hourly-tab-content">
                <canvas id="hourlyScoresChart" height="160"></canvas>
            </div>

            <!-- Tides Chart -->
            <div id="hourly-tab-tides" class="hourly-tab-content" style="display:none;">
                <canvas id="hourlyTidesChart" height="160"></canvas>
            </div>

            <!-- Best Hours Tab -->
            <div id="hourly-tab-best" class="hourly-tab-content" style="display:none;">
                <table style="width:100%;font-size:12px;border-collapse:collapse;">
                    <tr style="background:#e3f2fd;">
                        <th style="padding:8px;text-align:left;">Hora</th>
                        <th style="padding:8px;">Score</th>
                        <th style="padding:8px;">Marea</th>
                        <th style="padding:8px;">Razon</th>
                    </tr>
                    {best_hours_rows}
                </table>
            </div>

            <!-- Legend -->
            <div style="margin-top:12px;padding-top:10px;border-top:1px solid #eee;font-size:11px;color:#666;">
                <b>Pesos:</b> Marea 35% | Hora 25% | SST 20% | Zona 20%
            </div>
        </div>

        <!-- Minimized Button -->
        <div id="hourly-btn" style="
            display: none;
            position: fixed;
            bottom: 10px;
            right: 10px;
            z-index: 1000;
            background: #0d47a1;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        " onclick="toggleHourlyPanel()">
            Prediccion Horaria
        </div>

        <style>
            .hourly-tab-btn.active {{
                color: #0d47a1;
                border-bottom: 2px solid #0d47a1;
            }}
            #hourly-panel::-webkit-scrollbar {{
                width: 6px;
            }}
            #hourly-panel::-webkit-scrollbar-thumb {{
                background: #ccc;
                border-radius: 3px;
            }}
        </style>

        <script>
            // Hourly prediction data
            const hourlyPredictions = {json.dumps(predictions)};
            const hoursLabels = {hours_labels};
            const totalScores = {total_scores};
            const tideHeights = {tide_heights};
            const tideScores = {tide_scores};
            const hourScores = {hour_scores};

            // Initialize hourly charts
            setTimeout(initHourlyCharts, 600);

            function initHourlyCharts() {{
                // Scores chart
                const scoresCtx = document.getElementById('hourlyScoresChart');
                if (scoresCtx && !scoresCtx.chart) {{
                    scoresCtx.chart = new Chart(scoresCtx, {{
                        type: 'bar',
                        data: {{
                            labels: hoursLabels,
                            datasets: [{{
                                label: 'Score Total',
                                data: totalScores,
                                backgroundColor: totalScores.map(s =>
                                    s >= 75 ? 'rgba(76, 175, 80, 0.7)' :
                                    s >= 60 ? 'rgba(255, 193, 7, 0.7)' :
                                    'rgba(244, 67, 54, 0.5)'
                                ),
                                borderColor: totalScores.map(s =>
                                    s >= 75 ? 'rgba(76, 175, 80, 1)' :
                                    s >= 60 ? 'rgba(255, 193, 7, 1)' :
                                    'rgba(244, 67, 54, 1)'
                                ),
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{ display: false }},
                                tooltip: {{
                                    callbacks: {{
                                        afterBody: function(context) {{
                                            const i = context[0].dataIndex;
                                            const p = hourlyPredictions[i];
                                            return [
                                                'Marea: ' + (p.tide_phase_es || p.tide_phase),
                                                'Altura: ' + p.tide_height + 'm'
                                            ];
                                        }}
                                    }}
                                }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Score' }} }}
                            }},
                            onClick: function(e, elements) {{
                                if (elements.length > 0) {{
                                    const hour = elements[0].index;
                                    document.getElementById('hour-slider').value = hour;
                                    updateHourDisplay(hour);
                                }}
                            }}
                        }}
                    }});
                }}

                // Tides chart
                const tidesCtx = document.getElementById('hourlyTidesChart');
                if (tidesCtx && !tidesCtx.chart) {{
                    tidesCtx.chart = new Chart(tidesCtx, {{
                        type: 'line',
                        data: {{
                            labels: hoursLabels,
                            datasets: [{{
                                label: 'Altura Marea (m)',
                                data: tideHeights,
                                borderColor: 'rgba(33, 150, 243, 1)',
                                backgroundColor: 'rgba(33, 150, 243, 0.2)',
                                fill: true,
                                tension: 0.4,
                                yAxisID: 'y'
                            }}, {{
                                label: 'Score Marea',
                                data: tideScores,
                                borderColor: 'rgba(76, 175, 80, 1)',
                                borderWidth: 2,
                                fill: false,
                                tension: 0.3,
                                yAxisID: 'y1'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }}
                            }},
                            scales: {{
                                y: {{ position: 'left', title: {{ display: true, text: 'Altura (m)' }} }},
                                y1: {{ position: 'right', min: 0, max: 100, grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Score' }} }}
                            }}
                        }}
                    }});
                }}

                // Initialize display
                updateHourDisplay(6);
            }}

            function updateHourDisplay(hour) {{
                hour = parseInt(hour);
                document.getElementById('hour-display').textContent = String(hour).padStart(2, '0') + ':00';

                const p = hourlyPredictions[hour];
                if (p) {{
                    document.getElementById('detail-score').textContent = p.total_score.toFixed(0) + '/100';
                    document.getElementById('detail-tide').textContent = p.tide_phase_es || p.tide_phase;
                    document.getElementById('detail-height').textContent = p.tide_height.toFixed(2) + 'm';

                    // Update score color
                    const scoreEl = document.getElementById('detail-score');
                    if (p.total_score >= 75) scoreEl.style.color = '#4caf50';
                    else if (p.total_score >= 60) scoreEl.style.color = '#ff9800';
                    else scoreEl.style.color = '#f44336';
                }}
            }}

            function showHourlyTab(tabName) {{
                document.querySelectorAll('.hourly-tab-content').forEach(el => el.style.display = 'none');
                document.querySelectorAll('.hourly-tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById('hourly-tab-' + tabName).style.display = 'block';
                event.target.classList.add('active');
            }}

            function toggleHourlyPanel() {{
                const panel = document.getElementById('hourly-panel');
                const btn = document.getElementById('hourly-btn');
                if (panel.style.display === 'none') {{
                    panel.style.display = 'block';
                    btn.style.display = 'none';
                }} else {{
                    panel.style.display = 'none';
                    btn.style.display = 'block';
                }}
            }}
        </script>
        '''

        self.map.get_root().html.add_child(folium.Element(hourly_html))

    def _build_best_hours_rows(self, best_hours: List[Dict]) -> str:
        """Build HTML rows for best fishing hours table."""
        rows = []
        for i, h in enumerate(best_hours[:5]):
            score = h.get('total_score', 0)
            color = '#4caf50' if score >= 75 else '#ff9800' if score >= 60 else '#f44336'
            medal = ['🥇', '🥈', '🥉', '4.', '5.'][i]

            rows.append(f'''
                <tr style="border-bottom:1px solid #eee;">
                    <td style="padding:8px;">{medal} {h.get('time', '--')}</td>
                    <td style="padding:8px;text-align:center;color:{color};font-weight:bold;">{score:.0f}</td>
                    <td style="padding:8px;text-align:center;font-size:11px;">{h.get('tide_phase_es', h.get('tide_phase', '--'))}</td>
                    <td style="padding:8px;font-size:10px;color:#666;">{self._get_hour_reason(h)}</td>
                </tr>
            ''')
        return ''.join(rows)

    def _build_tide_extremes_html(self, extremes: List[Dict]) -> str:
        """Build HTML for tide extremes display."""
        html_parts = []
        for e in extremes:
            icon = '🔺' if e.get('type') == 'high' else '🔻'
            label = e.get('type_es', 'Pleamar' if e.get('type') == 'high' else 'Bajamar')
            html_parts.append(f'''
                <div style="background:white;padding:5px 10px;border-radius:5px;font-size:12px;">
                    {icon} <b>{e.get('time', '--')}</b> {label} ({e.get('height', 0):.2f}m)
                </div>
            ''')
        return ''.join(html_parts)

    def _get_hour_reason(self, hour_data: Dict) -> str:
        """Generate reason text for why this hour is good."""
        reasons = []

        tide_score = hour_data.get('tide_score', 0)
        hour_score = hour_data.get('hour_score', 0)

        if tide_score >= 80:
            reasons.append('Marea optima')
        if hour_score >= 90:
            reasons.append('Alba/Ocaso')
        elif hour_score >= 70:
            reasons.append('Buena hora')

        if not reasons:
            reasons.append('Condiciones moderadas')

        return ' + '.join(reasons)

    def finalize(self) -> folium.Map:
        """Add layer control and return map."""
        if self.map:
            folium.LayerControl().add_to(self.map)
        return self.map

    def save(self, filepath: str):
        """Save map to HTML file."""
        if self.map:
            self.map.save(filepath)

    # === Private helper methods ===

    def _get_zone_colors(self, intensity: float) -> Dict[str, str]:
        if intensity >= 0.8:
            return {'border': self.COLORS['zones']['high'], 'fill': self.COLORS['zones']['fill_high']}
        elif intensity >= 0.6:
            return {'border': self.COLORS['zones']['medium'], 'fill': self.COLORS['zones']['fill_medium']}
        return {'border': self.COLORS['zones']['low'], 'fill': self.COLORS['zones']['fill_low']}

    def _get_flow_color(self, speed: float) -> str:
        if speed > 0.3:
            return self.COLORS['flow']['very_fast']
        elif speed > 0.2:
            return self.COLORS['flow']['fast']
        elif speed > 0.1:
            return self.COLORS['flow']['moderate']
        return self.COLORS['flow']['slow']

    def _get_sst_color(self, sst: float) -> str:
        if sst <= 14:
            return self.COLORS['sst']['cold']
        elif sst <= 16:
            return self.COLORS['sst']['cool']
        elif sst <= 17:
            return self.COLORS['sst']['fresh']
        elif sst <= 18:
            return self.COLORS['sst']['optimal_low']
        elif sst <= 19:
            return self.COLORS['sst']['optimal']
        elif sst <= 20:
            return self.COLORS['sst']['optimal_high']
        elif sst <= 21:
            return self.COLORS['sst']['warm']
        elif sst <= 22:
            return self.COLORS['sst']['hot']
        return self.COLORS['sst']['very_hot']

    def _get_spot_color(self, score: float, is_best: bool) -> str:
        if is_best:
            return self.COLORS['spots']['best']
        elif score >= 80:
            return self.COLORS['spots']['excellent']
        elif score >= 60:
            return self.COLORS['spots']['good']
        elif score >= 40:
            return self.COLORS['spots']['regular']
        return self.COLORS['spots']['poor']

    def _add_movement_arrow(self, fg: folium.FeatureGroup, zone: Dict):
        direction = zone.get('movement_direction', 90)
        rad = np.radians(direction)
        arrow_len = 250

        end_lat = zone['lat'] + arrow_len / 111000 * np.cos(rad)
        end_lon = zone['lon'] + arrow_len / (111000 * np.cos(np.radians(zone['lat']))) * np.sin(rad)

        folium.PolyLine(
            locations=[[zone['lat'], zone['lon']], [end_lat, end_lon]],
            color=self.COLORS['movement'],
            weight=4,
            opacity=0.9
        ).add_to(fg)

        folium.RegularPolygonMarker(
            location=[end_lat, end_lon],
            number_of_sides=3,
            radius=8,
            rotation=direction - 90,
            color=self.COLORS['movement'],
            fill=True,
            fillColor=self.COLORS['movement'],
            fillOpacity=1.0
        ).add_to(fg)

    def _add_flow_arrow(self, fg: folium.FeatureGroup, line: List, color: str):
        if len(line) < 2:
            return

        end_lat, end_lon = line[-1]
        prev_lat, prev_lon = line[-2]

        angle = np.degrees(np.arctan2(end_lon - prev_lon, end_lat - prev_lat))

        folium.RegularPolygonMarker(
            location=[end_lat, end_lon],
            number_of_sides=3,
            radius=6,
            rotation=angle - 90,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.9
        ).add_to(fg)

    def _add_spot_label(self, fg: folium.FeatureGroup, spot: Dict, index: int):
        folium.Marker(
            location=[spot['lat'], spot['lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:14px;font-weight:bold;color:white;text-shadow:2px 2px 4px black;">#{index+1}</div>',
                icon_size=(30, 30),
                icon_anchor=(15, 15)
            )
        ).add_to(fg)

    def _zone_popup(self, zone: Dict) -> str:
        return f"""
        <div style="font-family:Arial;">
        <b>Zona de Peces #{zone.get('id', '?')}</b><br>
        <hr style="margin:4px 0;">
        Intensidad: <b>{zone.get('intensity', 0):.0%}</b><br>
        Causa: {zone.get('cause', 'N/A')}<br>
        SST: {zone.get('sst', 'N/A')}C
        </div>
        """

    def _anchovy_popup(self, zone: Dict) -> str:
        """Popup for anchovy concentration zones."""
        score = zone.get('intensity', 0) * 100
        rating = "Alta" if score >= 70 else "Media" if score >= 40 else "Baja"

        return f"""
        <div style="font-family:Arial;min-width:200px;">
        <h4 style="margin:0;color:#FF4500;">🐟 Zona Anchoveta</h4>
        <hr style="margin:5px 0;border-color:#FFD700;">
        <b>Probabilidad:</b> {rating} ({score:.0f}%)<br>
        <b>SST:</b> {zone.get('sst', 'N/A')}°C<br>
        <b>Historico:</b> {zone.get('historical_hours', 0):.0f}h pesca<br>
        <hr style="margin:5px 0;border-color:#eee;">
        <small style="color:#666;">
        Basado en patrones de migracion<br>
        y datos historicos GFW/IMARPE
        </small><br>
        <small>{zone.get('lat', 0):.4f}, {zone.get('lon', 0):.4f}</small>
        </div>
        """

    def _marine_popup(self, point, sst: float) -> str:
        wave = point.wave_height if point.wave_height else 0
        spd = point.current_speed if point.current_speed else 0
        dir_ = point.current_direction if point.current_direction else 0

        return f"""
        <div style="font-family:Arial;">
        <b>Datos Marinos</b><br>
        SST: <b>{sst:.1f}C</b><br>
        Olas: {wave:.1f}m<br>
        Corriente: {spd:.2f} m/s<br>
        Dir: {dir_:.0f}
        </div>
        """

    def _spot_popup(self, spot: Dict, index: int, is_best: bool) -> str:
        rating = self._get_rating(spot['score'])
        title = "MEJOR SPOT" if is_best else f"Spot #{spot.get('id', index+1)}"

        species_html = ""
        if spot.get('species') and index < 5:
            species_html = "<br><b>Especies:</b><br>"
            for sp in spot['species']:
                species_html += f"- {sp['name']}<br>"

        return f"""
        <div style="font-family:Arial;min-width:180px;">
        <h4 style="margin:0;color:{'#FF0000' if is_best else '#1a5f7a'};">{title}</h4>
        <hr style="margin:5px 0;">
        <b>Score:</b> {spot['score']:.1f}/100 ({rating})<br>
        <b>Dist. peces:</b> {spot.get('distance_to_fish', 0):.0f}m<br>
        {species_html}
        <small>{spot['lat']:.5f}, {spot['lon']:.5f}</small>
        </div>
        """

    def _spot_tooltip(self, spot: Dict, is_best: bool) -> str:
        prefix = "* " if is_best else ""
        return f"{prefix}Score: {spot['score']:.0f} - {self._get_rating(spot['score'])}"

    def _get_rating(self, score: float) -> str:
        if score >= 80:
            return "Excelente"
        elif score >= 60:
            return "Bueno"
        elif score >= 40:
            return "Regular"
        return "Bajo"

    def _build_legend_html(self) -> str:
        return '''
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:rgba(255,255,255,0.95);padding:12px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.4);font-size:10px;
                    font-family:Arial;max-height:90vh;overflow-y:auto;width:170px;">
            <b style="font-size:13px;">Predictor de Pesca</b><br>
            <small style="color:#666;">ML + Migracion Anchoveta</small>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>Spots:</b><br>
            <span style="color:#FF0000;">●</span> Mejor<br>
            <span style="color:#228B22;">●</span> Excelente (80+)<br>
            <span style="color:#32CD32;">●</span> Bueno (60-80)<br>
            <span style="color:#FFD700;">●</span> Regular (40-60)<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b style="color:#FF4500;">Anchoveta:</b><br>
            <span style="color:#FF4500;">◯</span> Alta probabilidad<br>
            <span style="color:#FF8C00;">◯</span> Media probabilidad<br>
            <span style="color:#FFD700;">◯</span> Baja probabilidad<br>
            🐟 Centro cardumen<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>SST:</b><br>
            <span style="color:#1E90FF;">◆</span> Frio<br>
            <span style="color:#00FF00;">◆</span> Optimo<br>
            <span style="color:#FF4500;">◆</span> Caliente<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>Corrientes:</b><br>
            <span style="color:#DA70D6;">→</span> Lento<br>
            <span style="color:#4B0082;">→</span> Rapido<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <span style="color:#00CED1;">◯</span> Zona peces<br>
            <span style="color:#FFD700;">→</span> Movimiento
        </div>
        '''
