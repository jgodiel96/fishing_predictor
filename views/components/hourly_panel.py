"""
Hourly predictions panel component.
Shows tide chart, hour slider, and best hours.
"""

import json
import folium
from typing import Dict, List

from views.styles.map_styles import COLORS


class HourlyPanel:
    """
    Hourly predictions panel with tide charts and hour selection.
    """

    def __init__(self, map_obj: folium.Map):
        self.map = map_obj

    def render(self, hourly_data: Dict):
        """Add hourly predictions panel to map."""
        if not self.map or not hourly_data:
            return

        html = self._build_html(hourly_data)
        self.map.get_root().html.add_child(folium.Element(html))

    def render_multiday(self, multiday_data: Dict):
        """Embed multi-day hourly predictions for dynamic date selection."""
        if not self.map or not multiday_data:
            return

        html = self._build_multiday_html(multiday_data)
        self.map.get_root().html.add_child(folium.Element(html))

    def render_hourly_spots(self, hourly_spots_data: Dict[int, List[Dict]]):
        """Embed 24-hour scoring data for dynamic marker updates."""
        if not self.map or not hourly_spots_data:
            return

        html = self._build_hourly_spots_html(hourly_spots_data)
        self.map.get_root().html.add_child(folium.Element(html))

    def _build_html(self, hourly_data: Dict) -> str:
        """Build hourly panel HTML."""
        predictions = hourly_data.get('predictions', [])
        tide_extremes = hourly_data.get('tide_extremes', [])
        best_hours = hourly_data.get('best_hours', [])
        date = hourly_data.get('date', 'N/A')
        location = hourly_data.get('location_name', 'Ubicacion')

        hours_labels = json.dumps([f"{h:02d}:00" for h in range(24)])
        total_scores = json.dumps([p.get('total_score', 0) for p in predictions])
        tide_heights = json.dumps([p.get('tide_height', 0) for p in predictions])
        tide_scores = json.dumps([p.get('tide_score', 0) for p in predictions])
        hour_scores = json.dumps([p.get('hour_score', 0) for p in predictions])

        best_hours_rows = self._build_best_hours_rows(best_hours)
        tide_extremes_html = self._build_tide_extremes_html(tide_extremes)

        return f'''
        <div id="hourly-panel" class="fishing-panel" style="
            position: fixed;
            bottom: 10px;
            right: 10px;
            width: 380px;
        ">
            <div class="panel-header">
                <div>
                    <h3 style="color:{COLORS['ui']['primary']};">Prediccion Horaria</h3>
                    <small style="color:#666;">{date} - {location}</small>
                </div>
                <button onclick="toggleHourlyPanel()" class="panel-minimize-btn">_</button>
            </div>

            <div class="info-box">
                <b>Mareas del Dia</b>
                <div class="tide-extremes" style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;">
                    {tide_extremes_html}
                </div>
            </div>

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
                        <b id="detail-score" style="color:{COLORS['ui']['primary']};">--</b>
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

            <div class="tab-container">
                <button class="hourly-tab-btn active" onclick="showHourlyTab('scores')">Scores</button>
                <button class="hourly-tab-btn" onclick="showHourlyTab('tides')">Mareas</button>
                <button class="hourly-tab-btn" onclick="showHourlyTab('best')">Mejores</button>
            </div>

            <div id="hourly-tab-scores" class="hourly-tab-content">
                <canvas id="hourlyScoresChart" height="160"></canvas>
            </div>

            <div id="hourly-tab-tides" class="hourly-tab-content" style="display:none;">
                <canvas id="hourlyTidesChart" height="160"></canvas>
            </div>

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

            <div style="margin-top:12px;padding-top:10px;border-top:1px solid #eee;font-size:11px;color:#666;">
                <b>Pesos:</b> Marea 35% | Hora 25% | SST 20% | Zona 20%
            </div>
        </div>

        <div id="hourly-btn" class="minimized-btn" style="
            bottom: 10px;
            right: 10px;
            background: {COLORS['ui']['primary']};
        " onclick="toggleHourlyPanel()">
            Prediccion Horaria
        </div>

        <style>
            .hourly-tab-btn.active {{
                color: {COLORS['ui']['primary']};
                border-bottom: 2px solid {COLORS['ui']['primary']};
            }}
        </style>

        <script>
            const hourlyPredictions = {json.dumps(predictions)};
            const hoursLabels = {hours_labels};
            const totalScores = {total_scores};
            const tideHeights = {tide_heights};
            const tideScores = {tide_scores};
            const hourScores = {hour_scores};

            setTimeout(initHourlyCharts, 600);

            function initHourlyCharts() {{
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
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{ legend: {{ display: false }} }},
                            scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
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
                                yAxisID: 'y1'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }} }},
                            scales: {{
                                y: {{ position: 'left', title: {{ display: true, text: 'Altura (m)' }} }},
                                y1: {{ position: 'right', min: 0, max: 100, grid: {{ drawOnChartArea: false }} }}
                            }}
                        }}
                    }});
                }}

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

    def _build_multiday_html(self, multiday_data: Dict) -> str:
        """Build multiday selector HTML."""
        days_data = multiday_data.get('days', {})
        location = multiday_data.get('location', {})

        js_data = {}
        for date_str, day_data in days_data.items():
            js_data[date_str] = {
                'day_name': day_data.get('day_name', ''),
                'location_name': location.get('name', 'Ubicacion'),
                'predictions': day_data.get('predictions', []),
                'tide_extremes': day_data.get('tide_extremes', []),
                'best_hours': day_data.get('best_hours', []),
                'avg_score': day_data.get('avg_score', 0)
            }

        day_buttons_html = []
        for i, (date_str, day_data) in enumerate(days_data.items()):
            is_first = i == 0
            score = day_data.get('avg_score', 0)
            score_color = '#4caf50' if score >= 60 else '#ff9800' if score >= 45 else '#f44336'
            day_buttons_html.append(f'''
                <button class="multiday-btn {'active' if is_first else ''}" data-date="{date_str}"
                    onclick="selectMultiday('{date_str}')"
                    style="padding:6px 10px;border:1px solid #ddd;border-radius:5px;cursor:pointer;
                           background:{('#0d47a1' if is_first else 'white')};
                           color:{('white' if is_first else '#333')};font-size:11px;text-align:center;">
                    <div style="font-weight:bold;">{day_data.get('day_name', '')}</div>
                    <div style="font-size:10px;">{date_str[5:]}</div>
                    <div style="color:{score_color};font-weight:bold;">{score:.0f}</div>
                </button>
            ''')

        return f'''
        <script>
            const multidayHourlyData = {json.dumps(js_data, default=str)};
            const multidayDates = {json.dumps(list(days_data.keys()))};
            const multidayLocation = {json.dumps(location)};

            function selectMultiday(date) {{
                const picker = document.getElementById('date-picker');
                if (picker) picker.value = date;

                document.querySelectorAll('.multiday-btn').forEach(btn => {{
                    if (btn.dataset.date === date) {{
                        btn.classList.add('active');
                        btn.style.background = '#0d47a1';
                        btn.style.color = 'white';
                    }} else {{
                        btn.classList.remove('active');
                        btn.style.background = 'white';
                        btn.style.color = '#333';
                    }}
                }});

                updateHourlyPanelForDate(date);
            }}

            function updateHourlyPanelForDate(date) {{
                if (typeof multidayHourlyData === 'undefined') return;
                const dayData = multidayHourlyData[date];
                if (!dayData) return;

                if (typeof hourlyPredictions !== 'undefined') {{
                    hourlyPredictions.length = 0;
                    dayData.predictions.forEach(p => hourlyPredictions.push(p));
                }}

                if (typeof totalScores !== 'undefined') {{
                    totalScores.length = 0;
                    dayData.predictions.forEach(p => totalScores.push(p.total_score || 0));
                }}

                refreshHourlyCharts();
                const slider = document.getElementById('hour-slider');
                if (slider) {{
                    slider.value = 6;
                    updateHourDisplay(6);
                }}
            }}

            function refreshHourlyCharts() {{
                const scoresCtx = document.getElementById('hourlyScoresChart');
                if (scoresCtx && scoresCtx.chart) {{
                    scoresCtx.chart.data.datasets[0].data = totalScores;
                    scoresCtx.chart.update('none');
                }}
            }}
        </script>

        <div id="multiday-bar" style="
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1001;
            background: rgba(255,255,255,0.97);
            padding: 8px 12px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            display: flex;
            gap: 6px;
            align-items: center;
        ">
            <span style="font-weight:bold;color:#0d47a1;font-size:12px;margin-right:5px;">7 Dias:</span>
            {''.join(day_buttons_html)}
        </div>

        <style>
            .multiday-btn:hover {{ transform: scale(1.05); transition: transform 0.15s; }}
            .multiday-btn.active {{ background: #0d47a1 !important; color: white !important; }}
        </style>
        '''

    def _build_hourly_spots_html(self, hourly_spots_data: Dict[int, List[Dict]]) -> str:
        """Build hourly spots JavaScript for dynamic updates."""
        js_data = {}
        for hour, spots in hourly_spots_data.items():
            js_data[str(hour)] = [
                {
                    'lat': s['lat'],
                    'lon': s['lon'],
                    'score': round(s['score'], 1),
                    'tide_phase': s.get('tide_phase', 'unknown'),
                    'tide_score': round(s.get('tide_score', 0.5) * 100, 1),
                    'hour_score': round(s.get('hour_score', 0.5) * 100, 1),
                }
                for s in spots
            ]

        return f'''
        <script>
            const hourlySpotsData = {json.dumps(js_data)};
            let currentDisplayHour = 6;

            function getScoreColor(score) {{
                if (score >= 80) return '#228B22';
                if (score >= 60) return '#32CD32';
                if (score >= 40) return '#FFD700';
                return '#DC143C';
            }}

            function updateSpotsForHour(hour) {{
                currentDisplayHour = hour;
                const spots = hourlySpotsData[hour.toString()];
                if (!spots) return;

                const hourDisplay = document.getElementById('unified-hour-display');
                if (hourDisplay) {{
                    hourDisplay.textContent = String(hour).padStart(2, '0') + ':00';
                }}

                console.log('Updated spots for hour:', hour, 'Count:', spots.length);
            }}
        </script>
        '''

    def _build_best_hours_rows(self, best_hours: List[Dict]) -> str:
        rows = []
        medals = ['First', 'Second', 'Third', '4.', '5.']
        for i, h in enumerate(best_hours[:5]):
            score = h.get('total_score', 0)
            color = '#4caf50' if score >= 75 else '#ff9800' if score >= 60 else '#f44336'
            rows.append(f'''
                <tr style="border-bottom:1px solid #eee;">
                    <td style="padding:8px;">{medals[i]} {h.get('time', '--')}</td>
                    <td style="padding:8px;text-align:center;color:{color};font-weight:bold;">{score:.0f}</td>
                    <td style="padding:8px;text-align:center;font-size:11px;">{h.get('tide_phase_es', h.get('tide_phase', '--'))}</td>
                    <td style="padding:8px;font-size:10px;color:#666;">Optimo</td>
                </tr>
            ''')
        return ''.join(rows)

    def _build_tide_extremes_html(self, tide_extremes: List[Dict]) -> str:
        html_parts = []
        for e in tide_extremes:
            icon = 'Up' if e.get('type') == 'high' else 'Down'
            label = e.get('type_es', 'Pleamar' if e.get('type') == 'high' else 'Bajamar')
            html_parts.append(f'''
                <div style="background:white;padding:5px 10px;border-radius:5px;font-size:12px;">
                    {icon} <b>{e.get('time', '--')}</b> {label} ({e.get('height', 0):.2f}m)
                </div>
            ''')
        return ''.join(html_parts)
