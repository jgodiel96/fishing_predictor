"""
Timeline panel component.
Shows monthly stats, weekly forecast, and yearly trends.
"""

import json
import folium
from typing import Dict, List

from views.styles.map_styles import COLORS, get_chart_js_cdn


class TimelinePanel:
    """
    Timeline panel with charts and date selection.
    """

    def __init__(self, map_obj: folium.Map):
        self.map = map_obj

    def render(self, timeline_data: Dict):
        """Add timeline controls, charts, and forecast panel to map."""
        if not self.map or not timeline_data:
            return

        html = self._build_html(timeline_data)
        self.map.get_root().html.add_child(folium.Element(html))

    def _build_html(self, data: Dict) -> str:
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
        {get_chart_js_cdn()}

        <div id="timeline-panel" class="fishing-panel" style="
            position: fixed;
            top: 10px;
            right: 10px;
            width: 320px;
        ">
            <div class="panel-header">
                <h3 style="color:{COLORS['ui']['secondary']};">Linea de Tiempo</h3>
                <button onclick="toggleTimeline()" class="panel-minimize-btn">_</button>
            </div>

            <div class="info-box">
                <b>Hoy - {data.get('today', 'N/A')}</b>
                <div style="display:flex;justify-content:space-between;margin-top:8px;">
                    <span>SST: <b>{today_sst if isinstance(today_sst, str) else f"{today_sst:.1f}"}C</b></span>
                    <span>Olas: <b>{today_wave if isinstance(today_wave, str) else f"{today_wave:.1f}"}m</b></span>
                    <span>Pesca: <b>{today_rate:.1f}%</b></span>
                </div>
            </div>

            <div style="margin-bottom:12px;">
                <label style="font-size:12px;color:#666;">Seleccionar fecha:</label>
                <input type="date" id="date-picker"
                    min="{date_range.get('min', '2020-01-01')}"
                    max="{date_range.get('max', '2026-01-31')}"
                    value="{data.get('today', '')}"
                    style="width:100%;padding:8px;border:1px solid #ddd;border-radius:5px;margin-top:4px;"
                    onchange="onDateChange(this.value)">
            </div>

            <div class="tab-container">
                <button class="tab-btn active" onclick="showTab('monthly')">Mensual</button>
                <button class="tab-btn" onclick="showTab('forecast')">7 Dias</button>
                <button class="tab-btn" onclick="showTab('yearly')">Anual</button>
            </div>

            <div id="tab-monthly" class="tab-content">
                <canvas id="monthlyChart" height="180"></canvas>
                <div style="margin-top:10px;font-size:11px;color:#666;">
                    <b>Mejor mes:</b> {self._get_best_month(monthly)}<br>
                    <b>Peor mes:</b> {self._get_worst_month(monthly)}
                </div>
            </div>

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

            <div id="tab-yearly" class="tab-content" style="display:none;">
                <table style="width:100%;font-size:11px;">
                    <tr style="background:#f5f5f5;">
                        <th>Ano</th><th>Eventos</th><th>Horas</th><th>Tasa</th>
                    </tr>
                    {self._build_yearly_rows(yearly)}
                </table>
            </div>

            <div style="margin-top:12px;padding-top:10px;border-top:1px solid #eee;">
                <label style="display:flex;align-items:center;cursor:pointer;">
                    <input type="checkbox" id="heatmap-toggle" onchange="toggleHeatmap(this.checked)" style="margin-right:8px;">
                    <span style="font-size:12px;">Mostrar heatmap historico</span>
                </label>
            </div>
        </div>

        <div id="timeline-btn" class="minimized-btn" style="
            top: 10px;
            right: 10px;
            background: {COLORS['ui']['secondary']};
        " onclick="toggleTimeline()">
            Linea de Tiempo
        </div>

        <style>
            #timeline-panel .tab-btn.active {{
                color: {COLORS['ui']['secondary']};
                border-bottom: 2px solid {COLORS['ui']['secondary']};
            }}
        </style>

        <script>
            const monthsLabels = {months_labels};
            const fishingRates = {fishing_rates};
            const sstValues = {sst_values};
            const forecastLabels = {forecast_labels};
            const forecastProbs = {forecast_probs};
            const forecastSst = {forecast_sst};

            document.addEventListener('DOMContentLoaded', function() {{
                initTimelineCharts();
            }});
            setTimeout(initTimelineCharts, 500);

            function initTimelineCharts() {{
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
                if (typeof multidayHourlyData !== 'undefined' && multidayHourlyData[date]) {{
                    updateHourlyPanelForDate(date);
                }} else {{
                    alert('Fecha seleccionada: ' + date + '\\nEjecuta: python main.py --date ' + date);
                }}
            }}

            function toggleHeatmap(show) {{
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
