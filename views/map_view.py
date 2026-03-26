"""
Map visualization coordinator.
Component-based architecture using deck.gl + MapLibre GL JS.

Components:
- MapComponent: Core map rendering (deck.gl layers)
- TimelinePanel: Historical data and forecasts (Charts.js)
- HourlyPanel: Hour-by-hour predictions
- Legend: Map legend
"""

import json
from typing import List, Dict, Tuple, Optional

from views.components.map_component import MapComponent, MapConfig
from views.components.timeline_panel import TimelinePanel
from views.components.hourly_panel import HourlyPanel
from views.components.legend import Legend

from views.styles.map_styles import (
    COLORS, get_base_css, get_mobile_css,
    get_sst_color, get_flow_color, get_zone_colors, get_heatmap_color
)


class MapView:
    """
    Coordinator for map visualization.
    Delegates to specialized components for rendering.
    """

    COLORS = COLORS

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self._map_component = MapComponent(self.config)
        self._timeline_panel: Optional[TimelinePanel] = None
        self._hourly_panel: Optional[HourlyPanel] = None
        self._legend: Optional[Legend] = None
        self._multiday_spots_html = ''

    @property
    def map(self):
        """Backward compat — returns MapComponent which handles add_child()."""
        return self._map_component

    def create_map(self, center: Tuple[float, float] = None, zoom: int = None):
        """Initialize the map."""
        self._map_component.create(center, zoom)

        # Initialize panel components (no map dependency)
        self._timeline_panel = TimelinePanel()
        self._hourly_panel = HourlyPanel()
        self._legend = Legend()

        return self._map_component

    # === Delegate methods to MapComponent ===

    def add_coastline(self, points: List[Tuple[float, float]],
                      segments: List[List[Tuple[float, float]]] = None):
        self._map_component.add_coastline(points, segments)

    def add_fish_zones(self, zones: List[Dict]):
        self._map_component.add_fish_zones(zones)

    def add_flow_lines(self, flow_lines: List[List[Tuple[float, float]]], vectors: List = None):
        self._map_component.add_flow_lines(flow_lines, vectors)

    def add_marine_points(self, points: List):
        self._map_component.add_marine_points(points)

    def add_fishing_spots(self, spots: List[Dict], top_n: int = 5):
        self._map_component.add_fishing_spots(spots, top_n)

    def add_user_location(self, lat: float, lon: float, radius_km: float = 5):
        self._map_component.add_user_location(lat, lon, radius_km)

    # === Delegate methods to Panel Components ===

    def add_legend(self):
        if self._legend:
            score_range = self._map_component.get_score_range()
            self._legend.render(score_range)

    def add_timeline(self, timeline_data: Dict):
        if self._timeline_panel:
            if timeline_data.get('heatmap'):
                self._map_component.add_heatmap(timeline_data['heatmap'])
            self._timeline_panel.render(timeline_data)

    def add_hourly_panel(self, hourly_data: Dict):
        if self._hourly_panel:
            self._hourly_panel.render(hourly_data)

    def add_multiday_hourly_data(self, multiday_data: Dict):
        if self._hourly_panel:
            self._hourly_panel.render_multiday(multiday_data)

    def add_multiday_spots(self, multiday_predictions: list):
        if not multiday_predictions:
            return
        js_data = {}
        for day in multiday_predictions:
            date_str = day.get('date', '')
            js_data[date_str] = day.get('top_spots', [])
        self._multiday_spots_html = f'''
        <script>
            const multidaySpotsData = {json.dumps(js_data)};
        </script>
        '''

    def add_hourly_spots_data(self, hourly_spots_data: Dict[int, List[Dict]]):
        if self._hourly_panel:
            self._hourly_panel.render_hourly_spots(hourly_spots_data)

    # === Finalization methods ===

    def finalize(self):
        return self._map_component

    def save(self, filepath: str):
        """Assemble all components and save as single HTML file."""
        base_html = self._map_component.render_html()

        # Collect panel HTML
        panels = []
        panels.append(get_base_css())
        panels.append(get_mobile_css())

        if self._multiday_spots_html:
            panels.append(self._multiday_spots_html)
        if self._timeline_panel:
            panels.append(self._timeline_panel.get_html())
        if self._hourly_panel:
            panels.append(self._hourly_panel.get_html())
        if self._legend:
            panels.append(self._legend.get_html())

        panels_html = '\n'.join(p for p in panels if p)

        # Inject panels before </body>
        full_html = base_html.replace('<!--PANELS_PLACEHOLDER-->', panels_html)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_html)

    # === Backward compatibility methods ===

    def _get_zone_colors(self, intensity: float) -> Dict[str, str]:
        return get_zone_colors(intensity)

    def _get_flow_color(self, speed: float) -> str:
        return get_flow_color(speed)

    def _get_sst_color(self, sst: float) -> str:
        return get_sst_color(sst)

    def _get_spot_color(self, score: float, is_best: bool, min_score: float = 0, max_score: float = 100) -> str:
        if is_best:
            return '#FF0000'
        return get_heatmap_color(score, min_score, max_score)

    def _get_rating(self, score: float) -> str:
        if score >= 80:
            return "Excelente"
        elif score >= 60:
            return "Bueno"
        elif score >= 40:
            return "Regular"
        return "Bajo"

    def _add_heatmap_layer(self, heatmap_data: List[Dict]):
        self._map_component.add_heatmap(heatmap_data)


__all__ = ['MapView', 'MapConfig', 'COLORS']
