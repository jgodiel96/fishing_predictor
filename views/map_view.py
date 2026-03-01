"""
Map visualization coordinator.
Refactored to use component-based architecture.

This module coordinates between:
- MapComponent: Core Folium map and markers
- TimelinePanel: Historical data and forecasts
- HourlyPanel: Hour-by-hour predictions
- Legend: Map legend
- Layouts: Desktop and mobile responsive layouts
"""

import folium
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import components
from views.components.map_component import MapComponent, MapConfig
from views.components.timeline_panel import TimelinePanel
from views.components.hourly_panel import HourlyPanel
from views.components.legend import Legend

# Import styles
from views.styles.map_styles import (
    COLORS, get_base_css, get_mobile_css,
    get_sst_color, get_flow_color, get_zone_colors, get_heatmap_color
)


class MapView:
    """
    Coordinator for map visualization.
    Delegates to specialized components for rendering.

    This class maintains backward compatibility with the original API
    while using the new component-based architecture internally.
    """

    # Expose COLORS for backward compatibility
    COLORS = COLORS

    def __init__(self, config: Optional[MapConfig] = None):
        """Initialize MapView with optional configuration."""
        self.config = config or MapConfig()
        self._map_component = MapComponent(self.config)
        self._timeline_panel: Optional[TimelinePanel] = None
        self._hourly_panel: Optional[HourlyPanel] = None
        self._legend: Optional[Legend] = None

    @property
    def map(self) -> Optional[folium.Map]:
        """Access the underlying Folium map."""
        return self._map_component.map

    def create_map(self, center: Tuple[float, float] = None, zoom: int = None) -> folium.Map:
        """Initialize the map."""
        folium_map = self._map_component.create(center, zoom)

        # Inject base CSS so panels render with proper background/borders/z-index
        folium_map.get_root().html.add_child(folium.Element(get_base_css()))

        # Initialize panel components
        self._timeline_panel = TimelinePanel(folium_map)
        self._hourly_panel = HourlyPanel(folium_map)
        self._legend = Legend(folium_map)

        return folium_map

    # === Delegate methods to MapComponent ===

    def add_coastline(self, points: List[Tuple[float, float]], segments: List[List[Tuple[float, float]]] = None):
        """Add coastline polyline(s)."""
        self._map_component.add_coastline(points, segments)

    def add_fish_zones(self, zones: List[Dict]):
        """Add fish zone circles with movement arrows."""
        self._map_component.add_fish_zones(zones)

    def add_flow_lines(self, flow_lines: List[List[Tuple[float, float]]], vectors: List = None):
        """Add current flow lines with direction arrows."""
        self._map_component.add_flow_lines(flow_lines, vectors)

    def add_marine_points(self, points: List):
        """Add SST sampling points as diamonds."""
        self._map_component.add_marine_points(points)

    def add_fishing_spots(self, spots: List[Dict], top_n: int = 5):
        """Add fishing spot markers with heatmap colors based on scores."""
        self._map_component.add_fishing_spots(spots, top_n)

    def add_user_location(self, lat: float, lon: float, radius_km: float = 5):
        """Add user location marker with radius circle."""
        self._map_component.add_user_location(lat, lon, radius_km)

    # === Delegate methods to Panel Components ===

    def add_legend(self):
        """Add map legend."""
        if self._legend:
            score_range = self._map_component.get_score_range()
            self._legend.render(score_range)

    def add_timeline(self, timeline_data: Dict):
        """Add timeline controls, charts, and forecast panel."""
        if self._timeline_panel:
            # Add heatmap layer first
            if timeline_data.get('heatmap'):
                self._map_component.add_heatmap(timeline_data['heatmap'])

            self._timeline_panel.render(timeline_data)

    def add_hourly_panel(self, hourly_data: Dict):
        """Add hourly predictions panel with tide chart and best hours."""
        if self._hourly_panel:
            self._hourly_panel.render(hourly_data)

    def add_multiday_hourly_data(self, multiday_data: Dict):
        """Embed multi-day hourly predictions data for dynamic date selection."""
        if self._hourly_panel:
            self._hourly_panel.render_multiday(multiday_data)

    def add_multiday_spots(self, multiday_predictions: list):
        """Embed multi-day top spots data as JS variable for day selector marker updates."""
        if not self.map or not multiday_predictions:
            return
        import json
        js_data = {}
        for day in multiday_predictions:
            date_str = day.get('date', '')
            js_data[date_str] = day.get('top_spots', [])
        html = f'''
        <script>
            const multidaySpotsData = {json.dumps(js_data)};
        </script>
        '''
        self.map.get_root().html.add_child(folium.Element(html))

    def add_hourly_spots_data(self, hourly_spots_data: Dict[int, List[Dict]]):
        """Embed 24-hour unified scoring data for dynamic marker updates."""
        if self._hourly_panel:
            self._hourly_panel.render_hourly_spots(hourly_spots_data)

    # === Finalization methods ===

    def finalize(self) -> folium.Map:
        """Add layer control and return map."""
        return self._map_component.finalize()

    def save(self, filepath: str):
        """Save map to HTML file."""
        self._map_component.save(filepath)

    # === Backward compatibility methods ===

    def _get_zone_colors(self, intensity: float) -> Dict[str, str]:
        """Get colors for fish zones. (Backward compatibility)"""
        return get_zone_colors(intensity)

    def _get_flow_color(self, speed: float) -> str:
        """Get color for current flow. (Backward compatibility)"""
        return get_flow_color(speed)

    def _get_sst_color(self, sst: float) -> str:
        """Get color for SST. (Backward compatibility)"""
        return get_sst_color(sst)

    def _get_spot_color(self, score: float, is_best: bool, min_score: float = 0, max_score: float = 100) -> str:
        """Get heatmap color for spot. (Backward compatibility)"""
        if is_best:
            return '#FF0000'
        return get_heatmap_color(score, min_score, max_score)

    def _get_rating(self, score: float) -> str:
        """Get text rating for score. (Backward compatibility)"""
        if score >= 80:
            return "Excelente"
        elif score >= 60:
            return "Bueno"
        elif score >= 40:
            return "Regular"
        return "Bajo"

    # === Helper for adding heatmap ===

    def _add_heatmap_layer(self, heatmap_data: List[Dict]):
        """Add historical fishing activity heatmap. (Backward compatibility)"""
        self._map_component.add_heatmap(heatmap_data)


# For backward compatibility, expose MapConfig at module level
__all__ = ['MapView', 'MapConfig', 'COLORS']
