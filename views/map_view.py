"""
Map visualization using Folium.
Separates rendering logic from data processing.
"""

import folium
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MapConfig:
    """Configuration for map rendering."""
    center: Tuple[float, float] = (-17.9, -71.0)
    zoom: int = 10
    tile_layer: str = "satellite"


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

        fg = folium.FeatureGroup(name="Zonas de Peces")

        for zone in zones:
            intensity = zone.get('intensity', 0.5)
            colors = self._get_zone_colors(intensity)

            # Zone circle
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

            # Center marker
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=8,
                color="#000",
                weight=2,
                fill=True,
                fillColor=colors['border'],
                fillOpacity=1.0
            ).add_to(fg)

            # Movement arrow
            self._add_movement_arrow(fg, zone)

        fg.add_to(self.map)

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
            <small style="color:#666;">ML + Datos Reales</small>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>Spots:</b><br>
            <span style="color:#FF0000;">●</span> Mejor<br>
            <span style="color:#228B22;">●</span> Excelente (80+)<br>
            <span style="color:#32CD32;">●</span> Bueno (60-80)<br>
            <span style="color:#FFD700;">●</span> Regular (40-60)<br>

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
