"""
Map component using Folium.
Handles map creation and marker rendering.
"""

import folium
import numpy as np
from folium.plugins import HeatMap
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
    Core map component using Folium.
    Handles map creation and all marker/layer rendering.
    """

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self.map: Optional[folium.Map] = None
        self._spot_score_range = (0, 100)

    def create(self, center: Tuple[float, float] = None, zoom: int = None) -> folium.Map:
        """Initialize the map with tile layers."""
        center = center or self.config.center
        zoom = zoom or self.config.zoom

        self.map = folium.Map(location=list(center), zoom_start=zoom, tiles=None)

        # Satellite layer
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satelite"
        ).add_to(self.map)

        # Street layer
        folium.TileLayer(tiles="OpenStreetMap", name="Calles").add_to(self.map)

        return self.map

    def add_coastline(self, points: List[Tuple[float, float]], segments: List[List[Tuple[float, float]]] = None):
        """Add coastline polyline(s)."""
        if not self.map:
            return

        if segments and len(segments) > 0:
            for i, segment in enumerate(segments):
                if len(segment) < 2:
                    continue
                folium.PolyLine(
                    locations=segment,
                    color=COLORS['coast'],
                    weight=3,
                    opacity=0.9,
                    popup=f"Segmento {i+1} ({len(segment)} puntos)"
                ).add_to(self.map)
        elif points:
            folium.PolyLine(
                locations=points,
                color=COLORS['coast'],
                weight=3,
                opacity=0.9,
                popup=f"Costa OSM ({len(points)} puntos)"
            ).add_to(self.map)

    def add_fish_zones(self, zones: List[Dict]):
        """Add fish zone circles with movement arrows."""
        if not self.map or not zones:
            return

        regular_zones = [z for z in zones if not z.get('is_anchovy')]
        anchovy_zones = [z for z in zones if z.get('is_anchovy')]

        # Regular fish zones
        fg = folium.FeatureGroup(name="Zonas de Peces")
        for zone in regular_zones:
            intensity = zone.get('intensity', 0.5)
            colors = get_zone_colors(intensity)

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

        # Anchovy zones
        if anchovy_zones:
            fg_anchovy = folium.FeatureGroup(name="Zonas Anchoveta")
            for zone in anchovy_zones:
                intensity = zone.get('intensity', 0.5)
                colors = get_anchovy_colors(intensity)

                folium.Circle(
                    location=[zone['lat'], zone['lon']],
                    radius=zone.get('radius', 400) * 1.8,
                    color=colors['border'],
                    weight=4,
                    fill=True,
                    fillColor=colors['fill'],
                    fillOpacity=0.4,
                    popup=self._anchovy_popup(zone)
                ).add_to(fg_anchovy)

                folium.Marker(
                    location=[zone['lat'], zone['lon']],
                    icon=folium.DivIcon(
                        html='<div style="font-size:20px;text-shadow:1px 1px 2px black;transform:translate(-10px,-10px);">🐟</div>',
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

            speed = vectors[i].speed if vectors and i < len(vectors) else 0.1
            color = get_flow_color(speed)

            folium.PolyLine(
                locations=line,
                color=color,
                weight=2,
                opacity=0.5
            ).add_to(fg)

            self._add_flow_arrow(fg, line, color)

        fg.add_to(self.map)

    def add_marine_points(self, points: List):
        """Add SST sampling points as diamonds."""
        if not self.map or not points:
            return

        fg = folium.FeatureGroup(name="Datos Marinos (SST)")

        for point in points:
            sst = point.sst if point.sst is not None else 17.0
            color = get_sst_color(sst)

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
        """Add fishing spot markers with heatmap colors based on scores."""
        if not self.map or not spots:
            return

        scores = [s['score'] for s in spots]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 100
        self._spot_score_range = (min_score, max_score)

        fg = folium.FeatureGroup(name="Puntos de Pesca")

        for i, spot in enumerate(spots):
            is_best = (i == 0)
            is_top = (i < top_n)

            if is_best:
                color = '#FF0000'
            else:
                color = get_heatmap_color(spot['score'], min_score, max_score)

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

            if is_top:
                self._add_spot_label(fg, spot, i)

        fg.add_to(self.map)

    def add_heatmap(self, heatmap_data: List[Dict], show: bool = False):
        """Add historical fishing activity heatmap."""
        if not self.map or not heatmap_data:
            return

        heat_data = [
            [h['lat'], h['lon'], h['intensity']]
            for h in heatmap_data
            if h['intensity'] > 0
        ]

        if heat_data:
            fg = folium.FeatureGroup(name="Heatmap Historico", show=show)
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_zoom=13,
                radius=20,
                blur=15,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}
            ).add_to(fg)
            fg.add_to(self.map)

    def add_user_location(self, lat: float, lon: float, radius_km: float = 5):
        """Add user location marker with radius circle."""
        if not self.map:
            return

        fg = folium.FeatureGroup(name="Tu Ubicacion")

        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000,
            color='#2196F3',
            weight=2,
            fill=True,
            fillColor='#2196F3',
            fillOpacity=0.1,
            popup=f'Radio de busqueda: {radius_km}km'
        ).add_to(fg)

        folium.Marker(
            location=[lat, lon],
            popup=f'''
                <div style="font-family: Arial; width: 150px;">
                    <b>Tu Ubicacion</b><br>
                    <small>Lat: {lat:.4f}</small><br>
                    <small>Lon: {lon:.4f}</small><br>
                    <small>Radio: {radius_km}km</small>
                </div>
            ''',
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(fg)

        fg.add_to(self.map)

    def finalize(self) -> folium.Map:
        """Add layer control and return map."""
        if self.map:
            folium.LayerControl().add_to(self.map)
        return self.map

    def save(self, filepath: str):
        """Save map to HTML file."""
        if self.map:
            self.map.save(filepath)

    def get_score_range(self) -> Tuple[float, float]:
        """Get the score range for legend."""
        return self._spot_score_range

    # === Private helper methods ===

    def _add_movement_arrow(self, fg: folium.FeatureGroup, zone: Dict):
        direction = zone.get('movement_direction', 90)
        rad = np.radians(direction)
        arrow_len = 250

        end_lat = zone['lat'] + arrow_len / 111000 * np.cos(rad)
        end_lon = zone['lon'] + arrow_len / (111000 * np.cos(np.radians(zone['lat']))) * np.sin(rad)

        folium.PolyLine(
            locations=[[zone['lat'], zone['lon']], [end_lat, end_lon]],
            color=COLORS['movement'],
            weight=4,
            opacity=0.9
        ).add_to(fg)

        folium.RegularPolygonMarker(
            location=[end_lat, end_lon],
            number_of_sides=3,
            radius=8,
            rotation=direction - 90,
            color=COLORS['movement'],
            fill=True,
            fillColor=COLORS['movement'],
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
            radius=4,
            rotation=angle - 90,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6
        ).add_to(fg)

    def _add_spot_label(self, fg: folium.FeatureGroup, spot: Dict, index: int):
        species = spot.get('species', [])
        species_text = ', '.join(s['name'] for s in species[:2]) if species else ''
        label = f'#{index+1}'
        if species_text:
            label += f' {species_text}'
        folium.Marker(
            location=[spot['lat'], spot['lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:12px;font-weight:bold;color:white;text-shadow:1px 1px 3px black;white-space:nowrap;">{label}</div>',
                icon_size=(100, 20),
                icon_anchor=(50, 10)
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
        score = zone.get('intensity', 0) * 100
        rating = "Alta" if score >= 70 else "Media" if score >= 40 else "Baja"

        return f"""
        <div style="font-family:Arial;min-width:200px;">
        <h4 style="margin:0;color:#FF4500;">🐟 Zona Anchoveta</h4>
        <hr style="margin:5px 0;border-color:#FFD700;">
        <b>Probabilidad:</b> {rating} ({score:.0f}%)<br>
        <b>SST:</b> {zone.get('sst', 'N/A')}C<br>
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
