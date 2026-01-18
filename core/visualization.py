"""
Visualizacion de mapas con transectos y flechas de movimiento.
Usa Folium para crear mapas interactivos.
"""

import folium
from folium import plugins
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .transects import Transect, TransectAnalyzer
from .fish_movement import MovementVector, MovementTrend
from .scoring import FishingScore
from .coastline import CoastlineModel, SubstrateType


# Colores por tipo de sustrato
SUBSTRATE_COLORS = {
    "roca": "#8B4513",  # Marron
    "arena": "#F4A460",  # Arena
    "mixto": "#CD853F"  # Peru
}

# Colores por tendencia de movimiento
TREND_COLORS = {
    MovementTrend.HACIA_COSTA: "#00FF00",  # Verde
    MovementTrend.HACIA_MAR: "#FF6600",    # Naranja
    MovementTrend.PARALELO_COSTA: "#0066FF",  # Azul
    MovementTrend.ESTACIONARIO: "#888888"  # Gris
}


class FishingMapBuilder:
    """
    Constructor de mapas de pesca con transectos y flechas de movimiento.
    """

    def __init__(
        self,
        center_lat: float = -17.85,
        center_lon: float = -71.15,
        zoom_start: int = 12
    ):
        """
        Inicializa el constructor de mapas.

        Args:
            center_lat: latitud del centro del mapa
            center_lon: longitud del centro del mapa
            zoom_start: nivel de zoom inicial
        """
        self.center = (center_lat, center_lon)
        self.zoom_start = zoom_start
        self.map = None

    def create_base_map(self) -> folium.Map:
        """Crea el mapa base con capas satelite y calles."""
        self.map = folium.Map(
            location=list(self.center),
            zoom_start=self.zoom_start,
            tiles=None
        )

        # Capa satelite (Esri)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satelite",
            overlay=False
        ).add_to(self.map)

        # Capa OpenStreetMap
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Calles",
            overlay=False
        ).add_to(self.map)

        return self.map

    def add_coastline_segments(
        self,
        coastline: CoastlineModel,
        show_all: bool = True
    ):
        """
        Agrega los segmentos de costa al mapa.

        Args:
            coastline: modelo de costa
            show_all: mostrar todos los segmentos
        """
        if self.map is None:
            self.create_base_map()

        fg = folium.FeatureGroup(name="Linea de Costa")

        for segment in coastline.segments:
            # Linea del segmento
            folium.PolyLine(
                locations=[
                    [segment.lat_start, segment.lon_start],
                    [segment.lat_end, segment.lon_end]
                ],
                color=SUBSTRATE_COLORS.get(segment.substrate.value, "#888"),
                weight=4,
                opacity=0.8,
                popup=f"<b>{segment.name}</b><br>Sustrato: {segment.substrate.value}<br>{segment.description}"
            ).add_to(fg)

            # Marcador en el centro
            center_lat, center_lon = segment.center
            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=6,
                color="#333",
                weight=1,
                fill=True,
                fillColor=SUBSTRATE_COLORS.get(segment.substrate.value, "#888"),
                fillOpacity=0.9,
                popup=f"<b>{segment.name}</b>",
                tooltip=segment.name
            ).add_to(fg)

        fg.add_to(self.map)

    def add_transects(
        self,
        transects: List[Transect],
        show_points: bool = True
    ):
        """
        Agrega transectos al mapa.

        Args:
            transects: lista de transectos
            show_points: mostrar puntos de muestreo
        """
        if self.map is None:
            self.create_base_map()

        fg = folium.FeatureGroup(name="Transectos")

        for transect in transects:
            if not transect.points:
                continue

            # Linea del transecto
            coordinates = [[p.lat, p.lon] for p in transect.points]

            # Color basado en SST promedio
            color = self._get_temperature_color(transect.avg_sst)

            folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=3,
                opacity=0.7,
                popup=self._create_transect_popup(transect),
                tooltip=f"{transect.name}: SST {transect.avg_sst:.1f}C" if transect.avg_sst else transect.name
            ).add_to(fg)

            # Punto de orilla (mas grande)
            shore = transect.points[0]
            folium.CircleMarker(
                location=[shore.lat, shore.lon],
                radius=8,
                color="#000",
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=f"<b>Orilla - {transect.name}</b>"
            ).add_to(fg)

            # Puntos de muestreo (si se solicita)
            if show_points:
                for point in transect.points[1:]:
                    folium.CircleMarker(
                        location=[point.lat, point.lon],
                        radius=4,
                        color=color,
                        weight=1,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.6,
                        popup=f"SST: {point.sst}C<br>Chl: {point.chlorophyll} mg/m3<br>Dist: {point.distance_from_shore_m:.0f}m"
                    ).add_to(fg)

        fg.add_to(self.map)

    def add_movement_arrows(
        self,
        vectors: List[MovementVector],
        scale: float = 200
    ):
        """
        Agrega flechas de movimiento de peces al mapa.

        Args:
            vectors: lista de vectores de movimiento
            scale: escala de las flechas en metros
        """
        if self.map is None:
            self.create_base_map()

        fg = folium.FeatureGroup(name="Movimiento de Peces")

        for vector in vectors:
            # Punto final de la flecha
            end_lat, end_lon = vector.get_arrow_endpoint(scale * vector.magnitude)

            # Color segun tendencia
            color = TREND_COLORS.get(vector.trend, "#888888")

            # Linea principal de la flecha
            folium.PolyLine(
                locations=[
                    [vector.lat, vector.lon],
                    [end_lat, end_lon]
                ],
                color=color,
                weight=4,
                opacity=0.8
            ).add_to(fg)

            # Punta de flecha (triangulo)
            self._add_arrow_head(
                fg,
                start=(vector.lat, vector.lon),
                end=(end_lat, end_lon),
                color=color,
                size=scale * 0.15 * vector.magnitude
            )

            # Popup en el origen
            popup_html = f"""
            <div style="font-family:Arial;width:180px;">
                <h4 style="margin:0 0 8px 0;color:#1a5f7a;">Movimiento</h4>
                <p><b>Tendencia:</b> {vector.trend.value}</p>
                <p><b>Direccion:</b> {vector.direction_deg:.0f}</p>
                <p><b>Intensidad:</b> {vector.magnitude:.0%}</p>
                <p><b>Confianza:</b> {vector.confidence:.0%}</p>
                <hr style="margin:8px 0;">
                <p><b>Especies:</b> {', '.join(vector.target_species) if vector.target_species else 'N/A'}</p>
            </div>
            """

            folium.CircleMarker(
                location=[vector.lat, vector.lon],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(fg)

        fg.add_to(self.map)

    def _add_arrow_head(
        self,
        feature_group: folium.FeatureGroup,
        start: Tuple[float, float],
        end: Tuple[float, float],
        color: str,
        size: float
    ):
        """Agrega la punta de flecha (triangulo)."""
        lat1, lon1 = start
        lat2, lon2 = end

        # Calcular direccion
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        angle = np.arctan2(dlon, dlat)

        # Puntos del triangulo
        size_deg = size / 111000  # Convertir a grados

        # Dos puntos detras formando un triangulo
        angle_offset = np.radians(25)

        p1_lat = lat2 - size_deg * np.cos(angle - angle_offset)
        p1_lon = lon2 - size_deg * np.sin(angle - angle_offset) / np.cos(np.radians(lat2))

        p2_lat = lat2 - size_deg * np.cos(angle + angle_offset)
        p2_lon = lon2 - size_deg * np.sin(angle + angle_offset) / np.cos(np.radians(lat2))

        # Dibujar triangulo
        folium.Polygon(
            locations=[
                [lat2, lon2],
                [p1_lat, p1_lon],
                [p2_lat, p2_lon]
            ],
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.9,
            weight=1
        ).add_to(feature_group)

    def add_fishing_scores(
        self,
        scores: List[FishingScore],
        show_rank: bool = True
    ):
        """
        Agrega marcadores de puntuacion de pesca.

        Args:
            scores: lista de scores
            show_rank: mostrar ranking numerico
        """
        if self.map is None:
            self.create_base_map()

        fg = folium.FeatureGroup(name="Scores de Pesca")

        for i, score in enumerate(scores):
            # Popup detallado
            popup_html = self._create_score_popup(score, rank=i+1 if show_rank else None)

            # Marcador circular con color segun score
            folium.CircleMarker(
                location=[score.latitude, score.longitude],
                radius=12,
                color="#000",
                weight=2,
                fill=True,
                fillColor=score.color,
                fillOpacity=0.85,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{score.location_name}: {score.total_score:.0f}/100"
            ).add_to(fg)

            # Etiqueta de ranking
            if show_rank and i < 5:
                folium.Marker(
                    location=[score.latitude, score.longitude],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:12px;font-weight:bold;color:white;text-shadow:1px 1px 2px black;">#{i+1}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    )
                ).add_to(fg)

        fg.add_to(self.map)

    def add_legend(self):
        """Agrega leyenda al mapa."""
        if self.map is None:
            return

        legend_html = '''
        <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
                    background:white;padding:15px;border-radius:8px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-family:Arial;
                    font-size:12px;max-width:200px;">
            <h4 style="margin:0 0 10px 0;border-bottom:1px solid #ccc;padding-bottom:5px;">
                Leyenda
            </h4>
            <div style="margin-bottom:8px;">
                <b>Score de Pesca:</b><br>
                <span style="background:#228b22;color:white;padding:2px 6px;border-radius:3px;">80-100</span> Excelente<br>
                <span style="background:#90ee90;padding:2px 6px;border-radius:3px;">60-80</span> Bueno<br>
                <span style="background:#ffff00;padding:2px 6px;border-radius:3px;">40-60</span> Promedio<br>
                <span style="background:#ff8c00;color:white;padding:2px 6px;border-radius:3px;">20-40</span> Bajo<br>
                <span style="background:#ff4444;color:white;padding:2px 6px;border-radius:3px;">0-20</span> Pobre
            </div>
            <div style="margin-bottom:8px;">
                <b>Movimiento Peces:</b><br>
                <span style="color:#00FF00;">→</span> Hacia costa<br>
                <span style="color:#FF6600;">→</span> Hacia mar<br>
                <span style="color:#0066FF;">→</span> Paralelo costa
            </div>
            <div>
                <b>Sustrato:</b><br>
                <span style="background:#8B4513;color:white;padding:2px 6px;border-radius:3px;">Roca</span>
                <span style="background:#F4A460;padding:2px 6px;border-radius:3px;">Arena</span>
                <span style="background:#CD853F;padding:2px 6px;border-radius:3px;">Mixto</span>
            </div>
        </div>
        '''

        self.map.get_root().html.add_child(folium.Element(legend_html))

    def _get_temperature_color(self, sst: Optional[float]) -> str:
        """Obtiene color basado en SST."""
        if sst is None:
            return "#888888"

        # Escala de colores de frio (azul) a calido (rojo)
        if sst < 15.0:
            return "#0000FF"  # Azul frio
        elif sst < 15.5:
            return "#0066FF"
        elif sst < 16.0:
            return "#00CCFF"
        elif sst < 16.5:
            return "#00FF66"
        elif sst < 17.0:
            return "#FFFF00"  # Amarillo
        elif sst < 17.5:
            return "#FF9900"
        else:
            return "#FF0000"  # Rojo calido

    def _create_transect_popup(self, transect: Transect) -> str:
        """Crea HTML del popup para un transecto."""
        gradient_text = f"{transect.sst_gradient:+.2f} C/km" if transect.sst_gradient else "N/A"
        front_text = f"{transect.thermal_front_distance_m:.0f}m" if transect.thermal_front_distance_m else "No detectado"

        return f"""
        <div style="font-family:Arial;width:200px;">
            <h4 style="margin:0 0 8px 0;color:#1a5f7a;">{transect.name}</h4>
            <p><b>Sustrato:</b> {transect.substrate.value}</p>
            <p><b>SST promedio:</b> {transect.avg_sst:.1f} C</p>
            <p><b>Clorofila:</b> {transect.avg_chlorophyll:.1f} mg/m3</p>
            <hr style="margin:8px 0;">
            <p><b>Gradiente SST:</b> {gradient_text}</p>
            <p><b>Frente termico:</b> {front_text}</p>
            <p><b>Extension:</b> {transect.max_distance_m:.0f}m</p>
        </div>
        """

    def _create_score_popup(self, score: FishingScore, rank: Optional[int] = None) -> str:
        """Crea HTML del popup para un score."""
        rank_text = f"<span style='font-size:18px;color:#1a5f7a;'>#{rank}</span> " if rank else ""
        safe_icon = "✅" if score.is_safe else "⚠️"

        species_html = ""
        if score.recommended_species:
            species_html = f"<p><b>Especies:</b> {', '.join(score.recommended_species)}</p>"

        return f"""
        <div style="font-family:Arial;width:220px;">
            <h4 style="margin:0 0 5px 0;">{rank_text}{score.location_name}</h4>
            <p style="font-size:20px;margin:5px 0;color:{score.color};">
                <b>{score.total_score:.0f}/100</b> - {score.category.value}
            </p>
            <p>{safe_icon} {'Seguro' if score.is_safe else 'Precaucion'}</p>
            <hr style="margin:8px 0;">
            <p><b>SST:</b> {score.sst_score:.0f} | <b>Chl:</b> {score.chlorophyll_score:.0f}</p>
            <p><b>Seguridad:</b> {score.safety_score:.0f} | <b>Hora:</b> {score.golden_hour_score:.0f}</p>
            <p><b>Sustrato:</b> {score.substrate_type}</p>
            {species_html}
            <p><b>Movimiento:</b> {score.movement_trend or 'N/A'}</p>
        </div>
        """

    def finalize(self, output_path: str) -> str:
        """
        Finaliza el mapa y lo guarda.

        Args:
            output_path: ruta donde guardar el archivo HTML

        Returns:
            Ruta del archivo guardado
        """
        if self.map is None:
            self.create_base_map()

        # Agregar control de capas
        folium.LayerControl().add_to(self.map)

        # Guardar
        self.map.save(output_path)

        return output_path
