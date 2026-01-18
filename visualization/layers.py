"""
Capas para el mapa de visualizacion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SCORE_COLORS, LOCATIONS


class LayerManager:
    """Gestiona las capas del mapa."""

    def __init__(self):
        """Inicializa el gestor de capas."""
        self.score_colormap = LinearColormap(
            colors=[
                SCORE_COLORS["poor"],
                SCORE_COLORS["below_avg"],
                SCORE_COLORS["average"],
                SCORE_COLORS["good"],
                SCORE_COLORS["excellent"]
            ],
            vmin=0,
            vmax=100,
            caption="Score de Pesca (0-100)"
        )

        self.sst_colormap = LinearColormap(
            colors=["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"],
            vmin=14,
            vmax=22,
            caption="SST (°C)"
        )

        self.chlorophyll_colormap = LinearColormap(
            colors=["#f0f0f0", "#90EE90", "#228B22", "#006400"],
            vmin=0,
            vmax=5,
            caption="Clorofila-a (mg/m³)"
        )

    def create_score_heatmap(
        self,
        df: pd.DataFrame,
        radius: int = 15,
        blur: int = 10
    ) -> HeatMap:
        """
        Crea heatmap de scores de pesca.

        Args:
            df: DataFrame con latitude, longitude, score
            radius: Radio del punto de calor
            blur: Difuminado

        Returns:
            HeatMap layer
        """
        # Preparar datos para heatmap
        heat_data = df[["latitude", "longitude", "score"]].values.tolist()

        # Normalizar scores a 0-1 para intensidad
        max_score = df["score"].max()
        if max_score > 0:
            heat_data = [
                [row[0], row[1], row[2] / max_score]
                for row in heat_data
            ]

        return HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            gradient={
                0.2: SCORE_COLORS["poor"],
                0.4: SCORE_COLORS["below_avg"],
                0.6: SCORE_COLORS["average"],
                0.8: SCORE_COLORS["good"],
                1.0: SCORE_COLORS["excellent"]
            },
            name="Score de Pesca"
        )

    def create_sst_layer(
        self,
        df: pd.DataFrame
    ) -> folium.FeatureGroup:
        """
        Crea capa de temperatura superficial del mar.

        Args:
            df: DataFrame con latitude, longitude, sst

        Returns:
            FeatureGroup con puntos de SST
        """
        layer = folium.FeatureGroup(name="SST (Temperatura)", show=False)

        for _, row in df.iterrows():
            if pd.isna(row.get("sst")):
                continue

            color = self.sst_colormap(row["sst"])

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"SST: {row['sst']:.1f}°C"
            ).add_to(layer)

        return layer

    def create_chlorophyll_layer(
        self,
        df: pd.DataFrame
    ) -> folium.FeatureGroup:
        """
        Crea capa de clorofila-a.

        Args:
            df: DataFrame con latitude, longitude, chlorophyll

        Returns:
            FeatureGroup con puntos de clorofila
        """
        layer = folium.FeatureGroup(name="Clorofila-a", show=False)

        for _, row in df.iterrows():
            if pd.isna(row.get("chlorophyll")):
                continue

            color = self.chlorophyll_colormap(row["chlorophyll"])

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"Chl-a: {row['chlorophyll']:.2f} mg/m³"
            ).add_to(layer)

        return layer

    def create_reference_points_layer(
        self,
        locations: Optional[Dict] = None
    ) -> folium.FeatureGroup:
        """
        Crea capa con puntos de referencia (playas).

        Args:
            locations: Dict de ubicaciones

        Returns:
            FeatureGroup con marcadores
        """
        if locations is None:
            locations = LOCATIONS

        layer = folium.FeatureGroup(name="Puntos de Pesca")

        for name, coords in locations.items():
            folium.Marker(
                location=[coords["lat"], coords["lon"]],
                popup=folium.Popup(
                    f"<b>{name}</b><br>Lat: {coords['lat']:.3f}<br>Lon: {coords['lon']:.3f}",
                    max_width=200
                ),
                icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
                tooltip=name
            ).add_to(layer)

        return layer

    def create_top_spots_layer(
        self,
        df: pd.DataFrame,
        n: int = 5
    ) -> folium.FeatureGroup:
        """
        Crea capa con los mejores spots.

        Args:
            df: DataFrame con scores
            n: Numero de top spots

        Returns:
            FeatureGroup con marcadores de top spots
        """
        layer = folium.FeatureGroup(name="Mejores Spots")

        top_spots = df.nlargest(n, "score")

        for rank, (_, row) in enumerate(top_spots.iterrows(), 1):
            # Color basado en ranking
            if rank == 1:
                color = "gold"
                icon_color = "green"
            elif rank <= 3:
                color = "lightgreen"
                icon_color = "lightgreen"
            else:
                color = "beige"
                icon_color = "orange"

            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0; color: {icon_color};">#{rank} Top Spot</h4>
                <hr style="margin: 5px 0;">
                <b>Score:</b> {row['score']:.0f}/100<br>
                <b>Categoria:</b> {row['category']}<br>
                <b>Lat:</b> {row['latitude']:.4f}<br>
                <b>Lon:</b> {row['longitude']:.4f}<br>
            </div>
            """

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=icon_color, icon="star", prefix="fa"),
                tooltip=f"#{rank} - Score: {row['score']:.0f}"
            ).add_to(layer)

        return layer

    def create_gradient_layer(
        self,
        df: pd.DataFrame
    ) -> folium.FeatureGroup:
        """
        Crea capa de gradientes de SST (frentes termicos).

        Args:
            df: DataFrame con latitude, longitude, sst_gradient, is_thermal_front

        Returns:
            FeatureGroup con indicadores de frentes
        """
        layer = folium.FeatureGroup(name="Frentes Termicos", show=False)

        # Filtrar solo puntos que son frentes
        fronts = df[df.get("is_thermal_front", df["sst_gradient"] > 0.5) == 1]

        for _, row in fronts.iterrows():
            intensity = min(row["sst_gradient"] / 2.0, 1.0)  # Normalizar

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color="#FF4500",
                fill=True,
                fillColor="#FF4500",
                fillOpacity=intensity,
                popup=f"Gradiente: {row['sst_gradient']:.2f} °C/km"
            ).add_to(layer)

        return layer

    def add_colormaps_to_map(self, m: folium.Map):
        """
        Agrega las escalas de colores al mapa.

        Args:
            m: Mapa de Folium
        """
        self.score_colormap.add_to(m)
