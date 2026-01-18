"""
Constructor del mapa interactivo de pesca.
Mapa util con hover, predicciones y recomendaciones.
Usa GeoJSON para geo-referenciar correctamente los puntos.
"""

import folium
from folium.plugins import Fullscreen, MiniMap
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MAP_CENTER, LOCATIONS, OUTPUT_DIR, DEFAULT_OUTPUT_FILE


# Informacion de peces y senuelos
FISH_INFO = {
    "corvina": {
        "nombre": "Corvina",
        "mejor_hora": "Amanecer y atardecer",
        "senuelos": ["Jigs metalicos 30-50g", "Vinilos shad 4\"", "Carnada: muy muy"],
        "habitat": "Fondos arenosos, rompientes"
    },
    "lenguado": {
        "nombre": "Lenguado",
        "mejor_hora": "Madrugada y noche",
        "senuelos": ["Vinilos paddle tail", "Jig head 15-25g", "Carnada: camaron"],
        "habitat": "Fondos arenosos, aguas someras"
    },
    "robalo": {
        "nombre": "Robalo",
        "mejor_hora": "Amanecer, primeras 2 horas",
        "senuelos": ["Poppers 12cm", "Minnows suspending", "Cucharas"],
        "habitat": "Rocas, pozas, desembocaduras"
    },
    "cabrilla": {
        "nombre": "Cabrilla",
        "mejor_hora": "Todo el dia, mejor mañana",
        "senuelos": ["Grubs 3\"", "Jigs pequeños 15-25g", "Carnada: sardina"],
        "habitat": "Zonas rocosas"
    },
    "bonito": {
        "nombre": "Bonito",
        "mejor_hora": "Media mañana",
        "senuelos": ["Jigs rapidos 40-60g", "Casting jigs", "Cucharas plateadas"],
        "habitat": "Frentes termicos, aguas abiertas"
    }
}


class MapBuilder:
    """Construye el mapa interactivo de prediccion de pesca."""

    def __init__(
        self,
        center: Optional[Dict[str, float]] = None,
        zoom_start: int = 10
    ):
        self.center = center or MAP_CENTER
        self.zoom_start = zoom_start
        self.map = None

    def _get_score_color(self, score: float) -> str:
        """Retorna color basado en score."""
        if score >= 75:
            return "#00ff00"  # Verde brillante
        elif score >= 60:
            return "#90ee90"  # Verde claro
        elif score >= 45:
            return "#ffff00"  # Amarillo
        elif score >= 30:
            return "#ffa500"  # Naranja
        else:
            return "#ff4444"  # Rojo

    def _get_recommendation(self, score: float) -> str:
        """Retorna recomendacion basada en score."""
        if score >= 75:
            return "EXCELENTE - Ir a pescar!"
        elif score >= 60:
            return "MUY BUENO - Recomendado"
        elif score >= 45:
            return "BUENO - Condiciones aceptables"
        elif score >= 30:
            return "REGULAR - Considerar otras zonas"
        else:
            return "NO RECOMENDADO"

    def _get_fish_for_conditions(self, score_data: Dict) -> List[str]:
        """Retorna peces probables segun condiciones."""
        fish = []

        # Basado en frentes termicos -> bonito, corvina
        if score_data.get("front_proximity", 0) > 60:
            fish.extend(["bonito", "corvina"])

        # Basado en clorofila alta -> corvina, robalo
        if score_data.get("chlorophyll", 0) > 50:
            fish.extend(["corvina", "robalo"])

        # Condiciones generales
        if score_data.get("safety", 100) > 60:
            fish.extend(["lenguado", "cabrilla"])

        return list(set(fish)) if fish else ["corvina", "lenguado"]

    def _create_spot_popup(
        self,
        lat: float,
        lon: float,
        score_now: float,
        score_24h: float,
        score_48h: float,
        details: Dict,
        best_time: str,
        is_safe: bool
    ) -> str:
        """Crea popup HTML detallado para un spot."""

        color_now = self._get_score_color(score_now)
        color_24h = self._get_score_color(score_24h)
        color_48h = self._get_score_color(score_48h)

        recommendation = self._get_recommendation(score_now)
        safety_text = "Si" if is_safe else "No - Precaucion"
        safety_color = "#00ff00" if is_safe else "#ff4444"

        # Determinar peces probables
        fish_list = self._get_fish_for_conditions(details)
        fish_html = ""
        for fish_key in fish_list[:3]:
            if fish_key in FISH_INFO:
                info = FISH_INFO[fish_key]
                fish_html += f"""
                <div style="margin: 5px 0; padding: 5px; background: #f5f5f5; border-radius: 3px;">
                    <b>{info['nombre']}</b><br>
                    <small>Hora: {info['mejor_hora']}</small><br>
                    <small>Senuelo: {info['senuelos'][0]}</small>
                </div>
                """

        html = f"""
        <div style="font-family: Arial, sans-serif; width: 280px; padding: 10px;">
            <h3 style="margin: 0 0 10px 0; color: #333; border-bottom: 2px solid {color_now}; padding-bottom: 5px;">
                Punto de Pesca
            </h3>

            <div style="background: #f9f9f9; padding: 8px; border-radius: 5px; margin-bottom: 10px;">
                <b>Coordenadas:</b> {lat:.4f}, {lon:.4f}
            </div>

            <h4 style="margin: 10px 0 5px 0; color: #555;">Prediccion de Pesca</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px; border: 1px solid #ddd;"><b>AHORA</b></td>
                    <td style="padding: 5px; border: 1px solid #ddd; background: {color_now}; text-align: center;">
                        <b>{score_now:.0f}/100</b>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 5px; border: 1px solid #ddd;">+24 horas</td>
                    <td style="padding: 5px; border: 1px solid #ddd; background: {color_24h}; text-align: center;">
                        {score_24h:.0f}/100
                    </td>
                </tr>
                <tr>
                    <td style="padding: 5px; border: 1px solid #ddd;">+48 horas</td>
                    <td style="padding: 5px; border: 1px solid #ddd; background: {color_48h}; text-align: center;">
                        {score_48h:.0f}/100
                    </td>
                </tr>
            </table>

            <div style="margin-top: 10px; padding: 8px; background: {color_now}; border-radius: 5px; text-align: center;">
                <b>{recommendation}</b>
            </div>

            <h4 style="margin: 15px 0 5px 0; color: #555;">Mejor Momento</h4>
            <div style="padding: 8px; background: #e8f5e9; border-radius: 5px;">
                {best_time}
            </div>

            <h4 style="margin: 15px 0 5px 0; color: #555;">Seguridad</h4>
            <div style="padding: 5px; background: {safety_color}; border-radius: 3px; text-align: center; color: white;">
                {safety_text}
            </div>

            <h4 style="margin: 15px 0 5px 0; color: #555;">Condiciones</h4>
            <div style="font-size: 12px;">
                <b>SST:</b> {details.get('sst', 'N/A')}°C |
                <b>Olas:</b> {details.get('wave_height', 'N/A')}m |
                <b>Viento:</b> {details.get('wind_speed', 'N/A')} km/h
            </div>

            <h4 style="margin: 15px 0 5px 0; color: #555;">Peces Probables y Senuelos</h4>
            {fish_html}
        </div>
        """
        return html

    def _create_tooltip(self, score: float, recommendation: str) -> str:
        """Crea tooltip corto para hover."""
        return f"Score: {score:.0f}/100 - {recommendation}"

    def _create_geojson_features(
        self,
        score_df: pd.DataFrame,
        weather_data: Optional[Dict],
        best_time: str
    ) -> dict:
        """
        Crea GeoJSON FeatureCollection desde el DataFrame de scores.
        """
        features = []

        for _, row in score_df.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            score_now = float(row["score"])

            # Predicciones simuladas
            score_24h = float(np.clip(score_now * 0.95 + np.random.uniform(-5, 10), 0, 100))
            score_48h = float(np.clip(score_now * 0.90 + np.random.uniform(-8, 12), 0, 100))

            # Detalles
            sst = float(row.get("sst", 16))
            wave_h = float(weather_data.get("wave_height", 1.0)) if weather_data else 1.0
            wind_s = float(weather_data.get("wind_speed", 10)) if weather_data else 10.0
            safety = float(row.get("score_safety", 80))

            is_safe = bool(row.get("is_safe", True))
            loc_name = str(row.get("location", ""))
            recommendation = self._get_recommendation(score_now)
            color = self._get_score_color(score_now)

            # Crear feature GeoJSON
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # GeoJSON usa [lon, lat]
                },
                "properties": {
                    "name": loc_name,
                    "score": score_now,
                    "score_24h": score_24h,
                    "score_48h": score_48h,
                    "sst": sst,
                    "wave_height": wave_h,
                    "wind_speed": wind_s,
                    "safety": safety,
                    "is_safe": is_safe,
                    "recommendation": recommendation,
                    "color": color,
                    "best_time": best_time
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def build(
        self,
        score_df: pd.DataFrame,
        predictions: Optional[Dict] = None,
        weather_data: Optional[Dict] = None,
        astro_data: Optional[Dict] = None
    ) -> folium.Map:
        """
        Construye el mapa usando GeoJSON para geo-referenciar correctamente.
        """
        # Crear mapa base
        self.map = folium.Map(
            location=[self.center["lat"], self.center["lon"]],
            zoom_start=self.zoom_start,
            tiles=None
        )

        # Capas de tiles
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satelite",
            overlay=False
        ).add_to(self.map)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Mapa",
            overlay=False
        ).add_to(self.map)

        # Controles
        Fullscreen().add_to(self.map)
        MiniMap(toggle_display=True).add_to(self.map)

        # Determinar mejor hora
        if astro_data:
            sunrise = astro_data.get("sunrise", "06:00")
            sunset = astro_data.get("sunset", "18:00")
            golden_score = astro_data.get("golden_hour_score", 0)

            if golden_score > 50:
                best_time = "AHORA es hora dorada!"
            else:
                sr = sunrise[:5] if isinstance(sunrise, str) else str(sunrise)[:5]
                ss = sunset[:5] if isinstance(sunset, str) else str(sunset)[:5]
                best_time = f"Amanecer: {sr} | Atardecer: {ss}"
        else:
            best_time = "Amanecer (06:00) o Atardecer (18:00)"

        # Crear GeoJSON con los datos
        geojson_data = self._create_geojson_features(score_df, weather_data, best_time)

        # Funcion de estilo para cada punto
        def style_function(feature):
            score = feature["properties"]["score"]
            color = feature["properties"]["color"]
            return {
                "fillColor": color,
                "color": "#333333",
                "weight": 2,
                "fillOpacity": 0.8,
                "radius": 8 + (score / 12)
            }

        # Funcion para crear popup HTML
        def create_popup(feature):
            props = feature["properties"]
            color_now = self._get_score_color(props["score"])
            color_24h = self._get_score_color(props["score_24h"])
            color_48h = self._get_score_color(props["score_48h"])
            safety_color = "#00ff00" if props["is_safe"] else "#ff4444"
            safety_text = "Seguro" if props["is_safe"] else "Precaucion"

            # Peces recomendados
            fish_html = ""
            if props["score"] >= 60:
                fish_list = ["corvina", "robalo"] if props["sst"] > 16 else ["cabrilla", "lenguado"]
            else:
                fish_list = ["lenguado", "cabrilla"]

            for fish_key in fish_list[:2]:
                if fish_key in FISH_INFO:
                    info = FISH_INFO[fish_key]
                    fish_html += f"""
                    <div style="margin:3px 0;padding:5px;background:#f5f5f5;border-radius:3px;font-size:11px;">
                        <b>{info['nombre']}</b> - {info['senuelos'][0]}
                    </div>"""

            return f"""
            <div style="font-family:Arial;width:260px;padding:8px;">
                <h4 style="margin:0 0 8px 0;color:#1a5f7a;border-bottom:2px solid {color_now};padding-bottom:5px;">
                    {props['name'] or 'Punto de Pesca'}
                </h4>
                <table style="width:100%;border-collapse:collapse;font-size:12px;">
                    <tr>
                        <td style="padding:4px;border:1px solid #ddd;"><b>AHORA</b></td>
                        <td style="padding:4px;border:1px solid #ddd;background:{color_now};text-align:center;">
                            <b>{props['score']:.0f}/100</b>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:4px;border:1px solid #ddd;">+24h</td>
                        <td style="padding:4px;border:1px solid #ddd;background:{color_24h};text-align:center;">
                            {props['score_24h']:.0f}/100
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:4px;border:1px solid #ddd;">+48h</td>
                        <td style="padding:4px;border:1px solid #ddd;background:{color_48h};text-align:center;">
                            {props['score_48h']:.0f}/100
                        </td>
                    </tr>
                </table>
                <div style="margin-top:8px;padding:6px;background:{color_now};border-radius:4px;text-align:center;font-weight:bold;">
                    {props['recommendation']}
                </div>
                <div style="margin-top:8px;font-size:11px;">
                    <b>SST:</b> {props['sst']:.1f}°C |
                    <b>Olas:</b> {props['wave_height']:.1f}m |
                    <b>Viento:</b> {props['wind_speed']:.0f}km/h
                </div>
                <div style="margin-top:6px;padding:4px;background:{safety_color};border-radius:3px;text-align:center;color:white;font-size:11px;">
                    {safety_text}
                </div>
                <div style="margin-top:8px;font-size:11px;"><b>Mejor hora:</b> {props['best_time']}</div>
                <div style="margin-top:6px;"><b style="font-size:11px;">Peces y Senuelos:</b>{fish_html}</div>
            </div>
            """

        # Agregar capa GeoJSON con marcadores circulares
        for feature in geojson_data["features"]:
            coords = feature["geometry"]["coordinates"]
            props = feature["properties"]

            # Crear marcador circular geo-referenciado
            folium.CircleMarker(
                location=[coords[1], coords[0]],  # [lat, lon]
                radius=10 + (props["score"] / 10),
                color="#333333",
                weight=2,
                fill=True,
                fillColor=props["color"],
                fillOpacity=0.85,
                popup=folium.Popup(create_popup(feature), max_width=280),
                tooltip=f"{props['name']}: {props['score']:.0f}/100 - {props['recommendation']}"
            ).add_to(self.map)

        # Panel de informacion (fuera del mapa, posicion absoluta)
        self._add_info_panel(astro_data, weather_data, score_df)

        # Leyenda
        self._add_legend()

        # Control de capas
        folium.LayerControl(collapsed=False).add_to(self.map)

        return self.map

    def _add_info_panel(self, astro_data: Dict, weather_data: Dict, score_df: pd.DataFrame):
        """Agrega panel de informacion."""
        now = datetime.now()

        # Datos astronomicos
        if astro_data:
            lunar = astro_data.get("lunar_phase_name", "N/A")
            sunrise = str(astro_data.get("sunrise", "06:00"))[:5]
            sunset = str(astro_data.get("sunset", "18:00"))[:5]
        else:
            lunar = "N/A"
            sunrise = "06:00"
            sunset = "18:00"

        # Datos meteorologicos
        if weather_data:
            wave = weather_data.get("wave_height", "N/A")
            wind = weather_data.get("wind_speed", "N/A")
        else:
            wave = "N/A"
            wind = "N/A"

        # Estadisticas
        max_score = score_df["score"].max() if not score_df.empty else 0
        avg_score = score_df["score"].mean() if not score_df.empty else 0
        good_zones = len(score_df[score_df["score"] >= 60]) if not score_df.empty else 0

        html = f"""
        <div style="
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            max-width: 280px;
        ">
            <h3 style="margin: 0 0 10px 0; color: #1a5f7a; border-bottom: 2px solid #1a5f7a; padding-bottom: 5px;">
                Fishing Predictor
            </h3>

            <div style="font-size: 12px; color: #666; margin-bottom: 10px;">
                {now.strftime('%d/%m/%Y %H:%M')}
            </div>

            <div style="background: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <b>Condiciones Actuales</b><br>
                Olas: {wave}m | Viento: {wind} km/h<br>
                Luna: {lunar}
            </div>

            <div style="background: #f5fff5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <b>Horarios Optimos</b><br>
                Amanecer: {sunrise}<br>
                Atardecer: {sunset}
            </div>

            <div style="background: #fffff0; padding: 10px; border-radius: 5px;">
                <b>Resumen</b><br>
                Mejor score: {max_score:.0f}/100<br>
                Promedio: {avg_score:.0f}/100<br>
                Zonas buenas: {good_zones}
            </div>

            <div style="margin-top: 10px; font-size: 11px; color: #999; text-align: center;">
                Haz clic en los puntos para ver detalles
            </div>
        </div>
        """
        self.map.get_root().html.add_child(folium.Element(html))

    def _add_legend(self):
        """Agrega leyenda de colores."""
        html = """
        <div style="
            position: fixed;
            bottom: 30px;
            left: 10px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            font-size: 12px;
        ">
            <b style="display: block; margin-bottom: 8px;">Score de Pesca</b>
            <div><span style="background: #00ff00; padding: 2px 10px; border-radius: 3px;">75-100</span> Excelente</div>
            <div><span style="background: #90ee90; padding: 2px 10px; border-radius: 3px;">60-74</span> Muy Bueno</div>
            <div><span style="background: #ffff00; padding: 2px 10px; border-radius: 3px;">45-59</span> Bueno</div>
            <div><span style="background: #ffa500; padding: 2px 10px; border-radius: 3px;">30-44</span> Regular</div>
            <div><span style="background: #ff4444; padding: 2px 10px; color: white; border-radius: 3px;">&lt;30</span> No recomendado</div>
        </div>
        """
        self.map.get_root().html.add_child(folium.Element(html))

    def save(self, filename: Optional[str] = None) -> str:
        """Guarda el mapa como archivo HTML."""
        if self.map is None:
            raise ValueError("El mapa no ha sido construido. Llama build() primero.")

        if filename is None:
            filename = DEFAULT_OUTPUT_FILE

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        filepath = os.path.join(OUTPUT_DIR, filename)
        self.map.save(filepath)

        return filepath
