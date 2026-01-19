#!/usr/bin/env python3
"""
Análisis de Pesca con Datos REALES de OpenStreetMap.
Usa la línea costera descargada de OSM para garantizar precisión.
Incluye datos meteorológicos y cálculos solunares.
"""

import sys
import os
import json
import folium
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.weather_solunar import get_fishing_conditions, WeatherFetcher, SolunarCalculator
from core.marine_data import FishZonePredictor, MarineDataFetcher, CurrentVector, SSTHistory


class RealCoastlineAnalyzer:
    """
    Analizador que usa datos REALES de línea costera.
    """

    def __init__(self):
        self.coastline_points: List[Tuple[float, float]] = []
        self.sampled_points: List[Dict] = []
        self.fish_zones: List[Dict] = []
        # Nuevos: datos marinos para visualización
        self.marine_fetcher: MarineDataFetcher = None
        self.flow_lines: List[List[Tuple[float, float]]] = []
        self.current_vectors: List[CurrentVector] = []
        self.sst_history: List[SSTHistory] = []

    def load_coastline(self, geojson_path: str) -> int:
        """
        Carga línea costera desde archivo GeoJSON.
        """
        with open(geojson_path, 'r') as f:
            data = json.load(f)

        self.coastline_points = []

        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'LineString':
                for coord in geom.get('coordinates', []):
                    lon, lat = coord[0], coord[1]
                    self.coastline_points.append((lat, lon))
            elif geom.get('type') == 'MultiLineString':
                for line in geom.get('coordinates', []):
                    for coord in line:
                        lon, lat = coord[0], coord[1]
                        self.coastline_points.append((lat, lon))

        # Ordenar por latitud (sur a norte)
        self.coastline_points.sort(key=lambda x: x[0])

        # Eliminar duplicados cercanos
        self.coastline_points = self._remove_close_duplicates(self.coastline_points)

        return len(self.coastline_points)

    def _remove_close_duplicates(
        self,
        points: List[Tuple[float, float]],
        threshold_m: float = 50
    ) -> List[Tuple[float, float]]:
        """Elimina puntos muy cercanos entre sí."""
        if not points:
            return []

        result = [points[0]]
        for lat, lon in points[1:]:
            last_lat, last_lon = result[-1]
            dist = self._distance_m(lat, lon, last_lat, last_lon)
            if dist > threshold_m:
                result.append((lat, lon))

        return result

    def _distance_m(self, lat1, lon1, lat2, lon2) -> float:
        """Distancia en metros entre dos puntos."""
        dlat = (lat2 - lat1) * 111000
        dlon = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(dlat**2 + dlon**2)

    def _bearing(self, lat1, lon1, lat2, lon2) -> float:
        """Calcula bearing de p1 a p2."""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return np.degrees(np.arctan2(dlon, dlat)) % 360

    def _perpendicular_to_sea(self, idx: int) -> float:
        """
        Calcula dirección perpendicular hacia el mar en un punto.
        """
        if len(self.coastline_points) < 2:
            return 270  # Default oeste

        # Usar puntos vecinos
        if idx == 0:
            p1, p2 = self.coastline_points[0], self.coastline_points[1]
        elif idx >= len(self.coastline_points) - 1:
            p1, p2 = self.coastline_points[-2], self.coastline_points[-1]
        else:
            p1 = self.coastline_points[idx - 1]
            p2 = self.coastline_points[idx + 1]

        # Dirección de la costa
        coast_bearing = self._bearing(p1[0], p1[1], p2[0], p2[1])

        # Perpendicular (el mar está al oeste en Peru)
        perp1 = (coast_bearing + 90) % 360
        perp2 = (coast_bearing - 90) % 360

        # Elegir la que apunta más hacia el oeste (longitud más negativa)
        # En la costa peruana, el mar tiene longitudes menores
        lat, lon = self.coastline_points[idx]

        # Test: ¿cuál perpendicular lleva a longitud menor?
        test_dist = 100  # metros
        for perp in [perp1, perp2]:
            rad = np.radians(perp)
            test_lon = lon + test_dist / (111000 * np.cos(np.radians(lat))) * np.sin(rad)
            if test_lon < lon:
                return perp

        return perp1

    def sample_coastline(self, num_points: int = 30) -> List[Dict]:
        """
        Muestrea puntos equidistantes a lo largo de la costa.
        """
        if len(self.coastline_points) <= num_points:
            indices = range(len(self.coastline_points))
        else:
            indices = np.linspace(0, len(self.coastline_points) - 1, num_points, dtype=int)

        self.sampled_points = []

        for i, idx in enumerate(indices):
            lat, lon = self.coastline_points[idx]
            bearing_to_sea = self._perpendicular_to_sea(idx)

            self.sampled_points.append({
                'id': i + 1,
                'lat': lat,
                'lon': lon,
                'bearing_to_sea': bearing_to_sea,
                'score': 0,
                'distance_to_fish': 0,
                'direction_to_fish': 0
            })

        return self.sampled_points

    def generate_fish_zones(self, num_zones: int = 6, use_real_data: bool = True):
        """
        Genera zonas de peces basadas en datos reales.

        Args:
            num_zones: número de zonas a generar
            use_real_data: True para usar SST real, False para simulación
        """
        if not self.sampled_points:
            self.sample_coastline()

        if use_real_data:
            # Usar predictor con datos marinos reales
            predictor = FishZonePredictor()
            self.fish_zones = predictor.predict_zones(
                self.coastline_points,
                num_zones=num_zones
            )
            # Guardar referencia al fetcher para datos de corrientes
            self.marine_fetcher = predictor.marine_fetcher
        else:
            # Fallback: usar zonas históricas conocidas de IMARPE
            self._use_historical_imarpe_zones(num_zones)

        return self.fish_zones

    def generate_flow_data(self):
        """
        Genera datos de flujo siguiendo la curvatura de la costa.
        Muestrea puntos perpendiculares a la línea costera.
        """
        if not self.coastline_points:
            return

        print("[INFO] Generando muestreo paralelo a la costa...")

        # Muestrear cada N puntos de la costa
        coast_sample_step = max(1, len(self.coastline_points) // 25)
        coast_samples = self.coastline_points[::coast_sample_step]

        # Para cada punto de costa, generar puntos mar adentro (perpendiculares)
        offshore_distances_km = [3, 8, 15, 25]  # km hacia el mar
        current_points = []

        for i, (lat, lon) in enumerate(coast_samples):
            # Calcular dirección perpendicular al mar
            bearing_to_sea = self._perpendicular_to_sea(
                self.coastline_points.index((lat, lon))
                if (lat, lon) in self.coastline_points
                else i * coast_sample_step
            )

            # Generar puntos a diferentes distancias mar adentro
            for dist_km in offshore_distances_km:
                rad = np.radians(bearing_to_sea)
                offset_lat = dist_km / 111.0 * np.cos(rad)
                offset_lon = dist_km / 111.0 * np.sin(rad) / np.cos(np.radians(lat))
                current_points.append((lat + offset_lat, lon + offset_lon))

        print(f"[INFO] Muestreando corrientes en {len(current_points)} puntos (paralelo a costa)...")

        if not self.marine_fetcher:
            self.marine_fetcher = MarineDataFetcher()

        self.current_vectors = self.marine_fetcher.fetch_current_vectors(current_points)
        self.flow_lines = self.marine_fetcher.get_flow_lines(num_steps=5, step_km=4.0)
        print(f"[OK] {len(self.current_vectors)} vectores, {len(self.flow_lines)} líneas de flujo")

        # Historial SST
        print("[INFO] Obteniendo historial SST (7 días)...")
        self.sst_history = [
            h for lat, lon in current_points[::15][:5]
            if (h := self.marine_fetcher.fetch_sst_history(lat, lon, days=7))
        ]

        if self.sst_history:
            avg_trend = np.mean([h.trend for h in self.sst_history])
            print(f"[OK] SST {len(self.sst_history)} puntos | Tendencia: {abs(avg_trend):.4f}°C/h ({'calentando' if avg_trend > 0 else 'enfriando'})")

    # Zonas históricas verificadas de IMARPE (Instituto del Mar del Perú)
    IMARPE_HISTORICAL_ZONES = [
        (-17.70, -71.35, "Punta Coles", 1.3),      # Reserva natural, alta biodiversidad
        (-17.78, -71.14, "Pozo Redondo", 1.2),     # Pozas naturales
        (-17.82, -71.10, "Punta Blanca", 1.25),    # Punta rocosa
        (-17.93, -70.99, "Ite", 1.15),             # Zona de surgencia
        (-18.02, -70.93, "Vila Vila", 1.2),        # Rocas con estructura
        (-18.12, -70.86, "Boca del Río", 1.1),     # Desembocadura
    ]

    def _use_historical_imarpe_zones(self, num_zones: int):
        """
        Usa zonas históricas REALES conocidas de IMARPE como respaldo.

        Datos basados en:
        - Reportes del Instituto del Mar del Perú (IMARPE)
        - Conocimiento de pescadores locales documentado
        - Cartas náuticas de zonas de pesca
        """
        print("[INFO] Usando zonas históricas verificadas de IMARPE")

        self.fish_zones = []

        for i, (lat, lon, name, intensity_factor) in enumerate(self.IMARPE_HISTORICAL_ZONES[:num_zones]):
            # Mover ligeramente hacia el mar (~2km)
            zone_lon = lon - 0.02

            self.fish_zones.append({
                'id': i + 1,
                'lat': lat,
                'lon': zone_lon,
                'radius': 250,
                'intensity': min(1.0, 0.6 * intensity_factor),
                'movement_direction': 90 + np.random.normal(0, 15),  # Hacia la costa
                'cause': f'historical_imarpe ({name})',
                'sst': 17.5,  # Climatología promedio
                'sst_source': 'climatology_imarpe'
            })

    # Especies por tipo de zona
    SPECIES_BY_SUBSTRATE = {
        "roca": ["Cabrilla", "Pintadilla", "Robalo", "Cherlo"],
        "arena": ["Corvina", "Lenguado", "Pejerrey", "Chita"],
        "mixto": ["Corvina", "Cabrilla", "Robalo", "Pejerrey"]
    }

    SPECIES_LURES = {
        "Cabrilla": "Grubs 3\", jigs 15-25g",
        "Pintadilla": "Vinilos pequeños, jigs ligeros",
        "Robalo": "Poppers 12cm, minnows, walk-the-dog",
        "Corvina": "Jigs metálicos 30-50g, vinilos paddle",
        "Lenguado": "Vinilos paddle tail, jigs de fondo",
        "Pejerrey": "Cucharillas pequeñas, sabikis",
        "Chita": "Carnada natural, jigs pequeños",
        "Cherlo": "Grubs, jigs de roca"
    }

    def get_species_for_point(self, point: Dict) -> List[Dict]:
        """Obtiene especies recomendadas para un punto."""
        # Determinar tipo de sustrato basado en características
        # Por ahora usamos una heurística simple
        lat = point['lat']

        # Zonas rocosas típicas (puntas y acantilados)
        rocky_zones = [(-17.7, -17.65), (-17.82, -17.78), (-18.0, -17.95)]
        sandy_zones = [(-18.15, -18.05), (-17.93, -17.88)]

        substrate = "mixto"
        for (lat_min, lat_max) in rocky_zones:
            if lat_min <= lat <= lat_max:
                substrate = "roca"
                break
        for (lat_min, lat_max) in sandy_zones:
            if lat_min <= lat <= lat_max:
                substrate = "arena"
                break

        species_names = self.SPECIES_BY_SUBSTRATE.get(substrate, ["Corvina", "Cabrilla"])

        species = []
        for name in species_names[:3]:
            species.append({
                "name": name,
                "lure": self.SPECIES_LURES.get(name, "Varios señuelos"),
                "substrate": substrate
            })

        return species

    def analyze(self):
        """
        Calcula scores para cada punto de la orilla.
        """
        if not self.fish_zones:
            self.generate_fish_zones()

        for point in self.sampled_points:
            # Agregar especies recomendadas
            point['species'] = self.get_species_for_point(point)
            best_score = 0
            best_dist = float('inf')
            best_dir = 0

            for zone in self.fish_zones:
                # Distancia desde orilla a zona
                dist = self._distance_m(
                    point['lat'], point['lon'],
                    zone['lat'], zone['lon']
                )

                # Dirección hacia la zona
                direction = self._bearing(
                    point['lat'], point['lon'],
                    zone['lat'], zone['lon']
                )

                # Factor de movimiento (peces viniendo hacia la costa = mejor)
                ideal_movement = (direction + 180) % 360
                angle_diff = abs(zone['movement_direction'] - ideal_movement)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                movement_factor = 1.0 + (1.0 - angle_diff / 180) * 0.4

                # Score
                dist_score = max(0, 100 - dist / 8)
                intensity_score = zone['intensity'] * 40
                zone_score = (dist_score * 0.6 + intensity_score * 0.4) * movement_factor

                if zone_score > best_score:
                    best_score = zone_score
                    best_dist = dist
                    best_dir = direction

            point['score'] = min(100, best_score)
            point['distance_to_fish'] = best_dist
            point['direction_to_fish'] = best_dir

        # Ordenar por score
        self.sampled_points.sort(key=lambda p: p['score'], reverse=True)

        return self.sampled_points

    def create_map(self, zoom: int = 10) -> folium.Map:
        """
        Crea mapa con la línea de costa REAL y el análisis.
        """
        # Centro
        if self.coastline_points:
            lats = [p[0] for p in self.coastline_points]
            lons = [p[1] for p in self.coastline_points]
            center = (np.mean(lats), np.mean(lons))
        else:
            center = (-17.9, -71.0)

        m = folium.Map(location=list(center), zoom_start=zoom, tiles=None)

        # Capas
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satélite"
        ).add_to(m)
        folium.TileLayer(tiles="OpenStreetMap", name="Calles").add_to(m)

        # === LÍNEA COSTERA REAL ===
        if self.coastline_points:
            folium.PolyLine(
                locations=self.coastline_points,
                color="yellow",
                weight=3,
                opacity=0.9,
                popup=f"Línea costera REAL ({len(self.coastline_points)} puntos)"
            ).add_to(m)

        # === ZONAS DE PECES ===
        # Paleta CYAN/TURQUESA (exclusiva para zonas de peces)
        fg_fish = folium.FeatureGroup(name="Zonas de Peces")
        for zone in self.fish_zones:
            # Color por intensidad (más intenso = más peces)
            intensity = zone.get('intensity', 0.5)
            if intensity >= 0.8:
                zone_color = "#00CED1"  # Turquesa oscuro (alta)
                zone_fill = "#00FFFF"
            elif intensity >= 0.6:
                zone_color = "#20B2AA"  # Verde mar claro (media-alta)
                zone_fill = "#40E0D0"
            else:
                zone_color = "#48D1CC"  # Turquesa medio (media)
                zone_fill = "#7FFFD4"

            # Círculo de la zona (más visible)
            folium.Circle(
                location=[zone['lat'], zone['lon']],
                radius=zone['radius'] * 1.5,  # Radio aumentado
                color=zone_color,
                weight=3,
                fill=True,
                fillColor=zone_fill,
                fillOpacity=0.35,
                popup=f"""
                <div style="font-family:Arial;">
                <b>Zona de Peces #{zone['id']}</b><br>
                <hr style="margin:4px 0;">
                Intensidad: <b>{zone['intensity']:.0%}</b><br>
                Causa: {zone.get('cause', 'N/A')}<br>
                SST: {zone.get('sst', 'N/A')}°C
                </div>
                """
            ).add_to(fg_fish)

            # Centro marcado
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=8,
                color="#000",
                weight=2,
                fill=True,
                fillColor=zone_color,
                fillOpacity=1.0
            ).add_to(fg_fish)

            # Flecha de movimiento (AMARILLO para distinguir)
            rad = np.radians(zone['movement_direction'])
            arrow_len = 250  # Flecha más larga
            end_lat = zone['lat'] + arrow_len / 111000 * np.cos(rad)
            end_lon = zone['lon'] + arrow_len / (111000 * np.cos(np.radians(zone['lat']))) * np.sin(rad)

            # Línea de movimiento
            folium.PolyLine(
                locations=[[zone['lat'], zone['lon']], [end_lat, end_lon]],
                color="#FFD700",  # Amarillo dorado
                weight=4,
                opacity=0.9
            ).add_to(fg_fish)

            # Punta de flecha
            folium.RegularPolygonMarker(
                location=[end_lat, end_lon],
                number_of_sides=3,
                radius=8,
                rotation=zone['movement_direction'] - 90,
                color="#FFD700",
                fill=True,
                fillColor="#FFD700",
                fillOpacity=1.0
            ).add_to(fg_fish)

        fg_fish.add_to(m)

        # === LÍNEAS DE FLUJO DE CORRIENTES ===
        # Paleta VIOLETA/PÚRPURA (exclusiva para corrientes, no se confunde con SST)
        if self.flow_lines:
            fg_flow = folium.FeatureGroup(name="Flujo de Corrientes")

            for i, line in enumerate(self.flow_lines):
                if len(line) > 1:
                    # Color en escala violeta basado en velocidad
                    if i < len(self.current_vectors):
                        speed = self.current_vectors[i].speed
                        # Más rápido = más intenso (violeta oscuro)
                        if speed > 0.3:
                            color = "#4B0082"  # Índigo (muy rápido)
                        elif speed > 0.2:
                            color = "#8B008B"  # Magenta oscuro (rápido)
                        elif speed > 0.1:
                            color = "#9932CC"  # Orquídea (moderado)
                        else:
                            color = "#DA70D6"  # Orquídea claro (lento)
                    else:
                        color = "#DA70D6"

                    # Línea de flujo
                    folium.PolyLine(
                        locations=line,
                        color=color,
                        weight=3,
                        opacity=0.8
                    ).add_to(fg_flow)

                    # Flecha al final para indicar dirección
                    if len(line) >= 2:
                        end_lat, end_lon = line[-1]
                        prev_lat, prev_lon = line[-2]

                        # Calcular ángulo para la flecha
                        angle = np.degrees(np.arctan2(
                            end_lon - prev_lon,
                            end_lat - prev_lat
                        ))

                        # Marcador de flecha al final
                        folium.RegularPolygonMarker(
                            location=[end_lat, end_lon],
                            number_of_sides=3,
                            radius=6,
                            rotation=angle - 90,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.9
                        ).add_to(fg_flow)

            fg_flow.add_to(m)

        # === PUNTOS DE MUESTREO MARINO (SST) ===
        # Paleta TÉRMICA CLÁSICA: azul frío -> verde óptimo -> rojo cálido
        if self.marine_fetcher and self.marine_fetcher.sampled_points:
            fg_marine = folium.FeatureGroup(name="Datos Marinos (SST)")

            for point in self.marine_fetcher.sampled_points:
                # Validar SST
                sst = point.sst if point.sst is not None else 17.0
                if sst <= 14:
                    sst_color = "#0000CD"  # Azul medio (muy frío)
                elif sst <= 16:
                    sst_color = "#1E90FF"  # Azul dodger (frío)
                elif sst <= 17:
                    sst_color = "#00BFFF"  # Azul cielo (fresco)
                elif sst <= 18:
                    sst_color = "#00FA9A"  # Verde primavera (óptimo bajo)
                elif sst <= 19:
                    sst_color = "#00FF00"  # Verde lima (ÓPTIMO)
                elif sst <= 20:
                    sst_color = "#ADFF2F"  # Verde amarillo (óptimo alto)
                elif sst <= 21:
                    sst_color = "#FFD700"  # Oro (cálido)
                elif sst <= 22:
                    sst_color = "#FF8C00"  # Naranja oscuro (muy cálido)
                else:
                    sst_color = "#FF4500"  # Naranja-rojo (caliente)

                wave = point.wave_height if point.wave_height else 0
                curr_spd = point.current_speed if point.current_speed else 0
                curr_dir = point.current_direction if point.current_direction else 0
                popup_text = f"""
                <div style="font-family:Arial;">
                <b>Datos Marinos</b><br>
                🌡️ SST: <b>{sst:.1f}°C</b><br>
                🌊 Olas: {wave:.1f}m<br>
                💨 Corriente: {curr_spd:.2f} m/s<br>
                🧭 Dir: {curr_dir:.0f}°
                </div>
                """

                # Marcador cuadrado para distinguir de otros elementos
                folium.RegularPolygonMarker(
                    location=[point.lat, point.lon],
                    number_of_sides=4,  # Cuadrado
                    radius=7,
                    rotation=45,  # Diamante
                    color="#000",
                    weight=1,
                    fill=True,
                    fillColor=sst_color,
                    fillOpacity=0.8,
                    popup=folium.Popup(popup_text, max_width=220),
                    tooltip=f"SST: {sst:.1f}°C"
                ).add_to(fg_marine)

            fg_marine.add_to(m)

        # === PUNTOS DE PESCA ===
        fg_spots = folium.FeatureGroup(name="Puntos de Pesca")

        def get_color(score):
            """Colores: rojo (malo) -> amarillo (medio) -> verde (bueno)"""
            if score >= 80:
                return "#228B22"  # Verde oscuro - Excelente
            if score >= 60:
                return "#32CD32"  # Verde lima - Bueno
            if score >= 40:
                return "#FFD700"  # Dorado - Regular
            if score >= 20:
                return "#FF8C00"  # Naranja - Bajo
            return "#DC143C"      # Rojo - Pobre

        def get_rating(score):
            if score >= 80: return "Excelente"
            if score >= 60: return "Bueno"
            if score >= 40: return "Regular"
            if score >= 20: return "Bajo"
            return "Pobre"

        for i, point in enumerate(self.sampled_points):
            is_best = (i == 0)
            is_top5 = (i < 5)
            color = "#FF0000" if is_best else get_color(point['score'])

            # Construir popup con información completa
            species_html = ""
            if 'species' in point and point['species'] and is_top5:
                species_html = "<br><b>🐟 Especies:</b><br>"
                for sp in point['species']:
                    species_html += f"• {sp['name']}<br>"
                    species_html += f"  <i>{sp['lure']}</i><br>"

            popup = f"""
            <div style="font-family:Arial;min-width:200px;">
                <h4 style="margin:0;color:{'#FF0000' if is_best else '#1a5f7a'};">
                    {'⭐ MEJOR SPOT' if is_best else f"Spot #{point['id']}"}
                </h4>
                <hr style="margin:5px 0;">
                <b>Score:</b> {point['score']:.1f}/100 ({get_rating(point['score'])})<br>
                <b>Distancia peces:</b> {point['distance_to_fish']:.0f}m<br>
                <b>Dirección:</b> {point['direction_to_fish']:.0f}°<br>
                {species_html}
                <hr style="margin:5px 0;">
                <small>
                    📍 {point['lat']:.5f}, {point['lon']:.5f}
                </small>
            </div>
            """

            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=12 if is_best else (9 if is_top5 else 6),
                color="#000" if is_top5 else "#333",
                weight=2 if is_top5 else 1,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=folium.Popup(popup, max_width=250),
                tooltip=f"{'⭐ ' if is_best else ''}Score: {point['score']:.0f} - {get_rating(point['score'])}"
            ).add_to(fg_spots)

            # Etiqueta top 5
            if i < 5:
                folium.Marker(
                    location=[point['lat'], point['lon']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:14px;font-weight:bold;color:white;text-shadow:2px 2px 4px black;">#{i+1}</div>',
                        icon_size=(30, 30),
                        icon_anchor=(15, 15)
                    )
                ).add_to(fg_spots)

        fg_spots.add_to(m)

        # === VECTORES ===
        fg_vectors = folium.FeatureGroup(name="Vectores a Peces")

        # Vectores desde top 3 a zonas cercanas
        for point in self.sampled_points[:3]:
            for zone in self.fish_zones:
                dist = self._distance_m(point['lat'], point['lon'], zone['lat'], zone['lon'])
                if dist < 800:
                    folium.PolyLine(
                        locations=[[point['lat'], point['lon']], [zone['lat'], zone['lon']]],
                        color="#00FFFF",
                        weight=2,
                        opacity=0.5,
                        dash_array="8,4"
                    ).add_to(fg_vectors)

        fg_vectors.add_to(m)

        # Leyenda actualizada con paletas exclusivas
        legend = '''
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:rgba(255,255,255,0.95);padding:12px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.4);font-size:10px;
                    font-family:Arial;max-height:90vh;overflow-y:auto;width:180px;">
            <b style="font-size:13px;">🎣 Predictor de Pesca</b><br>
            <small style="color:#666;">Datos: Open-Meteo Marine API</small>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b style="color:#333;">🎯 Spots de Pesca:</b><br>
            <span style="color:#FF0000;">●</span> Mejor (#1)<br>
            <span style="color:#228B22;">●</span> Excelente (80+)<br>
            <span style="color:#32CD32;">●</span> Bueno (60-80)<br>
            <span style="color:#FFD700;">●</span> Regular (40-60)<br>
            <span style="color:#DC143C;">●</span> Bajo (&lt;40)<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b style="color:#333;">◆ SST (Diamantes):</b><br>
            <span style="color:#1E90FF;">◆</span> Frío (&lt;16°C)<br>
            <span style="color:#00FF00;">◆</span> Óptimo (17-19°C)<br>
            <span style="color:#FFD700;">◆</span> Cálido (20-21°C)<br>
            <span style="color:#FF4500;">◆</span> Caliente (&gt;22°C)<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b style="color:#333;">🔮 Corrientes (Violeta):</b><br>
            <span style="color:#DA70D6;">→</span> Lento<br>
            <span style="color:#9932CC;">→</span> Moderado<br>
            <span style="color:#4B0082;">→</span> Rápido<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b style="color:#333;">🐟 Zonas Peces (Cyan):</b><br>
            <span style="color:#00CED1;">◯</span> Zona activa<br>
            <span style="color:#FFD700;">→</span> Dirección movimiento<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <span style="color:#CCCC00;">━━</span> Costa OSM
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend))

        folium.LayerControl().add_to(m)
        return m


def main():
    print("=" * 70)
    print("   🎣 ANÁLISIS CON LÍNEA COSTERA REAL (OpenStreetMap)")
    print("   + Clima en tiempo real + Datos solunares")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print()

    # Cargar costa real
    analyzer = RealCoastlineAnalyzer()

    coastline_file = "data/cache/coastline_real_osm.geojson"
    if not os.path.exists(coastline_file):
        print("ERROR: Primero descarga la costa con el comando anterior")
        return

    print("[1/6] Cargando línea costera REAL de OSM...")
    num_points = analyzer.load_coastline(coastline_file)
    print(f"      Puntos de costa: {num_points}")

    # Mostrar rango
    lats = [p[0] for p in analyzer.coastline_points]
    lons = [p[1] for p in analyzer.coastline_points]
    print(f"      Rango lat: {min(lats):.4f} a {max(lats):.4f}")
    print(f"      Rango lon: {min(lons):.4f} a {max(lons):.4f}")

    # Muestrear
    print()
    print("[2/6] Muestreando puntos de pesca...")
    samples = analyzer.sample_coastline(num_points=35)
    print(f"      Puntos muestreados: {len(samples)}")

    # Generar zonas de peces
    print()
    print("[3/6] Generando zonas de actividad de peces...")
    zones = analyzer.generate_fish_zones(num_zones=6)
    print(f"      Zonas generadas: {len(zones)}")

    # Generar datos de flujo de corrientes
    print()
    print("[4/6] Generando datos de flujo de corrientes...")
    analyzer.generate_flow_data()

    # Analizar
    print()
    print("[5/6] Analizando y optimizando...")
    results = analyzer.analyze()

    # Obtener condiciones meteorológicas y solunares
    print()
    print("[6/6] Obteniendo clima y datos solunares...")
    center_lat = np.mean([p[0] for p in analyzer.coastline_points])
    center_lon = np.mean([p[1] for p in analyzer.coastline_points])

    conditions = get_fishing_conditions(center_lat, center_lon)

    weather = conditions.get("weather", {})
    solunar = conditions.get("solunar", {})

    print(f"      🌡️  Temperatura: {weather.get('temperature', 'N/A')}°C")
    print(f"      💨 Viento: {weather.get('wind_speed', 'N/A')} km/h")
    print(f"      🌙 Luna: {solunar.get('moon_phase', 'N/A')} ({solunar.get('moon_illumination', 'N/A')})")
    print(f"      ⭐ Rating del día: {solunar.get('day_rating', 'N/A')}/100")

    if weather.get('warnings'):
        for warn in weather['warnings']:
            print(f"      ⚠️  {warn}")

    # Resultados
    print()
    print("=" * 70)
    print("🏆 TOP 5 MEJORES PUNTOS DE PESCA")
    print("=" * 70)

    for i, point in enumerate(results[:5]):
        emoji = "⭐" if i == 0 else "🎣"
        print(f"\n{emoji} #{i+1} - Score: {point['score']:.1f}/100")
        print(f"   Coordenadas: {point['lat']:.6f}, {point['lon']:.6f}")
        print(f"   Distancia a peces: {point['distance_to_fish']:.0f}m")

        # Mostrar especies
        if 'species' in point and point['species']:
            species_names = [s['name'] for s in point['species']]
            print(f"   Especies: {', '.join(species_names)}")

    # Mejor
    best = results[0]
    print()
    print("=" * 70)
    print("⭐ MEJOR PUNTO RECOMENDADO")
    print("=" * 70)
    print(f"   Lat: {best['lat']:.6f}")
    print(f"   Lon: {best['lon']:.6f}")
    print(f"   Score: {best['score']:.1f}/100")
    print(f"   Peces a {best['distance_to_fish']:.0f}m en dirección {best['direction_to_fish']:.0f}°")

    if 'species' in best and best['species']:
        print()
        print("   🐟 ESPECIES RECOMENDADAS:")
        for sp in best['species']:
            print(f"      • {sp['name']}: {sp['lure']}")

    print()
    print("   ⏰ MEJORES HORARIOS:")
    print(f"      {solunar.get('best_times', 'Amanecer y atardecer')}")
    print(f"      🌅 Amanecer: {solunar.get('sunrise', 'N/A')}")
    print(f"      🌇 Atardecer: {solunar.get('sunset', 'N/A')}")

    # Mapa
    print()
    print("Generando mapa...")
    m = analyzer.create_map(zoom=10)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "analisis_costa_real.html")
    m.save(output_file)

    print()
    print("=" * 70)
    print(f"✅ MAPA GUARDADO: {output_file}")
    print("=" * 70)

    return analyzer


if __name__ == "__main__":
    main()
