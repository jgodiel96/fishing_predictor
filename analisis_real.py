#!/usr/bin/env python3
"""
Análisis de Pesca con Datos REALES de OpenStreetMap.
Usa la línea costera descargada de OSM para garantizar precisión.
"""

import sys
import os
import json
import folium
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class RealCoastlineAnalyzer:
    """
    Analizador que usa datos REALES de línea costera.
    """

    def __init__(self):
        self.coastline_points: List[Tuple[float, float]] = []
        self.sampled_points: List[Dict] = []
        self.fish_zones: List[Dict] = []

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

    def generate_fish_zones(self, num_zones: int = 5, distance_range: Tuple[float, float] = (150, 500)):
        """
        Genera zonas de peces basadas en la geometría de la costa.
        Las zonas se colocan en el mar, perpendicular a la costa.
        """
        if not self.sampled_points:
            self.sample_coastline()

        np.random.seed(int(datetime.now().hour))

        # Seleccionar puntos aleatorios de la costa
        zone_indices = np.random.choice(
            len(self.sampled_points),
            min(num_zones, len(self.sampled_points)),
            replace=False
        )

        self.fish_zones = []

        for i, idx in enumerate(zone_indices):
            point = self.sampled_points[idx]
            lat, lon = point['lat'], point['lon']
            bearing = point['bearing_to_sea']

            # Distancia al mar
            distance = np.random.uniform(distance_range[0], distance_range[1])

            # Calcular posición de la zona
            rad = np.radians(bearing)
            zone_lat = lat + distance / 111000 * np.cos(rad)
            zone_lon = lon + distance / (111000 * np.cos(np.radians(lat))) * np.sin(rad)

            # Dirección de movimiento (con tendencia hacia la costa)
            movement = (bearing + 180 + np.random.normal(0, 40)) % 360

            self.fish_zones.append({
                'id': i + 1,
                'lat': zone_lat,
                'lon': zone_lon,
                'radius': np.random.uniform(100, 300),
                'intensity': np.random.uniform(0.5, 0.95),
                'movement_direction': movement,
                'cause': 'thermal_activity'
            })

        return self.fish_zones

    def analyze(self):
        """
        Calcula scores para cada punto de la orilla.
        """
        if not self.fish_zones:
            self.generate_fish_zones()

        for point in self.sampled_points:
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
        fg_fish = folium.FeatureGroup(name="Zonas de Peces")
        for zone in self.fish_zones:
            # Círculo de la zona
            folium.Circle(
                location=[zone['lat'], zone['lon']],
                radius=zone['radius'],
                color="#0066FF",
                weight=2,
                fill=True,
                fillColor="#0066FF",
                fillOpacity=0.25,
                popup=f"Zona #{zone['id']}<br>Intensidad: {zone['intensity']:.0%}"
            ).add_to(fg_fish)

            # Centro
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=5,
                color="#0066FF",
                fill=True,
                fillColor="#00FFFF",
                fillOpacity=0.9
            ).add_to(fg_fish)

            # Flecha de movimiento
            rad = np.radians(zone['movement_direction'])
            arrow_len = 150
            end_lat = zone['lat'] + arrow_len / 111000 * np.cos(rad)
            end_lon = zone['lon'] + arrow_len / (111000 * np.cos(np.radians(zone['lat']))) * np.sin(rad)
            folium.PolyLine(
                locations=[[zone['lat'], zone['lon']], [end_lat, end_lon]],
                color="#FF00FF",
                weight=3,
                opacity=0.8,
                dash_array="5,5"
            ).add_to(fg_fish)

        fg_fish.add_to(m)

        # === PUNTOS DE PESCA ===
        fg_spots = folium.FeatureGroup(name="Puntos de Pesca")

        def get_color(score, is_best):
            if is_best:
                return "#FF0000"
            if score >= 70:
                return "#00FF00"
            if score >= 50:
                return "#90EE90"
            if score >= 30:
                return "#FFFF00"
            return "#FFA500"

        for i, point in enumerate(self.sampled_points):
            is_best = (i == 0)
            color = get_color(point['score'], is_best)

            popup = f"""
            <b>{'⭐ MEJOR SPOT' if is_best else f"Spot #{point['id']}"}</b><br>
            Score: {point['score']:.1f}/100<br>
            Distancia peces: {point['distance_to_fish']:.0f}m<br>
            Dirección: {point['direction_to_fish']:.0f}°<br>
            <br>
            Lat: {point['lat']:.6f}<br>
            Lon: {point['lon']:.6f}
            """

            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=10 if is_best else 7,
                color="#000",
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=popup,
                tooltip=f"{'⭐ ' if is_best else ''}Score: {point['score']:.0f}"
            ).add_to(fg_spots)

            # Etiqueta top 3
            if i < 3:
                folium.Marker(
                    location=[point['lat'], point['lon']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:12px;font-weight:bold;color:white;text-shadow:1px 1px 3px black;">#{i+1}</div>',
                        icon_size=(25, 25),
                        icon_anchor=(12, 12)
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

        # Leyenda
        legend = '''
        <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
                    background:white;padding:12px;border-radius:8px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:11px;">
            <b>🎣 Análisis con Costa REAL</b><br><br>
            <span style="color:yellow;">━</span> Línea costera OSM<br>
            <span style="color:#FF0000;">●</span> Mejor spot<br>
            <span style="color:#00FF00;">●</span> Score 70+<br>
            <span style="color:#90EE90;">●</span> Score 50-70<br>
            <span style="color:#0066FF;">◯</span> Zona de peces<br>
            <span style="color:#FF00FF;">--</span> Movimiento peces
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend))

        folium.LayerControl().add_to(m)
        return m


def main():
    print("=" * 70)
    print("   🎣 ANÁLISIS CON LÍNEA COSTERA REAL (OpenStreetMap)")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print()

    # Cargar costa real
    analyzer = RealCoastlineAnalyzer()

    coastline_file = "data/cache/coastline_real_osm.geojson"
    if not os.path.exists(coastline_file):
        print("ERROR: Primero descarga la costa con el comando anterior")
        return

    print("[1/4] Cargando línea costera REAL de OSM...")
    num_points = analyzer.load_coastline(coastline_file)
    print(f"      Puntos de costa: {num_points}")

    # Mostrar rango
    lats = [p[0] for p in analyzer.coastline_points]
    lons = [p[1] for p in analyzer.coastline_points]
    print(f"      Rango lat: {min(lats):.4f} a {max(lats):.4f}")
    print(f"      Rango lon: {min(lons):.4f} a {max(lons):.4f}")

    # Muestrear
    print()
    print("[2/4] Muestreando puntos de pesca...")
    samples = analyzer.sample_coastline(num_points=35)
    print(f"      Puntos muestreados: {len(samples)}")

    # Generar zonas de peces
    print()
    print("[3/4] Generando zonas de actividad de peces...")
    zones = analyzer.generate_fish_zones(num_zones=6)
    print(f"      Zonas generadas: {len(zones)}")

    # Analizar
    print()
    print("[4/4] Analizando y optimizando...")
    results = analyzer.analyze()

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
