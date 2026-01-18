#!/usr/bin/env python3
"""
Análisis Optimizado de Pesca desde Orilla

Este script implementa el modelo geométrico completo:
1. Define la línea de orilla como curva desde puntos de referencia
2. Detecta/simula zonas de actividad de peces
3. Calcula vectores y distancias desde orilla a zonas de peces
4. Recomienda el punto óptimo de pesca

Uso:
    python analisis_optimizado.py
"""

import sys
import os
import json
import folium
from folium import plugins
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fishing_optimizer import FishingOptimizer, CoastlineCurve, FishZoneDetector


def create_analysis_map(
    optimizer: FishingOptimizer,
    center: tuple = None,
    zoom: int = 14
) -> folium.Map:
    """
    Crea mapa interactivo con el análisis completo.
    """
    # Calcular centro si no se proporciona
    if center is None and optimizer.shore_points:
        lats = [p.lat for p in optimizer.shore_points]
        lons = [p.lon for p in optimizer.shore_points]
        center = (np.mean(lats), np.mean(lons))

    # Crear mapa
    m = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=None
    )

    # Capas base
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite"
    ).add_to(m)

    folium.TileLayer(tiles="OpenStreetMap", name="Calles").add_to(m)

    # === CAPA: Línea de Orilla ===
    fg_coast = folium.FeatureGroup(name="Línea de Orilla")

    if optimizer.coastline.curve_points:
        coast_coords = [[p[0], p[1]] for p in optimizer.coastline.curve_points]
        folium.PolyLine(
            locations=coast_coords,
            color="yellow",
            weight=4,
            opacity=0.9,
            popup="Línea de Orilla (puntos de pesca)"
        ).add_to(fg_coast)

    fg_coast.add_to(m)

    # === CAPA: Zonas de Peces ===
    fg_fish = folium.FeatureGroup(name="Zonas de Peces")

    for zone in optimizer.fish_zones:
        # Círculo representando la zona
        folium.Circle(
            location=[zone.center_lat, zone.center_lon],
            radius=zone.radius_m,
            color="#0066FF",
            weight=2,
            fill=True,
            fillColor="#0066FF",
            fillOpacity=0.3 * zone.intensity,
            popup=f"""
            <b>Zona de Peces #{zone.id}</b><br>
            Intensidad: {zone.intensity:.0%}<br>
            Causa: {zone.cause}<br>
            Radio: {zone.radius_m:.0f}m
            """
        ).add_to(fg_fish)

        # Centro de la zona
        folium.CircleMarker(
            location=[zone.center_lat, zone.center_lon],
            radius=6,
            color="#0066FF",
            fill=True,
            fillColor="#00FFFF",
            fillOpacity=0.9
        ).add_to(fg_fish)

        # Flecha de dirección de movimiento
        if zone.movement_direction:
            arrow_length = 200  # metros
            dir_rad = np.radians(zone.movement_direction)
            end_lat = zone.center_lat + arrow_length / 111000 * np.cos(dir_rad)
            end_lon = zone.center_lon + arrow_length / (111000 * np.cos(np.radians(zone.center_lat))) * np.sin(dir_rad)

            folium.PolyLine(
                locations=[
                    [zone.center_lat, zone.center_lon],
                    [end_lat, end_lon]
                ],
                color="#FF00FF",
                weight=3,
                opacity=0.8,
                dash_array="5,10"
            ).add_to(fg_fish)

    fg_fish.add_to(m)

    # === CAPA: Puntos de Pesca ===
    fg_spots = folium.FeatureGroup(name="Puntos de Pesca")

    # Colores según score
    def get_color(score):
        if score >= 70:
            return "#00FF00"  # Verde brillante
        elif score >= 50:
            return "#90EE90"  # Verde claro
        elif score >= 30:
            return "#FFFF00"  # Amarillo
        else:
            return "#FFA500"  # Naranja

    for i, point in enumerate(optimizer.shore_points):
        is_best = (i == 0)
        color = "#FF0000" if is_best else get_color(point.score)
        radius = 12 if is_best else 8

        popup_html = f"""
        <div style="font-family:Arial;width:200px;">
            <h4 style="margin:0;color:{'#FF0000' if is_best else '#1a5f7a'};">
                {'⭐ MEJOR SPOT' if is_best else f'Spot #{point.id}'}
            </h4>
            <hr style="margin:5px 0;">
            <p><b>Score:</b> {point.score:.1f}/100</p>
            <p><b>Distancia a peces:</b> {point.distance_to_fish_zone:.0f}m</p>
            <p><b>Dirección:</b> {point.fish_zone_direction:.0f}°</p>
            <p><b>Coordenadas:</b><br>
               {point.lat:.6f}, {point.lon:.6f}</p>
        </div>
        """

        folium.CircleMarker(
            location=[point.lat, point.lon],
            radius=radius,
            color="#000000",
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.9,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{'⭐ MEJOR: ' if is_best else ''}Score: {point.score:.0f}"
        ).add_to(fg_spots)

        # Etiqueta para top 3
        if i < 3:
            folium.Marker(
                location=[point.lat, point.lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:14px;font-weight:bold;color:white;text-shadow:2px 2px 4px black;">#{i+1}</div>',
                    icon_size=(30, 30),
                    icon_anchor=(15, 15)
                )
            ).add_to(fg_spots)

    fg_spots.add_to(m)

    # === CAPA: Vectores a Zonas de Peces ===
    fg_vectors = folium.FeatureGroup(name="Vectores a Peces")

    # Solo mostrar vectores desde los mejores 3 spots
    for point in optimizer.shore_points[:3]:
        for zone in optimizer.fish_zones:
            # Calcular distancia
            dist = np.sqrt(
                ((zone.center_lat - point.lat) * 111000)**2 +
                ((zone.center_lon - point.lon) * 111000 * np.cos(np.radians(point.lat)))**2
            )

            # Solo mostrar si está en rango razonable
            if dist < 1000:
                opacity = max(0.3, 1.0 - dist / 1000)
                folium.PolyLine(
                    locations=[
                        [point.lat, point.lon],
                        [zone.center_lat, zone.center_lon]
                    ],
                    color="#00FFFF",
                    weight=2,
                    opacity=opacity,
                    dash_array="10,5",
                    popup=f"Distancia: {dist:.0f}m"
                ).add_to(fg_vectors)

    fg_vectors.add_to(m)

    # === LEYENDA ===
    legend_html = '''
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
                background:white;padding:15px;border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3);font-family:Arial;
                font-size:12px;max-width:220px;">
        <h4 style="margin:0 0 10px 0;border-bottom:1px solid #ccc;padding-bottom:5px;">
            🎣 Análisis de Pesca
        </h4>
        <div style="margin-bottom:8px;">
            <b>Línea de Orilla:</b><br>
            <span style="color:yellow;text-shadow:1px 1px 2px black;">━━━</span> Puntos de pesca
        </div>
        <div style="margin-bottom:8px;">
            <b>Puntos de Pesca:</b><br>
            <span style="color:#FF0000;">●</span> Mejor spot<br>
            <span style="color:#00FF00;">●</span> Score 70+<br>
            <span style="color:#90EE90;">●</span> Score 50-70<br>
            <span style="color:#FFFF00;">●</span> Score 30-50
        </div>
        <div style="margin-bottom:8px;">
            <b>Zonas de Peces:</b><br>
            <span style="color:#0066FF;">◯</span> Área de actividad<br>
            <span style="color:#FF00FF;">- -</span> Dirección movimiento
        </div>
        <div>
            <b>Vectores:</b><br>
            <span style="color:#00FFFF;">┄┄</span> Distancia a peces
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Control de capas
    folium.LayerControl().add_to(m)

    return m


def main():
    """
    Análisis principal.
    """
    print("=" * 70)
    print("   🎣 ANÁLISIS OPTIMIZADO DE PESCA DESDE ORILLA")
    print("   Modelo Geométrico: Orilla → Zonas de Peces → Mejor Punto")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print()

    # === 1. DEFINIR LÍNEA DE ORILLA ===
    print("[1/4] Definiendo línea de orilla...")

    # Puntos de referencia de la costa (de sur a norte)
    # Estos definen la curva del límite mar/playa
    reference_points = [
        (-18.2140, -70.5800),   # Zona sur Tacna
        (-18.1205, -70.8430),   # Boca del Río
        (-18.0870, -70.8680),   # Santa Rosa
        (-18.0520, -70.8830),   # Los Palos
        (-18.0180, -70.9120),   # Vila Vila
        (-17.9880, -70.9350),   # Punta Mesa
        (-17.9620, -70.9480),   # Carlepe
        (-17.9320, -70.9680),   # Ite Norte
        (-17.9020, -70.9920),   # Ite Centro
        (-17.8720, -71.0180),   # Ite Sur
        (-17.8420, -71.0480),   # Gentillar
        (-17.8120, -71.0820),   # Punta Blanca
        (-17.7820, -71.1220),   # Pozo Redondo
        (-17.7570, -71.1720),   # Fundición
        (-17.7320, -71.2220),   # Media Luna
        (-17.7020, -71.3320),   # Punta Coles
        (-17.6820, -71.2950),   # Pocoma
        (-17.6420, -71.3400),   # Pozo Lizas
        (-17.6320, -71.3450),   # Ilo Puerto
    ]

    optimizer = FishingOptimizer()
    optimizer.set_coastline(reference_points, num_points=40)

    print(f"      Puntos de referencia: {len(reference_points)}")
    print(f"      Puntos en curva: {len(optimizer.shore_points)}")

    # === 2. DETECTAR ZONAS DE PECES ===
    print()
    print("[2/4] Detectando zonas de actividad de peces...")

    # Por ahora usamos simulación basada en la geometría de la costa
    # En producción, esto usaría datos reales de SST y clorofila
    optimizer.detect_fish_zones(use_simulation=True)

    print(f"      Zonas detectadas: {len(optimizer.fish_zones)}")
    for zone in optimizer.fish_zones:
        print(f"        - Zona {zone.id}: intensidad {zone.intensity:.0%}, "
              f"movimiento {zone.movement_direction:.0f}°")

    # === 3. ANALIZAR Y OPTIMIZAR ===
    print()
    print("[3/4] Calculando distancias y optimizando...")

    results = optimizer.analyze()
    recommendation = optimizer.get_recommendation()

    print(f"      Puntos analizados: {len(results)}")

    # === 4. RESULTADOS ===
    print()
    print("=" * 70)
    print("RESULTADOS DEL ANÁLISIS")
    print("=" * 70)
    print()

    # Top 5 spots
    print("🏆 TOP 5 MEJORES PUNTOS DE PESCA:")
    print("-" * 50)
    for i, point in enumerate(results[:5]):
        emoji = "⭐" if i == 0 else "🎣"
        print(f"  {emoji} #{i+1} - Score: {point.score:.1f}/100")
        print(f"      Coordenadas: {point.lat:.6f}, {point.lon:.6f}")
        print(f"      Distancia a peces: {point.distance_to_fish_zone:.0f}m")
        print(f"      Dirección: {point.fish_zone_direction:.0f}°")
        print()

    # Mejor spot
    best = results[0]
    print("=" * 70)
    print("⭐ RECOMENDACIÓN: MEJOR PUNTO DE PESCA")
    print("=" * 70)
    print(f"  Latitud:  {best.lat:.6f}")
    print(f"  Longitud: {best.lon:.6f}")
    print(f"  Score:    {best.score:.1f}/100")
    print()
    print(f"  📍 Los peces están a {best.distance_to_fish_zone:.0f}m")
    print(f"  🧭 Dirección: {best.fish_zone_direction:.0f}° desde la orilla")
    print()

    # Generar mapa
    print("[4/4] Generando mapa interactivo...")

    m = create_analysis_map(optimizer, zoom=11)

    # Guardar
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "analisis_optimizado.html")
    m.save(output_file)

    print()
    print("=" * 70)
    print(f"✅ MAPA GUARDADO: {output_file}")
    print("=" * 70)

    # Guardar también el JSON con los resultados
    json_file = os.path.join(output_dir, "analisis_resultados.json")
    with open(json_file, 'w') as f:
        json.dump(recommendation, f, indent=2)
    print(f"📊 Resultados JSON: {json_file}")

    return optimizer, recommendation


if __name__ == "__main__":
    optimizer, results = main()
