#!/usr/bin/env python3
"""
Análisis de un tramo específico de playa para pesca desde orilla.
Divide el tramo en secciones y genera predicciones para cada una.
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetchers.coastal_sweep import CoastalSweep
from data.fetchers import OpenMeteoFetcher, AstronomicalCalculator
from config import THRESHOLDS, get_score_color, get_score_category
from visualization import MapBuilder
import folium


def analizar_tramo(
    lat_inicio: float,
    lon_inicio: float,
    lat_fin: float,
    lon_fin: float,
    num_secciones: int = 10,
    nombre_tramo: str = "Tramo Playa",
    tipo_sustrato: str = "arena"
):
    """
    Analiza un tramo de playa dividiéndolo en secciones.
    """
    print("=" * 60)
    print("   ANÁLISIS DE TRAMO COSTERO - Pesca desde Orilla")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print()
    print(f"Tramo: {nombre_tramo}")
    print(f"Inicio: {lat_inicio}, {lon_inicio}")
    print(f"Fin: {lat_fin}, {lon_fin}")
    print(f"Secciones: {num_secciones}")
    print(f"Sustrato: {tipo_sustrato}")
    print()

    # Crear barrido costero
    print("[1/4] Generando secciones del tramo...")
    sweep = CoastalSweep()
    sections = sweep.create_sweep(
        start_point=(lat_inicio, lon_inicio),
        end_point=(lat_fin, lon_fin),
        num_sections=num_secciones,
        tramo_name=nombre_tramo,
        tipo_sustrato=tipo_sustrato
    )

    for sec in sections:
        print(f"      Sección {sec.id}: {sec.lat_center:.5f}, {sec.lon_center:.5f} ({sec.length_m:.0f}m)")

    # Generar datos oceanográficos
    print("\n[2/4] Generando datos oceanográficos...")
    df = sweep.generate_analysis_data()

    # Obtener condiciones meteorológicas
    print("[3/4] Obteniendo condiciones actuales...")
    lat_centro = (lat_inicio + lat_fin) / 2
    lon_centro = (lon_inicio + lon_fin) / 2

    meteo = OpenMeteoFetcher(use_cache=True)
    weather = meteo.get_current_conditions(lat_centro, lon_centro)

    astro = AstronomicalCalculator(lat=lat_centro, lon=lon_centro)
    astro_data = astro.get_all_astronomical_data(datetime.now(), lat_centro, lon_centro)

    wave_h = weather.get('wave_height', 1.0)
    wind_s = weather.get('wind_speed', 10.0)
    lunar = astro_data.get('lunar_phase_name', 'N/A')

    print(f"      Olas: {wave_h}m | Viento: {wind_s} km/h")
    print(f"      Luna: {lunar}")

    # Calcular scores
    print("\n[4/4] Calculando scores de pesca...")

    # Score de seguridad
    wave_factor = min(wave_h / THRESHOLDS.WAVE_SAFETY_THRESHOLD, 1.0)
    wind_factor = min(wind_s / THRESHOLDS.WIND_SAFETY_THRESHOLD, 1.0)
    safety_score = (1 - (wave_factor + wind_factor) / 2) * 100

    # Score de hora dorada
    golden_score = astro_data.get('golden_hour_score', 50)
    lunar_score = astro_data.get('lunar_score', 50)

    # Calcular score para cada sección
    scores = []
    for _, row in df.iterrows():
        # Score SST (anomalía térmica)
        sst_score = min(100, (row['sst'] - 14.5) / 4 * 100)

        # Score clorofila
        chl_score = min(100, (row['chlorophyll'] - 0.5) / 2 * 100)

        # Score total
        score = (
            sst_score * 0.20 +
            chl_score * 0.25 +
            safety_score * 0.20 +
            golden_score * 0.15 +
            lunar_score * 0.10 +
            50 * 0.10  # Base por estar en zona costera
        )
        score = min(100, max(0, score))
        scores.append(score)

    df['score'] = scores
    df['score_safety'] = safety_score
    df['is_safe'] = safety_score >= 50
    df['category'] = df['score'].apply(get_score_category)
    df['color'] = df['score'].apply(get_score_color)

    # Resultados
    print()
    print("-" * 60)
    print("RESULTADOS POR SECCIÓN:")
    print("-" * 60)

    for _, row in df.iterrows():
        emoji = "🟢" if row['score'] >= 60 else "🟡" if row['score'] >= 40 else "🔴"
        print(f"  {emoji} {row['location']}: {row['score']:.0f}/100")
        print(f"      SST: {row['sst']:.1f}°C | Chl: {row['chlorophyll']:.1f} mg/m³")

    # Mejor sección
    best = df.loc[df['score'].idxmax()]
    print()
    print("-" * 60)
    print(f"MEJOR SECCIÓN: {best['location']}")
    print(f"Score: {best['score']:.0f}/100 - {best['category']}")
    print(f"Coordenadas: {best['latitude']:.5f}, {best['longitude']:.5f}")
    print("-" * 60)

    # Recomendaciones de peces según sustrato
    print()
    print("PECES RECOMENDADOS:")
    if tipo_sustrato == "roca":
        print("  🐟 Cabrilla - Jigs pequeños 15-25g")
        print("  🐟 Pintadilla - Grubs 3\"")
        print("  🐟 Robalo - Poppers 12cm, minnows")
    elif tipo_sustrato == "arena":
        print("  🐟 Corvina - Jigs metálicos 30-50g")
        print("  🐟 Lenguado - Vinilos paddle tail")
        print("  🐟 Pejerrey - Cucharillas pequeñas")
    else:
        print("  🐟 Corvina - Jigs metálicos 30-50g")
        print("  🐟 Cabrilla - Grubs 3\"")
        print("  🐟 Robalo - Poppers, minnows")

    # Generar mapa
    print()
    print("Generando mapa del tramo...")

    # Crear mapa centrado en el tramo
    m = folium.Map(
        location=[lat_centro, lon_centro],
        zoom_start=16,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri"
    )

    # Agregar capa OpenStreetMap
    folium.TileLayer(tiles="OpenStreetMap", name="Mapa").add_to(m)

    # Dibujar línea del tramo
    folium.PolyLine(
        locations=[[lat_inicio, lon_inicio], [lat_fin, lon_fin]],
        color="yellow",
        weight=3,
        opacity=0.8,
        popup=f"Tramo: {nombre_tramo}"
    ).add_to(m)

    # Agregar marcadores para cada sección
    for _, row in df.iterrows():
        color = row['color']
        popup_html = f"""
        <div style="font-family:Arial;width:200px;">
            <h4 style="margin:0;color:#1a5f7a;">{row['location']}</h4>
            <p style="margin:5px 0;"><b>Score:</b> {row['score']:.0f}/100</p>
            <p style="margin:5px 0;"><b>SST:</b> {row['sst']:.1f}°C</p>
            <p style="margin:5px 0;"><b>Clorofila:</b> {row['chlorophyll']:.1f} mg/m³</p>
            <p style="margin:5px 0;"><b>Sustrato:</b> {row['tipo_sustrato']}</p>
            <p style="margin:5px 0;"><b>Longitud:</b> {row['length_m']:.0f}m</p>
        </div>
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=12,
            color="#333",
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['location']}: {row['score']:.0f}/100"
        ).add_to(m)

    # Marcadores de inicio y fin
    folium.Marker(
        [lat_inicio, lon_inicio],
        popup="INICIO del tramo",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        [lat_fin, lon_fin],
        popup="FIN del tramo",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    # Control de capas
    folium.LayerControl().add_to(m)

    # Guardar
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tramo_analisis.html")
    m.save(output_path)

    print()
    print("=" * 60)
    print(f"MAPA GUARDADO: {output_path}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    # Tramo especificado por el usuario
    analizar_tramo(
        lat_inicio=-18.21437,
        lon_inicio=-70.57990,
        lat_fin=-18.2143692,
        lon_fin=-70.5798961,
        num_secciones=10,
        nombre_tramo="Playa Tacna Sur",
        tipo_sustrato="arena"
    )
