#!/usr/bin/env python3
"""
Fishing Spot Predictor - Main Entry Point

Sistema de prediccion de zonas optimas para pesca desde orilla
en la costa sur de Peru (Tacna - Ilo).

Uso:
    python main.py
    python main.py --verbose
"""

import argparse
from datetime import datetime
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOCATIONS, MAP_CENTER, OUTPUT_DIR, THRESHOLDS, get_score_color, get_score_category
from data.fetchers import ERDDAPFetcher, OpenMeteoFetcher, AstronomicalCalculator
from visualization import MapBuilder


def calculate_direct_scores(sst_df, chl_df, weather_data, astro_data):
    """
    Calcula scores directamente sobre los puntos costeros.
    Sin interpolacion de grilla - usa las coordenadas originales.
    """
    # Combinar SST y clorofila
    score_df = sst_df.copy()

    # Merge con clorofila por coordenadas cercanas
    if not chl_df.empty:
        score_df = score_df.merge(
            chl_df[['latitude', 'longitude', 'chlorophyll']],
            on=['latitude', 'longitude'],
            how='left'
        )
        score_df['chlorophyll'] = score_df['chlorophyll'].fillna(chl_df['chlorophyll'].mean())

    # Calcular sub-scores

    # 1. Score de SST (frentes termicos - gradiente local)
    sst_mean = score_df['sst'].mean()
    score_df['sst_anomaly'] = abs(score_df['sst'] - sst_mean)
    sst_max_anomaly = score_df['sst_anomaly'].max()
    if sst_max_anomaly > 0:
        score_df['score_sst'] = (score_df['sst_anomaly'] / sst_max_anomaly) * 100
    else:
        score_df['score_sst'] = 50

    # 2. Score de clorofila (productividad)
    if 'chlorophyll' in score_df.columns:
        chl_low = THRESHOLDS.CHL_LOW_THRESHOLD
        chl_high = THRESHOLDS.CHL_HIGH_THRESHOLD
        score_df['score_chlorophyll'] = ((score_df['chlorophyll'] - chl_low) / (chl_high - chl_low)) * 100
        score_df['score_chlorophyll'] = score_df['score_chlorophyll'].clip(0, 100)
    else:
        score_df['score_chlorophyll'] = 50

    # 3. Score de distancia a costa (mas cerca = mejor)
    if 'dist_costa_km' in score_df.columns:
        score_df['score_distance'] = 100 - (score_df['dist_costa_km'] / 3) * 50
        score_df['score_distance'] = score_df['score_distance'].clip(30, 100)
    else:
        score_df['score_distance'] = 70

    # 4. Score de seguridad
    wave_h = weather_data.get('wave_height', 1.0)
    wind_s = weather_data.get('wind_speed', 10.0)
    wave_factor = min(wave_h / THRESHOLDS.WAVE_SAFETY_THRESHOLD, 1.0)
    wind_factor = min(wind_s / THRESHOLDS.WIND_SAFETY_THRESHOLD, 1.0)
    safety_score = (1 - (wave_factor + wind_factor) / 2) * 100
    score_df['score_safety'] = safety_score

    # 5. Score de hora dorada
    golden_score = astro_data.get('golden_hour_score', 50) if astro_data else 50
    score_df['score_golden'] = golden_score

    # 6. Score lunar
    lunar_score = astro_data.get('lunar_score', 50) if astro_data else 50
    score_df['score_lunar'] = lunar_score

    # Calcular score total ponderado
    weights = {
        'sst': 0.25,
        'chlorophyll': 0.25,
        'distance': 0.15,
        'safety': 0.15,
        'golden': 0.10,
        'lunar': 0.10
    }

    score_df['score'] = (
        score_df['score_sst'] * weights['sst'] +
        score_df['score_chlorophyll'] * weights['chlorophyll'] +
        score_df['score_distance'] * weights['distance'] +
        score_df['score_safety'] * weights['safety'] +
        score_df['score_golden'] * weights['golden'] +
        score_df['score_lunar'] * weights['lunar']
    )

    # Normalizar a 0-100
    score_df['score'] = score_df['score'].clip(0, 100)

    # Agregar categoria y color
    score_df['category'] = score_df['score'].apply(get_score_category)
    score_df['color'] = score_df['score'].apply(get_score_color)
    score_df['is_safe'] = safety_score >= 50

    # Renombrar columnas de sub-scores para compatibilidad
    score_df['score_front_proximity'] = score_df['score_sst']
    score_df['score_chlorophyll_score'] = score_df['score_chlorophyll']
    score_df['score_safety_score'] = score_df['score_safety']

    return score_df


def main():
    """Funcion principal del programa."""
    parser = argparse.ArgumentParser(
        description="Fishing Spot Predictor - Prediccion de zonas de pesca"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fishing_map.html",
        help="Nombre del archivo de salida"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostrar informacion detallada"
    )

    args = parser.parse_args()
    target_date = datetime.now()

    print("=" * 60)
    print("   FISHING SPOT PREDICTOR - Costa Sur Peru")
    print("=" * 60)
    print(f"Fecha: {target_date.strftime('%d/%m/%Y %H:%M')}")
    print()

    # Coordenadas de referencia
    ref_lat = MAP_CENTER["lat"]
    ref_lon = MAP_CENTER["lon"]

    # 1. Obtener datos oceanograficos
    print("[1/5] Obteniendo datos oceanograficos...")
    erddap = ERDDAPFetcher(use_cache=True, use_fallback=True)
    sst_df, chl_df = erddap.fetch_all(date=target_date)

    if args.verbose:
        print(f"      SST: {len(sst_df)} puntos")
        print(f"      Clorofila: {len(chl_df)} puntos")

    # 2. Obtener datos meteorologicos
    print("[2/5] Obteniendo condiciones meteorologicas...")
    meteo = OpenMeteoFetcher(use_cache=True)
    weather_conditions = meteo.get_current_conditions(ref_lat, ref_lon)

    wave_h = weather_conditions.get('wave_height', 'N/A')
    wind_s = weather_conditions.get('wind_speed', 'N/A')
    print(f"      Olas: {wave_h}m | Viento: {wind_s} km/h")

    # 3. Calcular datos astronomicos
    print("[3/5] Calculando datos astronomicos...")
    astro = AstronomicalCalculator(lat=ref_lat, lon=ref_lon)
    astro_data = astro.get_all_astronomical_data(target_date, ref_lat, ref_lon)

    lunar = astro_data.get('lunar_phase_name', 'N/A')
    sunrise = str(astro_data.get('sunrise', ''))[:16]
    sunset = str(astro_data.get('sunset', ''))[:16]
    golden = astro_data.get('golden_hour_score', 0)

    print(f"      Luna: {lunar}")
    print(f"      Amanecer: {sunrise} | Atardecer: {sunset}")

    # 4. Calcular scores directamente sobre puntos costeros
    print("[4/5] Calculando scores de pesca...")
    score_df = calculate_direct_scores(sst_df, chl_df, weather_conditions, astro_data)

    # Estadisticas
    max_score = score_df["score"].max()
    avg_score = score_df["score"].mean()
    good_zones = len(score_df[score_df["score"] >= 60])
    excellent_zones = len(score_df[score_df["score"] >= 75])

    print()
    print("-" * 40)
    print("RESULTADOS:")
    print("-" * 40)
    print(f"  Mejor score:     {max_score:.0f}/100")
    print(f"  Score promedio:  {avg_score:.0f}/100")
    print(f"  Zonas buenas:    {good_zones}")
    print(f"  Zonas excelentes: {excellent_zones}")

    # TOP 5 spots
    top_5 = score_df.nlargest(5, "score")
    print()
    print("TOP 5 MEJORES SPOTS:")
    for i, (_, spot) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. Lat: {spot['latitude']:.4f}, Lon: {spot['longitude']:.4f} -> Score: {spot['score']:.0f}")

    # Recomendacion de horario
    print()
    print("-" * 40)
    print("RECOMENDACION DE HORARIO:")
    print("-" * 40)

    if golden > 50:
        print("  *** AHORA ES BUEN MOMENTO (Hora dorada) ***")
    else:
        print(f"  Mejores horas para pescar:")
        print(f"    - Amanecer: {sunrise} (+/- 1 hora)")
        print(f"    - Atardecer: {sunset} (+/- 1 hora)")

    if lunar in ["Luna Llena", "Luna Nueva"]:
        print(f"  {lunar}: Actividad de peces ALTA")
    elif "Cuarto" in lunar:
        print(f"  {lunar}: Actividad de peces MODERADA")

    # Seguridad
    print()
    if wave_h != 'N/A' and float(wave_h) > 2.0:
        print("  PRECAUCION: Olas altas, considerar seguridad")
    if wind_s != 'N/A' and float(wind_s) > 25:
        print("  PRECAUCION: Viento fuerte")

    # 5. Generar mapa
    print()
    print("[5/5] Generando mapa interactivo...")

    map_builder = MapBuilder()
    fishing_map = map_builder.build(
        score_df=score_df,
        predictions=None,
        weather_data=weather_conditions,
        astro_data=astro_data
    )

    output_path = map_builder.save(args.output)

    print()
    print("=" * 60)
    print(f"MAPA GUARDADO: {output_path}")
    print("=" * 60)
    print()
    print("Instrucciones:")
    print("  1. Abre el archivo HTML en tu navegador")
    print("  2. Pasa el mouse sobre los puntos para ver el score")
    print("  3. Haz clic en un punto para ver detalles completos:")
    print("     - Prediccion a 0h, 24h, 48h")
    print("     - Peces probables y senuelos recomendados")
    print("     - Condiciones de seguridad")
    print()


if __name__ == "__main__":
    main()
