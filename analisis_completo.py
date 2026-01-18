#!/usr/bin/env python3
"""
Analisis completo de pesca desde orilla.
Sistema robusto con transectos y prediccion de movimiento de peces.

Uso:
    python analisis_completo.py                    # Analiza toda la costa
    python analisis_completo.py --punto LAT LON   # Analiza punto especifico
    python analisis_completo.py --tramo LAT1 LON1 LAT2 LON2  # Analiza tramo
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Optional, Dict, List

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.coastline import CoastlineModel, SubstrateType
from core.transects import TransectAnalyzer, Transect
from core.fish_movement import FishMovementPredictor, MovementTrend
from core.scoring import ScoringEngine, FishingScore
from core.visualization import FishingMapBuilder

# Importar fetchers para datos meteorologicos
try:
    from data.fetchers import OpenMeteoFetcher, AstronomicalCalculator
    HAS_FETCHERS = True
except ImportError:
    HAS_FETCHERS = False
    print("[AVISO] No se encontraron fetchers de datos. Usando valores por defecto.")


def get_weather_data(lat: float, lon: float) -> Dict:
    """Obtiene datos meteorologicos actuales."""
    if not HAS_FETCHERS:
        return {
            'wave_height': 1.0,
            'wind_speed': 12.0,
            'wind_direction': 180,
            'temperature': 18.0
        }

    try:
        meteo = OpenMeteoFetcher(use_cache=True)
        return meteo.get_current_conditions(lat, lon)
    except Exception as e:
        print(f"[AVISO] Error obteniendo datos meteo: {e}")
        return {
            'wave_height': 1.0,
            'wind_speed': 12.0,
            'wind_direction': 180,
            'temperature': 18.0
        }


def get_astro_data(lat: float, lon: float) -> Dict:
    """Obtiene datos astronomicos actuales."""
    if not HAS_FETCHERS:
        return {
            'lunar_phase_name': 'Cuarto creciente',
            'lunar_score': 65,
            'golden_hour_score': 70
        }

    try:
        astro = AstronomicalCalculator(lat=lat, lon=lon)
        return astro.get_all_astronomical_data(datetime.now(), lat, lon)
    except Exception as e:
        print(f"[AVISO] Error obteniendo datos astro: {e}")
        return {
            'lunar_phase_name': 'N/A',
            'lunar_score': 50,
            'golden_hour_score': 50
        }


def analizar_costa_completa(output_dir: str = None) -> str:
    """
    Analiza toda la costa de Tacna a Ilo.

    Returns:
        Ruta del mapa generado
    """
    print("=" * 65)
    print("   ANALISIS COMPLETO DE COSTA - Pesca desde Orilla")
    print("   Sistema de Transectos con Prediccion de Movimiento")
    print("=" * 65)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print()

    # 1. Cargar modelo de costa
    print("[1/6] Cargando modelo de linea costera...")
    coastline = CoastlineModel()
    print(f"      {len(coastline.segments)} segmentos de costa cargados")

    # 2. Crear transectos para cada segmento
    print("\n[2/6] Generando transectos perpendiculares...")
    analyzer = TransectAnalyzer(coastline)

    for segment in coastline.segments:
        analyzer.create_transects_for_segment(
            segment=segment,
            num_transects=3,
            points_per_transect=5,
            max_distance_m=400
        )

    print(f"      {len(analyzer.transects)} transectos generados")

    # 3. Poblar datos oceanograficos
    print("\n[3/6] Obteniendo datos oceanograficos...")
    analyzer.populate_oceanographic_data()

    # 4. Predecir movimiento de peces
    print("\n[4/6] Prediciendo movimiento de peces...")
    predictor = FishMovementPredictor()
    hour = datetime.now().hour
    movement_vectors = predictor.predict_movement(analyzer.transects, hour=hour)

    summary = predictor.get_summary()
    print(f"      Tendencia dominante: {summary['dominant_trend']}")
    print(f"      Direccion promedio: {summary['average_direction']:.0f} grados")

    # 5. Calcular scores
    print("\n[5/6] Calculando scores de pesca...")

    # Obtener datos meteorologicos del centro de la region
    center_lat, center_lon = -17.85, -71.15
    weather_data = get_weather_data(center_lat, center_lon)
    astro_data = get_astro_data(center_lat, center_lon)

    print(f"      Olas: {weather_data.get('wave_height', 'N/A')}m")
    print(f"      Viento: {weather_data.get('wind_speed', 'N/A')} km/h")
    print(f"      Luna: {astro_data.get('lunar_phase_name', 'N/A')}")

    scoring = ScoringEngine()
    scores = scoring.calculate_batch(
        transects=analyzer.transects,
        movement_vectors=movement_vectors,
        weather_data=weather_data,
        astro_data=astro_data
    )

    # Mostrar top 5
    print("\n      TOP 5 MEJORES SPOTS:")
    for i, score in enumerate(scores[:5]):
        emoji = "🟢" if score.total_score >= 60 else "🟡" if score.total_score >= 40 else "🔴"
        safe = "✅" if score.is_safe else "⚠️"
        print(f"      {i+1}. {emoji} {score.location_name}: {score.total_score:.0f}/100 {safe}")
        if score.recommended_species:
            print(f"         Especies: {', '.join(score.recommended_species)}")

    # 6. Generar mapa
    print("\n[6/6] Generando mapa interactivo...")

    map_builder = FishingMapBuilder(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=10
    )

    map_builder.create_base_map()
    map_builder.add_coastline_segments(coastline)
    map_builder.add_transects(analyzer.transects, show_points=True)
    map_builder.add_movement_arrows(movement_vectors, scale=300)
    map_builder.add_fishing_scores(scores, show_rank=True)
    map_builder.add_legend()

    # Guardar
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "analisis_costa_completo.html")
    map_builder.finalize(output_path)

    print()
    print("=" * 65)
    print(f"MAPA GUARDADO: {output_path}")
    print("=" * 65)

    # Resumen final
    best = scores[0] if scores else None
    if best:
        print()
        print("MEJOR SPOT RECOMENDADO:")
        print(f"  Ubicacion: {best.location_name}")
        print(f"  Score: {best.total_score:.0f}/100 - {best.category.value}")
        print(f"  Sustrato: {best.substrate_type}")
        print(f"  Movimiento: {best.movement_trend}")
        if best.recommended_species:
            print(f"  Especies: {', '.join(best.recommended_species)}")

    return output_path


def analizar_punto(lat: float, lon: float, nombre: str = "Punto") -> str:
    """
    Analiza un punto especifico de la costa.

    Args:
        lat: latitud del punto
        lon: longitud del punto
        nombre: nombre del punto

    Returns:
        Ruta del mapa generado
    """
    print("=" * 65)
    print("   ANALISIS DE PUNTO ESPECIFICO - Pesca desde Orilla")
    print("=" * 65)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"Punto: {nombre}")
    print(f"Coordenadas: {lat}, {lon}")
    print()

    # Crear modelo y buscar segmento cercano
    coastline = CoastlineModel()
    analyzer = TransectAnalyzer(coastline)

    # Crear transecto en el punto
    print("[1/4] Creando transecto en el punto...")
    transect = analyzer.create_transect_at_point(
        lat=lat,
        lon=lon,
        name=nombre,
        points_per_transect=7,
        max_distance_m=500
    )

    # Poblaar datos
    print("[2/4] Obteniendo datos oceanograficos...")
    analyzer.populate_oceanographic_data()

    # Predecir movimiento
    print("[3/4] Prediciendo movimiento de peces...")
    predictor = FishMovementPredictor()
    vectors = predictor.predict_movement([transect], hour=datetime.now().hour)

    if vectors:
        print(f"      Tendencia: {vectors[0].trend.value}")
        print(f"      Direccion: {vectors[0].direction_deg:.0f} grados")

    # Calcular score
    print("[4/4] Calculando score...")
    weather = get_weather_data(lat, lon)
    astro = get_astro_data(lat, lon)

    scoring = ScoringEngine()
    score = scoring.calculate_score(
        transect=transect,
        movement_vector=vectors[0] if vectors else None,
        weather_data=weather,
        astro_data=astro
    )

    print()
    print("-" * 50)
    print(f"RESULTADO: {score.total_score:.0f}/100 - {score.category.value}")
    print("-" * 50)
    print(f"  SST Score: {score.sst_score:.0f}")
    print(f"  Clorofila Score: {score.chlorophyll_score:.0f}")
    print(f"  Seguridad: {score.safety_score:.0f} {'✅' if score.is_safe else '⚠️'}")
    print(f"  Hora dorada: {score.golden_hour_score:.0f}")
    print(f"  Sustrato: {score.substrate_type}")
    if score.recommended_species:
        print(f"  Especies: {', '.join(score.recommended_species)}")

    # Generar mapa
    map_builder = FishingMapBuilder(center_lat=lat, center_lon=lon, zoom_start=15)
    map_builder.create_base_map()
    map_builder.add_transects([transect], show_points=True)
    if vectors:
        map_builder.add_movement_arrows(vectors, scale=200)
    map_builder.add_fishing_scores([score], show_rank=False)
    map_builder.add_legend()

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analisis_punto.html")
    map_builder.finalize(output_path)

    print()
    print(f"MAPA GUARDADO: {output_path}")

    return output_path


def analizar_tramo(
    lat_inicio: float,
    lon_inicio: float,
    lat_fin: float,
    lon_fin: float,
    num_secciones: int = 10,
    nombre: str = "Tramo"
) -> str:
    """
    Analiza un tramo de costa entre dos puntos.

    Args:
        lat_inicio, lon_inicio: punto inicial
        lat_fin, lon_fin: punto final
        num_secciones: numero de secciones/transectos
        nombre: nombre del tramo

    Returns:
        Ruta del mapa generado
    """
    print("=" * 65)
    print("   ANALISIS DE TRAMO COSTERO - Pesca desde Orilla")
    print("=" * 65)
    print(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"Tramo: {nombre}")
    print(f"Inicio: {lat_inicio}, {lon_inicio}")
    print(f"Fin: {lat_fin}, {lon_fin}")
    print(f"Secciones: {num_secciones}")
    print()

    # Crear modelo y transectos
    coastline = CoastlineModel()
    analyzer = TransectAnalyzer(coastline)

    print("[1/5] Generando transectos del tramo...")
    transects = analyzer.create_sweep_transects(
        lat_start=lat_inicio,
        lon_start=lon_inicio,
        lat_end=lat_fin,
        lon_end=lon_fin,
        num_transects=num_secciones,
        points_per_transect=5,
        max_distance_m=400,
        name=nombre
    )

    for t in transects:
        print(f"      {t.name}: {t.shore_lat:.5f}, {t.shore_lon:.5f}")

    # Poblar datos
    print("\n[2/5] Obteniendo datos oceanograficos...")
    analyzer.populate_oceanographic_data()

    # Predecir movimiento
    print("[3/5] Prediciendo movimiento de peces...")
    predictor = FishMovementPredictor()
    vectors = predictor.predict_movement(transects, hour=datetime.now().hour)

    summary = predictor.get_summary()
    print(f"      Tendencia dominante: {summary['dominant_trend']}")

    # Calcular scores
    print("[4/5] Calculando scores...")
    center_lat = (lat_inicio + lat_fin) / 2
    center_lon = (lon_inicio + lon_fin) / 2

    weather = get_weather_data(center_lat, center_lon)
    astro = get_astro_data(center_lat, center_lon)

    print(f"      Olas: {weather.get('wave_height', 'N/A')}m")
    print(f"      Luna: {astro.get('lunar_phase_name', 'N/A')}")

    scoring = ScoringEngine()
    scores = scoring.calculate_batch(
        transects=transects,
        movement_vectors=vectors,
        weather_data=weather,
        astro_data=astro
    )

    # Mostrar resultados
    print("\n[5/5] Resultados por seccion:")
    print("-" * 50)
    for score in scores:
        emoji = "🟢" if score.total_score >= 60 else "🟡" if score.total_score >= 40 else "🔴"
        print(f"  {emoji} {score.location_name}: {score.total_score:.0f}/100")

    # Mejor seccion
    best = scores[0] if scores else None
    if best:
        print()
        print("-" * 50)
        print(f"MEJOR SECCION: {best.location_name}")
        print(f"Score: {best.total_score:.0f}/100 - {best.category.value}")
        print(f"Coordenadas: {best.latitude:.5f}, {best.longitude:.5f}")
        if best.recommended_species:
            print(f"Especies: {', '.join(best.recommended_species)}")

    # Generar mapa
    map_builder = FishingMapBuilder(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=14
    )

    map_builder.create_base_map()
    map_builder.add_transects(transects, show_points=True)
    map_builder.add_movement_arrows(vectors, scale=200)
    map_builder.add_fishing_scores(scores, show_rank=True)
    map_builder.add_legend()

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "analisis_tramo.html")
    map_builder.finalize(output_path)

    print()
    print("=" * 65)
    print(f"MAPA GUARDADO: {output_path}")
    print("=" * 65)

    return output_path


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Analisis de pesca desde orilla - Costa sur de Peru"
    )

    parser.add_argument(
        '--punto',
        nargs=2,
        type=float,
        metavar=('LAT', 'LON'),
        help='Analizar punto especifico'
    )

    parser.add_argument(
        '--tramo',
        nargs=4,
        type=float,
        metavar=('LAT1', 'LON1', 'LAT2', 'LON2'),
        help='Analizar tramo de costa'
    )

    parser.add_argument(
        '--secciones',
        type=int,
        default=10,
        help='Numero de secciones para analisis de tramo (default: 10)'
    )

    parser.add_argument(
        '--nombre',
        type=str,
        default='Analisis',
        help='Nombre del analisis'
    )

    args = parser.parse_args()

    if args.punto:
        analizar_punto(args.punto[0], args.punto[1], args.nombre)
    elif args.tramo:
        analizar_tramo(
            args.tramo[0], args.tramo[1],
            args.tramo[2], args.tramo[3],
            num_secciones=args.secciones,
            nombre=args.nombre
        )
    else:
        analizar_costa_completa()


if __name__ == "__main__":
    main()
