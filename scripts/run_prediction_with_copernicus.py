#!/usr/bin/env python3
"""
Ejecuta predicción de zonas de pesca usando datos reales de Copernicus.

Integra:
- SST de Copernicus
- Corrientes oceánicas (uo, vo)
- Olas (altura, período, dirección)
- Clorofila-a

con el modelo ML de predicción.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from datetime import datetime
from typing import List, Dict

from core.marine_data import MarineDataFetcher, ThermalFrontDetector
from core.copernicus_data_provider import CopernicusDataProvider, convert_to_marine_points
from models.predictor import FishingPredictor
from models.features import FeatureExtractor
from data.data_config import DataConfig


def run_prediction(date: str, verbose: bool = True):
    """
    Ejecuta predicción completa con datos de Copernicus.

    Args:
        date: Fecha en formato YYYY-MM-DD
        verbose: Mostrar información detallada

    Returns:
        Dict con resultados de predicción
    """
    if verbose:
        print("=" * 70)
        print(f"PREDICCIÓN DE ZONAS DE PESCA - {date}")
        print("Datos: Copernicus Marine Service")
        print("=" * 70)
        print()

    # 1. Cargar datos de Copernicus
    if verbose:
        print("[1/6] Cargando datos oceanográficos de Copernicus...")

    provider = CopernicusDataProvider()
    ocean_points = provider.get_data_for_date(date)

    if not ocean_points:
        print("[ERROR] No hay datos disponibles para esta fecha")
        return None

    stats = provider.get_statistics(date)
    if verbose:
        print(f"      Total puntos: {stats['total_points']}")
        print(f"      SST: {stats['sst']['count']} pts (rango: {stats['sst']['min']:.1f}°C - {stats['sst']['max']:.1f}°C)")
        print(f"      Corrientes: {stats['currents']['count']} pts (velocidad media: {stats['currents']['mean_speed']:.3f} m/s)")
        print(f"      Olas: {stats['waves']['count']} pts (altura media: {stats['waves']['mean_height']:.2f} m)")
        print(f"      Clorofila: {stats['chlorophyll']['count']} pts")
        print()

    # 2. Convertir a formato compatible
    if verbose:
        print("[2/6] Preparando datos para el modelo...")

    marine_points = convert_to_marine_points(ocean_points)
    if verbose:
        print(f"      {len(marine_points)} puntos válidos para análisis")
        print()

    # 3. Detectar frentes térmicos
    if verbose:
        print("[3/6] Detectando frentes térmicos...")

    # Crear MarinePoint compatibles para detector
    from core.marine_data import MarinePoint
    detector_points = [
        MarinePoint(
            lat=p.lat, lon=p.lon,
            sst=p.sst,
            wave_height=p.wave_height,
            wave_period=p.wave_period,
            current_speed=p.current_speed,
            current_direction=p.current_direction
        )
        for p in marine_points
    ]

    front_detector = ThermalFrontDetector(gradient_threshold=0.3)
    thermal_fronts = front_detector.detect_fronts(detector_points, min_intensity=0.2)

    if verbose:
        print(f"      {len(thermal_fronts)} frentes térmicos detectados")
        if thermal_fronts:
            top_front = thermal_fronts[0]
            print(f"      Frente más intenso: ({top_front.lat:.3f}, {top_front.lon:.3f}) - {top_front.gradient:.3f}°C/km")
        print()

    # 4. Extraer características
    if verbose:
        print("[4/6] Extrayendo características oceanográficas (32 features)...")

    extractor = FeatureExtractor()

    # Línea de costa de la región
    coastline = []
    lat_range = np.arange(DataConfig.REGION['lat_min'], DataConfig.REGION['lat_max'], 0.1)
    for lat in lat_range:
        coastline.append((lat, DataConfig.REGION['lon_max'] + 0.02))

    X = extractor.extract_from_marine_points(marine_points, coastline)

    if verbose:
        print(f"      Matriz de características: {X.shape}")
        print()

    # 5. Entrenar y predecir
    if verbose:
        print("[5/6] Ejecutando modelo de predicción...")

    predictor = FishingPredictor(n_components=6, n_clusters=5)
    predictor.fit_unsupervised(X, feature_names=extractor.feature_names)

    results = predictor.predict(X)

    # Asignar coordenadas
    for i, result in enumerate(results):
        result.lat = marine_points[i].lat
        result.lon = marine_points[i].lon

    if verbose:
        print(f"      Modelo: PCA ({predictor.pca.n_components_} componentes, {sum(predictor.pca_explained_variance):.1%} varianza)")
        print(f"      Clusters: {predictor.n_clusters}")
        print()

    # 6. Analizar resultados
    if verbose:
        print("[6/6] Analizando resultados...")
        print()

    scores = [r.score for r in results]
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
    importance = predictor.get_feature_importance()
    high_prob_zones = [r for r in results if r.score > 80]

    # Estadísticas
    if verbose:
        print("=" * 70)
        print("RESULTADOS DE PREDICCIÓN")
        print("=" * 70)
        print()

        print("📊 ESTADÍSTICAS DE SCORES:")
        print(f"   Mínimo:   {min(scores):.1f}")
        print(f"   Máximo:   {max(scores):.1f}")
        print(f"   Promedio: {np.mean(scores):.1f}")
        print(f"   Mediana:  {np.median(scores):.1f}")
        print(f"   Std Dev:  {np.std(scores):.1f}")
        print()

        # Zonas de alta probabilidad (score > 80)
        print(f"🎯 ZONAS DE ALTA PROBABILIDAD (score > 80): {len(high_prob_zones)}")
        print()

        # Top 10 zonas
        print("🏆 TOP 10 ZONAS DE PESCA RECOMENDADAS:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Latitud':>9} {'Longitud':>10} {'Score':>7} {'Confianza':>10} {'Cluster':>8}")
        print("-" * 70)

        for i, r in enumerate(results_sorted[:10], 1):
            print(f"{i:<5} {r.lat:>9.4f} {r.lon:>10.4f} {r.score:>7.1f} {r.confidence:>10.2f} {r.cluster_id:>8}")

        print()

        # Feature importance
        print("📈 FACTORES MÁS IMPORTANTES:")
        print("-" * 70)
        importance = predictor.get_feature_importance()
        for i, (name, imp) in enumerate(list(importance.items())[:10], 1):
            bar = "█" * int(imp * 40)
            print(f"   {i:2}. {name:<28} {imp:.3f} {bar}")

        print()

        # Clusters
        print("🗺️ DISTRIBUCIÓN POR ZONA (Clusters):")
        print("-" * 70)
        cluster_data = {}
        for r in results:
            if r.cluster_id not in cluster_data:
                cluster_data[r.cluster_id] = {'scores': [], 'lats': [], 'lons': []}
            cluster_data[r.cluster_id]['scores'].append(r.score)
            cluster_data[r.cluster_id]['lats'].append(r.lat)
            cluster_data[r.cluster_id]['lons'].append(r.lon)

        for cid in sorted(cluster_data.keys()):
            cd = cluster_data[cid]
            avg_score = np.mean(cd['scores'])
            avg_lat = np.mean(cd['lats'])
            avg_lon = np.mean(cd['lons'])
            quality = "⭐⭐⭐" if avg_score > 85 else "⭐⭐" if avg_score > 70 else "⭐"
            print(f"   Zona {cid}: {len(cd['scores']):>3} pts | Centro: ({avg_lat:.2f}, {avg_lon:.2f}) | Score: {avg_score:.1f} {quality}")

        print()
        print("=" * 70)
        print("PREDICCIÓN COMPLETADA")
        print("=" * 70)

    return {
        'date': date,
        'total_points': len(results),
        'statistics': {
            'min': min(scores),
            'max': max(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores)
        },
        'top_zones': [
            {'lat': r.lat, 'lon': r.lon, 'score': r.score, 'confidence': r.confidence, 'cluster': r.cluster_id}
            for r in results_sorted[:10]
        ],
        'thermal_fronts': len(thermal_fronts),
        'feature_importance': dict(list(importance.items())[:10]),
        'high_probability_zones': len(high_prob_zones)
    }


def compare_seasons():
    """Compara predicciones entre diferentes estaciones."""
    print("\n" + "=" * 70)
    print("COMPARACIÓN ESTACIONAL")
    print("=" * 70 + "\n")

    dates = {
        'Invierno (Jun 2024)': '2024-06-15',
        'Primavera (Sep 2024)': '2024-09-15',
        'Verano (Dic 2024)': '2024-12-15',
        'Otoño (Mar 2025)': '2025-03-15',
    }

    results = {}
    for name, date in dates.items():
        print(f"\n--- {name} ---")
        result = run_prediction(date, verbose=False)
        if result:
            results[name] = result
            print(f"   Score medio: {result['statistics']['mean']:.1f}")
            print(f"   Zonas alta prob: {result['high_probability_zones']}")
            print(f"   Frentes térmicos: {result['thermal_fronts']}")

    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Estación':<25} {'Score Medio':>12} {'Zonas >80':>10} {'Frentes':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<25} {r['statistics']['mean']:>12.1f} {r['high_probability_zones']:>10} {r['thermal_fronts']:>8}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predicción de zonas de pesca con Copernicus')
    parser.add_argument('--date', default=None, help='Fecha (YYYY-MM-DD)')
    parser.add_argument('--compare', action='store_true', help='Comparar estaciones')
    args = parser.parse_args()

    if args.compare:
        compare_seasons()
    else:
        # Usar fecha reciente por defecto
        date = args.date or '2025-01-15'
        run_prediction(date)
