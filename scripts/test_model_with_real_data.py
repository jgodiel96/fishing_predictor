#!/usr/bin/env python3
"""
Test del modelo de predicción con datos reales de Copernicus.

Integra:
- SST (Sea Surface Temperature)
- Chlorophyll-a (productividad primaria)
- Corrientes oceánicas (uo, vo) - NUEVO
- Olas (VHM0, VTPK, VMDR) - NUEVO
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from data.data_config import DataConfig
from models.predictor import FishingPredictor
from models.features import FeatureExtractor


@dataclass
class RealMarinePoint:
    """Punto marino con datos reales de Copernicus."""
    lat: float
    lon: float
    sst: Optional[float] = None
    chlorophyll: Optional[float] = None
    current_speed: Optional[float] = None
    current_direction: Optional[float] = None
    wave_height: Optional[float] = None
    wave_period: Optional[float] = None
    wave_direction: Optional[float] = None
    date: str = ""


class RealDataLoader:
    """Carga datos reales de los archivos parquet de Copernicus."""

    def __init__(self):
        self.data_dir = DataConfig.RAW_DIR
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_sst(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de SST."""
        path = self.data_dir / "sst" / "copernicus" / f"{year}-{month:02d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def load_chlorophyll(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de clorofila."""
        path = self.data_dir / "chla" / "copernicus" / f"{year}-{month:02d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def load_currents(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de corrientes."""
        path = self.data_dir / "currents" / "copernicus" / f"{year}-{month:02d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def load_waves(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Carga datos de olas."""
        path = self.data_dir / "waves" / "copernicus" / f"{year}-{month:02d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def get_merged_data(self, date: str) -> List[RealMarinePoint]:
        """
        Obtiene datos fusionados de todas las fuentes para una fecha.

        Args:
            date: Fecha en formato YYYY-MM-DD

        Returns:
            Lista de RealMarinePoint con datos fusionados
        """
        year, month = int(date[:4]), int(date[5:7])

        # Cargar datos
        sst_df = self.load_sst(year, month)
        chla_df = self.load_chlorophyll(year, month)
        currents_df = self.load_currents(year, month)
        waves_df = self.load_waves(year, month)

        if sst_df is None:
            print(f"[WARN] No hay datos SST para {date}")
            return []

        # Filtrar por fecha
        sst_day = sst_df[sst_df['date'] == date] if 'date' in sst_df.columns else sst_df

        points = []
        for _, row in sst_day.iterrows():
            lat, lon = row['lat'], row['lon']

            point = RealMarinePoint(
                lat=lat,
                lon=lon,
                sst=row.get('sst', row.get('analysed_sst')),
                date=date
            )

            # Agregar chlorophyll
            if chla_df is not None:
                # Manejar diferentes formatos de fecha
                if 'date' in chla_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(chla_df['date']):
                        chla_day = chla_df[chla_df['date'].dt.strftime('%Y-%m-%d') == date]
                    else:
                        chla_day = chla_df[chla_df['date'] == date]
                else:
                    chla_day = chla_df
                nearby = chla_day[
                    (abs(chla_day['lat'] - lat) < 0.1) &
                    (abs(chla_day['lon'] - lon) < 0.1)
                ]
                if not nearby.empty:
                    # Usar 'value' que es el nombre real de la columna
                    point.chlorophyll = nearby.iloc[0].get('value', nearby.iloc[0].get('CHL', nearby.iloc[0].get('chlorophyll')))

            # Agregar corrientes
            if currents_df is not None:
                curr_day = currents_df[currents_df['date'] == date] if 'date' in currents_df.columns else currents_df
                nearby = curr_day[
                    (abs(curr_day['lat'] - lat) < 0.1) &
                    (abs(curr_day['lon'] - lon) < 0.1)
                ]
                if not nearby.empty:
                    row_curr = nearby.iloc[0]
                    uo = row_curr.get('uo', 0)
                    vo = row_curr.get('vo', 0)
                    point.current_speed = row_curr.get('speed', np.sqrt(uo**2 + vo**2))
                    point.current_direction = row_curr.get('direction', np.degrees(np.arctan2(vo, uo)))

            # Agregar olas
            if waves_df is not None:
                waves_day = waves_df[waves_df['date'] == date] if 'date' in waves_df.columns else waves_df
                nearby = waves_day[
                    (abs(waves_day['lat'] - lat) < 0.1) &
                    (abs(waves_day['lon'] - lon) < 0.1)
                ]
                if not nearby.empty:
                    row_wave = nearby.iloc[0]
                    point.wave_height = row_wave.get('VHM0', row_wave.get('wave_height'))
                    point.wave_period = row_wave.get('VTPK', row_wave.get('wave_period'))
                    point.wave_direction = row_wave.get('VMDR', row_wave.get('wave_direction'))

            points.append(point)

        return points


def convert_to_feature_compatible(points: List[RealMarinePoint]) -> List:
    """Convierte RealMarinePoint a formato compatible con FeatureExtractor."""

    @dataclass
    class CompatiblePoint:
        lat: float
        lon: float
        sst: Optional[float]
        wave_height: Optional[float]
        wave_period: Optional[float]
        current_speed: Optional[float]
        current_direction: Optional[float]

    return [
        CompatiblePoint(
            lat=p.lat,
            lon=p.lon,
            sst=p.sst,
            wave_height=p.wave_height or 1.0,
            wave_period=p.wave_period or 8.0,
            current_speed=p.current_speed or 0.1,
            current_direction=p.current_direction or 180
        )
        for p in points if p.sst is not None
    ]


def run_model_test(date: str):
    """Ejecuta prueba del modelo con datos reales."""

    print("=" * 60)
    print(f"TEST DEL MODELO CON DATOS REALES - {date}")
    print("=" * 60)
    print()

    # 1. Cargar datos reales
    print("[1/5] Cargando datos de Copernicus...")
    loader = RealDataLoader()
    points = loader.get_merged_data(date)

    if not points:
        print("[ERROR] No se encontraron datos para la fecha especificada")
        return None

    print(f"      ✓ {len(points)} puntos cargados")

    # Estadísticas de datos
    has_sst = sum(1 for p in points if p.sst is not None)
    has_currents = sum(1 for p in points if p.current_speed is not None)
    has_waves = sum(1 for p in points if p.wave_height is not None)
    has_chla = sum(1 for p in points if p.chlorophyll is not None)

    print(f"      - SST: {has_sst} puntos")
    print(f"      - Corrientes: {has_currents} puntos")
    print(f"      - Olas: {has_waves} puntos")
    print(f"      - Clorofila: {has_chla} puntos")
    print()

    # 2. Convertir a formato compatible
    print("[2/5] Preparando datos para el modelo...")
    compatible_points = convert_to_feature_compatible(points)

    if len(compatible_points) < 10:
        print("[ERROR] Insuficientes puntos con datos SST válidos")
        return None

    print(f"      ✓ {len(compatible_points)} puntos procesados")
    print()

    # 3. Extraer features
    print("[3/5] Extrayendo características oceanográficas...")
    extractor = FeatureExtractor()

    # Crear línea de costa simulada para la región
    coastline = [
        (-17.0 + i*0.1, -71.35) for i in range(20)
    ] + [
        (-18.0 + i*0.1, -71.30) for i in range(20)
    ]

    X = extractor.extract_from_marine_points(compatible_points, coastline)

    print(f"      ✓ Matriz de features: {X.shape}")
    print(f"      - {len(extractor.feature_names)} características")
    print()

    # 4. Entrenar modelo
    print("[4/5] Entrenando modelo de predicción...")
    predictor = FishingPredictor(n_components=6, n_clusters=5)
    predictor.fit_unsupervised(X, feature_names=extractor.feature_names)

    print(f"      ✓ Modelo entrenado")
    print(f"      - PCA: {predictor.pca.n_components_} componentes")
    print(f"      - Varianza explicada: {sum(predictor.pca_explained_variance):.1%}")
    print()

    # 5. Predicciones
    print("[5/5] Generando predicciones...")
    results = predictor.predict(X)

    # Asignar coordenadas
    for i, result in enumerate(results):
        result.lat = compatible_points[i].lat
        result.lon = compatible_points[i].lon

    # Estadísticas de predicción
    scores = [r.score for r in results]
    print(f"      ✓ {len(results)} predicciones generadas")
    print()

    # Resultados
    print("=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print()

    print("📊 ESTADÍSTICAS DE SCORES:")
    print(f"   - Mínimo:  {min(scores):.1f}")
    print(f"   - Máximo:  {max(scores):.1f}")
    print(f"   - Promedio: {np.mean(scores):.1f}")
    print(f"   - Mediana:  {np.median(scores):.1f}")
    print(f"   - Std Dev:  {np.std(scores):.1f}")
    print()

    # Top 10 zonas
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)

    print("🎯 TOP 10 ZONAS DE PESCA:")
    print("-" * 60)
    print(f"{'#':<3} {'Lat':>8} {'Lon':>9} {'Score':>7} {'Conf':>6} {'Cluster':>8}")
    print("-" * 60)

    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:<3} {r.lat:>8.3f} {r.lon:>9.3f} {r.score:>7.1f} {r.confidence:>6.2f} {r.cluster_id:>8}")

    print()

    # Feature importance
    print("📈 IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
    print("-" * 60)
    importance = predictor.get_feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:10], 1):
        bar = "█" * int(imp * 50)
        print(f"   {i:2}. {name:<25} {imp:.3f} {bar}")

    print()

    # PCA Analysis
    print("🔬 ANÁLISIS PCA:")
    print("-" * 60)
    for i, var in enumerate(predictor.pca_explained_variance):
        cum = sum(predictor.pca_explained_variance[:i+1])
        bar = "█" * int(var * 30)
        print(f"   PC{i+1}: {var:>6.1%} (acum: {cum:>6.1%}) {bar}")

    print()

    # Distribución por cluster
    print("🗺️ DISTRIBUCIÓN POR CLUSTER:")
    print("-" * 60)
    cluster_stats = {}
    for r in results:
        if r.cluster_id not in cluster_stats:
            cluster_stats[r.cluster_id] = {'scores': [], 'count': 0}
        cluster_stats[r.cluster_id]['scores'].append(r.score)
        cluster_stats[r.cluster_id]['count'] += 1

    for cluster_id in sorted(cluster_stats.keys()):
        stats = cluster_stats[cluster_id]
        avg_score = np.mean(stats['scores'])
        print(f"   Cluster {cluster_id}: {stats['count']:>4} puntos | Score promedio: {avg_score:.1f}")

    print()
    print("=" * 60)
    print("TEST COMPLETADO EXITOSAMENTE")
    print("=" * 60)

    return {
        'date': date,
        'n_points': len(results),
        'scores': scores,
        'top_zones': results_sorted[:10],
        'feature_importance': importance,
        'pca_variance': predictor.pca_explained_variance.tolist()
    }


def test_multiple_dates():
    """Prueba el modelo con múltiples fechas."""

    print("\n" + "=" * 60)
    print("PRUEBA MULTI-FECHA")
    print("=" * 60 + "\n")

    # Fechas de prueba (días con datos completos)
    test_dates = [
        "2024-06-15",  # Invierno austral
        "2024-12-15",  # Verano austral
        "2025-03-15",  # Otoño austral
    ]

    results_summary = []

    for date in test_dates:
        print(f"\n--- Probando fecha: {date} ---\n")
        result = run_model_test(date)
        if result:
            results_summary.append(result)

    # Comparación entre fechas
    if len(results_summary) > 1:
        print("\n" + "=" * 60)
        print("COMPARACIÓN ENTRE FECHAS")
        print("=" * 60 + "\n")

        for r in results_summary:
            scores = r['scores']
            print(f"{r['date']}: Media={np.mean(scores):.1f}, Max={max(scores):.1f}, Puntos={r['n_points']}")


def verify_data_availability():
    """Verifica disponibilidad de datos antes de las pruebas."""

    print("=" * 60)
    print("VERIFICACIÓN DE DATOS DISPONIBLES")
    print("=" * 60 + "\n")

    loader = RealDataLoader()

    sources = {
        'SST': ('sst', 'copernicus'),
        'Chlorophyll': ('chla', 'copernicus'),
        'Currents': ('currents', 'copernicus'),
        'Waves': ('waves', 'copernicus'),
    }

    for name, (subdir, provider) in sources.items():
        path = loader.data_dir / subdir / provider
        if path.exists():
            files = list(path.glob("*.parquet"))
            files = [f for f in files if not f.name.startswith('_')]
            print(f"✓ {name}: {len(files)} archivos")
            if files:
                dates = sorted([f.stem for f in files])
                print(f"  Rango: {dates[0]} a {dates[-1]}")
        else:
            print(f"✗ {name}: No disponible")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test modelo con datos reales')
    parser.add_argument('--date', default="2025-01-15", help='Fecha a probar (YYYY-MM-DD)')
    parser.add_argument('--multi', action='store_true', help='Probar múltiples fechas')
    args = parser.parse_args()

    # Verificar datos
    verify_data_availability()

    if args.multi:
        # Probar múltiples fechas (estaciones)
        test_multiple_dates()
    else:
        # Ejecutar prueba principal
        result = run_model_test(args.date)
