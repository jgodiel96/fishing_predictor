#!/usr/bin/env python3
"""
Generador de Predicciones Horarias - Plan V3

Este script genera predicciones de pesca para cada hora del día,
integrando:
- Datos de mareas (TideFetcher)
- SST y condiciones marinas (Bronze layer)
- Modelo ML (FishingPredictor)

Uso:
    python scripts/generate_hourly_predictions.py --date 2026-01-30
    python scripts/generate_hourly_predictions.py --date 2026-01-30 --lat -17.702 --lon -71.332
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import sqlite3
import json

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.fetchers.tide_fetcher import TideFetcher, TideData
from data.data_config import DataConfig
from domain import HOTSPOTS, SPECIES, STUDY_AREA

# Importar fetcher de SSS/SLA si está disponible
try:
    from data.fetchers.copernicus_physics_fetcher import CopernicusPhysicsFetcher
    HAS_PHYSICS_FETCHER = True
except ImportError:
    HAS_PHYSICS_FETCHER = False


class HourlyPredictionGenerator:
    """
    Genera predicciones de pesca para cada hora del día.

    Combina:
    - Datos de mareas (calculados astronómicamente)
    - Condiciones temporales (alba, ocaso, hora)
    - SST del día (si disponible)
    - Hotspots conocidos
    """

    # Horas de alba y ocaso aproximadas para Tacna-Ilo (varían por estación)
    SOLAR_TIMES = {
        'summer': {'dawn': 5.5, 'dusk': 19.0},  # Dic-Feb
        'autumn': {'dawn': 6.0, 'dusk': 18.0},  # Mar-May
        'winter': {'dawn': 6.5, 'dusk': 17.5},  # Jun-Ago
        'spring': {'dawn': 5.5, 'dusk': 18.5},  # Sep-Nov
    }

    # Scores por hora del día (basado en experiencia de pescadores)
    HOUR_SCORES = {
        # Madrugada (baja actividad)
        0: 0.2, 1: 0.2, 2: 0.2, 3: 0.3,
        # Pre-alba (aumenta)
        4: 0.5, 5: 0.7,
        # Alba (excelente)
        6: 0.95, 7: 0.9,
        # Mañana (buena)
        8: 0.75, 9: 0.65, 10: 0.55, 11: 0.45,
        # Mediodía (baja)
        12: 0.35, 13: 0.3, 14: 0.35,
        # Tarde (aumenta)
        15: 0.45, 16: 0.55, 17: 0.7,
        # Atardecer (excelente)
        18: 0.9, 19: 0.85,
        # Noche (moderada)
        20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3
    }

    def __init__(self):
        self.tide_fetcher = TideFetcher()
        self.db_path = DataConfig.PROCESSED_DIR / "hourly_predictions.db"
        self._init_database()

        # Inicializar fetcher de física oceánica si está disponible
        self.physics_fetcher = None
        if HAS_PHYSICS_FETCHER:
            try:
                self.physics_fetcher = CopernicusPhysicsFetcher(verbose=False)
            except Exception:
                pass

    def _init_database(self):
        """Inicializa la base de datos de predicciones horarias."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hourly_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    location_name TEXT,

                    -- Scores componentes
                    tide_score REAL,
                    hour_score REAL,
                    sst_score REAL,
                    hotspot_score REAL,

                    -- Score final
                    total_score REAL NOT NULL,
                    confidence REAL,

                    -- Datos de marea
                    tide_phase TEXT,
                    tide_height REAL,
                    hours_to_high REAL,
                    hours_to_low REAL,

                    -- Metadatos
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(date, hour, lat, lon)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_date
                ON hourly_predictions(date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_location
                ON hourly_predictions(lat, lon)
            """)

    def _get_season(self, date: datetime) -> str:
        """Determina la estación del año (hemisferio sur)."""
        month = date.month
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'

    def _calculate_hour_score(self, hour: int, date: datetime) -> float:
        """
        Calcula el score basado en la hora del día.

        Las mejores horas son alba y atardecer.
        """
        base_score = self.HOUR_SCORES.get(hour, 0.5)

        # Ajuste por estación (alba/ocaso varían)
        season = self._get_season(date)
        solar = self.SOLAR_TIMES[season]

        # Bonus si está cerca del alba o atardecer
        dist_to_dawn = abs(hour - solar['dawn'])
        dist_to_dusk = abs(hour - solar['dusk'])
        min_dist = min(dist_to_dawn, dist_to_dusk)

        if min_dist < 1:
            base_score = min(1.0, base_score + 0.1)

        return base_score

    def _calculate_hotspot_score(self, lat: float, lon: float) -> Tuple[float, str]:
        """
        Calcula el score basado en proximidad a hotspots conocidos.

        Returns:
            Tuple de (score, nombre_hotspot_cercano)
        """
        min_dist = float('inf')
        closest_name = ""
        closest_bonus = 1.0

        for hotspot in HOTSPOTS:
            dist = np.sqrt((lat - hotspot.lat)**2 + (lon - hotspot.lon)**2)
            if dist < min_dist:
                min_dist = dist
                closest_name = hotspot.name
                closest_bonus = hotspot.bonus_factor

        # Convertir distancia a score (más cerca = mejor)
        # 0.05 grados ~ 5km
        if min_dist < 0.02:  # < 2km
            score = 1.0 * closest_bonus
        elif min_dist < 0.05:  # < 5km
            score = 0.8 * closest_bonus
        elif min_dist < 0.1:  # < 10km
            score = 0.5 * closest_bonus
        else:
            score = 0.2
            closest_name = "Zona abierta"

        return min(1.0, score), closest_name

    def _get_sst_for_date(self, date: str, lat: float, lon: float) -> Optional[float]:
        """
        Obtiene SST del día desde la base de datos consolidada.
        """
        try:
            marine_db = DataConfig.MARINE_DB
            if not marine_db.exists():
                return None

            with sqlite3.connect(marine_db) as conn:
                # Buscar SST cercana a la ubicación
                result = conn.execute("""
                    SELECT AVG(sst) as avg_sst
                    FROM sst
                    WHERE date LIKE ?
                    AND lat BETWEEN ? AND ?
                    AND lon BETWEEN ? AND ?
                """, (
                    f"{date}%",
                    lat - 0.1, lat + 0.1,
                    lon - 0.1, lon + 0.1
                )).fetchone()

                if result and result[0]:
                    return result[0]
        except Exception:
            pass

        return None

    def _calculate_sst_score(self, sst: Optional[float], target_species: str = None) -> float:
        """
        Calcula score basado en SST para especies objetivo.
        """
        if sst is None:
            return 0.5  # Score neutral si no hay datos

        # Rango óptimo general para costa peruana (Humboldt)
        optimal_min = 14.0
        optimal_max = 20.0

        # Si hay especie específica, usar su rango
        if target_species and target_species in SPECIES:
            species_data = SPECIES[target_species]
            optimal_min = species_data.temp_min
            optimal_max = species_data.temp_max

        # Calcular score
        if optimal_min <= sst <= optimal_max:
            # En rango óptimo
            center = (optimal_min + optimal_max) / 2
            dist_from_center = abs(sst - center) / ((optimal_max - optimal_min) / 2)
            score = 1.0 - (dist_from_center * 0.2)  # Máximo 0.2 penalización
        elif sst < optimal_min:
            # Demasiado frío
            diff = optimal_min - sst
            score = max(0.2, 0.8 - diff * 0.1)
        else:
            # Demasiado cálido
            diff = sst - optimal_max
            score = max(0.2, 0.8 - diff * 0.1)

        return score

    def generate_for_location(
        self,
        date: str,
        lat: float,
        lon: float,
        location_name: str = None
    ) -> List[Dict]:
        """
        Genera predicciones para las 24 horas en una ubicación específica.

        Args:
            date: Fecha en formato 'YYYY-MM-DD'
            lat: Latitud
            lon: Longitud
            location_name: Nombre opcional de la ubicación

        Returns:
            Lista de predicciones horarias
        """
        dt = datetime.strptime(date, '%Y-%m-%d')
        predictions = []

        # Obtener datos de marea para todo el día
        tide_data = self.tide_fetcher.fetch_tides_for_date(date, lat, lon)

        # Obtener SST del día
        sst = self._get_sst_for_date(date, lat, lon)
        sst_score = self._calculate_sst_score(sst)

        # Obtener SSS y SLA si están disponibles (Plan V3)
        sss = None
        sla = None
        sss_score = 0.5  # Neutral si no hay datos
        sla_score = 0.5

        if self.physics_fetcher:
            sss = self.physics_fetcher.get_sss_for_location(date, lat, lon)
            sla = self.physics_fetcher.get_sla_for_location(date, lat, lon)

            if sss is not None:
                sss_score = self.physics_fetcher.calculate_sss_score(sss)
            if sla is not None:
                sla_score = self.physics_fetcher.calculate_sla_score(sla)

        # Score de hotspot (constante para la ubicación)
        hotspot_score, hotspot_name = self._calculate_hotspot_score(lat, lon)

        if location_name is None:
            location_name = hotspot_name

        for hour in range(24):
            tide = tide_data[hour]

            # Calcular scores individuales
            hour_score = self._calculate_hour_score(hour, dt)

            # Score total ponderado
            # Pesos basados en literatura V2:
            # Si hay SSS/SLA disponibles, redistribuir pesos
            if sss is not None or sla is not None:
                # Pesos con física oceánica completa
                # - Mareas: 30%
                # - Hora: 20%
                # - SST: 15%
                # - SSS: 15% (variable #1 en papers)
                # - SLA: 10% (variable #2 en papers)
                # - Hotspot: 10%
                total_score = (
                    tide.fishing_score * 0.30 +
                    hour_score * 0.20 +
                    sst_score * 0.15 +
                    sss_score * 0.15 +
                    sla_score * 0.10 +
                    hotspot_score * 0.10
                ) * 100
            else:
                # Pesos sin física oceánica (original)
                # - Mareas: 35% (muy importante para costa)
                # - Hora: 25% (alba/ocaso críticos)
                # - SST: 20% (condiciona presencia de peces)
                # - Hotspot: 20% (zonas históricamente buenas)
                total_score = (
                    tide.fishing_score * 0.35 +
                    hour_score * 0.25 +
                    sst_score * 0.20 +
                    hotspot_score * 0.20
                ) * 100  # Escala 0-100

            # Confianza basada en disponibilidad de datos
            confidence = 0.6  # Base
            if sst is not None:
                confidence += 0.10
            if sss is not None:
                confidence += 0.10
            if sla is not None:
                confidence += 0.10
            if hotspot_score > 0.5:
                confidence += 0.10

            predictions.append({
                'date': date,
                'hour': hour,
                'time': f"{hour:02d}:00",
                'lat': lat,
                'lon': lon,
                'location_name': location_name,

                # Scores componentes
                'tide_score': round(tide.fishing_score * 100, 1),
                'hour_score': round(hour_score * 100, 1),
                'sst_score': round(sst_score * 100, 1),
                'sss_score': round(sss_score * 100, 1) if sss is not None else None,
                'sla_score': round(sla_score * 100, 1) if sla is not None else None,
                'hotspot_score': round(hotspot_score * 100, 1),

                # Score final
                'total_score': round(total_score, 1),
                'confidence': round(confidence, 2),

                # Datos de marea
                'tide_phase': tide.tide_phase,
                'tide_phase_es': self._translate_phase(tide.tide_phase),
                'tide_height': tide.tide_height,
                'hours_to_high': tide.hours_to_high,
                'hours_to_low': tide.hours_to_low,

                # Datos oceanográficos
                'sst': sst,
                'sss': sss,
                'sla': sla,
            })

        return predictions

    def _translate_phase(self, phase: str) -> str:
        """Traduce fase de marea al español."""
        translations = {
            'flooding': 'Marea entrante',
            'ebbing': 'Marea saliente',
            'slack_high': 'Reposo pleamar',
            'slack_low': 'Reposo bajamar'
        }
        return translations.get(phase, phase)

    def generate_for_hotspots(self, date: str) -> Dict[str, List[Dict]]:
        """
        Genera predicciones para todos los hotspots conocidos.

        Returns:
            Diccionario con nombre de hotspot -> predicciones
        """
        results = {}

        for hotspot in HOTSPOTS:
            predictions = self.generate_for_location(
                date,
                hotspot.lat,
                hotspot.lon,
                hotspot.name
            )
            results[hotspot.name] = predictions

        return results

    def get_best_hours(
        self,
        date: str,
        lat: float,
        lon: float,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Obtiene las mejores horas para pescar en una ubicación.

        Returns:
            Lista de las top_n mejores horas
        """
        predictions = self.generate_for_location(date, lat, lon)

        # Ordenar por score total
        sorted_preds = sorted(
            predictions,
            key=lambda x: x['total_score'],
            reverse=True
        )

        return sorted_preds[:top_n]

    def get_best_locations_for_hour(
        self,
        date: str,
        hour: int,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Obtiene las mejores ubicaciones para una hora específica.

        Returns:
            Lista de las top_n mejores ubicaciones
        """
        all_predictions = []

        for hotspot in HOTSPOTS:
            predictions = self.generate_for_location(
                date, hotspot.lat, hotspot.lon, hotspot.name
            )
            # Solo la predicción para la hora especificada
            hour_pred = predictions[hour]
            all_predictions.append(hour_pred)

        # Ordenar por score total
        sorted_preds = sorted(
            all_predictions,
            key=lambda x: x['total_score'],
            reverse=True
        )

        return sorted_preds[:top_n]

    def generate_multiday(
        self,
        start_date: str,
        num_days: int = 7,
        lat: float = None,
        lon: float = None,
        location_name: str = None
    ) -> Dict:
        """
        Genera predicciones para múltiples días consecutivos.

        Args:
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            num_days: Número de días a generar (default: 7)
            lat: Latitud (si None, usa Punta Coles)
            lon: Longitud (si None, usa Punta Coles)
            location_name: Nombre de la ubicación

        Returns:
            Dict con estructura:
            {
                'start_date': str,
                'end_date': str,
                'location': {'lat': float, 'lon': float, 'name': str},
                'days': {
                    'YYYY-MM-DD': {
                        'day_name': str,
                        'predictions': List[Dict],  # 24 horas
                        'tide_extremes': List[Dict],
                        'best_hours': List[Dict],  # Top 5
                        'avg_score': float
                    },
                    ...
                }
            }
        """
        # Valores por defecto: Punta Coles
        if lat is None:
            lat = -17.702
        if lon is None:
            lon = -71.332

        # Determinar nombre de ubicación
        if location_name is None:
            _, location_name = self._calculate_hotspot_score(lat, lon)

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=num_days - 1)

        # Nombres de días en español
        day_names_es = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

        result = {
            'start_date': start_date,
            'end_date': end_dt.strftime('%Y-%m-%d'),
            'location': {
                'lat': lat,
                'lon': lon,
                'name': location_name
            },
            'days': {}
        }

        for i in range(num_days):
            current_dt = start_dt + timedelta(days=i)
            date_str = current_dt.strftime('%Y-%m-%d')
            day_name = day_names_es[current_dt.weekday()]

            # Generar predicciones para las 24 horas
            predictions = self.generate_for_location(date_str, lat, lon, location_name)

            # Obtener extremos de marea
            tide_extremes = self.tide_fetcher.get_tide_extremes_for_date(date_str, lat, lon)

            # Obtener mejores 5 horas
            best_hours = sorted(
                predictions,
                key=lambda x: x['total_score'],
                reverse=True
            )[:5]

            # Calcular score promedio del día
            avg_score = sum(p['total_score'] for p in predictions) / len(predictions)

            result['days'][date_str] = {
                'day_name': day_name,
                'day_full': current_dt.strftime('%A'),
                'predictions': predictions,
                'tide_extremes': tide_extremes,
                'best_hours': best_hours,
                'avg_score': round(avg_score, 1)
            }

        return result

    def get_multiday_summary(self, multiday_data: Dict) -> List[Dict]:
        """
        Genera un resumen de los múltiples días para mostrar en el selector.

        Args:
            multiday_data: Resultado de generate_multiday()

        Returns:
            Lista de dicts con resumen por día para el selector
        """
        summary = []
        for date_str, day_data in multiday_data['days'].items():
            best_hour = day_data['best_hours'][0] if day_data['best_hours'] else None
            summary.append({
                'date': date_str,
                'day_name': day_data['day_name'],
                'avg_score': day_data['avg_score'],
                'best_hour': best_hour['time'] if best_hour else 'N/A',
                'best_score': best_hour['total_score'] if best_hour else 0,
                'tide_count': len(day_data['tide_extremes'])
            })
        return summary

    def save_to_database(self, predictions: List[Dict]):
        """
        Guarda predicciones en la base de datos.
        """
        with sqlite3.connect(self.db_path) as conn:
            for pred in predictions:
                conn.execute("""
                    INSERT OR REPLACE INTO hourly_predictions (
                        date, hour, lat, lon, location_name,
                        tide_score, hour_score, sst_score, hotspot_score,
                        total_score, confidence,
                        tide_phase, tide_height, hours_to_high, hours_to_low
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred['date'], pred['hour'], pred['lat'], pred['lon'],
                    pred['location_name'],
                    pred['tide_score'], pred['hour_score'],
                    pred['sst_score'], pred['hotspot_score'],
                    pred['total_score'], pred['confidence'],
                    pred['tide_phase'], pred['tide_height'],
                    pred['hours_to_high'], pred['hours_to_low']
                ))

    def print_daily_report(self, date: str, lat: float, lon: float):
        """
        Imprime un reporte diario de predicciones.
        """
        predictions = self.generate_for_location(date, lat, lon)
        best_hours = self.get_best_hours(date, lat, lon)

        # Extremos de marea
        tide_extremes = self.tide_fetcher.get_tide_extremes_for_date(date, lat, lon)

        print(f"\n{'='*60}")
        print(f"PREDICCION HORARIA - {date}")
        print(f"Ubicacion: ({lat}, {lon})")
        print(f"{'='*60}")

        print(f"\n--- MAREAS DEL DIA ---")
        for e in tide_extremes:
            print(f"  {e['time']} - {e['type_es']}: {e['height']}m")

        print(f"\n--- MEJORES 5 HORAS ---")
        for i, h in enumerate(best_hours, 1):
            print(f"  {i}. {h['time']} - Score: {h['total_score']:.0f}/100")
            print(f"     Marea: {h['tide_phase_es']} ({h['tide_score']:.0f})")
            print(f"     Hora: {h['hour_score']:.0f} | SST: {h['sst_score']:.0f} | Zona: {h['hotspot_score']:.0f}")

        print(f"\n--- TODAS LAS HORAS ---")
        print(f"{'Hora':<6} {'Score':<8} {'Marea':<18} {'Altura':<8}")
        print("-" * 45)
        for p in predictions:
            bar = '█' * int(p['total_score'] / 10)
            print(f"{p['time']:<6} {p['total_score']:>5.0f}  {bar:<10} {p['tide_phase_es']:<18} {p['tide_height']:>5.2f}m")


def main():
    parser = argparse.ArgumentParser(
        description='Genera predicciones horarias de pesca'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Fecha (YYYY-MM-DD). Default: hoy'
    )
    parser.add_argument(
        '--lat',
        type=float,
        default=-17.702,
        help='Latitud. Default: Punta Coles'
    )
    parser.add_argument(
        '--lon',
        type=float,
        default=-71.332,
        help='Longitud. Default: Punta Coles'
    )
    parser.add_argument(
        '--all-hotspots',
        action='store_true',
        help='Generar para todos los hotspots'
    )
    parser.add_argument(
        '--multiday',
        type=int,
        default=0,
        metavar='N',
        help='Generar predicciones para N días (default: 0 = solo un día)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Guardar en base de datos'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Salida en formato JSON'
    )

    args = parser.parse_args()

    generator = HourlyPredictionGenerator()

    # Modo multi-día
    if args.multiday > 0:
        print(f"Generando predicciones para {args.multiday} días desde {args.date}...")
        multiday_data = generator.generate_multiday(
            start_date=args.date,
            num_days=args.multiday,
            lat=args.lat,
            lon=args.lon
        )

        if args.json:
            print(json.dumps(multiday_data, indent=2, default=str))
        else:
            # Imprimir resumen
            summary = generator.get_multiday_summary(multiday_data)
            print(f"\n{'='*60}")
            print(f"PREDICCIÓN {args.multiday} DÍAS - {multiday_data['location']['name']}")
            print(f"{'='*60}")
            print(f"{'Fecha':<12} {'Día':<5} {'Prom':<6} {'Mejor Hora':<12} {'Score':<6}")
            print("-" * 45)
            for day in summary:
                print(f"{day['date']:<12} {day['day_name']:<5} {day['avg_score']:>5.0f} {day['best_hour']:<12} {day['best_score']:>5.0f}")

        if args.save:
            for date_str, day_data in multiday_data['days'].items():
                generator.save_to_database(day_data['predictions'])
            print(f"\nGuardado en {generator.db_path}")

    elif args.all_hotspots:
        print(f"Generando predicciones para todos los hotspots...")
        results = generator.generate_for_hotspots(args.date)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for name, predictions in results.items():
                best = sorted(predictions, key=lambda x: x['total_score'], reverse=True)[:3]
                print(f"\n{name}:")
                for h in best:
                    print(f"  {h['time']} - Score: {h['total_score']:.0f} ({h['tide_phase_es']})")

        if args.save:
            for predictions in results.values():
                generator.save_to_database(predictions)
            print(f"\nGuardado en {generator.db_path}")
    else:
        if args.json:
            predictions = generator.generate_for_location(args.date, args.lat, args.lon)
            print(json.dumps(predictions, indent=2, default=str))
        else:
            generator.print_daily_report(args.date, args.lat, args.lon)

        if args.save:
            predictions = generator.generate_for_location(args.date, args.lat, args.lon)
            generator.save_to_database(predictions)
            print(f"\nGuardado en {generator.db_path}")


if __name__ == '__main__':
    main()
