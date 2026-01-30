"""
Anchovy (Anchoveta) Migration Model for Peruvian Coast.

Based on IMARPE research and Humboldt Current dynamics.
Anchoveta (Engraulis ringens) follows predictable patterns:
- SST preference: 15-21°C (optimal 17-19°C)
- Depth: 0-50m during day, surface at dusk
- Migration: North-South seasonal, Coast-Offshore daily
- Peak activity: 4-7 PM (feeding time)
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import centralized data configuration
try:
    from data.data_config import DataConfig, LEGACY_DB
except ImportError:
    # Fallback if data_config not available
    DataConfig = None
    LEGACY_DB = Path("data/real_only/real_data_100.db")


class AnchovyMigrationModel:
    """
    Predicts anchovy concentration zones based on:
    - Historical sightings
    - SST patterns
    - Seasonal migration
    - Time of day
    - Upwelling indicators
    """

    # Anchovy SST preferences (IMARPE data)
    SST_OPTIMAL_MIN = 16.5
    SST_OPTIMAL_MAX = 19.5
    SST_TOLERABLE_MIN = 14.0
    SST_TOLERABLE_MAX = 22.0

    # Seasonal migration patterns (latitude shift by month)
    # Negative = moves south, Positive = moves north
    MONTHLY_LAT_SHIFT = {
        1: -0.05,   # January: slightly south (warm)
        2: -0.08,   # February: more south (warmest)
        3: -0.05,   # March: returning
        4: 0.0,     # April: neutral
        5: 0.05,    # May: moving north (cooling)
        6: 0.10,    # June: north (cold upwelling)
        7: 0.12,    # July: most north
        8: 0.10,    # August: north
        9: 0.05,    # September: returning
        10: 0.0,    # October: neutral
        11: -0.03,  # November: slight south
        12: -0.05   # December: south (warming)
    }

    # Daily offshore movement (km from coast by hour)
    # Anchovy moves offshore at night, returns to coast in afternoon
    HOURLY_OFFSHORE_KM = {
        0: 15, 1: 18, 2: 20, 3: 22, 4: 20, 5: 18,    # Night: offshore
        6: 15, 7: 12, 8: 10, 9: 8, 10: 6, 11: 5,     # Morning: approaching
        12: 4, 13: 3, 14: 3, 15: 2, 16: 2, 17: 3,    # Afternoon: near coast
        18: 4, 19: 6, 20: 8, 21: 10, 22: 12, 23: 14  # Evening: moving out
    }

    # Use centralized config or fallback to legacy path
    DB_PATH = LEGACY_DB if DataConfig is None else DataConfig.LEGACY_DB

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DB_PATH
        self.historical_hotspots: List[Dict] = []
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical fishing hotspots from database."""
        if not self.db_path.exists():
            return

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Get hotspots with significant fishing activity
        cur.execute('''
            SELECT
                ROUND(lat, 2) as lat_zone,
                ROUND(lon, 2) as lon_zone,
                month,
                SUM(fishing_hours) as total_hours,
                AVG(sst) as avg_sst,
                COUNT(*) as observations
            FROM training
            WHERE is_fishing = 1 AND fishing_hours > 0.5
            GROUP BY lat_zone, lon_zone, month
            HAVING total_hours > 2
            ORDER BY total_hours DESC
        ''')

        for row in cur.fetchall():
            self.historical_hotspots.append({
                'lat': row[0],
                'lon': row[1],
                'month': row[2],
                'total_hours': row[3],
                'avg_sst': row[4] or 17.0,
                'observations': row[5]
            })

        conn.close()

    def predict_concentration_zones(
        self,
        target_date: str,
        target_hour: int = 17,
        num_zones: int = 8
    ) -> List[Dict]:
        """
        Predict where anchovy schools are likely concentrated.

        Args:
            target_date: Date in YYYY-MM-DD format
            target_hour: Hour of day (0-23), default 5 PM
            num_zones: Number of zones to return

        Returns:
            List of predicted concentration zones with scores
        """
        dt = datetime.strptime(target_date, '%Y-%m-%d')
        month = dt.month

        # Get base hotspots for this month
        month_hotspots = [h for h in self.historical_hotspots if h['month'] == month]

        if not month_hotspots:
            # Use all hotspots if no month-specific data
            month_hotspots = self.historical_hotspots[:20]

        # Apply migration adjustments
        lat_shift = self.MONTHLY_LAT_SHIFT.get(month, 0)
        offshore_km = self.HOURLY_OFFSHORE_KM.get(target_hour, 5)

        # Score and adjust each hotspot
        predictions = []
        for hotspot in month_hotspots:
            # Adjust latitude for seasonal migration
            adjusted_lat = hotspot['lat'] + lat_shift

            # Adjust longitude for daily offshore movement
            # Negative longitude = west (offshore in Peru)
            offshore_shift = -offshore_km / 111.0  # Convert km to degrees
            adjusted_lon = hotspot['lon'] + offshore_shift * 0.3  # Partial adjustment

            # Calculate score
            score = self._calculate_zone_score(
                hotspot, adjusted_lat, adjusted_lon, month, target_hour
            )

            predictions.append({
                'lat': adjusted_lat,
                'lon': adjusted_lon,
                'original_lat': hotspot['lat'],
                'original_lon': hotspot['lon'],
                'score': score,
                'historical_hours': hotspot['total_hours'],
                'avg_sst': hotspot['avg_sst'],
                'month': month,
                'hour': target_hour,
                'migration_applied': True
            })

        # Sort by score and return top zones
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:num_zones]

    def _calculate_zone_score(
        self,
        hotspot: Dict,
        lat: float,
        lon: float,
        month: int,
        hour: int
    ) -> float:
        """Calculate prediction score for a zone."""
        score = 0.0

        # Historical activity weight (0-40 points)
        hours_score = min(hotspot['total_hours'] / 50 * 40, 40)
        score += hours_score

        # SST optimality (0-30 points)
        sst = hotspot['avg_sst']
        if self.SST_OPTIMAL_MIN <= sst <= self.SST_OPTIMAL_MAX:
            sst_score = 30
        elif self.SST_TOLERABLE_MIN <= sst <= self.SST_TOLERABLE_MAX:
            # Linear decrease outside optimal
            if sst < self.SST_OPTIMAL_MIN:
                sst_score = 30 * (sst - self.SST_TOLERABLE_MIN) / (self.SST_OPTIMAL_MIN - self.SST_TOLERABLE_MIN)
            else:
                sst_score = 30 * (self.SST_TOLERABLE_MAX - sst) / (self.SST_TOLERABLE_MAX - self.SST_OPTIMAL_MAX)
        else:
            sst_score = 0
        score += sst_score

        # Time of day bonus (0-20 points)
        # Peak hours: 4-7 PM (16-19)
        if 16 <= hour <= 19:
            time_score = 20
        elif 14 <= hour <= 20:
            time_score = 15
        elif 10 <= hour <= 22:
            time_score = 10
        else:
            time_score = 5
        score += time_score

        # Observation confidence (0-10 points)
        obs_score = min(hotspot['observations'] / 10 * 10, 10)
        score += obs_score

        return score

    def get_best_fishing_times(self, date: str) -> List[Dict]:
        """Get ranked fishing times for a given date."""
        times = []
        for hour in range(5, 21):  # 5 AM to 8 PM
            zones = self.predict_concentration_zones(date, hour, num_zones=3)
            avg_score = sum(z['score'] for z in zones) / len(zones) if zones else 0

            times.append({
                'hour': hour,
                'time_str': f"{hour:02d}:00",
                'avg_score': avg_score,
                'top_zone': zones[0] if zones else None,
                'offshore_km': self.HOURLY_OFFSHORE_KM.get(hour, 5)
            })

        times.sort(key=lambda x: x['avg_score'], reverse=True)
        return times

    def get_migration_forecast(self, start_date: str, days: int = 7) -> List[Dict]:
        """Get migration forecast for next N days."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        forecasts = []

        for i in range(days):
            forecast_date = start + timedelta(days=i)
            date_str = forecast_date.strftime('%Y-%m-%d')

            # Get best time predictions
            best_times = self.get_best_fishing_times(date_str)[:3]

            # Get concentration zones at peak time (5 PM)
            zones = self.predict_concentration_zones(date_str, 17, num_zones=5)

            forecasts.append({
                'date': date_str,
                'day_name': ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'][forecast_date.weekday()],
                'best_times': best_times,
                'concentration_zones': zones,
                'migration_direction': 'Sur' if self.MONTHLY_LAT_SHIFT.get(forecast_date.month, 0) < 0 else 'Norte',
                'month_pattern': self._get_month_description(forecast_date.month)
            })

        return forecasts

    def _get_month_description(self, month: int) -> str:
        """Get description of typical anchovy behavior for month."""
        patterns = {
            1: "Verano - anchoveta dispersa, cerca de costa",
            2: "Verano calido - cardumenes mas profundos",
            3: "Transicion - reagrupamiento",
            4: "Inicio temporada - concentracion",
            5: "Alta concentracion - surgencia activa",
            6: "Pico temporada fria - maxima concentracion",
            7: "Pico surgencia - anchoveta abundante",
            8: "Surgencia - buenos cardumenes",
            9: "Transicion - dispersion gradual",
            10: "Primavera - patrones mixtos",
            11: "Calentamiento - movimiento sur",
            12: "Pre-verano - cardumenes costeros"
        }
        return patterns.get(month, "Patron variable")

    def add_sighting(
        self,
        date: str,
        lat: float,
        lon: float,
        hours: float,
        notes: str = ""
    ) -> bool:
        """Add a user-reported sighting to improve predictions."""
        if not self.db_path.exists():
            return False

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            # Add to fishing table
            cur.execute('''
                INSERT OR REPLACE INTO fishing
                (date, lat, lon, fishing_hours, vessel_id, flag_state, gear_type, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, lat, lon, hours, f'user_{datetime.now().timestamp()}',
                  'PER', 'artisanal', 'user_report'))

            # Update training if exists
            month = int(date[5:7])
            cur.execute('''
                INSERT OR REPLACE INTO training
                (date, lat, lon, sst, fishing_hours, is_fishing, month, all_real)
                VALUES (?, ?, ?,
                    COALESCE((SELECT AVG(sst) FROM training WHERE month = ?), 18.0),
                    ?, 1, ?, 1)
            ''', (date, round(lat, 2), round(lon, 2), month, hours, month))

            conn.commit()

            # Reload historical data
            self._load_historical_data()

            return True
        except Exception as e:
            print(f"Error adding sighting: {e}")
            return False
        finally:
            conn.close()


def get_anchovy_predictions(target_date: str = None, hour: int = 17) -> Dict:
    """
    Get complete anchovy prediction for map visualization.

    Returns dict with:
    - concentration_zones: Top predicted zones
    - best_times: Ranked fishing times
    - migration_info: Current migration pattern
    """
    if not target_date:
        target_date = datetime.now().strftime('%Y-%m-%d')

    model = AnchovyMigrationModel()

    zones = model.predict_concentration_zones(target_date, hour, num_zones=8)
    times = model.get_best_fishing_times(target_date)
    forecast = model.get_migration_forecast(target_date, days=7)

    return {
        'date': target_date,
        'hour': hour,
        'concentration_zones': zones,
        'best_times': times[:5],
        'weekly_forecast': forecast,
        'total_historical_hotspots': len(model.historical_hotspots)
    }
