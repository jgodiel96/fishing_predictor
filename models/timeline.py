"""
Timeline analysis module for historical and forecast data.
Provides temporal analysis capabilities for fishing predictions.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, NamedTuple
from pathlib import Path

# Import centralized data configuration
try:
    from data.data_config import DataConfig, LEGACY_DB
except ImportError:
    # Fallback if data_config not available
    DataConfig = None
    LEGACY_DB = Path("data/real_only/real_data_100.db")


class DailyStats(NamedTuple):
    """Statistics for a single day."""
    date: str
    total_records: int
    fishing_events: int
    fishing_hours: float
    avg_sst: float
    avg_wave: float
    fishing_rate: float


class MonthlyStats(NamedTuple):
    """Statistics for a month."""
    month: int
    month_name: str
    total_records: int
    fishing_events: int
    total_hours: float
    avg_sst: float
    avg_wave: float
    fishing_rate: float


class ZoneActivity(NamedTuple):
    """Fishing activity for a zone."""
    lat: float
    lon: float
    total_hours: float
    event_count: int
    intensity: float


class TimelineAnalyzer:
    """
    Analyzes temporal patterns in fishing data.
    Provides historical analysis and forecasting capabilities.
    """

    MONTH_NAMES = (
        'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
    )

    # Use centralized config or fallback to legacy path
    DB_PATH = LEGACY_DB if DataConfig is None else DataConfig.LEGACY_DB

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DB_PATH

    def get_date_range(self) -> Tuple[str, str]:
        """Get the available date range in the database."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT MIN(date), MAX(date) FROM training")
        result = cur.fetchone()
        conn.close()
        return result if result[0] else ('2020-01-01', '2026-01-31')

    def get_daily_stats(self, date: str) -> Optional[DailyStats]:
        """Get statistics for a specific date."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute('''
            SELECT
                COUNT(*) as total,
                SUM(is_fishing) as events,
                SUM(fishing_hours) as hours,
                AVG(sst) as avg_sst,
                AVG(wave_height) as avg_wave
            FROM training
            WHERE date = ?
        ''', (date,))

        row = cur.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return None

        total, events, hours, avg_sst, avg_wave = row
        events = events or 0
        hours = hours or 0

        return DailyStats(
            date=date,
            total_records=total,
            fishing_events=int(events),
            fishing_hours=float(hours),
            avg_sst=float(avg_sst) if avg_sst else 17.0,
            avg_wave=float(avg_wave) if avg_wave else 1.5,
            fishing_rate=events / total * 100 if total > 0 else 0
        )

    def get_monthly_stats(self) -> List[MonthlyStats]:
        """Get statistics grouped by month."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute('''
            SELECT
                month,
                COUNT(*) as total,
                SUM(is_fishing) as events,
                SUM(fishing_hours) as hours,
                AVG(sst) as avg_sst,
                AVG(wave_height) as avg_wave
            FROM training
            GROUP BY month
            ORDER BY month
        ''')

        results = []
        for row in cur.fetchall():
            month, total, events, hours, avg_sst, avg_wave = row
            events = events or 0
            hours = hours or 0

            results.append(MonthlyStats(
                month=month,
                month_name=self.MONTH_NAMES[month - 1],
                total_records=total,
                fishing_events=int(events),
                total_hours=float(hours),
                avg_sst=float(avg_sst) if avg_sst else 17.0,
                avg_wave=float(avg_wave) if avg_wave else 1.5,
                fishing_rate=events / total * 100 if total > 0 else 0
            ))

        conn.close()
        return results

    def get_weekly_forecast(self, start_date: str = None) -> List[Dict]:
        """
        Generate 7-day forecast based on historical patterns.
        Uses same-month historical data for prediction.
        """
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')

        start = datetime.strptime(start_date, '%Y-%m-%d')
        forecasts = []

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        for i in range(7):
            forecast_date = start + timedelta(days=i)
            month = forecast_date.month
            day_of_week = forecast_date.weekday()

            # Get historical average for this month
            cur.execute('''
                SELECT
                    AVG(sst) as avg_sst,
                    AVG(wave_height) as avg_wave,
                    AVG(CASE WHEN is_fishing = 1 THEN 1.0 ELSE 0.0 END) * 100 as fishing_rate,
                    SUM(fishing_hours) / COUNT(DISTINCT date) as avg_daily_hours
                FROM training
                WHERE month = ?
            ''', (month,))

            row = cur.fetchone()

            # Adjust based on day of week (weekends typically have more fishing)
            weekend_factor = 1.2 if day_of_week >= 5 else 1.0

            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'day_name': ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'][day_of_week],
                'month': month,
                'predicted_sst': float(row[0]) if row[0] else 17.0,
                'predicted_wave': float(row[1]) if row[1] else 1.5,
                'fishing_probability': float(row[2] * weekend_factor) if row[2] else 2.0,
                'expected_hours': float(row[3] * weekend_factor) if row[3] else 10.0,
                'confidence': self._calculate_confidence(month, cur)
            })

        conn.close()
        return forecasts

    def get_heatmap_data(self, date: str = None, month: int = None) -> List[ZoneActivity]:
        """
        Get fishing activity heatmap data.
        If date is provided, shows that specific date.
        If month is provided, shows monthly aggregate.
        Otherwise, shows all-time data.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        if date:
            query = '''
                SELECT
                    ROUND(lat, 2) as lat_zone,
                    ROUND(lon, 2) as lon_zone,
                    SUM(fishing_hours) as total_hours,
                    COUNT(*) as events
                FROM training
                WHERE date = ? AND is_fishing = 1
                GROUP BY lat_zone, lon_zone
            '''
            cur.execute(query, (date,))
        elif month:
            query = '''
                SELECT
                    ROUND(lat, 2) as lat_zone,
                    ROUND(lon, 2) as lon_zone,
                    SUM(fishing_hours) as total_hours,
                    COUNT(*) as events
                FROM training
                WHERE month = ? AND is_fishing = 1
                GROUP BY lat_zone, lon_zone
            '''
            cur.execute(query, (month,))
        else:
            query = '''
                SELECT
                    ROUND(lat, 2) as lat_zone,
                    ROUND(lon, 2) as lon_zone,
                    SUM(fishing_hours) as total_hours,
                    COUNT(*) as events
                FROM training
                WHERE is_fishing = 1
                GROUP BY lat_zone, lon_zone
            '''
            cur.execute(query)

        results = []
        max_hours = 1
        rows = cur.fetchall()

        if rows:
            max_hours = max(r[2] for r in rows if r[2]) or 1

        for row in rows:
            lat, lon, hours, events = row
            hours = hours or 0

            results.append(ZoneActivity(
                lat=float(lat),
                lon=float(lon),
                total_hours=float(hours),
                event_count=int(events),
                intensity=min(hours / max_hours, 1.0)
            ))

        conn.close()
        return results

    def get_yearly_trend(self) -> List[Dict]:
        """Get yearly fishing trends."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute('''
            SELECT
                SUBSTR(date, 1, 4) as year,
                COUNT(*) as total,
                SUM(is_fishing) as events,
                SUM(fishing_hours) as hours,
                AVG(sst) as avg_sst
            FROM training
            GROUP BY year
            ORDER BY year
        ''')

        results = []
        for row in cur.fetchall():
            year, total, events, hours, avg_sst = row
            events = events or 0
            hours = hours or 0

            results.append({
                'year': year,
                'total_records': total,
                'fishing_events': int(events),
                'total_hours': float(hours),
                'avg_sst': float(avg_sst) if avg_sst else 17.0,
                'fishing_rate': events / total * 100 if total > 0 else 0
            })

        conn.close()
        return results

    def get_best_dates_in_month(self, month: int, top_n: int = 5) -> List[Dict]:
        """Get historically best fishing dates for a given month."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute('''
            SELECT
                date,
                SUM(fishing_hours) as hours,
                COUNT(*) as events,
                AVG(sst) as avg_sst,
                AVG(wave_height) as avg_wave
            FROM training
            WHERE month = ? AND is_fishing = 1
            GROUP BY date
            ORDER BY hours DESC
            LIMIT ?
        ''', (month, top_n))

        results = []
        for row in cur.fetchall():
            date, hours, events, avg_sst, avg_wave = row
            results.append({
                'date': date,
                'fishing_hours': float(hours) if hours else 0,
                'events': int(events),
                'avg_sst': float(avg_sst) if avg_sst else 17.0,
                'avg_wave': float(avg_wave) if avg_wave else 1.5
            })

        conn.close()
        return results

    def _calculate_confidence(self, month: int, cursor) -> float:
        """Calculate prediction confidence based on data availability."""
        cursor.execute(
            "SELECT COUNT(DISTINCT date) FROM training WHERE month = ?",
            (month,)
        )
        days = cursor.fetchone()[0]

        # More data = higher confidence (max at ~180 days = 6 years)
        return min(days / 180 * 100, 95)

    def get_conditions_for_date(self, date: str) -> Dict:
        """Get detailed conditions for a specific date."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute('''
            SELECT
                AVG(sst) as avg_sst,
                MIN(sst) as min_sst,
                MAX(sst) as max_sst,
                AVG(wave_height) as avg_wave,
                AVG(wind_speed) as avg_wind,
                SUM(fishing_hours) as total_hours,
                SUM(is_fishing) as fishing_events,
                COUNT(*) as total_records
            FROM training
            WHERE date = ?
        ''', (date,))

        row = cur.fetchone()
        conn.close()

        if not row or row[7] == 0:
            return {'date': date, 'available': False}

        return {
            'date': date,
            'available': True,
            'avg_sst': float(row[0]) if row[0] else None,
            'min_sst': float(row[1]) if row[1] else None,
            'max_sst': float(row[2]) if row[2] else None,
            'avg_wave': float(row[3]) if row[3] else None,
            'avg_wind': float(row[4]) if row[4] else None,
            'total_hours': float(row[5]) if row[5] else 0,
            'fishing_events': int(row[6]) if row[6] else 0,
            'total_records': int(row[7])
        }


def generate_timeline_data() -> Dict:
    """
    Generate all timeline data for the map visualization.
    Returns a dictionary ready to be serialized to JSON.
    """
    analyzer = TimelineAnalyzer()

    date_range = analyzer.get_date_range()
    monthly_stats = analyzer.get_monthly_stats()
    weekly_forecast = analyzer.get_weekly_forecast()
    yearly_trend = analyzer.get_yearly_trend()
    heatmap_data = analyzer.get_heatmap_data()

    # Get today's stats
    today = datetime.now().strftime('%Y-%m-%d')
    today_stats = analyzer.get_daily_stats(today)

    return {
        'date_range': {
            'min': date_range[0],
            'max': date_range[1]
        },
        'today': today,
        'today_stats': today_stats._asdict() if today_stats else None,
        'monthly_stats': [m._asdict() for m in monthly_stats],
        'weekly_forecast': weekly_forecast,
        'yearly_trend': yearly_trend,
        'heatmap': [
            {
                'lat': z.lat,
                'lon': z.lon,
                'hours': z.total_hours,
                'events': z.event_count,
                'intensity': z.intensity
            }
            for z in heatmap_data
        ]
    }
