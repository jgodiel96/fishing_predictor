"""
Provider para SST historico de Copernicus.

Aprovecha los 354,362 registros ya descargados para:
- Calcular anomalias termicas vs promedio mensual
- Detectar tendencias de temperatura
- Mejorar la precision de frentes termicos

La SST historica permite:
1. Detectar anomalias termicas - desviaciones del promedio historico
2. Identificar tendencias - calentamiento/enfriamiento estacional
3. Validar frentes termicos - comparar con promedios mensuales

Uso:
    provider = SSTHistoricalProvider()
    result = provider.get_sst_with_anomaly('2024-06-15', -17.8, -71.2)
    print(f"SST: {result.sst:.1f}C, Anomalia: {result.anomaly:.2f}C")
    trend = provider.get_temperature_trend('2024-06-15', -17.8, -71.2, days=30)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, NamedTuple
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from data.data_config import DataConfig

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES DE DOMINIO
# =============================================================================

SST_OPTIMAL_MIN: float = 14.0       # C - temperatura minima optima
SST_OPTIMAL_MAX: float = 24.0       # C - temperatura maxima optima
SST_OPTIMAL_CENTER: float = 19.0    # C - temperatura optima para especies locales
ANOMALY_THRESHOLD_WARM: float = 1.5 # C - anomalia calida significativa
ANOMALY_THRESHOLD_COLD: float = -1.5  # C - anomalia fria significativa (upwelling)


class SSTResult(NamedTuple):
    """Resultado de consulta SST."""
    sst: float              # Temperatura actual en C
    anomaly: float          # Desviacion del promedio mensual
    monthly_mean: float     # Promedio historico del mes
    trend_7d: float         # Tendencia ultimos 7 dias
    score: float            # Score normalizado 0-1


@dataclass(frozen=True, slots=True)
class SSTConfig:
    """Configuracion inmutable para SST provider."""
    data_dir: Path
    monthly_stats_cache: Path
    grid_resolution_deg: float = 0.05


class SSTHistoricalProvider:
    """
    Provider para datos historicos de SST.

    Caracteristicas de alta eficiencia:
    - Indice espacial KD-Tree para busquedas O(log N)
    - Estadisticas mensuales pre-calculadas en cache
    - Cache multinivel (memoria + disco)
    - Tipos de datos optimizados (float32)

    Performance esperado:
    - Query individual: ~1ms
    - 1000 queries: ~100ms (vs ~10s sin indice)

    Atributos:
        config: Configuracion del provider
        _data: DataFrame con todos los datos SST
        _spatial_index: KD-Tree para busquedas espaciales
        _monthly_stats: Estadisticas mensuales pre-calculadas
    """

    __slots__ = (
        'config', '_data', '_spatial_index',
        '_monthly_stats', '_cache', '_unique_coords'
    )

    def __init__(self, config: Optional[SSTConfig] = None):
        """
        Inicializa el provider.

        Args:
            config: Configuracion opcional
        """
        if config is None:
            config = SSTConfig(
                data_dir=DataConfig.RAW_SST_COPERNICUS,
                monthly_stats_cache=DataConfig.PROCESSED_DIR / "sst_monthly_stats.parquet"
            )

        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._spatial_index = None
        self._monthly_stats: Optional[pd.DataFrame] = None
        self._cache: Dict[str, float] = {}
        self._unique_coords: Optional[np.ndarray] = None

    def _ensure_loaded(self) -> None:
        """Carga datos si no estan en memoria (lazy loading)."""
        if self._data is not None:
            return

        logger.info("Cargando SST historico...")

        # Buscar archivos parquet
        parquet_files = list(self.config.data_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No se encontraron archivos SST en {self.config.data_dir}")
            # Crear DataFrame vacio con estructura correcta
            self._data = pd.DataFrame(columns=['date', 'lat', 'lon', 'sst', 'month', 'year'])
            return

        dfs = []
        for f in parquet_files:
            try:
                # Solo cargar columnas necesarias para eficiencia
                df = pd.read_parquet(f)

                # Normalizar nombres de columnas
                if 'analysed_sst' in df.columns and 'sst' not in df.columns:
                    df['sst'] = df['analysed_sst']

                # Asegurar columnas minimas
                required = ['lat', 'lon']
                if not all(c in df.columns for c in required):
                    logger.warning(f"Archivo {f.name} no tiene columnas requeridas, saltando")
                    continue

                # Seleccionar columnas relevantes
                cols_to_keep = [c for c in ['date', 'time', 'lat', 'lon', 'sst', 'analysed_sst'] if c in df.columns]
                df = df[cols_to_keep]

                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error leyendo {f.name}: {e}")
                continue

        if not dfs:
            logger.warning("No se pudo cargar ningun archivo SST")
            self._data = pd.DataFrame(columns=['date', 'lat', 'lon', 'sst', 'month', 'year'])
            return

        self._data = pd.concat(dfs, ignore_index=True)

        # Normalizar columna de fecha
        if 'time' in self._data.columns and 'date' not in self._data.columns:
            self._data['date'] = self._data['time']
        if 'analysed_sst' in self._data.columns and 'sst' not in self._data.columns:
            self._data['sst'] = self._data['analysed_sst']

        # Convertir SST de Kelvin a Celsius si es necesario
        if 'sst' in self._data.columns and self._data['sst'].mean() > 100:
            self._data['sst'] = self._data['sst'] - 273.15

        # Optimizar tipos
        self._data['lat'] = self._data['lat'].astype(np.float32)
        self._data['lon'] = self._data['lon'].astype(np.float32)
        if 'sst' in self._data.columns:
            self._data['sst'] = self._data['sst'].astype(np.float32)

        # Parsear fechas
        if 'date' in self._data.columns:
            self._data['date'] = pd.to_datetime(self._data['date'])
            self._data['month'] = self._data['date'].dt.month.astype(np.uint8)
            self._data['year'] = self._data['date'].dt.year.astype(np.uint16)

        # Construir indice espacial si scipy disponible
        if HAS_SCIPY and len(self._data) > 0:
            self._unique_coords = self._data[['lat', 'lon']].drop_duplicates().values.astype(np.float32)
            self._spatial_index = cKDTree(self._unique_coords)
            logger.debug(f"Indice espacial creado con {len(self._unique_coords)} puntos unicos")

        # Calcular estadisticas mensuales
        self._calculate_monthly_stats()

        logger.info(f"SST cargado: {len(self._data):,} registros")

    def _calculate_monthly_stats(self) -> None:
        """Calcula estadisticas mensuales por ubicacion."""
        cache_path = self.config.monthly_stats_cache

        # Intentar cargar desde cache
        if cache_path.exists():
            try:
                self._monthly_stats = pd.read_parquet(cache_path)
                logger.debug("Estadisticas mensuales cargadas desde cache")
                return
            except Exception as e:
                logger.warning(f"Error cargando cache de estadisticas: {e}")

        if self._data is None or self._data.empty or 'sst' not in self._data.columns:
            self._monthly_stats = pd.DataFrame()
            return

        logger.info("Calculando estadisticas mensuales SST...")

        # Redondear coordenadas para agrupar (0.05 grados ~ 5km)
        data_rounded = self._data.copy()
        data_rounded['lat_grid'] = (data_rounded['lat'] / 0.05).round() * 0.05
        data_rounded['lon_grid'] = (data_rounded['lon'] / 0.05).round() * 0.05

        # Agrupar por ubicacion y mes
        stats = data_rounded.groupby(
            ['lat_grid', 'lon_grid', 'month'],
            observed=True
        ).agg(
            sst_mean=('sst', 'mean'),
            sst_std=('sst', 'std'),
            sst_min=('sst', 'min'),
            sst_max=('sst', 'max'),
            count=('sst', 'count')
        ).reset_index()

        # Renombrar columnas
        stats = stats.rename(columns={'lat_grid': 'lat', 'lon_grid': 'lon'})

        # Optimizar tipos
        stats['sst_mean'] = stats['sst_mean'].astype(np.float32)
        stats['sst_std'] = stats['sst_std'].fillna(0).astype(np.float32)
        stats['sst_min'] = stats['sst_min'].astype(np.float32)
        stats['sst_max'] = stats['sst_max'].astype(np.float32)
        stats['count'] = stats['count'].astype(np.uint16)

        # Guardar cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            stats.to_parquet(cache_path, compression='snappy')
            logger.debug(f"Estadisticas guardadas en {cache_path}")
        except Exception as e:
            logger.warning(f"Error guardando cache de estadisticas: {e}")

        self._monthly_stats = stats

    @lru_cache(maxsize=512)
    def get_sst_with_anomaly(
        self,
        date: str,
        lat: float,
        lon: float
    ) -> SSTResult:
        """
        Obtiene SST y anomalia para ubicacion y fecha.

        Args:
            date: Fecha ISO (YYYY-MM-DD)
            lat: Latitud
            lon: Longitud

        Returns:
            SSTResult con sst, anomaly, monthly_mean, trend_7d, score
        """
        self._ensure_loaded()

        # Redondear para mejorar cache hits
        lat = round(lat, 4)
        lon = round(lon, 4)

        target_date = pd.Timestamp(date)
        month = target_date.month

        # Obtener promedio mensual historico
        monthly_mean = self._get_monthly_mean(lat, lon, month)

        if self._data is None or self._data.empty or 'sst' not in self._data.columns:
            # Fallback si no hay datos
            return SSTResult(
                sst=monthly_mean,
                anomaly=0.0,
                monthly_mean=monthly_mean,
                trend_7d=0.0,
                score=0.5
            )

        # Buscar punto mas cercano
        if self._spatial_index is not None and self._unique_coords is not None:
            _, idx = self._spatial_index.query([lat, lon])
            nearest_lat, nearest_lon = self._unique_coords[idx]
        else:
            nearest_lat, nearest_lon = lat, lon

        # Obtener SST del dia
        mask = (
            (np.abs(self._data['lat'] - nearest_lat) < 0.1) &
            (np.abs(self._data['lon'] - nearest_lon) < 0.1) &
            (self._data['date'] == target_date)
        )

        daily_data = self._data.loc[mask, 'sst']

        if daily_data.empty:
            # Fallback: usar promedio del mes
            sst = monthly_mean
            anomaly = 0.0
        else:
            sst = float(daily_data.iloc[0])
            anomaly = sst - monthly_mean

        # Calcular tendencia 7 dias
        trend = self._calculate_trend(nearest_lat, nearest_lon, target_date, days=7)

        # Calcular score
        score = self.calculate_score(sst, anomaly)

        return SSTResult(
            sst=round(sst, 2),
            anomaly=round(anomaly, 2),
            monthly_mean=round(monthly_mean, 2),
            trend_7d=round(trend, 3),
            score=round(score, 3)
        )

    def _get_monthly_mean(self, lat: float, lon: float, month: int) -> float:
        """Obtiene promedio mensual historico."""
        if self._monthly_stats is None or self._monthly_stats.empty:
            return SST_OPTIMAL_CENTER

        # Redondear a grid
        lat_grid = round(lat / 0.05) * 0.05
        lon_grid = round(lon / 0.05) * 0.05

        mask = (
            (np.abs(self._monthly_stats['lat'] - lat_grid) < 0.01) &
            (np.abs(self._monthly_stats['lon'] - lon_grid) < 0.01) &
            (self._monthly_stats['month'] == month)
        )

        result = self._monthly_stats.loc[mask, 'sst_mean']

        if result.empty:
            return SST_OPTIMAL_CENTER

        return float(result.iloc[0])

    def _calculate_trend(
        self,
        lat: float,
        lon: float,
        end_date: pd.Timestamp,
        days: int = 7
    ) -> float:
        """
        Calcula tendencia de temperatura.

        Args:
            lat: Latitud
            lon: Longitud
            end_date: Fecha final
            days: Numero de dias para tendencia

        Returns:
            Cambio de temperatura en C (positivo = calentamiento)
        """
        if self._data is None or self._data.empty:
            return 0.0

        start_date = end_date - pd.Timedelta(days=days)

        mask = (
            (np.abs(self._data['lat'] - lat) < 0.1) &
            (np.abs(self._data['lon'] - lon) < 0.1) &
            (self._data['date'] >= start_date) &
            (self._data['date'] <= end_date)
        )

        period_data = self._data.loc[mask].sort_values('date')

        if len(period_data) < 2:
            return 0.0

        # Regresion lineal simple
        x = np.arange(len(period_data))
        y = period_data['sst'].values

        if len(x) > 1:
            try:
                slope = np.polyfit(x, y, 1)[0]
                return float(slope * days)  # Cambio total en el periodo
            except:
                return 0.0

        return 0.0

    @staticmethod
    def calculate_score(sst: float, anomaly: float = 0.0) -> float:
        """
        Calcula score combinado de SST y anomalia.

        Args:
            sst: Temperatura en C
            anomaly: Desviacion del promedio mensual en C

        Returns:
            Score normalizado 0-1

        Logica:
        - SST optima (14-24C): Score base alto
        - Anomalia fria (upwelling): Bonus
        - Anomalia calida: Penalizacion
        """
        # Score base por temperatura absoluta
        if SST_OPTIMAL_MIN <= sst <= SST_OPTIMAL_MAX:
            # Dentro del rango optimo - score alto
            dist_from_center = abs(sst - SST_OPTIMAL_CENTER)
            range_half = (SST_OPTIMAL_MAX - SST_OPTIMAL_MIN) / 2
            base_score = 1.0 - (dist_from_center / range_half) * 0.2
        else:
            # Fuera del rango optimo
            if sst < SST_OPTIMAL_MIN:
                dist = SST_OPTIMAL_MIN - sst
            else:
                dist = sst - SST_OPTIMAL_MAX
            base_score = max(0.3, 0.7 - dist * 0.1)

        # Ajuste por anomalia
        anomaly_factor = 1.0
        if anomaly > ANOMALY_THRESHOLD_WARM:
            # Anomalia calida - puede ser negativo para pesca
            anomaly_factor = 0.85
        elif anomaly < ANOMALY_THRESHOLD_COLD:
            # Anomalia fria - upwelling, generalmente bueno
            anomaly_factor = 1.1

        return min(1.0, max(0.0, base_score * anomaly_factor))

    def get_temperature_trend(
        self,
        date: str,
        lat: float,
        lon: float,
        days: int = 30
    ) -> Dict:
        """
        Obtiene analisis de tendencia extendido.

        Args:
            date: Fecha ISO
            lat: Latitud
            lon: Longitud
            days: Dias para analisis

        Returns:
            Dict con trend, classification, confidence
        """
        self._ensure_loaded()

        target_date = pd.Timestamp(date)

        # Buscar punto mas cercano
        if self._spatial_index is not None and self._unique_coords is not None:
            _, idx = self._spatial_index.query([lat, lon])
            nearest_lat, nearest_lon = self._unique_coords[idx]
        else:
            nearest_lat, nearest_lon = lat, lon

        trend = self._calculate_trend(nearest_lat, nearest_lon, target_date, days)

        # Clasificar tendencia
        if trend > 1.0:
            classification = "warming_strong"
        elif trend > 0.3:
            classification = "warming_moderate"
        elif trend < -1.0:
            classification = "cooling_strong"
        elif trend < -0.3:
            classification = "cooling_moderate"
        else:
            classification = "stable"

        return {
            'trend_celsius': round(trend, 3),
            'classification': classification,
            'period_days': days,
            'confidence': min(1.0, days / 30)
        }

    def get_statistics_summary(self) -> Dict:
        """Obtiene resumen de estadisticas de los datos cargados."""
        self._ensure_loaded()

        if self._data is None or self._data.empty:
            return {'status': 'no_data'}

        return {
            'total_records': len(self._data),
            'unique_locations': len(self._unique_coords) if self._unique_coords is not None else 0,
            'date_range': {
                'start': str(self._data['date'].min()) if 'date' in self._data.columns else None,
                'end': str(self._data['date'].max()) if 'date' in self._data.columns else None
            },
            'sst_range': {
                'min': float(self._data['sst'].min()) if 'sst' in self._data.columns else None,
                'max': float(self._data['sst'].max()) if 'sst' in self._data.columns else None,
                'mean': float(self._data['sst'].mean()) if 'sst' in self._data.columns else None
            },
            'spatial_index': 'enabled' if self._spatial_index is not None else 'disabled'
        }

    def clear_cache(self) -> None:
        """Limpia todas las caches."""
        self._cache.clear()
        self.get_sst_with_anomaly.cache_clear()


# =============================================================================
# FUNCION DE CONVENIENCIA
# =============================================================================

def get_sst_historical_score(date: str, lat: float, lon: float) -> float:
    """
    Funcion de conveniencia para obtener score SST historico.

    Args:
        date: Fecha ISO
        lat: Latitud
        lon: Longitud

    Returns:
        Score 0-1
    """
    provider = SSTHistoricalProvider()
    result = provider.get_sst_with_anomaly(date, lat, lon)
    return result.score
