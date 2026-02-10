"""
Generador de hotspots dinamicos basado en datos GFW.

Usa clustering DBSCAN para identificar zonas de alta
actividad pesquera real a partir de datos AIS de Global Fishing Watch.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
- No requiere especificar numero de clusters
- Detecta clusters de forma irregular
- Identifica outliers (ruido)

Parametros DBSCAN:
- eps: Radio de vecindad (2km = 0.018 grados)
- min_samples: Minimo de puntos para formar cluster (5)

Uso:
    generator = GFWHotspotGenerator()
    hotspots = generator.generate_hotspots(min_fishing_hours=10)

    # Por temporada
    hotspots_summer = generator.generate_seasonal_hotspots(
        season='summer',
        min_samples=5
    )

    # Bonus por proximidad
    bonus = generator.calculate_proximity_bonus(-17.8, -71.2)
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, NamedTuple

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from data.data_config import DataConfig

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES
# =============================================================================

EARTH_RADIUS_KM: float = 6371.0
DEFAULT_EPS_KM: float = 2.0        # Radio de vecindad para DBSCAN
DEFAULT_MIN_SAMPLES: int = 5       # Minimo puntos por cluster
DEFAULT_MIN_FISHING_HOURS: float = 5.0  # Minimo horas de pesca


class DynamicHotspot(NamedTuple):
    """Hotspot generado dinamicamente desde datos GFW."""
    id: int                          # ID del cluster
    lat: float                       # Latitud del centroide
    lon: float                       # Longitud del centroide
    fishing_hours: float             # Total de horas de pesca
    vessel_count: int                # Numero de embarcaciones
    score: float                     # Score normalizado 0.5-1.0
    source: str = "GFW_dynamic"      # Fuente de datos
    season: Optional[str] = None     # Temporada si aplica


@dataclass(slots=True)
class HotspotConfig:
    """Configuracion para generacion de hotspots."""
    data_path: Path
    eps_km: float = DEFAULT_EPS_KM
    min_samples: int = DEFAULT_MIN_SAMPLES
    min_fishing_hours: float = DEFAULT_MIN_FISHING_HOURS
    output_cache: Path = field(
        default_factory=lambda: DataConfig.PROCESSED_DIR / "dynamic_hotspots.parquet"
    )


class GFWHotspotGenerator:
    """
    Generador de hotspots pesqueros dinamicos.

    Usa datos reales de Global Fishing Watch y clustering DBSCAN
    para identificar zonas de alta actividad pesquera.

    Caracteristicas de eficiencia:
    - Clustering espacial DBSCAN (O(n log n) con tree)
    - Indice KD-Tree para busquedas de proximidad
    - Cache de resultados para evitar recalculos
    - Tipos de datos optimizados (float32)

    Atributos:
        config: Configuracion del generador
        _data: DataFrame con datos GFW cargados
        _hotspots_cache: Cache de hotspots por configuracion
    """

    __slots__ = ('config', '_data', '_hotspots_cache', '_spatial_index')

    def __init__(self, config: Optional[HotspotConfig] = None):
        """
        Inicializa el generador.

        Args:
            config: Configuracion opcional
        """
        if config is None:
            config = HotspotConfig(
                data_path=DataConfig.RAW_GFW
            )

        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._hotspots_cache: Dict[str, List[DynamicHotspot]] = {}
        self._spatial_index = None

    def _ensure_loaded(self) -> None:
        """Carga datos GFW si no estan en memoria."""
        if self._data is not None:
            return

        logger.info("Cargando datos GFW...")

        parquet_files = list(self.config.data_path.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No se encontraron archivos GFW en {self.config.data_path}")
            self._data = pd.DataFrame()
            return

        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error leyendo {f.name}: {e}")
                continue

        if not dfs:
            logger.warning("No se pudo cargar ningun archivo GFW")
            self._data = pd.DataFrame()
            return

        self._data = pd.concat(dfs, ignore_index=True)

        # Estandarizar nombres de columnas
        col_mapping = {
            'cell_ll_lat': 'lat',
            'cell_ll_lon': 'lon',
            'Lat': 'lat',
            'Lon': 'lon',
            'fishing_hours': 'fishing_hours',
            'apparent_fishing_hours': 'fishing_hours',
            'mmsi_present': 'vessel_count'
        }

        for old, new in col_mapping.items():
            if old in self._data.columns and new not in self._data.columns:
                self._data[new] = self._data[old]

        # Asegurar columnas requeridas
        if 'lat' not in self._data.columns or 'lon' not in self._data.columns:
            logger.error("Datos GFW no tienen columnas lat/lon")
            self._data = pd.DataFrame()
            return

        if 'fishing_hours' not in self._data.columns:
            # Usar valor por defecto
            self._data['fishing_hours'] = 1.0

        if 'vessel_count' not in self._data.columns:
            self._data['vessel_count'] = 1

        # Optimizar tipos
        self._data['lat'] = self._data['lat'].astype(np.float32)
        self._data['lon'] = self._data['lon'].astype(np.float32)
        self._data['fishing_hours'] = self._data['fishing_hours'].astype(np.float32)

        # Agregar columna de mes si hay fecha
        date_cols = ['date', 'time', 'Time Range']
        for col in date_cols:
            if col in self._data.columns:
                try:
                    self._data['date'] = pd.to_datetime(self._data[col])
                    self._data['month'] = self._data['date'].dt.month.astype(np.uint8)
                    break
                except:
                    pass

        # Filtrar coordenadas invalidas
        self._data = self._data[
            (self._data['lat'].notna()) &
            (self._data['lon'].notna()) &
            (self._data['lat'].between(-90, 90)) &
            (self._data['lon'].between(-180, 180))
        ]

        logger.info(f"GFW cargado: {len(self._data):,} registros")

    def generate_hotspots(
        self,
        min_fishing_hours: Optional[float] = None,
        eps_km: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> List[DynamicHotspot]:
        """
        Genera hotspots usando clustering DBSCAN.

        Args:
            min_fishing_hours: Minimo de horas de pesca para incluir punto
            eps_km: Radio de vecindad en km
            min_samples: Minimo de puntos por cluster

        Returns:
            Lista de DynamicHotspot ordenados por score descendente
        """
        if not HAS_SKLEARN:
            logger.error("sklearn no instalado. Ejecuta: pip install scikit-learn")
            return []

        self._ensure_loaded()

        if self._data is None or self._data.empty:
            logger.warning("No hay datos GFW para generar hotspots")
            return []

        # Usar valores por defecto si no se especifican
        min_fishing_hours = min_fishing_hours or self.config.min_fishing_hours
        eps_km = eps_km or self.config.eps_km
        min_samples = min_samples or self.config.min_samples

        # Clave de cache
        cache_key = f"all_{min_fishing_hours}_{eps_km}_{min_samples}"
        if cache_key in self._hotspots_cache:
            return self._hotspots_cache[cache_key]

        # Filtrar por minimo de horas
        filtered = self._data[
            self._data['fishing_hours'] >= min_fishing_hours
        ].copy()

        if filtered.empty:
            logger.warning(f"No hay datos despues de filtrar por fishing_hours >= {min_fishing_hours}")
            return []

        logger.info(f"Ejecutando DBSCAN en {len(filtered)} puntos (eps={eps_km}km, min_samples={min_samples})...")

        # Preparar coordenadas para DBSCAN
        coords = filtered[['lat', 'lon']].values

        # Convertir eps de km a grados (aproximacion: 1 grado ~ 111km)
        eps_deg = eps_km / 111.0

        # Ejecutar DBSCAN
        clustering = DBSCAN(
            eps=eps_deg,
            min_samples=min_samples,
            metric='euclidean',  # Aproximacion valida para areas pequenas
            n_jobs=-1  # Usar todos los cores
        )

        filtered['cluster'] = clustering.fit_predict(coords)

        # Filtrar ruido (cluster = -1)
        clustered = filtered[filtered['cluster'] != -1]

        if clustered.empty:
            logger.warning("No se encontraron clusters")
            return []

        # Calcular centroides y metricas por cluster
        hotspots = []

        for cluster_id in clustered['cluster'].unique():
            cluster_data = clustered[clustered['cluster'] == cluster_id]

            # Centroide ponderado por fishing_hours
            total_hours = cluster_data['fishing_hours'].sum()

            if total_hours > 0:
                weighted_lat = (
                    cluster_data['lat'] * cluster_data['fishing_hours']
                ).sum() / total_hours
                weighted_lon = (
                    cluster_data['lon'] * cluster_data['fishing_hours']
                ).sum() / total_hours
            else:
                weighted_lat = cluster_data['lat'].mean()
                weighted_lon = cluster_data['lon'].mean()

            # Metricas
            vessel_count = int(cluster_data['vessel_count'].sum()) if 'vessel_count' in cluster_data else len(cluster_data)

            hotspots.append(DynamicHotspot(
                id=int(cluster_id),
                lat=float(weighted_lat),
                lon=float(weighted_lon),
                fishing_hours=float(total_hours),
                vessel_count=vessel_count,
                score=0.0,  # Se calcula despues
                source="GFW_dynamic"
            ))

        # Calcular scores normalizados
        hotspots = self._calculate_scores(hotspots)

        # Ordenar por score descendente
        hotspots.sort(key=lambda h: h.score, reverse=True)

        # Guardar en cache
        self._hotspots_cache[cache_key] = hotspots

        logger.info(f"Generados {len(hotspots)} hotspots dinamicos")

        return hotspots

    def generate_seasonal_hotspots(
        self,
        season: str,
        **kwargs
    ) -> List[DynamicHotspot]:
        """
        Genera hotspots filtrados por temporada.

        Args:
            season: 'summer' (Dic-Mar), 'winter' (Jun-Sep),
                   'spring' (Sep-Dic), 'autumn' (Mar-Jun)
            **kwargs: Argumentos adicionales para generate_hotspots

        Returns:
            Lista de DynamicHotspot para la temporada
        """
        self._ensure_loaded()

        if self._data is None or self._data.empty:
            return []

        if 'month' not in self._data.columns:
            logger.warning("No hay datos temporales, usando todos los registros")
            return self.generate_hotspots(**kwargs)

        # Mapear temporada a meses (Hemisferio Sur)
        season_months = {
            'summer': [12, 1, 2, 3],    # Verano: Dic-Mar
            'autumn': [3, 4, 5, 6],      # Otono: Mar-Jun
            'winter': [6, 7, 8, 9],      # Invierno: Jun-Sep
            'spring': [9, 10, 11, 12]    # Primavera: Sep-Dic
        }

        if season not in season_months:
            raise ValueError(f"Temporada invalida: {season}. Usar: {list(season_months.keys())}")

        months = season_months[season]

        # Filtrar datos por temporada
        original_data = self._data.copy()
        self._data = self._data[self._data['month'].isin(months)]

        try:
            hotspots = self.generate_hotspots(**kwargs)

            # Agregar informacion de temporada
            hotspots = [
                DynamicHotspot(
                    id=h.id,
                    lat=h.lat,
                    lon=h.lon,
                    fishing_hours=h.fishing_hours,
                    vessel_count=h.vessel_count,
                    score=h.score,
                    source=h.source,
                    season=season
                )
                for h in hotspots
            ]

        finally:
            # Restaurar datos originales
            self._data = original_data

        return hotspots

    def _calculate_scores(
        self,
        hotspots: List[DynamicHotspot]
    ) -> List[DynamicHotspot]:
        """
        Calcula scores normalizados para hotspots.

        Score basado en:
        - 70% fishing_hours (normalizado)
        - 30% vessel_count (normalizado)

        Rango final: 0.5 - 1.0 (hotspot real siempre tiene score minimo 0.5)
        """
        if not hotspots:
            return []

        # Extraer metricas
        hours = np.array([h.fishing_hours for h in hotspots], dtype=np.float32)
        vessels = np.array([h.vessel_count for h in hotspots], dtype=np.float32)

        # Normalizar con min-max (evitar division por cero)
        hours_range = hours.max() - hours.min()
        vessels_range = vessels.max() - vessels.min()

        if hours_range > 0:
            hours_norm = (hours - hours.min()) / hours_range
        else:
            hours_norm = np.ones_like(hours)

        if vessels_range > 0:
            vessels_norm = (vessels - vessels.min()) / vessels_range
        else:
            vessels_norm = np.ones_like(vessels)

        # Score compuesto
        scores = 0.7 * hours_norm + 0.3 * vessels_norm

        # Escalar a 0.5-1.0 (nunca score bajo para un hotspot real)
        scores = 0.5 + scores * 0.5

        # Crear nuevos hotspots con scores
        return [
            DynamicHotspot(
                id=h.id,
                lat=h.lat,
                lon=h.lon,
                fishing_hours=h.fishing_hours,
                vessel_count=h.vessel_count,
                score=float(round(scores[i], 3)),
                source=h.source,
                season=h.season
            )
            for i, h in enumerate(hotspots)
        ]

    def get_nearest_hotspot(
        self,
        lat: float,
        lon: float,
        hotspots: Optional[List[DynamicHotspot]] = None
    ) -> Tuple[Optional[DynamicHotspot], float]:
        """
        Encuentra el hotspot mas cercano a una ubicacion.

        Args:
            lat: Latitud del punto
            lon: Longitud del punto
            hotspots: Lista de hotspots (si None, genera nuevos)

        Returns:
            (hotspot_cercano, distancia_km)
        """
        if hotspots is None:
            hotspots = self.generate_hotspots()

        if not hotspots:
            return None, float('inf')

        # Construir KD-Tree si scipy disponible
        if HAS_SCIPY:
            coords = np.array([[h.lat, h.lon] for h in hotspots], dtype=np.float32)
            tree = cKDTree(coords)

            # Buscar mas cercano
            dist_deg, idx = tree.query([lat, lon])
            dist_km = dist_deg * 111.0  # Aproximacion

            return hotspots[idx], dist_km
        else:
            # Fallback: busqueda lineal
            min_dist = float('inf')
            nearest = None

            for h in hotspots:
                # Distancia euclidiana simple (aproximacion)
                dist = np.sqrt((h.lat - lat)**2 + (h.lon - lon)**2) * 111.0
                if dist < min_dist:
                    min_dist = dist
                    nearest = h

            return nearest, min_dist

    def calculate_proximity_bonus(
        self,
        lat: float,
        lon: float,
        hotspots: Optional[List[DynamicHotspot]] = None,
        max_distance_km: float = 10.0,
        max_bonus: float = 10.0
    ) -> float:
        """
        Calcula bonus por proximidad a hotspot dinamico.

        Args:
            lat: Latitud del punto
            lon: Longitud del punto
            hotspots: Lista de hotspots
            max_distance_km: Distancia maxima para bonus
            max_bonus: Bonus maximo en puntos

        Returns:
            Bonus en puntos (0 a max_bonus)
        """
        nearest, distance = self.get_nearest_hotspot(lat, lon, hotspots)

        if nearest is None or distance > max_distance_km:
            return 0.0

        # Bonus decae linealmente con distancia
        proximity_factor = 1.0 - (distance / max_distance_km)

        # Escalar por score del hotspot
        bonus = proximity_factor * nearest.score * max_bonus

        return round(bonus, 2)

    def export_to_geojson(
        self,
        hotspots: List[DynamicHotspot],
        output_path: Path
    ) -> None:
        """
        Exporta hotspots a GeoJSON para visualizacion.

        Args:
            hotspots: Lista de hotspots
            output_path: Ruta de salida
        """
        features = []
        for h in hotspots:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [h.lon, h.lat]
                },
                "properties": {
                    "id": h.id,
                    "fishing_hours": h.fishing_hours,
                    "vessel_count": h.vessel_count,
                    "score": h.score,
                    "source": h.source,
                    "season": h.season
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Exportado: {output_path} ({len(hotspots)} hotspots)")

    def export_to_parquet(
        self,
        hotspots: List[DynamicHotspot],
        output_path: Optional[Path] = None
    ) -> None:
        """
        Exporta hotspots a Parquet para uso interno.

        Args:
            hotspots: Lista de hotspots
            output_path: Ruta de salida (usa config.output_cache si None)
        """
        if output_path is None:
            output_path = self.config.output_cache

        df = pd.DataFrame([
            {
                'id': h.id,
                'lat': h.lat,
                'lon': h.lon,
                'fishing_hours': h.fishing_hours,
                'vessel_count': h.vessel_count,
                'score': h.score,
                'source': h.source,
                'season': h.season
            }
            for h in hotspots
        ])

        # Optimizar tipos
        df['lat'] = df['lat'].astype(np.float32)
        df['lon'] = df['lon'].astype(np.float32)
        df['fishing_hours'] = df['fishing_hours'].astype(np.float32)
        df['score'] = df['score'].astype(np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression='snappy', index=False)

        logger.info(f"Exportado: {output_path}")

    def get_statistics(self) -> Dict:
        """Obtiene estadisticas de los datos GFW."""
        self._ensure_loaded()

        if self._data is None or self._data.empty:
            return {'status': 'no_data'}

        return {
            'total_records': len(self._data),
            'total_fishing_hours': float(self._data['fishing_hours'].sum()),
            'unique_positions': len(self._data[['lat', 'lon']].drop_duplicates()),
            'lat_range': {
                'min': float(self._data['lat'].min()),
                'max': float(self._data['lat'].max())
            },
            'lon_range': {
                'min': float(self._data['lon'].min()),
                'max': float(self._data['lon'].max())
            },
            'has_temporal_data': 'month' in self._data.columns
        }

    def clear_cache(self) -> None:
        """Limpia la cache de hotspots."""
        self._hotspots_cache.clear()


# =============================================================================
# FUNCION DE CONVENIENCIA
# =============================================================================

def get_gfw_hotspot_bonus(
    lat: float,
    lon: float,
    max_distance_km: float = 10.0
) -> float:
    """
    Funcion de conveniencia para obtener bonus por proximidad a hotspot GFW.

    Args:
        lat: Latitud
        lon: Longitud
        max_distance_km: Distancia maxima para bonus

    Returns:
        Bonus 0-10 puntos
    """
    generator = GFWHotspotGenerator()
    return generator.calculate_proximity_bonus(lat, lon, max_distance_km=max_distance_km)
