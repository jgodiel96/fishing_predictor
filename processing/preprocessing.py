"""
Preprocesamiento de datos: creacion de grilla, interpolacion, alineacion.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.interpolate import griddata
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import BBOX, GRID_RESOLUTION_DEG, LOCATIONS


@dataclass
class Grid:
    """Representa una grilla geografica."""
    lat: np.ndarray  # Array 1D de latitudes
    lon: np.ndarray  # Array 1D de longitudes
    lat_grid: np.ndarray  # Meshgrid 2D de latitudes
    lon_grid: np.ndarray  # Meshgrid 2D de longitudes


class GridProcessor:
    """Procesa y alinea datos a una grilla comun."""

    def __init__(
        self,
        bbox: Optional[Dict[str, float]] = None,
        resolution_deg: Optional[float] = None
    ):
        """
        Inicializa el procesador de grilla.

        Args:
            bbox: Bounding box {north, south, east, west}
            resolution_deg: Resolucion en grados
        """
        self.bbox = bbox or BBOX
        self.resolution = resolution_deg or GRID_RESOLUTION_DEG
        self._grid = None

    @property
    def grid(self) -> Grid:
        """Retorna la grilla, creandola si es necesario."""
        if self._grid is None:
            self._grid = self._create_grid()
        return self._grid

    def _create_grid(self) -> Grid:
        """Crea la grilla base."""
        lat = np.arange(
            self.bbox["south"],
            self.bbox["north"],
            self.resolution
        )
        lon = np.arange(
            self.bbox["west"],
            self.bbox["east"],
            self.resolution
        )

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        return Grid(
            lat=lat,
            lon=lon,
            lat_grid=lat_grid,
            lon_grid=lon_grid
        )

    def interpolate_to_grid(
        self,
        df: pd.DataFrame,
        value_column: str,
        method: str = "linear"
    ) -> np.ndarray:
        """
        Interpola datos dispersos a la grilla regular.

        Args:
            df: DataFrame con columnas 'latitude', 'longitude' y value_column
            value_column: Nombre de la columna con valores a interpolar
            method: Metodo de interpolacion ('linear', 'nearest', 'cubic')

        Returns:
            Array 2D con valores interpolados
        """
        if df.empty:
            return np.full(self.grid.lat_grid.shape, np.nan)

        points = df[["longitude", "latitude"]].values
        values = df[value_column].values

        # Interpolar a grilla
        interpolated = griddata(
            points,
            values,
            (self.grid.lon_grid, self.grid.lat_grid),
            method=method
        )

        return interpolated

    def fill_gaps(
        self,
        data: np.ndarray,
        max_iterations: int = 10
    ) -> np.ndarray:
        """
        Rellena huecos (NaN) usando interpolacion iterativa.

        Args:
            data: Array 2D con posibles NaN
            max_iterations: Numero maximo de iteraciones

        Returns:
            Array con huecos rellenados
        """
        filled = data.copy()

        for _ in range(max_iterations):
            if not np.isnan(filled).any():
                break

            # Kernel para promediar vecinos
            kernel = np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ]) / 8.0

            # Promediar vecinos donde hay NaN
            from scipy.ndimage import convolve
            neighbor_avg = convolve(
                np.nan_to_num(filled, nan=0),
                kernel,
                mode="constant"
            )
            neighbor_count = convolve(
                (~np.isnan(filled)).astype(float),
                kernel,
                mode="constant"
            )

            # Evitar division por cero
            neighbor_count = np.where(neighbor_count == 0, 1, neighbor_count)
            neighbor_avg = neighbor_avg / neighbor_count * 8

            # Rellenar NaN con promedio de vecinos
            mask = np.isnan(filled) & (neighbor_count > 0)
            filled = np.where(mask, neighbor_avg, filled)

        return filled

    def _get_coastline_lon(self, lat: float) -> float:
        """
        Calcula la longitud de la costa para una latitud dada.
        La costa de Peru en esta zona corre de suroeste a noreste.
        Basado en las ubicaciones conocidas de pesca.
        """
        lat_ref = -18.12  # Boca del Rio
        lon_ref = -70.85
        slope = -1.02
        return lon_ref + slope * (lat - lat_ref)

    def calculate_distance_to_coast(
        self,
        coastline_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calcula la distancia de cada punto a la costa.
        Usa modelo de costa realista que varia con la latitud.

        Args:
            coastline_mask: Mascara donde True = tierra/costa
                           Si no se provee, usa modelo de costa diagonal

        Returns:
            Array 2D con distancias en km
        """
        if coastline_mask is None:
            # Crear mascara de costa basada en modelo diagonal
            coastline_mask = np.zeros(self.grid.lat_grid.shape, dtype=bool)

            for i in range(self.grid.lat_grid.shape[0]):
                lat = self.grid.lat_grid[i, 0]
                coast_lon = self._get_coastline_lon(lat)

                for j in range(self.grid.lon_grid.shape[1]):
                    lon = self.grid.lon_grid[i, j]
                    # Puntos al este de la costa son tierra
                    if lon > coast_lon:
                        coastline_mask[i, j] = True

        # Calcular distancia para cada punto
        distance_km = np.zeros(self.grid.lat_grid.shape)

        for i in range(self.grid.lat_grid.shape[0]):
            lat = self.grid.lat_grid[i, 0]
            coast_lon = self._get_coastline_lon(lat)

            for j in range(self.grid.lon_grid.shape[1]):
                lon = self.grid.lon_grid[i, j]
                # Distancia en km (negativo = tierra, positivo = mar)
                dist = (coast_lon - lon) * 111 * np.cos(np.radians(lat))
                distance_km[i, j] = max(0, dist)  # Solo valores positivos (en el mar)

        return distance_km

    def get_grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna las coordenadas de la grilla como arrays planos.

        Returns:
            Tupla de (latitudes, longitudes) como arrays 1D
        """
        return self.grid.lat_grid.flatten(), self.grid.lon_grid.flatten()

    def grid_to_dataframe(
        self,
        data_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Convierte datos en grilla a DataFrame.

        Args:
            data_dict: Dict de {nombre_columna: array_2d}

        Returns:
            DataFrame con coordenadas y valores
        """
        lats, lons = self.get_grid_coordinates()

        df = pd.DataFrame({
            "latitude": lats,
            "longitude": lons
        })

        for name, data in data_dict.items():
            df[name] = data.flatten()

        return df

    def mask_by_distance(
        self,
        data: np.ndarray,
        max_distance_km: float,
        coastline_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Enmascara datos que estan muy lejos de la costa.

        Args:
            data: Array 2D de datos
            max_distance_km: Distancia maxima en km
            coastline_mask: Mascara de costa

        Returns:
            Array con NaN donde la distancia excede el maximo
        """
        distance = self.calculate_distance_to_coast(coastline_mask)
        masked = np.where(distance <= max_distance_km, data, np.nan)
        return masked
