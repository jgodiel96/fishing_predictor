"""
Generador de barrido costero para análisis de tramos de playa.
Divide un tramo de costa en secciones para análisis detallado.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CoastalSection:
    """Representa una sección de costa."""
    id: int
    name: str
    lat_start: float
    lon_start: float
    lat_end: float
    lon_end: float
    lat_center: float
    lon_center: float
    length_m: float
    tipo: str
    descripcion: str


class CoastalSweep:
    """
    Genera un barrido de análisis a lo largo de un tramo de costa.
    Divide el tramo en secciones para análisis detallado de pesca.
    """

    def __init__(self):
        self.sections: List[CoastalSection] = []

    def create_sweep(
        self,
        start_point: Tuple[float, float],  # (lat, lon)
        end_point: Tuple[float, float],    # (lat, lon)
        num_sections: int = 10,
        tramo_name: str = "Tramo",
        tipo_sustrato: str = "arena"
    ) -> List[CoastalSection]:
        """
        Crea un barrido dividiendo el tramo en secciones.

        Args:
            start_point: (lat, lon) del punto inicial
            end_point: (lat, lon) del punto final
            num_sections: número de secciones a crear
            tramo_name: nombre base del tramo
            tipo_sustrato: tipo de sustrato predominante

        Returns:
            Lista de secciones costeras
        """
        lat1, lon1 = start_point
        lat2, lon2 = end_point

        # Calcular distancia total aproximada (en metros)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        lat_avg = (lat1 + lat2) / 2

        # Conversión aproximada a metros
        dist_lat_m = dlat * 111000  # 1 grado lat ≈ 111km
        dist_lon_m = dlon * 111000 * np.cos(np.radians(lat_avg))
        total_dist_m = np.sqrt(dist_lat_m**2 + dist_lon_m**2)

        # Si la distancia es muy pequeña, crear al menos algunos puntos
        if total_dist_m < 100:
            # Extender el área en ambas direcciones
            extension = 0.005  # ~500m
            lat1 -= extension
            lat2 += extension
            lon1 -= extension * 0.5
            lon2 += extension * 0.5

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            dist_lat_m = dlat * 111000
            dist_lon_m = dlon * 111000 * np.cos(np.radians(lat_avg))
            total_dist_m = np.sqrt(dist_lat_m**2 + dist_lon_m**2)

        section_length = total_dist_m / num_sections

        self.sections = []

        for i in range(num_sections):
            # Calcular puntos de inicio y fin de cada sección
            t_start = i / num_sections
            t_end = (i + 1) / num_sections
            t_center = (t_start + t_end) / 2

            sec_lat_start = lat1 + dlat * t_start
            sec_lon_start = lon1 + dlon * t_start
            sec_lat_end = lat1 + dlat * t_end
            sec_lon_end = lon1 + dlon * t_end
            sec_lat_center = lat1 + dlat * t_center
            sec_lon_center = lon1 + dlon * t_center

            section = CoastalSection(
                id=i + 1,
                name=f"{tramo_name} - Sec {i + 1}",
                lat_start=round(sec_lat_start, 6),
                lon_start=round(sec_lon_start, 6),
                lat_end=round(sec_lat_end, 6),
                lon_end=round(sec_lon_end, 6),
                lat_center=round(sec_lat_center, 6),
                lon_center=round(sec_lon_center, 6),
                length_m=round(section_length, 1),
                tipo=tipo_sustrato,
                descripcion=f"Sección {i + 1} de {num_sections}"
            )
            self.sections.append(section)

        return self.sections

    def generate_analysis_data(
        self,
        sections: Optional[List[CoastalSection]] = None
    ) -> pd.DataFrame:
        """
        Genera datos de análisis para cada sección del barrido.

        Returns:
            DataFrame con SST, clorofila y scores por sección
        """
        if sections is None:
            sections = self.sections

        if not sections:
            raise ValueError("No hay secciones definidas. Usa create_sweep primero.")

        np.random.seed(int(datetime.now().hour))

        data = []

        for sec in sections:
            # Generar SST según tipo de sustrato
            if sec.tipo == "roca":
                base_sst = 15.5 + np.random.uniform(0.5, 1.8)
                base_chl = 5.5 + np.random.uniform(1.5, 3.0)
            elif sec.tipo == "arena":
                base_sst = 16.0 + np.random.uniform(0.2, 1.0)
                base_chl = 4.0 + np.random.uniform(0.8, 2.0)
            else:  # mixto
                base_sst = 15.8 + np.random.uniform(0.3, 1.3)
                base_chl = 4.8 + np.random.uniform(1.0, 2.5)

            sst = round(base_sst + np.random.normal(0, 0.15), 2)
            sst = np.clip(sst, 14.5, 19.0)

            chl = round(base_chl + np.random.normal(0, 0.2), 2)
            chl = np.clip(chl, 0.8, 12.0)

            data.append({
                "section_id": sec.id,
                "location": sec.name,
                "latitude": sec.lat_center,
                "longitude": sec.lon_center,
                "lat_start": sec.lat_start,
                "lon_start": sec.lon_start,
                "lat_end": sec.lat_end,
                "lon_end": sec.lon_end,
                "length_m": sec.length_m,
                "tipo_sustrato": sec.tipo,
                "descripcion": sec.descripcion,
                "sst": sst,
                "chlorophyll": chl,
                "dist_costa_km": 0.0
            })

        return pd.DataFrame(data)

    def get_sections_geojson(self) -> dict:
        """
        Genera GeoJSON de las secciones como líneas.
        Útil para visualizar el barrido en el mapa.
        """
        features = []

        for sec in self.sections:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [sec.lon_start, sec.lat_start],
                        [sec.lon_end, sec.lat_end]
                    ]
                },
                "properties": {
                    "id": sec.id,
                    "name": sec.name,
                    "tipo": sec.tipo,
                    "length_m": sec.length_m
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }
