"""
Fetcher para datos oceanograficos - Enfocado en pesca desde orilla/roca.
Genera datos realistas para la franja costera de la costa sur de Peru.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import BBOX, LOCATIONS


class ERDDAPFetcher:
    """Genera datos oceanograficos para pesca costera."""

    def __init__(self, use_cache: bool = True, use_fallback: bool = True):
        pass

    def _generate_sst_coastal(self, bbox: Dict[str, float]) -> pd.DataFrame:
        """
        Genera SST exactamente en las ubicaciones de pesca conocidas.
        Un punto por ubicacion - directamente en la costa.
        Incluye tipo de sustrato y descripcion.
        """
        np.random.seed(int(datetime.now().hour))

        data = []

        # Generar UN punto por cada ubicacion de pesca conocida
        for loc_name, loc_info in LOCATIONS.items():
            lat = loc_info["lat"]
            lon = loc_info["lon"]
            tipo = loc_info.get("tipo", "mixto")
            descripcion = loc_info.get("descripcion", "")

            # SST base segun tipo de sustrato y ubicacion
            if tipo == "roca":
                # Zonas rocosas suelen tener mas variabilidad termica
                base_sst = 15.5 + np.random.uniform(0.5, 1.8)
            elif tipo == "arena":
                # Playas arenosas, aguas mas uniformes
                base_sst = 16.0 + np.random.uniform(0.2, 1.0)
            else:  # mixto
                base_sst = 15.8 + np.random.uniform(0.3, 1.3)

            # Ajuste por zona geografica
            if "Ite" in loc_name:
                base_sst -= 0.5  # Surgencia activa = agua mas fria
            elif "Coles" in loc_name or "Pocoma" in loc_name:
                base_sst -= 0.3  # Zona con corrientes

            sst = base_sst + np.random.normal(0, 0.15)
            sst = np.clip(sst, 14.5, 19.0)

            data.append({
                "latitude": lat,
                "longitude": lon,
                "sst": round(sst, 2),
                "dist_costa_km": 0.0,
                "location": loc_name,
                "tipo_sustrato": tipo,
                "descripcion": descripcion
            })

        return pd.DataFrame(data)

    def _generate_chlorophyll_coastal(self, bbox: Dict[str, float]) -> pd.DataFrame:
        """
        Genera clorofila exactamente en las ubicaciones de pesca conocidas.
        Incluye tipo de sustrato - rocas tienen mas productividad.
        """
        np.random.seed(int(datetime.now().hour) + 1)

        data = []

        for loc_name, loc_info in LOCATIONS.items():
            lat = loc_info["lat"]
            lon = loc_info["lon"]
            tipo = loc_info.get("tipo", "mixto")

            # Clorofila base segun tipo de sustrato
            if tipo == "roca":
                # Rocas tienen mas vida marina adherida
                base_chl = 5.5 + np.random.uniform(1.5, 3.0)
            elif tipo == "arena":
                # Playas arenosas menor productividad visible
                base_chl = 4.0 + np.random.uniform(0.8, 2.0)
            else:  # mixto
                base_chl = 4.8 + np.random.uniform(1.0, 2.5)

            # Ajuste por zona
            if "Ite" in loc_name:
                base_chl += 1.5  # Surgencia = alta productividad
            elif "Coles" in loc_name:
                base_chl += 1.0  # Reserva con mucha vida
            elif "Boca" in loc_name:
                base_chl += 0.8  # Nutrientes del rio

            chl = base_chl + np.random.normal(0, 0.2)
            chl = np.clip(chl, 0.8, 12.0)

            data.append({
                "latitude": lat,
                "longitude": lon,
                "chlorophyll": round(chl, 2)
            })

        return pd.DataFrame(data)

    def fetch_sst(
        self,
        date: Optional[datetime] = None,
        bbox: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        bbox = bbox or BBOX
        print("      Generando datos SST costeros...")
        return self._generate_sst_coastal(bbox)

    def fetch_chlorophyll(
        self,
        date: Optional[datetime] = None,
        bbox: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        bbox = bbox or BBOX
        print("      Generando datos Clorofila costeros...")
        return self._generate_chlorophyll_coastal(bbox)

    def fetch_all(
        self,
        date: Optional[datetime] = None,
        bbox: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sst_df = self.fetch_sst(date, bbox)
        chl_df = self.fetch_chlorophyll(date, bbox)
        return sst_df, chl_df
