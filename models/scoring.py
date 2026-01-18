"""
Modelo de scoring para prediccion de zonas de pesca.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import get_score_color, get_score_category
from processing.features import FeatureSet


@dataclass
class ScoringResult:
    """Resultado del scoring."""
    total_score: np.ndarray
    subscores: Dict[str, np.ndarray]
    is_safe: np.ndarray


class FishingScorer:
    """Calcula el score de pesca para cada celda de la grilla."""

    # Pesos optimizados para pesca
    DEFAULT_WEIGHTS = {
        "front_proximity": 0.30,      # Frentes termicos - muy importante
        "chlorophyll_score": 0.25,    # Productividad
        "upwelling_index": 0.10,      # Surgencia
        "golden_hour": 0.15,          # Hora dorada
        "lunar_score": 0.10,          # Fase lunar
        "safety_score": 0.10,         # Seguridad
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def calculate_subscores(
        self,
        features: FeatureSet,
        astronomical_data: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """Calcula sub-scores para cada factor."""
        subscores = {}

        # Front proximity - basado en gradiente SST
        # Normalizar front_proximity_score a 0-100
        front_score = features.front_proximity_score
        if front_score.max() > 0:
            front_score = (front_score / front_score.max()) * 100
        subscores["front_proximity"] = np.clip(front_score, 0, 100)

        # Chlorophyll score - ya viene calculado
        chl_score = features.chlorophyll_score
        subscores["chlorophyll_score"] = np.clip(chl_score, 0, 100)

        # Upwelling index - normalizar
        upwelling = features.upwelling_index
        upwelling_norm = upwelling - upwelling.min()
        if upwelling_norm.max() > 0:
            upwelling_norm = (upwelling_norm / upwelling_norm.max()) * 100
        subscores["upwelling_index"] = np.clip(upwelling_norm, 0, 100)

        # Golden hour score
        if astronomical_data and "golden_hour_score" in astronomical_data:
            golden = float(astronomical_data["golden_hour_score"])
        else:
            golden = 50.0
        subscores["golden_hour"] = np.full(features.sst.shape, golden)

        # Lunar score
        if astronomical_data and "lunar_score" in astronomical_data:
            lunar = float(astronomical_data["lunar_score"])
        else:
            lunar = 50.0
        subscores["lunar_score"] = np.full(features.sst.shape, lunar)

        # Safety score
        safety = features.safety_score
        subscores["safety_score"] = np.clip(safety, 0, 100)

        return subscores

    def calculate_total_score(
        self,
        subscores: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula el score total ponderado."""
        total = np.zeros_like(subscores["safety_score"], dtype=float)

        # Sumar scores ponderados
        for factor, weight in self.weights.items():
            if factor in subscores:
                total += subscores[factor] * weight

        # Determinar si es seguro (safety > 40)
        is_safe = subscores["safety_score"] >= 40

        # Penalizar ligeramente condiciones inseguras (pero no tanto)
        total = np.where(is_safe, total, total * 0.7)

        # Asegurar rango 0-100
        total = np.clip(total, 0, 100)

        return total, is_safe

    def score(
        self,
        features: FeatureSet,
        astronomical_data: Optional[Dict] = None
    ) -> ScoringResult:
        """Calcula el score completo."""
        subscores = self.calculate_subscores(features, astronomical_data)
        total_score, is_safe = self.calculate_total_score(subscores)

        return ScoringResult(
            total_score=total_score,
            subscores=subscores,
            is_safe=is_safe
        )

    def score_to_dataframe(
        self,
        result: ScoringResult,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray
    ) -> pd.DataFrame:
        """Convierte resultado de scoring a DataFrame."""
        df = pd.DataFrame({
            "latitude": lat_grid.flatten(),
            "longitude": lon_grid.flatten(),
            "score": result.total_score.flatten(),
            "is_safe": result.is_safe.flatten(),
        })

        # Agregar subscores
        for name, subscore in result.subscores.items():
            df[f"score_{name}"] = subscore.flatten()

        # Agregar categoria y color
        df["category"] = df["score"].apply(get_score_category)
        df["color"] = df["score"].apply(get_score_color)

        return df

    def get_top_spots(
        self,
        df: pd.DataFrame,
        n: int = 10,
        min_safety_score: float = 40.0
    ) -> pd.DataFrame:
        """Obtiene los mejores spots para pesca."""
        safe_df = df[df["score_safety_score"] >= min_safety_score].copy()
        if safe_df.empty:
            return df.nlargest(n, "score")
        return safe_df.nlargest(n, "score")

    def get_spot_summary(self, lat: float, lon: float, df: pd.DataFrame) -> Dict:
        """Obtiene resumen detallado de un spot."""
        df_copy = df.copy()
        df_copy["distance"] = np.sqrt(
            (df_copy["latitude"] - lat)**2 + (df_copy["longitude"] - lon)**2
        )
        closest = df_copy.loc[df_copy["distance"].idxmin()]

        return {
            "latitude": closest["latitude"],
            "longitude": closest["longitude"],
            "total_score": closest["score"],
            "category": closest["category"],
            "is_safe": closest["is_safe"],
            "breakdown": {
                "front_proximity": closest.get("score_front_proximity", 0),
                "chlorophyll": closest.get("score_chlorophyll_score", 0),
                "upwelling": closest.get("score_upwelling_index", 0),
                "golden_hour": closest.get("score_golden_hour", 0),
                "lunar": closest.get("score_lunar_score", 0),
                "safety": closest.get("score_safety_score", 0),
            }
        }
