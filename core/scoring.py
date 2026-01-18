"""
Motor de scoring unificado para prediccion de pesca.
Consolida toda la logica de puntuacion en un solo modulo.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

from .transects import Transect
from .fish_movement import MovementVector, MovementTrend
from .coastline import SubstrateType


class ScoreCategory(Enum):
    """Categorias de puntuacion."""
    EXCELENTE = "Excelente"
    BUENO = "Bueno"
    PROMEDIO = "Promedio"
    BAJO = "Bajo promedio"
    POBRE = "Pobre"


@dataclass
class ScoringWeights:
    """Pesos configurables para el scoring."""
    sst_weight: float = 0.20
    chlorophyll_weight: float = 0.25
    safety_weight: float = 0.15
    golden_hour_weight: float = 0.15
    lunar_weight: float = 0.10
    substrate_weight: float = 0.10
    movement_weight: float = 0.05

    def validate(self) -> bool:
        """Verifica que los pesos sumen 1.0."""
        total = (
            self.sst_weight +
            self.chlorophyll_weight +
            self.safety_weight +
            self.golden_hour_weight +
            self.lunar_weight +
            self.substrate_weight +
            self.movement_weight
        )
        return abs(total - 1.0) < 0.01

    def normalize(self):
        """Normaliza los pesos para que sumen 1.0."""
        total = (
            self.sst_weight +
            self.chlorophyll_weight +
            self.safety_weight +
            self.golden_hour_weight +
            self.lunar_weight +
            self.substrate_weight +
            self.movement_weight
        )
        if total > 0:
            self.sst_weight /= total
            self.chlorophyll_weight /= total
            self.safety_weight /= total
            self.golden_hour_weight /= total
            self.lunar_weight /= total
            self.substrate_weight /= total
            self.movement_weight /= total


@dataclass
class FishingScore:
    """Resultado de scoring para un punto de pesca."""
    location_name: str
    latitude: float
    longitude: float
    total_score: float
    category: ScoreCategory
    is_safe: bool

    # Scores componentes (0-100)
    sst_score: float = 0.0
    chlorophyll_score: float = 0.0
    safety_score: float = 0.0
    golden_hour_score: float = 0.0
    lunar_score: float = 0.0
    substrate_score: float = 0.0
    movement_score: float = 0.0

    # Datos adicionales
    substrate_type: str = ""
    recommended_species: List[str] = field(default_factory=list)
    movement_trend: str = ""

    @property
    def color(self) -> str:
        """Color para visualizacion basado en el score."""
        if self.total_score >= 80:
            return "#228b22"  # Verde oscuro
        elif self.total_score >= 60:
            return "#90ee90"  # Verde claro
        elif self.total_score >= 40:
            return "#ffff00"  # Amarillo
        elif self.total_score >= 20:
            return "#ff8c00"  # Naranja
        else:
            return "#ff4444"  # Rojo

    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            "location": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "total_score": round(self.total_score, 1),
            "category": self.category.value,
            "is_safe": self.is_safe,
            "color": self.color,
            "components": {
                "sst": round(self.sst_score, 1),
                "chlorophyll": round(self.chlorophyll_score, 1),
                "safety": round(self.safety_score, 1),
                "golden_hour": round(self.golden_hour_score, 1),
                "lunar": round(self.lunar_score, 1),
                "substrate": round(self.substrate_score, 1),
                "movement": round(self.movement_score, 1)
            },
            "substrate_type": self.substrate_type,
            "recommended_species": self.recommended_species,
            "movement_trend": self.movement_trend
        }


class ScoringEngine:
    """
    Motor de scoring unificado.
    Calcula puntuaciones de pesca basadas en multiples factores.
    """

    # Configuracion por defecto
    DEFAULT_WEIGHTS = ScoringWeights()

    # Umbrales de seguridad
    WAVE_SAFETY_THRESHOLD = 2.0  # metros
    WIND_SAFETY_THRESHOLD = 25.0  # km/h

    # Rangos optimos SST para costa peruana
    SST_OPTIMAL_MIN = 15.0
    SST_OPTIMAL_MAX = 17.5

    # Rangos optimos clorofila
    CHL_OPTIMAL_MIN = 3.0
    CHL_OPTIMAL_MAX = 8.0

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Inicializa el motor de scoring.

        Args:
            weights: pesos personalizados (usa DEFAULT_WEIGHTS si no se especifica)
        """
        self.weights = weights or ScoringWeights()
        if not self.weights.validate():
            self.weights.normalize()

    def calculate_score(
        self,
        transect: Transect,
        movement_vector: Optional[MovementVector] = None,
        weather_data: Optional[Dict] = None,
        astro_data: Optional[Dict] = None,
        current_hour: Optional[int] = None
    ) -> FishingScore:
        """
        Calcula el score de pesca para un transecto.

        Args:
            transect: transecto con datos oceanograficos
            movement_vector: vector de movimiento predicho
            weather_data: datos meteorologicos (wave_height, wind_speed)
            astro_data: datos astronomicos (lunar_phase, golden_hour_score)
            current_hour: hora actual (0-23)

        Returns:
            FishingScore con todos los componentes
        """
        if current_hour is None:
            current_hour = datetime.now().hour

        # Calcular scores componentes
        sst_score = self._calculate_sst_score(transect.avg_sst)
        chl_score = self._calculate_chlorophyll_score(transect.avg_chlorophyll)
        safety_score = self._calculate_safety_score(weather_data)
        golden_score = self._calculate_golden_hour_score(current_hour, astro_data)
        lunar_score = self._calculate_lunar_score(astro_data)
        substrate_score = self._calculate_substrate_score(transect.substrate)
        movement_score = self._calculate_movement_score(movement_vector)

        # Score total ponderado
        total = (
            sst_score * self.weights.sst_weight +
            chl_score * self.weights.chlorophyll_weight +
            safety_score * self.weights.safety_weight +
            golden_score * self.weights.golden_hour_weight +
            lunar_score * self.weights.lunar_weight +
            substrate_score * self.weights.substrate_weight +
            movement_score * self.weights.movement_weight
        )

        # Penalizacion por inseguridad
        is_safe = safety_score >= 50
        if not is_safe:
            total *= 0.5  # Reducir 50% si es inseguro

        total = min(100, max(0, total))

        # Categoria
        category = self._get_category(total)

        # Especies recomendadas
        species = []
        if movement_vector and movement_vector.target_species:
            species = movement_vector.target_species

        # Tendencia de movimiento
        trend = ""
        if movement_vector:
            trend = movement_vector.trend.value

        return FishingScore(
            location_name=transect.name,
            latitude=transect.shore_lat,
            longitude=transect.shore_lon,
            total_score=total,
            category=category,
            is_safe=is_safe,
            sst_score=sst_score,
            chlorophyll_score=chl_score,
            safety_score=safety_score,
            golden_hour_score=golden_score,
            lunar_score=lunar_score,
            substrate_score=substrate_score,
            movement_score=movement_score,
            substrate_type=transect.substrate.value,
            recommended_species=species,
            movement_trend=trend
        )

    def _calculate_sst_score(self, sst: Optional[float]) -> float:
        """Calcula score de SST (0-100)."""
        if sst is None:
            return 50.0

        # Score maximo en rango optimo
        if self.SST_OPTIMAL_MIN <= sst <= self.SST_OPTIMAL_MAX:
            return 100.0

        # Reduccion fuera del rango optimo
        if sst < self.SST_OPTIMAL_MIN:
            diff = self.SST_OPTIMAL_MIN - sst
            return max(0, 100 - diff * 25)
        else:
            diff = sst - self.SST_OPTIMAL_MAX
            return max(0, 100 - diff * 25)

    def _calculate_chlorophyll_score(self, chl: Optional[float]) -> float:
        """Calcula score de clorofila (0-100)."""
        if chl is None:
            return 50.0

        # Score maximo en rango optimo
        if self.CHL_OPTIMAL_MIN <= chl <= self.CHL_OPTIMAL_MAX:
            return 100.0

        # Muy baja clorofila = poco alimento
        if chl < self.CHL_OPTIMAL_MIN:
            return max(0, (chl / self.CHL_OPTIMAL_MIN) * 80)

        # Muy alta puede indicar bloom (no optimo)
        if chl > self.CHL_OPTIMAL_MAX:
            excess = chl - self.CHL_OPTIMAL_MAX
            return max(50, 100 - excess * 5)

        return 50.0

    def _calculate_safety_score(self, weather_data: Optional[Dict]) -> float:
        """Calcula score de seguridad (0-100)."""
        if weather_data is None:
            return 70.0  # Asumir condiciones moderadas

        wave_height = weather_data.get('wave_height', 1.0)
        wind_speed = weather_data.get('wind_speed', 10.0)

        # Factor de olas
        wave_factor = min(wave_height / self.WAVE_SAFETY_THRESHOLD, 1.0)

        # Factor de viento
        wind_factor = min(wind_speed / self.WIND_SAFETY_THRESHOLD, 1.0)

        # Score combinado (ambos factores importan)
        safety = (1 - (wave_factor * 0.6 + wind_factor * 0.4)) * 100

        return max(0, min(100, safety))

    def _calculate_golden_hour_score(
        self,
        hour: int,
        astro_data: Optional[Dict] = None
    ) -> float:
        """Calcula score de hora dorada (0-100)."""
        # Si tenemos datos astronomicos, usarlos
        if astro_data and 'golden_hour_score' in astro_data:
            return astro_data['golden_hour_score']

        # Calcular basado en hora
        # Amanecer ~5:30-7:30, Atardecer ~17:30-19:30
        if 5 <= hour <= 7:
            return 90 + (7 - hour) * 5  # Maximo al amanecer
        elif 17 <= hour <= 19:
            return 90 + (hour - 17) * 5  # Maximo al atardecer
        elif 7 < hour < 10 or 15 < hour < 17:
            return 70.0  # Horas cercanas
        elif 10 <= hour <= 15:
            return 40.0  # Mediodia - bajo
        else:
            return 30.0  # Noche

    def _calculate_lunar_score(self, astro_data: Optional[Dict]) -> float:
        """Calcula score lunar (0-100)."""
        if astro_data is None:
            return 50.0

        # Si tenemos score precalculado
        if 'lunar_score' in astro_data:
            return astro_data['lunar_score']

        # Calcular basado en fase
        phase = astro_data.get('lunar_phase_name', '').lower()

        phase_scores = {
            'luna nueva': 90,
            'new moon': 90,
            'luna llena': 85,
            'full moon': 85,
            'cuarto creciente': 70,
            'first quarter': 70,
            'cuarto menguante': 70,
            'last quarter': 70,
        }

        for key, score in phase_scores.items():
            if key in phase:
                return float(score)

        return 60.0

    def _calculate_substrate_score(self, substrate: SubstrateType) -> float:
        """Calcula score de sustrato (0-100)."""
        # Rocas tienen mas estructura = mas peces
        scores = {
            SubstrateType.ROCA: 85,
            SubstrateType.MIXTO: 75,
            SubstrateType.ARENA: 65
        }
        return float(scores.get(substrate, 70))

    def _calculate_movement_score(
        self,
        movement: Optional[MovementVector]
    ) -> float:
        """Calcula score basado en movimiento de peces (0-100)."""
        if movement is None:
            return 50.0

        # Peces moviendose hacia la costa = mejor para pesca desde orilla
        trend_scores = {
            MovementTrend.HACIA_COSTA: 95,
            MovementTrend.PARALELO_COSTA: 70,
            MovementTrend.ESTACIONARIO: 60,
            MovementTrend.HACIA_MAR: 40
        }

        base_score = trend_scores.get(movement.trend, 50)

        # Ajustar por magnitud y confianza
        score = base_score * (0.5 + movement.magnitude * 0.3 + movement.confidence * 0.2)

        return min(100, max(0, score))

    def _get_category(self, score: float) -> ScoreCategory:
        """Obtiene la categoria del score."""
        if score >= 80:
            return ScoreCategory.EXCELENTE
        elif score >= 60:
            return ScoreCategory.BUENO
        elif score >= 40:
            return ScoreCategory.PROMEDIO
        elif score >= 20:
            return ScoreCategory.BAJO
        else:
            return ScoreCategory.POBRE

    def calculate_batch(
        self,
        transects: List[Transect],
        movement_vectors: Optional[List[MovementVector]] = None,
        weather_data: Optional[Dict] = None,
        astro_data: Optional[Dict] = None
    ) -> List[FishingScore]:
        """
        Calcula scores para multiples transectos.

        Returns:
            Lista de FishingScore ordenada por score descendente
        """
        scores = []

        for i, transect in enumerate(transects):
            movement = None
            if movement_vectors and i < len(movement_vectors):
                movement = movement_vectors[i]

            score = self.calculate_score(
                transect=transect,
                movement_vector=movement,
                weather_data=weather_data,
                astro_data=astro_data
            )
            scores.append(score)

        # Ordenar por score descendente
        scores.sort(key=lambda s: s.total_score, reverse=True)

        return scores

    def get_best_spot(
        self,
        scores: List[FishingScore],
        require_safe: bool = True
    ) -> Optional[FishingScore]:
        """Obtiene el mejor punto de pesca."""
        if not scores:
            return None

        if require_safe:
            safe_scores = [s for s in scores if s.is_safe]
            return safe_scores[0] if safe_scores else None

        return scores[0]
