"""
ML Predictor for fishing zones using PCA and ensemble methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Result of fishing zone prediction."""
    lat: float
    lon: float
    score: float  # 0-100
    confidence: float  # 0-1
    cluster_id: int
    principal_components: np.ndarray
    contributing_factors: Dict[str, float]


class FishingPredictor:
    """
    ML-based fishing zone predictor.

    Uses:
    - PCA for dimensionality reduction and feature analysis
    - KMeans for zone clustering
    - GradientBoosting for score regression
    """

    def __init__(self, n_components: int = 4, n_clusters: int = 5):
        self.n_components = n_components
        self.n_clusters = n_clusters

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )

        self.is_fitted = False
        self.feature_names: List[str] = []
        self.pca_explained_variance: Optional[np.ndarray] = None
        self.pca_components: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: List[str] = None):
        """
        Fit the predictor on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Optional target scores (for supervised learning)
            feature_names: Names of features for interpretation
        """
        if X.shape[0] < 5:
            raise ValueError("Need at least 5 samples to fit")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        n_comp = min(self.n_components, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_comp)
        X_pca = self.pca.fit_transform(X_scaled)

        self.pca_explained_variance = self.pca.explained_variance_ratio_
        self.pca_components = self.pca.components_

        # Clustering
        n_clust = min(self.n_clusters, X.shape[0])
        self.clusterer = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        self.clusterer.fit(X_pca)

        # Regression (if targets provided)
        if y is not None and len(y) == X.shape[0]:
            self.regressor.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def fit_unsupervised(self, X: np.ndarray, feature_names: List[str] = None):
        """
        Fit using unsupervised approach with domain-knowledge scoring.

        NOTE: This method uses oceanographic domain knowledge to estimate
        fishing potential. For supervised learning with REAL fishing data,
        use fit() with ground truth labels from Global Fishing Watch.

        Domain knowledge scoring is based on:
        - SST in optimal range (15-20°C for Humboldt species) = higher score
        - Thermal fronts (fish aggregation zones) = higher score
        - Calm waves (safe fishing conditions) = higher score
        - Proximity to coast (artisanal fishing zones) = higher score
        - Upwelling conditions (nutrient-rich waters) = higher score
        """
        if X.shape[0] < 5:
            raise ValueError("Need at least 5 samples")

        # Generate scores based on oceanographic domain knowledge
        y = self._calculate_domain_knowledge_scores(X)

        return self.fit(X, y, feature_names)

    def _calculate_domain_knowledge_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate fishing potential scores using oceanographic domain knowledge.

        This is NOT synthetic data - it's a heuristic scoring function based on
        peer-reviewed oceanographic research for the Humboldt Current System:
        - IMARPE (Instituto del Mar del Perú) research
        - FAO fisheries reports for Peru
        - Chavez et al. (2008) - Climate and fisheries in the Humboldt Current

        For REAL ground truth, use Global Fishing Watch data instead.

        Feature indices (32 features):
        - SST (6): 0-5: sst, sst_anomaly, sst_optimal_score, sst_species_score, sst_variability, sst_trend
        - Thermal fronts (5): 6-10: gradient_mag, grad_dir_sin, grad_dir_cos, is_front, front_intensity
        - Currents (6): 11-16: speed, u, v, convergence, shear, toward_coast
        - Waves (3): 17-19: height, period, favorable
        - Upwelling (3): 20-22: index, ekman, favorable
        - Spatial (4): 23-26: dist_coast, depth, coastal_zone, offshore_zone
        - Historical (2): 27-28: hotspot_dist, hotspot_sim
        - Temporal (3): 29-31: hour_score, moon_score, season_score
        """
        scores = np.zeros(X.shape[0])
        n_feat = X.shape[1]

        for i in range(X.shape[0]):
            score = 40.0  # Base score

            # === SST Features (high importance) ===
            sst_optimal = X[i, 2] if n_feat > 2 else 0.5
            sst_species = X[i, 3] if n_feat > 3 else 0.5
            sst_anomaly = X[i, 1] if n_feat > 1 else 0

            score += sst_optimal * 15  # Up to +15 for optimal SST
            score += sst_species * 10  # Up to +10 for species match
            score += max(0, 5 - sst_anomaly)  # Penalty for anomaly

            # === Thermal Front Features (very high importance) ===
            is_front = X[i, 9] if n_feat > 9 else 0
            front_intensity = X[i, 10] if n_feat > 10 else 0

            if is_front > 0.5:
                score += 12 + front_intensity * 8  # +12 to +20 for fronts

            # === Current Features ===
            current_speed = X[i, 11] if n_feat > 11 else 0.1
            convergence = X[i, 14] if n_feat > 14 else 0
            toward_coast = X[i, 16] if n_feat > 16 else 0

            # Moderate current optimal
            if 0.05 <= current_speed <= 0.4:
                score += 5

            # Convergence zones attract fish
            score += min(8, convergence * 20)

            # Current toward coast brings nutrients
            score += min(5, toward_coast * 10)

            # === Wave Features ===
            wave_favorable = X[i, 19] if n_feat > 19 else 1
            score += wave_favorable * 8  # Up to +8 for calm seas

            # === Upwelling Features (high importance for Humboldt) ===
            upwelling_index = X[i, 20] if n_feat > 20 else 0
            upwelling_favorable = X[i, 22] if n_feat > 22 else 0

            score += upwelling_index * 10  # Up to +10
            score += upwelling_favorable * 5  # +5 bonus

            # === Spatial Features ===
            dist_coast_km = X[i, 23] if n_feat > 23 else 5
            coastal_zone = X[i, 25] if n_feat > 25 else 0

            # Shore fishing proximity bonus
            if coastal_zone > 0.5:
                score += 10
            elif dist_coast_km < 8:
                score += 6
            elif dist_coast_km < 15:
                score += 3

            # === Historical Features ===
            hotspot_sim = X[i, 28] if n_feat > 28 else 0
            score += hotspot_sim * 12  # Up to +12 for hotspot similarity

            # === Temporal Features ===
            hour_score = X[i, 29] if n_feat > 29 else 0.5
            moon_score = X[i, 30] if n_feat > 30 else 0.5
            season_score = X[i, 31] if n_feat > 31 else 0.5

            score += hour_score * 5  # Dawn/dusk bonus
            score += moon_score * 4  # Lunar phase bonus
            score += season_score * 3  # Seasonal bonus

            scores[i] = min(100, max(0, score))

        return scores

    def predict(self, X: np.ndarray) -> List[PredictionResult]:
        """
        Predict fishing scores for new data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            List of PredictionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        results = []
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Get cluster assignments
        clusters = self.clusterer.predict(X_pca)

        # Get scores from regressor
        scores = self.regressor.predict(X_scaled)
        scores = np.clip(scores, 0, 100)

        # Calculate confidence based on distance to cluster center
        distances = self.clusterer.transform(X_pca)
        min_distances = distances.min(axis=1)
        max_dist = min_distances.max() if min_distances.max() > 0 else 1
        confidences = 1 - (min_distances / max_dist)

        # Get feature contributions
        feature_importance = self.regressor.feature_importances_

        for i in range(X.shape[0]):
            # Contributing factors
            contrib = {}
            for j, name in enumerate(self.feature_names):
                if j < len(feature_importance):
                    contrib[name] = float(feature_importance[j] * X_scaled[i, j])

            results.append(PredictionResult(
                lat=0.0,  # Will be set by caller
                lon=0.0,
                score=float(scores[i]),
                confidence=float(confidences[i]),
                cluster_id=int(clusters[i]),
                principal_components=X_pca[i],
                contributing_factors=contrib
            ))

        return results

    def get_pca_analysis(self) -> Dict:
        """Get PCA analysis results."""
        if not self.is_fitted:
            return {}

        analysis = {
            'explained_variance_ratio': self.pca_explained_variance.tolist(),
            'cumulative_variance': np.cumsum(self.pca_explained_variance).tolist(),
            'n_components': self.pca.n_components_,
            'component_loadings': []
        }

        # Feature loadings for each component
        for i, comp in enumerate(self.pca_components):
            loadings = {
                self.feature_names[j]: float(comp[j])
                for j in range(min(len(comp), len(self.feature_names)))
            }
            # Sort by absolute value
            loadings = dict(sorted(
                loadings.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))
            analysis['component_loadings'].append({
                'component': i + 1,
                'explained_variance': float(self.pca_explained_variance[i]),
                'loadings': loadings
            })

        return analysis

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers in PCA space."""
        if not self.is_fitted:
            return np.array([])
        return self.clusterer.cluster_centers_

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from regressor."""
        if not self.is_fitted:
            return {}

        importance = {}
        for i, name in enumerate(self.feature_names):
            if i < len(self.regressor.feature_importances_):
                importance[name] = float(self.regressor.feature_importances_[i])

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_model_summary(self) -> str:
        """Get human-readable model summary."""
        if not self.is_fitted:
            return "Model not fitted"

        lines = [
            "=" * 50,
            "RESUMEN DEL MODELO ML",
            "=" * 50,
            "",
            f"Muestras entrenadas: {self.scaler.n_samples_seen_}",
            f"Features: {len(self.feature_names)}",
            f"Componentes PCA: {self.pca.n_components_}",
            f"Clusters: {self.n_clusters}",
            "",
            "VARIANZA EXPLICADA (PCA):",
        ]

        for i, var in enumerate(self.pca_explained_variance):
            cum = sum(self.pca_explained_variance[:i+1])
            lines.append(f"  PC{i+1}: {var:.1%} (acumulado: {cum:.1%})")

        lines.append("")
        lines.append("TOP 5 FEATURES MÁS IMPORTANTES:")

        importance = self.get_feature_importance()
        for i, (name, imp) in enumerate(list(importance.items())[:5]):
            lines.append(f"  {i+1}. {name}: {imp:.3f}")

        return "\n".join(lines)
