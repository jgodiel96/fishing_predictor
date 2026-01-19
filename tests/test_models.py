"""
Tests for ML models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.features import FeatureExtractor, MarineFeatures
from models.predictor import FishingPredictor, PredictionResult


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_initialization(self):
        """Test extractor initializes correctly."""
        extractor = FeatureExtractor()
        assert extractor.SST_OPTIMAL['min'] == 14.0
        assert extractor.SST_OPTIMAL['max'] == 24.0
        assert len(extractor.feature_names) == 32

    def test_feature_vector_length(self):
        """Test feature vector has correct length (32 features)."""
        extractor = FeatureExtractor()
        mf = MarineFeatures(
            lat=-17.5, lon=-71.3,
            sst=17.5, sst_anomaly=0,
            current_speed=0.1, current_direction=180,
            wave_height=1.0, wave_period=8.0,
            distance_to_coast=1000, depth_proxy=20,
            gradient_magnitude=0.05, gradient_direction=90,
            is_thermal_front=True, front_intensity=0.5,
            upwelling_index=0.3, ekman_transport=0.1,
            current_convergence=0.01, current_shear=0.01,
            productivity_index=0.5,
            historical_hotspot_distance=5000, hotspot_similarity=0.6,
            hour_sin=0.5, hour_cos=0.866,
            day_of_year_sin=0.5, day_of_year_cos=0.866,
            moon_phase=0.5, is_major_period=False
        )
        vector = extractor._to_vector(mf)
        assert len(vector) == 32

    def test_sst_optimal_score(self):
        """Test SST optimal scoring."""
        extractor = FeatureExtractor()

        # Optimal SST at center (17.5)
        score_optimal = extractor._sst_optimal_score(17.5)
        assert score_optimal > 0.9  # Near 1.0

        # SST at edge of range
        score_edge = extractor._sst_optimal_score(14.0)
        assert 0.1 < score_edge < 0.5

        # SST out of range
        score_cold = extractor._sst_optimal_score(10.0)
        assert score_cold == 0.1

    def test_sst_species_score(self):
        """Test species-specific SST scoring."""
        extractor = FeatureExtractor()

        # Optimal for Cabrilla (16-19°C)
        score = extractor._sst_species_score(17.5)
        assert score > 0.8

        # Out of range for all species
        score_cold = extractor._sst_species_score(10.0)
        assert score_cold == 0.1

    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        extractor = FeatureExtractor()

        # Same point = 0 distance
        dist = extractor._haversine(-17.5, -71.3, -17.5, -71.3)
        assert dist == 0.0

        # ~1 degree latitude ≈ 111km
        dist = extractor._haversine(-17.0, -71.0, -18.0, -71.0)
        assert 110000 < dist < 112000

    def test_thermal_front_detection(self):
        """Test gradient threshold for thermal fronts."""
        extractor = FeatureExtractor()
        assert extractor.GRADIENT_THRESHOLD == 0.04
        assert extractor.STRONG_FRONT_THRESHOLD == 0.1


class TestFishingPredictor:
    """Tests for FishingPredictor."""

    def test_initialization(self):
        """Test predictor initializes correctly."""
        predictor = FishingPredictor(n_components=3, n_clusters=4)
        assert predictor.n_components == 3
        assert predictor.n_clusters == 4
        assert predictor.is_fitted == False

    def test_fit_requires_minimum_samples(self):
        """Test fit raises error with too few samples."""
        predictor = FishingPredictor()
        X = np.random.rand(3, 5)  # Only 3 samples

        with pytest.raises(ValueError):
            predictor.fit(X)

    def test_fit_unsupervised(self):
        """Test unsupervised fitting with 32 features."""
        predictor = FishingPredictor(n_components=5, n_clusters=3)

        # Create sample data with 32 features
        X = np.random.rand(20, 32)
        X[:, 0] = np.random.uniform(14, 22, 20)  # SST
        X[:, 2] = np.clip((X[:, 0] - 14) / 10, 0, 1)  # sst_optimal_score

        predictor.fit_unsupervised(X)

        assert predictor.is_fitted == True
        assert predictor.pca_explained_variance is not None
        assert len(predictor.pca_explained_variance) <= 5

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        predictor = FishingPredictor(n_components=5, n_clusters=3)

        X_train = np.random.rand(20, 32)
        predictor.fit_unsupervised(X_train)

        X_test = np.random.rand(5, 32)
        results = predictor.predict(X_test)

        assert len(results) == 5
        assert all(isinstance(r, PredictionResult) for r in results)
        assert all(0 <= r.score <= 100 for r in results)
        assert all(0 <= r.confidence <= 1 for r in results)

    def test_predict_without_fit_raises_error(self):
        """Test predict raises error if not fitted."""
        predictor = FishingPredictor()
        X = np.random.rand(5, 32)

        with pytest.raises(ValueError):
            predictor.predict(X)

    def test_pca_analysis(self):
        """Test PCA analysis output."""
        predictor = FishingPredictor(n_components=6, n_clusters=3)

        X = np.random.rand(30, 32)
        predictor.fit_unsupervised(X, feature_names=[f"f{i}" for i in range(32)])

        analysis = predictor.get_pca_analysis()

        assert 'explained_variance_ratio' in analysis
        assert 'cumulative_variance' in analysis
        assert 'component_loadings' in analysis
        assert len(analysis['component_loadings']) == 6

    def test_feature_importance(self):
        """Test feature importance extraction."""
        predictor = FishingPredictor()

        X = np.random.rand(30, 32)
        names = [f"feature_{i}" for i in range(32)]
        predictor.fit_unsupervised(X, feature_names=names)

        importance = predictor.get_feature_importance()

        assert len(importance) == 32
        assert all(name in importance for name in names)
        assert sum(importance.values()) > 0

    def test_domain_knowledge_scores_calculation(self):
        """Test domain-knowledge score calculation logic with 32 features."""
        predictor = FishingPredictor()

        # Create data with known characteristics (32 features)
        X = np.zeros((10, 32))

        # Sample 1: Optimal conditions (based on Humboldt Current research)
        # SST features (0-5)
        X[0, 0] = 17.5  # sst (optimal for Humboldt species)
        X[0, 1] = 0     # sst_anomaly
        X[0, 2] = 0.95  # sst_optimal_score
        X[0, 3] = 0.9   # sst_species_score
        # Thermal front (6-10)
        X[0, 9] = 1     # is_thermal_front
        X[0, 10] = 0.8  # front_intensity
        # Currents (11-16)
        X[0, 11] = 0.2  # current_speed
        X[0, 14] = 0.1  # convergence
        # Waves (17-19)
        X[0, 19] = 1    # wave_favorable
        # Upwelling (20-22)
        X[0, 20] = 0.5  # upwelling_index
        X[0, 22] = 1    # upwelling_favorable
        # Spatial (23-26)
        X[0, 23] = 3    # dist_coast_km
        X[0, 25] = 1    # coastal_zone
        # Historical (27-28)
        X[0, 28] = 0.8  # hotspot_similarity
        # Temporal (29-31)
        X[0, 29] = 0.9  # hour_score
        X[0, 30] = 0.9  # moon_score
        X[0, 31] = 0.8  # season_score

        # Sample 2: Poor conditions
        X[1, 0] = 12.0  # cold SST
        X[1, 1] = 5.5   # high anomaly
        X[1, 2] = 0.1   # low optimal score
        X[1, 9] = 0     # no front
        X[1, 19] = 0    # unfavorable waves
        X[1, 23] = 30   # far from coast
        X[1, 28] = 0    # no hotspot similarity

        scores = predictor._calculate_domain_knowledge_scores(X)

        # Optimal conditions should score significantly higher
        assert scores[0] > scores[1]
        assert scores[0] > 70  # Should be high
        assert scores[1] < 60  # Should be lower


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full feature extraction and prediction pipeline."""
        # Create mock marine data
        from dataclasses import dataclass

        @dataclass
        class MockMarinePoint:
            lat: float
            lon: float
            sst: float
            wave_height: float
            wave_period: float
            current_speed: float
            current_direction: float

        # Generate mock data with varying conditions
        points = [
            MockMarinePoint(-17.5 + i*0.1, -71.3, 15.0 + i*0.3, 1.0 + i*0.05, 8.0, 0.1 + i*0.01, 180 + i*5)
            for i in range(20)
        ]
        coastline = [(-17.0 + i*0.1, -71.5) for i in range(30)]

        # Extract features
        extractor = FeatureExtractor()
        X = extractor.extract_from_marine_points(points, coastline)

        assert X.shape[0] == 20
        assert X.shape[1] == 32  # Updated to 32 features

        # Train and predict
        predictor = FishingPredictor(n_components=6, n_clusters=3)
        predictor.fit_unsupervised(X, feature_names=extractor.feature_names)
        results = predictor.predict(X)

        assert len(results) == 20
        assert predictor.is_fitted

    def test_upwelling_calculation(self):
        """Test upwelling index calculation."""
        extractor = FeatureExtractor()

        # Wind from south (upwelling favorable in Peru)
        upwelling, ekman = extractor._calculate_upwelling(-17.5, -71.3, 10, 180)
        assert upwelling >= 0

        # Wind from north (not favorable)
        upwelling_north, _ = extractor._calculate_upwelling(-17.5, -71.3, 10, 0)
        assert upwelling_north == 0

    def test_productivity_estimation(self):
        """Test productivity index estimation."""
        extractor = FeatureExtractor()

        # Cold water + upwelling + front = high productivity
        prod_high = extractor._estimate_productivity(15.0, 0.8, 0.9)
        assert prod_high > 0.5

        # Warm water + no upwelling + no front = low productivity
        prod_low = extractor._estimate_productivity(22.0, 0.0, 0.0)
        assert prod_low < prod_high

    def test_hotspot_analysis(self):
        """Test historical hotspot analysis."""
        extractor = FeatureExtractor()

        # Near Punta Coles hotspot
        dist, sim = extractor._analyze_hotspots(-17.70, -71.35, 17.5)
        assert dist < 1000  # Should be very close
        assert sim > 0.5  # Should have similarity

        # Far from any hotspot
        dist_far, sim_far = extractor._analyze_hotspots(-19.0, -70.0, 17.5)
        assert dist_far > 50000  # Far away
        assert sim_far == 0  # No similarity


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
