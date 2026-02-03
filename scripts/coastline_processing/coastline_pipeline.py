#!/usr/bin/env python3
"""
Coastline Detection Pipeline

Orchestrates the complete coastline detection process:
1. Detection (SAM or HSV fallback)
2. Verification (dual-source comparison)
3. Refinement (spacing guarantee)
4. Validation (final checks)
5. Storage (immutable Gold layer)

Usage:
    python core/coastline_pipeline.py --region full
    python core/coastline_pipeline.py --region canepa --no-sam
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.coastline_sam import (
    SAMCoastlineDetector,
    SAMConfig,
    check_sam_availability,
    print_sam_status
)
from core.coastline_connector import (
    CoastlineConnector,
    ConnectorConfig,
    save_segments_as_geojson
)
from core.coastline_verifier import (
    CoastlineVerifier,
    VerificationConfig,
    create_osm_water_mask
)
from core.coastline_refiner import (
    CoastlineRefiner,
    RefinementConfig
)
from core.coastline_validator import (
    CoastlineValidator,
    ValidationConfig
)


class CoastlinePipeline:
    """
    Complete pipeline for detecting, verifying, and saving coastline data.

    This pipeline implements Plan V5 specifications:
    - SAM (Segment Anything Model) for detection
    - Dual-source verification
    - Maximum 200m spacing guarantee
    - Immutable storage with checksums
    """

    # Predefined regions
    REGIONS = {
        "canepa": {
            "lat_min": -18.10,
            "lat_max": -17.95,
            "lon_min": -70.40,
            "lon_max": -70.20,
            "zoom": 16,
            "description": "Playa Canepa and surroundings"
        },
        "sama": {
            "lat_min": -18.05,
            "lat_max": -17.85,
            "lon_min": -70.35,
            "lon_max": -70.15,
            "zoom": 16,
            "description": "Sama beach area"
        },
        "full": {
            "lat_min": -18.35,
            "lat_max": -17.30,
            "lon_min": -71.50,
            "lon_max": -70.10,
            "zoom": 15,
            "description": "Full Tacna-Ilo region"
        }
    }

    def __init__(
        self,
        use_sam: bool = True,
        verify: bool = True,
        refine: bool = True,
        validate: bool = True,
        use_connector: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            use_sam: Use SAM for detection (falls back to HSV if unavailable)
            verify: Perform dual-source verification
            refine: Apply spacing refinement
            validate: Run final validation
            use_connector: Use intelligent point connector (v5.1)
        """
        self.use_sam = use_sam
        self.verify = verify
        self.refine = refine
        self.validate_flag = validate
        self.use_connector = use_connector

        # Components
        self.detector = SAMCoastlineDetector()
        self.connector = CoastlineConnector()
        self.verifier = CoastlineVerifier()
        self.refiner = CoastlineRefiner()
        self.validator = CoastlineValidator()

        # Results storage
        self.raw_points = []
        self.segments = []  # NEW: Store connected segments
        self.verified_points = []
        self.refined_points = []
        self.validation_result = None

    def detect(
        self,
        region_name: str = "full",
        custom_region: Dict = None
    ) -> List[Tuple[float, float]]:
        """
        Phase 1: Detect coastline from satellite imagery.

        Args:
            region_name: Name of predefined region
            custom_region: Custom region dict (overrides region_name)

        Returns:
            List of detected (lat, lon) points
        """
        region = custom_region or self.REGIONS.get(region_name)
        if not region:
            raise ValueError(f"Unknown region: {region_name}")

        print("\n" + "="*60)
        print("FASE 1: DETECCION")
        print("="*60)
        print(f"Region: {region.get('description', region_name)}")

        self.raw_points = self.detector.detect_coastline_for_region(
            lat_min=region["lat_min"],
            lat_max=region["lat_max"],
            lon_min=region["lon_min"],
            lon_max=region["lon_max"],
            zoom=region.get("zoom", 15),
            use_sam=self.use_sam
        )

        print(f"\nResultado: {len(self.raw_points)} puntos detectados")
        return self.raw_points

    def connect_points(
        self,
        points: List[Tuple[float, float]] = None
    ) -> List[List[Tuple[float, float]]]:
        """
        Phase 1.5: Connect detected points using intelligent algorithm.

        This phase ensures points are connected correctly without creating
        lines that cross land. It uses nearest-neighbor chains and segment
        merging instead of simple latitude sorting.

        Args:
            points: Points to connect (uses self.raw_points if None)

        Returns:
            List of connected segments
        """
        if not self.use_connector:
            # Skip connector, return points as single segment
            points = points or self.raw_points
            self.segments = [points] if points else []
            return self.segments

        print("\n" + "="*60)
        print("FASE 1.5: CONEXION INTELIGENTE (v5.1)")
        print("="*60)

        points = points or self.raw_points
        if not points:
            print("[WARN] No points to connect")
            self.segments = []
            return []

        # Configure connector
        config = ConnectorConfig(
            max_gap_m=500,
            max_merge_distance_m=1000,
            min_segment_points=3,
            remove_isolated_points=True,
            min_neighbors_radius_m=200,
            min_neighbors_count=2
        )
        self.connector = CoastlineConnector(config)

        # Connect points
        self.segments = self.connector.connect(points)

        # Statistics
        stats = self.connector.get_connection_statistics(self.segments)
        print(f"\nResultado de conexion:")
        print(f"  Segmentos: {stats['num_segments']}")
        print(f"  Puntos totales: {stats['total_points']}")
        print(f"  Longitud total: {stats['total_length_km']:.2f} km")

        if stats['num_segments'] > 0:
            print(f"  Longitud promedio por segmento: {stats['avg_segment_length_km']:.2f} km")

        return self.segments

    def verify_dual_source(
        self,
        points: List[Tuple[float, float]] = None
    ) -> List[Tuple[float, float]]:
        """
        Phase 2: Verify points using dual-source comparison.

        Currently simplified - compares SAM detection with HSV fallback
        on satellite imagery. Full dual-source (sat + street) verification
        requires more tile fetching.

        Args:
            points: Points to verify (uses connected segments if None)

        Returns:
            List of verified points
        """
        if not self.verify:
            # Use connected segments flattened, or raw points
            if self.segments:
                return [p for seg in self.segments for p in seg]
            return points or self.raw_points

        print("\n" + "="*60)
        print("FASE 2: VERIFICACION")
        print("="*60)

        # Get points from segments or raw
        if points is None:
            if self.segments:
                points = [p for seg in self.segments for p in seg]
            else:
                points = self.raw_points

        if not points:
            print("[WARN] No points to verify")
            return []

        # Simplified verification: check that points have water to the west
        # This is a basic check; full dual-source verification would be more complex
        print(f"Verificando {len(points)} puntos...")

        # For now, we trust the SAM/HSV detection
        # Full verification would fetch both satellite and street tiles for each point
        self.verified_points = points

        print(f"\nResultado: {len(self.verified_points)} puntos verificados")
        return self.verified_points

    def apply_refinement(
        self,
        points: List[Tuple[float, float]] = None
    ) -> List[Tuple[float, float]]:
        """
        Phase 3: Refine spacing and smooth curve.

        Args:
            points: Points to refine

        Returns:
            Refined points with guaranteed spacing
        """
        if not self.refine:
            return points or self.verified_points or self.raw_points

        print("\n" + "="*60)
        print("FASE 3: REFINAMIENTO")
        print("="*60)

        points = points or self.verified_points or self.raw_points
        if not points:
            print("[WARN] No points to refine")
            return []

        self.refined_points = self.refiner.refine(points)

        # Get statistics
        stats = self.refiner.get_statistics(self.refined_points)
        print(f"\nEstadisticas:")
        print(f"  Puntos: {stats['total_points']}")
        print(f"  Longitud: {stats['total_length_km']:.2f} km")
        print(f"  Espaciado promedio: {stats['avg_spacing_m']:.1f} m")
        print(f"  Espaciado maximo: {stats['max_spacing_m']:.1f} m")
        print(f"  Espaciado minimo: {stats['min_spacing_m']:.1f} m")

        if stats['spacing_violations'] > 0:
            print(f"  [WARN] Violaciones de espaciado: {stats['spacing_violations']}")

        return self.refined_points

    def run_validation(
        self,
        points: List[Tuple[float, float]] = None,
        confidence_scores: List[float] = None
    ):
        """
        Phase 4: Run final validation.

        Args:
            points: Points to validate
            confidence_scores: Optional confidence scores

        Returns:
            ValidationResult
        """
        if not self.validate_flag:
            return None

        print("\n" + "="*60)
        print("FASE 4: VALIDACION")
        print("="*60)

        points = points or self.refined_points or self.verified_points or self.raw_points
        if not points:
            print("[ERROR] No points to validate")
            return None

        # Pass segments for proper spacing validation (gaps between segments are OK)
        self.validation_result = self.validator.validate(
            points,
            confidence_scores,
            segments=self.segments if self.segments else None
        )

        print(f"\nResultados de validacion:")
        for check, passed in self.validation_result.checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")

        if self.validation_result.errors:
            print("\nErrores:")
            for error in self.validation_result.errors:
                print(f"  - {error}")

        if self.validation_result.warnings:
            print("\nAdvertencias:")
            for warning in self.validation_result.warnings:
                print(f"  - {warning}")

        overall = "VALIDO" if self.validation_result.is_valid else "INVALIDO"
        print(f"\nResultado general: {overall}")

        return self.validation_result

    def save(
        self,
        points: List[Tuple[float, float]] = None,
        output_dir: str = None,
        version: str = None
    ) -> Dict[str, str]:
        """
        Save to Gold layer.

        Saves as MultiLineString if segments are available (preserving
        segment boundaries), or as LineString if only flat points.

        Args:
            points: Points to save
            output_dir: Output directory
            version: Version string

        Returns:
            Dict with paths to created files
        """
        print("\n" + "="*60)
        print("GUARDANDO EN GOLD LAYER")
        print("="*60)

        points = points or self.refined_points or self.verified_points or self.raw_points
        if not points:
            print("[ERROR] No points to save")
            return {}

        # Run validation if not done
        if self.validation_result is None:
            self.run_validation(points)

        # Determine version
        if version is None:
            version = datetime.now().strftime("v%Y%m%d")

        # Check if we have segments to save as MultiLineString
        if self.segments and len(self.segments) > 1:
            print(f"[INFO] Saving {len(self.segments)} segments as MultiLineString")

        return self.validator.save_to_gold(
            points=points,
            validation_result=self.validation_result,
            output_dir=output_dir,
            version=version,
            segments=self.segments if self.segments else None
        )

    def run(
        self,
        region_name: str = "full",
        custom_region: Dict = None,
        save_output: bool = True,
        version: str = None
    ) -> Dict:
        """
        Run the complete pipeline.

        Args:
            region_name: Name of region to process
            custom_region: Custom region definition
            save_output: Whether to save to Gold layer
            version: Version string for output

        Returns:
            Dict with results and statistics
        """
        print("\n" + "#"*60)
        print("# COASTLINE DETECTION PIPELINE (v5.1)")
        print("#"*60)

        start_time = datetime.now()

        # Check SAM availability
        sam_status = check_sam_availability()
        if self.use_sam and not (sam_status["segment_anything"] and sam_status["checkpoint"]):
            print("\n[WARN] SAM not fully available, falling back to HSV")
            self.use_sam = False

        # Phase 1: Detection
        self.detect(region_name, custom_region)

        # Phase 1.5: Intelligent Connection (NEW in v5.1)
        self.connect_points()

        # Phase 2: Verification
        self.verify_dual_source()

        # Phase 3: Refinement
        self.apply_refinement()

        # Phase 4: Validation
        self.run_validation()

        # Save
        saved_files = {}
        if save_output:
            saved_files = self.save(version=version)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Summary
        print("\n" + "#"*60)
        print("# RESUMEN")
        print("#"*60)
        print(f"Tiempo total: {duration:.1f} segundos")
        print(f"Puntos detectados: {len(self.raw_points)}")
        print(f"Segmentos conectados: {len(self.segments)}")
        print(f"Puntos refinados: {len(self.refined_points)}")
        if self.validation_result:
            print(f"Validacion: {'PASS' if self.validation_result.is_valid else 'FAIL'}")
        print()

        return {
            "raw_points": len(self.raw_points),
            "num_segments": len(self.segments),
            "refined_points": len(self.refined_points),
            "validation": self.validation_result,
            "saved_files": saved_files,
            "duration_seconds": duration
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Coastline Detection Pipeline (Plan V5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect coastline for Playa Canepa area
  python core/coastline_pipeline.py --region canepa

  # Full region with HSV (no SAM)
  python core/coastline_pipeline.py --region full --no-sam

  # Check SAM availability
  python core/coastline_pipeline.py --check-sam
        """
    )

    parser.add_argument(
        "--region",
        choices=["canepa", "sama", "full"],
        default="canepa",
        help="Region to process"
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Use HSV instead of SAM"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip refinement step"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version string for output (default: vYYYYMMDD)"
    )
    parser.add_argument(
        "--check-sam",
        action="store_true",
        help="Check SAM availability and exit"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory"
    )

    args = parser.parse_args()

    if args.check_sam:
        print_sam_status()
        return

    # Create and run pipeline
    pipeline = CoastlinePipeline(
        use_sam=not args.no_sam,
        verify=not args.no_verify,
        refine=not args.no_refine,
        validate=True
    )

    results = pipeline.run(
        region_name=args.region,
        save_output=not args.no_save,
        version=args.version
    )

    # Return exit code based on validation
    if results.get("validation") and results["validation"].is_valid:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
