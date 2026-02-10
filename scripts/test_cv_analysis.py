#!/usr/bin/env python3
"""
Test CV Analysis Pipeline - V8

Ejecuta analisis de vision computacional en zona de prueba:
- Zona rocosa costera de Ilo, Peru
- Coordenadas: -17.65 to -17.68 lat, -71.33 to -71.36 lon

Genera:
- Linea costera precisa
- Mapa de sustrato (rock/sand/mixed)
- Estimacion de profundidad
- Zonas de especies con colores
- Archivo GeoJSON para visualizacion

Usage:
    python scripts/test_cv_analysis.py
"""

import sys
import logging
from pathlib import Path

# Agregar directorio raiz al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Verifica dependencias necesarias."""
    missing = []

    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        from PIL import Image
        logger.info("PIL disponible")
    except ImportError:
        missing.append("Pillow")

    try:
        import requests
        logger.info("Requests disponible")
    except ImportError:
        missing.append("requests")

    if missing:
        logger.error(f"Dependencias faltantes: {', '.join(missing)}")
        logger.error("Instala con: pip install " + " ".join(missing))
        return False

    return True


def run_test():
    """Ejecuta prueba de CV analysis."""
    from core.cv_analysis import (
        CVAnalysisPipeline,
        TileConfig,
        SPECIES_DATABASE
    )

    # Zona de prueba: Costa rocosa de Ilo
    # Area pequena para prueba rapida
    LAT_MIN = -17.68
    LAT_MAX = -17.65
    LON_MIN = -71.36
    LON_MAX = -71.33

    logger.info("=" * 60)
    logger.info("TEST CV ANALYSIS PIPELINE - V8")
    logger.info("=" * 60)
    logger.info(f"Zona: ({LAT_MIN}, {LON_MIN}) - ({LAT_MAX}, {LON_MAX})")

    # Configurar pipeline con zoom alto para detalle
    config = TileConfig(zoom=17, source='esri')
    pipeline = CVAnalysisPipeline(tile_config=config, grid_size=32)

    logger.info("\nIniciando analisis...")

    # Ejecutar analisis
    result = pipeline.analyze_area(
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        include_species_zones=True
    )

    # Mostrar resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)

    logger.info(f"\nTiempo de procesamiento: {result.processing_time_s:.2f}s")
    logger.info(f"Puntos de linea costera: {len(result.coastline)}")
    logger.info(f"Zonas de especies generadas: {len(result.species_zones)}")

    # Estadisticas
    if 'water_percentage' in result.stats:
        logger.info(f"\nPorcentaje de agua: {result.stats['water_percentage']:.1f}%")

    if 'substrate_distribution' in result.stats:
        logger.info("\nDistribucion de sustrato:")
        for key, value in result.stats['substrate_distribution'].items():
            logger.info(f"  - {key}: {value:.1f}%")

    if 'depth_range' in result.stats:
        logger.info("\nRango de profundidad:")
        dr = result.stats['depth_range']
        logger.info(f"  - Min: {dr['min']:.1f}m")
        logger.info(f"  - Max: {dr['max']:.1f}m")
        logger.info(f"  - Media: {dr['mean']:.1f}m")

    # Mostrar zonas de especies
    if result.species_zones:
        logger.info("\n" + "-" * 40)
        logger.info("ZONAS DE ESPECIES")
        logger.info("-" * 40)

        # Agrupar por especie principal
        species_zones = {}
        for zone in result.species_zones:
            sp = zone.primary_species
            if sp not in species_zones:
                species_zones[sp] = []
            species_zones[sp].append(zone)

        for species_id, zones in sorted(species_zones.items()):
            species = SPECIES_DATABASE.get(species_id)
            name = species.name_es if species else species_id
            total_area = sum(z.area_km2 for z in zones)
            logger.info(f"\n{name}:")
            logger.info(f"  - Zonas: {len(zones)}")
            logger.info(f"  - Area total: {total_area:.4f} km^2")
            logger.info(f"  - Profundidad media: {zones[0].avg_depth:.1f}m")
            logger.info(f"  - Sustrato: {zones[0].substrate.value}")

    # Guardar GeoJSON
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    geojson_path = output_dir / "cv_analysis_test.geojson"
    result.save_geojson(geojson_path)
    logger.info(f"\nGeoJSON guardado: {geojson_path}")

    # Guardar resumen JSON
    import json
    summary_path = output_dir / "cv_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info(f"Resumen guardado: {summary_path}")

    # Probar consulta puntual
    logger.info("\n" + "-" * 40)
    logger.info("CONSULTA PUNTUAL")
    logger.info("-" * 40)

    test_lat = (LAT_MIN + LAT_MAX) / 2
    test_lon = (LON_MIN + LON_MAX) / 2

    depth_result = pipeline.get_depth_at_point(test_lat, test_lon)
    substrate_result = pipeline.get_substrate_at_point(test_lat, test_lon)

    logger.info(f"\nPunto: ({test_lat:.4f}, {test_lon:.4f})")
    logger.info(f"Profundidad: {depth_result.depth:.1f}m (fuente: {depth_result.source})")
    logger.info(f"Sustrato: {substrate_result.substrate_type.value}")
    logger.info(f"  - Prob. roca: {substrate_result.rock_probability:.2f}")
    logger.info(f"  - Prob. arena: {substrate_result.sand_probability:.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

    return result


def main():
    """Punto de entrada principal."""
    if not check_dependencies():
        sys.exit(1)

    try:
        result = run_test()
        return 0
    except Exception as e:
        logger.error(f"Error durante el test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
