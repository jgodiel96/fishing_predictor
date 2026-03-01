#!/usr/bin/env python3
"""
Validacion de registros de pesca vs predicciones del modelo.

Compara capturas reales de pescadores con los scores que el modelo
predice para esas mismas fechas y ubicaciones.

Uso:
    python scripts/validar_registros.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from controllers.analysis import AnalysisController
from config import COASTLINE_FILE


def cargar_registros() -> list:
    """Carga registros de pesca desde encuestas_pesca.json."""
    json_path = PROJECT_ROOT / "data" / "encuestas_pesca.json"
    if not json_path.exists():
        print(f"ERROR: No se encuentra {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("encuestas", [])


def encontrar_spot_cercano(spots: list, lat: float, lon: float) -> dict:
    """Encuentra el spot mas cercano a las coordenadas dadas."""
    mejor = None
    mejor_dist = float('inf')

    for spot in spots:
        dist = ((spot['lat'] - lat)**2 + (spot['lon'] - lon)**2)**0.5
        if dist < mejor_dist:
            mejor_dist = dist
            mejor = spot

    return mejor


def run_validation():
    """Ejecuta validacion completa."""
    registros = cargar_registros()
    if not registros:
        print("No hay registros para validar.")
        return

    print("=" * 70)
    print("   VALIDACION: REGISTROS DE PESCA vs PREDICCION DEL MODELO")
    print("=" * 70)
    print(f"Registros a validar: {len(registros)}\n")

    # Agrupar por fecha para evitar correr el modelo multiples veces
    por_fecha = {}
    for reg in registros:
        fecha = reg["fecha"]
        if fecha not in por_fecha:
            por_fecha[fecha] = []
        por_fecha[fecha].append(reg)

    resultados = []

    for fecha, regs in sorted(por_fecha.items()):
        print(f"\n{'─' * 70}")
        print(f"FECHA: {fecha}")
        print(f"{'─' * 70}")

        # Inicializar controller para esta fecha
        controller = AnalysisController()
        target_dt = datetime.strptime(fecha, "%Y-%m-%d")
        controller.analysis_datetime = target_dt

        # 1. Cargar costa
        coastline_path = str(COASTLINE_FILE)
        if not Path(coastline_path).exists():
            print(f"  ERROR: No se encuentra coastline: {coastline_path}")
            continue

        n_pts = controller.load_coastline(coastline_path)
        print(f"  Costa: {n_pts} puntos")

        # 2. Muestrear spots
        spots = controller.sample_fishing_spots(spacing_m=300, max_spots=600)
        print(f"  Spots: {len(spots)}")

        # 3. Generar zonas
        controller.generate_fish_zones()

        # 4. Fetch datos marinos
        print(f"  Obteniendo datos marinos para {fecha}...")
        n_vectors = controller.fetch_marine_data()
        print(f"  Vectores: {n_vectors}")

        # 5. Condiciones
        conditions = controller.get_conditions()

        # 6. ML prediction
        pca = controller.run_ml_prediction()

        # 7. Analizar spots para la hora de cada registro
        for reg in regs:
            hora_str = reg.get("hora_detalle", "10:00")
            try:
                hora = int(hora_str.split(":")[0])
            except (ValueError, IndexError):
                hora_map = {
                    "madrugada": 5, "manana": 9, "mediodia": 13,
                    "tarde": 17, "noche": 21
                }
                hora = hora_map.get(reg.get("hora", "manana"), 10)

            analyzed = controller.analyze_spots(target_hour=hora)

            # Buscar spot mas cercano al punto de pesca
            spot = encontrar_spot_cercano(analyzed, reg["lat"], reg["lon"])

            if not spot:
                print(f"\n  [!] No se encontro spot cercano para ({reg['lat']}, {reg['lon']})")
                continue

            dist_km = ((spot['lat'] - reg['lat'])**2 + (spot['lon'] - reg['lon'])**2)**0.5 * 111

            resultado = {
                "fecha": reg["fecha"],
                "hora": hora,
                "zona": reg.get("zona_nombre", reg.get("zona", "?")),
                "lat": reg["lat"],
                "lon": reg["lon"],
                "especie": ", ".join(reg.get("especies", [])),
                "kg": reg.get("cantidad_kg", 0),
                "metodo": reg.get("metodo", "?"),
                "score_modelo": spot["score"],
                "spot_lat": spot["lat"],
                "spot_lon": spot["lon"],
                "dist_spot_km": dist_km,
                "tide_phase": spot.get("tide_phase", "?"),
                "tide_score": spot.get("tide_score", 0),
                "hour_score": spot.get("hour_score", 0),
                "sss_score": spot.get("sss_score", 0),
                "sla_score": spot.get("sla_score", 0),
                "chla_score": spot.get("chla_score", 0),
            }
            resultados.append(resultado)

            # Mostrar resultado
            print(f"\n  REGISTRO: {reg.get('zona_nombre', reg.get('zona'))}")
            print(f"    Ubicacion: ({reg['lat']:.4f}, {reg['lon']:.4f})")
            print(f"    Hora: {hora}:00 | Especie: {resultado['especie']} | Captura: {resultado['kg']}kg")
            print(f"    Metodo: {reg.get('metodo_detalle', reg.get('metodo', '?'))}")
            print(f"    Mar: {reg.get('mar', '?')}")
            print(f"  PREDICCION DEL MODELO:")
            print(f"    Score: {spot['score']:.1f}/100")
            print(f"    Spot cercano: ({spot['lat']:.4f}, {spot['lon']:.4f}) a {dist_km:.1f}km")
            print(f"    Marea: {spot.get('tide_phase', '?')} (score: {spot.get('tide_score', 0):.2f})")
            print(f"    Hour score: {spot.get('hour_score', 0):.2f}")

            # Evaluacion
            score = spot['score']
            if score >= 60:
                veredicto = "COINCIDE - modelo predijo zona buena"
            elif score >= 40:
                veredicto = "PARCIAL - modelo predijo zona promedio"
            else:
                veredicto = "FALLA - modelo NO predijo esta zona como buena"
            print(f"  VEREDICTO: {veredicto}")

    # Resumen final
    print(f"\n\n{'=' * 70}")
    print("   RESUMEN DE VALIDACION")
    print(f"{'=' * 70}")

    if not resultados:
        print("No se obtuvieron resultados.")
        return

    scores = [r["score_modelo"] for r in resultados]
    kgs = [r["kg"] for r in resultados]

    print(f"\n  Total registros validados: {len(resultados)}")
    print(f"  Score promedio del modelo: {sum(scores)/len(scores):.1f}/100")
    print(f"  Score min/max: {min(scores):.1f} / {max(scores):.1f}")
    print(f"  Captura promedio: {sum(kgs)/len(kgs):.1f} kg")

    # Tabla resumen
    print(f"\n  {'Fecha':<12} {'Zona':<35} {'Kg':>6} {'Score':>6} {'Veredicto'}")
    print(f"  {'─'*12} {'─'*35} {'─'*6} {'─'*6} {'─'*20}")
    for r in resultados:
        score = r["score_modelo"]
        if score >= 60:
            v = "OK"
        elif score >= 40:
            v = "PARCIAL"
        else:
            v = "FALLA"
        print(f"  {r['fecha']:<12} {r['zona']:<35} {r['kg']:>5.0f} {score:>5.1f} {v}")

    # Correlacion simple
    if len(resultados) >= 3:
        mean_kg = sum(kgs) / len(kgs)
        mean_score = sum(scores) / len(scores)
        numerator = sum((k - mean_kg) * (s - mean_score) for k, s in zip(kgs, scores))
        denom_kg = sum((k - mean_kg)**2 for k in kgs)**0.5
        denom_score = sum((s - mean_score)**2 for s in scores)**0.5

        if denom_kg > 0 and denom_score > 0:
            correlation = numerator / (denom_kg * denom_score)
            print(f"\n  Correlacion Score-Captura: {correlation:.3f}")
            if correlation > 0.5:
                print("  -> Buena correlacion positiva")
            elif correlation > 0:
                print("  -> Correlacion positiva debil")
            elif correlation > -0.5:
                print("  -> Correlacion negativa debil")
            else:
                print("  -> Correlacion negativa (modelo necesita ajuste)")

    # Guardar resultados
    output_path = PROJECT_ROOT / "output" / "validacion_registros.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "fecha_validacion": datetime.now().isoformat(),
            "total_registros": len(resultados),
            "score_promedio": sum(scores) / len(scores),
            "resultados": resultados
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run_validation()
