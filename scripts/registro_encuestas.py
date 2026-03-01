#!/usr/bin/env python3
"""
Registro de encuestas de pesca desde orilla.

Uso:
    python scripts/registro_encuestas.py              # Agregar nueva encuesta
    python scripts/registro_encuestas.py --ver       # Ver todas las encuestas
    python scripts/registro_encuestas.py --stats     # Ver estadisticas
    python scripts/registro_encuestas.py --exportar  # Exportar a CSV
"""

import json
import csv
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# Archivo de datos
DATA_FILE = Path(__file__).parent.parent / "data" / "encuestas_pesca.json"

# Coordenadas por zona (para el modelo)
ZONAS = {
    "punta_coles": {"lat": -17.702, "lon": -71.385, "nombre": "Punta Coles / Reserva"},
    "vila_vila": {"lat": -17.630, "lon": -71.340, "nombre": "Vila Vila"},
    "pozo_redondo": {"lat": -17.680, "lon": -71.370, "nombre": "Pozo Redondo"},
    "ilo_puerto": {"lat": -17.640, "lon": -71.340, "nombre": "Ilo Puerto"},
    "fundicion": {"lat": -17.660, "lon": -71.350, "nombre": "Fundicion"},
    "punta_mesa": {"lat": -17.850, "lon": -71.200, "nombre": "Punta Mesa"},
    "santa_rosa": {"lat": -17.750, "lon": -71.300, "nombre": "Santa Rosa"},
    "boca_rio": {"lat": -18.050, "lon": -70.850, "nombre": "Boca del Rio"},
    "playa_blanca_gentillar": {"lat": -17.822, "lon": -71.140, "nombre": "Playa Blanca-Gentillar"},
    "llostay": {"lat": -17.96, "lon": -70.88, "nombre": "Llostay"},
    "punta_picata": {"lat": -18.200, "lon": -70.900, "nombre": "Punta Picata"},
}

ESPECIES = ["cabrilla", "corvina", "robalo", "pejerrey", "lenguado", "bonito", "lorna", "chita", "tramboyo", "otro", "nada"]
METODOS = ["spinning", "rockfishing", "carnada", "mosca", "otro"]
HORAS = ["madrugada", "manana", "mediodia", "tarde", "noche"]
MAR = ["calmo", "regular", "picado"]
TEMP_AGUA = ["fria", "normal", "tibia"]
CANTIDADES = ["0", "1-2", "3-5", "6-10", "10+"]


def cargar_datos():
    """Carga datos existentes."""
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"encuestas": [], "metadata": {"creado": datetime.now().isoformat()}}


def guardar_datos(data):
    """Guarda datos."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["metadata"]["actualizado"] = datetime.now().isoformat()
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def seleccionar_opcion(opciones: list, prompt: str, permitir_otro: bool = False) -> str:
    """Muestra opciones numeradas y retorna seleccion."""
    print(f"\n{prompt}")
    for i, op in enumerate(opciones, 1):
        print(f"  {i}. {op}")
    if permitir_otro:
        print(f"  {len(opciones)+1}. Otro (especificar)")

    while True:
        try:
            sel = input("Selecciona numero: ").strip()
            if not sel:
                return opciones[0]  # Default
            idx = int(sel) - 1
            if 0 <= idx < len(opciones):
                return opciones[idx]
            if permitir_otro and idx == len(opciones):
                return input("Especifica: ").strip().lower()
            print("Opcion invalida")
        except ValueError:
            print("Ingresa un numero")


def seleccionar_multiple(opciones: list, prompt: str) -> list:
    """Permite seleccionar multiples opciones."""
    print(f"\n{prompt} (separar con comas, ej: 1,3,4)")
    for i, op in enumerate(opciones, 1):
        print(f"  {i}. {op}")

    while True:
        try:
            sel = input("Selecciona: ").strip()
            if not sel:
                return []
            indices = [int(x.strip()) - 1 for x in sel.split(",")]
            return [opciones[i] for i in indices if 0 <= i < len(opciones)]
        except ValueError:
            print("Formato invalido. Usa numeros separados por comas")


def nueva_encuesta():
    """Registra una nueva encuesta interactivamente."""
    print("\n" + "="*60)
    print("NUEVA ENCUESTA DE PESCA")
    print("="*60)

    encuesta = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Zona
    zonas_lista = list(ZONAS.keys())
    zonas_nombres = [ZONAS[z]["nombre"] for z in zonas_lista]
    zona_nombre = seleccionar_opcion(zonas_nombres, "1. ZONA DE PESCA:", permitir_otro=True)

    # Buscar zona o crear nueva
    zona_key = None
    for k, v in ZONAS.items():
        if v["nombre"] == zona_nombre:
            zona_key = k
            encuesta["lat"] = v["lat"]
            encuesta["lon"] = v["lon"]
            break

    if not zona_key:
        zona_key = zona_nombre.lower().replace(" ", "_")
        encuesta["lat"] = None  # Se puede agregar despues
        encuesta["lon"] = None

    encuesta["zona"] = zona_key
    encuesta["zona_nombre"] = zona_nombre

    # 2. Fecha
    print("\n2. FECHA DE PESCA:")
    fecha_str = input("   Fecha (DD/MM/YYYY o 'hoy' o 'ayer'): ").strip().lower()
    if fecha_str == "hoy" or not fecha_str:
        encuesta["fecha"] = date.today().isoformat()
    elif fecha_str == "ayer":
        from datetime import timedelta
        encuesta["fecha"] = (date.today() - timedelta(days=1)).isoformat()
    else:
        try:
            d = datetime.strptime(fecha_str, "%d/%m/%Y")
            encuesta["fecha"] = d.date().isoformat()
        except:
            encuesta["fecha"] = date.today().isoformat()
            print(f"   Fecha invalida, usando hoy: {encuesta['fecha']}")

    # 3. Hora
    encuesta["hora"] = seleccionar_opcion(HORAS, "3. HORA DE PESCA:")

    # 4. Duracion
    duracion = input("\n4. DURACION (horas, ej: 3): ").strip()
    encuesta["duracion_horas"] = float(duracion) if duracion else 3.0

    # 5. Especies
    encuesta["especies"] = seleccionar_multiple(ESPECIES, "5. ESPECIES CAPTURADAS:")

    # 6. Cantidad
    encuesta["cantidad"] = seleccionar_opcion(CANTIDADES, "6. CANTIDAD TOTAL:")
    encuesta["exito"] = encuesta["cantidad"] != "0"

    # 7. Metodo
    encuesta["metodo"] = seleccionar_opcion(METODOS, "7. METODO DE PESCA:", permitir_otro=True)

    # 8. Condiciones
    encuesta["mar"] = seleccionar_opcion(MAR, "8. ESTADO DEL MAR:")
    encuesta["temp_agua"] = seleccionar_opcion(TEMP_AGUA, "9. TEMPERATURA DEL AGUA:")

    # 10. Notas opcionales
    notas = input("\n10. NOTAS (opcional, Enter para omitir): ").strip()
    if notas:
        encuesta["notas"] = notas

    # Confirmar
    print("\n" + "-"*60)
    print("RESUMEN:")
    print(f"  Zona: {encuesta['zona_nombre']}")
    print(f"  Fecha: {encuesta['fecha']} - {encuesta['hora']}")
    print(f"  Especies: {', '.join(encuesta['especies']) if encuesta['especies'] else 'Ninguna'}")
    print(f"  Cantidad: {encuesta['cantidad']}")
    print(f"  Metodo: {encuesta['metodo']}")
    print(f"  Mar: {encuesta['mar']}, Agua: {encuesta['temp_agua']}")
    print("-"*60)

    confirmar = input("¿Guardar? (S/n): ").strip().lower()
    if confirmar != 'n':
        data = cargar_datos()
        data["encuestas"].append(encuesta)
        guardar_datos(data)
        print(f"\n✓ Encuesta #{len(data['encuestas'])} guardada!")
        return encuesta
    else:
        print("Cancelado.")
        return None


def ver_encuestas():
    """Muestra todas las encuestas."""
    data = cargar_datos()
    encuestas = data.get("encuestas", [])

    if not encuestas:
        print("No hay encuestas registradas.")
        return

    print(f"\n{'='*80}")
    print(f"ENCUESTAS REGISTRADAS: {len(encuestas)}")
    print(f"{'='*80}")

    for i, e in enumerate(encuestas, 1):
        exito = "✓" if e.get("exito") else "✗"
        especies = ", ".join(e.get("especies", [])) or "nada"
        print(f"{i:3}. [{exito}] {e.get('fecha', '?')} | {e.get('zona_nombre', '?'):20} | "
              f"{especies:20} | {e.get('cantidad', '?'):5} | {e.get('metodo', '?')}")


def ver_estadisticas():
    """Muestra estadisticas de las encuestas."""
    data = cargar_datos()
    encuestas = data.get("encuestas", [])

    if not encuestas:
        print("No hay encuestas registradas.")
        return

    print(f"\n{'='*60}")
    print("ESTADISTICAS")
    print(f"{'='*60}")

    total = len(encuestas)
    exitos = sum(1 for e in encuestas if e.get("exito"))

    print(f"\nTotal encuestas: {total}")
    print(f"Con captura: {exitos} ({100*exitos/total:.1f}%)")
    print(f"Sin captura: {total - exitos} ({100*(total-exitos)/total:.1f}%)")

    # Por zona
    print("\n--- Por Zona ---")
    zonas = {}
    for e in encuestas:
        z = e.get("zona_nombre", "?")
        if z not in zonas:
            zonas[z] = {"total": 0, "exitos": 0}
        zonas[z]["total"] += 1
        if e.get("exito"):
            zonas[z]["exitos"] += 1

    for z, stats in sorted(zonas.items(), key=lambda x: x[1]["total"], reverse=True):
        tasa = 100 * stats["exitos"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {z}: {stats['total']} encuestas, {tasa:.0f}% exito")

    # Por especie
    print("\n--- Por Especie ---")
    especies = {}
    for e in encuestas:
        for sp in e.get("especies", []):
            if sp != "nada":
                especies[sp] = especies.get(sp, 0) + 1

    for sp, count in sorted(especies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sp}: {count} capturas")

    # Por condiciones
    print("\n--- Por Estado del Mar ---")
    mar_stats = {}
    for e in encuestas:
        m = e.get("mar", "?")
        if m not in mar_stats:
            mar_stats[m] = {"total": 0, "exitos": 0}
        mar_stats[m]["total"] += 1
        if e.get("exito"):
            mar_stats[m]["exitos"] += 1

    for m, stats in mar_stats.items():
        tasa = 100 * stats["exitos"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {m}: {stats['total']} dias, {tasa:.0f}% exito")


def exportar_csv():
    """Exporta a CSV para analisis."""
    data = cargar_datos()
    encuestas = data.get("encuestas", [])

    if not encuestas:
        print("No hay encuestas para exportar.")
        return

    csv_path = DATA_FILE.with_suffix('.csv')

    # Campos
    campos = ["id", "fecha", "zona", "lat", "lon", "hora", "duracion_horas",
              "especies", "cantidad", "exito", "metodo", "mar", "temp_agua", "notas"]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
        writer.writeheader()
        for e in encuestas:
            row = e.copy()
            row["especies"] = "|".join(e.get("especies", []))
            writer.writerow(row)

    print(f"Exportado a: {csv_path}")
    print(f"Total registros: {len(encuestas)}")


def registro_rapido():
    """Registro ultra-rapido desde linea de comandos."""
    print("\nREGISTRO RAPIDO (formato: zona fecha hora especie cantidad metodo mar)")
    print("Ejemplo: coles 15/02 manana cabrilla 3 spinning calmo")

    linea = input("\n> ").strip().lower().split()

    if len(linea) < 7:
        print("Faltan datos. Usa el formato completo.")
        return

    zona_input, fecha, hora, especie, cantidad, metodo, mar = linea[:7]

    # Buscar zona
    zona_key = None
    for k in ZONAS:
        if zona_input in k:
            zona_key = k
            break

    if not zona_key:
        zona_key = zona_input

    encuesta = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "zona": zona_key,
        "zona_nombre": ZONAS.get(zona_key, {}).get("nombre", zona_key),
        "lat": ZONAS.get(zona_key, {}).get("lat"),
        "lon": ZONAS.get(zona_key, {}).get("lon"),
        "fecha": f"2026-{fecha.replace('/', '-')}" if '/' in fecha else fecha,
        "hora": hora,
        "especies": [especie] if especie != "nada" else [],
        "cantidad": cantidad,
        "exito": especie != "nada" and cantidad != "0",
        "metodo": metodo,
        "mar": mar,
    }

    data = cargar_datos()
    data["encuestas"].append(encuesta)
    guardar_datos(data)
    print(f"✓ Guardado: {encuesta['zona_nombre']} | {especie} x{cantidad}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--ver", "-v", "ver"]:
            ver_encuestas()
        elif arg in ["--stats", "-s", "stats"]:
            ver_estadisticas()
        elif arg in ["--exportar", "-e", "csv"]:
            exportar_csv()
        elif arg in ["--rapido", "-r", "rapido"]:
            registro_rapido()
        elif arg in ["--help", "-h"]:
            print(__doc__)
        else:
            print(f"Opcion desconocida: {arg}")
            print(__doc__)
    else:
        nueva_encuesta()
