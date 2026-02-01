"""
Gestor de datos para el sistema de predicción de pesca.
Maneja descarga incremental, cache y actualización de datos.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib


class DataManager:
    """
    Gestiona la descarga y almacenamiento de datos oceanográficos.
    Soporta actualizaciones incrementales.
    """

    # Configuración ERDDAP - Datos reales
    ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap"

    # Datasets disponibles
    DATASETS = {
        "sst": {
            "id": "jplMURSST41",  # MUR SST - 1km resolución
            "variable": "analysed_sst",
            "description": "Sea Surface Temperature (MUR)"
        },
        "chlorophyll": {
            "id": "erdMH1chla8day",  # MODIS Chlorophyll 8-day
            "variable": "chlorophyll",
            "description": "Chlorophyll-a concentration"
        }
    }

    # BBOX para costa sur de Perú (Tacna - Ilo - Sama - Canepa)
    # Expandido para incluir Playa Canepa y zona de Sama
    DEFAULT_BBOX = {
        "north": -17.50,
        "south": -18.35,  # Más al sur
        "west": -71.45,
        "east": -70.10    # Más al este para incluir Playa Canepa (-70.25)
    }

    def __init__(self, cache_dir: str = None):
        """
        Inicializa el gestor de datos.

        Args:
            cache_dir: directorio para almacenar datos en cache
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Archivo de metadata para tracking de descargas
        self.metadata_file = self.cache_dir / "download_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Carga metadata de descargas previas."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "downloads": [],
            "last_update": None,
            "coastline_version": None
        }

    def _save_metadata(self):
        """Guarda metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    # =========================================================================
    # COASTLINE DATA - OpenStreetMap
    # =========================================================================

    def download_coastline(self, force: bool = False) -> Path:
        """
        Descarga datos de línea costera de OpenStreetMap.

        Args:
            force: forzar descarga aunque ya exista

        Returns:
            Path al archivo de coastline
        """
        coastline_file = self.cache_dir / "coastline_peru.geojson"

        # Verificar si ya existe y no forzamos
        if coastline_file.exists() and not force:
            print(f"[INFO] Coastline ya existe: {coastline_file}")
            return coastline_file

        print("[INFO] Descargando línea costera de Overpass API...")

        # Query Overpass para obtener coastline de la región
        overpass_url = "https://overpass-api.de/api/interpreter"

        # Query para obtener la línea costera en nuestra área
        query = f"""
        [out:json][timeout:120];
        (
          way["natural"="coastline"]
            ({self.DEFAULT_BBOX['south']},{self.DEFAULT_BBOX['west']},
             {self.DEFAULT_BBOX['north']},{self.DEFAULT_BBOX['east']});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = requests.post(overpass_url, data={"data": query}, timeout=180)
            response.raise_for_status()
            data = response.json()

            # Convertir a GeoJSON
            geojson = self._osm_to_geojson(data)

            # Guardar
            with open(coastline_file, 'w') as f:
                json.dump(geojson, f)

            self.metadata["coastline_version"] = datetime.now().isoformat()
            self._save_metadata()

            print(f"[OK] Coastline guardado: {coastline_file}")
            return coastline_file

        except Exception as e:
            print(f"[ERROR] Error descargando coastline: {e}")
            # Crear coastline manual como fallback
            return self._create_fallback_coastline(coastline_file)

    def _osm_to_geojson(self, osm_data: Dict) -> Dict:
        """Convierte datos OSM a GeoJSON."""
        nodes = {}
        ways = []

        for element in osm_data.get("elements", []):
            if element["type"] == "node":
                nodes[element["id"]] = (element["lon"], element["lat"])
            elif element["type"] == "way":
                ways.append(element)

        features = []
        for way in ways:
            coords = []
            for node_id in way.get("nodes", []):
                if node_id in nodes:
                    coords.append(list(nodes[node_id]))

            if len(coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "osm_id": way["id"],
                        "natural": "coastline"
                    }
                })

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def _create_fallback_coastline(self, output_file: Path) -> Path:
        """Crea coastline manual como respaldo."""
        print("[INFO] Creando coastline de respaldo...")

        # Puntos conocidos de la costa (de NW a SE)
        coast_points = [
            [-71.3500, -17.6270],  # Ilo Puerto
            [-71.3450, -17.6320],
            [-71.3400, -17.6420],  # Pozo Lizas
            [-71.3320, -17.7020],  # Punta Coles
            [-71.2220, -17.7320],  # Media Luna
            [-71.1720, -17.7570],  # Fundicion
            [-71.1220, -17.7820],  # Pozo Redondo
            [-71.0820, -17.8120],  # Punta Blanca
            [-71.0480, -17.8420],  # Gentillar
            [-71.0180, -17.8720],  # Ite Sur
            [-70.9920, -17.9020],  # Ite Centro
            [-70.9680, -17.9320],  # Ite Norte
            [-70.9480, -17.9620],  # Carlepe
            [-70.9350, -17.9880],  # Punta Mesa
            [-70.9120, -18.0180],  # Vila Vila
            [-70.8830, -18.0520],  # Los Palos
            [-70.8680, -18.0870],  # Santa Rosa
            [-70.8430, -18.1205],  # Boca del Rio
            [-70.5800, -18.2140],  # Zona sur
        ]

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coast_points
                },
                "properties": {
                    "name": "Costa Tacna-Ilo",
                    "source": "manual"
                }
            }]
        }

        with open(output_file, 'w') as f:
            json.dump(geojson, f)

        return output_file

    def get_coastline_points(self) -> List[Tuple[float, float]]:
        """
        Obtiene puntos de la línea costera como lista de (lat, lon).
        """
        coastline_file = self.download_coastline()

        with open(coastline_file, 'r') as f:
            geojson = json.load(f)

        points = []
        for feature in geojson.get("features", []):
            coords = feature.get("geometry", {}).get("coordinates", [])
            for lon, lat in coords:
                points.append((lat, lon))

        return points

    # =========================================================================
    # OCEANOGRAPHIC DATA - ERDDAP
    # =========================================================================

    def fetch_erddap_data(
        self,
        dataset: str,
        start_date: datetime,
        end_date: datetime,
        bbox: Dict = None
    ) -> Optional[pd.DataFrame]:
        """
        Descarga datos de ERDDAP.

        Args:
            dataset: 'sst' o 'chlorophyll'
            start_date: fecha inicial
            end_date: fecha final
            bbox: bounding box (usa DEFAULT_BBOX si no se especifica)

        Returns:
            DataFrame con los datos
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Dataset desconocido: {dataset}")

        bbox = bbox or self.DEFAULT_BBOX
        ds_info = self.DATASETS[dataset]

        # Formatear fechas
        start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")

        # Construir URL ERDDAP
        url = (
            f"{self.ERDDAP_BASE}/griddap/{ds_info['id']}.csv?"
            f"{ds_info['variable']}"
            f"[({start_str}):1:({end_str})]"
            f"[({bbox['south']}):1:({bbox['north']})]"
            f"[({bbox['west']}):1:({bbox['east']})]"
        )

        print(f"[INFO] Descargando {dataset} desde ERDDAP...")
        print(f"       Periodo: {start_date.date()} a {end_date.date()}")

        try:
            response = requests.get(url, timeout=300)

            if response.status_code == 200:
                # Parsear CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text), skiprows=[1])

                # Renombrar columnas
                df = df.rename(columns={
                    "latitude": "lat",
                    "longitude": "lon",
                    ds_info['variable']: dataset
                })

                # Convertir SST de Kelvin a Celsius si es necesario
                if dataset == "sst" and df[dataset].mean() > 100:
                    df[dataset] = df[dataset] - 273.15

                print(f"[OK] Descargados {len(df)} registros de {dataset}")
                return df
            else:
                print(f"[ERROR] ERDDAP respondió con código {response.status_code}")
                return None

        except Exception as e:
            print(f"[ERROR] Error descargando {dataset}: {e}")
            return None

    def update_oceanographic_data(
        self,
        months_back: int = 4,
        force_full: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Actualiza datos oceanográficos.
        Solo descarga datos nuevos si ya tenemos datos previos.

        Args:
            months_back: meses hacia atrás para descargar
            force_full: forzar descarga completa

        Returns:
            Dict con DataFrames de SST y chlorophyll
        """
        end_date = datetime.now() - timedelta(days=1)  # Ayer

        results = {}

        for dataset in ["sst", "chlorophyll"]:
            cache_file = self.cache_dir / f"{dataset}_data.parquet"

            # Verificar datos existentes
            existing_df = None
            last_date = None

            if cache_file.exists() and not force_full:
                existing_df = pd.read_parquet(cache_file)
                if 'time' in existing_df.columns:
                    existing_df['time'] = pd.to_datetime(existing_df['time'])
                    last_date = existing_df['time'].max()
                    print(f"[INFO] {dataset}: datos existentes hasta {last_date.date()}")

            # Determinar rango de descarga
            if last_date and not force_full:
                # Solo descargar datos nuevos
                start_date = last_date + timedelta(days=1)
                if start_date >= end_date:
                    print(f"[INFO] {dataset}: datos ya actualizados")
                    results[dataset] = existing_df
                    continue
            else:
                # Descarga completa
                start_date = end_date - timedelta(days=months_back * 30)

            # Descargar
            new_df = self.fetch_erddap_data(dataset, start_date, end_date)

            if new_df is not None:
                # Combinar con datos existentes
                if existing_df is not None and not force_full:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(
                        subset=['time', 'lat', 'lon'],
                        keep='last'
                    )
                else:
                    combined_df = new_df

                # Guardar
                combined_df.to_parquet(cache_file)
                results[dataset] = combined_df

                # Actualizar metadata
                download_info = {
                    "dataset": dataset,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "records": len(new_df),
                    "timestamp": datetime.now().isoformat()
                }
                self.metadata["downloads"].append(download_info)
                self.metadata["last_update"] = datetime.now().isoformat()
                self._save_metadata()
            else:
                results[dataset] = existing_df

        return results

    def get_latest_data(
        self,
        dataset: str,
        days_back: int = 8
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene los datos más recientes para un dataset.

        Args:
            dataset: 'sst' o 'chlorophyll'
            days_back: días hacia atrás para promediar

        Returns:
            DataFrame con datos promediados
        """
        cache_file = self.cache_dir / f"{dataset}_data.parquet"

        if not cache_file.exists():
            print(f"[WARN] No hay datos cacheados de {dataset}")
            return None

        df = pd.read_parquet(cache_file)

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            cutoff = df['time'].max() - timedelta(days=days_back)
            df = df[df['time'] >= cutoff]

        # Promediar por ubicación
        if len(df) > 0:
            avg_df = df.groupby(['lat', 'lon'])[dataset].mean().reset_index()
            return avg_df

        return None

    def get_data_summary(self) -> Dict:
        """Obtiene resumen del estado de los datos."""
        summary = {
            "cache_dir": str(self.cache_dir),
            "last_update": self.metadata.get("last_update"),
            "coastline_version": self.metadata.get("coastline_version"),
            "datasets": {}
        }

        for dataset in ["sst", "chlorophyll"]:
            cache_file = self.cache_dir / f"{dataset}_data.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    summary["datasets"][dataset] = {
                        "records": len(df),
                        "date_min": df['time'].min().isoformat() if len(df) > 0 else None,
                        "date_max": df['time'].max().isoformat() if len(df) > 0 else None,
                        "file_size_mb": cache_file.stat().st_size / (1024*1024)
                    }
            else:
                summary["datasets"][dataset] = {"status": "not_downloaded"}

        return summary


# Función de conveniencia
def get_data_manager() -> DataManager:
    """Obtiene instancia del gestor de datos."""
    return DataManager()
