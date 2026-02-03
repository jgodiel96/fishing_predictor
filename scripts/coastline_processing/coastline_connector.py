#!/usr/bin/env python3
"""
Coastline Connector - Robust Point Ordering Algorithm

Uses scipy and numpy for efficient, scientifically-sound coastline reconstruction
from unordered points.

Algorithm:
1. Build a k-nearest neighbors graph
2. Use minimum spanning tree to find optimal connections
3. Extract the longest path (the coastline)
4. Split into segments at natural breaks
5. Order segments geographically

Author: Fishing Predictor Project
Date: 2026-02-02
Version: 5.2
"""

import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple, FrozenSet
from dataclasses import dataclass
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Scientific computing imports
try:
    from scipy.spatial import KDTree
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not available. Install with: pip install scipy")


class CoastPoint(NamedTuple):
    """Immutable point on the coastline."""
    lat: float
    lon: float


@dataclass(frozen=True)
class ConnectorConfig:
    """Immutable configuration for coastline connector."""
    # Earth radius for distance calculations
    earth_radius_m: float = 6371000.0

    # Graph construction
    k_neighbors: int = 8  # Number of neighbors to consider

    # Segment detection
    max_edge_length_m: float = 500.0  # Maximum edge length in MST
    min_segment_points: int = 5  # Minimum points to form a segment

    # Filtering
    remove_outliers: bool = True
    outlier_std_threshold: float = 3.0  # Points > 3 std from mean distance

    # Geographic context (Peru south coast)
    coast_direction: str = "north-south"  # Main axis of the coast
    ocean_side: str = "west"  # Ocean is to the west


class CoastlineConnector:
    """
    Robust coastline point connector using graph algorithms.

    This implementation uses:
    - scipy.spatial.KDTree for efficient neighbor search
    - scipy.sparse.csgraph for minimum spanning tree
    - numpy for vectorized distance calculations

    The algorithm reconstructs a coastline from unordered points by:
    1. Building a k-NN graph with geographic distances
    2. Computing the minimum spanning tree (MST)
    3. Extracting the longest path through the MST
    4. Splitting at natural gaps to form segments
    """

    def __init__(self, config: Optional[ConnectorConfig] = None):
        """Initialize with configuration."""
        self.config = config or ConnectorConfig()

        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for CoastlineConnector")

    def haversine_distance_vectorized(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized haversine distance calculation.

        Args:
            points1: Array of shape (n, 2) with [lat, lon]
            points2: Array of shape (m, 2) with [lat, lon]

        Returns:
            Distance matrix of shape (n, m) in meters
        """
        lat1 = np.radians(points1[:, 0:1])
        lon1 = np.radians(points1[:, 1:2])
        lat2 = np.radians(points2[:, 0:1].T)
        lon2 = np.radians(points2[:, 1:2].T)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return self.config.earth_radius_m * c

    def haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate haversine distance between two points in meters."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.config.earth_radius_m * c

    def _points_to_array(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Convert list of (lat, lon) tuples to numpy array."""
        return np.array(points, dtype=np.float64)

    def _array_to_points(self, array: np.ndarray) -> List[CoastPoint]:
        """Convert numpy array to list of CoastPoint."""
        return [CoastPoint(lat=float(row[0]), lon=float(row[1])) for row in array]

    def remove_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Remove outlier points that are far from the main cluster.

        Uses statistical analysis of distances to nearest neighbors.
        """
        if len(points) < 10 or not self.config.remove_outliers:
            return points

        # Build KDTree with scaled coordinates
        avg_lat = np.mean(points[:, 0])
        scale = np.array([111000, 111000 * np.cos(np.radians(avg_lat))])
        scaled = points * scale

        tree = KDTree(scaled)

        # Get distance to 3rd nearest neighbor (skip self and immediate neighbor)
        distances, _ = tree.query(scaled, k=4)
        dist_to_3rd = distances[:, 3]

        # Remove points with unusually large distances (outliers)
        mean_dist = np.mean(dist_to_3rd)
        std_dist = np.std(dist_to_3rd)
        threshold = mean_dist + self.config.outlier_std_threshold * std_dist

        mask = dist_to_3rd <= threshold
        removed = np.sum(~mask)

        if removed > 0:
            print(f"[INFO] Removed {removed} outlier points")

        return points[mask]

    def build_distance_graph(self, points: np.ndarray) -> csr_matrix:
        """
        Build a sparse distance graph using k-nearest neighbors.

        Returns a sparse matrix where entry (i,j) is the distance
        between points i and j if they are neighbors, 0 otherwise.
        """
        n = len(points)
        k = min(self.config.k_neighbors, n - 1)

        # Scale coordinates for KDTree (approximate meters)
        avg_lat = np.mean(points[:, 0])
        scale = np.array([111000, 111000 * np.cos(np.radians(avg_lat))])
        scaled = points * scale

        # Find k nearest neighbors
        tree = KDTree(scaled)
        distances, indices = tree.query(scaled, k=k+1)  # +1 because includes self

        # Build sparse matrix
        row_indices = []
        col_indices = []
        dist_values = []

        for i in range(n):
            for j_idx in range(1, k+1):  # Skip self (index 0)
                j = indices[i, j_idx]
                dist = distances[i, j_idx]

                # Add both directions for undirected graph
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                dist_values.extend([dist, dist])

        graph = csr_matrix(
            (dist_values, (row_indices, col_indices)),
            shape=(n, n)
        )

        return graph

    def compute_mst(self, graph: csr_matrix) -> csr_matrix:
        """
        Compute the Minimum Spanning Tree of the distance graph.

        The MST connects all points with minimum total distance,
        which naturally follows the coastline shape.
        """
        mst = minimum_spanning_tree(graph)

        # Make symmetric (MST is returned as lower triangular)
        mst = mst + mst.T

        return mst

    def extract_paths_from_mst(
        self,
        mst: csr_matrix,
        points: np.ndarray
    ) -> List[List[int]]:
        """
        Extract paths from MST, handling multiple connected components.

        Returns a list of paths, one per component.
        """
        n = mst.shape[0]

        # Find connected components
        n_components, labels = connected_components(mst, directed=False)
        print(f"[INFO] Found {n_components} connected components")

        # Convert to adjacency list
        adj = [[] for _ in range(n)]
        rows, cols = mst.nonzero()

        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates
                weight = mst[i, j]
                adj[i].append((j, weight))
                adj[j].append((i, weight))

        # Extract path for each component
        paths = []

        for comp_id in range(n_components):
            # Get nodes in this component
            comp_nodes = np.where(labels == comp_id)[0]

            if len(comp_nodes) < self.config.min_segment_points:
                continue

            # Find endpoints (degree 1) in this component
            comp_degrees = [len(adj[i]) for i in comp_nodes]
            endpoints = [comp_nodes[i] for i, d in enumerate(comp_degrees) if d == 1]

            if len(endpoints) >= 2:
                # BFS from first endpoint
                start = endpoints[0]
            else:
                # Use any node from the component
                start = comp_nodes[0]

            _, path = self._bfs_farthest(adj, start)

            if path and len(path) >= self.config.min_segment_points:
                paths.append(path)

        # Sort paths by starting latitude (south to north)
        paths.sort(key=lambda p: points[p[0], 0] if p else float('inf'))

        return paths

    def extract_path_from_mst(self, mst: csr_matrix) -> List[int]:
        """
        Legacy method - extract single longest path.
        Use extract_paths_from_mst for proper handling.
        """
        n = mst.shape[0]

        # Convert to adjacency list
        adj = [[] for _ in range(n)]
        rows, cols = mst.nonzero()

        for i, j in zip(rows, cols):
            if i < j:
                weight = mst[i, j]
                adj[i].append((j, weight))
                adj[j].append((i, weight))

        degrees = [len(adj[i]) for i in range(n)]
        endpoints = [i for i in range(n) if degrees[i] == 1]

        if len(endpoints) < 2:
            start = 0
            _, path = self._bfs_farthest(adj, start)
            if path:
                _, path = self._bfs_farthest(adj, path[-1])
            return path if path else list(range(n))

        start = endpoints[0]
        _, path = self._bfs_farthest(adj, start)

        return path

    def _bfs_farthest(
        self,
        adj: List[List[Tuple[int, float]]],
        start: int
    ) -> Tuple[float, List[int]]:
        """
        BFS to find the farthest node from start and the path to it.
        """
        from collections import deque

        n = len(adj)
        visited = [False] * n
        parent = [-1] * n
        dist = [0.0] * n

        queue = deque([start])
        visited[start] = True
        farthest = start
        max_dist = 0.0

        while queue:
            node = queue.popleft()

            for neighbor, weight in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    dist[neighbor] = dist[node] + weight
                    queue.append(neighbor)

                    if dist[neighbor] > max_dist:
                        max_dist = dist[neighbor]
                        farthest = neighbor

        # Reconstruct path
        path = []
        node = farthest
        while node != -1:
            path.append(node)
            node = parent[node]
        path.reverse()

        return max_dist, path

    def split_into_segments(
        self,
        points: np.ndarray,
        path: List[int]
    ) -> List[np.ndarray]:
        """
        Split the path into segments at natural breaks (large gaps).
        """
        if len(path) < 2:
            return [points[path]] if path else []

        max_edge = self.config.max_edge_length_m
        min_points = self.config.min_segment_points

        segments = []
        current_segment = [path[0]]

        for i in range(1, len(path)):
            prev_idx = path[i - 1]
            curr_idx = path[i]

            dist = self.haversine_distance(
                points[prev_idx, 0], points[prev_idx, 1],
                points[curr_idx, 0], points[curr_idx, 1]
            )

            if dist > max_edge:
                # Gap detected - save current segment if large enough
                if len(current_segment) >= min_points:
                    segments.append(points[current_segment])
                current_segment = [curr_idx]
            else:
                current_segment.append(curr_idx)

        # Add final segment
        if len(current_segment) >= min_points:
            segments.append(points[current_segment])

        return segments

    def order_segment_south_to_north(self, segment: np.ndarray) -> np.ndarray:
        """
        Order segment from south to north (increasing latitude).
        """
        if len(segment) < 2:
            return segment

        if segment[0, 0] > segment[-1, 0]:
            return segment[::-1]

        return segment

    def connect(
        self,
        points: List[Tuple[float, float]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Main entry point: connect unordered points into coastline segments.

        Args:
            points: List of (lat, lon) tuples

        Returns:
            List of segments, each a list of ordered (lat, lon) tuples
        """
        if not points:
            return []

        print(f"\n[INFO] Connecting {len(points)} coastline points...")

        # Convert to numpy array
        points_array = self._points_to_array(points)

        # Step 1: Remove outliers
        points_array = self.remove_outliers(points_array)

        if len(points_array) < self.config.min_segment_points:
            print("[WARN] Too few points after outlier removal")
            return []

        # Step 2: Build k-NN distance graph
        print("[INFO] Building distance graph...")
        graph = self.build_distance_graph(points_array)

        # Step 3: Compute minimum spanning tree
        print("[INFO] Computing minimum spanning tree...")
        mst = self.compute_mst(graph)

        # Step 4: Extract paths from MST (handles multiple components)
        print("[INFO] Extracting coastline paths...")
        paths = self.extract_paths_from_mst(mst, points_array)

        if not paths:
            print("[WARN] No valid paths extracted")
            return []

        # Step 5: Convert paths to segments and split at gaps
        print("[INFO] Processing segments...")
        all_segments = []

        for path in paths:
            # Split this path at large gaps
            path_segments = self.split_into_segments(points_array, path)
            all_segments.extend(path_segments)

        # Step 6: Order each segment south to north
        all_segments = [self.order_segment_south_to_north(seg) for seg in all_segments]

        # Sort segments by starting latitude
        all_segments.sort(key=lambda seg: seg[0, 0])

        # Convert back to list of tuples
        result = [[(float(p[0]), float(p[1])) for p in seg] for seg in all_segments]

        total_points = sum(len(seg) for seg in result)
        print(f"[OK] Connected into {len(result)} segments with {total_points} total points")

        return result

    def get_statistics(
        self,
        segments: List[List[Tuple[float, float]]]
    ) -> dict:
        """Calculate statistics about the connected segments."""
        if not segments:
            return {
                "num_segments": 0,
                "total_points": 0,
                "total_length_km": 0,
            }

        total_points = sum(len(seg) for seg in segments)

        # Calculate total length
        total_length = 0.0
        segment_lengths = []

        for seg in segments:
            length = 0.0
            for i in range(1, len(seg)):
                length += self.haversine_distance(
                    seg[i-1][0], seg[i-1][1],
                    seg[i][0], seg[i][1]
                )
            segment_lengths.append(length)
            total_length += length

        return {
            "num_segments": len(segments),
            "total_points": total_points,
            "total_length_km": total_length / 1000,
            "segment_lengths_km": [l / 1000 for l in segment_lengths],
            "points_per_segment": [len(seg) for seg in segments],
        }


def save_segments_as_geojson(
    segments: List[List[Tuple[float, float]]],
    output_path: str,
    metadata: Optional[dict] = None
) -> str:
    """
    Save connected segments as GeoJSON MultiLineString.
    """
    import json

    # Convert to GeoJSON coordinates (lon, lat)
    coordinates = []
    for seg in segments:
        seg_coords = [[lon, lat] for lat, lon in seg]
        coordinates.append(seg_coords)

    total_points = sum(len(seg) for seg in segments)

    properties = {
        "name": "Connected Coastline",
        "source": "coastline_connector_v5.2",
        "algorithm": "minimum_spanning_tree",
        "num_segments": len(segments),
        "total_points": total_points,
    }
    if metadata:
        properties.update(metadata)

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": coordinates
            },
            "properties": properties
        }]
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"[OK] Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Coastline point connector v5.2")
    parser.add_argument("input", help="Input GeoJSON file")
    parser.add_argument("-o", "--output", help="Output GeoJSON file")
    parser.add_argument("--max-edge", type=float, default=500,
                       help="Maximum edge length in meters")
    parser.add_argument("--k-neighbors", type=int, default=8,
                       help="Number of neighbors for graph")

    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        data = json.load(f)

    # Extract points
    points = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            for lon, lat in geom.get("coordinates", []):
                points.append((lat, lon))
        elif geom.get("type") == "MultiLineString":
            for line in geom.get("coordinates", []):
                for lon, lat in line:
                    points.append((lat, lon))

    print(f"Loaded {len(points)} points")

    # Connect
    config = ConnectorConfig(
        max_edge_length_m=args.max_edge,
        k_neighbors=args.k_neighbors
    )
    connector = CoastlineConnector(config)
    segments = connector.connect(points)

    # Statistics
    stats = connector.get_statistics(segments)
    print(f"\nStatistics:")
    print(f"  Segments: {stats['num_segments']}")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Total length: {stats['total_length_km']:.2f} km")

    # Save
    if args.output:
        save_segments_as_geojson(segments, args.output)
