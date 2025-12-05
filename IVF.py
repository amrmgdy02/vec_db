from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
import os

class TwoLevelIVF:
    """
    Two-level hierarchical IVF.

    Top-level clusters: K1
    Second-level clusters per top: K2

    This implementation trains a top-level MiniBatchKMeans and then, for each
    top cluster, trains a MiniBatchKMeans on the vectors assigned to that top
    cluster to produce K2 second-level centroids. Inverted lists are built for
    the flattened (top*K2 + sub) cluster ids.
    """
    def __init__(self, K1: int, K2: int, seed: int = 42) -> None:
        self.K1 = K1
        self.K2 = K2
        self.seed = seed

        self.centroids1: np.ndarray = None            # (K1, D)
        self.centroids2: List[np.ndarray] = []        # list of (k2_i, D) arrays (k2_i <= K2)

    def fit(self, vectors: np.ndarray, batch_size: int) -> None:
        """Train the two-level index from `vectors`.

        This will train top-level centroids and then per-top-cluster KMeans for
        second-level centroids (on the points assigned to that top cluster).
        """
        num_vectors = vectors.shape[0]

        # Train top-level kmeans
        kmeans1 = MiniBatchKMeans(n_clusters=self.K1, batch_size=batch_size, random_state=self.seed, max_iter=100)
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            batch = vectors[start:end].astype(np.float32)
            kmeans1.partial_fit(batch)
        self.centroids1 = kmeans1.cluster_centers_.astype(np.float32)

        # Assign every vector to its top cluster
        top_assignments = np.empty(num_vectors, dtype=np.int32)
        centroid_norms = np.sum(self.centroids1 * self.centroids1, axis=1)
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            batch = vectors[start:end].astype(np.float32)
            batch_norms = np.sum(batch * batch, axis=1)
            dots = batch @ self.centroids1.T
            dists = batch_norms[:, None] + centroid_norms[None, :] - 2.0 * dots
            top_assignments[start:end] = np.argmin(dists, axis=1)

        # For each top cluster, collect its points and run KMeans to produce K2 centroids
        self.centroids2 = [None] * self.K1

        for c in range(self.K1):
            idxs = np.nonzero(top_assignments == c)[0]
            if idxs.size == 0:
                # empty top cluster
                self.centroids2[c] = np.zeros((0, vectors.shape[1]), dtype=np.float32)
                continue

            points = vectors[idxs].astype(np.float32)
            # choose k for this cluster (cannot exceed number of points)
            k_here = min(self.K2, points.shape[0])
            kmeans2 = MiniBatchKMeans(n_clusters=k_here, batch_size=min(batch_size, points.shape[0]), random_state=self.seed, max_iter=100)
            # fit on the points for this top cluster
            for start in range(0, points.shape[0], batch_size):
                end = min(start + batch_size, points.shape[0])
                kmeans2.partial_fit(points[start:end])

            centers2 = kmeans2.cluster_centers_.astype(np.float32)
            self.centroids2[c] = centers2

    def assign(self, vectors: np.ndarray, batch_size: int) -> np.ndarray:
        """Assign vectors to flattened second-level cluster ids: id = top * K2 + sub.

        Returns an array of shape (num_vectors,) with values in [0, K1*K2].
        """
        if self.centroids1 is None:
            raise ValueError("TwoLevelIVF is not trained. Call fit() first.")

        num_vectors = vectors.shape[0]
        assignments = np.empty(num_vectors, dtype=np.int32)

        centroid1 = self.centroids1
        centroid1_norms = np.sum(centroid1 * centroid1, axis=1)

        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            batch = vectors[start:end].astype(np.float32)

            batch_norms = np.sum(batch * batch, axis=1)
            dots1 = batch @ centroid1.T
            dists1 = batch_norms[:, None] + centroid1_norms[None, :] - 2.0 * dots1
            top_ids = np.argmin(dists1, axis=1)

            # For points in this batch, group by top_id and compute subcluster assignments
            for top in np.unique(top_ids):
                mask = (top_ids == top)
                points_idx = np.nonzero(mask)[0]
                pts = batch[points_idx]
                centers2 = self.centroids2[top]
                if centers2.size == 0:
                    # all map to empty; assign a special -1 (should be avoided)
                    assignments[start:end][points_idx] = top * self.K2
                    continue
                c2_norms = np.sum(centers2 * centers2, axis=1)
                dots2 = pts @ centers2.T
                bnorms = np.sum(pts * pts, axis=1)
                d2 = bnorms[:, None] + c2_norms[None, :] - 2.0 * dots2
                sub_ids = np.argmin(d2, axis=1)
                assignments[start:end][points_idx] = top * self.K2 + sub_ids

        return assignments

    def search(self, query: np.ndarray,
               nprobe1: int, nprobe2: int, 
               cluster_dir: str = "indexes", 
               centroids_1_path: str = None,
               centroids_2_path: str = None) -> np.ndarray:
        """Search the two-level index and return candidate ids.
        Loads cluster files on-demand from hierarchical directory structure.

        - `nprobe1`: number of top-level clusters to probe
        - `nprobe2`: number of second-level subclusters to probe per top cluster
        - `cluster_dir`: base directory containing cluster_L1_* subdirectories
        """
        centroids1 = np.load(centroids_1_path) if centroids_1_path else self.centroids1
        centroids_2 = pickle.load(open(centroids_2_path, "rb")) if centroids_2_path else self.centroids2
        
        q = query.astype(np.float32)
        qnorm = np.sum(q * q)
        centroid1_norms = np.sum(centroids1 * centroids1, axis=1)
        dots1 = q @ centroids1.T
        d1 = qnorm + centroid1_norms - 2.0 * dots1
        top_ids = np.argsort(d1)[:min(nprobe1, self.K1)]

        all_candidate_ids = []
        
        for top in top_ids:
            centers2 = centroids_2[top]
            if centers2.size == 0:
                continue
            c2_norms = np.sum(centers2 * centers2, axis=1)
            dots2 = q @ centers2.T
            d2 = qnorm + c2_norms - 2.0 * dots2
            sub_ids = np.argsort(d2)[:min(nprobe2, centers2.shape[0])]
            del centers2
            
            # Load cluster files on-demand
            for sub in sub_ids:
                cluster_file = f"{cluster_dir}/cluster_L1_{top}/cluster_L2_{sub}.npy"
                if os.path.exists(cluster_file):
                    cluster_ids = np.load(cluster_file)
                    all_candidate_ids.append(cluster_ids)
            
        return np.concatenate(all_candidate_ids) if all_candidate_ids else np.array([], dtype=np.int32)
