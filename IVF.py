from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle


class InvertedFileIndex:
    """
    Inverted File Index (IVF) for coarse quantization.
    """
    
    def __init__(self, num_clusters: int, seed: int = 42) -> None:
        self.num_clusters = num_clusters
        self.seed = seed
        self.centroids: np.ndarray = None  # Shape: (num_clusters, dimension)
        self.inverted_offsets: np.ndarray = None  # Shape: (num_clusters + 1,)
        self.inverted_ids: np.ndarray = None  # Flat array of all vector IDs
        
    def fit(self, vectors: np.ndarray, batch_size: int) -> None:
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=batch_size,
            random_state=self.seed,
            max_iter=100,
            verbose=0
        )
        
        num_vectors = vectors.shape[0]
        
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            batch = vectors[start:end].astype(np.float32)
            kmeans.partial_fit(batch)
        
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
    
    def assign(self, vectors: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Assign vectors to their nearest cluster
        """
        num_vectors = vectors.shape[0]
        assignments = np.zeros(num_vectors, dtype=np.int32)
        
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            batch = vectors[start:end].astype(np.float32)
            distances = np.sum((batch[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
            assignments[start:end] = np.argmin(distances, axis=1)
        
        return assignments
    
    def build_inverted_lists(self, assignments: np.ndarray) -> None:
        """Build inverted lists using numpy arrays for memory efficiency."""
        # Count vectors per cluster
        cluster_sizes = np.bincount(assignments, minlength=self.num_clusters)
        
        # Pre-allocate flat array for all IDs
        self.inverted_ids = np.zeros(len(assignments), dtype=np.int32)
        self.inverted_offsets = np.zeros(self.num_clusters + 1, dtype=np.int32)
        
        # Compute offsets (cumulative sum)
        self.inverted_offsets[1:] = np.cumsum(cluster_sizes)
        
        # Fill IDs
        current_pos = self.inverted_offsets.copy()
        for vec_id, cluster_id in enumerate(assignments):
            pos = current_pos[cluster_id]
            self.inverted_ids[pos] = vec_id
            current_pos[cluster_id] += 1
        
        # Report statistics
        print(f"Inverted lists built. Min size: {cluster_sizes.min()}, "
            f"Max size: {cluster_sizes.max()}, Avg size: {cluster_sizes.mean():.1f}")
    
    def search_clusters(self, query: np.ndarray, nprobe: int) -> List[int]:
        """
        Find the nprobe nearest clusters to the query vector.
        """
        query = query.astype(np.float32)
        
        # Compute distances to all centroids
        distances = np.sum((self.centroids - query[None, :]) ** 2, axis=1)
        
        # Get nprobe nearest clusters
        nearest_clusters = np.argsort(distances)[:nprobe]
        
        return nearest_clusters.tolist()
    

    def get_candidate_ids(self, cluster_ids: List[int]) -> np.ndarray:
        """Get candidate IDs using offset-based lookup."""
        candidates = []
        for cluster_id in cluster_ids:
            start = self.inverted_offsets[cluster_id]
            end = self.inverted_offsets[cluster_id + 1]
            candidates.append(self.inverted_ids[start:end])
        
        return np.concatenate(candidates) if candidates else np.array([], dtype=np.int32)
        
    def save(self, filepath: str = "ivf_model.pkl") -> None:
        """
        Save the IVF model to disk.
        
        Args:
            filepath (str): Path to save the model.
        """
        state = {
            'num_clusters': self.num_clusters,
            'seed': self.seed,
            'centroids': self.centroids,
            'inverted_offsets': self.inverted_offsets,
            'inverted_ids': self.inverted_ids
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"IVF model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = "ivf_model.pkl"):
        """
        Load an IVF model from disk.
        
        Args:
            filepath (str): Path to load the model from.
            
        Returns:
            InvertedFileIndex: Loaded IVF instance.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(
            num_clusters=state['num_clusters'],
            seed=state['seed']
        )
        instance.centroids = state['centroids']
        instance.inverted_offsets = state['inverted_offsets']
        instance.inverted_ids = state['inverted_ids']
        print(f"IVF model loaded from {filepath}")
        return instance


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

        # flattened inverted lists across all K1 * K2 second-level clusters
        self.inverted_offsets: np.ndarray = None      # shape (K1*K2 + 1,)
        self.inverted_ids: np.ndarray = None          # flat array of vector ids

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

        # Assign every vector to its top cluster (in batches to be memmap-friendly)
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
        second_level_assignments = np.empty(num_vectors, dtype=np.int32)

        current_global_index = 0
        flattened_ids_list = []
        offsets = [0]

        for c in range(self.K1):
            idxs = np.nonzero(top_assignments == c)[0]
            if idxs.size == 0:
                # empty top cluster
                self.centroids2[c] = np.zeros((0, vectors.shape[1]), dtype=np.float32)
                # extend offsets by K2 empty lists
                for _ in range(self.K2):
                    offsets.append(offsets[-1])
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

            # assign points to second-level clusters
            centroid2_norms = np.sum(centers2 * centers2, axis=1)
            dots2 = points @ centers2.T
            batch_norms = np.sum(points * points, axis=1)
            dists2 = batch_norms[:, None] + centroid2_norms[None, :] - 2.0 * dots2
            sub_assign = np.argmin(dists2, axis=1)
            second_level_assignments[idxs] = sub_assign

            # Build flattened inverted lists by subcluster
            for sub in range(k_here):
                mask = (sub_assign == sub)
                ids_in_sub = idxs[mask]
                flattened_ids_list.append(ids_in_sub)
                offsets.append(offsets[-1] + ids_in_sub.size)

            # If k_here < K2, add empty subclusters for the remaining slots
            for _ in range(self.K2 - k_here):
                flattened_ids_list.append(np.array([], dtype=np.int32))
                offsets.append(offsets[-1])

        # concatenate flattened ids into one array
        if flattened_ids_list:
            self.inverted_ids = np.concatenate([arr.astype(np.int32) for arr in flattened_ids_list])
        else:
            self.inverted_ids = np.array([], dtype=np.int32)

        self.inverted_offsets = np.array(offsets, dtype=np.int32)

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

    def get_candidate_ids(self, top_sub_ids: List[int]) -> np.ndarray:
        """Given a list (or array) of flattened second-level ids, return concatenated candidate vector ids."""
        candidates = []
        for fsid in top_sub_ids:
            if fsid < 0 or fsid >= (self.K1 * self.K2):
                continue
            start = self.inverted_offsets[fsid]
            end = self.inverted_offsets[fsid + 1]
            candidates.append(self.inverted_ids[start:end])
        return np.concatenate(candidates) if candidates else np.array([], dtype=np.int32)

    def search(self, query: np.ndarray, nprobe1: int, nprobe2: int) -> np.ndarray:
        """Search the two-level index and return concatenated candidate ids.

        - `nprobe1`: number of top-level clusters to probe
        - `nprobe2`: number of second-level subclusters to probe per top cluster
        """
        q = query.astype(np.float32)
        qnorm = np.sum(q * q)
        centroid1_norms = np.sum(self.centroids1 * self.centroids1, axis=1)
        dots1 = q @ self.centroids1.T
        d1 = qnorm + centroid1_norms - 2.0 * dots1
        top_ids = np.argsort(d1)[:min(nprobe1, self.K1)]

        probe_flat_ids = []
        for top in top_ids:
            centers2 = self.centroids2[top]
            if centers2.size == 0:
                continue
            c2_norms = np.sum(centers2 * centers2, axis=1)
            dots2 = q @ centers2.T
            d2 = qnorm + c2_norms - 2.0 * dots2
            sub_ids = np.argsort(d2)[:min(nprobe2, centers2.shape[0])]
            for sub in sub_ids:
                probe_flat_ids.append(top * self.K2 + sub)

        return self.get_candidate_ids(probe_flat_ids)
