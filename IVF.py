from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
import os


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
        self.cluster_dir: str = None  # Directory for per-cluster files (optional)
        
    def fit(self, vectors: np.ndarray, batch_size: int) -> None:
        #train IVF using k-means
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=batch_size,
            random_state=self.seed,
            max_iter=100,
            verbose=0
        )
        
        num_vectors = vectors.shape[0]
        
        # Train using partial_fit for memory efficiency
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            batch = vectors[start:end].astype(np.float32)
            kmeans.partial_fit(batch)
        
        # Store the trained centroids
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
    
    def assign(self, vectors: np.memmap, batch_size: int) -> np.memmap:
        """
        Assign vectors to their nearest cluster using matrix multiplication.
        """
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Call fit() before assign().")

        num_vectors = vectors.shape[0]
        assignments = np.memmap("assignments.dat", dtype=np.int32, mode='w+', shape=(num_vectors,))
        
        # Precompute centroid norms (Equation part: ||B||^2)
        # Shape: (Num_Centroids,)
        centroid_norms = np.sum(self.centroids ** 2, axis=1)
        
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            # slice from vectors (this will be a contiguous read if `vectors` is a memmap)
            batch = vectors[start:end].astype(np.float32)
            
            # 1. Compute Batch norms (Equation part: ||A||^2)
            # Shape: (Batch_Size, 1)
            batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)
            
            # 2. Compute Dot Product (Equation part: 2A.B)
            # Shape: (Batch_Size, Num_Centroids)
            dot_product = np.dot(batch, self.centroids.T)
            
            # 3. Combine to get Squared Distance
            # Broadcasting: (B, 1) + (K,) - (B, K) -> (B, K)
            distances = batch_norms + centroid_norms - 2 * dot_product
            
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
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)
        
        # Compute distances without intermediate arrays
        centroid_norms_sq = np.einsum('ij,ij->i', self.centroids, self.centroids)
        dot_products = np.dot(self.centroids, query)
        distances = centroid_norms_sq - 2 * dot_products
        
        if nprobe < len(distances):
            nearest_clusters = np.argpartition(distances, nprobe)[:nprobe]
        else:
            nearest_clusters = np.arange(len(distances))
            
        # Cleanup
        del distances
        del dot_products
        del centroid_norms_sq
        
        return nearest_clusters.tolist()
    

    def get_candidate_ids(self, cluster_ids: List[int]) -> np.ndarray:
        candidates = []
        
        for cluster_id in cluster_ids:
            cluster_file = f"{self.cluster_dir}/cluster_{cluster_id}.npy"
            if os.path.exists(cluster_file):
                # Use mmap_mode to avoid loading entire file into memory
                cluster_data = np.load(cluster_file, mmap_mode='r')
                # Copy only what we need
                candidates.append(cluster_data.copy())
                del cluster_data
        
        if not candidates:
            return np.array([], dtype=np.int32)
        
        # Concatenate more efficiently
        total_size = sum(c.shape[0] for c in candidates)
        result = np.empty(total_size, dtype=np.int32)
        offset = 0
        for c in candidates:
            size = c.shape[0]
            result[offset:offset + size] = c
            offset += size
            del c
        
        del candidates
        return result
        
