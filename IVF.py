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
        self.inverted_lists: Dict[int, List[int]] = {}  # Maps cluster_id -> list of vector IDs
        
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
            'inverted_lists': self.inverted_lists
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
        instance.inverted_lists = state['inverted_lists']
        print(f"IVF model loaded from {filepath}")
        return instance
