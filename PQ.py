from typing import List, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

class ProductQuantizer:
    def __init__(self, num_subvectors: int, num_centroids: int, seed: int = 42) -> None:
        """
        Initialize the Product Quantizer.
        Args:
            num_subvectors M(int): Number of subvectors to split the original vector into. here each subvector will be of size 64 / num_subvectors
            num_centroids Ks(int): Number of centroids per subquantizer.
            seed (int): Random seed for reproducibility.
        """
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids
        self.codebooks: List[np.ndarray] = [] # List of codebooks for each subvector , len=num_subvectors
        self.seed = seed

    def fit(self, vectors: np.ndarray, batch_size: int) -> None:
        """
        Train PQ codebooks on database vectors using MiniBatchKMeans with streaming (partial_fit).
        to works for huge datasets under strict RAM limits.

        NOTE: Upstream step (before calling PQ.fit):
        NOTE: The input vectors should already be preprocessed with PCA or OPQ rotation.(not implemented yet)
        call example after PCA/OPQ rotation:
            pq = ProductQuantizer(num_subvectors=10, num_centroids=256)
            pq.fit(rotated_vectors)  #where rotated_vectors come from PCA/OPQ step   
            
        Args:
            vectors (np.ndarray): Input vectors to train on, shape (num_vectors, dimension).
            batch_size (int): Size of mini-batches for training.
        """

        dim = vectors.shape[1]
        if dim % self.num_subvectors != 0:
            raise ValueError("Dimension must be divisible by number of subvectors")

        subvector_dim = dim // self.num_subvectors
        self.codebooks = []

        num_vectors = vectors.shape[0]

        # Train 1 sub-vector KMeans at a time (saves memory)
        for i in range(self.num_subvectors):

            kmeans = MiniBatchKMeans(
                n_clusters=self.num_centroids,
                batch_size=batch_size,
                random_state=self.seed + i  # different seed per subspace
            )

            start = i * subvector_dim
            end = (i + 1) * subvector_dim

            # Stream training using partial_fit
            for b in range(0, num_vectors, batch_size): # iterate over batches up to num_vectors
                batch = vectors[b:b+batch_size, start:end].astype(np.float32)
                kmeans.partial_fit(batch) #MiniBatchKMeans has now seen all subvector pieces but never loaded all into memory at once.


            # Store centroid codebook for this subvector
            self.codebooks.append(kmeans.cluster_centers_.astype(np.float32))


    def encode(self, vectors: np.ndarray, batch_size: int) -> np.ndarray:
        """
        compress database vectors into PQ codes using the trained codebooks.
        
        we use batch size We split the vectors into smaller chunks (batches).
        Example: batch_size = 100_000
        Compute distances for 100k vectors at a time so fits easily in memory.
        Repeat for all batches until all 20M vectors are encoded.
        we use squared L2 distance for efficiency (no sqrt needed for argmin).
        Args:
            vectors (np.ndarray): Input vectors to encode, shape (num_vectors, dimension)
            batch_size (int): Batch size for processing.
        Returns:
            np.ndarray: PQ codes, shape (num_vectors, num_subvectors)
        """

        dim = vectors.shape[1]
        if dim % self.num_subvectors != 0:
            raise ValueError("Dimension must be divisible by number of subvectors")

        subvector_dim = dim // self.num_subvectors
        codes = np.zeros((vectors.shape[0], self.num_subvectors), dtype=np.uint8)

        num_vectors = vectors.shape[0]

        for i in range(self.num_subvectors):
            centroids = self.codebooks[i].astype(np.float32) 

            for start in range(0, num_vectors, batch_size):
                end = min(start + batch_size, num_vectors)
                batch = vectors[start:end, i*subvector_dim:(i+1)*subvector_dim].astype(np.float32)

                # Compute squared L2 distances (no sqrt)
                distances = ((batch[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

                codes[start:end, i] = np.argmin(distances, axis=1)

        return codes



    # def decode(self, codes: np.ndarray) -> np.ndarray:
    #     """
    #     Decode PQ codes back to approximate vectors using codebooks.
    #     Args:
    #         codes (np.ndarray): PQ codes, shape (num_vectors, num_subvectors)
    #     Returns:
    #         np.ndarray: Reconstructed vectors, shape (num_vectors, dimension)
    #     """

    #     num_vectors = codes.shape[0]
    #     dim = self.num_subvectors * self.codebooks[0].shape[1]
    #     reconstructed = np.zeros((num_vectors, dim), dtype=np.float32)

    #     subvector_dim = dim // self.num_subvectors

    #     for i in range(self.num_subvectors):
    #         centroids = self.codebooks[i].astype(np.float32) 
    #         reconstructed[:, i*subvector_dim:(i+1)*subvector_dim] = centroids[codes[:, i]]

    #     return reconstructed


def adc_distance(query: np.ndarray, codes: np.ndarray, pq: ProductQuantizer) -> np.ndarray:
    """
    Compute ADC distances from a query to PQ-coded database.
    Optimized for large datasets.
    Args:
        query (np.ndarray): Query vector, shape (dimension,)
        codes (np.ndarray): PQ codes of database vectors, shape (num_vectors, num_subvectors)
        pq (ProductQuantizer): Trained ProductQuantizer instance.
    Returns:
        np.ndarray: Distances from query to each database vector, shape (num_vectors,)
    """
    M = pq.num_subvectors
    Ks = pq.num_centroids
    subdim = query.shape[0] // M

    # 1) Build Lookup Table: shape (M, Ks)
    lookup = np.zeros((M, Ks), dtype=np.float32)
    for m in range(M):
        start, end = m*subdim, (m+1)*subdim
        subq = query[start:end].astype(np.float32)
        centroids = pq.codebooks[m].astype(np.float32)
        lookup[m] = ((centroids - subq[None, :])**2).sum(axis=1)  # squared L2 distance

    # 2) Vectorized scoring using lookup table
    distances = lookup[np.arange(M)[:, None], codes.T].sum(axis=0)  # shape (N,)
    return distances
