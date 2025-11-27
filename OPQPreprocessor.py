from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.linalg import orthogonal_procrustes
import pickle
import numpy as np

class OPQPreprocessor:
    def __init__(self, num_subvectors: int, num_centroids: int, iterations: int = 20, seed: int = 42):
        self.M = num_subvectors
        self.Ks = num_centroids
        self.iterations = iterations
        self.seed = seed
        self.R = None # Rotation matrix

    def fit(self, X):
        D = X.shape[1]
        
        pca = PCA(n_components=D, random_state=self.seed)
        pca.fit(X)
        
        self.R = pca.components_.T

        for i in range(self.iterations):
            
            X_rotated = X @ self.R
            
            X_target = np.zeros_like(X_rotated)
            
            d_sub = D // self.M
            
            for m in range(self.M):
                # Slice the m-th part of the vectors
                start = m * d_sub
                end = (m + 1) * d_sub
                sub_vectors = X_rotated[:, start:end]
                
                # Run fast K-Means on this slice
                kmeans = KMeans(n_clusters=self.Ks, n_init=1, random_state=self.seed + m)
                kmeans.fit(sub_vectors)
                
                # Find nearest centroids (Assignment)
                assignments = kmeans.predict(sub_vectors)
                
                # Update target vectors with assigned centroids
                X_target[:, start:end] = kmeans.cluster_centers_[assignments]
            
            self.R, _ = orthogonal_procrustes(X, X_target)
            
            #print(f"Iteration {i+1} complete.")
    
    def transform(self, X):
        if self.R is None:
            raise ValueError("The OPQPreprocessor has not been fitted yet.")
        return X @ self.R

    def save(self, filepath: str = "opq_model.pkl"):
        """Saves the learned rotation matrix and config to disk."""
        state = {
            'num_subvectors': self.M,
            'num_centroids': self.Ks,
            'seed': self.seed,
            'R': self.R
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"OPQ model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "opq_model.pkl"):
        """Loads a trained OPQ model from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance with the loaded config
        instance = cls(
            num_subvectors=state['num_subvectors'],
            num_centroids=state['num_centroids'],
            seed=state.get('seed', 42)  # Default to 42 for backward compatibility
        )
        instance.R = state['R'] # Restore the rotation matrix
        return instance