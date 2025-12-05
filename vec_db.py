from typing import Dict, List, Annotated
import numpy as np
import os
from IVF import TwoLevelIVF

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat",
                 index_file_path = "saved_index",
                 new_db = True, 
                 db_size = None, 
                 first_level_num_clusters=256,
                 second_level_num_clusters=256,
                 nprobe1=5, 
                 nprobe2=10,
                 batch_size=131_072) -> None:
        
        self.db_path = database_file_path
        self.first_level_num_clusters = first_level_num_clusters
        self.second_level_num_clusters = second_level_num_clusters
        self.nprobe1 = nprobe1
        self.nprobe2 = nprobe2
        self.batch_size = batch_size
        self.twolevel: TwoLevelIVF = None      # Two-level IVF
        self.index_path = index_file_path
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """Retrieve using the trained two-level IVF with on-demand cluster file loading."""
        # Load index only once (not every query)
        if self.twolevel is None or self.twolevel.centroids1 is None:
            self.load_index()
        
        query = query.flatten() / np.linalg.norm(query)

        candidate_ids = self.twolevel.search(query, 
                                            nprobe1=self.nprobe1, 
                                            nprobe2=self.nprobe2,
                                            cluster_dir=self.index_path)

        if len(candidate_ids) == 0:
            print("Warning: No candidates found in two-level IVF search")
            return []
        
        num_records = self._get_num_records()
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        candidate_vectors = np.asarray(db_vectors[candidate_ids])

        dot_products = candidate_vectors @ query
        distances = dot_products

        if len(candidate_ids) < top_k:
            top_k_indices = np.argsort(-distances)
        else:
            top_k_indices = np.argpartition(-distances, top_k)[:top_k]
            top_k_indices = top_k_indices[np.argsort(-distances[top_k_indices])]

        result_ids = [candidate_ids[i] for i in top_k_indices]
        return result_ids
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
            IVF_SAMPLE_SIZE = 500_000
            
            """
            Build PQ index:
            1) Load vectors in batches
            """
            num_records = self._get_num_records()
            
            print(" Building Index for ", num_records, "records")
            
            if num_records <= 1_000_000:
                self.total_clusters_num = num_records // 1000
            else:
                self.total_clusters_num = int(np.sqrt(num_records))
                
            self.first_level_num_clusters = int(np.sqrt(self.total_clusters_num))
            self.second_level_num_clusters = self.total_clusters_num // self.first_level_num_clusters
            
            nprobe = int(np.sqrt(self.total_clusters_num))
            self.nprobe1 = int(np.sqrt(nprobe))
            self.nprobe2 = nprobe // self.nprobe1
            # check num of rows
            if num_records == 20_000_000:
                self.nprobe1 = 5
                self.nprobe2 = 5
            
            vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=(num_records, DIMENSION)) 
            # normalize all vectors
            for start in range(0, num_records, self.batch_size):
                end = min(start + self.batch_size, num_records)
                batch = vectors[start:end]
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                batch /= norms
                vectors[start:end] = batch
            

            rng = np.random.default_rng(DB_SEED_NUMBER)
            ivf_indices = rng.choice(num_records, size=min(IVF_SAMPLE_SIZE, num_records), replace=False)

            ivf_train_data = vectors[np.sort(ivf_indices)]
            
            print("Training Two-Level IVF (Coarse Quantization)...")
            self.twolevel = TwoLevelIVF(K1=self.first_level_num_clusters, K2=self.second_level_num_clusters, seed=DB_SEED_NUMBER)
            self.twolevel.fit(ivf_train_data, batch_size=self.batch_size)
            print("Two-level IVF training completed.")

            assignments = self.twolevel.assign(vectors, batch_size=self.batch_size)
            print("Assignment Completed")

            if not os.path.exists("indexes"):
                os.makedirs("indexes")
            
            np.save("indexes/ivf1_centroids.npy", self.twolevel.centroids1.astype(np.float32))
            # Save centroids2 as a list of arrays using pickle
            import pickle
            with open("indexes/ivf2_centroids.pkl", "wb") as f:
                pickle.dump(self.twolevel.centroids2, f)
            
            print("Saving cluster files in hierarchical structure...")
            
            # Group vectors by their cluster assignments
            total_clusters = self.twolevel.K1 * self.twolevel.K2
            cluster_vectors = [[] for _ in range(total_clusters)]
            
            for vec_id, cluster_id in enumerate(assignments):
                cluster_vectors[cluster_id].append(vec_id)
            
            # Save to hierarchical file structure
            for first_level_id in range(self.twolevel.K1):
                first_level_dir = f"indexes/cluster_L1_{first_level_id}"
                os.makedirs(first_level_dir, exist_ok=True)
                
                # Save each second-level cluster within this first-level cluster
                for second_level_id in range(self.twolevel.K2):
                    # Calculate flattened cluster ID
                    flat_cluster_id = first_level_id * self.twolevel.K2 + second_level_id
                    
                    # Get vector IDs for this cluster
                    cluster_vector_ids = np.array(cluster_vectors[flat_cluster_id], dtype=np.int32)
                    
                    # Save to file
                    cluster_file = f"{first_level_dir}/cluster_L2_{second_level_id}.npy"
                    np.save(cluster_file, cluster_vector_ids)
            
            print(f"Saved {self.twolevel.K1} first-level directories with {self.twolevel.K2} files each")

    def load_index(self):
        """Load centroids only - cluster files will be loaded on-demand during retrieval."""
        c1 = np.load("indexes/ivf1_centroids.npy")
        # Load centroids2 list from pickle
        import pickle
        with open("indexes/ivf2_centroids.pkl", "rb") as f:
            centroids2 = pickle.load(f)

        self.twolevel = TwoLevelIVF(K1=c1.shape[0], K2=self.second_level_num_clusters, seed=DB_SEED_NUMBER)
        self.twolevel.centroids1 = c1
        self.twolevel.centroids2 = centroids2
        # No inverted lists - will load cluster files on demand
        return



