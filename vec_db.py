from typing import Dict, List, Annotated
import numpy as np
import os
from IVF import TwoLevelIVF

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", 
                 new_db = True, 
                 db_size = None, 
                 first_level_num_clusters=1000,
                 second_level_num_clusters=1000,
                 nprobe1=50, 
                 nprobe2=50,
                 batch_size=32_768) -> None:
        
        self.db_path = database_file_path
        self.first_level_num_clusters = first_level_num_clusters
        self.second_level_num_clusters = second_level_num_clusters
        self.nprobe1 = nprobe1
        self.nprobe2 = nprobe2
        self.batch_size = batch_size
        self.twolevel: TwoLevelIVF = None      # Two-level IVF
        
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
        """Retrieve using the trained two-level IVF (with OPQ rotation applied if present)."""
        self.load_index(use_pq=False)
        query = query.flatten() / np.linalg.norm(query)

        candidate_ids = self.twolevel.search(query, nprobe1=self.nprobe1, nprobe2=self.nprobe2)

        if len(candidate_ids) == 0:
            print("Warning: No candidates found in two-level IVF search")
            return []

        # Read candidate vectors from DB memmap
        num_records = self._get_num_records()
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        candidate_vectors = np.asarray(db_vectors[candidate_ids])

        # compute cosine similarities
        distances = np.array([self._cal_score(query, vec) for vec in candidate_vectors])

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
            IVF_SAMPLE_SIZE = 262_144
            
            """
            Build PQ index:
            1) Load vectors in batches
            """
            num_records = self._get_num_records()
            vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=(num_records, DIMENSION)) 

            batch = self.batch_size
            for start in range(0, num_records, batch):
                end = min(start + batch, num_records)
                block = vectors[start:end]        
                norms = np.linalg.norm(block.astype(np.float32), axis=1, keepdims=True).astype(np.float32)
                norms[norms == 0] = 1.0
                block /= norms
            
            vectors.flush()

            rng = np.random.default_rng(DB_SEED_NUMBER)
            ivf_indices = rng.choice(num_records, size=min(IVF_SAMPLE_SIZE, num_records), replace=False)

            ivf_train_data = vectors[np.sort(ivf_indices)]
            
            print("Training Two-Level IVF (Coarse Quantization)...")
            self.twolevel = TwoLevelIVF(K1=self.first_level_num_clusters, K2=self.second_level_num_clusters, seed=DB_SEED_NUMBER)
            self.twolevel.fit(ivf_train_data, batch_size=self.batch_size)
            print("Two-level IVF training completed.")

            assignments = self.twolevel.assign(vectors, batch_size=self.batch_size)
            print("Assignment Completed")

            # Build flattened inverted lists from assignments
            total_clusters = self.twolevel.K1 * self.twolevel.K2
            cluster_sizes = np.bincount(assignments, minlength=total_clusters)
            self.twolevel.inverted_ids = np.zeros(len(assignments), dtype=np.int32)
            self.twolevel.inverted_offsets = np.zeros(total_clusters + 1, dtype=np.int32)
            self.twolevel.inverted_offsets[1:] = np.cumsum(cluster_sizes)
            current_pos = self.twolevel.inverted_offsets.copy()
            for vec_id, cluster_id in enumerate(assignments):
                pos = current_pos[cluster_id]
                self.twolevel.inverted_ids[pos] = vec_id
                current_pos[cluster_id] += 1

            print("Two-level assignment and inverted list building completed.")

            # ensure index directory exists
            if not os.path.exists("indexes"):
                os.makedirs("indexes")
            np.save("indexes/ivf1_centroids.npy", self.twolevel.centroids1.astype(np.float32), allow_pickle=False)
            np.savez("indexes/ivf2_centroids.npz", *self.twolevel.centroids2)
            np.save("indexes/inverted_ids.npy", self.twolevel.inverted_ids.astype(np.int32))
            np.save("indexes/inverted_offsets.npy", self.twolevel.inverted_offsets.astype(np.int32))
            

    def load_index(self, use_pq=True):
        """Load all needed index components from disk."""
        c1 = np.load("indexes/ivf1_centroids.npy")
        c2_npz = np.load("indexes/ivf2_centroids.npz")
        centroids2 = [c2_npz[key] for key in c2_npz]

        self.twolevel = TwoLevelIVF(K1=c1.shape[0], K2=self.second_level_num_clusters, seed=DB_SEED_NUMBER)
        self.twolevel.centroids1 = c1
        self.twolevel.centroids2 = centroids2
        self.twolevel.inverted_offsets = np.load("indexes/inverted_offsets.npy")
        self.twolevel.inverted_ids = np.load("indexes/inverted_ids.npy", mmap_mode="r")
        return

