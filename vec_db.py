from typing import Dict, List, Annotated
import numpy as np
import os

from PQ import ProductQuantizer
from OPQPreprocessor import OPQPreprocessor

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None,M=8, Ks=256, batch_size=131_072) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.M = M
        self.Ks = Ks
        self.batch_size = batch_size

        self.pq: ProductQuantizer = None       # PQ object
        self.opq: OPQPreprocessor = None       # OPQ object
        self.pq_codes: np.memmap = None        # PQ codes stored on disk
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
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, apply_pca=False):
        OPQ_SAMPLE_SIZE = 32_768 
        PQ_SAMPLE_SIZE = 262_144
        
        """
        Build PQ index:
        1) Load vectors in batches 
        2) Optional PCA rotation for better PQ accuracy
        3) Train PQ codebooks
        4) Encode all vectors into PQ codes
        5) Store PQ codes as memmap on disk
        """
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION)) 

        #TOOOOODOOOOOO: IMPLEMENT ROTATION
        rng = np.random.default_rng(DB_SEED_NUMBER)
        pq_indices  = rng.choice(num_records, size=PQ_SAMPLE_SIZE, replace=False)
        opq_indices = pq_indices[:OPQ_SAMPLE_SIZE]

        # Sort indices for faster disk seek
        pq_train_data  = vectors[np.sort(pq_indices)]
        opq_train_data = vectors[np.sort(opq_indices)]
            
        print(" Training OPQ (Rotation)...")
        self.opq = OPQPreprocessor(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)
        self.opq.fit(opq_train_data)
        self.opq.save("models/opq_model.pkl")

        print(" Rotating PQ training sample...")
        # We must rotate the PQ training data using the learned matrix so PQ learns codebooks on the *rotated* space.
        pq_train_data = self.opq.transform(pq_train_data)

        # Initialize PQ
        self.pq = ProductQuantizer(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)

        # Fit PQ codebooks (batch processing inside PQ)
        self.pq.fit(pq_train_data, batch_size=self.batch_size)

        # Encode vectors into PQ codes
        if os.path.exists(self.index_path):
            os.remove(self.index_path) #Clean old index file if it exists
        self.pq_codes = np.memmap(self.index_path, dtype=np.uint8, mode='w+', shape=(num_records, self.M))
        
        #Encode vectors into PQ codes in batches to save memory
        for start in range(0, num_records, self.batch_size):
            end = min(start + self.batch_size, num_records)
            batch_vectors = vectors[start:end]
            batch_vectors = self.opq.transform(batch_vectors)
            codes_batch = self.pq.encode(batch_vectors)
            self.pq_codes[start:end] = codes_batch
        self.pq_codes.flush() #Ensure all memmap changes are written to disk.


