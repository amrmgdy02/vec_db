from typing import Dict, List, Annotated
import numpy as np
import os

from PQ import ProductQuantizer, adc_distance
from OPQPreprocessor import OPQPreprocessor
from IVF import InvertedFileIndex

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None, M=8, Ks=256, num_clusters=100, nprobe=30, batch_size=131_072) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.M = M
        self.Ks = Ks
        self.num_clusters = num_clusters  
        self.nprobe = nprobe              
        self.batch_size = batch_size

        self.pq: ProductQuantizer = None       # PQ object
        self.opq: OPQPreprocessor = None       # OPQ object
        self.ivf: InvertedFileIndex = None     # IVF object
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
        """
        Retrieve top_k most similar vectors using IVF-PQ.
        
        Search algorithm:
        1. Rotate query using OPQ
        2. Find nprobe nearest IVF clusters
        3. Get candidate vector IDs from those clusters
        4. Compute ADC distances to candidates using PQ codes
        5. Return top_k nearest vectors
        """
        query = query.flatten()  # Ensure query is 1D
        
        # 1. Find nprobe nearest clusters
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        
        # 2. Get candidate vector IDs from inverted lists
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids)
        
        if len(candidate_ids) == 0:
            print("Warning: No candidates found in IVF search")
            return []
        
        # 3. Get PQ codes for candidates
        candidate_codes = self.pq_codes[candidate_ids]
        
        # 4. Rotate query using OPQ
        query_rotated = self.opq.transform(query.reshape(1, -1)).flatten()
        
        # 5. Compute ADC distances using PQ
        distances = adc_distance(query_rotated, candidate_codes, self.pq)
        
        # 6. Get top_k nearest (smallest distances)
        # Note: ADC returns squared distances, so smaller is better
        if len(candidate_ids) < top_k:
            top_k_indices = np.argsort(distances)
        else:
            top_k_indices = np.argpartition(distances, top_k)[:top_k]
            top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        # Map back to original vector IDs
        result_ids = [candidate_ids[i] for i in top_k_indices]
        
        return result_ids
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, apply_pca=False):
            OPQ_SAMPLE_SIZE = 32_768 
            PQ_SAMPLE_SIZE = 262_144
            IVF_SAMPLE_SIZE = 262_144
            
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
            ivf_indices = rng.choice(num_records, size=min(IVF_SAMPLE_SIZE, num_records), replace=False)
            pq_indices  = rng.choice(num_records, size=PQ_SAMPLE_SIZE, replace=False)
            opq_indices = pq_indices[:OPQ_SAMPLE_SIZE]

            # Sort indices for faster disk seek
            pq_train_data  = vectors[np.sort(pq_indices)]
            opq_train_data = vectors[np.sort(opq_indices)]
            ivf_train_data = vectors[np.sort(ivf_indices)]
            
            print("Training IVF (Coarse Quantization)...")
            self.ivf = InvertedFileIndex(num_clusters=self.num_clusters, seed=DB_SEED_NUMBER)
            self.ivf.fit(ivf_train_data, batch_size=self.batch_size)
            print("IVF training completed.")
            
            print("Assigning vectors to IVF clusters...")
            assignments = self.ivf.assign(vectors, batch_size=self.batch_size)
            self.ivf.build_inverted_lists(assignments)
            self.ivf.save("models/ivf_model.pkl")
            print("IVF assignment and inverted list building completed.")
            
            
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
                codes_batch = self.pq.encode(batch_vectors, batch_size=self.batch_size)
                self.pq_codes[start:end] = codes_batch
            self.pq_codes.flush() #Ensure all memmap changes are written to disk.


