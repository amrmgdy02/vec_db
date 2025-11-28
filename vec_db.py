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
    def __init__(self, database_file_path = "saved_db.dat", 
                 index_file_path = "index.dat", 
                 use_pq=True, 
                 new_db = True, 
                 db_size = None, 
                 M=8, 
                 Ks=256, 
                 num_clusters=16384, 
                 nprobe=128, 
                 S_ivf=131_072,
                 batch_size=131_072) -> None:
        
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.use_pq = use_pq
        self.M = M
        self.Ks = Ks
        self.num_clusters = num_clusters  
        self.nprobe = nprobe              
        self.batch_size = batch_size
        self.S_ivf = S_ivf

        self.pq: ProductQuantizer = ProductQuantizer(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)       # PQ object
        self.opq: OPQPreprocessor = OPQPreprocessor(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)       # OPQ object
        self.ivf: InvertedFileIndex = InvertedFileIndex(num_clusters=self.num_clusters, seed=DB_SEED_NUMBER)     # IVF object
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
        if self.use_pq:
            return self.retrieve_with_pq(query, top_k)
        else:
            return self.retrieve_without_pq(query, top_k)
    
    def retrieve_with_pq(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, refine_factor=100):
        # 1. Ask PQ for MORE results than we need (The Shortlist)
        shortlist_k = top_k * refine_factor  # e.g., 5 * 100 = 500

        self.load_index(use_pq=True)
        query = query.flatten() / np.linalg.norm(query)
        
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids)
        
        if len(candidate_ids) == 0:
            print("Warning: No candidates found in IVF search")
            return []
        
        # 3. Get PQ codes
        candidate_codes = self.pq_codes[candidate_ids]
        
        # 4. Rotate query
        query_rotated = self.opq.transform(query.reshape(1, -1)).flatten()
        
        # 5. Compute ADC distances
        distances = adc_distance(query_rotated, candidate_codes, self.pq)
        
        # -------------------------------------------------------------
        # FIX 1: Use 'shortlist_k' here, NOT 'top_k'
        # -------------------------------------------------------------
        # We want the top 500 candidates from PQ, not just the top 5.
        
        k_to_fetch = min(shortlist_k, len(candidate_ids))

        if len(candidate_ids) <= k_to_fetch:
            shortlist_indices = np.argsort(distances)
        else:
            # Partition to get the best 'shortlist_k' (unsorted order is fine for re-ranking)
            shortlist_indices = np.argpartition(distances, k_to_fetch)[:k_to_fetch]
        
        # Map back to global IDs
        # These are the 500 candidates we will load from disk
        result_ids = [candidate_ids[i] for i in shortlist_indices]

        # -------------------------------------------------------------
        # Re-Ranking Step (Now applied to 500 vectors)
        # -------------------------------------------------------------
        if len(result_ids) == 0:
            return []

        # 1. Load RAW vectors
        raw_db = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                           shape=(self._get_num_records(), DIMENSION))
        
        shortlist_vectors = np.zeros((len(result_ids), DIMENSION), dtype=np.float32)
        
        # Sort IDs for sequential disk access
        sorted_pairs = sorted(enumerate(result_ids), key=lambda x: x[1])
        sorted_ids = [p[1] for p in sorted_pairs]
        original_indices = [p[0] for p in sorted_pairs] # Keep track of where they go
        
        for i, vector_id in enumerate(sorted_ids):
            shortlist_vectors[i] = raw_db[vector_id]
        
        # -------------------------------------------------------------
        # FIX 2: Safety Normalization
        # -------------------------------------------------------------
        # Even if DB is normalized, re-normalizing here costs nothing (N=500) 
        # and prevents bugs if raw_db has floating point drift.
        norms = np.linalg.norm(shortlist_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        shortlist_vectors /= norms
        
        # 2. Exact Cosine Similarity
        # Note: 'query' was already normalized at the start of the function
        exact_scores = np.dot(shortlist_vectors, query)
        
        # 3. Final Sort
        # We match scores back to the IDs
        final_pairs = list(zip(sorted_ids, exact_scores))
        
        # Sort descending (Higher Score = Better)
        final_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Return Final Top K
        final_top_k_ids = [p[0] for p in final_pairs[:top_k]]
        
        return final_top_k_ids
    
    def retrieve_without_pq(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """
        Retrieve top_k most similar vectors using IVF without PQ (brute-force within clusters).
        
        Search algorithm:
        1. Find nprobe nearest IVF clusters
        2. Get candidate vector IDs from those clusters
        3. Compute exact distances to candidates
        4. Return top_k nearest vectors
        """
        self.load_index(use_pq=False)
        
        query = query.flatten() / np.linalg.norm(query)
        
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids)
        
        if len(candidate_ids) == 0:
            print("Warning: No candidates found in IVF search")
            return []
        # Filter invalid IDs (defensive) and load candidate vectors in a single memmap slice
        num_records = self._get_num_records()
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

        # Read candidate vectors directly from the DB memmap (faster than calling get_one_row repeatedly)
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        candidate_vectors = np.asarray(db_vectors[candidate_ids])

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

    def _build_index(self, apply_pca=False):
            OPQ_SAMPLE_SIZE = 32_768 
            PQ_SAMPLE_SIZE = 262_144
            IVF_SAMPLE_SIZE = self.S_ivf

            db_size_str = self.db_path.split("_emb_")[0]  # get the number before 'M'

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
            pq_indices  = rng.choice(num_records, size=PQ_SAMPLE_SIZE, replace=False)
            opq_indices = rng.choice(num_records, size=OPQ_SAMPLE_SIZE, replace=False)

            # Sort indices for faster disk seek
            pq_train_data  = vectors[np.sort(pq_indices)]
            opq_train_data = vectors[np.sort(opq_indices)]
            ivf_train_data = vectors[np.sort(ivf_indices)]
            
            print("Training IVF (Coarse Quantization)...")
            self.ivf.fit(ivf_train_data, batch_size=self.batch_size)
            print("IVF training completed.")
            
            # save the centroids as numpy file
            centroids = self.ivf.centroids
            # create indexes folder if not exists
            if not os.path.exists("indexes"):
                os.makedirs("indexes")
            np.save(f"indexes/{db_size_str}_ivf_centroids.npy", centroids.astype(np.float32), allow_pickle=False)
            
            print("Assigning vectors to IVF clusters...")
            assignments = self.ivf.assign(vectors, batch_size=32_768)
            print("Assignment Completed")
            self.ivf.build_inverted_lists(assignments)
            #self.ivf.save("models/ivf_model.pkl")
            print("IVF assignment and inverted list building completed.")
            
            # save the inverted lists to index file
            np.save(f"indexes/{db_size_str}_inverted_ids.npy", self.ivf.inverted_ids.astype(np.int32)) # large
            np.save(f"indexes/{db_size_str}_inverted_offsets.npy", self.ivf.inverted_offsets.astype(np.int32)) # small
            
            print(" Training OPQ (Rotation)...")
            self.opq.fit(opq_train_data)
            
            #self.opq.save("models/opq_model.pkl")
            print(" OPQ training completed.")
            
            # save the rotation matrix
            np.save(f"indexes/{db_size_str}_opq_rotation.npy", self.opq.R)

            pq_train_data = self.opq.transform(pq_train_data)

            # Fit PQ codebooks (batch processing inside PQ)
            self.pq.fit(pq_train_data, batch_size=self.batch_size)
            
            # Save the codebooks
            np.save(f"indexes/{db_size_str}_pq_codebooks.npy", self.pq.codebooks)

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

    def load_index(self, use_pq=True):
        """Load all needed index components from disk."""
        
        self.ivf.centroids = np.load("indexes/ivf_centroids.npy")
        self.ivf.inverted_offsets = np.load("indexes/inverted_offsets.npy")
        # load inverted ids as memmap for large size
        self.ivf.inverted_ids = np.load("indexes/inverted_ids.npy", mmap_mode="r")
        
        if use_pq:
            self.opq.R = np.load("indexes/opq_rotation.npy")
            self.pq.codebooks = np.load("indexes/pq_codebooks.npy")
            
            num_records = self._get_num_records()
            self.pq_codes = np.memmap(self.index_path, dtype=np.uint8, mode='r', shape=(num_records, self.M))

