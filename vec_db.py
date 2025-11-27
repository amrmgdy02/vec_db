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
                 num_clusters=1000, 
                 nprobe=100, 
                 batch_size=131_072) -> None:
        
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.use_pq = use_pq
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
        if self.use_pq:
            return self.retrieve_with_pq(query, top_k)
        else:
            return self.retrieve_without_pq(query, top_k)
    
    def retrieve_with_pq(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """
        Retrieve top_k most similar vectors using IVF-PQ.
        
        Search algorithm:
        1. Rotate query using OPQ
        2. Find nprobe nearest IVF clusters
        3. Get candidate vector IDs from those clusters
        4. Compute ADC distances to candidates using PQ codes
        5. Return top_k nearest vectors
        """
        self.load_index(use_pq=True)
        query = query.flatten() / np.linalg.norm(query)
        
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids) # Needs to get inverted lists (ids and offsets) from index file
        
        if len(candidate_ids) == 0:
            print("Warning: No candidates found in IVF search")
            return []
        
        # 3. Get PQ codes for candidates
        candidate_codes = self.pq_codes[candidate_ids] # Needs to load pq_codes from disk (for each vector, List of size M)
        
        # 4. Rotate query using OPQ
        query_rotated = self.opq.transform(query.reshape(1, -1)).flatten()  # Needs to get R from index file
        
        # 5. Compute ADC distances using PQ
        distances = adc_distance(query_rotated, candidate_codes, self.pq) # Needs to load Centroids (codebooks) from index file
        
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
            self.ivf = InvertedFileIndex(num_clusters=self.num_clusters, seed=DB_SEED_NUMBER)
            self.ivf.fit(ivf_train_data, batch_size=self.batch_size)
            print("IVF training completed.")
            
            # save the centroids as numpy file
            centroids = self.ivf.centroids
            # create indexes folder if not exists
            if not os.path.exists("indexes"):
                os.makedirs("indexes")
            np.save("indexes/ivf_centroids.npy", centroids.astype(np.float32), allow_pickle=False)
            
            print("Assigning vectors to IVF clusters...")
            assignments = self.ivf.assign(vectors, batch_size=self.batch_size)
            print("Assignment Completed")
            self.ivf.build_inverted_lists(assignments)
            #self.ivf.save("models/ivf_model.pkl")
            print("IVF assignment and inverted list building completed.")
            
            # save the inverted lists to index file
            np.save("indexes/inverted_ids.npy", self.ivf.inverted_ids.astype(np.int32)) # large
            np.save("indexes/inverted_offsets.npy", self.ivf.inverted_offsets.astype(np.int32)) # small
            
            print(" Training OPQ (Rotation)...")
            self.opq = OPQPreprocessor(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)
            self.opq.fit(opq_train_data)
            
            #self.opq.save("models/opq_model.pkl")
            print(" OPQ training completed.")
            
            # save the rotation matrix
            np.save("indexes/opq_rotation.npy", self.opq.R)

            pq_train_data = self.opq.transform(pq_train_data)

            # Initialize PQ
            self.pq = ProductQuantizer(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)

            # Fit PQ codebooks (batch processing inside PQ)
            self.pq.fit(pq_train_data, batch_size=self.batch_size)
            
            # Save the codebooks
            np.save("indexes/pq_codebooks.npy", self.pq.codebooks)

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

