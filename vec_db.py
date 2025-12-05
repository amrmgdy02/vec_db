from typing import Annotated
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
                 index_file_path = None, 
                 use_pq=True, 
                 new_db = True, 
                 db_size = None, 
                 M=8, 
                 Ks=256, 
                 num_clusters=16384, 
                 nprobe=128, 
                 S_ivf=131_072,
                 batch_size=131_072,
                 use_cluster_files=True) -> None:
        
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.use_pq = use_pq
        self.use_cluster_files = use_cluster_files
        self.M = M
        self.Ks = Ks
        self.num_clusters = num_clusters  
        self.nprobe = nprobe              
        self.batch_size = batch_size
        self.S_ivf = S_ivf
        # check which number does db_path contain 
        if self.db_path is not None and "1M" in self.db_path:
            self.M = 4
            self.num_clusters = 1024
            self.nprobe = 64
        elif self.db_path is not None and "10M" in self.db_path:
            self.M = 4
            self.num_clusters = 4096
            self.nprobe = 64
        elif self.db_path is not None and "20M" in self.db_path:
            self.M = 4
            self.num_clusters = 16384
            self.nprobe = 128
        
        # if index_file_path is not None and os.path.exists(index_file_path):
        #     metadata_file_path = os.path.join(index_file_path, f"{self.db_path.split('_emb_')[0]}_metadata.npy")
            
        #     if os.path.exists(metadata_file_path):
        #         metadata = np.load(metadata_file_path)
        #         self.M, self.Ks, self.num_clusters, self.nprobe = metadata
        #         print(f"Loaded index metadata: M={self.M}, Ks={self.Ks}, num_clusters={self.num_clusters}, nprobe={self.nprobe}")

        self.pq: ProductQuantizer = ProductQuantizer(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)  
        self.opq: OPQPreprocessor = OPQPreprocessor(num_subvectors=self.M, num_centroids=self.Ks, seed=DB_SEED_NUMBER)  
        self.ivf: InvertedFileIndex = InvertedFileIndex(num_clusters=self.num_clusters, seed=DB_SEED_NUMBER)
        self.pq_codes: np.memmap = None        
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
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """Ultra memory-optimized retrieval - minimize all allocations"""
        refine_factor = 100
        shortlist_k = top_k * refine_factor 

        self.load_index()
        
        # Normalize query in-place, reuse query array
        query = query.flatten()
        norm = np.linalg.norm(query)
        if norm > 0:
            query /= norm
        
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids)
        
        if len(candidate_ids) == 0:
            return []
        
        batch_size = 5000
        query_rotated = self.opq.transform(query.reshape(1, -1)).flatten()
        
        distances = np.empty(len(candidate_ids), dtype=np.float32)
        
        offset = 0
        for i in range(0, len(candidate_ids), batch_size):
            end_idx = min(i + batch_size, len(candidate_ids))
            batch_ids = candidate_ids[i:end_idx]
            
            candidate_codes = self.pq_codes[batch_ids]
            
            batch_distances = adc_distance(query_rotated, candidate_codes, self.pq)
            
            distances[offset:offset + len(batch_distances)] = batch_distances
            offset += len(batch_distances)
            
            del candidate_codes
            del batch_distances
        
        k_to_fetch = min(shortlist_k, len(candidate_ids))
        if k_to_fetch >= len(distances):
            shortlist_indices = np.argsort(distances)
        else:
            shortlist_indices = np.argpartition(distances, k_to_fetch)[:k_to_fetch]
        
        result_ids = candidate_ids[shortlist_indices]
        
        del distances
        del candidate_ids
        del shortlist_indices
        
        if len(result_ids) == 0:
            return []

        # Re-rank with MINIMAL memory: process in tiny batches
        raw_db = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                        shape=(self._get_num_records(), DIMENSION))
        
        # Ultra-small batches for re-ranking
        rerank_batch_size = 500
        top_results = []  # Store (id, score) tuples
        
        for i in range(0, len(result_ids), rerank_batch_size):
            batch_ids = result_ids[i:i + rerank_batch_size]
            
            # Process each vector individually to minimize peak memory
            for vid in batch_ids:
                vec = raw_db[vid].copy()  # Load single vector
                
                # Normalize
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    vec /= vec_norm
                
                score = np.dot(vec, query)
                top_results.append((int(vid), float(score)))
                
                del vec  
        
        top_results.sort(key=lambda x: x[1], reverse=True)
        
        final_ids = [p[0] for p in top_results[:top_k]]
        
        # Cleanup
        del top_results
        del result_ids
        
        return final_ids

        
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
            OPQ_SAMPLE_SIZE = 32_768 
            PQ_SAMPLE_SIZE = 262_144
            IVF_SAMPLE_SIZE = self.S_ivf
            
            if os.path.exists(self.index_path):
                return 
            
            # create index directory
            os.makedirs(self.index_path, exist_ok=True)
                
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

            pq_train_data  = vectors[np.sort(pq_indices)]
            opq_train_data = vectors[np.sort(opq_indices)]
            ivf_train_data = vectors[np.sort(ivf_indices)]
            
            print("Training IVF (Coarse Quantization)...")
            self.ivf.fit(ivf_train_data, batch_size=self.batch_size)
            print("IVF training completed.")
            
            print("Assigning vectors to IVF clusters...")
            assignments = self.ivf.assign(vectors, batch_size=32_768)
            print("Assignment Completed")
            self.ivf.build_inverted_lists(assignments)
            print("IVF inverted lists building completed.")
            
            # if os.path.exists("assignments.dat"):
            #     os.remove("assignments.dat")
            
            print(" Training OPQ (Rotation)...")
            self.opq.fit(opq_train_data)
            print(" OPQ training completed.")
            
            pq_train_data = self.opq.transform(pq_train_data)
            self.pq.fit(pq_train_data, batch_size=self.batch_size)
    
            print("Saving index to disk...")
            np.save(f"{self.index_path}/{db_size_str}_ivf_centroids.npy", self.ivf.centroids.astype(np.float32), allow_pickle=False)
            
            # Save each cluster's IDs in separate file
            cluster_dir = f"{self.index_path}/{db_size_str}_clusters"
            os.makedirs(cluster_dir, exist_ok=True)
            for cluster_id in range(self.num_clusters):
                start = self.ivf.inverted_offsets[cluster_id]
                end = self.ivf.inverted_offsets[cluster_id + 1]
                cluster_ids = self.ivf.inverted_ids[start:end]
                np.save(f"{cluster_dir}/cluster_{cluster_id}.npy", cluster_ids.astype(np.int32), allow_pickle=False)
            print(f"Saved {self.num_clusters} cluster files")
            
            np.save(f"{self.index_path}/{db_size_str}_opq_rotation.npy", self.opq.R)
            np.save(f"{self.index_path}/{db_size_str}_pq_codebooks.npy", self.pq.codebooks)
            
            self.pq_codes = np.memmap(f"{self.index_path}/{db_size_str}_pq_codes.dat", dtype=np.uint8, mode='w+', shape=(num_records, self.M))
            
            for start in range(0, num_records, self.batch_size):
                end = min(start + self.batch_size, num_records)
                batch_vectors = vectors[start:end]
                batch_vectors = self.opq.transform(batch_vectors)
                codes_batch = self.pq.encode(batch_vectors, batch_size=self.batch_size)
                self.pq_codes[start:end] = codes_batch
                
            self.pq_codes.flush()
            
            print("Index saved to disk.")

    def load_index(self) -> None:
        """Lazy load only metadata, keep heavy data as memmap"""
        db_size_str = self.db_path.split("_emb_")[0]  
        
        self.ivf.centroids = np.load(f"{self.index_path}/{db_size_str}_ivf_centroids.npy")
        self.ivf.cluster_dir = f"{self.index_path}/{db_size_str}_clusters"
        num_records = self._get_num_records()
        self.opq.R = np.load(f"{self.index_path}/{db_size_str}_opq_rotation.npy")
        self.pq.codebooks = np.load(f"{self.index_path}/{db_size_str}_pq_codebooks.npy")
        
        # Keep PQ codes as memmap
        self.pq_codes = np.memmap(
            f"{self.index_path}/{db_size_str}_pq_codes.dat", 
            dtype=np.uint8, 
            mode='r', 
            shape=(num_records, self.M)
        )
