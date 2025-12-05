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
        # check which number does db_path contain 
        if self.db_path is not None and "1M" in self.db_path:
            self.M = 16
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
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        if self.use_pq:
            return self.retrieve_with_pq(query, top_k)
        else:
            return self.retrieve_without_pq(query, top_k)
    
    def retrieve_with_pq(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, refine_factor=100):
        """
        Retrieve top_k most similar vectors using IVF with PQ with additional re-ranking.
        Args:
            query (Annotated[np.ndarray, (1, DIMENSION)]): Query vector for retrieval.
            top_k (int, optional): Number of top similar vectors to retrieve. Defaults to 5.
            refine_factor (int, optional): Refinement factor for re-ranking. Defaults to 100.

        Returns:
            List of top_k most similar vector IDs.
        """
        shortlist_k = top_k * refine_factor 

        self.load_index(use_pq=True)
        query = query.flatten() / np.linalg.norm(query)
        
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids)
        
        if len(candidate_ids) == 0:
            print("Warning: No candidates found in IVF search")
            return []
        
        candidate_codes_list = []
        for cid in cluster_ids:
            start = self.ivf.inverted_offsets[cid]
            end = self.ivf.inverted_offsets[cid+1]
            
            # This reads a continuous block from disk
            candidate_codes_list.append(self.pq_codes[start:end])

        candidate_codes = np.concatenate(candidate_codes_list)
        
        query_rotated = self.opq.transform(query.reshape(1, -1)).flatten()
        distances = adc_distance(query_rotated, candidate_codes, self.pq)
    
        k_to_fetch = min(shortlist_k, len(candidate_ids))
        if len(candidate_ids) <= k_to_fetch:
            shortlist_indices = np.argsort(distances)
        else:
            shortlist_indices = np.argpartition(distances, k_to_fetch)[:k_to_fetch]
        
        # Map back to global IDs
        result_ids = [candidate_ids[i] for i in shortlist_indices]

        # Re-Ranking
        if len(result_ids) == 0:
            return []

        raw_db = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                           shape=(self._get_num_records(), DIMENSION))
        
        sorted_ids = np.array(sorted(result_ids))
        
        # Read in one chunk (shortlist is small, ~500 vectors max)
        shortlist_vectors = np.array(raw_db[sorted_ids])
        
        norms = np.linalg.norm(shortlist_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        shortlist_vectors /= norms
        
        exact_scores = np.dot(shortlist_vectors, query)
        
        final_pairs = list(zip(sorted_ids, exact_scores))
        final_pairs.sort(key=lambda x: x[1], reverse=True)
        
        final_top_k_ids = [int(p[0]) for p in final_pairs[:top_k]]
        return final_top_k_ids
    
    def retrieve_without_pq(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5, chunk_size=10000):
        """
        Retrieve top_k most similar vectors using IVF without PQ (brute-force within clusters).
        Memory-efficient: processes candidates in chunks instead of loading all into RAM.
        
        Search algorithm:
        1. Find nprobe nearest IVF clusters
        2. Get candidate vector IDs from those clusters
        3. Compute exact distances to candidates in chunks
        4. Return top_k nearest vectors
        """
        
        self.load_index(use_pq=False)
        
        query = query.flatten() / np.linalg.norm(query)
        
        cluster_ids = self.ivf.search_clusters(query, self.nprobe)
        candidate_ids = self.ivf.get_candidate_ids(cluster_ids)
        
        if len(candidate_ids) == 0:
            print("Warning: No candidates found in IVF search")
            return []
        
        num_records = self._get_num_records()
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        sort_indices = np.argsort(candidate_ids)
        sorted_candidate_ids = candidate_ids[sort_indices]
        
        import heapq
        top_k_heap = []
        
        for chunk_start in range(0, len(sorted_candidate_ids), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sorted_candidate_ids))
            chunk_ids = sorted_candidate_ids[chunk_start:chunk_end]
            
            chunk_vectors = np.array(db_vectors[chunk_ids])
            
            chunk_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
            chunk_norms[chunk_norms == 0] = 1.0
            chunk_vectors_normalized = chunk_vectors / chunk_norms
            chunk_scores = np.dot(chunk_vectors_normalized, query)
            
            for local_idx, score in enumerate(chunk_scores):
                original_idx = sort_indices[chunk_start + local_idx]
                if len(top_k_heap) < top_k:
                    heapq.heappush(top_k_heap, (score, original_idx))
                elif score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (score, original_idx))
        
        top_k_heap.sort(reverse=True)
        result_ids = [candidate_ids[idx] for _, idx in top_k_heap]
        
        return result_ids
    
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
            np.save(f"{self.index_path}/{db_size_str}_inverted_ids.npy", self.ivf.inverted_ids.astype(np.int32))
            np.save(f"{self.index_path}/{db_size_str}_inverted_offsets.npy", self.ivf.inverted_offsets.astype(np.int32))
            np.save(f"{self.index_path}/{db_size_str}_opq_rotation.npy", self.opq.R)
            np.save(f"{self.index_path}/{db_size_str}_pq_codebooks.npy", self.pq.codebooks)
            
            self.pq_codes = np.memmap(f"{self.index_path}/{db_size_str}_pq_codes.dat", dtype=np.uint8, mode='w+', shape=(num_records, self.M))
            
            metadata = np.array([self.M, self.Ks, self.num_clusters, self.nprobe], dtype=np.int32)
            np.save(os.path.join(self.index_path, f"{db_size_str}_metadata.npy"), metadata)
            
            for start in range(0, num_records, self.batch_size):
                end = min(start + self.batch_size, num_records)
                batch_vectors = vectors[start:end]
                batch_vectors = self.opq.transform(batch_vectors)
                codes_batch = self.pq.encode(batch_vectors, batch_size=self.batch_size)
                self.pq_codes[start:end] = codes_batch
                
            self.pq_codes.flush()

            print("Re-arranging PQ codes to match IVF clusters...")

            # 1. Create a READ-ONLY view of the unsorted codes you just wrote
            unsorted_codes = np.memmap(
                os.path.join(self.index_path, f"{db_size_str}_pq_codes.dat"), 
                dtype=np.uint8, mode='r', shape=(num_records, self.M)
            )

            # 2. Create a new file for SORTED codes
            sorted_codes_path = os.path.join(self.index_path, f"{db_size_str}_pq_codes_sorted.dat")
            sorted_codes = np.memmap(
                sorted_codes_path, 
                dtype=np.uint8, mode='w+', shape=(num_records, self.M)
            )

            # 3. Write codes in the order of inverted_ids
            # Doing this in batches to respect RAM
            batch_size = 100_000 # Adjust based on M
            for i in range(0, num_records, batch_size):
                # Get a batch of the SORTED IDs (which represent the cluster order)
                batch_ids = self.ivf.inverted_ids[i : i + batch_size]
                
                # Read those specific codes using fancy indexing (This is slow, but we only do it once!)
                # Ideally, you'd do this smarter, but for building, this works.
                sorted_codes[i : i + batch_size] = unsorted_codes[batch_ids]

            sorted_codes.flush()

            # 4. Point self.pq_codes to this new sorted file for the rest of the runtime
            self.pq_codes = sorted_codes

            # remove the unsorted codes file and rename the sorted one
            os.remove(os.path.join(self.index_path, f"{db_size_str}_pq_codes.dat"))
            os.rename(sorted_codes_path, os.path.join(self.index_path, f"{db_size_str}_pq_codes.dat"))
            
            print("Index saved to disk.")

    def load_index(self, use_pq=True) -> None:
        """
        Load all needed index components from disk.
        Args:
            use_pq (bool, optional): Whether to load PQ components. Defaults to True.
        
        """
        
        db_size_str = self.db_path.split("_emb_")[0]  
        self.ivf.centroids = np.load(f"{self.index_path}/{db_size_str}_ivf_centroids.npy")
        self.ivf.inverted_offsets = np.load(f"{self.index_path}/{db_size_str}_inverted_offsets.npy")
        self.ivf.inverted_ids = np.load(f"{self.index_path}/{db_size_str}_inverted_ids.npy", mmap_mode="r")
        
        # metadata = np.load(os.path.join(self.index_path, f"{db_size_str}_metadata.npy"))
        # self.M, self.Ks, self.num_clusters, self.nprobe = metadata
        # Loaded in init
        
        if use_pq:
            num_records = self._get_num_records()
            self.opq.R = np.load(f"{self.index_path}/{db_size_str}_opq_rotation.npy")
            self.pq.codebooks = np.load(f"{self.index_path}/{db_size_str}_pq_codebooks.npy")
            self.pq_codes = np.memmap(f"{self.index_path}/{db_size_str}_pq_codes.dat", dtype=np.uint8, mode='r', shape=(num_records, self.M))

