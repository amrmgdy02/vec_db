import numpy as np
import os
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
from memory_profiler import memory_usage
import gc

QUERY_SEED_NUMBER = 10

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, queries, top_k, actual_ids, num_runs):
    """
    Run queries on the database and record results for each query.

    Parameters:
    - db: Database instance to run queries on.
    - queries: List of query vectors.
    - top_k: Number of top results to retrieve.
    - actual_ids: List of actual results to evaluate accuracy.
    - num_runs: Number of query executions to perform for testing.

    Returns:
    - List of Result
    """
    global results
    results = []
    for i in range(num_runs):
        tic = time.time()
        db_ids = db.retrieve(queries[i], top_k)
        toc = time.time()
        run_time = toc - tic
        results.append(Result(run_time, top_k, db_ids, actual_ids[i]))
    return results

def memory_usage_run_queries(args):
    """
    Run queries and measure memory usage during the execution.

    Parameters:
    - args: Arguments to be passed to the run_queries function.

    Returns:
    - results: The results of the run_queries.
    - memory_diff: The difference in memory usage before and after running the queries.
    """
    global results
    mem_before = max(memory_usage())
    mem = memory_usage(proc=(run_queries, args, {}), interval = 1e-3)
    return results, max(mem) - mem_before

def evaluate_result(results: List[Result]):
    """
    Evaluate the results based on accuracy and runtime.
    Scores are negative. So getting 0 is the best score.

    Parameters:
    - results: A list of Result objects

    Returns:
    - avg_score: The average score across all queries.
    - avg_runtime: The average runtime for all queries.
    """
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retrieving number not equal to top_k, score will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_actual_ids_first_k(actual_sorted_ids, k):
    """
    Retrieve the IDs from the sorted list of actual IDs.
    actual IDs has the top_k for the 20 M database but for other databases we have to remove the numbers higher than the max size of the DB.

    Parameters:
    - actual_sorted_ids: A list of lists containing the sorted actual IDs for each query.
    - k: The DB size.

    Returns:
    - List of lists containing the actual IDs for each query for this DB.
    """
    return [[id for id in actual_sorted_ids_one_q if id < k] for actual_sorted_ids_one_q in actual_sorted_ids]

def eval():
    needed_top_k = 10000
    rng = np.random.default_rng(QUERY_SEED_NUMBER)
    query1 = rng.random((1, DIMENSION), dtype=np.float32)
    query2 = rng.random((1, DIMENSION), dtype=np.float32)
    query3 = rng.random((1, DIMENSION), dtype=np.float32)
    query_dummy = rng.random((1, DIMENSION), dtype=np.float32)

    db = VecDB(database_file_path="OpenSubtitles_en_20M_emb_64.dat", new_db=False)

    vectors = db.get_all_rows()

    # Add epsilon to avoid division by zero
    eps = 1e-10
    actual_sorted_ids_20m_q1 = np.argsort(vectors.dot(query1.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query1) + eps), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
    gc.collect()
    actual_sorted_ids_20m_q2 = np.argsort(vectors.dot(query2.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query2) + eps), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
    gc.collect()
    actual_sorted_ids_20m_q3 = np.argsort(vectors.dot(query3.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query3) + eps), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
    gc.collect()

    queries = [query1, query2, query3]
    actual_sorted_ids_20m = [actual_sorted_ids_20m_q1, actual_sorted_ids_20m_q2, actual_sorted_ids_20m_q3]

    # No more need to the actual vectors so delete it
    del vectors
    gc.collect()

    to_print_arr = []

    database_info = {
        "1M": {
            "database_file_path": "OpenSubtitles_en_1M_emb_64.dat",
            "index_file_path": "OpenSubtitles_en_1M_index_64",
            "size": 10**6,
            #"M": 16,     
            "nprobe": 64
        },
        "10M": {
            "database_file_path": "OpenSubtitles_en_10M_emb_64.dat",
            "index_file_path": "OpenSubtitles_en_10M_index_64",
            "size": 10 * 10**6,
            #"M": 16,      
            "nprobe": 64
        },
        # "15M": {
        #     "database_file_path": "OpenSubtitles_en_15M_emb_64.dat",
        #     "index_file_path": "OpenSubtitles_en_15M_index_64",
        #     "size": 15 * 10**6,
        #     "M": 4,
        #     "nprobe": 64
        # },
        "20M": {
            "database_file_path": "OpenSubtitles_en_20M_emb_64.dat",
            "index_file_path": "OpenSubtitles_en_20M_index_64",
            "size": 20 * 10**6,
            #"M": 16, 
            "nprobe": 128
        }
    }

    for db_name, info in database_info.items():
        # UPDATE: Pass M and nprobe to VecDB
        db = VecDB(
            database_file_path=info["database_file_path"], 
            index_file_path=info["index_file_path"], 
            new_db=False,
            #M=info["M"], 
            nprobe=info["nprobe"]  # Pass nprobe for consistent recall
        )
        
        actual_ids = get_actual_ids_first_k(actual_sorted_ids_20m, info["size"])
        # Make a dummy run query to make everything fresh and loaded (wrap up)
        res = run_queries(db, query_dummy, 5, actual_ids, 1)
        # actual runs to evaluate
        res, mem = memory_usage_run_queries((db, queries, 5, actual_ids, 3))
        eval_result = evaluate_result(res)
        to_print = f"{db_name}\tscore\t{eval_result[0]}\ttime\t{eval_result[1]:.2f}\tRAM\t{mem:.2f} MB"
        print(to_print)
        to_print_arr.append(to_print)
        del db
        del actual_ids
        del res
        del mem
        del eval_result
        gc.collect()

def build_indices():
    database_files = {
            "OpenSubtitles_en_1M_emb_64.dat": {
                "M": 16,
                "Ks": 256,
                "nprobe": 64,
                "n_clusters": 1024,
                "S_ivf": 131_072
            }, 
            # "OpenSubtitles_en_10M_emb_64.dat": {
            #     "M": 4,
            #     "Ks": 256,
            #     "nprobe": 64,
            #     "n_clusters": 4096,
            #     "S_ivf": 262_144
            # }, 
            # "OpenSubtitles_en_15M_emb_64.dat": {
            #     "M": 4,
            #     "Ks": 256,
            #     "nprobe": 64,
            #     "n_clusters": 8192,
            #     "S_ivf": 393_216
            # }, 
            "OpenSubtitles_en_20M_emb_64.dat": {
                "M": 4,
                "Ks": 256,
                "nprobe": 128,
                "n_clusters": 16384,
                "S_ivf": 655_360
            }
        }
    for db_file in database_files:
        index_file = db_file.replace("_emb_", "_index_").replace(".dat", "")
        db = VecDB(database_file_path=db_file, index_file_path=index_file, new_db=False, 
                   M=database_files[db_file]["M"],
                   Ks=database_files[db_file]["Ks"],
                   nprobe=database_files[db_file]["nprobe"],
                   num_clusters=database_files[db_file]["n_clusters"],
                   S_ivf=database_files[db_file]["S_ivf"])
        db._build_index()
        del db
        gc.collect()

"Each database size is a subset of the 20M database."
import os
import gc
from itertools import islice

# Define constants
DIMENSION = 64
ELEMENT_SIZE = 4 # float32 = 4 bytes

def generate_dbs():
    _20M_db_file = "20M_emb_64.dat"
    _20M_txt_file = "20M_sorted.txt"

    sizes = {
        "1M_emb_64.dat": 10**6,
        "10M_emb_64.dat": 10 * 10**6,
        "15M_emb_64.dat": 15 * 10**6
    }

    txt_files = {
        "1M_emb_64.dat": "1M_sorted.txt",
        "10M_emb_64.dat": "10M_sorted.txt",
        "15M_emb_64.dat": "15M_sorted.txt"
    }
    
    # 1. GENERATE VECTOR FILES (.dat)
    # ------------------------------------------------
    for db_file, size in sizes.items():
        print(f"Generating vector file: {db_file}...")
        
        num_bytes = size * DIMENSION * ELEMENT_SIZE
        chunk_size = 10 * 1024 * 1024  # Increase chunk to 10MB for faster disk I/O
        
        with open(_20M_db_file, 'rb') as f_in, open(db_file, 'wb') as f_out:
            bytes_copied = 0
            while bytes_copied < num_bytes:
                # Ensure we don't read past the required size
                bytes_to_read = min(chunk_size, num_bytes - bytes_copied)
                chunk = f_in.read(bytes_to_read)
                
                if not chunk:
                    break 
                
                f_out.write(chunk)
                bytes_copied += len(chunk)

    # 2. GENERATE TEXT FILES (.txt)
    # ------------------------------------------------
    for db_file, size in sizes.items():
        target_txt = txt_files[db_file]
        print(f"Generating text file: {target_txt}...")

        with open(_20M_txt_file, 'r', encoding='utf-8') as f_in, \
             open(target_txt, 'w', encoding='utf-8') as f_out:
            
            # Use islice to efficiently grab the first 'size' lines
            # This is much faster than a manual for-loop with readline()
            f_out.writelines(islice(f_in, size))

    gc.collect()
    print("Generation complete.")

if __name__ == "__main__":
    # generate_dbs()
    #build_indices()
    eval()