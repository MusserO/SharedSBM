from main_functions import *
from shared_block_optimization import *
from gt_extension import *
from utils import *
from datetime import datetime
import pickle

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

algorithms = [
    (shared_blocks_random, "Random"),
    (optimize_shared_blocks_greedy, "Greedy"),
    (optimize_shared_blocks_ILP, "ILP"),
]

seed = 1
np.random.seed(seed)
gt.seed_rng(seed)

sbm_mean_running_times = []
alg_mean_running_times = []
n_graphs_range = range(2, 8)
n_vertices = 200
n_blocks = 4
n_tests = 5
n_shared_blocks = 2
using_directed_graphs = True
verbose = True
log_file = open("shared_block_algs_running_times_test.log", "w")

hard_time_limit = 30*60 # 30 minutes
soft_time_limit = 5*60 # 5 minutes
time_limit_exceeded = np.repeat(False, len(algorithms))
if hard_time_limit is not None:
    signal.signal(signal.SIGALRM, signal_handler)
for n_graphs in n_graphs_range:
    sbm_running_times = []
    alg_running_times = [np.zeros(n_tests) for _ in range(len(algorithms))]
    print_log(f"Testing {n_graphs} graphs", log_file, verbose)
    for test_index in range(n_tests):
        block_counts = n_graphs * [n_blocks]
        vertex_counts = n_graphs * [n_vertices]
        Gs, Ps, shared_block_sets, vertex_to_block_maps = generate_shared_SBMs(n_graphs, block_counts, vertex_counts, n_shared_blocks, using_directed_graphs)
        sbm_params = {"Gs": Gs, "block_counts": block_counts}
        (_, inferred_block_assignments), running_time = time_and_run_function(fit_SBMs, "fit_SBMs", results_file=log_file, verbose=verbose, **sbm_params)
        sbm_running_times.append(running_time.total_seconds())
        edge_count_matrices, missing_edges_matrices = calculate_edge_counts_and_missing_edges(inferred_block_assignments, Gs, using_directed_graphs)
        inferred_Ps = compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices)
        inferred_block_counts = [len(np.unique(b)) for b in inferred_block_assignments]
        params = {"n_shared_blocks": n_shared_blocks, "inferred_Ps": inferred_Ps, "inferred_block_counts": inferred_block_counts,
                  "edge_count_matrices": edge_count_matrices, "missing_edges_matrices": missing_edges_matrices}
        for i, (algorithm, algorithm_name) in enumerate(algorithms):
            running_time, time_limit_exceeded[i] = test_runtime_with_hard_time_limit(algorithm, algorithm_name, params, time_limit_exceeded[i], hard_time_limit, log_file, verbose)
            alg_running_times[i][test_index] = running_time
        print_log("", log_file, verbose)
    if soft_time_limit is not None:
        for i in range(len(algorithms)):
            time_limit_exceeded[i] = (alg_running_times[i] > soft_time_limit).any()
    sbm_mean_running_times.append((n_graphs, np.mean(sbm_running_times)))
    alg_mean_running_times.append((n_graphs, np.mean(alg_running_times, axis=1)))
    print_log(f"{n_graphs} - mean running time over {n_tests} runs: SBM: {np.mean(sbm_running_times)}, algorithms: {np.mean(alg_running_times, axis=1)}", log_file, verbose=True)
    print_log("", log_file, verbose)
if log_file is not None:
    log_file.close()

with open("shared_block_alg_mean_running_times.pkl", "wb") as pkl_file:
    pickle.dump(alg_mean_running_times, pkl_file)
