from main_functions import *
from shared_block_optimization import *
from gt_extension import *
from utils import *
from datetime import datetime
import pickle

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

seed = 123
np.random.seed(seed)
gt.seed_rng(seed)

n_graphs = 2
block_counts = n_graphs * [4]
n_shared_blocks = 2
using_directed_graphs = True
n_steps = 500
n_tests = 5

verbose = True
log_file = open("inference_algs_running_times_test.log", "w")

inference_algorithms = [
    (fit_SBMs_bernoulli, "Single SBMs", n_steps),
    (fit_shared_SBM, "Shared SBM", n_steps),
    (fit_SBMs, "Multilevel single SBMs", None),
    (fit_single_SBMs_twostep, "Two-step single SBMs", n_steps),
    (fit_shared_SBM_twostep, "Two-step shared SBMs", n_steps),
]

inference_running_times = [[] for _ in range(len(inference_algorithms))]
n_edges_lists = []
n_vertices_range = range(100, 1901, 300)
for n_vertices in n_vertices_range:
    print_log(f"Number of vertices: {n_vertices}", log_file, verbose=True)
    sbm_running_times = []
    for i in range(n_tests):
        vertex_counts = n_graphs * [n_vertices]
        Gs, Ps, shared_block_sets, vertex_to_block_maps = generate_shared_SBMs(n_graphs, block_counts, vertex_counts, n_shared_blocks, using_directed_graphs, seed=seed+i)
        n_edges = [G.num_edges() for G in Gs]
        n_edges_lists.append(n_edges)
        print_log(f"    Numbers of edges: {n_edges}", log_file, verbose=True)

        for inference_alg_i, (inference_algorithm, inference_name, n_iter) in enumerate(inference_algorithms):
            np.random.seed(seed+i)
            gt.seed_rng(seed+i)
            inference_start_time = datetime.now()
            if inference_algorithm == fit_SBMs:
                states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks)
            else:
                states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks, n_iter=n_iter)
            inference_running_time = datetime.now() - inference_start_time
            inference_running_times[inference_alg_i].append(inference_running_time)
            print_log(f"    Fitting SBMs done in: {inference_running_time}", log_file, verbose=True)

    for inference_alg_i, (_, inference_alg_name, _) in enumerate(inference_algorithms):
        print_log(f"    {inference_alg_name}: mean over {n_tests} runs: running time (s): {np.mean(inference_running_times[inference_alg_i]).total_seconds():.2f}", log_file, verbose=True)

if log_file is not None:
    log_file.close()

with open("inference_running_times.pkl", "wb") as pkl_file:
    results = inference_algorithms, inference_running_times, n_edges_lists, n_vertices_range, n_tests
    pickle.dump(results, pkl_file)
