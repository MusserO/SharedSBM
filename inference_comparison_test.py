from main_functions import *
from shared_block_optimization import *
from gt_extension import *
from utils import *
from datetime import datetime
import pickle

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def fit_shared_SBM_twostep(Gs, block_counts, n_shared_blocks, shared_block_alg=optimize_shared_blocks_ILP, n_iter=1, verbose=0):
    states, inferred_block_assignments = fit_SBMs(Gs, block_counts, n_shared_blocks=n_shared_blocks)
    inferred_block_counts = get_block_counts(inferred_block_assignments)
    edge_count_matrices, missing_edges_matrices = calculate_edge_counts_and_missing_edges(inferred_block_assignments, Gs, using_directed_graphs)
    inferred_Ps = compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices)
    # find which blocks should be shared and reorder so that shared blocks are first and have matching indices
    _, inferred_shared_blocks, _ = shared_block_alg(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices)
    relabel_shared_blocks_first(states, inferred_shared_blocks)
    states, inferred_block_assignments = fit_shared_SBM(Gs, block_counts, n_shared_blocks, states=states, n_iter=n_iter, verbose=verbose)
    return states, inferred_block_assignments

def fit_single_SBMs_twostep(Gs, block_counts, n_shared_blocks=None, n_iter=1, verbose=0):
    states, _ = fit_SBMs(Gs, block_counts, n_shared_blocks=n_shared_blocks)
    states, inferred_block_assignments = fit_SBMs_bernoulli(Gs, block_counts, states=states, n_iter=n_iter, verbose=verbose)
    return states, inferred_block_assignments

total_start_time = datetime.now()

n_graphs = 3
block_counts = [5] * n_graphs
vertex_counts = [500] * n_graphs
n_shared_blocks = 3
using_directed_graphs = True

seed = 1
np.random.seed(seed)
gt.seed_rng(seed)

n_steps = 500
verbose = True
n_tests = 10

results_file_name = "inference_comparison_test"
if results_file_name is not None:
    log_file = open(results_file_name+".log", "w")

inference_algorithms = [
    (fit_SBMs_bernoulli, "bernoulli single SBMs", n_steps),
    (fit_shared_SBM, "bernoulli shared SBM", n_steps),
    (fit_SBMs, "poisson multilevel single SBMs", None),
    (fit_single_SBMs_twostep, "two-step single SBMs", n_steps),
    (fit_shared_SBM_twostep, "two-step shared SBMs", n_steps),
]
shared_block_optimization_algs = [
    (shared_blocks_random, "Random"),
    (firsts_as_shared_blocks, "Firsts"),
    (optimize_shared_blocks_greedy, "Greedy"),
    (optimize_shared_blocks_ILP, "ILP"),
]
mean_partition_ARIs = [[] for _ in range(len(inference_algorithms))]
inference_running_times = [[] for _ in range(len(inference_algorithms))]
single_log_likelihoods = [[] for _ in range(len(inference_algorithms))]
single_BICs = [[] for _ in range(len(inference_algorithms))]
single_edge_P_MAEs = [[] for _ in range(len(inference_algorithms))]

shared_block_mean_ARIs = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]
total_running_times = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]
shared_log_likelihoods = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]
shared_BICs = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]
shared_edge_P_MAEs = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]

for i in range(n_tests):
    Gs, Ps, shared_block_sets, block_assignments = generate_shared_SBMs(n_graphs, block_counts, vertex_counts, n_shared_blocks, using_directed_graphs, seed=seed+i)
    for inference_alg_i, (inference_algorithm, inference_name, n_iter) in enumerate(inference_algorithms):
        np.random.seed(seed+i)
        gt.seed_rng(seed+i)
        inference_start_time = datetime.now()
        if inference_name == "poisson multilevel single SBMs":
            states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks)
        else:
            states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks, n_iter=n_iter)

        inferred_block_counts = get_block_counts(inferred_block_assignments)
        inference_running_time = datetime.now() - inference_start_time
        partition_ARIs = compute_ARIs(inferred_block_assignments, block_assignments, verbose=False)
        mean_partition_ARIs[inference_alg_i].append(np.mean(partition_ARIs))
        inference_running_times[inference_alg_i].append(inference_running_time)

        edge_count_matrices, missing_edges_matrices = calculate_edge_counts_and_missing_edges(inferred_block_assignments, Gs, using_directed_graphs)
        inferred_Ps = compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices)
        log_likelihood, BIC = log_likelihood_and_BIC(inferred_Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs, n_shared_blocks=0)
        edge_P_MAE = compute_edge_probability_MAE(Ps, inferred_Ps, vertex_counts, using_directed_graphs, block_assignments, inferred_block_assignments)
        single_log_likelihoods[inference_alg_i].append(log_likelihood)
        single_BICs[inference_alg_i].append(BIC)
        single_edge_P_MAEs[inference_alg_i].append(edge_P_MAE)
        print_log(f"{inference_name} done in: {inference_running_time}, partition ARIs: {partition_ARIs}, log-likelihood: {log_likelihood}, BIC: {BIC}, edge P MAE': {edge_P_MAE}", log_file, verbose)
        for shared_alg_i, (shared_block_alg, shared_block_alg_name) in enumerate(shared_block_optimization_algs):
            shared_block_start_time = datetime.now()
            inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets = shared_block_alg(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices)
            shared_block_running_time = datetime.now() - shared_block_start_time
            total_running_time = inference_running_time + shared_block_running_time
            ARIs = compare_shared_blocks_ARI(Gs, inferred_block_assignments, inferred_shared_block_sets, block_assignments, shared_block_sets)
            log_likelihood, BIC = log_likelihood_and_BIC(inferred_shared_Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs, n_shared_blocks)
            edge_P_MAE = compute_edge_probability_MAE(Ps, inferred_shared_Ps, vertex_counts, using_directed_graphs, block_assignments, inferred_block_assignments)
            shared_block_mean_ARIs[inference_alg_i][shared_alg_i].append(np.mean(ARIs, axis=0))
            total_running_times[inference_alg_i][shared_alg_i].append(total_running_time)
            shared_log_likelihoods[inference_alg_i][shared_alg_i].append(log_likelihood)
            shared_BICs[inference_alg_i][shared_alg_i].append(BIC)
            shared_edge_P_MAEs[inference_alg_i][shared_alg_i].append(edge_P_MAE)
            print_log(f"{inference_name} with {shared_block_alg_name} shared blocks done in: {total_running_time}, ARIs: {list(ARIs)}, log-likelihood: {log_likelihood}, BIC: {BIC}, edge P MAE: {edge_P_MAE}\n", log_file, verbose)
for inference_alg_i, (_, inference_alg_name, _) in enumerate(inference_algorithms):
    print_log(f"{inference_alg_name}: mean over {n_tests} runs: partition ARI: {np.mean(mean_partition_ARIs[inference_alg_i]):.2f}, running time (s): {np.mean(inference_running_times[inference_alg_i]).total_seconds():.2f}, log-likelihood: {np.mean(single_log_likelihoods[inference_alg_i]):.2f}, BIC: {np.mean(single_BICs[inference_alg_i]):.2f}, edge P MAE: {np.mean(single_edge_P_MAEs[inference_alg_i]):.2f}", log_file, verbose=True)
print_log("\n", log_file, verbose=True)
for inference_alg_i, (_, inference_alg_name, _) in enumerate(inference_algorithms):
    for shared_alg_i, (_, shared_block_alg_name) in enumerate(shared_block_optimization_algs):
        print_log(f"{inference_alg_name} with {shared_block_alg_name} shared blocks: mean over {n_tests} runs: shared ARIs: {np.mean(shared_block_mean_ARIs[inference_alg_i][shared_alg_i], axis=0)}, running time (s): {np.mean(total_running_times[inference_alg_i][shared_alg_i]).total_seconds():.2f}, log likelihood: {np.mean(shared_log_likelihoods[inference_alg_i][shared_alg_i]):.2f}, BIC: {np.mean(shared_BICs[inference_alg_i][shared_alg_i]):.2f}, edge P MAE: {np.mean(shared_edge_P_MAEs[inference_alg_i][shared_alg_i]):.2f}", log_file, verbose=True)

if results_file_name is not None:
    results = inference_algorithms, shared_block_optimization_algs, mean_partition_ARIs, inference_running_times, single_log_likelihoods, single_BICs, single_edge_P_MAEs, shared_block_mean_ARIs, total_running_times, shared_log_likelihoods, shared_BICs, shared_edge_P_MAEs
    with open(results_file_name+".pkl", "wb") as pkl_file:
        pickle.dump(results, pkl_file)

print_log(f"Total running time: {datetime.now() - total_start_time}", log_file, verbose=True)
