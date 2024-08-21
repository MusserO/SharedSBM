from main_functions import *
from shared_block_optimization import *
from gt_extension import *
from utils import *
from datetime import datetime
import matplotlib.pyplot as plt

import pickle
import os

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


dataset_paths = [
    #'../data/wikipedia_link_fi/out.wikipedia_link_fi', # nodes: 684613, edges: 14658697, average degree: 42.82, communities: 167 (took 0:08:16.38) to find)
    '../data/wikipedia_link_zh/out.wikipedia_link_zh', # nodes: 1786381, edges: 72614837, average degree: 81.30, communities: 272 (took 0:46:09.26) to find)
    #'../data/wikipedia_link_de/out.wikipedia_link_de', # nodes: 3603726, edges: 96865851, average degree: 53.76
    '../data/wikipedia_link_en/out.wikipedia_link_en', # nodes: 13593032, edges: 437217424, average degree: 64.33
]
use_communities = False

Gs = []
community_lists = []
for file_path in dataset_paths:
    print(f"Reading dataset: {file_path}")
    if os.path.exists(file_path + '.pickle'):
        G = pickle.load(open(file_path + '.pickle', "rb"))
    else:
        print("Dataset not processed...")
        continue
    if use_communities:
        if os.path.exists(file_path + '_communities.pickle'):
            communities = pickle.load(open(file_path + '_communities.pickle', "rb"))
        else:
            # Community detection
            start = datetime.now()
            communities = nx.community.louvain_communities(G)
            print(f"Communities: {len(communities)}")
            print(f"Community detection took {(datetime.now() - start)}")
            with open(file_path + '_communities.pickle', 'wb') as file:
                pickle.dump(communities, file)
        community_lists.append(communities)

    Gs.append(G)

using_directed_graphs = Gs[0].is_directed()
Gs = convert_to_gt_graphs(Gs, using_directed_graphs)

seed = 123
verbose = True

#block_counts = [None] * len(Gs)
block_counts = [20] * len(Gs)
#block_counts = [len(c) for c in community_lists]
n_steps = 100
n_shared_blocks = 0

inference_algorithms = [
    #(fit_SBMs_bernoulli, "bernoulli single SBMs", n_steps),
    (fit_shared_SBM, "bernoulli shared SBM", n_steps),
    #(fit_SBMs, "poisson multilevel single SBMs", None),
    #(fit_single_SBMs_twostep, "two-step single SBMs", n_steps),
    #(fit_shared_SBM_twostep, "two-step shared SBMs", n_steps),
]
shared_block_optimization_algs = [
    #(shared_blocks_random, "Random"),
    #(firsts_as_shared_blocks, "Firsts"),
    (optimize_shared_blocks_greedy, "Greedy"),
    #(optimize_shared_blocks_ILP, "ILP"),
]
datasets_languages = "_".join([file_path[-2:] for file_path in dataset_paths])
results_file_name = "wiki_networks_results" + "_"+datasets_languages

total_start_time = datetime.now()

if seed is not None:
    np.random.seed(seed)
    gt.seed_rng(seed)

if results_file_name is not None:
    log_file = open(results_file_name+".log", "w")

inference_running_times = [[] for _ in range(len(inference_algorithms))]
single_log_likelihoods = [[] for _ in range(len(inference_algorithms))]
single_BICs = [[] for _ in range(len(inference_algorithms))]

total_running_times = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]
shared_log_likelihoods = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]
shared_BICs = [[[] for _ in range(len(shared_block_optimization_algs))] for _ in range(len(inference_algorithms))]

for inference_alg_i, (inference_algorithm, inference_name, n_iter) in enumerate(inference_algorithms):
    if seed is not None:
        np.random.seed(seed)
        gt.seed_rng(seed)
    inference_start_time = datetime.now()
    if inference_algorithm == fit_SBMs:
        multilevel_mcmc_args = dict(parallel=True)
        states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks, multilevel_mcmc_args=multilevel_mcmc_args)
    elif inference_algorithm == fit_shared_SBM_twostep:
        states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks, using_directed_graphs=using_directed_graphs)
    else:
        states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks, n_iter=n_iter)

    inferred_block_counts = get_block_counts(inferred_block_assignments)
    if block_counts[0] is None:
        print(f"inferred_block_counts: {inferred_block_counts}")
    inference_running_time = datetime.now() - inference_start_time
    inference_running_times[inference_alg_i].append(inference_running_time)

    edge_count_matrices, missing_edges_matrices = calculate_edge_counts_and_missing_edges(inferred_block_assignments, Gs, using_directed_graphs)
    inferred_Ps = compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices)
    log_likelihood, BIC = log_likelihood_and_BIC(inferred_Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs, n_shared_blocks=0)
    single_log_likelihoods[inference_alg_i].append(log_likelihood)
    single_BICs[inference_alg_i].append(BIC)
    print_log(f"{inference_name} done in: {inference_running_time}, log-likelihood: {log_likelihood}, BIC: {BIC}", log_file, verbose)
    #results = states, inferred_block_assignments
    results = inferred_block_assignments
    with open(results_file_name+"_"+inference_name+".pkl", "wb") as pkl_file:
        pickle.dump(results, pkl_file)
    for shared_alg_i, (shared_block_alg, shared_block_alg_name) in enumerate(shared_block_optimization_algs):
        shared_block_start_time = datetime.now()
        inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets = shared_block_alg(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices)
        shared_block_running_time = datetime.now() - shared_block_start_time
        total_running_time = inference_running_time + shared_block_running_time
        log_likelihood, BIC = log_likelihood_and_BIC(inferred_shared_Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs, n_shared_blocks)
        total_running_times[inference_alg_i][shared_alg_i].append(total_running_time)
        shared_log_likelihoods[inference_alg_i][shared_alg_i].append(log_likelihood)
        shared_BICs[inference_alg_i][shared_alg_i].append(BIC)
        print_log(f"{inference_name} with {shared_block_alg_name} shared blocks done in: {total_running_time}, log-likelihood: {log_likelihood}, BIC: {BIC}\n", log_file, verbose)
        results = inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets
        with open(results_file_name+"_"+inference_name+"_"+shared_block_alg_name+".pkl", "wb") as pkl_file:
            pickle.dump(results, pkl_file)

if results_file_name is not None:
    results = inference_algorithms, shared_block_optimization_algs, inference_running_times, single_log_likelihoods, single_BICs, total_running_times, shared_log_likelihoods, shared_BICs
    with open(results_file_name+".pkl", "wb") as pkl_file:
        pickle.dump(results, pkl_file)

print_log(f"Total running time: {datetime.now() - total_start_time}", log_file, verbose=True)
