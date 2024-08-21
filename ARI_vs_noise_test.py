from main_functions import *
from shared_block_optimization import *
from gt_extension import *
from utils import *
from datetime import datetime
import pickle

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def add_noise_to_block_assignments(block_assignments, block_counts, noise_level):
    noisy_block_assignments = []
    for k in range(len(block_assignments)):
        noisy_block_assignments.append(np.copy(block_assignments[k]))
        noisy_block_assignments[k][np.where(np.random.rand(len(block_assignments[k])) < noise_level)] = np.random.randint(block_counts[k])
    return noisy_block_assignments


seed = 1
np.random.seed(seed)
gt.seed_rng(seed)

algorithms = [
    (shared_blocks_random, "Random"),
    (optimize_shared_blocks_greedy, "Greedy"),
    (optimize_shared_blocks_ILP, "ILP"),
]

n_graphs = 2
block_counts = (4, 4)
vertex_counts = (200, 200)
n_shared_blocks = 2
using_directed_graphs = True
n_tests = 500
total_running_times = [[] for _ in range(len(algorithms))]
plotting = False
verbose = True
noise_level_range = np.arange(0, 0.2+1e-9, 0.01)
noise_mean_ARIs = [np.zeros((len(noise_level_range), 2)) for _ in range(len(algorithms))]
for noise_level_index, noise_level in enumerate(noise_level_range):
    shared_block_mean_ARIs = [[] for _ in range(len(algorithms))]
    shared_log_likelihoods = [[] for _ in range(len(algorithms))]
    if verbose:
        print(f"Noise level:{noise_level}")
    for i in range(n_tests):
        Gs, Ps, shared_block_sets, block_assignments = generate_shared_SBMs(n_graphs, block_counts, vertex_counts, n_shared_blocks, using_directed_graphs)
        # Add noise to block_assignments
        inferred_block_assignments = add_noise_to_block_assignments(block_assignments, block_counts, noise_level)
        inferred_block_counts = get_block_counts(inferred_block_assignments)
        edge_count_matrices, missing_edges_matrices = calculate_edge_counts_and_missing_edges(inferred_block_assignments, Gs, using_directed_graphs)
        inferred_Ps = compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices)
        for i, algorithm in enumerate(algorithms):
            alg_function, alg_name = algorithm
            inference_start_time = datetime.now()
            inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets = alg_function(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices)
            total_running_time = datetime.now() - inference_start_time
            ARIs = compare_shared_blocks_ARI(Gs, inferred_block_assignments, inferred_shared_block_sets, block_assignments, shared_block_sets)
            #log_likelihood = compute_log_likelihood(inferred_shared_Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs)
            if verbose:
                print(f"{alg_name} done in: {total_running_time}, ARIs: {list(ARIs)}")
            shared_block_mean_ARIs[i].append(np.mean(ARIs, axis=0))
            total_running_times[i].append(total_running_time)
            #log_likelihoods[i].append(log_likelihood)
        if verbose:
            print()
    for i in range(len(algorithms)):
        noise_mean_ARIs[i][noise_level_index] = np.mean(shared_block_mean_ARIs[i], axis=0)
    for i, (_, alg_name) in enumerate(algorithms):
        print(f"{alg_name}: mean over {n_tests} runs: ARIs: {np.mean(shared_block_mean_ARIs[i], axis=0)}, running time (s): {np.mean(total_running_times[i]).total_seconds():.3f}")
        print()

import matplotlib.pyplot as plt

for ARI_index in range(len(noise_mean_ARIs[0][0])):
    ARI_values = [[ARI[ARI_index] for ARI in noise_mean_ARIs[i]] for i in range(len(algorithms))]

    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        _, alg_name = algorithm
        plt.plot(noise_level_range, ARI_values[i], marker='o', label=alg_name)

    plt.xlabel('Noise level')
    plt.ylabel('ARI')
    plt.title(f'ARI vs noise level for ARI {ARI_index+1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.0)
    plt.savefig(f'ARI vs noise level for ARI {ARI_index+1}.pdf')
    #plt.show()
