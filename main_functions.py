import graph_tool.all as gt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from utils import *

def generate_shared_SBMs(n_graphs, block_counts, vertex_counts, n_shared_blocks, using_directed_graphs, using_gt_graph=True, seed=None):
    Gs = []
    Ps = []
    shared_block_sets = []
    vertex_to_block_maps = []

    if seed is not None:
        np.random.seed(seed)

    # Generate random graphs
    for k in range(n_graphs):
        n_vertices = vertex_counts[k]
        n_blocks = block_counts[k]

        # Split vertices randomly into random size blocks
        block_split_index_choices = sorted(np.random.choice(np.arange(1,n_vertices), n_blocks-1, replace=False))
        node_permutation = np.random.permutation(n_vertices)
        blocks = np.split(node_permutation, block_split_index_choices)

        # Generate random edge probabilities between blocks
        a, b = 0.5, 1.0
        P = np.random.beta(a, b, size=(n_blocks, n_blocks))
        if not using_directed_graphs:
            P = (P + P.T)/2 # Make p symmetric

        # Choose which blocks are shared for each graph
        shared_block_sets.append(np.random.choice(np.arange(n_blocks), n_shared_blocks, replace=False))
        if k > 0:
            # Share the same edge probabilities for the shared blocks
            P[np.ix_(shared_block_sets[k], shared_block_sets[k])] = Ps[0][np.ix_(shared_block_sets[0], shared_block_sets[0])]
        Ps.append(P)

        # Generate SBM graph
        nx_graph = nx.stochastic_block_model([len(block) for block in blocks], P, seed=seed, directed=True)
        nx_graph = nx.relabel_nodes(nx_graph, dict(zip(np.arange(n_vertices), node_permutation)))
        if using_gt_graph:
            G = gt.Graph(directed=using_directed_graphs)
            G.add_edge_list(nx_graph.edges())
            Gs.append(G)
        else:
            Gs.append(nx_graph)

        vertex_to_block = np.zeros(n_vertices, dtype=int)
        for i, block in enumerate(blocks):
            for v in block:
                vertex_to_block[v] = i
        vertex_to_block_maps.append(vertex_to_block)
    return Gs, Ps, shared_block_sets, vertex_to_block_maps

def is_weighted_graph(nx_graph):
    for _, _, data in nx_graph.edges(data=True):
        if 'weight' in data:
            if data['weight'] != 1.0 and data['weight'] != 0.0:
                return True
        else:
            return False
    return False

def convert_to_gt_graphs(Gs, using_directed_graphs):
    gt_graphs = []
    for nx_graph in Gs:
        G = gt.Graph(directed=using_directed_graphs)
        if is_weighted_graph(nx_graph):
            G.ep.weight = G.new_ep("double")
            weighted_edges = np.array([(e[0], e[1], e[2]['weight']) for e in nx_graph.edges(data=True)])
            G.add_edge_list(weighted_edges, eprops=[G.ep.weight])
        else:
            G.add_edge_list(nx_graph.edges())
        # also include any vertices that don't have any edges
        if G.num_vertices() < nx_graph.number_of_nodes():
            G.add_vertex(n=nx_graph.number_of_nodes() - G.num_vertices())
        gt_graphs.append(G)
    return gt_graphs

def plot_ground_truth_SBMs(Gs, vertex_to_block_maps):
    layouts = []
    for k, G in enumerate(Gs):
        vertex_to_block = vertex_to_block_maps[k]
        # Create colors for the vertices based on their block
        node_colors = [plt.cm.tab10(vertex_to_block[v]) for v in G.get_vertices()]

        plot_color = G.new_vertex_property('vector<double>')
        G.vertex_properties['plot_color'] = plot_color
        for v in G.get_vertices():
            plot_color[v] = node_colors[v]

        layout = gt.sfdp_layout(G)
        layouts.append(layout)

        gt.graph_draw(G, pos=layout, vertex_text=G.vertex_index, vertex_size=20, vertex_fill_color=G.vertex_properties['plot_color'])
    return layouts

def plot_inferred_SBMs(Gs, states, layouts):
    for k, G in enumerate(Gs):
        state = states[k]
        state.draw(pos=layouts[k], vertex_text=G.vertex_index, vertex_size=20)


def fit_multilevel_SBMs(Gs, block_counts=None, n_shared_blocks=None, state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict()):
    states=[]
    inferred_block_assignments = []
    if block_counts is None:
        block_counts = [None]*len(Gs)
    for k, G in enumerate(Gs):
        n_blocks = block_counts[k]
        state_args, entropy_args, multilevel_mcmc_args = fill_default_params(G=G, n_blocks=n_blocks, n_shared_blocks=n_shared_blocks, state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args)
        state = gt.minimize_blockmodel_dl(G, state=state_class, state_args=state_args, multilevel_mcmc_args=multilevel_mcmc_args)

        #gt.mcmc_anneal(state, beta_range=(1, 10), niter=10, mcmc_equilibrate_args=dict(force_niter=10, mcmc_args=dict(allow_vacate=False)))
        #gt.mcmc_equilibrate(state, wait=10, nbreaks=2, mcmc_args=dict(niter=10, allow_vacate=False))

        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        states.append(state)
        inferred_block_assignments.append(b.a)
    return states, inferred_block_assignments

def fit_SBMs(Gs, block_counts=None, n_shared_blocks=None, state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict()):
    return fit_multilevel_SBMs(Gs, block_counts=block_counts, n_shared_blocks=n_shared_blocks, state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args)

def get_block_counts(block_assignments):
    return [len(np.unique(b)) for b in block_assignments]


def _compute_single_ARI(inferred_blocks, true_blocks, k=0, verbose=True):
    ari = adjusted_rand_score(inferred_blocks, true_blocks)
    if verbose:
        n_vertices_in_inferred_blocks = sorted(np.unique(inferred_blocks, return_counts=True)[1])
        n_vertices_in_true_blocks = sorted(np.unique(true_blocks, return_counts=True)[1])
        print(f"Adjusted Rand Index for graph {k}: {ari}")
        print(f"    blocks: {n_vertices_in_inferred_blocks} vs {n_vertices_in_true_blocks}")
    return ari

def compute_ARIs(predicted_blocks, ground_truth_blocks, k=0, verbose=True):
    # Check if the input is an array of arrays or a single array
    if isinstance(predicted_blocks[0], np.ndarray) or isinstance(predicted_blocks[0], list):
        aris = np.zeros(len(predicted_blocks))
        for k in range(len(predicted_blocks)):
            aris[k] = _compute_single_ARI(predicted_blocks[k], ground_truth_blocks[k], k=k, verbose=verbose)
        return aris
    else:
        return _compute_single_ARI(predicted_blocks, ground_truth_blocks, k=k, verbose=verbose)

def calculate_edge_counts_and_missing_edges(block_assignments, Gs, using_directed_graphs):
    edge_count_matrices = []
    missing_edges_matrices = []
    for k, G in enumerate(Gs):
        b = block_assignments[k]
        n_vertices_in_blocks = np.bincount(b)
        n_blocks = len(n_vertices_in_blocks)
        # Count the number of edges between each pair of blocks
        edge_count_matrix = np.zeros((n_blocks, n_blocks))
        if 'weight' in G.ep:
            for (source, target), weight in zip(G.get_edges(), G.ep.weight.fa):
                source_block = b[source]
                target_block = b[target]
                edge_count_matrix[source_block, target_block] += weight
                if not using_directed_graphs and source_block != target_block:
                    edge_count_matrix[target_block, source_block] += weight
        else:
            for edge in G.edges():
                source_block = b[int(edge.source())]
                target_block = b[int(edge.target())]
                edge_count_matrix[source_block, target_block] += 1
                if not using_directed_graphs and source_block != target_block:
                    edge_count_matrix[target_block, source_block] += 1
        edge_count_matrices.append(edge_count_matrix)

        # Compute the maximum number of edges between each pair of blocks
        # Currently this does not take weights into account
        max_edges_between_blocks = np.outer(n_vertices_in_blocks, n_vertices_in_blocks)
        # Adjust the diagonal elements
        if using_directed_graphs:
            np.fill_diagonal(max_edges_between_blocks, n_vertices_in_blocks * (n_vertices_in_blocks - 1))
        else:
            np.fill_diagonal(max_edges_between_blocks, n_vertices_in_blocks * (n_vertices_in_blocks - 1)//2)
        missing_edges_matrix = max_edges_between_blocks - edge_count_matrix
        missing_edges_matrices.append(missing_edges_matrix)
    return edge_count_matrices, missing_edges_matrices

def compute_log_likelihood(Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs=True):
    log_likelihood = 0
    for k in range(len(Ps)):
        for i in range(len(Ps[k])):
            for j in range(0 if using_directed_graphs else i, len(Ps[k][i])):
                log_likelihood += edge_count_matrices[k][i][j] * np.log(Ps[k][i][j]) + missing_edges_matrices[k][i][j] * np.log(1 - Ps[k][i][j])
    return log_likelihood

def log_likelihood_and_BIC(Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs, n_shared_blocks=0):
    log_likelihood = compute_log_likelihood(Ps, edge_count_matrices, missing_edges_matrices, using_directed_graphs)
    n_duplicated_parameters = (len(Ps)-1)*n_shared_blocks*n_shared_blocks
    n_parameters = np.sum([np.prod(P.shape) for P in Ps]) - n_duplicated_parameters
    # set the number of observations as the total number of edges or missing edges
    n_observations = np.sum([np.sum(edge_count_matrices[k] + missing_edges_matrices[k]) for k in range(len(Ps))])
    BIC = n_parameters * np.log(n_observations) - 2*log_likelihood
    return log_likelihood, BIC

def compute_edge_probability_MAE(Ps, inferred_Ps, vertex_counts, using_directed_graphs, block_assignments, inferred_block_assignments, verbose=False):
    """Computes the mean absoulte error (MAE) between inferred and ground truth edge probabilities over the all pairs of vertices in all graphs"""
    total_edge_P_MAE = 0
    for k, P in enumerate(Ps):
        if using_directed_graphs:
            u_indices, v_indices = np.meshgrid(range(vertex_counts[k]), range(vertex_counts[k]))
            u_indices, v_indices = u_indices[u_indices != v_indices], v_indices[u_indices != v_indices] # exclude self-loops
        else:
            u_indices, v_indices = np.triu_indices(vertex_counts[k], k=1)

        edge_probabilities = P[block_assignments[k][u_indices], block_assignments[k][v_indices]]
        inferred_edge_probabilities = inferred_Ps[k][inferred_block_assignments[k][u_indices], inferred_block_assignments[k][v_indices]]
        edge_P_diff = np.sum(np.abs(edge_probabilities - inferred_edge_probabilities))

        if using_directed_graphs:
            n_possible_edges = vertex_counts[k] * (vertex_counts[k] - 1)
        else:
            n_possible_edges = vertex_counts[k] * (vertex_counts[k] - 1) // 2

        edge_P_MAE = edge_P_diff / n_possible_edges
        total_edge_P_MAE += edge_P_MAE
        if verbose:
            print(f"Graph {k} - total absolute edge probability difference: {edge_P_diff}, mean absolute edge probability difference: {edge_P_MAE}")
    return total_edge_P_MAE / len(Ps)

def compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices, epsilon=1e-10):
    inferred_Ps = []
    for k in range(len(edge_count_matrices)):
        total_edges_matrix = edge_count_matrices[k] + missing_edges_matrices[k]
        non_zero_indices = np.where(total_edges_matrix > 0)
        inferred_P = np.zeros(edge_count_matrices[k].shape)
        # Compute the maximum likelihood parameteters for the inferred block structure
        inferred_P[non_zero_indices] = edge_count_matrices[k][non_zero_indices] / total_edges_matrix[non_zero_indices]
        # Adjust the probabilities by epsilon to prevent log(p) and log(1-p) from becoming -inf
        inferred_P = np.where(inferred_P < epsilon, epsilon, inferred_P)
        inferred_P = np.where(inferred_P > 1-epsilon, 1-epsilon, inferred_P)
        inferred_Ps.append(inferred_P)
    return inferred_Ps

def compare_shared_blocks_ARI(Gs, inferred_block_assingments, inferred_shared_block_sets, block_assignments, shared_block_sets):
    scores = np.zeros((len(Gs), 2))
    for k, G in enumerate(Gs):
        b = inferred_block_assingments[k]
        predicted_shared_or_other_blocks_binary = np.array([1 if b[int(v)] in inferred_shared_block_sets[k] else -1 for v in G.vertices()])
        ground_truth_shared_or_other_blocks_binary = np.array([1 if block_assignments[k][int(v)] in shared_block_sets[k] else -1 for v in G.vertices()])
        ARI1 = adjusted_rand_score(predicted_shared_or_other_blocks_binary, ground_truth_shared_or_other_blocks_binary)

        predicted_shared_or_other_blocks = np.array([b[int(v)] if b[int(v)] in inferred_shared_block_sets[k] else -1 for v in G.vertices()])
        ground_truth_shared_or_other_blocks = np.array([block_assignments[k][int(v)] if block_assignments[k][int(v)] in shared_block_sets[k] else -1 for v in G.vertices()])
        ARI2 = adjusted_rand_score(ground_truth_shared_or_other_blocks, predicted_shared_or_other_blocks)
        scores[k] = ARI1, ARI2
    return scores

def permute_shared_first(Ps, shared_block_sets):
    """Reorder rows and columns so that shared blocks become the first s by s submatrix"""
    permuted_Ps = []
    for k in range(len(Ps)):
        nonshared_block_indices = [i for i in np.arange(Ps[k].shape[0]) if i not in shared_block_sets[k]]
        # Sort the shared and nonshared blocks by their diagonal value to make comparing blocks with the ground truth easier
        sorted_nonshared_block_indices = sorted(nonshared_block_indices, key = lambda i: Ps[k][i,i])
        sorted_shared_block_indices = sorted(shared_block_sets[k], key = lambda i: Ps[k][i,i])
        block_indices = np.concatenate((sorted_shared_block_indices, sorted_nonshared_block_indices))
        permuted_Ps.append(Ps[k][block_indices][:, block_indices])
    return permuted_Ps

def relabel_shared_blocks_first(states, shared_blocks):
    """Relabel blocks in states so that shared blocks in each graph have matching indices from 1 to n_shared_blocks"""
    new_labels = [np.full(states[k].get_B(), -1, dtype=int) for k in range(len(states))]
    # map shared blocks to indices from 0 to n_shared_blocks-1
    for shared_block_i, shared_block in enumerate(shared_blocks):
        for k, block_i in enumerate(shared_block):
            new_labels[k][block_i] = shared_block_i
    for k in range(len(states)):
        # map remaining blocks to the indices from n_shared_blocks onwards
        remaining_blocks = np.where(new_labels[k] == -1)[0]
        new_labels[k][remaining_blocks] = np.arange(len(shared_blocks), len(shared_blocks) + len(remaining_blocks))
        states[k].set_state(new_labels[k][states[k].get_blocks()])



from shared_block_optimization import *
from gt_extension import *
import pickle

def fit_shared_SBM_twostep(Gs, block_counts, n_shared_blocks, shared_block_alg=optimize_shared_blocks_ILP, using_directed_graphs=True, n_iter=1,
                           state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict(), verbose=0):
    states, inferred_block_assignments = fit_SBMs(Gs, block_counts, n_shared_blocks=n_shared_blocks,
                                                  state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args)
    inferred_block_counts = get_block_counts(inferred_block_assignments)
    edge_count_matrices, missing_edges_matrices = calculate_edge_counts_and_missing_edges(inferred_block_assignments, Gs, using_directed_graphs)
    inferred_Ps = compute_max_likelihood_params(edge_count_matrices, missing_edges_matrices)
    # find which blocks should be shared and reorder so that shared blocks are first and have matching indices
    _, inferred_shared_blocks, _ = shared_block_alg(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices)
    relabel_shared_blocks_first(states, inferred_shared_blocks)
    states, inferred_block_assignments = fit_shared_SBM(Gs, block_counts, n_shared_blocks, states=states, n_iter=n_iter,
                                                        state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args, verbose=verbose)
    return states, inferred_block_assignments

def fit_single_SBMs_twostep(Gs, block_counts, n_shared_blocks=None, n_iter=1,
                            state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict(), verbose=0):
    states, _ = fit_SBMs(Gs, block_counts, n_shared_blocks=n_shared_blocks)
    states, inferred_block_assignments = fit_SBMs_bernoulli(Gs, block_counts, states=states, n_iter=n_iter,
                                                            state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args, verbose=verbose)
    return states, inferred_block_assignments

def run_experiments(inference_algorithms, shared_block_optimization_algs, Gs, block_counts=None, n_shared_blocks=0, using_directed_graphs=True, results_file_name=None, seed=None, verbose=True):
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
            states, inferred_block_assignments = inference_algorithm(Gs, block_counts, n_shared_blocks=n_shared_blocks)
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

    if results_file_name is not None:
        results = inference_algorithms, shared_block_optimization_algs, inference_running_times, single_log_likelihoods, single_BICs, total_running_times, shared_log_likelihoods, shared_BICs
        with open(results_file_name+".pkl", "wb") as pkl_file:
            pickle.dump(results, pkl_file)

    print_log(f"Total running time: {datetime.now() - total_start_time}", log_file, verbose=True)
