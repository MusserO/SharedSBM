import numpy as np
import sys
sys.path.insert(0, '..')
import pysbm.pysbm as pysbm

def get_edge_counts_between_blocks(partition, r, s, change_in_r_vertices=0, change_in_s_vertices=0, change_in_edges=0, directed=True):
    vertices_in_r = partition.get_number_of_nodes_in_block(r) + change_in_r_vertices
    vertices_in_s = partition.get_number_of_nodes_in_block(s) + change_in_s_vertices
    edges_from_r_to_s = partition.get_edge_count(r,s) + change_in_edges
    if r != s:
        total_max_edges_from_r_to_s = vertices_in_r * vertices_in_s
    else:
        if directed:
            total_max_edges_from_r_to_s = vertices_in_r * (vertices_in_r - 1)
        else:
            total_max_edges_from_r_to_s = vertices_in_r * (vertices_in_r - 1) / 2
    return edges_from_r_to_s, total_max_edges_from_r_to_s
    
def bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=0, change_in_s_vertices=0, change_in_edges=0, directed=True):
    # compute log likelihood term for edges from block r to block s
    edges_from_r_to_s, total_max_edges_from_r_to_s = \
        get_edge_counts_between_blocks(partition, r, s, change_in_r_vertices, change_in_s_vertices, change_in_edges, directed=directed)
    missing_edges_from_r_to_s = total_max_edges_from_r_to_s - edges_from_r_to_s
    if edges_from_r_to_s == 0 or missing_edges_from_r_to_s == 0:
        return 0
    return edges_from_r_to_s * np.log(edges_from_r_to_s / total_max_edges_from_r_to_s) \
                    + missing_edges_from_r_to_s * np.log(missing_edges_from_r_to_s / total_max_edges_from_r_to_s)

def shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=0, change_in_s_vertices=0, change_in_edges=0, directed=True):
    # compute shared log likelihood term for edges from block r to block s
    # if both r and s are shared blocks compute the new likelihood for each partition
    if r < shared_partition.n_shared_blocks and s < shared_partition.n_shared_blocks:
        total_shared_edges_from_r_to_s = 0
        total_shared_max_edges_from_r_to_s = 0
        # compute sums of edges and max edges between shared blocks for all partitions
        for k, partition in enumerate(shared_partition.partitions):
            if k == partition_index:
                edges_from_r_to_s, total_max_edges_from_r_to_s = \
                    get_edge_counts_between_blocks(partition, r, s, change_in_r_vertices, change_in_s_vertices, change_in_edges, directed=directed)
            else:
                edges_from_r_to_s, total_max_edges_from_r_to_s = \
                    get_edge_counts_between_blocks(partition, r, s, directed=directed)
            total_shared_edges_from_r_to_s += edges_from_r_to_s
            total_shared_max_edges_from_r_to_s += total_max_edges_from_r_to_s
            
        total_shared_missing_edges_from_r_to_s = total_shared_max_edges_from_r_to_s - total_shared_edges_from_r_to_s
        if total_shared_edges_from_r_to_s == 0 or total_shared_missing_edges_from_r_to_s == 0:
            return 0
        
        # compute maximum likelihood parameter p as sum of edges between shared blocks divided by sum of total max edges
        p = total_shared_edges_from_r_to_s / total_shared_max_edges_from_r_to_s

        # compute the new likelihood between blocks r and s for each partition
        log_likelihood = total_shared_edges_from_r_to_s * np.log(p) \
                       + total_shared_missing_edges_from_r_to_s * np.log(1-p)
        return log_likelihood
    else:
        partition = shared_partition.partitions[partition_index]
        return bernoulli_log_likelihood(partition, r, s, change_in_r_vertices, change_in_s_vertices, change_in_edges)
    

def calculate_complete_bernoulli_log_likelihood(partition):
    likelihood = 0.0
    for r in range(partition.B):
        for s in range(partition.B):
            likelihood += bernoulli_log_likelihood(partition, r, s)
    return likelihood

def calculate_delta_bernoulli_log_likelihood_undirected(partition, from_block, to_block, *args):
    raise NotImplementedError()

def calculate_delta_bernoulli_log_likelihood_directed(partition, from_block, to_block, *args):
    if from_block == to_block:
        return 0.0
    if len(args) == 5:
        # kit: counts of edges from node being moved towards each block 
        # kti: counts of edges from each block towards node being moved
        kit, kti, selfloops, in_degree, out_degree = args
    else:
        raise ValueError()
    delta = 0.0

    for s, edges_from_vertex_to_block in kit.items():
        if s != from_block and s != to_block:
            # compute the change in log-likelihood for the term for edges from from_block to another block s
            r = from_block
            old_likelihood = bernoulli_log_likelihood(partition, r, s)
            new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=-1, change_in_edges=-edges_from_vertex_to_block)
            delta += new_likelihood - old_likelihood

            # the change in likelihoods for the term for edges from to_block to s
            r = to_block
            old_likelihood = bernoulli_log_likelihood(partition, r, s)
            new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=1, change_in_edges=edges_from_vertex_to_block)
            delta += new_likelihood - old_likelihood


    for r, edges_from_block_to_vertex in kti.items():
        if r != from_block and r != to_block:
            # compute the change in log-likelihood for the term for edges from a block r to from_block
            s = from_block
            old_likelihood = bernoulli_log_likelihood(partition, r, s)
            new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_s_vertices=-1, change_in_edges=-edges_from_block_to_vertex)
            delta += new_likelihood - old_likelihood

            # the change in likelihoods for the term for edges from r to to_block
            s = to_block
            old_likelihood = bernoulli_log_likelihood(partition, r, s)
            new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_s_vertices=1, change_in_edges=edges_from_block_to_vertex)
            delta += new_likelihood - old_likelihood

    # change in the term for edges from from_block to to_block
    r = from_block
    s = to_block
    old_likelihood = bernoulli_log_likelihood(partition, r, s)
    new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=-1, change_in_s_vertices=1, change_in_edges=-kit[to_block]+kti[from_block])
    delta += new_likelihood - old_likelihood

    # change in the term for edges from to_block to from_block
    r = to_block
    s = from_block
    old_likelihood = bernoulli_log_likelihood(partition, r, s)
    new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=1, change_in_s_vertices=-1, change_in_edges=kit[from_block]-kti[to_block])
    delta += new_likelihood - old_likelihood

    # change in the term for edges within from_block
    r = from_block
    s = from_block
    old_likelihood = bernoulli_log_likelihood(partition, r, s)
    new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=-1, change_in_s_vertices=-1, change_in_edges=-kit[from_block]-kti[from_block])
    delta += new_likelihood - old_likelihood

    # change in the term for edges within to_block
    r = to_block
    s = to_block
    old_likelihood = bernoulli_log_likelihood(partition, r, s)
    new_likelihood = bernoulli_log_likelihood(partition, r, s, change_in_r_vertices=1, change_in_s_vertices=1, change_in_edges=kit[to_block]+kti[to_block])
    delta += new_likelihood - old_likelihood

    return delta

class BernoulliLogLikelihood(pysbm.ObjectiveFunction):
    """Bernoulli non-degree-corrected log-likelihood"""
    title = "Bernoulli Log-Likelihood"
    short_title = "BLL"

    def __init__(self, is_directed):
        super(BernoulliLogLikelihood, self).__init__(
            is_directed,
            calculate_complete_bernoulli_log_likelihood,
            calculate_complete_bernoulli_log_likelihood,
            calculate_delta_bernoulli_log_likelihood_undirected,
            calculate_delta_bernoulli_log_likelihood_directed)


def calculate_complete_shared_bernoulli_log_likelihood(shared_partition):
    raise NotImplementedError()

def calculate_delta_shared_bernoulli_log_likelihood_undirected(shared_partition, from_block, to_block, *args):
    raise NotImplementedError()

def calculate_delta_shared_bernoulli_log_likelihood_directed(shared_partition, partition_index, from_block, to_block, *args):
    if from_block == to_block:
        return 0.0
    if len(args) == 5:
        # kit: counts of edges from node being moved towards each block 
        # kti: counts of edges from each block towards node being moved
        kit, kti, selfloops, in_degree, out_degree = args
    else:
        raise ValueError()
    delta = 0.0

    for s, edges_from_vertex_to_block in kit.items():
        if s != from_block and s != to_block:
            # compute the change in log-likelihood for the term for edges from from_block to another block s
            r = from_block
            old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
            new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=-1, change_in_edges=-edges_from_vertex_to_block)
            delta += new_likelihood - old_likelihood

            # the change in likelihoods for the term for edges from to_block to s
            r = to_block
            old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
            new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=1, change_in_edges=edges_from_vertex_to_block)
            delta += new_likelihood - old_likelihood


    for r, edges_from_block_to_vertex in kti.items():
        if r != from_block and r != to_block:
            # compute the change in log-likelihood for the term for edges from a block r to from_block
            s = from_block
            old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
            new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_s_vertices=-1, change_in_edges=-edges_from_block_to_vertex)
            delta += new_likelihood - old_likelihood

            # the change in likelihoods for the term for edges from r to to_block
            s = to_block
            old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
            new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_s_vertices=1, change_in_edges=edges_from_block_to_vertex)
            delta += new_likelihood - old_likelihood

    # change in the term for edges from from_block to to_block
    r = from_block
    s = to_block
    old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
    new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=-1, change_in_s_vertices=1, change_in_edges=-kit[to_block]+kti[from_block])
    delta += new_likelihood - old_likelihood

    # change in the term for edges from to_block to from_block
    r = to_block
    s = from_block
    old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
    new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=1, change_in_s_vertices=-1, change_in_edges=kit[from_block]-kti[to_block])
    delta += new_likelihood - old_likelihood

    # change in the term for edges within from_block
    r = from_block
    s = from_block
    old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
    new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=-1, change_in_s_vertices=-1, change_in_edges=-kit[from_block]-kti[from_block])
    delta += new_likelihood - old_likelihood

    # change in the term for edges within to_block
    r = to_block
    s = to_block
    old_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s)
    new_likelihood = shared_bernoulli_log_likelihood(shared_partition, partition_index, r, s, change_in_r_vertices=1, change_in_s_vertices=1, change_in_edges=kit[to_block]+kti[to_block])
    delta += new_likelihood - old_likelihood

    return delta

class SharedBernoulliLogLikelihood(pysbm.ObjectiveFunction):
    """Shared Bernoulli non-degree-corrected log-likelihood"""
    title = "Shared Bernoulli Log-Likelihood"
    short_title = "SBLL"

    def __init__(self, is_directed):
        super(SharedBernoulliLogLikelihood, self).__init__(
            is_directed,
            calculate_complete_shared_bernoulli_log_likelihood,
            calculate_complete_shared_bernoulli_log_likelihood,
            self.calculate_delta_shared_bernoulli_log_likelihood_undirected,
            self.calculate_delta_shared_bernoulli_log_likelihood_directed)
        
    def set_shared_partition(self, shared_partition):
        self.shared_partition = shared_partition
        self.partition_to_index = {partition: i for i, partition in enumerate(shared_partition.partitions)}

    def calculate_delta_shared_bernoulli_log_likelihood_undirected(self, partition, from_block, to_block, *args):
        partition_index = self.partition_to_index[partition]
        return calculate_delta_shared_bernoulli_log_likelihood_undirected(self.shared_partition, partition_index, from_block, to_block, *args)
    
    def calculate_delta_shared_bernoulli_log_likelihood_directed(self, partition, from_block, to_block, *args):
        partition_index = self.partition_to_index[partition]
        return calculate_delta_shared_bernoulli_log_likelihood_directed(self.shared_partition, partition_index, from_block, to_block, *args)
    
class NxSharedPartition():
    """
    Shared Partition of NetworkX Graphs
    """
    def __init__(self, Gs, n_shared_blocks, block_counts, using_directed_graphs=True, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=True, save_neighbor_edges=False,
                 weighted_graph=False, representation=None):
        self.partitions = [pysbm.NxPartition(graph=G, number_of_blocks=block_counts[k], calculate_degree_of_blocks=calculate_degree_of_blocks,
                                           save_neighbor_of_blocks=save_neighbor_of_blocks, fill_random=fill_random,
                                           save_neighbor_edges=save_neighbor_edges, weighted_graph=weighted_graph, representation=representation) for k, G in enumerate(Gs)]
        self.n_shared_blocks = n_shared_blocks
        self.block_counts = block_counts
        self.using_directed_graphs = using_directed_graphs

    def is_graph_directed(self):
        return self.using_directed_graphs

    def get_random_partition(self):
        partition_index = np.random.randint(len(self.partitions))
        return self.partitions[partition_index]
    
    def set_save_neighbor_edges(self, save_neighbor_edges):
        # switch edge saving on for each partition
        for partition in self.partitions:
            partition.set_save_neighbor_edges(save_neighbor_edges)


class SharedMetropolisHastingsInference(pysbm.MetropolisHastingInference):
    """Shared Metropolis-Hastings Inference Algorithm"""
    title = "Shared Metropolis-Hastings Inference"
    short_title = "SMHA"

    def __init__(self, objective_function, shared_partition, limit_possible_blocks=False):
        super(SharedMetropolisHastingsInference, self).__init__(None, objective_function, shared_partition, limit_possible_blocks)
        self.shared_partition = shared_partition
        self._objective_function.set_shared_partition(shared_partition)

    def infer_stepwise_undirected(self):
        self.partition = self.shared_partition.get_random_partition()
        super().infer_stepwise_undirected()

    def infer_stepwise_directed(self):
        self.partition = self.shared_partition.get_random_partition()
        super().infer_stepwise_directed()