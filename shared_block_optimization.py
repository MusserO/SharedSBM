import numpy as np
import itertools
from gurobipy import Model, GRB, quicksum

def compute_neg_log_params(inferred_Ps):
    ts = []
    t_bars = []
    for k in range(len(inferred_Ps)):
        t = -np.log(inferred_Ps[k])
        t_bar = -np.log(1-inferred_Ps[k])
        ts.append(t)
        t_bars.append(t_bar)
    return ts, t_bars

def possible_shared_block_to_index(shared_block, block_counts):
    index = 0
    for i, block_index in enumerate(shared_block):
        index *= block_counts[i]
        index += block_index
    return index

def index_to_possible_shared_block(index, block_counts):
    shared_block = []
    for block_count in reversed(block_counts):
        index, block_index = divmod(index, block_count)
        shared_block.append(block_index)
    return tuple(reversed(shared_block))

def generate_possible_shared_blocks(block_counts):
    ranges = [range(block_count) for block_count in block_counts]
    return itertools.product(*ranges)

def max_likelihood_param_for_shared_blocks(shared_block1, shared_block2, edge_count_matrices, missing_edges_matrices, epsilon=1e-10):
    total_edges = 0
    total_possible_edges = 0
    for k in range(len(edge_count_matrices)):
        i, j = shared_block1[k], shared_block2[k]
        total_edges += edge_count_matrices[k][i,j]
        total_possible_edges += edge_count_matrices[k][i,j] + missing_edges_matrices[k][i,j]
    if total_possible_edges == 0:
        p = 0
    else:
        p = total_edges / total_possible_edges
    if p <= epsilon:
        p = epsilon
    elif p >= 1-epsilon:
        p = 1-epsilon
    return p

def compute_possible_shared_neg_log_params(inferred_block_counts, edge_count_matrices, missing_edges_matrices, epsilon=1e-10):
    n_possible_shared_blocks = np.prod(inferred_block_counts)

    q = np.zeros((n_possible_shared_blocks, n_possible_shared_blocks))
    q_bar = np.zeros((n_possible_shared_blocks, n_possible_shared_blocks))
    for shared_index1, shared_block1 in enumerate(generate_possible_shared_blocks(inferred_block_counts)):
        for shared_index2, shared_block2 in enumerate(generate_possible_shared_blocks(inferred_block_counts)):
            p = max_likelihood_param_for_shared_blocks(shared_block1, shared_block2, edge_count_matrices, missing_edges_matrices, epsilon=epsilon)
            q[shared_index1, shared_index2] = -np.log(p)
            q_bar[shared_index1, shared_index2] = -np.log(1-p)
    return q, q_bar

def set_shared_block_parameters(inferred_Ps, inferred_block_counts, inferred_shared_blocks, edge_count_matrices=None, missing_edges_matrices=None, q=None):
    # Set the new maximum likelihood parameters for shared blocks
    inferred_shared_Ps = [np.array(matrix) for matrix in inferred_Ps]
    for inferred_shared_block1 in inferred_shared_blocks:
        for inferred_shared_block2 in inferred_shared_blocks:
            if q is not None:
                shared_index1 = possible_shared_block_to_index(inferred_shared_block1, inferred_block_counts)
                shared_index2 = possible_shared_block_to_index(inferred_shared_block2, inferred_block_counts)
                shared_p = np.exp(-q[shared_index1, shared_index2])
            else:
                shared_p = max_likelihood_param_for_shared_blocks(inferred_shared_block1, inferred_shared_block2, edge_count_matrices, missing_edges_matrices)
            for k in range(len(inferred_shared_Ps)):
                inferred_shared_Ps[k][inferred_shared_block1[k], inferred_shared_block2[k]] = shared_p
    return inferred_shared_Ps

def solve_ILP(n, s, B, E, F, ts, t_bars, q, q_bar, verbose=False):
    m = Model("ssbm_fixed")
    if not verbose:
        m.setParam('OutputFlag', 0)

    n_possible_shared_blocks = np.prod(B)

    # Create variables
    x = [[[m.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{r}") for r in range(s)] for i in range(B[k])] for k in range(n)]
    y = [[m.addVar(vtype=GRB.CONTINUOUS, name=f"y_{k}_{i}") for i in range(B[k])] for k in range(n)]
    z = [[[m.addVar(vtype=GRB.CONTINUOUS, name=f"z_{k}_{i}_{j}") for j in range(B[k])] for i in range(B[k])] for k in range(n)]
    c = [[m.addVar(vtype=GRB.CONTINUOUS, name=f"c_{b}_{r}") for r in range(s)] for b in range(n_possible_shared_blocks)]
    d = [m.addVar(vtype=GRB.CONTINUOUS, name=f"d_{b}") for b in range(n_possible_shared_blocks)]
    w = [[m.addVar(vtype=GRB.CONTINUOUS, name=f"w_{b}_{b_prime}") for b_prime in range(n_possible_shared_blocks)] for b in range(n_possible_shared_blocks)]

    # Set objective
    m.setObjective(
        quicksum(quicksum(quicksum(
                (1 - z[k][i][j]) * (E[k][i,j] * ts[k][i,j] + F[k][i,j] * t_bars[k][i,j]) +
                quicksum(
                    quicksum(
                        w[b][b_prime] * (E[k][i,j] * q[b,b_prime] + F[k][i,j] * q_bar[b,b_prime]) \
                        if j == index_to_possible_shared_block(b_prime, B)[k] else 0 for b_prime in range(n_possible_shared_blocks)) \
                    if i == index_to_possible_shared_block(b, B)[k] else 0 for b in range(n_possible_shared_blocks)) \
                for i in range(B[k])) \
            for j in range(B[k])) \
        for k in range(n)),
        GRB.MINIMIZE
    )

    # Add constraints
    for k in range(n):
        for r in range(s):
            m.addConstr(quicksum(x[k][i][r] for i in range(B[k])) == 1)
        for i in range(B[k]):
            m.addConstr(y[k][i] == quicksum(x[k][i][r] for r in range(s)))
            m.addConstr(y[k][i] <= 1)
            for j in range(B[k]):
                m.addConstr(z[k][i][j] <= y[k][i])
                m.addConstr(z[k][i][j] <= y[k][j])
                m.addConstr(z[k][i][j] >= y[k][i] + y[k][j] - 1)

    for b in range(n_possible_shared_blocks):
        shared_block = index_to_possible_shared_block(b, B)
        for r in range(s):
            for k in range(n):
                m.addConstr(c[b][r] <= x[k][shared_block[k]][r])
            m.addConstr(c[b][r] >= quicksum(x[k][shared_block[k]][r] for k in range(n))-n+1)
        m.addConstr(d[b] == quicksum(c[b][r] for r in range(s)))

    for b in range(n_possible_shared_blocks):
        for b_prime in range(n_possible_shared_blocks):
            m.addConstr(w[b][b_prime] <= d[b])
            m.addConstr(w[b][b_prime] <= d[b_prime])
            m.addConstr(w[b][b_prime] >= d[b] + d[b_prime] - 1)

    # Optimize model
    m.optimize()

    # Get optimal variable values
    c_values = np.array([[c[b][r].X for r in range(s)] for b in range(n_possible_shared_blocks)])
    return c_values

def optimize_shared_blocks_ILP(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices, verbose=False):
    """Use an ILP to find an optimal set of shared blocks that minimizes the negative log-likelihood"""
    n = len(inferred_Ps)
    s = n_shared_blocks
    B = inferred_block_counts
    E = edge_count_matrices
    F = missing_edges_matrices

    ts, t_bars = compute_neg_log_params(inferred_Ps)
    q, q_bar = compute_possible_shared_neg_log_params(inferred_block_counts, edge_count_matrices, missing_edges_matrices)

    c_values = solve_ILP(n, s, B, E, F, ts, t_bars, q, q_bar, verbose=verbose)

    # inferred_shared_blocks contains n_shared_blocks entries of size n_graphs that indicate which block from each graph corresponds that shared block
    inferred_shared_blocks = []
    for r in range(n_shared_blocks):
        shared_block_index = np.where(c_values[:,r] > 0.999)[0][0]
        inferred_shared_blocks.append(index_to_possible_shared_block(shared_block_index, B))

    # inferred_shared_block_sets contains n_graphs entries of size n_shared_blocks, the sets of blocks that are shared for each graph
    inferred_shared_block_sets = list(zip(*inferred_shared_blocks)) 
    inferred_shared_Ps = set_shared_block_parameters(inferred_Ps, inferred_block_counts, inferred_shared_blocks, q=q)

    return inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets

def compute_delta_nll_for_shared_block(shared_block, current_shared_blocks, inferred_block_counts, ts, t_bars, q, q_bar, E, F):
    delta_nll = 0
    b = possible_shared_block_to_index(shared_block, inferred_block_counts)
    for k in range(len(E)):
        i = shared_block[k]
        delta_nll += E[k][i,i] * (q[b,b]-ts[k][i,i]) + F[k][i,i] * (q_bar[b,b]-t_bars[k][i,i])
    for shared_block2 in current_shared_blocks:
        b_prime = possible_shared_block_to_index(shared_block2, inferred_block_counts)
        for k in range(len(E)):
            i = shared_block[k]
            j = shared_block2[k]
            delta_nll += E[k][i,j] * (q[b,b_prime]-ts[k][i,j]) + F[k][i,j] * (q_bar[b,b_prime]-t_bars[k][i,j])
            delta_nll += E[k][j,i] * (q[b_prime,b]-ts[k][j,i]) + F[k][j,i] * (q_bar[b_prime,b]-t_bars[k][j,i])
    return delta_nll

def optimize_shared_blocks_greedy(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices):
    """Choose shared blocks greedily, picking a shared block that minimizes the negative log likelihood at each iteration"""
    ts, t_bars = compute_neg_log_params(inferred_Ps)
    q, q_bar = compute_possible_shared_neg_log_params(inferred_block_counts, edge_count_matrices, missing_edges_matrices)

    current_shared_blocks = list()
    current_shared_block_sets = [set() for _ in range(len(inferred_Ps))]
    while len(current_shared_blocks) < n_shared_blocks:
        best_delta_nll = np.inf
        possible_block_ranges = [set(range(block_count))-current_shared_block_sets[i] for i, block_count in enumerate(inferred_block_counts)]
        for shared_block in itertools.product(*possible_block_ranges):
            delta_nll = compute_delta_nll_for_shared_block(shared_block, current_shared_blocks, inferred_block_counts, ts, t_bars, q, q_bar, edge_count_matrices, missing_edges_matrices)
            if delta_nll < best_delta_nll:
                best_delta_nll = delta_nll
                best_block = shared_block
        current_shared_blocks.append(best_block)
        current_shared_block_sets = [set(block_set) for block_set in zip(*current_shared_blocks)]

    inferred_shared_blocks = current_shared_blocks
    inferred_shared_block_sets = list(zip(*inferred_shared_blocks))
    inferred_shared_Ps = set_shared_block_parameters(inferred_Ps, inferred_block_counts, inferred_shared_blocks, q=q)

    return inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets

def shared_blocks_random(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices):
    """Choose shared blocks randomly"""
    inferred_shared_block_sets = [tuple(np.random.choice(range(block_count), min(n_shared_blocks, block_count), replace=False)) for block_count in inferred_block_counts]
    inferred_shared_blocks = list(zip(*inferred_shared_block_sets))
    inferred_shared_Ps = set_shared_block_parameters(inferred_Ps, inferred_block_counts, inferred_shared_blocks, edge_count_matrices, missing_edges_matrices)

    return inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets

def firsts_as_shared_blocks(n_shared_blocks, inferred_Ps, inferred_block_counts, edge_count_matrices, missing_edges_matrices):
    """Set the first n_shared_blocks in each graph as the shared blocks"""
    inferred_shared_blocks = [np.repeat(i, len(inferred_Ps)) for i in np.arange(n_shared_blocks)]
    inferred_shared_block_sets = list(zip(*inferred_shared_blocks))
    inferred_shared_Ps = set_shared_block_parameters(inferred_Ps, inferred_block_counts, inferred_shared_blocks, edge_count_matrices, missing_edges_matrices)
    return inferred_shared_Ps, inferred_shared_blocks, inferred_shared_block_sets