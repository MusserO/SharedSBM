import graph_tool.all as gt
import numpy as np
from utils import fill_default_params

from graph_tool_extension import libshared_mcmc

from graph_tool import Vector_size_t
from graph_tool import libcore

class DictState(dict):
    """Dictionary with (key,value) pairs accessible via attributes."""
    def __init__(self, d):
        self.update(d)
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, val):
        self[attr] = val

def shared_mcmc_sweep(states, n_shared_blocks, beta=1., c=.5, d=.01, niter=1, entropy_args={},
                   allow_vacate=True, sequential=True, deterministic=False, verbose=0):

        mcmc_state = DictState(locals())
        mcmc_state.oentropy_args = states[0]._get_entropy_args(entropy_args)

        vlists = []
        state_list = []
        for i, state in enumerate(states):
            state_list.append(states[i]._state)
            vlist = Vector_size_t()
            vertices = state.g.vertex_index.copy().fa
            vlist.resize(len(vertices))
            vlist.a = vertices
            vlists.append(vlist)
        mcmc_state.py_states = state_list
        mcmc_state.py_vlists = vlists

        mcmc_state.n_shared_blocks = n_shared_blocks
        mcmc_state.verbose = verbose

        return libshared_mcmc.shared_mcmc_sweep(mcmc_state, states[0]._state, libcore.get_rng())

def fit_shared_SBM(Gs, block_counts, n_shared_blocks, states=None, n_iter=1, state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict(), verbose=0):
    if states is None:
        states = []
        for k, G in enumerate(Gs):
            n_blocks = block_counts[k]

            state_args, entropy_args, _ = fill_default_params(G=G, n_blocks=n_blocks, n_shared_blocks=n_shared_blocks, state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args)
            #state_args = dict(deg_corr=False, B=n_blocks)

            state = state_class(G, **state_args)
            states.append(state)
    else:
        _, entropy_args, _ = fill_default_params(n_shared_blocks=n_shared_blocks, entropy_args=entropy_args)
    result = shared_mcmc_sweep(states, n_shared_blocks, niter=n_iter, beta=np.inf, d=0, allow_vacate=False, entropy_args=entropy_args, verbose=verbose)
    if verbose:
        print(f"Entropy, n_attempts, n_moves: {result}")

    inferred_block_assignments = []
    for state in states:
        inferred_block_assignments.append(state.get_blocks().a)

    return states, inferred_block_assignments

def fit_SBMs_bernoulli(Gs, block_counts, n_shared_blocks=0, states=None, n_iter=1, state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict(), verbose=0):
    # same as shared SBM fitting but with n_shared_blocks set to 0
    return fit_shared_SBM(Gs, block_counts, 0, states=states, n_iter=n_iter, state_class=state_class, state_args=state_args, entropy_args=entropy_args, multilevel_mcmc_args=multilevel_mcmc_args, verbose=verbose)
