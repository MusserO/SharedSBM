import re
import signal
from datetime import datetime
import numpy as np
import graph_tool.all as gt

def latex_table_to_mathjax(latex_table):
    # Replace unsupported LaTeX commands and environments
    mathjax_table = latex_table.replace('tabular', 'array')
    mathjax_table = mathjax_table.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
    mathjax_table = mathjax_table.replace('$', '')

    # Remove the caption and label
    mathjax_table = re.sub(r'\\caption\{.*?\}', '', mathjax_table)
    mathjax_table = re.sub(r'\\label\{.*?\}', '', mathjax_table)

    # Replace spaces with ~ to show them correctly
    mathjax_table = mathjax_table.strip()
    mathjax_table = mathjax_table.replace(' ', '~')

    return mathjax_table

def print_log(message, log_file=None, verbose=True):
    if verbose:
        print(message, flush=True)
    if log_file is not None:
        print(message, file=log_file)

def time_and_run_function(function, function_name, results_file=None, verbose=True, **params):
    start_time = datetime.now()
    results = function(**params)
    running_time = datetime.now() - start_time
    print_log(f"{function_name} done: {running_time}", results_file, verbose)
    return results, running_time

def signal_handler(signum, frame): raise TimeoutError()

def test_runtime_with_hard_time_limit(algorithm, algorithm_name, params, time_limit_exceeded, hard_time_limit, log_file=None, verbose=True):
    if not time_limit_exceeded:
        if hard_time_limit is None:
            _, running_time = time_and_run_function(algorithm, algorithm_name, results_file=log_file, verbose=verbose, **params)
        else:
            # Run algorithm with a hard time limit
            signal.alarm(hard_time_limit)
            try:
                _, running_time = time_and_run_function(algorithm, algorithm_name, results_file=log_file, verbose=verbose, **params)
            except TimeoutError:
                print_log(f"Time limit of {hard_time_limit} seconds exceeded for {algorithm_name}", log_file, verbose)
                running_time = np.inf
                time_limit_exceeded = True
            signal.alarm(0)

        if running_time != np.inf:
            # Convert running time to seconds
            running_time = running_time.total_seconds()
    else:
        running_time = np.inf
    return running_time, time_limit_exceeded

def fill_default_params(G=None, n_blocks=None, n_shared_blocks=None, state_class=gt.BlockState, state_args=dict(), entropy_args=dict(), multilevel_mcmc_args=dict()):
    if G is not None and 'weight' in G.ep:
        if not 'recs' in state_args:
            state_args['recs'] = [G.ep.weight]
        if not 'rec_types' in state_args:
            state_args['rec_types'] = ['discrete-geometric']
    if state_class == gt.BlockState:
        if not 'deg_corr' in state_args:
            state_args['deg_corr'] = False
        if not 'B' in state_args:
            state_args['B'] = n_blocks
        if not 'multigraph' in entropy_args:
            entropy_args['multigraph'] = False
        if not 'dense' in entropy_args:
            entropy_args['dense'] = True
    if not 'entropy_args' in multilevel_mcmc_args:
        multilevel_mcmc_args['entropy_args'] = entropy_args
    if n_blocks is not None:
        multilevel_mcmc_args.update({'B_min': n_blocks, 'B_max': n_blocks})
    elif n_shared_blocks is not None:
        multilevel_mcmc_args.update({'B_min': n_shared_blocks})
    return state_args, entropy_args, multilevel_mcmc_args