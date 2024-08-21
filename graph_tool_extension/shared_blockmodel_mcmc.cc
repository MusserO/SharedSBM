#include <graph_tool.hh>
#include <random.hh>

#include <boost/python.hpp>

#include <inference/blockmodel/graph_blockmodel_util.hh>
#include <inference/blockmodel/graph_blockmodel.hh>
#include <inference/loops/mcmc_loop.hh>

#include "shared_blockmodel_mcmc.hh"

#include <iostream>

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_shared_block_state, MCMC<State>::template MCMCSharedBlockState,
             MCMC_SHARED_BLOCK_STATE_params(State))

template <class MCMCState, class RNG>
auto shared_mcmc_sweep(MCMCState& state, RNG& rng)
{
    GILRelease gil;

    auto& combined_vlist = state.get_combined_vlist();
    auto beta = state.get_beta();

    typedef std::remove_const_t<decltype(state._null_move)> move_t;
    constexpr bool single_step =
        std::is_same_v<std::remove_reference_t<decltype(state.move_proposal(combined_vlist.front().first, combined_vlist.front().second, rng))>,
                       move_t>;

    double S = 0;
    size_t nattempts = 0;
    size_t nmoves = 0;

    for (size_t iter = 0; iter < state.get_niter(); ++iter)
    {
        if (state.is_sequential() && !state.is_deterministic())
            std::shuffle(combined_vlist.begin(), combined_vlist.end(), rng);

        size_t nsteps = 1;
        auto get_N =
            [&]
            {
                if constexpr (single_step)
                    return combined_vlist.size();
                else
                    return state.get_N();
            };

        state.init_iter(rng);

        for (size_t vi = 0; vi < get_N(); ++vi)
        {
            auto& v_index_pair = (state.is_sequential()) ?
                combined_vlist[vi] : uniform_sample(combined_vlist, rng);
            auto& graph_i = v_index_pair.first;
            auto& v = v_index_pair.second;

            if (state.skip_node(graph_i, v))
                continue;

            if (state._verbose > 1)
            {
                auto&& r = state.node_state(graph_i, v);
                std::cout << "graph: " << graph_i << " v: " << v << ": " << r;
            }

            auto&& ret = state.move_proposal(graph_i, v, rng);

            auto get_s =
                [&]() -> move_t&
                {
                    if constexpr (single_step)
                    {
                        return ret;
                    }
                    else
                    {
                        nsteps = get<1>(ret);
                        return get<0>(ret);
                    }
                };

            move_t& s = get_s();

            if (s == state._null_move)
            {
                if (state._verbose > 1)
                    std::cout << " (null proposal)" << std::endl;
                continue;
            }

            double dS, mP;
            std::tie(dS, mP) = state.virtual_move_dS(graph_i, v, s);

            nattempts += nsteps;

            bool accept = false;
            if (metropolis_accept(dS, mP, beta, rng))
            {
                state.perform_move(graph_i, v, s);
                nmoves += nsteps;
                S += dS;
                accept = true;
            }

            state.step(graph_i, v, s);

            if (state._verbose > 1)
                std::cout << " -> " << s << " accept: " << accept << " dS: " << dS << " mP: " << mP << " " << -dS * beta + mP << " " << S << std::endl;
        }

        if (state.is_sequential() && state.is_deterministic())
            std::reverse(combined_vlist.begin(), combined_vlist.end());
    }
    return make_tuple(S, nattempts, nmoves);
}

python::object do_shared_mcmc_sweep(python::object omcmc_state,
                             python::object oblock_state,
                             rng_t& rng)
{
    python::object ret;

    auto dispatch = [&](auto& block_state)
    {
        typedef typename std::remove_reference<decltype(block_state)>::type
            state_t;

        mcmc_shared_block_state<state_t>::make_dispatch
           (omcmc_state,
            [&](auto& s)
            {
                auto ret_ = shared_mcmc_sweep(*s, rng);
                ret = tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
            });
    };
    block_state::dispatch(oblock_state, dispatch);
    return ret;
}


BOOST_PYTHON_MODULE(libshared_mcmc)
{
    using namespace boost::python;    
    def("shared_mcmc_sweep", &do_shared_mcmc_sweep);
}