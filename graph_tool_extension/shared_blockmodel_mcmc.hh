#ifndef SHARED_BLOCKMODEL_MCMC_HH
#define SHARED_BLOCKMODEL_MCMC_HH

#include <config.h>

#include <vector>

#include <graph_tool.hh>
#include <inference/support/graph_state.hh>
#include <boost/mpl/vector.hpp>

#include <iostream>
#include "from_python_conversion.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_SHARED_BLOCK_STATE_params(State)                                    \
    ((__class__,&, mpl::vector<python::object>, 1))                              \
    ((py_states, &, python::list, 0))                                            \
    ((n_shared_blocks,, size_t, 0))                                              \
    ((py_vlists, &, python::list, 0))                                            \
    ((beta,, double, 0))                                                         \
    ((c,, double, 0))                                                            \
    ((d,, double, 0))                                                            \
    ((oentropy_args,, python::object, 0))                                        \
    ((allow_vacate,, bool, 0))                                                   \
    ((sequential,, bool, 0))                                                     \
    ((deterministic,, bool, 0))                                                  \
    ((verbose,, int, 0))                                                         \
    ((niter,, size_t, 0))

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCSharedBlockStateBase, MCMC_SHARED_BLOCK_STATE_params(State))

    template <class... Ts>
    class MCMCSharedBlockState
        : public MCMCSharedBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCSharedBlockStateBase<Ts...>,
                         MCMC_SHARED_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_SHARED_BLOCK_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCSharedBlockState(ATs&&... as)
           : MCMCSharedBlockStateBase<Ts...>(as...),
            _entropy_args(python::extract<typename State::_entropy_args_t&>(_oentropy_args))
        {
            _states = python_list_of_states_to_vec_of_states<State>(_py_states);
            _vlists = python_list_of_lists_to_vec_of_vecs(_py_vlists);

            for (auto& state : _states)
            {
                state.init_mcmc(*this);
                _m_entries_list.push_back(num_vertices(state._bg));
            }
            for (size_t i = 0; i < _vlists.size(); ++i)
            {
                for (auto v : _vlists[i])
                {
                    _combined_vlist.push_back(std::make_pair(i, v));
                }
            }
        }
        constexpr static size_t _null_move = null_group;
        typename State::_entropy_args_t& _entropy_args;

        std::vector<State> _states;
        std::vector<std::vector<size_t>> _vlists;
        std::vector<std::pair<size_t, size_t>> _combined_vlist;
        std::vector<typename State::m_entries_t> _m_entries_list;

        template <class Graph>
        [[gnu::pure]]
        uint64_t total_possible_edges_between_blocks(size_t r, size_t s, uint64_t wr_r, uint64_t wr_s, bool multigraph, const Graph& g) 
        {
            uint64_t nrns;

            assert(wr_r + wr_s > 0);

            if (graph_tool::is_directed(g)) {
                if (r != s || multigraph) 
                    nrns = wr_r * wr_s;
                else 
                    nrns = wr_r * (wr_r-1);
            } else {
                if (r != s) {
                    nrns = wr_r * wr_s;
                } else {
                    if (multigraph)
                        nrns = (wr_r * (wr_r + 1)) / 2;
                    else
                        nrns = (wr_r * (wr_r - 1)) / 2;
                }
            }
            return nrns;
        }

        template <class Graph>
        [[gnu::pure]]
        double eterm_shared_dense_bernoulli(size_t i, size_t r, size_t s, uint64_t ers, uint64_t wr_r, 
                                uint64_t wr_s, bool multigraph, const Graph& g) 
        {
            uint64_t nrns = total_possible_edges_between_blocks(r, s, wr_r, wr_s, multigraph, g);

            if (r < _n_shared_blocks && s < _n_shared_blocks)
            {
                // compute sum of ers (number of edges from block r to block s) and nrns (number of possible edges between the two blocks) for shared bernoulli log-likelihood
                // could get rid of this for loop by keeping track of total sums of ers and nrns in the state
                for (size_t j = 0; j < _states.size(); ++j)
                {   
                    if (j != i) {
                        ers += (r != null_group) ? get_beprop(r, s, _states[j]._mrs, _states[j]._emat) : 0;
                        nrns += total_possible_edges_between_blocks(r, s, _states[j]._wr[r], _states[j]._wr[s], multigraph, _states[j]._bg);
                    }
                }
            }
            if (ers == 0)
                return 0.;

            uint64_t frs = nrns - ers;
            double log_p = log(static_cast<double>(ers) / nrns);
            double log_one_minus_p = log(static_cast<double>(frs) / nrns);
            double log_likelihood = ers * log_p + frs * log_one_minus_p;
            return -log_likelihood;
        }

        template <class Graph>
        double virtual_move_shared_dense_bernoulli(size_t i, size_t v, size_t r, size_t nr, bool multigraph, const Graph& _g)
        {
            typedef Graph g_t;
            typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;

            if (r == nr)
                return 0;

            auto _bg = _states[i]._bg;
            auto _eweight = _states[i]._eweight;
            auto _b = _states[i]._b;
            auto _vweight = _states[i]._vweight;
            auto _mrs = _states[i]._mrs;
            auto _emat = _states[i]._emat;
            auto _wr = _states[i]._wr;

            vector<int> deltap(num_vertices(_bg), 0);
            int deltal = 0;
            for (auto e : out_edges_range(v, _g))
            {
                vertex_t u = target(e, _g);
                vertex_t s = _b[u];
                if (u == v)
                    deltal += _eweight[e];
                else
                    deltap[s] += _eweight[e];
            }
            if constexpr (!is_directed_::apply<g_t>::type::value)
                deltal /= 2;

            vector<int> deltam(num_vertices(_bg), 0);
            if (is_directed_::apply<g_t>::type::value)
            {
                for (auto e : in_edges_range(v, _g))
                {
                    vertex_t u = source(e, _g);
                    if (u == v)
                        continue;
                    vertex_t s = _b[u];
                    deltam[s] += _eweight[e];
                }
            }

            double dS = 0;
            int dwr = _vweight[v];
            int dwnr = dwr;

            if (r == null_group && dwnr == 0)
                dwnr = 1;

            if (nr == null_group)
            {
                std::fill(deltap.begin(), deltap.end(), 0);
                std::fill(deltam.begin(), deltam.end(), 0);
                if (dwr != _wr[r])
                    deltal = 0;
            }

            double Si = 0, Sf = 0;
            for (vertex_t s = 0; s < num_vertices(_bg); ++s)
            {
                if (_wr[s] == 0 && s != r && s != nr)
                    continue;

                int ers = (r != null_group) ? get_beprop(r, s, _mrs, _emat) : 0;
                int enrs = (nr != null_group) ? get_beprop(nr, s, _mrs, _emat) : 0;

                if (!is_directed_::apply<g_t>::type::value)
                {
                    if (s != nr && s != r)
                    {
                        if (r != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, r,  s, ers,              _wr[r],         _wr[s], multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, r,  s, ers - deltap[s],  _wr[r] - dwr,   _wr[s], multigraph, _bg);
                        }

                        if (nr != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, nr, s, enrs,             _wr[nr],        _wr[s], multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, nr, s, enrs + deltap[s], _wr[nr] + dwnr, _wr[s], multigraph, _bg);
                        }
                    }

                    if (s == r)
                    {
                        Si += eterm_shared_dense_bernoulli(i, r, r, ers,                      _wr[r],       _wr[r],       multigraph, _bg);
                        Sf += eterm_shared_dense_bernoulli(i, r, r, ers - deltap[r] - deltal, _wr[r] - dwr, _wr[r] - dwr, multigraph, _bg);
                    }

                    if (s == nr)
                    {
                        Si += eterm_shared_dense_bernoulli(i, nr, nr, enrs,                       _wr[nr],        _wr[nr],        multigraph, _bg);
                        Sf += eterm_shared_dense_bernoulli(i, nr, nr, enrs + deltap[nr] + deltal, _wr[nr] + dwnr, _wr[nr] + dwnr, multigraph, _bg);

                        if (r != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, r, nr, ers,                          _wr[r],       _wr[nr],        multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, r, nr, ers - deltap[nr] + deltap[r], _wr[r] - dwr, _wr[nr] + dwnr, multigraph, _bg);
                        }
                    }
                }
                else
                {
                    int esr = (r != null_group) ? get_beprop(s, r, _mrs, _emat) : 0;
                    int esnr  = (nr != null_group) ? get_beprop(s, nr, _mrs, _emat) : 0;

                    if (s != nr && s != r)
                    {
                        if (r != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, r, s, ers            , _wr[r]      , _wr[s]      , multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, r, s, ers - deltap[s], _wr[r] - dwr, _wr[s]      , multigraph, _bg);
                            Si += eterm_shared_dense_bernoulli(i, s, r, esr            , _wr[s]      , _wr[r]      , multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, s, r, esr - deltam[s], _wr[s]      , _wr[r] - dwr, multigraph, _bg);
                        }

                        if (nr != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, nr, s, enrs            , _wr[nr]       , _wr[s]        , multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, nr, s, enrs + deltap[s], _wr[nr] + dwnr, _wr[s]        , multigraph, _bg);
                            Si += eterm_shared_dense_bernoulli(i, s, nr, esnr            , _wr[s]        , _wr[nr]       , multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, s, nr, esnr + deltam[s], _wr[s]        , _wr[nr] + dwnr, multigraph, _bg);
                        }
                    }

                    if(s == r)
                    {
                        Si += eterm_shared_dense_bernoulli(i, r, r, ers                                  , _wr[r]      , _wr[r]      , multigraph, _bg);
                        Sf += eterm_shared_dense_bernoulli(i, r, r, ers - deltap[r]  - deltam[r] - deltal, _wr[r] - dwr, _wr[r] - dwr, multigraph, _bg);

                        if (nr != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, r, nr, esnr                         , _wr[r]      , _wr[nr]       , multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, r, nr, esnr - deltap[nr] + deltam[r], _wr[r] - dwr, _wr[nr] + dwnr, multigraph, _bg);
                        }
                    }

                    if(s == nr)
                    {
                        Si += eterm_shared_dense_bernoulli(i, nr, nr, esnr                                   , _wr[nr]       , _wr[nr]       , multigraph, _bg);
                        Sf += eterm_shared_dense_bernoulli(i, nr, nr, esnr + deltap[nr] + deltam[nr] + deltal, _wr[nr] + dwnr, _wr[nr] + dwnr, multigraph, _bg);

                        if (r != null_group)
                        {
                            Si += eterm_shared_dense_bernoulli(i, nr, r, esr                         , _wr[nr]       , _wr[r]      , multigraph, _bg);
                            Sf += eterm_shared_dense_bernoulli(i, nr, r, esr + deltap[r] - deltam[nr], _wr[nr] + dwnr, _wr[r] - dwr, multigraph, _bg);
                        }
                    }
                }
            }

            return Sf - Si + dS;
        }

        template <class MEntries>
        [[gnu::hot]]
        double virtual_move(size_t i, size_t v, size_t r, size_t nr, const entropy_args_t& ea,
                            MEntries& m_entries)
        {
            _states[i].get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

            double dS = 0;
            if (ea.adjacency)
            {
                if (ea.dense)
                {
                    dS = virtual_move_shared_dense_bernoulli(i, v, r, nr, ea.multigraph, _states[i]._g);
                }
                else
                {   
                    //todo sparse version of shared bernoulli log-likelihood
                    //if (ea.exact)
                    //    dS = _states[i].virtual_move_sparse<true>(v, r, nr, m_entries);
                    //else
                    //    dS = _states[i].virtual_move_sparse<false>(v, r, nr, m_entries);
                }
            }
            //todo compute description length if want to do model selection for number of blocks or number of shared blocks
            return dS;
        }


        size_t node_state(size_t i, size_t v)
        {
            return _states[i]._b[v];
        }

        bool skip_node(size_t i, size_t v)
        {
            return _states[i].node_weight(v) == 0;
        }

        template <class RNG>
        size_t move_proposal(size_t i, size_t v, RNG& rng)
        {
            if (!_allow_vacate && _states[i].is_last(v))
                return _null_move;
            size_t s = _states[i].sample_block(v, _c, _d, rng);
            if (s == node_state(i, v))
                return _null_move;
            return s;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t i, size_t v, size_t nr)
        {
            size_t r = _states[i]._b[v];
            if (r == nr)
                return std::make_tuple(0., 0.);

            double dS = virtual_move(i, v, r, nr, _entropy_args, _m_entries_list[i]);

            double a = 0;
            if (!std::isinf(_beta))
            {
                double pf = _states[i].get_move_prob(v, r, nr, _c, _d, false,
                                                 _m_entries_list[i]);
                double pb = _states[i].get_move_prob(v, nr, r, _c, _d, true,
                                                 _m_entries_list[i]);
                a = pb - pf;
            }
            return std::make_tuple(dS, a);
        }

        void perform_move(size_t i, size_t v, size_t nr)
        {
            //could update total ers and nrns to avoid for-loop in eterm_shared_dense_bernoulli
            _states[i].move_vertex(v, nr, _m_entries_list[i]);
        }

        bool is_deterministic()
        {
            return _deterministic;
        }

        bool is_sequential()
        {
            return _sequential;
        }

        auto& get_vlist(size_t i)
        {
            return _vlists[i];
        }
        auto& get_combined_vlist()
        {
            return _combined_vlist;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }
        size_t n_shared_blocks()
        {
            return _n_shared_blocks;
        }

        void step(int, size_t, size_t)
        {
        }

        template <class RNG>
        void init_iter(RNG&)
        {
        }
    };
};


#ifndef GRAPH_VIEWS
#define GRAPH_VIEWS all_graph_views
#endif

} // graph_tool namespace

#endif //SHARED_BLOCKMODEL_MCMC_HH
