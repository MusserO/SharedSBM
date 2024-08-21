#include <boost/python.hpp> 
#include <vector>

namespace py = boost::python;

std::vector<std::vector<size_t>> python_list_of_lists_to_vec_of_vecs(py::list iterable) {
    std::vector<std::vector<size_t>> result;
    for (py::ssize_t i = 0; i < py::len(iterable); ++i) {
        py::object inner_list = iterable[i];
        std::vector<size_t> inner_vector;
        for (py::ssize_t j = 0; j < py::len(inner_list); ++j) {
            inner_vector.push_back(py::extract<size_t>(inner_list[j]));
        }
        result.push_back(inner_vector);
    }
    return result;
}

template <class State>
std::vector<State> python_list_of_states_to_vec_of_states(py::list iterable) {
    std::vector<State> result;
    for (py::ssize_t i = 0; i < py::len(iterable); ++i) {
        result.push_back(py::extract<State>(iterable[i]));
    }
    return result;
}
