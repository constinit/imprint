#pragma once
#include <pybind11/pybind11.h>

#include <kevlar_bits/util/types.hpp>

namespace kevlar {
namespace model {

namespace py = pybind11;

template <class SGSB>
void add_base_tests(py::module_& m) {
    using sbs_t = SGSB;
    using ss_t = typename sbs_t::sim_state_t;

    m.def("test_py_sgs", [](const sbs_t& s) { return s.make_sim_state(); });
    m.def("test_py_ss_simulate", [](ss_t& s) {
        using gen_t = typename sbs_t::gen_t;
        using uint_t = typename sbs_t::uint_t;
        gen_t gen;
        colvec_type<uint_t> rej_len(10);
        s.simulate(gen, rej_len);
        return rej_len;
    });
    m.def("test_py_ss_score", [](const ss_t& s) {
        using value_t = typename sbs_t::value_t;
        colvec_type<value_t> score(3);
        s.score(0, score);
        return score;
    });
}

}  // namespace model
}  // namespace kevlar
