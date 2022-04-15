#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iostream>
#include <kevlar_bits/grid/grid_range.hpp>
#include <kevlar_bits/grid/gridder.hpp>
#include <kevlar_bits/grid/hyperplane.hpp>
#include <kevlar_bits/grid/tile.hpp>
#include <kevlar_bits/model/direct_bayes_binomial_control_k_treatment.hpp>
#include <kevlar_bits/util/macros.hpp>
#include <testutil/base_fixture.hpp>

namespace kevlar {
namespace {
struct MockHyperPlane;
using value_t = double;
using mat_t = DirectBayesBinomialControlkTreatment<value_t>::mat_t;
using vec_t = DirectBayesBinomialControlkTreatment<value_t>::vec_t;
using uint_t = uint32_t;
using tile_t = Tile<value_t>;
using hp_t = MockHyperPlane;
using gr_t = GridRange<value_t, uint_t, tile_t>;
using bckt_t = DirectBayesBinomialControlkTreatment<value_t>;

const Eigen::Vector<value_t, 1> critical_values{0.95};
const auto tol = 1e-8;
const int n_arm_size = 15;
const int n_integration_points = 50;
const size_t n_arms = 2;
const size_t n_samples = 250;
const size_t n_thetas = 10;
const value_t alpha_prior = 0.0005;
const value_t beta_prior = 0.000005;
const value_t efficacy_threshold = 0.3;
const value_t mu_sig_sq = 100;

struct MockHyperPlane : HyperPlane<value_t> {
    using base_t = HyperPlane<value_t>;
    using base_t::base_t;
};

vec_t get_efficacy_thresholds(int n) {
    Eigen::Vector<value_t, Eigen::Dynamic> efficacy_thresholds(n);
    efficacy_thresholds.fill(efficacy_threshold);
    return efficacy_thresholds;
}

gr_t get_grid_range() {
    using hp_t = MockHyperPlane;
    auto theta_1d = Gridder::make_grid(n_thetas, -1., 0.);
    auto radius = Gridder::radius(n_thetas, -1., 0.);

    colvec_type<value_t> normal(n_arms);
    std::vector<hp_t> hps;
    for (int i = 0; i < n_arms; ++i) {
        normal.setZero();
        normal(i) = -1;
        hps.emplace_back(normal, logit(efficacy_threshold));
    }

    // populate theta as the cartesian product of theta_1d
    gr_t grid_range(n_arms, ipow(n_thetas, n_arms));
    auto& thetas = grid_range.thetas();
    dAryInt bits(n_thetas, n_arms);
    for (size_t j = 0; j < grid_range.n_gridpts(); ++j) {
        for (size_t i = 0; i < n_arms; ++i) {
            thetas(i, j) = theta_1d[bits()[i]];
        }
        ++bits;
    }

    // populate radii as fixed radius
    grid_range.radii().array() = radius;

    // create tile information
    grid_range.create_tiles(hps);
    grid_range.prune();

    return grid_range;
}

DirectBayesBinomialControlkTreatment<value_t> get_test_class() {
    // TODO: n_samples should really be n_arm_size, but causes the rej_len test
    // to fail
    bckt_t b_new(n_arms, n_samples, critical_values,
                 get_efficacy_thresholds(n_arms));
    return b_new;
}

TEST(Test, TestConditionalExceedProbGivenSigma) {
    Eigen::Vector<value_t, 4> logit_efficacy_thresholds;
    logit_efficacy_thresholds.fill(-0.40546511);
    for (bool use_fast : {true, false}) {
        mat_t got = DirectBayesBinomialControlkTreatment<value_t>::
            conditional_exceed_prob_given_sigma(
                1.10517092, 0.1,
                Eigen::Vector<value_t, 4>{12.32, 10.08, 11.22, 10.08},
                Eigen::Vector<value_t, 4>{0.24116206, -0.94446161, 0.66329422,
                                          0.94446161},
                logit_efficacy_thresholds,
                Eigen::Vector<value_t, 4>{0, 0, 0, 0}, use_fast);
        Eigen::Vector<value_t, 4> want;
        want << 0.9892854091921082, 0.0656701203047288, 0.999810960134644,
            0.9999877861068269;
        EXPECT_TRUE(got.isApprox(want, tol));
        got = DirectBayesBinomialControlkTreatment<value_t>::
            conditional_exceed_prob_given_sigma(
                1.01445965e-8, 0.1,
                Eigen::Vector<value_t, 4>{12.32, 10.08, 11.22, 10.08},
                Eigen::Vector<value_t, 4>{0.24116206, -0.94446161, 0.66329422,
                                          0.94446161},
                logit_efficacy_thresholds,
                Eigen::Vector<value_t, 4>{0, 0, 0, 0}, use_fast);
        want << 0.9999943915784785, 0.999994391552775, 0.9999943915861994,
            0.9999943915892988;
        EXPECT_TRUE(got.isApprox(want, tol));
    }
};

TEST(Test, TestGetPosteriorExcedanceProbs) {
    const auto [quadrature_points, weighted_density_logspace] =
        DirectBayesBinomialControlkTreatment<value_t>::get_quadrature(
            alpha_prior, beta_prior, n_integration_points, n_arm_size);
    vec_t phat = Eigen::Vector<value_t, 4>{3, 8, 5, 4};
    phat.array() /= 15;
    Eigen::Vector<value_t, 4> want{0.64462095, 0.80224266, 0.71778699,
                                   0.67847136};
    for (bool use_optimized : {true, false}) {
        auto got = DirectBayesBinomialControlkTreatment<
            value_t>::get_posterior_exceedance_probs(phat, quadrature_points,
                                                     weighted_density_logspace,
                                                     get_efficacy_thresholds(4),
                                                     n_arm_size, mu_sig_sq,
                                                     use_optimized);
        EXPECT_TRUE(got.isApprox(want, tol));
    }
};

TEST(Test, TestFasterInvert) {
    auto v = Eigen::Vector<value_t, 4>{1, 2, 3, 4};
    value_t d = 0.5;
    const auto got =
        DirectBayesBinomialControlkTreatment<value_t>::faster_invert(
            1. / v.array(), d);
    mat_t m = v.asDiagonal();
    m.array() += d;
    mat_t want = m.inverse();
    EXPECT_TRUE(want.isApprox(got, tol));
};

TEST(Test, GetGridRange) {
    auto grid_range = get_grid_range();
    EXPECT_EQ(grid_range.n_tiles(0), 1);
    EXPECT_EQ(grid_range.n_tiles(1), 1);
    EXPECT_EQ(grid_range.n_tiles(2), 1);
    EXPECT_EQ(grid_range.n_tiles(3), 1);
};

TEST(Test, TestRejLen) {
    auto model = get_test_class();
    auto grid_range = get_grid_range();
    model.set_grid_range(grid_range);
    auto state = model.make_state();
    size_t seed = 3214;
    std::mt19937 gen;
    gen.seed(seed);
    state->gen_rng(gen);
    state->gen_suff_stat();
    colvec_type<uint_t> actual(grid_range.n_tiles());
    state->rej_len(actual);
    colvec_type<uint_t> expected(grid_range.n_tiles());
    expected << 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1;
    EXPECT_EQ(expected, actual);
};
}  // namespace
}  // namespace kevlar
