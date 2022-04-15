#include <kevlar_bits/stat/log_rank_test.hpp>
#include <kevlar_bits/util/algorithm.hpp>
#include <testutil/base_fixture.hpp>

namespace kevlar {
namespace stat {
namespace {

using value_t = double;
using uint_t = uint32_t;
using stat_t = LogRankTest<value_t, uint_t>;

inline constexpr double tol = 2e-15;

TEST(LogRankTestSuite, distinct_outcomes) {
    size_t n_c = 20;
    size_t n_t = 30;

    colvec_type<value_t> control(n_c);
    colvec_type<value_t> treatment(n_t);
    colvec_type<value_t> censor_times(n_c + n_t);
    colvec_type<value_t> expected(n_c + n_t);

    // initialize data
    control << 0.05541601363179369, 0.04387435311767324, 0.04209432757256784,
        0.13130642149327754, 0.87171591010661675, 0.37023849846880258,
        0.16250612466037723, 0.28803598124012464, 0.04427762436132266,
        0.41876998759723877, 0.22945795105994382, 0.37265990687588985,
        1.33210906885963132, 0.31753784020194742, 0.31172657212608890,
        0.56490068425421480, 0.19715345896235326, 0.10145542799901511,
        0.17719955478863020, 0.71198893496646976;

    treatment << 1.0842780206616767, 0.4968249791213689, 0.9558539266534452,
        0.1791767285180253, 0.1004039871544149, 0.9775548367886930,
        6.6873865745987864, 1.9819461246972483, 1.6838056676416289,
        2.4244684854274929, 0.0629535918690365, 0.5473144473570928,
        2.2305201508678905, 0.3437674831614287, 1.7275774954561653,
        0.5096975238867910, 1.2250241611978696, 1.2694977900761568,
        0.3970058291739337, 1.8241235425590265, 1.7369035536317814,
        2.1828744048550002, 2.1167320176576649, 0.9368942528207963,
        0.5089239803625836, 2.1843321894441714, 1.6799928852578678,
        0.8685376622550491, 3.3916087876341243, 0.7132473732093098;

    censor_times << 0.04209432757256784, 0.04387435311767324,
        0.04427762436132266, 0.05541601363179369, 0.06295359186903650,
        0.10040398715441493, 0.10145542799901511, 0.13130642149327754,
        0.16250612466037723, 0.17719955478863020, 0.17917672851802530,
        0.19715345896235326, 0.22945795105994382, 0.28803598124012464,
        0.31172657212608890, 0.31753784020194742, 0.34376748316142869,
        0.37023849846880258, 0.37265990687588985, 0.39700582917393373,
        0.41876998759723877, 0.49682497912136891, 0.50892398036258357,
        0.50969752388679102, 0.54731444735709278, 0.56490068425421480,
        0.71198893496646976, 0.71324737320930975, 0.86853766225504914,
        0.87171591010661675, 0.93689425282079630, 0.95585392665344515,
        0.97755483678869304, 1.08427802066167667, 1.22502416119786961,
        1.26949779007615682, 1.33210906885963132, 1.67999288525786783,
        1.68380566764162887, 1.72757749545616535, 1.73690355363178139,
        1.82412354255902653, 1.98194612469724829, 2.11673201765766494,
        2.18287440485500017, 2.18433218944417140, 2.23052015086789046,
        2.42446848542749294, 3.39160878763412432, 6.68738657459878638;

    expected << 1.500000000000000, 3.078203517587940, 4.742318400765956,
        6.501154647866653, 3.871061000791012, 2.245453063178386,
        3.558817062103615, 5.040798697615941, 6.677732943973894,
        8.467873796335361, 6.738456428063629, 8.479434880700760,
        10.388770496738115, 12.479923795131521, 14.771989940568323,
        17.290235777109910, 15.651207480235721, 18.248911859065245,
        21.131780727684685, 19.721736888718091, 22.773822013620290,
        21.535643959627233, 20.334593845284630, 19.169043515177890,
        18.037469254141662, 20.871625094169453, 24.108790809323697,
        23.355787844066363, 22.598117058655671, 26.111818706466686,
        25.656006567539976, 25.186649057402967, 24.702512233518405,
        24.202172747626026, 23.683975499482813, 23.145978480708962,
        26.771333904241825, 26.771333904241825, 26.771333904241811,
        26.771333904241811, 26.771333904241839, 26.771333904241839,
        26.771333904241811, 26.771333904241811, 26.771333904241811,
        26.771333904241779, 26.771333904241811, 26.771333904241811,
        26.771333904241779, 26.771333904241747;

    // end initialize data

    // Run my logrank test
    sort_cols(control);
    sort_cols(treatment);
    stat_t lrt(control, treatment);
    lrt.run();

    // compare with expected
    for (int i = 0; i < censor_times.size(); ++i) {
        value_t actual = lrt.stat(censor_times[i], false);
        EXPECT_NEAR(actual * actual, expected[i], tol * expected[i]);
    }
}

TEST(LogRankTestSuite, with_repeat_times) {
    size_t n_c = 20;
    size_t n_t = 30;
    size_t n_unique = 39;

    colvec_type<value_t> control(n_c);
    colvec_type<value_t> treatment(n_t);
    colvec_type<value_t> censor_times(n_unique);
    colvec_type<value_t> expected(n_unique);

    // initialize data
    control << 0.04209432757256784, 0.04387435311767324, 0.04387435311767324,
        0.04387435311767324, 0.10145542799901511, 0.13130642149327754,
        0.16250612466037723, 0.16250612466037723, 0.16250612466037723,
        0.16250612466037723, 0.16250612466037723, 0.31172657212608890,
        0.31753784020194742, 0.37023849846880258, 0.37265990687588985,
        0.41876998759723877, 0.49682497912136891, 0.71198893496646976,
        0.87171591010661675, 1.33210906885963132;

    treatment << 0.0629535918690365, 0.1004039871544149, 0.16250612466037723,
        0.3437674831614287, 0.3970058291739337, 0.4968249791213689,
        0.4968249791213689, 0.4968249791213689, 0.4968249791213689,
        0.7132473732093098, 0.8685376622550491, 0.9368942528207963,
        0.9558539266534452, 0.9775548367886930, 1.0842780206616767,
        1.2250241611978696, 1.2694977900761568, 1.6799928852578678,
        1.6838056676416289, 1.7275774954561653, 31.7369035536317814,
        1.8241235425590265, 1.9819461246972483, 2.1167320176576649,
        2.1828744048550002, 2.1843321894441714, 2.2305201508678905,
        2.4244684854274929, 3.3916087876341243, 6.6873865745987864;

    censor_times << 0.04209432757256784, 0.04387435311767324,
        0.06295359186903650, 0.10040398715441493, 0.10145542799901511,
        0.13130642149327754, 0.16250612466037723, 0.31172657212608890,
        0.31753784020194742, 0.34376748316142869, 0.37023849846880258,
        0.37265990687588985, 0.39700582917393373, 0.41876998759723877,
        0.4968249791213689, 0.71198893496646976, 0.71324737320930975,
        0.86853766225504914, 0.87171591010661675, 0.93689425282079630,
        0.95585392665344515, 0.97755483678869304, 1.08427802066167667,
        1.22502416119786961, 1.26949779007615682, 1.33210906885963132,
        1.67999288525786783, 1.68380566764162887, 1.72757749545616535,
        1.73690355363178139, 1.82412354255902653, 1.98194612469724829,
        2.11673201765766494, 2.18287440485500017, 2.18433218944417140,
        2.23052015086789046, 2.42446848542749294, 3.39160878763412432,
        6.68738657459878638;

    expected << 1.500000000000000, 6.436308967534842, 3.796457677824932,
        2.179541174565480, 3.488114424032636, 4.967559009746862,
        12.052353255752125, 14.347605006197698, 16.872129814887373,
        15.225288158872281, 17.828209952644901, 20.719866629570301,
        19.301340421066527, 22.362693766896985, 21.212381923342569,
        24.541159204489787, 23.750182479812096, 22.955575460583653,
        26.568729040321980, 26.090557639595716, 25.598610993042723,
        25.091636662412512, 24.568190450723225, 24.026593614875935,
        23.464877156586645, 27.187653620156468, 27.187653620156468,
        27.187653620156468, 27.187653620156446, 27.187653620156446,
        27.187653620156446, 27.187653620156446, 27.187653620156446,
        27.187653620156446, 27.187653620156446, 27.187653620156478,
        27.187653620156446, 27.187653620156446, 27.187653620156414;
    // end initialize data

    // Run my logrank test
    sort_cols(control);
    sort_cols(treatment);
    stat_t lrt(control, treatment);
    lrt.run();

    // compare with expected
    for (int i = 0; i < censor_times.size(); ++i) {
        value_t actual = lrt.stat(censor_times[i], false);
        EXPECT_NEAR(actual * actual, expected[i], tol * expected[i]);
    }
}

}  // namespace
}  // namespace stat
}  // namespace kevlar
