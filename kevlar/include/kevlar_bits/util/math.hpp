#pragma once
#define _USE_MATH_DEFINES
#include <cmath>

namespace kevlar {
namespace details {

template <class ValueType, class IntType>
inline ValueType ipow_pos(ValueType base, IntType exp)
{
    if (exp == 1) return base;
    if (exp % 2 == 0) {
        auto t = ipow_pos(base, exp / 2);
        return t * t;
    } else {
        return ipow_pos(base, exp-1) * base;
    }
}

} // namespace details 

template <class ValueType, class IntType>
inline auto ipow(ValueType base, IntType exp)
{
    if (exp == 0) return ValueType(1);
    if (exp < 0) return ValueType(1) / ipow(base, -exp);
    return details::ipow_pos(base, exp);
};

// Inverse Normal CDF (Acklam's algorithm)
// https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
template <class ValueType>
constexpr inline auto qnorm(ValueType p)
{
    constexpr double a1 = -39.69683028665376;
    constexpr double a2 = 220.9460984245205;
    constexpr double a3 = -275.9285104469687;
    constexpr double a4 = 138.3577518672690;
    constexpr double a5 =-30.66479806614716;
    constexpr double a6 = 2.506628277459239;

    constexpr double b1 = -54.47609879822406;
    constexpr double b2 = 161.5858368580409;
    constexpr double b3 = -155.6989798598866;
    constexpr double b4 = 66.80131188771972;
    constexpr double b5 = -13.28068155288572;

    constexpr double c1 = -0.007784894002430293;
    constexpr double c2 = -0.3223964580411365;
    constexpr double c3 = -2.400758277161838;
    constexpr double c4 = -2.549732539343734;
    constexpr double c5 = 4.374664141464968;
    constexpr double c6 = 2.938163982698783;

    constexpr double d1 = 0.007784695709041462;
    constexpr double d2 = 0.3224671290700398;
    constexpr double d3 = 2.445134137142996;
    constexpr double d4 = 3.754408661907416;

    constexpr double sqrt_2 = 1.4142135623730951455;
    constexpr double sqrt_2_pi = 2.5066282746310002416;

    // Define break-points.
    constexpr double p_low =  0.02425;
    constexpr double p_high = 1 - p_low;
    long double q, r, e, u;
    long double x = 0.0;

    // Rational approximation for lower region.
    if (0 < p && p < p_low) {
        q = std::sqrt(-2*std::log(p));
        x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }

    // Rational approximation for central region.
    if (p_low <= p && p <= p_high) {
        q = p - 0.5;
        r = q*q;
        x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
    }

    // Rational approximation for upper region.
    if (p_high < p && p < 1) {
        q = std::sqrt(-2*std::log(1-p));
        x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }

    // Pseudo-code algorithm for refinement
    if((0 < p) && (p < 1)){
        e = 0.5 * std::erfc(-x/sqrt_2) - p;
        u = e * sqrt_2_pi * std::exp(x*x/2);
        x -= u/(1 + x*u/2);
    }

    return x;
}

} // namespace kevlar

#undef _USE_MATH_DEFINES
