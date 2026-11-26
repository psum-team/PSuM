#pragma once

#include <algorithm>
#include <cmath>

inline double calc_tau(double T_alpha_eV, double T_beta_eV,
                double m_alpha, double m_beta,
                double n_alpha, double n_beta,
                double q_alpha, double q_beta) {
    double eV_to_J = 1.602176634e-19;
    double eps0_loc = 8.854187817e-12;
    double hbar_loc = 1.054571817e-34;
    double kT_alpha = T_alpha_eV * eV_to_J;
    double kT_beta = T_beta_eV * eV_to_J;
    double v_th = std::sqrt(kT_alpha / m_alpha + kT_beta / m_beta);
    double m12 = m_alpha * m_beta / (m_alpha + m_beta);
    double lambdaD_sq_inv = n_alpha * q_alpha * q_alpha / (eps0_loc * kT_alpha)
                          + n_beta * q_beta * q_beta / (eps0_loc * kT_beta);
    double lambdaD = 1.0 / std::sqrt(lambdaD_sq_inv);
    double b_min = std::max(hbar_loc / (m12 * v_th), std::abs(q_alpha * q_beta) / (v_th * v_th * 4.0 * M_PI * eps0_loc * m12));
    double Lambda = std::max(lambdaD / b_min, 1.01);
    double lnLambda = std::log(Lambda);
    double coeff = 3.0 * std::pow(4.0 * M_PI * eps0_loc, 2) * m_alpha * m_beta;
    double denom = 8.0 * std::sqrt(2.0 * M_PI) * n_beta
                   * q_alpha * q_alpha * q_beta * q_beta * lnLambda;
    double thermal = std::pow(kT_alpha / m_alpha + kT_beta / m_beta, 1.5);
    return coeff / denom * thermal;
}
