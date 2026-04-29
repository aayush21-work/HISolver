// model_potential.hpp
// ===================
// User-supplied potential V(phi) and its derivative dV/dphi.
//
// This is the ONLY file you need to edit when changing potentials.
// The background and perturbation solvers include this header.
//
// Current model: plateau potential
//   V(phi) = V0 * phi^2 / (m^2 + phi^2)
//
// This gives ~60-70 e-folds for phi_ini = 7.5, m = 1, V0 = 1e-10.

#pragma once
#include <cmath>

// ── Potential parameters ──────────────────────────────────────────
static constexpr double V0    = 1.0e-10;   // amplitude  (Planck units)
static constexpr double m_phi = 1.0;       // mass scale (Planck units)

// ── V(phi) ────────────────────────────────────────────────────────
inline double V_phi(double phi) {
    double phi2 = phi * phi;
    double m2   = m_phi * m_phi;
    return V0 * phi2 / (m2 + phi2);
}

// ── dV/dphi ───────────────────────────────────────────────────────
inline double dV_phi(double phi) {
    double phi2 = phi * phi;
    double m2   = m_phi * m_phi;
    return V0 * 2.0 * m2 * phi / ((m2 + phi2) * (m2 + phi2));
}
