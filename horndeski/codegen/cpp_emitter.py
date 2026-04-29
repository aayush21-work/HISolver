"""
codegen/cpp_emitter.py
======================
Takes SymPy expressions from algebra/background.py and
algebra/perturbations.py, substitutes a user-defined Horndeski model,
and emits a self-contained C++ header  model_generated.hpp  that the
numerical solvers (background.cpp, perturbations.cpp) can include.

Usage
-----
    from horndeski.codegen.cpp_emitter import emit_model

    from horndeski.symbols import phi_sym, X_sym, Mpl, M
    from sympy import Function

    V    = Function('V')(phi_sym)
    G2   = X_sym - V
    G3   = 0 * X_sym
    G4   = Mpl**2/2 + X_sym/M**2
    G5   = 0 * X_sym

    emit_model(
        G2, G3, G4, G5,
        model_name   = "NMDC",
        V_expr       = V,
        dV_expr      = V.diff(phi_sym),
        output_path  = "model_generated.hpp",
        params       = {"Mpl": 1.0, "M": 1e-3},
    )

The generated header exposes these C functions (all inline):

    double rho_phi   (double phi, double dphi, double H)
    double ddphi_rhs (double phi, double dphi, double H, double dH)
    double dH_rhs    (double phi, double dphi, double H, double ddphi)
    double epsilon_H (double phi, double dphi, double H)

    double QT        (double phi, double dphi, double H)
    double cT2       (double phi, double dphi, double H)
    double Qs        (double phi, double dphi, double H)
    double cs2       (double phi, double dphi, double H, double dH, double ddphi)

    double V_phi     (double phi)
    double dV_phi    (double phi)
"""

import os
from sympy import (
    symbols, Function, Rational, diff,
    simplify, cse, numbered_symbols, Symbol
)
from sympy.printing.c import C99CodePrinter
from horndeski.symbols import (
    phi_sym, X_sym, dphi, H, dH, ddphi, Mpl, M
)
from horndeski.algebra.background import (
    energy_density, pressure, KG_coefficients, substitute_model
)
from horndeski.algebra.perturbations import substitute_perturbations

# =============================================================================
# Helper: substitution dict  SymPy functions -> plain C variable names
# =============================================================================

def _make_subs(V_expr, dV_expr):
    """
    Build a substitution dict that replaces SymPy Function objects
    (dphi(t), H(t), etc.) with plain symbols suitable for C code.
    """
    # plain C variable names
    phi_c   = Symbol('phi')
    dphi_c  = Symbol('dphi')
    ddphi_c = Symbol('ddphi')
    H_c     = Symbol('H')
    dH_c    = Symbol('dH')
    Mpl_c   = Symbol('Mpl')
    M_c     = Symbol('M')
    V_c     = Symbol('V_phi_val')
    dV_c    = Symbol('dV_phi_val')

    return {
        dphi    : dphi_c,
        ddphi   : ddphi_c,
        H       : H_c,
        dH      : dH_c,
        phi_sym : phi_c,
        Mpl     : Mpl_c,
        M       : M_c,
        V_expr  : V_c,
        dV_expr : dV_c,
    }


# =============================================================================
# Helper: emit a single inline C function
# =============================================================================

def _emit_function(name, args, expr, subs, printer,
                   V_expr=None, dV_expr=None,
                   params=None):
    """
    Emit a single C99 inline double function.

    Parameters
    ----------
    name   : C function name  (e.g. "rho_phi")
    args   : list of (C_type, C_name) pairs
    expr   : SymPy expression
    subs   : substitution dict from _make_subs
    printer: C99CodePrinter instance
    V_expr : SymPy expression for V(phi) — will be pre-computed if present
    dV_expr: SymPy expression for dV/dphi
    params : dict of {symbol_name: value} for hardcoded parameters
    """
    lines = []

    # function signature
    arg_str = ", ".join(f"double {n}" for _, n in args)
    lines.append(f"inline double {name}({arg_str}) {{")

    # hardcode parameter values if provided
    if params:
        for pname, pval in params.items():
            lines.append(f"    const double {pname} = {pval};")

    # pre-compute V and dV if needed
    if V_expr is not None:
        lines.append(f"    const double V_phi_val  = V_phi(phi);")
    if dV_expr is not None:
        lines.append(f"    const double dV_phi_val = dV_phi(phi);")

    # apply substitution and generate C expression
    expr_c = expr.subs(subs)

    # use CSE (common subexpression elimination) for cleaner output
    replacements, reduced = cse([expr_c],
                                 symbols=numbered_symbols('_t'),
                                 optimizations='basic')

    for sym, val in replacements:
        lines.append(f"    const double {sym} = {printer.doprint(val)};")

    lines.append(f"    return {printer.doprint(reduced[0])};")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main emitter
# =============================================================================

def emit_model(G2_model, G3_model, G4_model, G5_model,
               model_name  = "Horndeski",
               V_expr      = None,
               dV_expr     = None,
               output_path = "model_generated.hpp",
               params      = None):
    """
    Generate model_generated.hpp for a given Horndeski model.

    Parameters
    ----------
    G2_model .. G5_model : SymPy expressions in (phi_sym, X_sym)
    model_name  : string label for the header comment
    V_expr      : SymPy expression for V(phi_sym) — used for V_phi() function
    dV_expr     : SymPy expression for dV/dphi
    output_path : path to write the generated header
    params      : dict {param_name: value} for numerical parameter values
                  e.g. {"Mpl": 1.0, "M": 1e-3}
                  If None, parameters appear as C variables (must be set elsewhere)
    """

    printer = C99CodePrinter()

    # ── Step 1: substitute model into all expressions ────────────────
    print(f"[codegen] Substituting {model_name} model into background equations...")

    # Background
    rho_expr = substitute_model(energy_density(), G2_model, G3_model,
                                 G4_model, G5_model)
    rho_expr = simplify(rho_expr.subs(X_sym, dphi**2/2))
    # rho_phi is the RHS of 3H^2 = rho -- move -3Mpl^2 H^2 to LHS
    rho_rhs  = simplify(rho_expr + 3*Mpl**2*H**2)

    P_ddphi_expr, P_dH_expr, P0_expr = [
        simplify(substitute_model(c, G2_model, G3_model, G4_model, G5_model)
                 .subs(X_sym, dphi**2/2))
        for c in KG_coefficients()
    ]

    # Raychaudhuri: -2*dH = rho + p  -> solve for dH
    p_expr = substitute_model(pressure(), G2_model, G3_model,
                               G4_model, G5_model)
    p_expr = simplify(p_expr.subs(X_sym, dphi**2/2))

    from sympy import solve, Eq
    ray_eq  = Eq(-2*dH, rho_expr + p_expr)
    dH_sol  = solve(ray_eq, dH)[0]
    dH_rhs  = simplify(dH_sol)

    # ddphi from KG: solve for ddphi using dH_rhs
    ddphi_rhs = simplify(
        -(P_dH_expr * dH_rhs + P0_expr) / P_ddphi_expr
    )

    # epsilon_H = -dH/H^2  (using dH_rhs)
    epsilon_expr = simplify(-dH_rhs / H**2)

    print(f"[codegen] Computing perturbation coefficients...")

    # Perturbations
    res = substitute_perturbations(G2_model, G3_model, G4_model, G5_model)
    QT_expr  = res['QT']
    cT2_expr = res['cT2']
    Qs_expr  = res['Qs']
    cs2_expr = res['cs2']

    # ── Step 2: build substitution dict ─────────────────────────────
    if V_expr is None:
        # create a placeholder
        V_expr  = Function('V')(phi_sym)
        dV_expr = V_expr.diff(phi_sym)

    subs = _make_subs(V_expr, dV_expr)

    # ── Step 3: assemble the header ──────────────────────────────────
    print(f"[codegen] Emitting C++ header to {output_path}...")

    lines = []

    # ---- file header ----
    lines.append(f"// model_generated.hpp")
    lines.append(f"// Auto-generated by horndeski/codegen/cpp_emitter.py")
    lines.append(f"// Model: {model_name}")
    lines.append(f"//")
    lines.append(f"// DO NOT EDIT — regenerate by running cpp_emitter.py")
    lines.append(f"//")
    lines.append(f"// Conventions:")
    lines.append(f"//   - Units: 8*pi*G = 1  (Mpl = 1 unless overridden)")
    lines.append(f"//   - phi   = inflaton field value")
    lines.append(f"//   - dphi  = d phi / dt  (cosmic time derivative)")
    lines.append(f"//   - ddphi = d^2 phi / dt^2")
    lines.append(f"//   - H     = Hubble parameter")
    lines.append(f"//   - dH    = d H / dt")
    lines.append(f"//   - X     = dphi^2 / 2  (kinetic term)")
    lines.append(f"//")
    lines.append(f"#pragma once")
    lines.append(f"#include <cmath>")
    lines.append(f"")

    # ---- parameter block ----
    if params:
        lines.append(f"// ── Model parameters ─────────────────────────────────────────")
        for pname, pval in params.items():
            lines.append(f"static constexpr double {pname} = {pval};")
        lines.append(f"")

    # ---- V and dV (forward declarations — user must implement) ------
    lines.append(f"// ── Potential: implement these in model.hpp or model.cpp ─────────")
    lines.append(f"// double V_phi  (double phi);   // V(phi)")
    lines.append(f"// double dV_phi (double phi);   // dV/dphi")
    lines.append(f"//")
    lines.append(f"// Or inline them here:")
    lines.append(f"")

    if V_expr is not None and not isinstance(V_expr, Function):
        V_c = printer.doprint(V_expr.subs(subs))
        lines.append(f"inline double V_phi(double phi) {{")
        lines.append(f"    return {V_c};")
        lines.append(f"}}")
        lines.append(f"")

    # ---- background functions ----------------------------------------
    lines.append(f"// ── Background ───────────────────────────────────────────────────")
    lines.append(f"//")
    lines.append(f"// Friedmann constraint:  3*H^2 = rho_phi(phi, dphi, H)")
    lines.append(f"// Use this to solve for H given phi, dphi.")
    lines.append(f"")

    # needs_V flags
    # needs_V flags — check if V or dV appear after substitution
    from sympy import Symbol as Sym
    V_c   = Sym('V_phi_val')
    dV_c  = Sym('dV_phi_val')

    def _needs(expr, sym_c):
        return sym_c in expr.subs(subs).free_symbols

    lines.append(_emit_function(
        "rho_phi",
        [("double","phi"), ("double","dphi"), ("double","H")],
        rho_rhs, subs, printer,
        V_expr  = V_expr  if _needs(rho_rhs,   V_c)  else None,
        dV_expr = dV_expr if _needs(rho_rhs,   dV_c) else None,
        params  = params,
    ))

    lines.append(_emit_function(
        "dH_rhs",
        [("double","phi"), ("double","dphi"), ("double","H"), ("double","ddphi")],
        dH_rhs, subs, printer,
        V_expr  = V_expr  if _needs(dH_rhs,    V_c)  else None,
        dV_expr = dV_expr if _needs(dH_rhs,    dV_c) else None,
        params  = params,
    ))

    lines.append(_emit_function(
        "ddphi_rhs",
        [("double","phi"), ("double","dphi"), ("double","H")],
        ddphi_rhs, subs, printer,
        V_expr  = V_expr  if _needs(ddphi_rhs, V_c)  else None,
        dV_expr = dV_expr if _needs(ddphi_rhs, dV_c) else None,
        params  = params,
    ))

    lines.append(_emit_function(
        "epsilon_H",
        [("double","phi"), ("double","dphi"), ("double","H")],
        epsilon_expr, subs, printer,
        V_expr  = V_expr  if _needs(epsilon_expr, V_c)  else None,
        dV_expr = dV_expr if _needs(epsilon_expr, dV_c) else None,
        params  = params,
    ))

    # ---- perturbation functions --------------------------------------
    lines.append(f"// ── Perturbations ────────────────────────────────────────────────")
    lines.append(f"//")
    lines.append(f"// Stability conditions:")
    lines.append(f"//   QT  > 0   (no tensor ghost)")
    lines.append(f"//   Qs  > 0   (no scalar ghost)")
    lines.append(f"//   cT2 > 0   (no tensor gradient instability)")
    lines.append(f"//   cs2 > 0   (no scalar gradient instability)")
    lines.append(f"")

    for fname, expr in [("QT",  QT_expr),
                         ("cT2", cT2_expr),
                         ("Qs",  Qs_expr)]:
        lines.append(_emit_function(
            fname,
            [("double","phi"), ("double","dphi"), ("double","H")],
            expr, subs, printer,
            params=params,
        ))

    # cs2 needs dH and ddphi
    lines.append(_emit_function(
        "cs2",
        [("double","phi"), ("double","dphi"), ("double","H"),
         ("double","dH"),  ("double","ddphi")],
        cs2_expr, subs, printer,
        params=params,
    ))

    # ---- Mukhanov-Sasaki pump field helper ---------------------------
    lines.append(f"// ── Mukhanov-Sasaki pump fields ───────────────────────────────────")
    lines.append(f"//")
    lines.append(f"// z^2 = a^2 * 2*Qs  ->  used in scalar MS equation")
    lines.append(f"// u^2 = a^2 * QT/2  ->  used in tensor MS equation")
    lines.append(f"//")
    lines.append(f"// z''/z and u''/u are computed numerically in perturbations.cpp")
    lines.append(f"// using finite differences on the background trajectory.")
    lines.append(f"")
    lines.append(f"inline double z2_over_a2(double phi, double dphi, double H) {{")
    lines.append(f"    return 2.0 * Qs(phi, dphi, H);")
    lines.append(f"}}")
    lines.append(f"")
    lines.append(f"inline double u2_over_a2(double phi, double dphi, double H) {{")
    lines.append(f"    return 0.5 * QT(phi, dphi, H);")
    lines.append(f"}}")
    lines.append(f"")

    # ── write file ────────────────────────────────────────────────────
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"[codegen] Done. Written to: {output_path}")
    print(f"[codegen] Functions emitted:")
    print(f"          Background:    rho_phi, dH_rhs, ddphi_rhs, epsilon_H")
    print(f"          Perturbations: QT, cT2, Qs, cs2")
    print(f"          MS helpers:    z2_over_a2, u2_over_a2")

    return content


# =============================================================================
# Run directly to generate NMDC model header
# =============================================================================

if __name__ == "__main__":
    from horndeski.symbols import phi_sym, X_sym, Mpl, M
    from sympy import Function, symbols as syms

    print("=" * 60)
    print("  Generating model_generated.hpp for NMDC model")
    print("  G4 = Mpl^2/2 + X/M^2")
    print("=" * 60)

    V   = Function('V')(phi_sym)
    dV  = V.diff(phi_sym)

    G2 = X_sym - V
    G3 = 0 * X_sym
    G4 = Mpl**2/2 + X_sym/M**2
    G5 = 0 * X_sym

    content = emit_model(
        G2, G3, G4, G5,
        model_name  = "NMDC (Einstein tensor non-minimal derivative coupling)",
        V_expr      = V,
        dV_expr     = dV,
        output_path = "model_generated.hpp",
        params      = {"Mpl": 1.0, "M": 1e-5},
    )

    print()
    print("=" * 60)
    print("  Preview of generated header:")
    print("=" * 60)
    print(content[:3000])
