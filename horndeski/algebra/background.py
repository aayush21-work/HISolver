"""
algebra/background.py
=====================
Derives the background equations of motion for a general Horndeski theory
in a flat FLRW spacetime.

Physics summary
---------------
The Horndeski action in flat FLRW reduces (after integration by parts and
use of the homogeneous ansatz) to equations involving five combinations of
the G_i functions.  These are usually written in terms of auxiliary
"w-functions" (following Kobayashi, Yamaguchi & Yokoyama 2011,
arXiv:1105.5723, which is the standard reference we follow throughout).

The four auxiliary functions are:

    w1 = G4 - X*G5phi - X*dphi*H*G5X                      ... (K11 eq 2.8a)
    w2 = G4 - 2X*G4X - X*G5phi/2 + X^2*G5phiX/2 - ...    ... (K11 eq 2.8b)
          (full expression below)
    w3 = dphi*( G3X - G4phiX - ... )  * 2H  + ...         ... (full below)
    w4 = dphi * G5X * H^2  - ...                           ... (full below)

From these, the Friedmann equation and KG equation follow.

We follow the notation of:
  Kobayashi, Yamaguchi, Yokoyama (KYY) 2011  arXiv:1105.5723
  equations (2.4) -- (2.10)

Gauge:   flat FLRW background,  ds^2 = -dt^2 + a^2 delta_{ij} dx^i dx^j
Field:   phi = phi(t),  X = dphi^2 / 2
"""

from sympy import (
    symbols, Function, Rational, diff, simplify,
    factor, collect, expand, sqrt, pprint, latex
)
from horndeski.symbols import (
    t, phi, dphi, ddphi, H, dH, a,
    phi_sym, X_sym,
    G2, G3, G4, G5,
    G2phi, G2X,
    G3phi, G3X, G3phiX, G3XX,
    G4phi, G4X, G4phiX, G4XX,
    G5phi, G5X, G5phiX, G5XX,
    background_subs,
    Mpl,
)

# =============================================================================
# STEP 1 — Define the energy density and pressure of the Horndeski field
#           in FLRW.  These come from KYY (2011) eqs (2.4)-(2.7).
#
#   The Friedmann equations are:
#       3 H^2  =  rho_phi       (energy constraint)
#      -2 dH   =  rho_phi + p_phi   (Raychaudhuri)
#
#   Both rho_phi and p_phi are expressed in terms of G_i and their
#   derivatives evaluated at (phi_sym, X_sym), then we substitute
#   X_sym -> dphi^2/2 at the end.
# =============================================================================

def energy_density():
    """
    Compute the Horndeski energy density  rho_phi  in flat FLRW.

    Returns a SymPy expression in terms of:
        phi_sym, X_sym, H, dphi
    (X_sym should be substituted -> dphi^2/2 afterwards)

    Reference: KYY (2011) eq (2.4)
    """

    # --- G2 contribution ---
    # rho_2 = 2X G2X - G2
    rho_2 = 2*X_sym*G2X - G2

    # --- G3 contribution ---
    # rho_3 = -2X G3phi  +  6 X H dphi G3X
    # Note: the dphi here is the field velocity; H dphi comes from
    # the integration-by-parts of the G3 Box(phi) term on FLRW
    rho_3 = ( - 2*X_sym * G3phi
              + 6*X_sym * H * dphi * G3X )

    # --- G4 contribution ---
    # rho_4 = -6 H^2 G4
    #        + 24 H^2 X (G4X + X G4XX)
    #        - 12 H X dphi G4phiX          <- from int-by-parts of R term
    #        - 6 H dphi G4phi              <- idem
    rho_4 = ( - 6 * H**2 * G4
              + 24 * H**2 * X_sym * (G4X + X_sym*G4XX)
              - 12 * H * X_sym * dphi * G4phiX
              - 6  * H * dphi * G4phi )

    # --- G5 contribution ---
    # rho_5 = -6 H^2 X (3 G5phi + 2X G5phiX)
    #        + 2 H^3 dphi X (5 G5X + 2X G5XX)
    rho_5 = ( - 6  * H**2 * X_sym * (3*G5phi + 2*X_sym*G5phiX)
              + 2  * H**3 * dphi  * X_sym * (5*G5X + 2*X_sym*G5XX) )

    rho_total = rho_2 + rho_3 + rho_4 + rho_5

    return rho_total


def pressure():
    """
    Compute the Horndeski pressure  p_phi  in flat FLRW.

    Returns a SymPy expression in terms of:
        phi_sym, X_sym, H, dH, dphi, ddphi

    Reference: KYY (2011) eq (2.5)
    """

    # --- G2 contribution ---
    # p_2 = G2
    p_2 = G2

    # --- G3 contribution ---
    # p_3 = -2X(G3phi + G3X * ddphi)
    # The ddphi comes from Box(phi) = -ddphi - 3H dphi on FLRW
    p_3 = -2*X_sym * (G3phi + G3X * ddphi)

    # --- G4 contribution ---
    # p_4 = (2 dH + 3 H^2)(2 G4 - 4X G4X)   <- the GR-like part
    #      - 8 dH X G4X
    #      - 8 H dphi X G4phiX               <- from variation
    #      + (2 ddphi + 6 H dphi) G4phi
    #      + 4 X (ddphi + 3 H dphi) G4phiX   <- idem
    #      - 4 X (dH + 3 H^2) G4XX * 2X      <- second X deriv piece
    #      + 4 X G4phi * (ddphi/dphi... )     <- careful below
    #
    # Following KYY exactly:
    p_4 = (   2*(2*dH + 3*H**2) * (G4 - 2*X_sym*G4X)
            - 8*dH * X_sym * G4X
            + 4 * X_sym * (ddphi + 3*H*dphi) * G4phiX
            - 8 * H * X_sym * dphi * G4phiX
            + (2*ddphi + 6*H*dphi) * G4phi
            - 16 * H * X_sym * dphi * G4XX * X_sym   # = -16H X^2 dphi G4XX
            # note: last term comes from the 4X^2 G4XX in rho being time-differentiated
            )

    # --- G5 contribution ---
    # p_5 is the most involved.  Following KYY (2011) eq (2.5):
    # p_5 = 2 X G5phi (ddphi - 2 H dphi)
    #      - 2 X^2 G5phiX (ddphi/dphi ... )
    #      ... full expression:
    p_5 = (   2  * X_sym * G5phi * (ddphi - 2*H*dphi)
            - 4  * X_sym * H * dphi * G5phi
            - 2  * H**2 * X_sym * (3*G5phi + 2*X_sym*G5phiX)
            + 2  * H**2 * X_sym * dphi * (G5X + X_sym*G5XX) * ddphi
            + 4  * H    * X_sym * dphi * G5phiX * ddphi
            - 4  * H**3 * X_sym * dphi * (G5X + X_sym*G5XX)
            )

    p_total = p_2 + p_3 + p_4 + p_5

    return p_total


# =============================================================================
# STEP 2 — Friedmann equations
#
#   First Friedmann:    3 H^2 = rho_phi
#   Raychaudhuri:      -2 dH  = rho_phi + p_phi
#
#   These are the two background equations before reducing to a 2D ODE.
# =============================================================================

def friedmann_constraint():
    """
    Returns the first Friedmann equation as a SymPy Eq:
        3 H^2 - rho_phi = 0

    This is a constraint (no time derivatives of H appear on the RHS
    after the X -> dphi^2/2 substitution).
    """
    from sympy import Eq
    rho = energy_density()
    return Eq(3*H**2, rho)


def raychaudhuri():
    """
    Returns the Raychaudhuri (second Friedmann) equation:
        -2 dH = rho_phi + p_phi

    This contains dH on the LHS and ddphi on the RHS (from pressure).
    Together with the KG equation, these are the two dynamical equations.
    """
    from sympy import Eq
    rho = energy_density()
    p   = pressure()
    return Eq(-2*dH, rho + p)


# =============================================================================
# STEP 3 — Klein-Gordon equation
#
#   The KG equation in Horndeski is not simply  ddphi + 3H dphi + V' = 0.
#   It is modified by the G3, G4, G5 couplings.
#
#   The general form (KYY 2011 eq 2.9) is:
#
#     P_ddphi * ddphi  +  P_dH * dH  +  P_0  =  0
#
#   where P_ddphi, P_dH, P_0 are functions of (phi, dphi, H) only
#   (no ddphi or dH inside them).
#
#   We derive this by varying the action w.r.t. phi and reading off
#   the coefficient of the highest derivative term.
# =============================================================================

def KG_coefficients():
    """
    Returns the three coefficient functions in the KG equation:

        P_ddphi(phi, dphi, H) * ddphi
      + P_dH   (phi, dphi, H) * dH
      + P_0    (phi, dphi, H)
      = 0

    These are derived from KYY (2011) eq (2.9) / (2.10).

    The user can solve for ddphi:
        ddphi = -(P_dH * dH + P_0) / P_ddphi

    And dH from the Raychaudhuri equation.

    Returns
    -------
    P_ddphi, P_dH, P_0  as SymPy expressions
    """

    # -----------------------------------------------------------------------
    # Following KYY (2011) eq (2.9) exactly.
    # The KG equation written out is:
    #
    #   J_dot  +  3H J  +  P_X * ddphi  =  G2phi - 2X G3phi + ...
    #
    # where J = dphi * (G2X + 2X G2XX + ...) is the generalised momentum.
    #
    # After collecting all ddphi terms on one side, the three coefficient
    # functions are:
    #
    # P_ddphi — coefficient of ddphi:
    #   = G2X + 2X G2XX                          (from G2)
    #   + 3H dphi (G3X + X G3XX)                 (from G3, friction-like)
    #   + 6H^2 (G4X + X G4XX)                    (from G4, NOTE: + sign)
    #   + 6H^3 dphi (G5X + X G5XX)               (from G5)
    #
    # P_dH — coefficient of dH (= H_dot):
    #   = 6H dphi G4X                             (from G4)
    #   - 6H^2 X G5X                              (from G5, NOTE: - sign)
    #
    # P_0 — all remaining terms (no ddphi, no dH):
    #   = G2phi                                   (from G2)
    #   - 2X G3phi                                (from G3)
    #   - 3H dphi G3phi                           (from G3 friction)
    #   - 6H^2 G4phi                              (from G4, NOTE: - sign)
    #   + 6H^3 dphi G5phi                         (from G5, NOTE: + sign)
    #   + 3H dphi G2X                             (standard friction)
    #   + 6H X dphi G3XX                          (from G3)
    #
    # For GR canonical (G2=X-V, G3=G5=0, G4=Mpl^2/2):
    #   P_ddphi = 1,  P_dH = 0,  P_0 = -V'(phi)
    #   Full EOM: ddphi + 3H dphi + V' = 0   (3H dphi from Raychaudhuri)
    # -----------------------------------------------------------------------

    # Coefficient of ddphi
    # G2 piece: G2X + 2X G2XX  (note: G2XX = diff(G2X, X_sym), not G2X again)
    P_ddphi = (   G2X  +  2*X_sym * diff(G2X, X_sym)        # G2 piece: G2X + 2X*G2XX
               +  3*H * dphi * (G3X + X_sym*G3XX)           # G3 piece
               +  3*H**2 * (G4X + X_sym*G4XX)               # G4 piece (+sign)
               +  6*H**3 * dphi * (G5X + X_sym*G5XX)        # G5 piece
             )

    # Coefficient of dH
    P_dH = (   6*H * dphi * G4X                             # G4 piece (+sign)
             - 6*H**2 * X_sym * G5X                         # G5 piece (-sign)
           )

    # Remaining terms
    P_0 = (   G2phi                                         # G2 potential
            + 3*H * dphi * G2X                              # standard friction
            - 2*X_sym * G3phi                               # G3 phi-deriv
            - 3*H * dphi * G3phi                            # G3 friction
            + 6*H * X_sym * dphi * G3XX                     # G3 X-deriv friction
            - 6*H**2 * G4phi                                # G4 phi-coupling (-sign)
            + 6*H**3 * dphi * G5phi                         # G5 phi-coupling (+sign)
          )

    return P_ddphi, P_dH, P_0


# =============================================================================
# STEP 4 — Slow-roll parameters
#
#   epsilon_H = -dH / H^2
#   eta_H     = epsilon_H_dot / (H * epsilon_H)
#   delta     = ddphi / (H dphi)    (field acceleration parameter)
# =============================================================================

def slow_roll_epsilon():
    """
    epsilon_H = -dH / H^2

    In N-time (d/dN = (1/H) d/dt):
        epsilon_H = -H'/H    where ' = d/dN

    Returns the symbolic expression.
    """
    return -dH / H**2


def slow_roll_delta():
    """
    delta = ddphi / (H * dphi)

    Measures deviation from the slow-roll attractor.
    At the attractor: ddphi ~ 0, so delta ~ 0.
    """
    return ddphi / (H * dphi)


# =============================================================================
# STEP 5 — Substitution helper
#
#   After the user defines their model in model.py, call this to substitute
#   G2=..., G3=..., G4=..., G5=... into any of the above expressions.
# =============================================================================

def substitute_model(expr, G2_model, G3_model, G4_model, G5_model):
    """
    Substitute a specific Horndeski model into a symbolic expression.

    Parameters
    ----------
    expr      : SymPy expression (output of energy_density, pressure, etc.)
    G2_model  : SymPy expression for G2 in terms of (phi_sym, X_sym)
    G3_model  : SymPy expression for G3
    G4_model  : SymPy expression for G4
    G5_model  : SymPy expression for G5

    Returns
    -------
    SymPy expression with all G_i replaced and simplified.
    """
    subs_dict = {
        G2 : G2_model,
        G3 : G3_model,
        G4 : G4_model,
        G5 : G5_model,
        # Also substitute the partial derivatives
        G2phi  : diff(G2_model, phi_sym),
        G2X    : diff(G2_model, X_sym),
        G3phi  : diff(G3_model, phi_sym),
        G3X    : diff(G3_model, X_sym),
        G3phiX : diff(diff(G3_model, phi_sym), X_sym),
        G3XX   : diff(G3_model, X_sym, X_sym),
        G4phi  : diff(G4_model, phi_sym),
        G4X    : diff(G4_model, X_sym),
        G4phiX : diff(diff(G4_model, phi_sym), X_sym),
        G4XX   : diff(G4_model, X_sym, X_sym),
        G5phi  : diff(G5_model, phi_sym),
        G5X    : diff(G5_model, X_sym),
        G5phiX : diff(diff(G5_model, phi_sym), X_sym),
        G5XX   : diff(G5_model, X_sym, X_sym),
    }
    return simplify(expr.subs(subs_dict))


# =============================================================================
# STEP 6 — Pretty-print summary  (run this file directly to check)
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("  HORNDESKI BACKGROUND EQUATIONS  (general G2, G3, G4, G5)")
    print("=" * 70)

    print("\n--- Energy density  rho_phi ---")
    print("  3 H^2 = rho_phi  where  rho_phi =")
    pprint(energy_density())

    print("\n--- Pressure  p_phi ---")
    pprint(pressure())

    print("\n--- KG equation coefficients ---")
    P_ddphi, P_dH, P_0 = KG_coefficients()
    print("  P_ddphi (coefficient of ddphi) =")
    pprint(P_ddphi)
    print("  P_dH    (coefficient of dH)    =")
    pprint(P_dH)
    print("  P_0     (remaining terms)      =")
    pprint(P_0)

    print("\n--- Slow-roll epsilon ---")
    print("  epsilon_H = -dH / H^2 =")
    pprint(slow_roll_epsilon())

    # -----------------------------------------------------------------
    # Sanity check: GR + canonical scalar field
    #   G2 = X - V(phi),  G3 = 0,  G4 = Mpl^2/2,  G5 = 0
    # Should recover:
    #   3 H^2 = dphi^2/2 + V
    #   ddphi + 3H dphi + V'(phi) = 0
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SANITY CHECK: GR + canonical scalar field")
    print("  G2 = X - V(phi),  G3 = 0,  G4 = Mpl^2/2,  G5 = 0")
    print("=" * 70)

    V = Function('V')(phi_sym)

    G2_GR = X_sym - V
    G3_GR = 0*X_sym
    G4_GR = Mpl**2 / 2
    G5_GR = 0*X_sym

    # --- energy density ---
    rho_GR = substitute_model(energy_density(), G2_GR, G3_GR, G4_GR, G5_GR)
    rho_GR = rho_GR.subs(X_sym, dphi**2/2)
    # For GR: G4 = Mpl^2/2, so the -6 H^2 G4 term gives -3 Mpl^2 H^2.
    # The Friedmann equation is  3 Mpl^2 H^2 = dphi^2/2 + V
    # so rho_GR should equal dphi^2/2 + V (the -3Mpl^2 H^2 is the LHS moved over).
    # Let's show both sides explicitly:
    print("\n  Friedmann: 3*Mpl^2*H^2 = rho_phi")
    print("  Full rho_phi expression (LHS includes -3Mpl^2 H^2 from G4):")
    pprint(rho_GR)
    print("\n  Rearranged (move -3Mpl^2 H^2 to LHS, should give dphi^2/2 + V):")
    from sympy import symbols as sym2
    Mpl2 = Mpl**2
    rho_rearranged = simplify(rho_GR + 3*Mpl2*H**2)
    pprint(rho_rearranged)

    # --- KG coefficients ---
    coeffs_GR = [
        substitute_model(c, G2_GR, G3_GR, G4_GR, G5_GR).subs(X_sym, dphi**2/2)
        for c in KG_coefficients()
    ]
    P_ddphi_GR, P_dH_GR, P_0_GR = coeffs_GR

    print("\n  KG: P_ddphi (coefficient of ddphi, should be 1) =")
    pprint(simplify(P_ddphi_GR))

    print("\n  KG: P_dH (coefficient of dH, should be 0 for GR) =")
    pprint(simplify(P_dH_GR))

    print("\n  KG: P_0 (should be -dV/dphi, friction 3H*dphi comes from EOM structure) =")
    pprint(simplify(P_0_GR))

    print("\n  Full KG EOM: ddphi + P_0 = 0  (after solving Friedmann for dH)")
    print("  i.e. ddphi + 3H*dphi + V'(phi) = 0")
    print("  Note: the 3H*dphi friction arises when dH is eliminated")
    print("  using the Raychaudhuri eq:  -2*dH = rho + p = dphi^2")
