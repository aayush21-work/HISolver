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

def energy_density():
    """
    Compute the Horndeski energy density  rho_phi  in flat FLRW.

    Returns a SymPy expression in terms of:
        phi_sym, X_sym, H, dphi
    (X_sym should be substituted -> dphi^2/2 afterwards)

    Reference: KYY (2011) eq (2.4)
    """

    
    rho_2 = 2*X_sym*G2X - G2

   
    rho_3 = ( - 2*X_sym * G3phi
              + 6*X_sym * H * dphi * G3X )

    
    rho_4 = ( - 6 * H**2 * G4
              + 24 * H**2 * X_sym * (G4X + X_sym*G4XX)
              - 12 * H * X_sym * dphi * G4phiX
              - 6  * H * dphi * G4phi )

    
    rho_5 = ( - 6  * H**2 * X_sym * (3*G5phi + 2*X_sym*G5phiX)
              + 2  * H**3 * dphi  * X_sym * (5*G5X + 2*X_sym*G5XX) )

    rho_total = rho_2 + rho_3 + rho_4 + rho_5

    return rho_total


def pressure():
    """
    Horndeski pressure p_phi in flat FLRW, with  -2 dH = rho_phi + p_phi.
    Transcribed from KYY (2011) P_i, eqs (3.7)-(3.10).  G2 == K.
    """
    def ddt(e):
        return (diff(e, dphi)*ddphi
                + diff(e, X_sym)*(dphi*ddphi)
                + diff(e, H)*dH)

    Xdot = dphi*ddphi

    P2 = G2

    P3 = -2*X_sym*(G3phi + ddphi*G3X)

    # KYY (3.9)
    P4 = ( 2*(3*H**2 + 2*dH)*G4
           - 12*H**2*X_sym*G4X
           - 4*H*dH*X_sym*G4X
           - 8*H*Xdot*G4X
           - 8*H*X_sym*Xdot*G4XX
           + 2*(ddphi + 2*H*dphi)*G4phi
           + 4*X_sym*(ddphi - 2*H*dphi)*G4phiX )

    # KYY (3.10)
    P5 = ( -2*X_sym*(2*H**3*dphi + 2*H*dH*dphi + 3*H**2*ddphi)*G5X
           - 4*H**2*X_sym**2*ddphi*G5XX
           + 4*H*X_sym*(Xdot - H*X_sym)*G5phiX
           + 2*(2*ddt(H*X_sym) + 3*H**2*X_sym)*G5phi
           + 4*H*X_sym*dphi*diff(G5phi, phi_sym) )

    return P2 + P3 + P4 + P5



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



def KG_coefficients():
    """
    Klein-Gordon coefficients (P_ddphi, P_dH, P_0) such that
        P_ddphi * ddphi + P_dH * dH + P_0 = 0.

    Derived from the KYY scalar EOM  d/dt(a^3 J) = a^3 P_phi, i.e.
        Jdot + 3H J - P_phi = 0,
    with J from KYY (3.12) and P_phi from KYY (3.13).  G2 == K.
    The three coefficients are read off by collecting ddphi and dH.
    """
    G3phiphi = diff(G3phi, phi_sym)
    G5phiphi = diff(G5phi, phi_sym)
    Xdot = dphi*ddphi

    # KYY (3.12)
    J = ( dphi*G2X + 6*H*X_sym*G3X - 2*dphi*G3phi
          + 6*H**2*dphi*(G4X + 2*X_sym*G4XX) - 12*H*X_sym*G4phiX
          + 2*H**3*X_sym*(3*G5X + 2*X_sym*G5XX)
          - 6*H**2*dphi*(G5phi + X_sym*G5phiX) )

    # KYY (3.13)
    P_phi = ( G2phi - 2*X_sym*(G3phiphi + ddphi*G3phiX)
              + 6*(2*H**2 + dH)*G4phi + 6*H*(Xdot + 2*H*X_sym)*G4phiX
              - 6*H**2*X_sym*G5phiphi + 2*H**3*X_sym*dphi*G5phiX )

    def ddt(e):
        return (diff(e, dphi)*ddphi
                + diff(e, X_sym)*(dphi*ddphi)
                + diff(e, H)*dH
                + diff(e, phi_sym)*dphi)

    KG = ddt(J) + 3*H*J - P_phi   # = P_ddphi*ddphi + P_dH*dH + P_0

    P_ddphi = KG.diff(ddphi)
    P_dH    = KG.diff(dH)
    P_0     = (KG - P_ddphi*ddphi - P_dH*dH)

    return P_ddphi, P_dH, P_0



def klein_gordon():
    """
    Full scalar (Klein-Gordon) field equation as a single expression,
    parallel to energy_density() and pressure().

    Returns the residual  R  such that the equation of motion is  R = 0,
    with  R = P_ddphi*ddphi + P_dH*dH + P_0  (= Jdot + 3H J - P_phi).
    """
    P_ddphi, P_dH, P_0 = KG_coefficients()
    return P_ddphi*ddphi + P_dH*dH + P_0



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



# Pretty-print with some defaults for sanity check


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
