"""
algebra/perturbations.py
========================
Scalar and tensor perturbation coefficients for a general Horndeski theory
in flat FLRW, uniform-field gauge.

Transcribed directly from Kobayashi, Yamaguchi & Yokoyama 2011
(arXiv:1105.5723), equations (4.4), (4.5), (4.25), (4.26), (4.32), (4.33).

Public quantities
-----------------
    QT  = G_T          tensor kinetic coefficient        (KYY 4.5)
    cT2 = F_T / G_T     tensor speed squared              (KYY 4.4, 4.7)
    Qs  = G_S          scalar kinetic coefficient        (KYY 4.33)
    cs2 = F_S / G_S     scalar sound speed squared        (KYY 4.32, 4.34)

Quadratic actions (KYY 4.3, 4.31):
    S_T = (1/8) int a^3 [ G_T hdot^2 - (F_T/a^2)(grad h)^2 ]
    S_S =       int a^3 [ G_S zdot^2 - (F_S/a^2)(grad z)^2 ]

Stability: G_T>0, F_T>0 (tensor), G_S>0, F_S>0 (scalar).

In KYY the L2 function is called K; here the symbol is G2 (G2 == K).
"""

from sympy import Rational, diff, simplify
from horndeski.symbols import (
    t, dphi, ddphi, H, dH,
    phi_sym, X_sym,
    G2, G3, G4, G5,
    G2phi, G2X,
    G3phi, G3X, G3phiX, G3XX,
    G4phi, G4X, G4phiX, G4XX,
    G5phi, G5X, G5phiX, G5XX,
)
from horndeski.algebra.background import substitute_model

# K and its X-derivatives (KYY's L2 function == our G2)
KX  = G2X
KXX = diff(G2X, X_sym)

# Higher X-derivatives appearing in Sigma (KYY 4.25)
G4XXX   = diff(G4XX, X_sym)
G5XXX   = diff(G5XX, X_sym)
G4phiXX = diff(G4phiX, X_sym)
G5phiXX = diff(G5phiX, X_sym)




def F_T():
    # KYY (4.4):  F_T = 2[ G4 - X( ddphi G5X + G5phi ) ]
    return 2 * ( G4 - X_sym * (ddphi * G5X + G5phi) )


def G_T():
    # KYY (4.5):  G_T = 2[ G4 - 2X G4X - X( H dphi G5X - G5phi ) ]
    return 2 * ( G4 - 2*X_sym*G4X - X_sym * (H*dphi*G5X - G5phi) )


def QT():
    # Tensor kinetic coefficient = G_T. Ghost-free: QT > 0.
    return G_T()


def cT2():
    # Tensor speed squared = F_T / G_T.
    return F_T() / G_T()




def Sigma():
    # KYY (4.25)
    return (
        X_sym*KX + 2*X_sym**2*KXX
        + 12*H*dphi*X_sym*G3X + 6*H*dphi*X_sym**2*G3XX
        - 2*X_sym*G3phi - 2*X_sym**2*G3phiX
        - 6*H**2*G4
        + 6*( H**2*(7*X_sym*G4X + 16*X_sym**2*G4XX + 4*X_sym**3*G4XXX)
              - H*dphi*(G4phi + 5*X_sym*G4phiX + 2*X_sym**2*G4phiXX) )
        + 30*H**3*dphi*X_sym*G5X + 26*H**3*dphi*X_sym**2*G5XX
        + 4*H**3*dphi*X_sym**3*G5XXX
        - 6*H**2*X_sym*(6*G5phi + 9*X_sym*G5phiX + 2*X_sym**2*G5phiXX)
    )


def Theta():
    # KYY (4.26)
    return (
        - dphi*X_sym*G3X
        + 2*H*G4 - 8*H*X_sym*G4X - 8*H*X_sym**2*G4XX
        + dphi*G4phi + 2*X_sym*dphi*G4phiX
        - H**2*dphi*(5*X_sym*G5X + 2*X_sym**2*G5XX)
        + 2*H*X_sym*(3*G5phi + 2*X_sym*G5phiX)
    )



def G_S():
    # KYY (4.33):  G_S = (Sigma / Theta^2) G_T^2 + 3 G_T
    _GT = G_T()
    return Sigma() / Theta()**2 * _GT**2 + 3*_GT


def _ddt(expr):
    # Total d/dt of a background expression in (dphi, X_sym, H):
    #   dphi -> ddphi,  X = dphi^2/2 -> Xdot = dphi*ddphi,  H -> dH.
    return (
        diff(expr, dphi) * ddphi
        + diff(expr, X_sym) * (dphi * ddphi)
        + diff(expr, H) * dH
    )


def F_S():
    # KYY (4.32):  F_S = (1/a) d/dt[ (a/Theta) G_T^2 ] - F_T
    #            = H G_T^2/Theta + d/dt[G_T^2/Theta] - F_T
    _GT = G_T()
    inner = _GT**2 / Theta()
    return H * inner + _ddt(inner) - F_T()


def Qs():
    # Scalar kinetic coefficient = G_S. Ghost-free: Qs > 0.
    return G_S()


def cs2():
    # Scalar sound speed squared = F_S / G_S.
    return F_S() / G_S()


def pump_scalar():
    return 2 * Qs()


def pump_tensor():
    return QT()



def substitute_perturbations(G2_model, G3_model, G4_model, G5_model,
                             X_val=None):
    """
    Substitute a Horndeski model into QT, cT2, Qs, cs2.

    Returns a dict with keys 'QT', 'cT2', 'Qs', 'cs2'.
    """
    if X_val is None:
        X_val = dphi**2 / 2

    results = {}
    for name, expr in [('QT',  QT()),
                       ('cT2', cT2()),
                       ('Qs',  Qs()),
                       ('cs2', cs2())]:
        r = substitute_model(expr, G2_model, G3_model, G4_model, G5_model)
        r = r.subs(X_sym, X_val)
        results[name] = simplify(r)

    return results



# Self-test and sanity checks


if __name__ == "__main__":
    from sympy import Function, symbols, pprint
    from horndeski.symbols import Mpl, M

    V = Function('V')(phi_sym)

    print("="*64)
    print("  GATE 1 - GR canonical: G2=X-V, G3=0, G4=Mpl^2/2, G5=0")
    print("="*64)
    G2g, G3g, G4g, G5g = X_sym - V, 0*X_sym, Mpl**2/2, 0*X_sym
    rGR = substitute_perturbations(G2g, G3g, G4g, G5g)
    print("QT  (expect Mpl^2):");                 pprint(rGR['QT'])
    print("cT2 (expect 1):");                     pprint(rGR['cT2'])
    print("Qs  (expect dphi^2/(2H^2) = eps Mpl^2):"); pprint(rGR['Qs'])
    print("cs2 (expect 1 after Raychaudhuri -2dH=dphi^2/Mpl^2):")
    pprint(rGR['cs2'])
    # impose GR Raychaudhuri to confirm cs2 -> 1
    cs2_onshell = simplify(rGR['cs2'].subs(dH, -dphi**2/(2*Mpl**2)))
    print("cs2 on-shell (expect 1):");            pprint(cs2_onshell)

    print()
    print("="*64)
    print("  GATE 2 - NMDC Gmunu coupling: G4 = Mpl^2/2 + X/M^2")
    print("="*64)
    G2n, G3n, G4n, G5n = X_sym - V, 0*X_sym, Mpl**2/2 + X_sym/M**2, 0*X_sym
    rNM = substitute_perturbations(G2n, G3n, G4n, G5n)
    print("QT:");  pprint(rNM['QT'])
    print("cT2 (expect superluminal, cf. Yang et al eq 72):")
    pprint(rNM['cT2'])
    print("Qs:");  pprint(rNM['Qs'])
    print("cs2:"); pprint(rNM['cs2'])
