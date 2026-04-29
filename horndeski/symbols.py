"""
symbols.py
==========
Central definition of every SymPy symbol and function used across the
Horndeski solver.  Import everything from here so that all modules share
the exact same objects (avoids subtle "different Symbol with same name"
bugs in SymPy).

Convention
----------
- Cosmic time is 't', e-fold time is 'N'.
- Overdots  ( . )  mean d/dt.
- Primes    ( ' )  mean d/dN.
- We work at background level first: all fields depend only on time.
- X = dphi^2 / 2  (note: positive, because phi is real and g^{00} = -1)
"""

from sympy import symbols, Function, Rational, sqrt, diff, Symbol

# ---------------------------------------------------------------------------
# 1.  Independent variables
# ---------------------------------------------------------------------------

t = symbols('t', real=True)          # cosmic time
N = symbols('N', real=True)          # e-fold time  N = ln(a/a0)

# ---------------------------------------------------------------------------
# 2.  Background scalar field  (function of time)
# ---------------------------------------------------------------------------

phi   = Function('phi')(t)           # phi(t)
dphi  = Function('dphi')(t)          # dphi/dt  ≡  phi_dot
ddphi = Function('ddphi')(t)         # d^2 phi / dt^2  ≡  phi_doubledot

# Kinetic term  X = (1/2) dphi^2
# At background level in flat FLRW:  X = +dphi^2/2  (spatial grads vanish)
X_expr = Rational(1, 2) * dphi**2   # the actual substitution expression

# We also keep X as a free symbol for writing G_i(phi, X) before substitution
phi_sym = symbols('phi_sym', real=True)   # free symbol for field argument
X_sym   = symbols('X_sym',   real=True, positive=True)  # free symbol for X argument

# ---------------------------------------------------------------------------
# 3.  Hubble parameter and scale factor  (functions of time)
# ---------------------------------------------------------------------------

H  = Function('H')(t)                # Hubble  H = a_dot / a
dH = Function('dH')(t)               # dH/dt   (H_dot)

a  = Function('a')(t)                # scale factor  a(t)

# ---------------------------------------------------------------------------
# 4.  Slow-roll / derived parameters  (will be computed, listed here for
#     reference so other modules can import the symbol names)
# ---------------------------------------------------------------------------

epsilon_sym = symbols('epsilon', real=True, positive=True)
# epsilon_H = - H_dot / H^2  = 1 - H'/H  in N-time
# Defined symbolically as:  epsilon = -dH / H^2

eta_sym     = symbols('eta',     real=True)
# eta = epsilon' / (epsilon * H)  (will be derived)

# ---------------------------------------------------------------------------
# 5.  Horndeski free functions  G2, G3, G4, G5
#     These are Functions of two arguments: (phi_sym, X_sym).
#     The user defines them in model.py as SymPy expressions.
#     Here we define them as generic Function objects so algebra.py
#     can manipulate them before the user's model is substituted.
# ---------------------------------------------------------------------------

G2 = Function('G2')(phi_sym, X_sym)
G3 = Function('G3')(phi_sym, X_sym)
G4 = Function('G4')(phi_sym, X_sym)
G5 = Function('G5')(phi_sym, X_sym)

# ---------------------------------------------------------------------------
# 6.  Partial derivatives of Gi  (SymPy diff applied to the Functions above)
#
#     Naming convention:
#       G2phi  = dG2/dphi
#       G2X    = dG2/dX
#       G4XX   = d^2 G4 / dX^2
#       G5phiX = d^2 G5 / (dphi dX)    etc.
# ---------------------------------------------------------------------------

# --- G2 partials ---
G2phi  = diff(G2, phi_sym)
G2X    = diff(G2, X_sym)

# --- G3 partials ---
G3phi  = diff(G3, phi_sym)
G3X    = diff(G3, X_sym)
G3phiX = diff(G3, phi_sym, X_sym)
G3XX   = diff(G3, X_sym, X_sym)

# --- G4 partials ---
G4phi  = diff(G4, phi_sym)
G4X    = diff(G4, X_sym)
G4phiX = diff(G4, phi_sym, X_sym)
G4XX   = diff(G4, X_sym, X_sym)

# --- G5 partials ---
G5phi  = diff(G5, phi_sym)
G5X    = diff(G5, X_sym)
G5phiX = diff(G5, phi_sym, X_sym)
G5XX   = diff(G5, X_sym, X_sym)

# ---------------------------------------------------------------------------
# 7.  Perturbation symbols  (scalar sector, uniform field gauge)
#
#     In uniform field gauge:  delta_phi = 0
#     The only scalar dof is the curvature perturbation  R(t, x)
#     Its Fourier mode is  R_k(t).
# ---------------------------------------------------------------------------

k  = symbols('k',  real=True, positive=True)   # comoving wavenumber
tau = symbols('tau', real=True)                 # conformal time (for ICs)

# Mukhanov variable  v_k = z * R_k   (z defined in perturbations.py)
# Mode function written as a real + imaginary part for numerical integration
Rk_re = Function('Rk_re')(t)    # Re[ R_k(t) ]
Rk_im = Function('Rk_im')(t)    # Im[ R_k(t) ]

# ---------------------------------------------------------------------------
# 8.  Perturbation coefficient symbols
#     (computed in algebra/perturbations.py, listed here for import)
# ---------------------------------------------------------------------------

Qs_sym  = symbols('Qs',  real=True, positive=True)   # scalar kinetic coeff
cs2_sym = symbols('cs2', real=True, positive=True)   # scalar sound speed^2
QT_sym  = symbols('QT',  real=True, positive=True)   # tensor kinetic coeff
cT2_sym = symbols('cT2', real=True, positive=True)   # tensor speed^2

# ---------------------------------------------------------------------------
# 9.  Useful constant symbols
# ---------------------------------------------------------------------------

Mpl = symbols('Mpl', real=True, positive=True)   # reduced Planck mass
M   = symbols('M',   real=True, positive=True)   # model mass scale
V0  = symbols('V0',  real=True, positive=True)   # potential amplitude

# ---------------------------------------------------------------------------
# 10.  Helper: the substitution dictionary
#      Use this to go from abstract (phi_sym, X_sym) to actual time functions
#      once the background solution is known.
# ---------------------------------------------------------------------------

def background_subs(phi_val, dphi_val):
    """
    Return a substitution dict that replaces the free symbols
    phi_sym, X_sym with the actual background values.

    Parameters
    ----------
    phi_val  : SymPy expression for phi   (e.g. phi  the Function, or a number)
    dphi_val : SymPy expression for dphi  (e.g. dphi the Function, or a number)

    Returns
    -------
    dict  suitable for .subs(d) calls
    """
    return {
        phi_sym : phi_val,
        X_sym   : Rational(1, 2) * dphi_val**2,
    }
