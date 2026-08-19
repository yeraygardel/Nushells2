import scipy
import numpy as np
from scipy.special import erf


# -----------------------------------------------------------------------
#  Fermi-Dirac sampler
# -----------------------------------------------------------------------
def _build_fd_icdf(shells, qmax=20.0, ngrid=10_000):
    """Build inverse CDF for f(hat_q) * hat_q^2 (momentum-space FD)."""
    q   = np.linspace(0.0, qmax, ngrid)
    pdf = q**2 / (np.exp(np.clip(q, 0, 500)) + 1.0)
    cdf = scipy.integrate.cumulative_trapezoid(pdf, q, initial=0.0)
    cdf /= cdf[-1]
    shells.inv_cdf = scipy.interpolate.interp1d(
        cdf, q, kind='linear',
        bounds_error=False,
        fill_value=(0.0, qmax)
    )

def sample_q(shells, N):
    """Return N samples of hat_q from the Fermi-Dirac distribution."""
    if shells.inv_cdf is None:
        _build_fd_icdf(shells)
    shells.log.debug(f"[IC] Sampled {N} momenta from Fermi-Dirac")
    return shells.inv_cdf(np.random.rand(N))


# -----------------------------------------------------------------------
# Weight computation
# -----------------------------------------------------------------------
def compute_weights(r, dr, q, Psi, log):
    """
    Phase-space weight for one shell.

    q is drawn (in ic.sample_q) via importance sampling from a density
    already proportional to q^2*f0(q), so for the Monte Carlo estimator
    weight = g(q)/p(q) that background shape cancels against the sampling
    density itself and must not be multiplied in again here -- only the
    linear perturbation factor (1+Theta) remains explicit.

    Parameters
    ----------
    r   : hat_r
    dr  : radial bin width
    q   : hat_q_total = q/T_nu, O(1)
    Psi : dimensionless Newtonian perturbation potential

    Returns
    -------
    w  : float, proportional to r^2 dr * (1 + perturbation)
    df : float, delta f perturbation
    """

    f0     = 1.0 / (np.exp(q) + 1.0)
    dfdlnq = -(q * np.exp(q)) / (np.exp(q) + 1.0)**2   # = q * df/dq
    pert   = 1.0 + (dfdlnq / f0) * Psi

    weight = (
        8.0 * np.pi**2    # solid angle factor
        * r**2 * dr       # radial volume element [1/m_phi]^3
        * (1.0 - Psi)     # metric perturbation correction
        * pert            # linearised perturbation, background cancels against sample density
    )

    log.debug(f"[IC] Computed weights")
    return weight, dfdlnq * Psi


# -----------------------------------------------------------------------
# Grav. potential computation
# -----------------------------------------------------------------------
def compute_Psi(R, shells):
    """ Seed potential, scaled by shells.psi_boost (default 1.0, no change). """
    a      = shells.a
    mphi   = shells.m_phi_hat
    rho    = R / shells.R0
    delta0 = 5*10**(-3)
    aini=10**(-4)
    omega0 = shells.omega0
    rini0=1

    norm = - (2.0*np.pi/3)*(1/mphi)**2*omega0*(delta0/aini)*rini0**3
    psi = norm * (1/R)*erf(R/(2**(1/2)*rini0))
    shells.log.debug(f"[IC] Computed gravitational potential Psi")
    return psi


