import scipy
import numpy as np
from timing import timed
import warnings
warnings.filterwarnings("ignore")


@timed("interpPhi")
def interpPhi(R_old, phi_old, R_new):
    """
    Interpolate phi from old shell positions to new ones.
    R_old, phi_old : sorted, self-consistent from previous step
    R_new          : new positions after drift (sorted by _sort before solvePhi)
    """
    return np.interp(R_new, R_old, phi_old,
                     left=phi_old[0],    # extrapolate as constant at boundaries
                     right=phi_old[-1])

def _computePhi(shells, xi_cap=500.0):
    """
    Compute the dimensionless Yukawa potential hat_phi at each shell position.

    The potential sourced by shell j at the position of shell i is:
        K_{phi,ij} = Theta(R_i > R_j) * exp(-xi_i)/R_i * sinh(xi_j)/xi_j
                   + Theta(R_i < R_j) * sinh(xi_i)/R_i * exp(-xi_j)/xi_j
    where xi = a * R is the dimensionless Yukawa argument.

    Shells must be sorted ascending in R so that cumulative sums correctly
    partition into outer (j < i) and inner (j > i) contributions.

    Parameters
    ----------
    shells  : Shells object with attributes a, R, m, eps, w, alpha, N
    xi_cap  : Yukawa screening threshold, the default value of 100.0 is a safe
              and conservative cutoff.
    """
    ## Retrieve shell state
    N = shells.N
    a, R, m, eps, w, alpha = shells.a, shells.R, shells.m, \
                             shells.eps, shells.w, shells.alpha

    xi_raw  = a * R
    xi      = np.minimum(xi_raw, xi_cap)
    capped  = xi_raw >= xi_cap
    xi_max  = a*140

    e        = np.exp(-xi)
    sinh_xi  = np.sinh(xi)

    # Source weights: properties of shell j as a source.
    out_weight = sinh_xi / xi
    inn_weight = e       / xi

    moe  = w * m / eps
    out  = moe * out_weight
    inn  = moe * inn_weight

    # Cumulative sums over sorted shells.
    # sum_outer[i] = sum_{j <= i} out_j : all shells inside R_i, INCLUDING
    #                the shell's own self-contribution (added here so it
    #                isn't double-counted via sum_inner below -- the outer
    #                and inner kernels are equal exactly at j=i, so the
    #                self-term only needs to appear in one of the two sums)
    # sum_inner[i] = sum_{j > i}  inn_j : all shells strictly outside R_i
    cumOut = np.cumsum(out)
    cumInn = np.cumsum(inn)
    totInn = cumInn[-1]

    sum_outer = cumOut.copy()

    sum_inner      = np.empty(N)
    sum_inner[-1]  = 0.0
    sum_inner[:-1] = totInn - cumInn[:-1]

    # Field-point dependence
    # exp(-xi_i)/R_i  multiplies the outer sum (shells inside  R_i)
    # sinh(xi_i)/R_i  multiplies the inner sum (shells outside R_i)
    exp_over_r  = e / R
    sinh_over_r = np.where(capped, 0.0, sinh_xi / R)
    boundaryphi = ((4.0 * np.pi)/alpha)*sinh_over_r/(a)*(np.exp(-xi_max))*(1+xi_max)*(shells.phi_bkg)
    boundaryphi = boundaryphi * 0.0   # TEMP: disabled for with/without comparison

    # Combine terms: hat_phi_i = -alpha/(4*pi) * (outer_term + inner_term)
    # The field eq. (k/a)^2 dphi + m_phi^2 dphi = -g n_nu delta_nu (Eq. 10 of
    # 2412.20766) carries no 4*pi on the source, so its Green's function is
    # exp(-xi)/(4*pi*r) -- matching the 4*pi already used in solveYukawaForce.
    phi = -(exp_over_r * sum_outer + sinh_over_r * sum_inner)+boundaryphi
    phi *= alpha / (4.0 * np.pi)

    # Cache, so no need to recompute force (after last phi iteration)
    shells._cache = {
        'xi': xi, 'capped': capped,
        'e': e, 'sinh_xi': sinh_xi,
        'sum_outer': sum_outer,
        'sum_inner': sum_inner,
    }
    return phi


def _stepPhi(shells, phi):
    """One iterative step: set phi -> update m -> compute new phi."""
    shells.data["phi"] = phi
    phi_min = -0.9999*shells.m0
    shells.data['phi'] = np.maximum(shells.data['phi'], phi_min)
    shells._update_mass()
    shells._update_eps()
    return _computePhi(shells)


def _relativeErr(new, old):
    return np.max(np.abs(new - old) / (1.0 + np.abs(old)))


def _solveNaive(shells, phi0, tol, max_iter, damp):
    """
    """
    phi = phi0.copy()

    for i in range(max_iter):
        phi_new = _stepPhi(shells, phi)
        err = _relativeErr(phi_new, phi)
        phi = phi + damp * (phi_new - phi)   # damped update
        if err < tol:
            return phi, i + 1, True, err

    return phi, max_iter, False, err

def _solveAnderson(shells, phi0, tol, max_iter, memory, alpha_mix):
    # scipy.optimize.anderson solves F(x)=0, so residual = phi_new - phi
    call_count = [0]

    def F(phi):
        call_count[0] += 1
        return _stepPhi(shells, phi) - phi

    try:
        phi = scipy.optimize.anderson(
              F, phi0,
              f_tol=tol,
              alpha=alpha_mix,
              M=memory,
              maxiter=max_iter,
              verbose=False,
              )
        err = _relativeErr(F(phi) + phi, phi)
        return phi, call_count[0], True, err

    except Exception as e:
        return phi0, call_count[0], False, float("inf")


def _solveNoIter(shells, phi0):
    """ """
    phi = _stepPhi(shells, phi0)
    return phi


@timed("solvePhi")
def solvePhi(
    shells,
    *,
    method   = "anderson",
    tol      = 1e-5,
    max_iter = 200,
    damp     = None,
    memory   = 20,
    alpha_mix = None,
    verbose  = True
):
    """
    Solve the self-consistent problem for the hat_phi potential.

    Parameters
    ----------
    shells      : Shell state
    method      : "naive" or "anderson" or "noiter"
    tol         : relative convergence tolerance
    max_iter    : maximum iterations / function evaluations
    damp        : (naive)    mixing parameter in (0, 1]
    memory      : (anderson) number of history vectors
    alpha_mix   : (anderson) step size
    verbose     : print convergence summary
    """

    phi0 = shells.phi.copy()

    alpha = shells.alpha
    if damp is None:
        damp = float(np.clip(2.0 / (1.0 + alpha * float(shells.w.mean())), 1e-3, 1.0))
    if alpha_mix is None:
        alpha_mix = float(np.clip(1.0 / alpha, 1e-6, 1.0))

    if method == "naive":
        phi, n_iter, converged, err = _solveNaive(shells, phi0, tol, max_iter, damp)
    elif method == "anderson":
        phi, n_iter, converged, err = _solveAnderson(shells, phi0, tol, max_iter, memory, alpha_mix)
    elif method == "noiter":
        phi = _solveNoIter(shells, phi0)
        converged = True
        n_iter = 1
        err = np.nan
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: naive | anderson")

    if converged == False:
        print(f"[solve_phi] {method:10s} iters={n_iter:4d}  err={err:.2e}  α={alpha:.1e}")
        raise scipy.optimize.NoConvergence("The potential calculation has not converged!")

    # commit solution
    shells.data["phi"] = phi
    phi_min = -shells.m0*0.9999 
    shells.data['phi'] = np.maximum(shells.data['phi'], phi_min)
    shells._update_mass()
    shells._update_eps()

    status = "converged" if converged else "NOT converged"
    shells.log.debug(f"[solve_phi] {method:10s} iters={n_iter:3d}  err={err:.2e}")

    return phi
