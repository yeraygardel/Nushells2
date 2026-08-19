import os, sys
cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)
import numpy as np
import matplotlib.pyplot as plt
from shells import Shells

# --- set up the same way we've done throughout the session ---
s = Shells()
s.init(Nshells=2000, g=1e-26, m_phi=1e-29, m_nu=0.1, kappa=54.24, kappa2=2.0,
       dt_frac=0.3, iter_m='anderson', iter_tol=1e-3, soft=1e-2,
       hdf5_io=False, seed=9, odir='/tmp/x', verb=0, to_file=False,
       psi_boost=1e8)   # amplify the seed potential so the initial density
                         # profile isn't nearly flat -- try 1.0 for the default

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Panel 1: initial density distribution ---
# shells.density() bins shells by radius, weighted by their w, and returns
# (bin-center radii, number density) -- exactly what the density_anim GIFs used.
r_c, n, n_err = s.density(nbins=100)
valid = np.isfinite(n)
n_bar = np.nanmean(n[valid])

ax = axes[0]
ax.loglog(r_c[valid], n[valid]/n_bar, color='steelblue', lw=1.8)
ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.6, label=r'$\bar n$')
ax.axvline(1/s.a, color='navy', lw=1.2, label=r'$\lambda_\phi$')
ax.set_xlabel(r'$r\, m_\phi$')
ax.set_ylabel(r'$n(r)/\bar n$')
ax.set_title('Initial density profile')
ax.legend()

# --- Panel 2: initial momentum distribution ---
# q_r (radial) and ell/R (transverse) combine into the full momentum magnitude
q_total = np.sqrt(s.q**2 + (s.ell / s.R)**2)

ax = axes[1]
ax.hist(q_total, bins=50, weights=s.w, density=True, color='darkorange', alpha=0.8,
        label='sampled (weighted)')
q_grid = np.linspace(0, q_total.max(), 200)
fd_shape = q_grid**2 / (np.exp(q_grid) + 1.0)
fd_shape /= np.trapezoid(fd_shape, q_grid)
ax.plot(q_grid, fd_shape, 'k-', lw=1.5, label=r'true $q^2 f_0(q)$')
ax.set_xlabel(r'$\hat q$')
ax.set_ylabel('normalized density')
ax.set_title('Initial momentum distribution')
ax.legend()

fig.tight_layout()
outpath = os.path.join(pdir, "ic_plots.png")
fig.savefig(outpath, dpi=120)
print(f"saved {outpath}")
