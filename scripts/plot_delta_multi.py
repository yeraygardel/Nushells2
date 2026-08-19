import os, sys, glob
import numpy as np
import scipy.ndimage as sn
import matplotlib.pyplot as plt
plt.style.use('sty.mplstyle')

cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)

from shells import Shells

data_dir = sys.argv[1]
sig      = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
rcut     = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
n_curves = int(sys.argv[4])   if len(sys.argv) > 4 else 5

pattern = os.path.join(data_dir, "states/shells_*.hdf5")
files   = sorted(glob.glob(pattern))
use_hdf5 = True
if len(files) < 1:
    pattern = os.path.join(data_dir, "states/shells_*.txt")
    files   = sorted(glob.glob(pattern))
    use_hdf5 = False

if len(files) < 1:
    print("No field files found, check the output directory!")
    exit(0)

nfiles = len(files)
print(f"Found {nfiles} field files, last one is {files[-1]}")

shells = Shells()
shells.hdf5_io = use_hdf5

# Pick n_curves snapshots evenly spaced across the run (always including
# the first and last), so the plot shows the evolution across redshift.
idx = np.unique(np.linspace(0, nfiles - 1, n_curves).astype(int))

fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("viridis")

for k, i in enumerate(idx):
    shells._load(data_dir, i)
    r_c, n, n_err = shells.density()

    valid = np.isfinite(n) & (r_c < rcut)
    if valid.sum() < 3:
        continue

    n_bar = np.nanmean(n[np.isfinite(n)])   # background from the FULL profile,
                                             # not just the r<rcut window
    y    = sn.gaussian_filter(n[valid] / n_bar, sigma=sig)
    yerr = sn.gaussian_filter(n_err[valid] / n_bar, sigma=sig)

    z = 1.0 / shells.a - 1.0
    color = cmap(k / max(1, len(idx) - 1))

    ax.plot(r_c[valid], y, color=color, lw=1.8, label=rf'$z={z:.2f}$')
    ax.fill_between(r_c[valid], y - yerr, y + yerr, color=color, alpha=0.25)

ax.axhline(1.0, color='k', lw=1.5, ls='--', alpha=0.6)
ax.set_xlim(0, rcut)
ax.set_xlabel(r'$r m_\phi$')
ax.set_ylabel(r'$n(r)/\bar n$')
ax.legend()

plt.tight_layout()
plt.show()
