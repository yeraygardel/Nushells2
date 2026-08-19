import os, sys
import glob
import numpy as np
import scipy.ndimage as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.style.use('sty.mplstyle')

cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)

from shells import Shells

data_dir = sys.argv[1]
speed    = float(sys.argv[2])
sig      = float(sys.argv[3])

use_hdf5 = True
pattern = os.path.join(data_dir, "states/shells_*.hdf5")
files   = sorted(glob.glob(pattern))
if len(files) < 1:
    pattern = os.path.join(data_dir, "states/shells_*.txt")
    files   = sorted(glob.glob(pattern))
    use_hdf5 = False

if len(files) < 1:
    print(f"No field files found, check the output directory!")
    exit(0)

print(f"Found {len(files)} field files, last one is {files[-1]}")

shells = Shells()
shells.hdf5_io = True if use_hdf5 else False
shells._load(data_dir, 0,)

plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

r_c, n, n_err = shells.density()
valid  = np.isfinite(n)
n_bar = np.nanmean(n[valid])

y     = sn.gaussian_filter(n[valid]/n_bar, sigma=sig)
# NOTE: smoothing the error the same way as the central value is a
# conservative simplification (real smoothing reduces variance further,
# so this slightly overestimates the true error of the smoothed curve),
# but keeps the band visually consistent with the smoothed line.
yerr  = sn.gaussian_filter(n_err[valid]/n_bar, sigma=sig)

line_n,   = ax.plot(r_c[valid], y, color='steelblue', lw=1.8, label=r'$n(r)$')
band      = ax.fill_between(r_c[valid], y-yerr, y+yerr, color='steelblue', alpha=0.25)
line_nbar = ax.axhline(n_bar/n_bar, color='k', lw=2, ls='--',alpha=0.6)
line_phi = ax.axvline(1/shells.a, color='navy', lw=1.5, label=r'$\lambda_\phi$')

ax.set_xlabel(r'$rm_\phi$')
ax.set_ylabel(r'$\delta_\nu$')
ax.legend()

title = fig.suptitle(r'$z=%.2f$' % (1/shells.a - 1))

for i in range(len(files)):
    shells._load(data_dir, i,)

    r_c, n, n_err = shells.density()
    valid  = np.isfinite(n)
    n_bar = np.nanmean(n[valid])
    y    = sn.gaussian_filter(n[valid]/n_bar, sigma=sig)
    yerr = sn.gaussian_filter(n_err[valid]/n_bar, sigma=sig)
    line_n.set_data(r_c[valid], y)
    line_phi.set_xdata([1/shells.a, 1/shells.a])

    band.remove()
    band = ax.fill_between(r_c[valid], y-yerr, y+yerr, color='steelblue', alpha=0.25)

    # Rescale to the current shell extent each frame, so the plot follows
    # the collapsing core instead of clipping it out at a fixed r=5 floor.
    ax.set_xlim(0.9 * r_c[valid].min(), 1.1 * r_c[valid].max())
    ax.set_ylim(0.5 * np.nanmin(y), 2.0 * np.nanmax(y))

    z = 1/shells.a - 1
    title.set_text(r'$z=%.2f$' % z)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(speed)

plt.ioff()
plt.show()
