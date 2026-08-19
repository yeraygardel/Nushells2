import os, sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.style.use('sty.mplstyle')

cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)

from shells import Shells
from plot import circles

data_dir = sys.argv[1]
speed    = float(sys.argv[2])

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

def log_r(r, rmin):
    return np.log10(r / rmin)

shells = Shells()
shells.hdf5_io = True if use_hdf5 else False
shells._load(data_dir, 0)
skip = 10 if shells.N > 1000 else 1

rmin = shells.R.min()
rmax = shells.R.max()

w_thr = np.percentile(shells.w, 5)   # scale-independent: hide the dimmest 5%
R_vals = shells.R[::skip]
W_vals = shells.w[::skip]

mask = W_vals > w_thr

#q_vals = np.abs(shells.q[::skip])           # radial momentum (hat_q_r)
q_vals = np.sqrt(shells.q[::skip]**2 + (shells.ell[::skip]/shells.R[::skip])**2)
w_vals = shells.w[::skip]
f_vals = w_vals / np.where(q_vals > 0, q_vals**2, np.inf)

norm = mpl.colors.Normalize(vmin=np.nanmin(W_vals[mask]), vmax=np.nanmax(W_vals[mask]))
#norm = mpl.colors.LogNorm(vmin=np.nanmin(W_vals[mask]), vmax=np.nanmax(W_vals[mask]))

cmap = plt.get_cmap("RdBu_r")

plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax, label=r"$w$")

circles = []
for r, w in zip(R_vals[mask], W_vals[mask]):
    color = cmap(norm(w))
    c = Circle((0,0), r, fill=False, color=color, alpha=0.8)
    ax.add_patch(c)
    circles.append(c)

force_bar, = ax.plot([0, 0], [1.15, 1.15], color='navy', lw=3,
                      transform=ax.transAxes, clip_on=False)
force_label = ax.text(0.01, 1.17, '', transform=ax.transAxes,
                       color='navy', fontsize=12, va='bottom')

ax.set_xlim(-shells.Rmax, shells.Rmax)
ax.set_ylim(-shells.Rmax, shells.Rmax)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

for s in ax.spines.values():
    s.set_visible(False)

# Update loop
for i in range(len(files)):
    shells._load(data_dir, i)

    R_vals = shells.R[::skip]
    W_vals = shells.w[::skip]
    mask = W_vals > w_thr

    for circle, r, w in zip(circles, R_vals[mask], W_vals[mask]):
        circle.set_radius(r)
        circle.set_color(cmap(norm(w)))

    bar_length = log_r(1/shells.a, rmin) / log_r(rmax, rmin)
    bar_length = np.maximum(bar_length, 0.002)
    force_bar.set_xdata([0, bar_length])
    force_label.set_text(r'$\lambda_\phi = %.2f$' % (1/shells.a))
    z = 1/shells.a-1
    ax.set_title(r'$z=%.2f$'%z)
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()
    plt.pause(speed)

plt.ioff()
plt.show()
