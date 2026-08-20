import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('sty.mplstyle')

cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)

from shells import Shells

data_dir  = sys.argv[1]
shell_arg = sys.argv[2] if len(sys.argv) > 2 else 'auto'

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
shells._load(data_dir, 0)
w0 = shells.w.copy()
N  = len(w0)
Rmin, Rmax = shells.Rmin, shells.Rmax
m0 = shells.m0

# Track every shell's R across all snapshots (matched by ID -- weights are
# NOT reliably unique, some CSVs have many shells sharing the same rounded
# weight, which silently breaks weight-based matching for most of them)
# so we can auto-pick an interesting one and/or plot the chosen one.
id0 = shells.ID.copy()

R_traj   = np.full((nfiles, N), np.nan)
q_traj   = np.full((nfiles, N), np.nan)
phi_traj = np.full((nfiles, N), np.nan)
a_traj   = np.full(nfiles, np.nan)

for i in range(nfiles):
    shells._load(data_dir, i)
    a_traj[i] = shells.a
    # map each current shell back to its index in the initial ordering
    sorter = np.argsort(id0)
    pos    = np.searchsorted(id0, shells.ID, sorter=sorter)
    pos    = np.clip(pos, 0, len(id0) - 1)
    good   = id0[sorter][pos] == shells.ID
    idx0   = sorter[pos][good]
    R_traj[i, idx0]   = shells.R[good]
    q_traj[i, idx0]   = shells.q[good]
    phi_traj[i, idx0] = shells.phi[good]

z_traj = 1.0 / a_traj - 1.0

if shell_arg == 'sustained':
    # pick the shell that spends the largest FRACTION of the whole run
    # near the center (not just one that happens to end up there late)
    thr = 10.0
    frac_near = np.mean(R_traj < thr, axis=0)
    shell_idx = int(np.argmax(frac_near))
    first_arrival = np.where(R_traj[:, shell_idx] < thr)[0]
    first_arrival = first_arrival[0] if len(first_arrival) else -1
    print(f"Picked shell {shell_idx}: spends {frac_near[shell_idx]*100:.1f}% of the run "
          f"within R<{thr}, first arriving at step {first_arrival} "
          f"(R0={R_traj[0,shell_idx]:.2f})")
elif shell_arg == 'collapse':
    # pick the shell with the deepest final R, tie-broken toward fewer
    # turnarounds (a "clean" collapse rather than a violently oscillating one)
    center_thr = 10 * Rmin
    rebound_counts = np.zeros(N, dtype=int)
    for j in range(N):
        r = R_traj[:, j]
        r = r[~np.isnan(r)]
        if len(r) < 3:
            continue
        dr = np.diff(r)
        sign = np.sign(dr)
        sign = sign[sign != 0]
        if len(sign) >= 2:
            rebound_counts[j] = np.sum(sign[1:] != sign[:-1])

    final_R = R_traj[-1, :]
    deep = np.where(final_R < center_thr)[0]
    if len(deep) == 0:
        shell_idx = int(np.argmin(final_R))
        print("No shell ended within 10*Rmin; picking the deepest overall.")
    else:
        shell_idx = deep[np.argmin(rebound_counts[deep])]
    print(f"Picked collapsing shell {shell_idx} "
          f"(R0={np.nanmax(R_traj[:,shell_idx]):.2f} -> "
          f"R_final={final_R[shell_idx]:.3f}, turnarounds={rebound_counts[shell_idx]})")
elif shell_arg == 'auto':
    # pick a shell that both rebounds a lot AND actually reaches near the
    # center, as the most illustrative example of real infall/rebound
    center_thr = 10 * Rmin
    rebound_counts = np.zeros(N, dtype=int)
    reaches_center = np.zeros(N, dtype=bool)
    for j in range(N):
        r = R_traj[:, j]
        r = r[~np.isnan(r)]
        if len(r) < 3:
            continue
        dr = np.diff(r)
        sign = np.sign(dr)
        sign = sign[sign != 0]
        if len(sign) >= 2:
            rebound_counts[j] = np.sum(sign[1:] != sign[:-1])
        reaches_center[j] = r.min() < center_thr

    candidates = np.where(reaches_center)[0]
    if len(candidates) == 0:
        shell_idx = int(np.argmax(rebound_counts))
        print("No shell reached the center; picking the most-rebounding shell instead.")
    else:
        shell_idx = candidates[np.argmax(rebound_counts[candidates])]
    print(f"Auto-picked shell {shell_idx} "
          f"(turnarounds={rebound_counts[shell_idx]}, "
          f"R_min={np.nanmin(R_traj[:, shell_idx]):.3f})")
else:
    shell_idx = int(shell_arg)

R   = R_traj[:, shell_idx]
q   = q_traj[:, shell_idx]
phi = phi_traj[:, shell_idx]

fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

axes[0].plot(z_traj, R, 'o-', ms=3)
axes[0].axhline(Rmin, color='C3', ls='--', alpha=0.6, label=r'$R_{\rm min}$')
axes[0].axhline(Rmax, color='k',  ls='--', alpha=0.6, label=r'$R_{\rm max}$')
axes[0].set_ylabel(r'$R$')
axes[0].legend()
axes[0].set_title(f'Shell {shell_idx}')

axes[1].plot(z_traj, q, 'o-', ms=3, color='C1')
axes[1].axhline(0, color='gray', lw=1)
axes[1].set_ylabel(r'$q_r$')

axes[2].plot(z_traj, (m0 + phi) / m0, 'o-', ms=3, color='C2')
axes[2].axhline(1, color='gray', lw=1)
axes[2].set_ylabel(r'$(m_0+\phi)/m_0$')
axes[2].set_xlabel(r'$z$')

axes[0].invert_xaxis()  # so time runs left -> right (z decreasing)

plt.tight_layout()
plt.show()
