import os
import tqdm
import shutil
import numpy as np

from timing import timed, report, reset, start_wall, stop_wall
from shells import Shells
from phi import solvePhi, interpPhi
from force import solveYukawaForce, solveGravityForce


# === Main params ============================================================
Nshells = 6000          # Number of simulation shells
g       = 1e-26        # Yukawa coupling constant
m_nu    = 0.06         # Neutrino  mass (eV)
m_phi   = 1e-29        # Scalar field mass (eV)
seed    = 9            # IC random seed 
dt_frac = 0.3          # Time step fractions (Courant-like)
ti_frac = 54.241701019743985  # Initial time controller (a_i=ti_frac*a_NR); z_ini~10, verified to survive past step 3200 (54.24 rounded crashes at step 2957)
tf_frac = 2.0          # Final time controller (a_f=tf_frac*1/mphi)
a_ini   = 0.0389507  
a_end   = 0.125   # z_end = 7
Rmin    = 0.204         # Directly set the inner domain boundary (reflecting wall)
Rmax    = 140           # Directly set the outer domain boundary (reflecting wall)
ic_file = 'shells.csv'  # qt=0 (no angular momentum barrier), qr/w unchanged
phi_bkg_file = 'phiback.csv'  # (a, phi) table for the boundary term beyond Rmax

# === Output params ==========================================================
odir = 'output'        # Output directory (data and logs)
nmeas = 100             # Number of measurements
logmeas = True         # Logarithmic spacing (in a) for measurements
hdf5_io = True         # Saved in HDF5 format (True) or .txt (False)
to_file = True         # Print to file (True) or to screen (False)
verb = 1               # Verbosity level (0:INFO, 1:DEBUG)
ic_only = False        # Only save ics

# === Force params ===========================================================
method = 'anderson'    # Iteration method:
                       #   'naive'    : brute force, ok for small g 
                       #   'anderson' : uses scipy.optimise.anderson
                       #                to accelerate the iteration
                       #   'noiter'   : one evaluation only
tol = 1e-2             # Tolerance level for the iterations
soft = 1e-2            # Softening parameter to define minimum radius 
keep_fs = True         # Keep free-streaming in the loop (False for testing)
keep_lr = True         # Keep Yukawa force in the loop (False for testing)

# ============================================================================

if __name__ == "__main__":

    start_wall()
    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)
    os.makedirs(odir+"/states")

    shells = Shells()
    shells.init(Nshells, g=g, m_phi=m_phi, m_nu=m_nu, kappa=ti_frac,
                kappa2=tf_frac, a_ini=a_ini, a_end=a_end, Rmin=Rmin, Rmax=Rmax,
                dt_frac=dt_frac, iter_m=method, iter_tol=tol, soft=soft,
                hdf5_io=hdf5_io, seed=seed, odir=odir, verb=verb,
                to_file=to_file, ic_file=ic_file, phi_bkg_file=phi_bkg_file)

    if ic_only:
        shells._save(odir, 0)
        print(f"ICs saved in {odir}")
        exit(0)

    if logmeas:
        saves_a = np.geomspace(shells.a_ini, shells.a_end, nmeas)
    else:
        saves_a = np.linspace(shells.a_ini, shells.a_end, nmeas)

    j = 0
    pbar = tqdm.tqdm(total=1)

    while True:
        shells.step(fs=keep_fs, lr=keep_lr)

        z = 1/shells.a - 1
        pbar.set_description(f"z={z:.1f}")
        pbar.update(1)

        while j < len(saves_a) and shells.a >= saves_a[j]:

            shells._save(odir, j)
            j += 1

        if shells.a >= shells.a_end:
            pbar.close()
            z_end = 1/shells.a_end-1
            print(f'\nReached z_end={z_end:.1f}, ending simulation!')
            break

    stop_wall()
    report(unit="s", path=odir+'/profile.log')


