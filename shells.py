import time
import scipy
import h5py as h5
import numpy as np
import warnings

import ic
from timing import timed
from logger import createLog
from phi import solvePhi, interpPhi
from force import solveYukawaForce, solveGravityForce



## Constants
T_NU_EV = 1.68e-4             # Neutrino temperature today [eV]
Q_MEAN  = 3.1514               # <q/T_nu> for Fermi-Dirac distribution
EV_2_PC = 1/(1.57e23)          # eV to pc conversion
PC_2_M  = 3.08567758128e16     # pc to meter conversion
H0_EV   = 1.446e-33            # Hubble constant today  [eV] (Assuming h=0.678)
M_PL    = 2.435e27               # Reduced Planck mass [eV]




# ---------------------------------------------------------------------------
class Shells:
# ---------------------------------------------------------------------------
    def __init__(self):
        self._dtype = np.dtype([
            ('ID',  np.int32),     # Shell ID
            ('R',   np.float64),   # hat_r   = r / r_phi 
            ('q',   np.float64),   # hat_qr  = q_r / T_nu
            ('ell', np.float64),   # hat_ell = r * q_perp / T_nu
            ('w',   np.float64),   # weight (dimensionless)
            ('phi', np.float64),   # hat_phi = phi / T_nu
            ('psi', np.float64),   # hat_psi (grav. potential)
            ('m',   np.float64),   # hat_m   = m_tilde / T_nu
            ('eps', np.float64),   # hat_eps = eps / T_nu
            ('F_fs', np.float64),  # Free-streaming term
            ('F_lr', np.float64),  # Long-range force term
            ('F_g',  np.float64),  # Gravity force term (DM only)
        ])

        self.data       = None
        self.inv_cdf    = None
        self.cumMass    = None
        self.hdf5_io    = None
        self.iter_m     = None
        self.iter_tol   = None
        self.log        = None
        self.verb       = None
        self.grav       = None
        self.df         = None

        # Physical inputs [eV]
        self.g      = None
        self.m_phi  = None
        self.r_phi  = None
        self.m_nu   = None
        self.T_nu   = None
        self.H0     = None
        self.frange = None

        # Derived dimensionless quantities
        self.alpha     = None   # g^2 * T^2 / m_phi^2
        self.alphap    = None   # g^2 * mnu^2 / m_phi^2
        self.beta      = None   # T^4 / (m_phi^2 * m_pl^2)
        self.eta       = None   # beta / alpha
        self.hat_M_phi = None   # M_phi / H0
        self.m0_hat    = None   # m_nu / T_nu 
        self.m0        = None   # alias for m0_hat
        self.rhobar    = None   # 

        # Grid bounds [tilde_r]
        self.Rmin  = None
        self.Rmax  = None
        self.R0    = None

        # Time
        self.a      = None
        self.a_NR   = None
        self.eta    = None
        self.dt     = None
        self.curr   = None
        self.meas   = None
        self.a_ini  = None
        self.a_end  = None   # when force range shrinks below Rmin

        # External background field table
        self._phi_bkg_table_a = None
        self._phi_bkg_table_v = None


    @timed("Init")
    def init(self,
             Nshells  = 1000,
             g        = 1e-26,
             m_phi    = 1e-29,          # eV
             m_nu     = 0.06,            # eV
             kappa    = 0.75,           # a_ini = kappa * a_NR (ignored if a_ini given directly)
             kappa2   = 0.75,           # a_end = kappa2 * 1 / R_min (ignored if a_end given directly)
             a_ini    = None,           # Directly set the starting scale factor, overrides kappa
             a_end    = None,           # Directly set the ending scale factor, overrides kappa2
             Rmin     = None,           # Directly set the inner domain boundary
             Rmax     = None,           # Directly set the outer domain boundary
             dt_frac  = 0.3,            # Courant factor
             soft     = 1e-3,           # softening length: "minimum Radius"
             iter_m   = 'anderson',     # method in phi iteration
             iter_tol = 1e-3,           # tolerance in phi iteration
             hdf5_io  = False,          # HDF5 I/O
             seed     = 9,              # IC seed
             odir     = 'output',       # Output directory
             verb     = 0,              # Verbosity level
             to_file  = True,           # Log to file instead of print
             psi_boost = 1.0,           # Multiplier on the seed potential Psi(r)
             ic_file  = None,           # CSV path (columns R,qr,qt,w) to load ICs from
             phi_bkg_file = None        # CSV path (columns a,phi_bkg) for the background
                                        # field beyond Rmax, precomputed externally
            ):
        """
        Initialize N shells in dimensionless units.
        Parameters
        ----------
        Nshells : int, optional
            Number of shells. Default is 1000.
        g : float, optional
            Coupling constant. Default is 1e-26.
        m_phi : float, optional
            Mediator mass in eV. Default is 1e-29.
        m_nu : float, optional
            Neutrino mass in eV. Default is 0.1.
        kappa : float, optional
            Sets the initial scale factor: a_ini = kappa * a_NR.
            Ignored if a_ini is given directly.
        kappa2 : float, optional
            Sets the final scale factor: a_end = kappa2 / R_min.
            Ignored if a_end is given directly.
        a_ini : float, optional
            Directly set the starting scale factor, bypassing kappa's
            physically-motivated derivation. Default is None (use kappa).
        a_end : float, optional
            Directly set the ending scale factor, bypassing kappa2's
            physically-motivated derivation. Default is None (use kappa2).
        Rmin : float, optional
            Directly set the inner domain boundary (reflecting wall),
            bypassing the default 0.01*l_phi derivation.
        Rmax : float, optional
            Directly set the outer domain boundary (reflecting wall),
            bypassing the default 10*max(l_phi, l_fs) derivation.
        dt_frac : float, optional
            Courant factor controlling timestep size.
        soft : float, optional
            Minimum radius to be resolved.
        iter_m : str, optional
            Method used for phi iteration (e.g. 'anderson').
        iter_tol : float, optional
            Convergence tolerance for phi iteration.
        hdf5_io : bool, optional
            If True, enable HDF5 output.
        seed : int, optional
            Random seed for initial condition generation.
        odir : str, optional
            Output directory.
        verb : int, optional
            Verbosity level (0: INFO, 1: DEBUG).
        to_file : bool, optional
            If True, log output to file instead of printing.
        psi_boost : float, optional
            Multiplier applied to the seed potential Psi(r). Default is 1.0
            (no change). Useful for testing with a stronger/weaker seed
            perturbation without changing the underlying physical params.
        ic_file : str, optional
            Path to a CSV file with a header row and columns R, qr, qt, w,
            giving each shell's radius, radial momentum, transverse
            momentum, and phase-space weight. If given, these replace the
            built-in IC generator entirely (no Fermi-Dirac/Psi sampling),
            and Nshells is overridden by the file's row count. Rows are
            re-sorted ascending in R internally (required by the force
            solver), so input order does not matter.
        """

        start = time.time()
        self.verb = verb
        self.psi_boost = psi_boost
        self.log = createLog(self.verb, toFile=to_file)
        self.to_file = to_file
        self.hdf5_io  = hdf5_io
        self.iter_m   = iter_m
        self.iter_tol = iter_tol
        np.random.seed(seed)

        # -------------------------------------------------------------------
        # Store eV inputs for reference and check g
        # -------------------------------------------------------------------
        self.g     = g
        self.m_phi = m_phi
        self.r_phi = 1/m_phi
        self.m_nu  = m_nu
        self.T_nu  = T_NU_EV
        self.H0    = H0_EV
        self.frange = self.r_phi*EV_2_PC*1e-6 # Mpc

        self._check_g()

        # -------------------------------------------------------------------
        # Derived dimensionless ratios
        # -------------------------------------------------------------------
        self.alpha     = g**2 * self.T_nu**2 / m_phi**2
        self.alphap    = g**2 * self.m_nu**2 / m_phi**2
        self.beta      = self.T_nu**4 / (self.m_phi**2 * M_PL**2)
        self.eta       = self.beta / self.alpha
        self.m_phi_hat = m_phi / self.H0     # for the da/d(tilde_eta) update
        self.m0_hat    = m_nu  / self.T_nu
        self.m0        = self.m0_hat

        # -------------------------------------------------------------------
        # Scale factors (Not used if I input initial conditions)
        # -------------------------------------------------------------------
        self.a_NR  = 1.0 / self.m0_hat
        if a_ini is None:
            a_ini = kappa * self.a_NR
        self.a     = a_ini
        self.a_ini = a_ini
        self.eta   = 2/self.H0*np.sqrt(self.a)
        self.curr  = 0
        self.meas  = 0

        # -------------------------------------------------------------------
        # Length/time scales (Not used for input initial conditions)
        # -------------------------------------------------------------------
        self.l_phi = 1.0 / a_ini
        l_fs_rad   = self._lambdaFS_rad()
        self.l_fs  = l_fs_rad + 2.0 / self.m_phi_hat * \
                         (1.0/np.sqrt(a_ini) - 1.0/np.sqrt(self.a_NR))

        self.Rmin  = 0.01 * self.l_phi if Rmin is None else Rmin
        self.Rmax  = 10.0 * max(self.l_phi, self.l_fs) if Rmax is None else Rmax
        if a_end is None:
            a_end = 1.0 / self.Rmin * kappa2
        self.a_end = a_end

        R0 = self.l_phi
        self.R0 = R0

        self.dt_frac = dt_frac
        self.dt = 0.001
        self.soft = soft

        # -------------------------------------------------------------------
        # Redshifts
        # -------------------------------------------------------------------
        z_ini = 1/self.a_ini-1
        z_end = 1/self.a_end-1

        # -------------------------------------------------------------------
        # Cosmological perturbation params (needed either way, for gravity)
        # -------------------------------------------------------------------
        #self.delta0 = 1.73342 / (z_ini**(0.7358) * self.frange**(0.0992))
        self.delta0 = 0.0
        self.omega0 = 0.3
        self.omegaL = 0.7   # Omega_DarkEnergy, flat universe: omega0 + omegaL = 1

        # -------------------------------------------------------------------
        # r_hat, q_hat -- either generated internally, or loaded from a CSV
        # -------------------------------------------------------------------
        if ic_file is not None:
            ic_data = np.genfromtxt(ic_file, delimiter=',', names=True)
            order   = np.argsort(ic_data['R'])   # force solver requires ascending R
            r_grid  = ic_data['R'][order]
            hat_qr  = ic_data['qr'][order]
            hat_qT  = ic_data['qt'][order]
            weights = ic_data['w'][order]
            hat_ell = r_grid * hat_qT
            Nshells = len(r_grid)

            psi = ic.compute_Psi(r_grid, self)
            self.df = None
            self.log.info(f"[IC] Loaded {Nshells} shells from {ic_file}")
        else:
            r_grid  = np.geomspace(self.Rmin, self.Rmax, Nshells)
            dr      = np.empty(Nshells)
            dr[:-1] = r_grid[1:] - r_grid[:-1]
            dr[-1]  = dr[-2]

            q_arr  = ic.sample_q(self, Nshells)
            mu_arr = np.random.uniform(-1.0, 1.0, Nshells)  # cos(theta)

            psi = ic.compute_Psi(r_grid, self)
            weights, df = ic.compute_weights(r_grid, dr, q_arr, psi, self.log)
            self.df = df

            hat_qr = q_arr * mu_arr
            hat_qT = q_arr * np.sqrt(np.maximum(1.0 - mu_arr**2, 0.0))
            hat_ell = r_grid * hat_qT

        # -------------------------------------------------------------------
        # Initial guess for Yukawa potential
        # -------------------------------------------------------------------
        _, phi_guess = self._solve_background()
        self.phi_bkg = -0.3179603255361576

        # -------------------------------------------------------------------
        # External background field table (a, phi_bkg), for the boundary
        # term representing material beyond Rmax -- precomputed externally
        # and interpolated here since evolving it in-loop is far too slow
        # (~23ms per _solve_background() call vs ~3ms per full step).
        # -------------------------------------------------------------------
        self._phi_bkg_table_a = None
        self._phi_bkg_table_v = None
        if phi_bkg_file is not None:
            tbl = np.genfromtxt(phi_bkg_file, delimiter=',', names=True)
            order = np.argsort(tbl['a'])
            self._phi_bkg_table_a = tbl['a'][order]
            self._phi_bkg_table_v = tbl['phi'][order]
            self.log.info(f"[IC] Loaded background field table from {phi_bkg_file}")
            self._update_phi_bkg()

        # -------------------------------------------------------------------
        # Initialiase data structure
        # -------------------------------------------------------------------
        self.data = np.zeros(Nshells, dtype=self._dtype)
        self.data['ID']   = np.arange(Nshells)
        self.data['R']    = r_grid
        self.data['q']    = hat_qr
        self.data['ell']  = hat_ell
        self.data['w']    = weights
        self.data['phi']  = np.full(Nshells, phi_guess)
        self.data['psi']  = psi
        self.log.debug("[IC] Arrays created!")

        # -------------------------------------------------------------------
        # Compute initial hat_phi self-consistently 
        # Then, mass, energy, cumulative mass are updated due to new phi value
        # For the first step we always choose anderson to make the convergence
        # is reached, however the tolerance can still be chosen
        # -------------------------------------------------------------------
        #_ = solvePhi(self, method="anderson", tol=iter_tol, verbose=verb)
        try:
            _ = solvePhi(self, method=self.iter_m,\
                         tol=self.iter_tol, verbose=self.verb)
        except scipy.optimize.NoConvergence:
            self.log.info("[Init] anderson solve_phi did not converge, "
                           "retrying with 'naive'")
            _ = solvePhi(self, method="naive", tol=self.iter_tol,
                         max_iter=3000, damp=0.3, verbose=self.verb)

        # -------------------------------------------------------------------
        # Set self.data["m"] and self.data["eps"]
        # -------------------------------------------------------------------
        self._update_mass()
        self._update_eps()
        min_m = np.min(self.m/self.m0)
        max_m = np.max(self.m/self.m0)

        # -------------------------------------------------------------------
        # Commit forces for the first half-kick
        # -------------------------------------------------------------------
        F_fs, F_lr = solveYukawaForce(self)
        self.data["F_fs"] = F_fs
        self.data["F_lr"] = F_lr

        self.grad_psi    = np.gradient(psi)
        self.data["F_g"] = self.eps * np.gradient(psi) # this won't change

        # -------------------------------------------------------------------
        # Uncomment this to turn on neutrino self-gravity, expected
        # however to be fairly negligible
        # self._update_cumMass()
        # self._update_rhobar()
        # F_sg = solveGravityForce(self)
        # self.data["F_g"] += F_sg
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # Find max acceleration and impose limit on dt
        # -------------------------------------------------------------------
        self._update_dt()

        # -------------------------------------------------------------------
        # Summary print
        # -------------------------------------------------------------------
        self.printParams(to_file=to_file)
        self.log.info(f"ICs took {time.time()-start:.5f} s")



    # -----------------------------------------------------------------------
    # Mass update
    # -----------------------------------------------------------------------
    @timed("update_mass")
    def _update_mass(self):
        """ """
        self.data['m'] = self.m0  + self.data["phi"]
        if np.any(self.m < 0):
            self.log.info("The effective mass has negative values!")


    # -----------------------------------------------------------------------
    # Energy update
    # -----------------------------------------------------------------------
    @timed("update_eps")
    def _update_eps(self):
        """ """
        self.data['eps'] = np.sqrt(self.q**2 + self.ell**2 / self.R**2
                                 + self.a**2 * self.m**2)


    # -----------------------------------------------------------------------
    # Update cumulative weighted mass
    # -----------------------------------------------------------------------
    @timed("update_cumMass")
    def _update_cumMass(self):
        """
        Cumulative sum M(<tilde_r) = sum_{R_i <= tilde_r} w_i * eps_i
        Assumes data is already sorted ascending by R.
        """
        self._cumMass = np.cumsum(self.w * self.eps)


    # -----------------------------------------------------------------------
    # Compute background neutrino energy density
    # -----------------------------------------------------------------------
    @timed("update_rhobar")
    def _update_rhobar(self):
        """
        Compute dimensionless background energy density
        rho_bar = a^4 * rho_nu / T^4
                = 1/(2pi^2) * int q^2 * eps(q) * f0(q) dq
        In the ultra-relativistic limit this is 7*pi^2/120 (constant).
        """
        a  = self.a
        m0 = self.m0

        def integrand(q):
            ep = np.sqrt(q**2 + (a * m0)**2)
            return q**2 * ep / (np.exp(np.clip(q, 0, 500)) + 1.0)

        result, _ = scipy.integrate.quad(integrand, 0, 30)
        self.rhobar = result / (2.0 * np.pi**2)


    # -----------------------------------------------------------------------
    # Scale factor update
    # -----------------------------------------------------------------------
    def _update_a(self):
        """
        Scale factor update
        Matter + dark energy: H^2 = H0^2*(Omega_m/a^3 + Omega_L)
                    da/deta = a^2*H = H0*sqrt(Omega_m*a + Omega_L*a^4)
                    da/d(hat_eta) = sqrt(omega0*a + omegaL*a**4) / m_phi_hat
        H0 here is the real, measured present-day Hubble constant (H0_EV),
        so this needs the explicit Omega_m, Omega_L factors -- omitting them
        implicitly assumes Omega_m=1 (Einstein-de Sitter) while still using
        the real H0.
        """
        self.a += self.dt * np.sqrt(self.omega0*self.a + self.omegaL*self.a**4) \
                   / self.m_phi_hat
        z = 1/self.a-1
        self.log.debug(f"[da] Time update -> {self.a:.5f}")


    # -----------------------------------------------------------------------
    # Background field update (from externally-supplied table, if any)
    # -----------------------------------------------------------------------
    def _update_phi_bkg(self):
        """
        Interpolate self.phi_bkg from the externally-loaded (a, phi_bkg)
        table at the current scale factor. No-op if no table was loaded
        (phi_bkg_file=None), in which case phi_bkg keeps whatever value it
        already has. Clamps at the table edges rather than extrapolating,
        same convention as interpPhi.
        """
        if self._phi_bkg_table_a is not None:
            self.phi_bkg = float(np.interp(
                self.a, self._phi_bkg_table_a, self._phi_bkg_table_v,
                left=self._phi_bkg_table_v[0], right=self._phi_bkg_table_v[-1]))


    # -----------------------------------------------------------------------
    # Time=step update
    # -----------------------------------------------------------------------
    def _update_dt(self):
        """ Update timestep according to acceleration"""
        F_tot = np.abs(self.F_fs - self.F_lr - self.F_g)

        mask = F_tot > 0
        dt_acc = np.full(self.N, np.inf)
        dt_acc[mask] = np.sqrt(2 * self.soft / F_tot[mask])

        #mask1 = self.F_fs > 0
        #dt_acc1 = np.full(self.N, np.inf)
        #dt_acc1[mask1] = np.sqrt(2 * self.soft / self.F_fs[mask1])

        #mask2 = self.F_lr > 0
        #dt_acc2 = np.full(self.N, np.inf)
        #dt_acc2[mask2] = np.sqrt(2 * self.soft / self.F_lr[mask2])

        dt_new = self.dt_frac * np.min(dt_acc)
        self.log.debug(f"[dt] Timestep update {self.dt:.5f} -> {dt_new:.5f}")
        self.dt = dt_new


    # -----------------------------------------------------------------------
    # Sorting routine
    # -----------------------------------------------------------------------
    @timed("Sort")
    def _sort(self):
        """Sort shells in ascending hat_r order (stable / merge sort)."""
        idx = np.argsort(self.R, kind='stable')
        for arr in (self.ID, self.R, self.q, self.ell, self.w, \
                    self.m, self.eps, self.phi):
            arr[:] = arr[idx]

    # -----------------------------------------------------------------------
    #  Time step: kick-drift-kick leapfrog
    # -----------------------------------------------------------------------
    def step(self, fs=True, lr=True, gr=True):
        """
        Advance by one tilde_eta step using kick-drift-kick (KDK) leapfrog.

        EOM:
            d(hat_r)/d(hat_eta) = hat_q / hat_eps

            d(hat_q)/d(hat_eta) = hat_ell^2 / (hat_eps * hat_r^3)          [FS]
                                - a^2 * alpha * hat_m / hat_eps * F_kernel [LR]
                                - hat_eps * dPsi/dhat_r                    [GR]

            da      /d(hat_eta) = sqrt(a) / m_phi_hat                [update_a]

        Force prefactor:
            alpha = g^2 * T^2_nu / m^2_phi
        """
        self.log.debug("-" * 50)
        self.log.debug(f"step {self.curr:5d}")
        self.log.debug("-" * 50)

        dt    = self.dt
        F_fs  = self.F_fs if fs else np.zeros(self.N)
        F_lr  = self.F_lr if lr else np.zeros(self.N)
        F_g   = self.F_g
        F_tot = F_fs - F_lr - F_g

        # -- Half kick --
        self.data['q'] += 0.5 * dt * F_tot
        self._update_eps()

        # -- Store pre-drift state for phi interpolation --
        R_old   = self.data['R'].copy()
        phi_old = self.data['phi'].copy()

        # -- Full drift --
        ## Add dt check based on q/eps
        self.data['R'] += dt * self.data['q'] / self.data['eps']

        # -- Reflecting boundary conditions --
        lo = self.data['R'] < self.soft
        self.data['R'][lo]  = 2.0*self.Rmin - self.data['R'][lo]
        self.data['q'][lo] *= -1.0

        hi = self.data['R'] > self.Rmax
        self.data['R'][hi]  = 2.0*self.Rmax - self.data['R'][hi]
        self.data['q'][hi] *= -1.0
        if hi.any():
            if not hasattr(self, 'reflect_count_hi'):
                self.reflect_count_hi = np.zeros(self.N, dtype=int)
            self.reflect_count_hi[self.data['ID'][hi]] += 1

        # -- Sort and updates --
        self._sort()
        self._update_a()
        self._update_phi_bkg()
        self._update_eps()

        self.curr += 1

        # -- Phi and Force updates
        phi0_interp = interpPhi(R_old, phi_old, self.data['R'])
        self.data['phi'] = phi0_interp
        try:
            _ = solvePhi(self, method=self.iter_m,\
                         tol=self.iter_tol, verbose=self.verb)
        except scipy.optimize.NoConvergence:
            # Anderson mixing can fail to converge once a shell's mass is
            # pinned against the -0.999*m0 floor (a non-smooth kink in the
            # fixed-point map), which happens transiently during deep
            # collapse. Fall back to the slower but more robust damped
            # 'naive' iteration instead of aborting the run.
            self.log.info(f"[step {self.curr:5d}] {self.iter_m} solve_phi "
                           "did not converge, retrying with 'naive'")
            _ = solvePhi(self, method="naive", tol=self.iter_tol,
                         max_iter=3000, damp=0.3, verbose=self.verb)

        self._update_mass()
        self._update_eps()

        F_fs, F_lr = solveYukawaForce(self)
        self.data["F_fs"] = F_fs if fs else np.zeros(self.N)
        self.data["F_lr"] = F_lr if lr else np.zeros(self.N)
        self.data["F_g"]  = self.data["eps"] * self.grad_psi

        # Uncomment this for neutrino self-gravity
        # Self._update_cumMass()
        # Self._update_rhobar()
        # F_sg = solveGravityForce(self)
        # self.data["F_g"] += F_sg

        self._update_dt()

        # -- Second half kick  --
        self.data['q'] += 0.5 * dt * (F_fs - F_lr - F_g)
        self._update_eps()


    # -----------------------------------------------------------------------
    # Free-streaming utility
    # -----------------------------------------------------------------------
    def _lambdaFS_rad(self, ai=1e-8):
        """Return the free-streaming scale in R.D. in units of 1/m_phi."""
        Omega_r = 9.2e-5
        def v(a):
            p = Q_MEAN * T_NU_EV / a
            E = np.sqrt(p**2 + self.m_nu**2)
            return p / E / a

        I, err = scipy.integrate.quad(v, ai, self.a_ini)
        lambda_FS_H0 = I / np.sqrt(Omega_r)

        return lambda_FS_H0/self.m_phi_hat


    # -----------------------------------------------------------------------
    # Check g utility
    # -----------------------------------------------------------------------
    def _check_g(self):
        """
        Checks magnitude of g to enforce the right parameter space
        for the long rangbe force to have an effect
        Based on Eqs. (4), (6) in https://arxiv.org/pdf/2412.20766
        """
        # Gravity check
        f_nu = 0.0045 * self.m_nu / 0.06
        gc1 = self.m_nu / (M_PL * np.sqrt(f_nu))

        # mphi check
        gc2 = 12 * self.m_phi / self.m_nu

        self.log.debug(f"Bounds on g>{gc1:.3e}, g>{gc2:.3e}, g is {self.g:.3e}")
        # Get the max as a minimum requirement
        gc = max(gc1, gc2)
        if gc >= self.g:
            raise RuntimeError(f"Set g larger than {gc:.3e}!")


    # -----------------------------------------------------------------------
    # Save configuration
    # -----------------------------------------------------------------------
    def _save_hdf5(self, path, step_index):
        """Save shell state to a hdf5 file."""
        with h5.File(f"{path}/states/shells_{step_index:05d}.hdf5", 'w') as f:

            head = f.create_group("Header")
            head.attrs['N'] = self.N
            head.attrs['a'] = self.a
            head.attrs['g'] = self.g
            head.attrs['m_phi'] = self.m_phi_hat
            head.attrs['alpha'] = self.alpha
            head.attrs['Rmin'] = self.Rmin
            head.attrs['Rmax'] = self.Rmax
            head.attrs['R0'] = self.R0
            head.attrs['m0'] = self.m0
            head.attrs['delta0'] = self.delta0
            head.attrs['omega0'] = self.omega0
            head.attrs['omegaL'] = self.omegaL
            head.attrs['dt'] = self.dt
            head.attrs['phib'] = self.phi_bkg

            data = f.create_group("Data")
            data.create_dataset("ID",   data=self.ID,    dtype=np.int32)
            data.create_dataset("R",    data=self.R,     dtype=np.float32)
            data.create_dataset("q",    data=self.q,     dtype=np.float32)
            data.create_dataset("ell",  data=self.ell,   dtype=np.float32)
            data.create_dataset("m",    data=self.m,     dtype=np.float32)
            data.create_dataset("eps",  data=self.eps,   dtype=np.float32)
            data.create_dataset("w",    data=self.w,     dtype=np.float32)
            data.create_dataset("phi",  data=self.phi,   dtype=np.float32)
            data.create_dataset("psi",  data=self.psi,   dtype=np.float32)
            data.create_dataset("F_fs", data=self.F_fs,  dtype=np.float32)
            data.create_dataset("F_lr", data=self.F_lr,  dtype=np.float32)
            data.create_dataset("F_g",  data=self.F_g,   dtype=np.float32)


    @timed("I/O")
    def _save(self, path, step_index):
        """Save shell state to a text or hdf5 file."""
        self.meas += 1
        if self.hdf5_io:
            self._save_hdf5(path, step_index)
        else:
            header = (
                f"ID R q ell m w phi F_fs F_lr\n"
                f"a={self.a:.6e}\n"
                f"Rmin={self.Rmin:.2e}, Rmax={self.Rmax:.2e}"
            )
            np.savetxt(
                f"{path}/states/shells_{step_index:05d}.txt",
                np.column_stack([
                    self.data['ID'], self.data['R'],
                    self.data['q'],  self.data['ell'],
                    self.data['m'],  self.data['w'],
                    self.data['phi'], self.data['F_fs'],
                    self.data['F_lr'], self.data['F_g']
                ]),
                header=header,
                fmt="%d %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e"
            )

        if self.to_file:
            self.log.info("=======================================================")
            self.log.info(f"Meas #{self.meas:3d} a:{self.a:.5f} z:{1/self.a-1:.1f}")
            self.log.info("=======================================================\n")
        else:
            print("\n=======================================================")
            print(f"Meas #{self.meas:3d} a:{self.a:.5f} z:{1/self.a-1:.1f}")
            print("=======================================================\n")


    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------
    def _load_hdf5(self, path, step_index):
        """Load shell state from hdf5 file."""
        with h5.File(f"{path}/states/shells_{step_index:05d}.hdf5", 'r') as f:
            self.a =  float(f['Header'].attrs['a'])
            self.g =  float(f['Header'].attrs['g'])
            self.m0 = float(f['Header'].attrs['m0'])
            self.R0 = float(f['Header'].attrs['R0'])
            self.omega0 = float(f['Header'].attrs['omega0'])
            self.omegaL = float(f['Header'].attrs['omegaL'])
            self.delta0 = float(f['Header'].attrs['delta0'])
            self.dt = float(f['Header'].attrs['dt'])
            self.m_phi = float(f['Header'].attrs['m_phi'])
            self.alpha = float(f['Header'].attrs['alpha'])
            self.Rmin  = float(f['Header'].attrs['Rmin'])
            self.Rmax  = float(f['Header'].attrs['Rmax'])
            self.phi_bkg = float(f['Header'].attrs['phib'])

            N =  int(f['Header'].attrs['N'])
            self.data = np.zeros(N, dtype=self._dtype)

            self.data['ID']   = f['Data/ID']
            self.data['R']    = f['Data/R']
            self.data['q']    = f['Data/q']
            self.data['ell']  = f['Data/ell']
            self.data['m']    = f['Data/m']
            self.data['w']    = f['Data/w']
            self.data['eps']  = f['Data/eps']
            self.data['phi']  = f['Data/phi']
            self.data['psi']  = f['Data/psi']
            self.data['F_fs'] = f['Data/F_fs']
            self.data['F_lr'] = f['Data/F_lr']
            self.data['F_g']  = f['Data/F_g']


    @timed("I/O")
    def _load(self, path, step_index):
        """Load shell state from text or hdf5 file."""
        if self.hdf5_io:
            self._load_hdf5(path, step_index)
        else:
            fname = f"{path}/states/shells_{step_index:05d}.txt"

            # Read header to extract scale factor
            with open(fname, "r") as f:
                f.readline()
                line = f.readline()
                self.a = float(line.split("=")[-1])
                line = f.readline()
                self.Rmin = float(line.split(",")[0].split('=')[1])
                self.Rmax = float(line.split(",")[1].split('=')[1])

            raw = np.loadtxt(fname)

            N = raw.shape[0]
            self.data = np.zeros(N, dtype=self._dtype)

            self.data['ID']  = raw[:,0].astype(np.int32)
            self.data['R']   = raw[:,1]
            self.data['q']   = raw[:,2]
            self.data['ell'] = raw[:,3]
            self.data['m']   = raw[:,4]
            self.data['w']   = raw[:,5]
            self.data['phi'] = raw[:,6]
            self.data['F_fs'] = raw[:,7]
            self.data['F_lr'] = raw[:,8]
            self.data['F_g']  = raw[:,9]



    # -----------------------------------------------------------------------
    # Summary print
    # -----------------------------------------------------------------------
    def printParams(self, to_file):
        min_m = np.min(self.m/self.m0)
        max_m = np.max(self.m/self.m0)

        message = []
        message.append("")
        message.append("=" * 60)
        message.append("  Simulation parameters")
        message.append("=" * 60)
        message.append(f" \n   Nshells = {self.N}\n")
        message.append(f"   g       = {self.g:.3e}")
        message.append(f"   m_phi   = {self.m_phi:.3e} eV")
        message.append(f"   m_nu    = {self.m_nu:.3e} eV")
        message.append(f"   T_nu    = {self.T_nu:.3e} eV")
        message.append(f"   Range   = {self.frange:.3e} Mpc\n")
        message.append(f"   alpha   = {self.alpha:.3e}  ")
        message.append(f"   alpha'  = {self.alphap:.3e}  ")
        message.append(f"   delta0  = {self.delta0:.3f}  ")
        message.append(f"   m/m0    = [{min_m:.5f},{max_m:5f}]\n")
        message.append(f"   m_nu/T_nu = {self.m0_hat:.3e} ")
        message.append(f"   m_phi/H0  = {self.m_phi_hat:.3e}\n")
        message.append(f"   a_NR  = {self.a_NR:.3e}   ")
        message.append(f"   a_ini = {self.a_ini:.3e}  (z={1/self.a_ini-1:.1f})")
        message.append(f"   a_end = {self.a_end:.3e}  (z={1/self.a_end-1:.1f})")
        message.append(f"   dt    = {self.dt:.3e}     \n")
        message.append(f"   lambda_FS  = {self.l_fs:.3e}  (free-streaming)")
        message.append(f"   lambda_phi = {self.l_phi:.3e}  (Yukawa range)")
        message.append(f"   Rmin       = {self.Rmin:.3e}")
        message.append(f"   Rmax       = {self.Rmax:.3e}\n")
        message.append("=" * 60)
        message = "\n".join(message)

        if self.to_file:
            self.log.info(message)
        print(message)


    # -----------------------------------------------------------------------
    # Neutrino number density
    # -----------------------------------------------------------------------
    def density(self, nbins=200):
        """
        Bin shells by weight in hat_r and return (r_bins, delta).

        Returns
        -------
        r_c   : array  hat_r bin centres
        n     : array  number density
        n_err : array  standard error on n, from sqrt(sum(w_i^2)) per bin
                (the appropriate estimator for a weighted histogram, since
                shells carry unequal importance-sampling weights -- reduces
                to the usual sqrt(N) Poisson error only if all w_i are equal)
        """
        self._sort()
        edges = np.geomspace(np.min(self.R), np.max(self.R), nbins + 1)
        r_c   = np.sqrt(edges[:-1] * edges[1:])
        vol   = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)

        mass, _  = np.histogram(self.data['R'], bins=edges,
                                weights=self.data['w'])
        sumw2, _ = np.histogram(self.data['R'], bins=edges,
                                weights=self.data['w']**2)

        occ = mass > 0
        n      = np.full_like(r_c, np.nan)
        n_err  = np.full_like(r_c, np.nan)
        n[occ]     = mass[occ]  / vol[occ]
        n_err[occ] = np.sqrt(sumw2[occ]) / vol[occ]
        return r_c[occ], n[occ], n_err[occ]


    # -----------------------------------------------------------------------
    # Background phi/mass
    # -----------------------------------------------------------------------
    def _solve_background(self, max_iter=100, tol=1e-5):
        """
        Solve for the self-consistent effective mass hat_m = hat_m0 + <hat_phi>
        by iterating hat_m = hat_m0 / (1 + alpha/(4pi) * I(a, hat_m)), with
               I(a, hat_m) = int dq q^2/(exp(q)+1) * 1/sqrt(q^2 + a^2*M^2)
        This form guarantees hat_m > 0
        Returns
        -------
        hat_m     : float   self-consistent effective mass  hat_m0 + <hat_phi>
        phi_bg    : float   background <hat_phi> = hat_m - hat_m0
                            Units are 1/T and g/T, respectively.
        """
        alpha  = self.alpha
        m0     = self.m0_hat
        a      = self.a

        def I(M):
            if M <= 0:
                return np.inf
            def integrand(q):
                den = np.sqrt(q**2 + a**2 * M**2)
                return q**2 / (np.exp(np.clip(q, 0, 500)) + 1) / den
            result, _ = scipy.integrate.quad(integrand, 0, 30)
            return result

        # Initial guess: start from bare mass (phi=0)
        M = m0

        for i in range(max_iter):
            M_new = m0 / (1.0 + alpha / (a**2 * 12.0) * I(M))

            if M_new <= 0:
                raise ValueError(f"M went negative at iteration {i}. "
                                 f"alpha={alpha:.3e} may be too large.")

            if abs(M_new - M) < tol * m0:
                phi_bkg = M_new - m0       # always negative
                return M_new, phi_bkg

            M = M_new

        raise ValueError(f"Background phi did not converge after {max_iter} iterations. "
                         f"Last M={M:.6e}, m0={m0:.6e}")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------
    @property
    def ID(self):   return self.data['ID']
    @property
    def R(self):    return self.data['R']
    @property
    def q(self):    return self.data['q']
    @property
    def ell(self):  return self.data['ell']
    @property
    def w(self):    return self.data['w']
    @property
    def m(self):    return self.data['m']
    @property
    def eps(self):  return self.data['eps']
    @property
    def phi(self):  return self.data['phi']
    @property
    def psi(self):  return self.data['psi']
    @property
    def F_fs(self):  return self.data['F_fs']
    @property
    def F_lr(self):  return self.data['F_lr']
    @property
    def F_g(self):  return self.data['F_g']
    @property
    def N(self):    return len(self.data)


