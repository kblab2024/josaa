"""
Main simulation driver for grating reflectivity calculation.

Implements the complete workflow from Chang et al. (2006):
1. Setup multilayer film and grating geometry
2. Compute plane-wave basis vectors
3. Build Green's function tensor
4. Solve Lippmann-Schwinger equation
5. Compute reflection coefficients

Optimized for Apple Silicon (M3) via NumPy/SciPy.
"""

import numpy as np
from .transfer_matrix import compute_reflection_coefficients, compute_Q_per_layer
from .greens_function import GreensFunctionTensor
from .solver import LippmannSchwinger
from .cylindrical import (compute_W, compute_segment_integral_cylindrical,
                           compute_diagonal_weight_cylindrical)


class GratingSimulation:
    """
    Complete simulation for 3D grating reflectivity on multilayer films.

    Supports:
    - Rectangular lattice with arbitrary periodicity
    - Cylindrical contact holes
    - TE and TM polarization
    - Multiple wavelengths
    """

    def __init__(self, a1, a2, layers, Nx=15, Ny=15):
        """
        Parameters
        ----------
        a1, a2 : array-like, shape (2,)
            Primitive lattice vectors in the x-y plane (nm).
        layers : list of dict
            Layer specifications from top to bottom:
            [{'epsilon': complex, 'thickness': float}, ...]
            First = superstrate (thickness=0), last = substrate (thickness=0).
        Nx, Ny : int
            Number of plane waves in x and y directions.
        """
        self.a1 = np.array(a1, dtype=float)
        self.a2 = np.array(a2, dtype=float)
        self.layers = layers
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny

        # Compute reciprocal lattice vectors: b_i · a_j = 2π δ_{ij}
        self._compute_reciprocal_vectors()

    def _compute_reciprocal_vectors(self):
        """Compute reciprocal lattice vectors b1, b2."""
        a1x, a1y = self.a1
        a2x, a2y = self.a2
        det = a1x * a2y - a1y * a2x
        self.b1 = 2 * np.pi / det * np.array([a2y, -a2x])
        self.b2 = 2 * np.pi / det * np.array([-a1y, a1x])

    def _compute_kn_vectors(self, k0_vec):
        """
        Generate all in-plane wavevectors kn = k0 + gn.

        Parameters
        ----------
        k0_vec : array-like, shape (2,)
            Incident in-plane wavevector (k0x, k0y).

        Returns
        -------
        kn_vectors : list of ndarray, shape (2,)
            All plane-wave in-plane wavevectors.
        """
        k0_vec = np.array(k0_vec)
        kn_vectors = []
        for i in range(-(self.Nx // 2), self.Nx // 2 + 1):
            for j in range(-(self.Ny // 2), self.Ny // 2 + 1):
                gn = i * self.b1 + j * self.b2
                kn_vectors.append(k0_vec + gn)
        # Truncate to N vectors
        return kn_vectors[:self.N]

    def compute_reflectivity(self, wavelengths, theta=0.0, phi=0.0,
                             M=None, tol=1e-6, maxiter=100):
        """
        Compute reflection coefficients for given wavelengths.

        Parameters
        ----------
        wavelengths : array-like
            Wavelengths in nm.
        theta : float
            Polar angle of incidence (radians).
        phi : float
            Azimuthal angle of incidence (radians).
        M : int or None
            Number of z-segments. If None, uses M = 15*d/λ.
        tol : float
            Convergence tolerance.
        maxiter : int
            Maximum iterations.

        Returns
        -------
        results : dict
            Keys: 'wavelength', 'r_TE', 'r_TM', 'R_TE', 'R_TM'
        """
        wavelengths = np.atleast_1d(wavelengths)
        eps_superstrate = self.layers[0]['epsilon']

        results = {
            'wavelength': wavelengths,
            'r_TE': np.zeros(len(wavelengths), dtype=complex),
            'r_TM': np.zeros(len(wavelengths), dtype=complex),
            'R_TE': np.zeros(len(wavelengths)),
            'R_TM': np.zeros(len(wavelengths)),
        }

        d = sum(l['thickness'] for l in self.layers if l['thickness'] > 0)

        for idx, lam in enumerate(wavelengths):
            k0 = 2 * np.pi / lam
            n_sup = np.sqrt(eps_superstrate + 0j)

            # Incident wavevector components
            k0x = k0 * n_sup.real * np.sin(theta) * np.cos(phi)
            k0y = k0 * n_sup.real * np.sin(theta) * np.sin(phi)
            k0_vec = np.array([k0x, k0y])

            # Generate plane-wave basis
            kn_vectors = self._compute_kn_vectors(k0_vec)

            # Number of segments: M = 15*d/λ gives convergent results per paper §5,
            # with a minimum of 5 segments for numerical stability
            M_val = M if M is not None else max(5, int(15 * d / lam))

            # Build solver
            solver = LippmannSchwinger(
                self.layers, k0, kn_vectors, M=M_val
            )

            # Perturbation: V = (ε_host - ε_grating) * k0²
            eps_host = self.layers[0]['epsilon']  # superstrate
            # Find grating layer
            eps_grating = 1.0  # air hole (contact hole)
            for l in self.layers:
                if l['thickness'] > 0:
                    eps_grating = l.get('epsilon_grating', 1.0)
                    break

            V_val = (eps_host - eps_grating) * k0**2

            # Incident field (TE: Ey polarization)
            psi0_TE = np.zeros((self.N, M_val), dtype=complex)
            psi0_TM = np.zeros((self.N, M_val), dtype=complex)

            # Principal order (n=0)
            for j in range(M_val):
                z_j = solver.z_mesh[j]
                # Simple plane wave in TE
                kn0 = np.array(kn_vectors[0])
                kn0_mag_sq = float(np.sum(np.abs(kn0)**2))
                qn0 = np.sqrt(kn0_mag_sq - eps_superstrate * k0**2 + 0j)
                psi0_TE[0, j] = np.exp(-1j * qn0 * z_j)  # downward propagating
                psi0_TM[0, j] = np.exp(-1j * qn0 * z_j)

            # Solve TE
            Ey, info_TE = solver.solve_decoupled_TE(psi0_TE, V_val, tol=tol, maxiter=maxiter)

            # Solve TM with errata correction #4
            Kx = np.array([abs(kn[0]) for kn in kn_vectors])
            eps_inv = 1.0 / eps_host if abs(eps_host) > 1e-30 else 0.0
            Hy, info_TM = solver.solve_decoupled_TM(
                psi0_TM, V_val, eps_inv, Kx, tol=tol, maxiter=maxiter
            )

            # Extract reflection coefficient from solved fields
            # r = r0 + projected LS integral (Eq. 2)
            r0_TE = self._compute_r0(k0, eps_superstrate, theta, "TE")
            r0_TM = self._compute_r0(k0, eps_superstrate, theta, "TM")

            # Projection integral for principal order
            delta_r_TE = self._project_scattered_field(Ey, solver, V_val, 0)
            delta_r_TM = self._project_scattered_field(Hy, solver, V_val, 0)

            results['r_TE'][idx] = r0_TE + delta_r_TE
            results['r_TM'][idx] = r0_TM + delta_r_TM
            results['R_TE'][idx] = abs(results['r_TE'][idx])**2
            results['R_TM'][idx] = abs(results['r_TM'][idx])**2

        return results

    def _compute_r0(self, k0, eps_sup, theta, mode):
        """
        Compute zeroth-order reflection coefficient for the multilayer
        without grating.
        """
        n_sup = np.sqrt(eps_sup + 0j)
        kz_sup = k0 * n_sup * np.cos(theta)

        # Simple Fresnel for first interface
        if len(self.layers) < 2:
            return 0.0

        eps_next = self.layers[1]['epsilon']
        n_next = np.sqrt(eps_next + 0j)
        kz_next = k0 * np.sqrt(eps_next - eps_sup * np.sin(theta)**2 + 0j)

        if mode == "TE":
            r0 = (kz_sup - kz_next) / (kz_sup + kz_next)
        else:  # TM
            r0 = (eps_next * kz_sup - eps_sup * kz_next) / (eps_next * kz_sup + eps_sup * kz_next)

        # For full multilayer, apply transfer matrix
        for l_idx in range(1, len(self.layers) - 1):
            d_l = self.layers[l_idx]['thickness']
            if d_l <= 0:
                continue
            eps_l = self.layers[l_idx]['epsilon']
            kz_l = k0 * np.sqrt(eps_l - eps_sup * np.sin(theta)**2 + 0j)
            phase = np.exp(2j * kz_l * d_l)

            if l_idx + 1 < len(self.layers):
                eps_next = self.layers[l_idx + 1]['epsilon']
                kz_next = k0 * np.sqrt(eps_next - eps_sup * np.sin(theta)**2 + 0j)
                if mode == "TE":
                    r_interface = (kz_l - kz_next) / (kz_l + kz_next)
                else:
                    r_interface = (eps_next * kz_l - eps_l * kz_next) / (eps_next * kz_l + eps_l * kz_next)
                r0 = (r0 + r_interface * phase) / (1 + r0 * r_interface * phase)

        return r0

    def _project_scattered_field(self, field, solver, V_val, n_idx):
        """
        Project scattered field to extract reflection coefficient
        contribution (Eq. 2).
        """
        delta_r = 0.0
        for j in range(solver.M):
            delta_r += field[n_idx, j] * V_val * solver.dz
        return delta_r


class ContactHoleSimulation(GratingSimulation):
    """
    Specialized simulation for 2D periodic arrays of cylindrical
    contact holes on multilayer films.

    Uses the cylindrical basis expansion from Section 4 of the paper.
    """

    def __init__(self, period_x, period_y, hole_diameter, hole_depth,
                 epsilon_film, epsilon_substrate, epsilon_hole=1.0,
                 Nx=15, Ny=15):
        """
        Parameters
        ----------
        period_x, period_y : float
            Grating period in x and y (nm).
        hole_diameter : float
            Contact hole diameter (nm).
        hole_depth : float
            Contact hole depth (nm).
        epsilon_film : complex
            Dielectric constant of the film (e.g., poly-Si).
        epsilon_substrate : complex
            Dielectric constant of the substrate (e.g., c-Si).
        epsilon_hole : complex
            Dielectric constant inside the hole (default: 1.0 for air).
        Nx, Ny : int
            Number of plane waves.
        """
        self.hole_radius = hole_diameter / 2.0
        self.hole_depth = hole_depth
        self.epsilon_hole = epsilon_hole

        layers = [
            {'epsilon': 1.0, 'thickness': 0},          # superstrate (air)
            {'epsilon': epsilon_film, 'thickness': hole_depth,
             'epsilon_grating': epsilon_hole},           # grating layer
            {'epsilon': epsilon_substrate, 'thickness': 0}  # substrate
        ]

        a1 = [period_x, 0]
        a2 = [0, period_y]

        super().__init__(a1, a2, layers, Nx, Ny)

    def compute_W_integrals(self, kn, m_max=2, nu_max=2):
        """
        Compute W_j(m, ν) integrals for the cylindrical expansion.

        Parameters
        ----------
        kn : float
            In-plane wavenumber.
        m_max, nu_max : int
            Maximum radial and angular indices.

        Returns
        -------
        W_table : dict
            W[(m, nu)] values.
        """
        W_table = {}
        for m in range(m_max + 1):
            for nu in range(-nu_max, nu_max + 1):
                W_table[(m, nu)] = compute_W(m, abs(nu), kn, self.hole_radius)
        return W_table
