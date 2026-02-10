"""
Lippmann-Schwinger equation solver using QMR iteration.

Implements Eqs. (19)-(28) of Chang et al. (2006), including:
- Analytical segment integration (Eq. 24)
- Matrix equation assembly (Eq. 25-28)
- QMR iterative solver

Errata correction #4 applied: Eq. (23) uses Hy - Hy^0(z), not Hy - Ez^0(z).
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from .greens_function import GreensFunctionTensor
from .transfer_matrix import compute_Q_per_layer


def _compute_Wj(Q, delta_z):
    """
    Compute segment integration weight W_j = (1 - exp(-Q*Δz)) * Q^{-1}.

    Parameters
    ----------
    Q : complex
        Vertical wave vector.
    delta_z : float
        Segment width.

    Returns
    -------
    Wj : complex
        Integration weight.
    """
    Qd = Q * delta_z
    if abs(Qd) < 1e-8:
        # Taylor expansion: (1 - exp(-x))/x ≈ 1 - x/2 + x²/6 - x³/24 + ...
        # so W = dz * (1 - Qd/2 + Qd²/6 - ...)
        return delta_z * (1.0 - Qd / 2.0 + Qd**2 / 6.0 - Qd**3 / 24.0)
    return (1.0 - np.exp(-Q * delta_z)) / Q


def _compute_Wj0(Q, delta_z):
    """
    Compute diagonal segment weight W_j^0 (Eq. 24).

    W_j^0 = 2 * [Δz_j - (1 - exp(-Q*Δz_j)) * Q^{-1}] * Q^{-1}

    Parameters
    ----------
    Q : complex
        Vertical wave vector.
    delta_z : float
        Segment width.

    Returns
    -------
    Wj0 : complex
        Diagonal integration weight.
    """
    Qd = Q * delta_z
    if abs(Qd) < 1e-8:
        # Taylor: dz - W = dz * (Qd/2 - Qd²/6 + ...) = dz²*Q*(1/2 - Qd/6 + ...)
        # Wj0 = 2*(dz - W)/Q = dz² * (1 - Qd/3 + Qd²/12 - ...)
        return delta_z**2 * (1.0 - Qd / 3.0 + Qd**2 / 12.0 - Qd**3 / 60.0)
    Wj_plus = _compute_Wj(Q, delta_z)
    return 2.0 * (delta_z - Wj_plus) / Q


class LippmannSchwinger:
    """
    Solves the Lippmann-Schwinger equation for electromagnetic scattering
    from 3D gratings on multilayer films.

    The LS equation:
        Ψ(z) = Ψ₀(z) + ∫ G(z,z') V(z') Ψ(z') dz'

    is discretized using piecewise-constant (m=0) or piecewise-linear (m=1)
    basis functions, and solved iteratively via GMRES (similar to QMR).
    """

    def __init__(self, layers, k0, kn_vectors, M=20, basis_order=0):
        """
        Parameters
        ----------
        layers : list of dict
            Layer specifications.
        k0 : float
            Free-space wavenumber.
        kn_vectors : list of array-like
            In-plane wavevectors [(kxn, kyn), ...] for each plane wave.
        M : int
            Number of segments for z-integration.
        basis_order : int
            0 for piecewise constant, 1 for piecewise linear.
        """
        self.layers = layers
        self.k0 = k0
        self.kn_vectors = [np.array(k, dtype=complex) for k in kn_vectors]
        self.N = len(kn_vectors)
        self.M = M
        self.basis_order = basis_order

        # Total grating depth
        self.d = sum(l['thickness'] for l in layers if l['thickness'] > 0)

        # Setup mesh
        self._setup_mesh()

        # Precompute Green's functions for each plane-wave component
        self._setup_greens_functions()

    def _setup_mesh(self):
        """Create uniform mesh over grating depth [0, d]."""
        self.dz = self.d / self.M
        self.z_mesh = np.array([
            (j + 0.5) * self.dz for j in range(self.M)
        ])

    def _setup_greens_functions(self):
        """Precompute Green's function objects for each kn."""
        self.gf_tensors = []
        for kn in self.kn_vectors:
            gf = GreensFunctionTensor(self.layers, self.k0, kn)
            self.gf_tensors.append(gf)

    def compute_V_matrix(self, epsilon_grating, epsilon_host):
        """
        Compute perturbation matrix V = (1 - ε(r)) * k0² in coupled-wave basis.

        For a Toeplitz structure, the matrix-vector product can be done in
        O(N log N) via FFT.

        Parameters
        ----------
        epsilon_grating : complex
            Dielectric constant of the grating material.
        epsilon_host : complex
            Dielectric constant of the host medium.

        Returns
        -------
        V_diag : complex
            The perturbation strength (epsilon_grating - epsilon_host) * k0^2.
        """
        return (epsilon_host - epsilon_grating) * self.k0**2

    def _build_Gbar_matrix(self, n_idx, j, jp):
        """
        Build integrated Green's function matrix element Ḡ_{n}(j,j').

        Uses analytical integration over segments (Eq. 28).

        Parameters
        ----------
        n_idx : int
            Plane-wave index.
        j, jp : int
            Segment indices.

        Returns
        -------
        G_elem : ndarray, shape (3,3)
            Integrated Green's tensor element.
        """
        gf = self.gf_tensors[n_idx]
        z_j = self.z_mesh[j]
        z_jp = self.z_mesh[jp]
        layer_idx = gf._get_layer_index(z_j)
        qn = gf.get_qn(layer_idx)

        dz = self.dz
        Wj = _compute_Wj(qn, dz)

        if j == jp:
            # Diagonal: use W_j^0 for the exp(-Q|z-z'|) term
            Wj0 = _compute_Wj0(qn, dz)
            # The Green's tensor at z=z' with segment integration
            G_base = gf.compute_full_tensor(z_j, z_jp, layer_idx)
            # Scale by integration weight
            return G_base * Wj0
        else:
            # Off-diagonal: standard segment integration
            G_base = gf.compute_full_tensor(z_j, z_jp, layer_idx)
            Wj_off = abs(Wj)**2
            return G_base * Wj * _compute_Wj(qn, dz)

    def build_system_matrix(self, V_val):
        """
        Build the full system matrix Â = O - Ḡ*V for the LS equation (Eq. 25).

        For efficiency, returns a function that computes Â*X.

        Parameters
        ----------
        V_val : complex
            Perturbation strength.

        Returns
        -------
        matvec : callable
            Function that computes Â*X for given X.
        dim : int
            Dimension of the system (3*N*M).
        """
        N = self.N
        M = self.M
        dim = 3 * N * M

        def matvec(x):
            x = x.reshape(N, M, 3)
            result = np.copy(x)

            for n in range(N):
                gf = self.gf_tensors[n]
                for j in range(M):
                    # -G * V * X term
                    for jp in range(M):
                        G_elem = self._build_Gbar_matrix(n, j, jp)
                        result[n, j, :] -= G_elem @ (V_val * x[n, jp, :])

            return result.ravel()

        return matvec, dim

    def solve(self, psi0, V_val, tol=1e-6, maxiter=100):
        """
        Solve the LS equation: Â*X = X₀.

        Parameters
        ----------
        psi0 : ndarray, shape (N, M, 3)
            Incident wave function on mesh points.
        V_val : complex
            Perturbation strength.
        tol : float
            Convergence tolerance.
        maxiter : int
            Maximum iterations.

        Returns
        -------
        psi : ndarray, shape (N, M, 3)
            Scattered wave function.
        info : int
            0 if converged.
        """
        N = self.N
        M = self.M
        dim = 3 * N * M

        matvec, _ = self.build_system_matrix(V_val)
        A_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)

        # Right-hand side
        rhs = psi0.ravel().copy()

        # Solve using GMRES (similar to QMR, both are Krylov methods)
        x0 = psi0.ravel().copy()
        solution, info = gmres(A_op, rhs, x0=x0, rtol=tol, maxiter=maxiter)

        psi = solution.reshape(N, M, 3)
        return psi, info

    def solve_decoupled_TE(self, Ey0, V_val, tol=1e-6, maxiter=100):
        """
        Solve decoupled TE mode (Ky=0): Eq. (22).

        Ey(z) - Ey0(z) = ∫ G22(z,z') V Ey(z') dz'

        Parameters
        ----------
        Ey0 : ndarray, shape (N, M)
            Incident Ey field on mesh.
        V_val : complex
            Perturbation.

        Returns
        -------
        Ey : ndarray, shape (N, M)
            Scattered Ey field.
        info : int
            Convergence info.
        """
        N = self.N
        M = self.M
        dim = N * M

        def matvec(x):
            x = x.reshape(N, M)
            result = np.copy(x)
            for n in range(N):
                gf = self.gf_tensors[n]
                for j in range(M):
                    for jp in range(M):
                        z_j = self.z_mesh[j]
                        z_jp = self.z_mesh[jp]
                        layer_idx = gf._get_layer_index(z_j)
                        qn = gf.get_qn(layer_idx)

                        if j == jp:
                            Wj0 = _compute_Wj0(qn, self.dz)
                            gyy = gf.Gyy(z_j, z_jp, layer_idx)
                            result[n, j] -= gyy * Wj0 * V_val * x[n, jp]
                        else:
                            Wj = _compute_Wj(qn, self.dz)
                            gyy = gf.Gyy(z_j, z_jp, layer_idx)
                            result[n, j] -= gyy * Wj * _compute_Wj(qn, self.dz) * V_val * x[n, jp]
            return result.ravel()

        A_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
        rhs = Ey0.ravel().copy()
        x0 = Ey0.ravel().copy()
        solution, info = gmres(A_op, rhs, x0=x0, rtol=tol, maxiter=maxiter)
        return solution.reshape(N, M), info

    def solve_decoupled_TM(self, Hy0, V_val, epsilon_inv, Kx, tol=1e-6, maxiter=100):
        """
        Solve decoupled TM mode (Ky=0): Eq. (23).

        ERRATA CORRECTION #4: The equation is:
            Hy(z) - Hy0(z) = ∫ [∂z' Ḡ(z,z') Ṽ ∂z' Hy + Ḡ(z,z') Kx V ε⁻¹ Kx Hy] dz'

        NOT "Hy - Ez0(z)" as erroneously printed in the paper.

        Parameters
        ----------
        Hy0 : ndarray, shape (N, M)
            Incident Hy field on mesh.
        V_val : complex
            Perturbation for TM.
        epsilon_inv : complex
            Inverse dielectric constant for ε⁻¹ term.
        Kx : ndarray, shape (N,)
            x-component of in-plane wavevectors.

        Returns
        -------
        Hy : ndarray, shape (N, M)
            Scattered Hy field.
        info : int
            Convergence info.
        """
        N = self.N
        M = self.M
        dim = N * M

        def matvec(x):
            x = x.reshape(N, M)
            result = np.copy(x)
            for n in range(N):
                gf = self.gf_tensors[n]
                kn = abs(gf.kn)
                for j in range(M):
                    for jp in range(M):
                        z_j = self.z_mesh[j]
                        z_jp = self.z_mesh[jp]
                        layer_idx = gf._get_layer_index(z_j)
                        qn = gf.get_qn(layer_idx)
                        k2 = gf.get_k2(layer_idx)

                        if abs(qn) < 1e-30:
                            continue

                        Wj = _compute_Wj(qn, self.dz)
                        if j == jp:
                            Wj0 = _compute_Wj0(qn, self.dz)
                            weight = Wj0
                        else:
                            weight = Wj * _compute_Wj(qn, self.dz)

                        # Ḡ(z,z') for TM uses modified Green's function
                        gyy = gf.Gyy(z_j, z_jp, layer_idx)
                        if kn > 1e-30:
                            gxz = gf.Gxz(z_j, z_jp, layer_idx)
                            gbar = gyy - (kn / (1j * qn**2)) * gxz
                        else:
                            gbar = gyy

                        # Kx V ε⁻¹ Kx term
                        kx_term = Kx[n]**2 * V_val * epsilon_inv
                        result[n, j] -= weight * gbar * kx_term * x[n, jp]

            return result.ravel()

        A_op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
        rhs = Hy0.ravel().copy()
        x0 = Hy0.ravel().copy()
        solution, info = gmres(A_op, rhs, x0=x0, rtol=tol, maxiter=maxiter)
        return solution.reshape(N, M), info
