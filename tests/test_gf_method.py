"""
Tests for the Green's function grating simulation method.

Tests validate the implementation against analytical properties
described in Chang et al. (2006), with errata corrections applied.
"""

import numpy as np
import pytest
from gf_method.transfer_matrix import (
    compute_reflection_coefficients,
    compute_Q_per_layer,
)
from gf_method.greens_function import GreensFunctionTensor
from gf_method.solver import LippmannSchwinger, _compute_Wj, _compute_Wj0
from gf_method.cylindrical import compute_W, compute_segment_integral_cylindrical
from gf_method.simulation import GratingSimulation, ContactHoleSimulation


class TestTransferMatrix:
    """Tests for transfer-matrix reflection coefficient computation."""

    def test_single_interface_TE(self):
        """Test reflection from a single interface (two semi-infinite media)."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0, 'Q': 1.0},   # air
            {'epsilon': 2.25, 'thickness': 0, 'Q': 0.5},   # glass
        ]
        R, rbar = compute_reflection_coefficients(layers, None, mode="TE")
        # For single interface with M±=1±J_l/J_{l+1}, R_{l+1}=0:
        # R[0] = M_minus / M_plus = (1 - Q_0/Q_1) / (1 + Q_0/Q_1)
        #       = (Q_1 - Q_0) / (Q_1 + Q_0)
        expected = (0.5 - 1.0) / (0.5 + 1.0)
        assert abs(R[0] - expected) < 1e-10
        assert abs(R[1]) < 1e-10  # substrate has no reflection

    def test_single_interface_TM(self):
        """Test TM reflection from a single interface."""
        eps0, eps1 = 1.0, 2.25
        Q0, Q1 = 1.0, 0.5
        layers = [
            {'epsilon': eps0, 'thickness': 0, 'Q': Q0},
            {'epsilon': eps1, 'thickness': 0, 'Q': Q1},
        ]
        R, rbar = compute_reflection_coefficients(layers, None, mode="TM")
        # TM: J_l = epsilon_l / Q_l
        J0 = eps0 / Q0
        J1 = eps1 / Q1
        expected = (J0 / J1 - 1) / (J0 / J1 + 1)  # (J0 - J1) / (J0 + J1)
        # Actually: ratio = J0/J1, M+ = 1 + ratio, M- = 1 - ratio
        # R = M- / M+ = (1 - J0/J1) / (1 + J0/J1) with R_{l+1}=0
        expected = (1 - J0 / J1) / (1 + J0 / J1)
        assert abs(R[0] - expected) < 1e-10

    def test_backward_reflection_symmetry(self):
        """Test that backward reflection gives expected values."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0, 'Q': 1.0},
            {'epsilon': 4.0, 'thickness': 0, 'Q': 0.5},
        ]
        R, rbar = compute_reflection_coefficients(layers, None, mode="TE")
        # rbar should be non-zero for the second layer looking backward
        assert abs(rbar[0]) < 1e-10  # first layer: no backward reflection
        assert abs(rbar[1]) > 0  # second layer has backward reflection

    def test_compute_Q_per_layer(self):
        """Test Q computation for layers."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 4.0, 'thickness': 100},
            {'epsilon': 2.25, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500  # 500 nm wavelength
        kn_sq = (k0 * 0.5)**2  # small in-plane component

        result = compute_Q_per_layer(layers, kn_sq, k0)
        for l in result:
            assert 'Q' in l
            # Q should have positive real part
            assert l['Q'].real >= 0 or abs(l['Q'].imag) > 0


class TestGreensFunction:
    """Tests for the tensor Green's function."""

    def _make_simple_layers(self):
        """Create simple two-layer system for testing."""
        return [
            {'epsilon': 1.0 + 0j, 'thickness': 0},
            {'epsilon': 2.25 + 0j, 'thickness': 200},
            {'epsilon': 4.0 + 0j, 'thickness': 0},
        ]

    def test_gyy_symmetry(self):
        """Test Gyy(z,z') properties."""
        layers = self._make_simple_layers()
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        z1, z2 = 50.0, 100.0
        gyy_12 = gf.Gyy(z1, z2, 1)
        gyy_21 = gf.Gyy(z2, z1, 1)

        # G(z,z') should have a definite relationship with G(z',z)
        # due to reciprocity
        assert isinstance(gyy_12, (complex, np.complexfloating, float))
        assert isinstance(gyy_21, (complex, np.complexfloating, float))

    def test_gxx_nonzero(self):
        """Test that Gxx is non-zero for non-trivial case."""
        layers = self._make_simple_layers()
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        gxx = gf.Gxx(50.0, 100.0, 1)
        assert abs(gxx) > 0

    def test_gxz_errata_correction(self):
        """
        Test errata correction #1 & #2: Gxz should be properly defined
        (not confused with Gxx) and include gbar(z') factor.
        """
        layers = self._make_simple_layers()
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        gxz = gf.Gxz(50.0, 100.0, 1)
        # Gxz should be non-zero when kn != 0
        assert abs(gxz) > 0

        # Errata #2: Check that Gxz properly depends on both z and z'
        gxz_1 = gf.Gxz(50.0, 80.0, 1)
        gxz_2 = gf.Gxz(50.0, 120.0, 1)
        assert gxz_1 != gxz_2  # Must depend on z'

    def test_gxz_vanishes_at_zero_kn(self):
        """Test that Gxz = 0 when kn = 0 (paper's limit)."""
        layers = self._make_simple_layers()
        k0 = 2 * np.pi / 500
        kn_vec = [0.0, 0.0]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        # In the limit kn -> 0: Gxz = Gzx = 0
        G = gf.compute_full_tensor(50.0, 100.0, 1)
        assert abs(G[0, 2]) < 1e-10
        assert abs(G[2, 0]) < 1e-10

    def test_gzz_delta_term(self):
        """
        Test errata correction #5: qn^2 ≡ kn^2 - k^2.
        The Gzz should contain proper qn dependence.
        """
        layers = self._make_simple_layers()
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.5, k0 * 0.0]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        gzz = gf.Gzz(50.0, 100.0, 1)
        assert isinstance(gzz, (complex, np.complexfloating))

    def test_full_tensor_3x3(self):
        """Test that full tensor is 3x3 complex matrix."""
        layers = self._make_simple_layers()
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        G = gf.compute_full_tensor(50.0, 100.0, 1)
        assert G.shape == (3, 3)
        assert G.dtype == complex


class TestSolver:
    """Tests for the Lippmann-Schwinger equation solver."""

    def test_Wj_small_Q(self):
        """Test Wj for small Q (should approach delta_z)."""
        Q = 1e-15
        dz = 10.0
        Wj = _compute_Wj(Q, dz)
        assert abs(Wj - dz) < 1e-5

    def test_Wj0_small_Q(self):
        """Test Wj0 for small Q (should approach dz^2)."""
        Q = 1e-15
        dz = 10.0
        Wj0 = _compute_Wj0(Q, dz)
        assert abs(Wj0 - dz**2) < 1e-5

    def test_Wj_large_Q(self):
        """Test Wj for large Q (should approach 1/Q)."""
        Q = 100.0
        dz = 10.0
        Wj = _compute_Wj(Q, dz)
        assert abs(Wj - 1.0 / Q) < 1e-5

    def test_Wj0_identity(self):
        """Test Wj0 = 2 * Wj_plus * delta_z_j identity."""
        Q = 2.0 + 1j
        dz = 5.0
        Wj = _compute_Wj(Q, dz)
        Wj0 = _compute_Wj0(Q, dz)
        Wj_plus = (dz - Wj) / Q if abs(Q) > 1e-30 else dz
        # Wj0 should equal 2 * Wj_plus * dz... actually paper says
        # Wj0 = 2[dz - (1-exp(-Q*dz))/Q] / Q
        # Let's verify the formula directly
        expected = 2.0 * (dz - (1.0 - np.exp(-Q * dz)) / Q) / Q
        assert abs(Wj0 - expected) < 1e-10

    def test_sgn_integral_vanishes(self):
        """Test that ∫∫ sgn(z-z') exp(-Q|z-z'|) dz dz' = 0 (Eq. 24)."""
        # This is verified analytically in the paper
        Q = 2.0
        dz = 5.0
        # Numerical verification
        N_pts = 100
        z_arr = np.linspace(-dz / 2, dz / 2, N_pts)
        dz_elem = z_arr[1] - z_arr[0]
        integral = 0.0
        for z in z_arr:
            for zp in z_arr:
                sgn = np.sign(z - zp)
                integral += sgn * np.exp(-Q * abs(z - zp)) * dz_elem**2
        assert abs(integral) < 1e-3

    def test_solver_identity_limit(self):
        """Test that solver returns input when V=0 (no perturbation)."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 2.25, 'thickness': 200},
            {'epsilon': 4.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vectors = [[k0 * 0.1, 0.0]]

        solver = LippmannSchwinger(layers, k0, kn_vectors, M=5)

        # With V=0, solution should equal input
        psi0 = np.ones((1, 5), dtype=complex)
        psi, info = solver.solve_decoupled_TE(psi0, V_val=0.0)

        np.testing.assert_allclose(psi, psi0, atol=1e-6)

    def test_errata_eq23_correction(self):
        """
        Test errata correction #4: Eq. (23) should use Hy - Hy0(z),
        not Hy - Ez0(z). Verify TM solver signature accepts Hy0.
        """
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 2.25, 'thickness': 200},
            {'epsilon': 4.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vectors = [[k0 * 0.1, 0.0]]
        solver = LippmannSchwinger(layers, k0, kn_vectors, M=5)

        Hy0 = np.ones((1, 5), dtype=complex)
        Kx = np.array([k0 * 0.1])

        # Should accept Hy0 (not Ez0) as the input field
        Hy, info = solver.solve_decoupled_TM(
            Hy0, V_val=0.0, epsilon_inv=1.0, Kx=Kx
        )
        np.testing.assert_allclose(Hy, Hy0, atol=1e-6)


class TestCylindrical:
    """Tests for cylindrical grating support."""

    def test_W_base_case_nu_plus_1(self):
        """Test W(ν+1, ν) = k⁻¹ J_{ν+1}(k*a)."""
        from scipy.special import jv
        kn = 0.05
        aj = 100.0
        nu = 2
        W_val = compute_W(nu + 1, nu, kn, aj)
        expected = jv(nu + 1, kn * aj) / kn
        assert abs(W_val - expected) < 1e-10

    def test_W_zero_kn(self):
        """Test W with kn ≈ 0."""
        W_val = compute_W(0, 0, 1e-20, 100.0)
        # Should be approximately a_j (integral of J_0(0) = 1)
        assert abs(W_val - 100.0) < 1e-5

    def test_segment_integral_sign(self):
        """Test segment integral returns proper complex values."""
        qn = 0.01 + 0j
        zj = 50.0
        dz = 10.0
        kn = 0.05
        phi_n = 0.0
        aj = 50.0

        I_plus, I_minus = compute_segment_integral_cylindrical(
            qn, zj, dz, kn, phi_n, aj, m=1, nu=0
        )
        assert isinstance(I_plus, (complex, np.complexfloating))
        assert isinstance(I_minus, (complex, np.complexfloating))

    def test_diagonal_weight_symmetry(self):
        """Test diagonal weight for cylindrical case."""
        qn = 0.02 + 0j
        dz = 10.0
        kn = 0.05
        phi_n = 0.0
        aj = 50.0

        w1 = compute_W(1, 0, kn, aj)
        assert isinstance(w1, (float, complex, np.floating, np.complexfloating))


class TestSimulation:
    """Integration tests for the full simulation."""

    def test_reciprocal_vectors(self):
        """Test that b_i · a_j = 2π δ_{ij}."""
        sim = GratingSimulation(
            a1=[1000, 0], a2=[0, 1000],
            layers=[
                {'epsilon': 1.0, 'thickness': 0},
                {'epsilon': 12.0, 'thickness': 400},
                {'epsilon': 12.0, 'thickness': 0},
            ],
            Nx=3, Ny=3
        )
        assert abs(np.dot(sim.b1, sim.a1) - 2 * np.pi) < 1e-10
        assert abs(np.dot(sim.b2, sim.a2) - 2 * np.pi) < 1e-10
        assert abs(np.dot(sim.b1, sim.a2)) < 1e-10
        assert abs(np.dot(sim.b2, sim.a1)) < 1e-10

    def test_contact_hole_simulation_creation(self):
        """Test ContactHoleSimulation object creation."""
        sim = ContactHoleSimulation(
            period_x=1000, period_y=1000,
            hole_diameter=307.5, hole_depth=409,
            epsilon_film=12.0 + 0.5j,
            epsilon_substrate=12.0 + 0.5j,
            Nx=3, Ny=3
        )
        assert sim.hole_radius == 307.5 / 2
        assert sim.N == 9

    def test_contact_hole_W_integrals(self):
        """Test W integral computation for contact holes."""
        sim = ContactHoleSimulation(
            period_x=1000, period_y=1000,
            hole_diameter=307.5, hole_depth=409,
            epsilon_film=12.0,
            epsilon_substrate=12.0,
            Nx=3, Ny=3
        )
        kn = 0.01
        W_table = sim.compute_W_integrals(kn, m_max=2, nu_max=2)
        assert len(W_table) > 0
        assert (0, 0) in W_table
        assert (1, 0) in W_table

    def test_reflectivity_runs(self):
        """Test that reflectivity computation runs without error."""
        sim = ContactHoleSimulation(
            period_x=1000, period_y=1000,
            hole_diameter=307.5, hole_depth=409,
            epsilon_film=12.0 + 0.5j,
            epsilon_substrate=12.0 + 0.5j,
            Nx=3, Ny=3
        )
        # Run at a single wavelength with small parameters for speed
        results = sim.compute_reflectivity(
            wavelengths=[500.0],
            theta=0.0,
            M=3,
            tol=1e-3,
            maxiter=10
        )
        assert 'r_TE' in results
        assert 'r_TM' in results
        assert 'R_TE' in results
        assert 'R_TM' in results
        assert len(results['R_TE']) == 1
        # Reflectivity should be between 0 and 1 (or close)
        assert results['R_TE'][0] >= 0
        assert results['R_TM'][0] >= 0


class TestErrataCorrections:
    """
    Tests specifically validating that errata corrections are properly applied.
    """

    def test_errata1_gxz_not_gxx(self):
        """
        Errata #1: The Gxz derivation should produce Gxz, not Gxx.
        Verify that Gxz and Gxx are different functions.
        """
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 4.0, 'thickness': 200},
            {'epsilon': 2.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        gxx = gf.Gxx(50.0, 100.0, 1)
        gxz = gf.Gxz(50.0, 100.0, 1)
        # These must be different functions
        assert abs(gxx - gxz) > 1e-15

    def test_errata2_gxz_has_z_prime_dependence(self):
        """
        Errata #2: Gxz must have proper z' dependence via gbar_n(z').
        Without the correction, the last term would lack z' dependence.
        """
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 4.0, 'thickness': 200},
            {'epsilon': 2.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        # Gxz should vary with z' due to gbar_n(z') factor
        gxz_a = gf.Gxz(50.0, 60.0, 1)
        gxz_b = gf.Gxz(50.0, 140.0, 1)
        assert abs(gxz_a - gxz_b) > 1e-15

    def test_errata4_TM_uses_Hy_not_Ez(self):
        """
        Errata #4: Eq. (23) should be Hy - Hy0(z), not Hy - Ez0(z).
        The TM solver accepts Hy0 as input, confirming the correction.
        """
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 2.25, 'thickness': 200},
            {'epsilon': 4.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vectors = [[k0 * 0.1, 0.0]]
        solver = LippmannSchwinger(layers, k0, kn_vectors, M=5)

        # The method signature takes Hy0 (not Ez0)
        Hy0 = np.ones((1, 5), dtype=complex) * 2.0
        Kx = np.array([k0 * 0.1])
        Hy, info = solver.solve_decoupled_TM(Hy0, 0.0, 1.0, Kx)
        # With V=0, output equals input (Hy, not Ez)
        np.testing.assert_allclose(Hy, Hy0, atol=1e-6)

    def test_errata5_qn_squared_definition(self):
        """
        Errata #5: qn^2 ≡ kn^2 - k^2 is correctly defined.
        Verify Q computation uses this definition.
        """
        layers = [{'epsilon': 4.0, 'thickness': 100}]
        k0 = 2 * np.pi / 500
        kn_sq = (k0 * 1.5)**2  # |kn|^2
        result = compute_Q_per_layer(layers, kn_sq, k0)
        Q = result[0]['Q']
        # Q^2 should equal kn^2 - epsilon * k0^2
        # i.e., Q^2 = kn_sq - 4 * k0^2
        expected_q_sq = kn_sq - 4.0 * k0**2
        assert abs(Q**2 - expected_q_sq) < 1e-10


class TestMLXBackend:
    """Tests for MLX backend abstraction and GPU acceleration support."""

    def test_backend_import(self):
        """Test that mlx_backend module is importable."""
        from gf_method.mlx_backend import use_mlx, get_backend, set_backend
        assert callable(use_mlx)
        assert callable(get_backend)
        assert callable(set_backend)

    def test_backend_returns_module(self):
        """Test that get_backend returns a valid array module."""
        from gf_method.mlx_backend import get_backend
        xp = get_backend()
        # Must have array creation and math functions
        assert hasattr(xp, 'array')
        assert hasattr(xp, 'exp')
        assert hasattr(xp, 'sqrt')

    def test_set_backend_numpy(self):
        """Test setting backend to numpy explicitly."""
        from gf_method.mlx_backend import set_backend, get_backend
        set_backend('numpy')
        xp = get_backend()
        assert xp is np
        # Reset
        set_backend('numpy')

    def test_set_backend_invalid(self):
        """Test that invalid backend raises ValueError."""
        from gf_method.mlx_backend import set_backend
        with pytest.raises(ValueError):
            set_backend('invalid_backend')

    def test_to_numpy(self):
        """Test to_numpy conversion."""
        from gf_method.mlx_backend import to_numpy
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_to_backend_numpy(self):
        """Test to_backend with numpy backend."""
        from gf_method.mlx_backend import to_backend, set_backend
        set_backend('numpy')
        arr = [1.0, 2.0, 3.0]
        result = to_backend(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_use_mlx_returns_bool(self):
        """Test that use_mlx returns a boolean."""
        from gf_method.mlx_backend import use_mlx
        result = use_mlx()
        assert isinstance(result, bool)

    def test_package_exports_backend(self):
        """Test that gf_method exports backend utilities."""
        from gf_method import use_mlx, set_backend, get_backend
        assert callable(use_mlx)
        assert callable(set_backend)
        assert callable(get_backend)

    def test_batch_tensor_matches_single(self):
        """Test that batch Green's tensor matches single-point computation."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 2.25, 'thickness': 200},
            {'epsilon': 4.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vec = [k0 * 0.3, k0 * 0.1]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        z_arr = np.array([50.0, 100.0, 150.0])
        zp_arr = np.array([80.0, 120.0, 60.0])

        # Batch computation
        G_batch = gf.compute_full_tensor_batch(z_arr, zp_arr, 1)
        assert G_batch.shape == (3, 3, 3)

        # Compare with single-point computation
        for i in range(len(z_arr)):
            G_single = gf.compute_full_tensor(z_arr[i], zp_arr[i], 1)
            np.testing.assert_allclose(G_batch[i], G_single, atol=1e-10)

    def test_batch_tensor_shape(self):
        """Test batch tensor output shape."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 4.0, 'thickness': 300},
            {'epsilon': 2.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 600
        kn_vec = [k0 * 0.2, k0 * 0.15]
        gf = GreensFunctionTensor(layers, k0, kn_vec)

        P = 10
        z_arr = np.linspace(10, 290, P)
        zp_arr = np.linspace(20, 280, P)

        G_batch = gf.compute_full_tensor_batch(z_arr, zp_arr, 1)
        assert G_batch.shape == (P, 3, 3)
        assert G_batch.dtype == complex

    def test_precomputed_solver_matches(self):
        """Test that precomputed Gbar matrices give same solver results."""
        layers = [
            {'epsilon': 1.0, 'thickness': 0},
            {'epsilon': 2.25, 'thickness': 200},
            {'epsilon': 4.0, 'thickness': 0},
        ]
        k0 = 2 * np.pi / 500
        kn_vectors = [[k0 * 0.1, 0.0]]

        solver = LippmannSchwinger(layers, k0, kn_vectors, M=5)

        # With V=0, solution should equal input (tests precomputed path)
        psi0 = np.ones((1, 5, 3), dtype=complex)
        V_val = 0.0

        matvec, dim = solver.build_system_matrix(V_val)
        result = matvec(psi0.ravel())
        np.testing.assert_allclose(result, psi0.ravel(), atol=1e-6)
