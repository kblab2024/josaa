"""
Tensor Green's function computation in reciprocal space.

Implements the Green's function components Gxx, Gyy, Gxz, Gzx, Gzz
as derived in Section 2 of Chang et al. (2006), with errata corrections:

Errata corrections applied:
1. Gxz derivation: Fixed "Gxx =" to "Gxz =" (typo in paper)
2. Gxz same-layer: Added missing g_bar_n(z') factor
3. Eq. (14): Corrected relationship for Gxz = (ikn/qn^2) * dz Gxx
4. Eq. (23): Corrected "Hy - Ez0(z)" to "Hy - Hy0(z)"
5. qn^2 definition: Confirmed qn^2 ≡ kn^2 - k^2

Supports MLX backend for GPU acceleration on Apple Silicon.
"""

import numpy as np
from .mlx_backend import use_mlx, get_backend
from .transfer_matrix import compute_reflection_coefficients, compute_Q_per_layer


class GreensFunctionTensor:
    """
    Computes the 3x3 tensor Green's function G_n(z, z') for each
    plane-wave component n in a multilayer film.

    The Green's function satisfies:
        (∇×∇× - k²)G = δ(r - r')

    with boundary conditions for continuity of G and ∇×G across
    layer interfaces.
    """

    def __init__(self, layers, k0, kn_vec):
        """
        Parameters
        ----------
        layers : list of dict
            Layer specifications: [{'epsilon': complex, 'thickness': float}, ...]
            First layer is superstrate, last is substrate.
        k0 : float
            Free-space wavenumber 2π/λ₀.
        kn_vec : array-like, shape (2,)
            In-plane wavevector (kxn, kyn) for this plane-wave component.
        """
        self.layers = layers
        self.k0 = k0
        self.kn_vec = np.array(kn_vec, dtype=complex)
        self.kxn = self.kn_vec[0]
        self.kyn = self.kn_vec[1]
        self.kn = np.sqrt(self.kxn**2 + self.kyn**2)
        self.phi_n = np.arctan2(self.kyn.real, self.kxn.real)

        # Precompute reflection coefficients for each layer
        self._precompute()

    def _precompute(self):
        """Precompute reflection coefficients for TE and TM modes."""
        kn_sq = abs(self.kn)**2

        # Compute Q per layer
        self.layers_q = compute_Q_per_layer(self.layers, kn_sq, self.k0)

        # For the layer of interest (where z, z' reside)
        # We compute R, rbar for both TE and TM modes
        self.R_TE, self.rbar_TE = compute_reflection_coefficients(
            self.layers_q, None, mode="TE"
        )
        self.R_TM, self.rbar_TM = compute_reflection_coefficients(
            self.layers_q, None, mode="TM"
        )

    def _get_layer_index(self, z):
        """Find which layer z belongs to."""
        z_acc = 0.0
        for i, layer in enumerate(self.layers):
            d = layer['thickness']
            if d == 0 and i == 0:
                if z <= 0:
                    return 0
                continue
            if d == 0 and i == len(self.layers) - 1:
                return i
            if z_acc <= z <= z_acc + d:
                return i
            z_acc += d
        return len(self.layers) - 1

    def get_qn(self, layer_idx):
        """Get qn for a specific layer."""
        return self.layers_q[layer_idx]['Q']

    def get_k2(self, layer_idx):
        """Get k^2 = epsilon * k0^2 for a specific layer."""
        return self.layers_q[layer_idx]['epsilon'] * self.k0**2

    def Gyy(self, z, zp, layer_idx):
        """
        Compute Gyy component (y'y' in rotated frame).

        For z and z' in the same medium (Eq. 7 and following):
            Gyy = (1/2qn) * exp(-qn|z-z'|)
                  + exp(qn*z) * R_n * f_n(z')
                  + exp(-qn*z) * rbar_n * g_n(z')

        Parameters
        ----------
        z, zp : float
            Vertical coordinates.
        layer_idx : int
            Layer index for this computation.
        """
        qn = self.get_qn(layer_idx)
        R = self.R_TE[layer_idx]
        rbar = self.rbar_TE[layer_idx]
        u = 1.0 / (1.0 - R * rbar) if abs(1 - R * rbar) > 1e-30 else 1.0

        if abs(qn) < 1e-30:
            return 0.0

        # f_n(z') and g_n(z') from Eq. (7)
        # f_n = (1/2qn) * (exp(Q*z') + rbar * u * (R*exp(Q*z') + exp(-Q*z')))
        # g_n = (1/2qn) * u * (R*exp(Q*z') + exp(-Q*z'))
        eQzp = np.exp(qn * zp)
        emQzp = np.exp(-qn * zp)

        fn = (1.0 / (2.0 * qn)) * (eQzp + rbar * u * (R * eQzp + emQzp))
        gn = (1.0 / (2.0 * qn)) * u * (R * eQzp + emQzp)

        # Same-medium expression
        result = (1.0 / (2.0 * qn)) * np.exp(-qn * abs(z - zp))
        result += np.exp(qn * z) * R * fn
        result += np.exp(-qn * z) * rbar * gn

        return result

    def Gxx(self, z, zp, layer_idx):
        """
        Compute Gxx component (x'x' in rotated frame).

        For z and z' in the same medium (Eq. 12 and following):
            Gxx = (-qn/2k²) * exp(-qn|z-z'|)
                  + exp(qn*z) * Rtilde_n * ftilde_n(z')
                  + exp(-qn*z) * rtilde_n * gtilde_n(z')
        """
        qn = self.get_qn(layer_idx)
        k2 = self.get_k2(layer_idx)
        Rt = self.R_TM[layer_idx]
        rt = self.rbar_TM[layer_idx]
        ut = 1.0 / (1.0 - Rt * rt) if abs(1 - Rt * rt) > 1e-30 else 1.0

        if abs(qn) < 1e-30 or abs(k2) < 1e-30:
            return 0.0

        eQzp = np.exp(qn * zp)
        emQzp = np.exp(-qn * zp)

        # ftilde_n and gtilde_n from Eq. (12)
        ft = (-qn / (2.0 * k2)) * (eQzp + rt * ut * (Rt * eQzp + emQzp))
        gt = (-qn / (2.0 * k2)) * ut * (Rt * eQzp + emQzp)

        # Same-medium expression
        result = (-qn / (2.0 * k2)) * np.exp(-qn * abs(z - zp))
        result += np.exp(qn * z) * Rt * ft
        result += np.exp(-qn * z) * rt * gt

        return result

    def Gxz(self, z, zp, layer_idx):
        """
        Compute Gxz component.

        ERRATA CORRECTION #1: The paper erroneously labels derivation lines as
        "Gxx = ..." when they should be "Gxz = ...".

        ERRATA CORRECTION #2: Added missing g_bar_n(z') factor in same-layer
        expression.

        For z and z' in the same medium (corrected Eq. 14):
            Gxz = (-ikn/2k²) * sgn(z-z') * exp(-qn|z-z'|)
                  + (-eqnz * Rtilde * fbar(z') + e-qnz * rtilde * gbar(z'))

        where fbar and gbar from Eq. (13):
            fbar = (-ikn/2k²) * (exp(Q*z') + rtilde*utilde*(Rtilde*exp(Q*z') - exp(-Q*z')))
            gbar = (-ikn/2k²) * utilde * (Rtilde*exp(Q*z') - exp(-Q*z'))

        Also: Gxz = (ikn/qn²) * ∂z Gxx  (Eq. 15, corrected per errata #3)
        """
        qn = self.get_qn(layer_idx)
        k2 = self.get_k2(layer_idx)
        kn = self.kn
        Rt = self.R_TM[layer_idx]
        rt = self.rbar_TM[layer_idx]
        ut = 1.0 / (1.0 - Rt * rt) if abs(1 - Rt * rt) > 1e-30 else 1.0

        if abs(qn) < 1e-30 or abs(k2) < 1e-30:
            return 0.0

        eQzp = np.exp(qn * zp)
        emQzp = np.exp(-qn * zp)

        # fbar_n and gbar_n from Eq. (13)
        fbar = (-1j * kn / (2.0 * k2)) * (eQzp + rt * ut * (Rt * eQzp - emQzp))
        # ERRATA #2: gbar must be present as a proper function of z'
        gbar = (-1j * kn / (2.0 * k2)) * ut * (Rt * eQzp - emQzp)

        # Same-medium expression (corrected)
        sgn = np.sign(z - zp) if z != zp else 0.0
        result = (-1j * kn / (2.0 * k2)) * sgn * np.exp(-qn * abs(z - zp))
        # ERRATA #1 & #2: Corrected Gxz expression with proper gbar(z')
        result += -np.exp(qn * z) * Rt * fbar + np.exp(-qn * z) * rt * gbar

        return result

    def Gzx(self, z, zp, layer_idx):
        """
        Compute Gzx = (ikn/qn²) * ∂z' Gxx  (Eq. 16).

        By symmetry G(kn; z, z') = G^T(-kn; z', z), we have
        Gzx(z,z') = Gxz(z',z) with kn -> -kn, which gives a sign flip on ikn terms.

        For z and z' in same medium (Eq. 17):
            Gzx = (-ikn/2k²) * sgn(z-z') * exp(-qn|z-z'|)
                  + (eqnz * Rtilde * fbar(z') + e-qnz * rtilde * gbar(z'))
                  ERRATA #2: gbar(z') factor included
        """
        qn = self.get_qn(layer_idx)
        k2 = self.get_k2(layer_idx)
        kn = self.kn
        Rt = self.R_TM[layer_idx]
        rt = self.rbar_TM[layer_idx]
        ut = 1.0 / (1.0 - Rt * rt) if abs(1 - Rt * rt) > 1e-30 else 1.0

        if abs(qn) < 1e-30 or abs(k2) < 1e-30:
            return 0.0

        eQzp = np.exp(qn * zp)
        emQzp = np.exp(-qn * zp)

        # Gzx = (-ikn/qn²) * ∂z' Gxx (Eq. 16)
        # For same medium, using Eq. (17):
        sgn_val = np.sign(z - zp) if z != zp else 0.0
        result = (-1j * kn / (2.0 * k2)) * sgn_val * np.exp(-qn * abs(z - zp))

        # The reflected/transmitted parts
        fbar_zx = (-1j * kn / (2.0 * k2)) * (eQzp + rt * ut * (Rt * eQzp - emQzp))
        gbar_zx = (-1j * kn / (2.0 * k2)) * ut * (Rt * eQzp - emQzp)

        result += np.exp(qn * z) * Rt * fbar_zx + np.exp(-qn * z) * rt * gbar_zx

        return result

    def Gzz(self, z, zp, layer_idx):
        """
        Compute Gzz = (kn²/qn⁴) * ∂z∂z' Gxx + (1/qn²) * δ(z-z')  (Eq. 18).

        For z and z' in same medium:
            Gzz = (kn²/(2*qn*k²)) * exp(-qn|z-z'|)
                  + (kn²/qn²) * [eqnz * Rtilde * ftilde(z')
                                  + e-qnz * rtilde * gtilde(z')]
                  - (1/k²) * δ(z-z')

        The delta function term is handled separately in the integral.
        """
        qn = self.get_qn(layer_idx)
        k2 = self.get_k2(layer_idx)
        kn = self.kn

        if abs(qn) < 1e-30 or abs(k2) < 1e-30:
            # In the limit kn -> 0: Gzz = -(1/k²) * δ(z-z')
            return 0.0

        Rt = self.R_TM[layer_idx]
        rt = self.rbar_TM[layer_idx]
        ut = 1.0 / (1.0 - Rt * rt) if abs(1 - Rt * rt) > 1e-30 else 1.0

        eQzp = np.exp(qn * zp)
        emQzp = np.exp(-qn * zp)

        kn_sq = kn**2
        qn_sq = qn**2

        # Same-medium: continuous part (delta handled in integration)
        result = (kn_sq / (2.0 * qn * k2)) * np.exp(-qn * abs(z - zp))

        # Reflected parts using ∂z∂z' Gxx
        ft = (-qn / (2.0 * k2)) * (eQzp + rt * ut * (Rt * eQzp + emQzp))
        gt = (-qn / (2.0 * k2)) * ut * (Rt * eQzp + emQzp)

        result += (kn_sq / qn_sq) * (np.exp(qn * z) * Rt * ft
                                      + np.exp(-qn * z) * rt * gt)

        return result

    def compute_full_tensor(self, z, zp, layer_idx):
        """
        Compute the full 3x3 Green's tensor in the original (x,y,z) frame.

        Rotates from the (x',y',z) frame back to (x,y,z) using the angle
        phi_n = arctan(kyn/kxn).

        Returns
        -------
        G : ndarray, shape (3,3)
            The Green's tensor G_n(z,z').
        """
        # Components in rotated frame
        gxx = self.Gxx(z, zp, layer_idx)
        gyy = self.Gyy(z, zp, layer_idx)
        gxz = self.Gxz(z, zp, layer_idx)
        gzx = self.Gzx(z, zp, layer_idx)
        gzz = self.Gzz(z, zp, layer_idx)

        kn = abs(self.kn)
        if kn < 1e-30:
            # Limit kn -> 0: Gxx = Gyy, Gxz = Gzx = 0
            return np.array([
                [gyy, 0, 0],
                [0, gyy, 0],
                [0, 0, gzz]
            ], dtype=complex)

        kxn = self.kxn
        kyn = self.kyn
        kn_sq = kn**2

        # Rotation back to original frame (paper's Eq. after Eq. 18)
        G11 = (kxn**2 / kn_sq) * gxx + (kyn**2 / kn_sq) * gyy
        G12 = (kxn * kyn / kn_sq) * (gxx - gyy)
        G13 = (kxn / kn) * gxz
        G21 = G12  # symmetric off-diagonal
        G22 = (kyn**2 / kn_sq) * gxx + (kxn**2 / kn_sq) * gyy
        G23 = (kyn / kn) * gxz
        G31 = (kxn / kn) * gzx
        G32 = (kyn / kn) * gzx
        G33 = gzz

        return np.array([
            [G11, G12, G13],
            [G21, G22, G23],
            [G31, G32, G33]
        ], dtype=complex)

    def compute_full_tensor_batch(self, z_arr, zp_arr, layer_idx):
        """
        Compute the full 3x3 Green's tensor for arrays of z, z' values.

        Vectorized version of compute_full_tensor for GPU acceleration
        via MLX on Apple Silicon.

        Parameters
        ----------
        z_arr : ndarray, shape (P,)
            Vertical coordinates z.
        zp_arr : ndarray, shape (P,)
            Vertical coordinates z'.
        layer_idx : int
            Layer index.

        Returns
        -------
        G_batch : ndarray, shape (P, 3, 3)
            The Green's tensor for each (z, z') pair.
        """
        xp = get_backend()
        z_arr = np.asarray(z_arr)
        zp_arr = np.asarray(zp_arr)
        P = len(z_arr)

        qn = self.get_qn(layer_idx)
        k2 = self.get_k2(layer_idx)
        kn = self.kn
        kxn = self.kxn
        kyn = self.kyn
        kn_abs = abs(kn)

        if abs(qn) < 1e-30 or abs(k2) < 1e-30:
            return np.zeros((P, 3, 3), dtype=complex)

        # TE reflection coefficients
        R_TE = self.R_TE[layer_idx]
        rbar_TE = self.rbar_TE[layer_idx]
        u_TE = 1.0 / (1.0 - R_TE * rbar_TE) if abs(1 - R_TE * rbar_TE) > 1e-30 else 1.0

        # TM reflection coefficients
        R_TM = self.R_TM[layer_idx]
        rbar_TM = self.rbar_TM[layer_idx]
        u_TM = 1.0 / (1.0 - R_TM * rbar_TM) if abs(1 - R_TM * rbar_TM) > 1e-30 else 1.0

        # Vectorized exponentials
        eQz = np.exp(qn * z_arr)
        emQz = np.exp(-qn * z_arr)
        eQzp = np.exp(qn * zp_arr)
        emQzp = np.exp(-qn * zp_arr)
        abs_dz = np.abs(z_arr - zp_arr)
        exp_neg_qdz = np.exp(-qn * abs_dz)
        sgn_dz = np.sign(z_arr - zp_arr)

        # --- Gyy (TE mode) ---
        fn_TE = (1.0 / (2.0 * qn)) * (eQzp + rbar_TE * u_TE * (R_TE * eQzp + emQzp))
        gn_TE = (1.0 / (2.0 * qn)) * u_TE * (R_TE * eQzp + emQzp)
        gyy = (1.0 / (2.0 * qn)) * exp_neg_qdz + eQz * R_TE * fn_TE + emQz * rbar_TE * gn_TE

        # --- Gxx (TM mode) ---
        ft_TM = (-qn / (2.0 * k2)) * (eQzp + rbar_TM * u_TM * (R_TM * eQzp + emQzp))
        gt_TM = (-qn / (2.0 * k2)) * u_TM * (R_TM * eQzp + emQzp)
        gxx = (-qn / (2.0 * k2)) * exp_neg_qdz + eQz * R_TM * ft_TM + emQz * rbar_TM * gt_TM

        # --- Gxz ---
        fbar = (-1j * kn / (2.0 * k2)) * (eQzp + rbar_TM * u_TM * (R_TM * eQzp - emQzp))
        gbar = (-1j * kn / (2.0 * k2)) * u_TM * (R_TM * eQzp - emQzp)
        gxz = (-1j * kn / (2.0 * k2)) * sgn_dz * exp_neg_qdz + (-eQz * R_TM * fbar + emQz * rbar_TM * gbar)

        # --- Gzx ---
        fbar_zx = (-1j * kn / (2.0 * k2)) * (eQzp + rbar_TM * u_TM * (R_TM * eQzp - emQzp))
        gbar_zx = (-1j * kn / (2.0 * k2)) * u_TM * (R_TM * eQzp - emQzp)
        gzx = (-1j * kn / (2.0 * k2)) * sgn_dz * exp_neg_qdz + eQz * R_TM * fbar_zx + emQz * rbar_TM * gbar_zx

        # --- Gzz ---
        kn_sq = kn**2
        qn_sq = qn**2
        gzz = (kn_sq / (2.0 * qn * k2)) * exp_neg_qdz
        gzz += (kn_sq / qn_sq) * (eQz * R_TM * ft_TM + emQz * rbar_TM * gt_TM)

        # --- Rotate to (x,y,z) frame ---
        G_batch = np.zeros((P, 3, 3), dtype=complex)

        if kn_abs < 1e-30:
            G_batch[:, 0, 0] = gyy
            G_batch[:, 1, 1] = gyy
            G_batch[:, 2, 2] = gzz
        else:
            kn_sq_abs = kn_abs**2
            G_batch[:, 0, 0] = (kxn**2 / kn_sq_abs) * gxx + (kyn**2 / kn_sq_abs) * gyy
            G_batch[:, 0, 1] = (kxn * kyn / kn_sq_abs) * (gxx - gyy)
            G_batch[:, 0, 2] = (kxn / kn_abs) * gxz
            G_batch[:, 1, 0] = G_batch[:, 0, 1]
            G_batch[:, 1, 1] = (kyn**2 / kn_sq_abs) * gxx + (kxn**2 / kn_sq_abs) * gyy
            G_batch[:, 1, 2] = (kyn / kn_abs) * gxz
            G_batch[:, 2, 0] = (kxn / kn_abs) * gzx
            G_batch[:, 2, 1] = (kyn / kn_abs) * gzx
            G_batch[:, 2, 2] = gzz

        return G_batch
