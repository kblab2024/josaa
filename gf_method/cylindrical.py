"""
Cylindrical grating support with Bessel/Struve function integrals.

Implements Section 4 of Chang et al. (2006): Application for
three-dimensional cylindrical objects.

Provides analytical integration over cylindrical disc segments
using Bessel and Struve functions.
"""

import numpy as np
from scipy.special import jv, struve


def compute_W(m, nu, kn, aj):
    """
    Compute the radial integral W_j(m, ν) = ∫₀^{a_j} (ρ/a_j)^m J_ν(k_n ρ) dρ.

    Uses the recursion relations from the paper:
    - W(ν+1, ν) = k⁻¹ J_{ν+1}(k*a_j)
    - W(ν, ν) uses Bessel-Struve formula
    - Recursion for m > ν: W(m+1, ν-1) = (2ν/(k*a_j)) W(m,ν) - W(m+1, ν+1)
    - Recursion for ν: W(m, ν+1) = W(m, ν-1) + 2[a_j J_ν(k*a_j) - m W(m-1,ν)]/(k*a_j)

    Parameters
    ----------
    m : int
        Radial power index.
    nu : int
        Bessel function order.
    kn : float
        In-plane wavenumber magnitude.
    aj : float
        Disc radius for segment j.

    Returns
    -------
    W_val : complex
        The radial integral value.
    """
    ka = kn * aj

    if abs(ka) < 1e-30:
        # Special case: kn ≈ 0
        if nu == 0:
            return aj / (m + 1) if m >= 0 else aj
        return 0.0

    # Base cases
    if m == nu + 1:
        return jv(nu + 1, ka) / kn

    if m == nu:
        # W(ν, ν) = 2^{ν-1} (k*a_j)^{-ν} a_j √π Γ(ν+1/2)
        #           × [J_ν(ka) H_{ν-1}(ka) - J_{ν-1}(ka) H_ν(ka)]
        from scipy.special import gamma
        if nu == 0:
            # W(0, 0) = ∫₀^{a_j} J_0(kρ) dρ
            # Use series or numerical integration for small arguments
            return _W_0_0(kn, aj)
        prefactor = (2.0**(nu - 1) * ka**(-nu) * aj *
                     np.sqrt(np.pi) * gamma(nu + 0.5))
        H_nu_minus1 = struve(nu - 1, ka)
        H_nu = struve(nu, ka)
        J_nu = jv(nu, ka)
        J_nu_minus1 = jv(nu - 1, ka)
        return prefactor * (J_nu * H_nu_minus1 - J_nu_minus1 * H_nu)

    # Use recursion for m > ν
    if m > nu + 1:
        return _W_recursion_m(m, nu, kn, aj)

    # Use recursion for general (m, ν) with m < ν
    if m < nu:
        return _W_recursion_nu(m, nu, kn, aj)

    return 0.0


def _W_0_0(kn, aj):
    """Compute W(0,0) = ∫₀^{a_j} J_0(kρ) dρ."""
    ka = kn * aj
    if abs(ka) < 0.1:
        # Taylor series for small ka
        result = aj
        term = aj
        for n in range(1, 20):
            term *= -(ka / (2 * n))**2 / (2 * n + 1)
            result += term
            if abs(term) < 1e-15 * abs(result):
                break
        return result
    # For larger ka: ∫₀^a J_0(kρ)dρ = a_j [J_0(ka)*H_1(ka) - J_1(ka)*H_0(ka)] * π/2
    # Or use: W(0,0) = sin(ka)/k (only for asymptotic)
    # Numerical integration fallback
    from scipy.integrate import quad
    result, _ = quad(lambda rho: jv(0, kn * rho), 0, aj)
    return result


def _W_recursion_m(m, nu, kn, aj):
    """
    Recursion for m > ν+1:
    W(m+1, ν-1) = (2ν/(k*a_j)) W(m, ν) - W(m+1, ν+1)
    """
    ka = kn * aj
    # Build up from base cases
    cache = {}

    def W_cached(mm, nn):
        if (mm, nn) in cache:
            return cache[(mm, nn)]
        if mm == nn + 1:
            val = jv(nn + 1, ka) / kn
        elif mm == nn:
            val = compute_W(mm, nn, kn, aj)
        elif mm > nn + 1:
            # W(mm, nn) from recursion:
            # W(mm, nn) = (2(nn+1)/(ka)) * W(mm-1, nn+1) - W(mm, nn+2)
            val = (2 * (nn + 1) / ka) * W_cached(mm - 1, nn + 1) - W_cached(mm, nn + 2)
        else:
            val = 0.0
        cache[(mm, nn)] = val
        return val

    return W_cached(m, nu)


def _W_recursion_nu(m, nu, kn, aj):
    """
    Recursion for ν:
    W(m, ν+1) = W(m, ν-1) + 2[a_j J_ν(k*a_j) - m*W(m-1, ν)] / (k*a_j)
    """
    ka = kn * aj
    cache = {}

    def W_cached(mm, nn):
        if (mm, nn) in cache:
            return cache[(mm, nn)]

        if nn < 0:
            # J_{-n}(x) = (-1)^n J_n(x), symmetry for W
            val = ((-1)**abs(nn)) * W_cached(mm, abs(nn))
        elif mm == nn + 1 and nn >= 0:
            val = jv(nn + 1, ka) / kn
        elif mm == nn and nn >= 0:
            if nn == 0:
                val = _W_0_0(kn, aj)
            else:
                val = compute_W(nn, nn, kn, aj)
        elif mm > nn + 1 and nn >= 0:
            # Use m-recursion: W(mm, nn) via W(mm-1, nn+1) and W(mm, nn+2)
            val = (2 * (nn + 1) / ka) * W_cached(mm - 1, nn + 1) - W_cached(mm, nn + 2)
        elif nn <= 1:
            # Base case: use numerical integration for small nu
            from scipy.integrate import quad
            if nn == 0:
                val = _W_0_0(kn, aj) if mm == 0 else 0.0
            elif nn == 1:
                if mm == 0:
                    result, _ = quad(lambda rho: jv(1, kn * rho), 0, aj)
                    val = result
                else:
                    result, _ = quad(lambda rho: (rho / aj)**mm * jv(1, kn * rho), 0, aj)
                    val = result
            else:
                val = 0.0
        else:
            # Build up from nu=0 and nu=1 using the recursion
            # W(m, nn) = W(m, nn-2) + 2[aj*J_{nn-1}(ka) - m*W(m-1,nn-1)] / ka
            w_nm2 = W_cached(mm, nn - 2)
            w_m1_nm1 = W_cached(mm - 1, nn - 1) if mm > 0 else 0.0
            val = w_nm2 + 2 * (aj * jv(nn - 1, ka) - mm * w_m1_nm1) / ka

        cache[(mm, nn)] = val
        return val

    return W_cached(m, nu)


def compute_segment_integral_cylindrical(qn, zj, delta_z, kn, phi_n, aj, m, nu):
    """
    Compute the cylindrical segment integral (Eq. 29):

    S⁻¹_m ∫∫∫ ρ^m dφ exp(i sin(φ+φ_n) k_n ρ) exp(iνφ) exp(±q_n z) dρ dz
    = 2√(πm) exp(iνφ_n) W_j(m,ν) exp(±q_n z_j) (1 - exp(-q_n Δz_j)) q_n⁻¹

    Parameters
    ----------
    qn : complex
        Vertical wave vector.
    zj : float
        Segment midpoint.
    delta_z : float
        Segment width.
    kn : float
        In-plane wavenumber magnitude.
    phi_n : float
        Azimuthal angle of k_n.
    aj : float
        Disc radius.
    m : int
        Radial power.
    nu : int
        Angular mode number.

    Returns
    -------
    integral_plus, integral_minus : complex
        Integrals for exp(+qn*z) and exp(-qn*z) terms.
    """
    W_val = compute_W(m, nu, kn, aj)
    phase = np.exp(1j * nu * phi_n)
    prefactor = 2 * np.sqrt(np.pi * max(m, 1)) * phase * W_val

    if abs(qn) < 1e-30:
        z_integral = delta_z
        integral_plus = prefactor * z_integral
        integral_minus = prefactor * z_integral
    else:
        z_factor = (1.0 - np.exp(-qn * delta_z)) / qn
        integral_plus = prefactor * np.exp(qn * zj) * z_factor
        integral_minus = prefactor * np.exp(-qn * zj) * z_factor

    return integral_plus, integral_minus


def compute_diagonal_weight_cylindrical(qn, delta_z, kn, phi_n, aj, m, nu, mp, mu):
    """
    Compute diagonal segment weight for j = j' (cylindrical case):

    exp(i(ν-μ)φ_n) ∫∫ (ρ/a)^m (ρ'/a)^{m'} J_μ(k_n ρ') J_ν(k_n ρ)
                     × exp(-q_n|z-z'|) dρ dρ' dz dz'
    = exp(i(ν-μ)φ_n) W_j(m',μ) W_j(m,ν) W_n^0(j)

    where W_n^0(j) = 2[Δs - (1-exp(-q_n Δs))/q_n] / q_n.

    Parameters
    ----------
    qn : complex
        Vertical wave vector.
    delta_z : float
        Segment width.
    kn : float
        In-plane wavenumber.
    phi_n : float
        Azimuthal angle.
    aj : float
        Disc radius.
    m, nu, mp, mu : int
        Radial and angular indices.

    Returns
    -------
    weight : complex
        Diagonal weight.
    """
    phase = np.exp(1j * (nu - mu) * phi_n)
    W_m_nu = compute_W(m, nu, kn, aj)
    W_mp_mu = compute_W(mp, mu, kn, aj)

    if abs(qn) < 1e-30:
        Wn0 = delta_z**2
    else:
        Wn0 = 2.0 * (delta_z - (1.0 - np.exp(-qn * delta_z)) / qn) / qn

    return phase * W_mp_mu * W_m_nu * Wn0
