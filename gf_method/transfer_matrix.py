"""
Transfer-matrix method for computing forward and backward reflection
coefficients in multilayer films.

Implements Eqs. (8)-(11) of Chang et al. (2006).
"""

import numpy as np


def compute_reflection_coefficients(layers, qn, mode="TE"):
    """
    Compute forward (R) and backward (r_bar) reflection coefficients
    for all layers using the transfer-matrix recursion.

    Parameters
    ----------
    layers : list of dict
        Each dict has keys:
            'epsilon': complex dielectric constant
            'thickness': float, layer thickness (0 for substrate/superstrate)
    qn : complex
        Vertical wave vector component q_n = sqrt(k_n^2 - k^2)
    mode : str
        "TE" for Ey boundary conditions (J_l = Q_l)
        "TM" for Ex boundary conditions (J_l = epsilon_l * Q_l^{-1})

    Returns
    -------
    R_list : list of complex
        Forward reflection coefficients per layer
    rbar_list : list of complex
        Backward reflection coefficients per layer
    """
    num_layers = len(layers)

    # Compute Q_l and J_l for each layer
    # q_n^2 = k_n^2 - k^2, but Q_l depends on layer's dielectric constant
    # k^2 = epsilon * k0^2, so Q_l = sqrt(k_n^2 - epsilon_l * k0^2)
    # However, the paper uses Q = qn within each layer.
    # qn passed here is the in-plane-dependent quantity; Q_l = sqrt(qn_base^2 + delta)
    # Actually: q_l = sqrt(k_n^2 - k_l^2) where k_l^2 = epsilon_l * k0^2

    Q = []
    J = []
    for layer in layers:
        eps_l = layer['epsilon']
        # k_l^2 = eps_l * k0^2; k_n^2 = |k_n|^2
        # We receive qn as the base vertical wavevector for the reference medium
        # For each layer l, Q_l = sqrt(k_n^2 - eps_l * k0^2)
        # But we parametrize via the quantities already computed
        Q_l = layer.get('Q', qn)  # Q for this layer
        Q.append(Q_l)
        if mode == "TE":
            J.append(Q_l)
        else:  # TM
            J.append(eps_l * (1.0 / Q_l) if Q_l != 0 else 0.0)

    # Forward recursion for R: Eq. (8)-(9)
    # R_l = exp(-Q_l * d_l) * (M_l^- + M_l^+ * R_{l+1}) * T_l * exp(-Q_l * d_l)
    # T_l = (M_l^+ + M_l^- * R_{l+1})^{-1}
    # M_l^± = 1 ± J_{l-1} * J_{l+1}  (but indices need care -- see note below)
    # Actually from paper: M_l^± = 1 ± J_l / J_{l+1}  (ratio)

    R = [0.0] * num_layers
    # Start from the last layer (substrate): R_{num_layers-1} = 0
    R[-1] = 0.0

    for l in range(num_layers - 2, -1, -1):
        d_l = layers[l]['thickness']
        Q_l = Q[l]

        # M_l^± ≡ 1 ± J_l^{-1} * J_{l+1}
        # From paper context: J_l for TE is Q_l, so ratio is Q_l / Q_{l+1}
        if J[l + 1] != 0:
            ratio = J[l] / J[l + 1]
        else:
            ratio = 0.0

        M_plus = 1.0 + ratio
        M_minus = 1.0 - ratio

        # T_l = (M_l^+ + M_l^- * R_{l+1})^{-1}
        denom = M_plus + M_minus * R[l + 1]
        if abs(denom) > 1e-30:
            T_l = 1.0 / denom
        else:
            T_l = 0.0

        # R_l = exp(-Q_l * d_l) * (M_minus + M_plus * R_{l+1}) * T_l * exp(-Q_l * d_l)
        exp_factor = np.exp(-Q_l * d_l) if d_l > 0 else 1.0
        R[l] = exp_factor * (M_minus + M_plus * R[l + 1]) * T_l * exp_factor

    # Backward recursion for r_bar: Eq. (10)-(11)
    # r_bar_{l+1} = (m^- + m^+ * exp(-Q_l*d_l) * r_bar_l * exp(-Q_l*d_l)) * t_l
    # t_l = (m^+ + m^- * exp(-Q_l*d_l) * r_bar_l * exp(-Q_l*d_l))^{-1}
    # m_l^± = 1 ± J_{l+1}^{-1} * J_l
    rbar = [0.0] * num_layers
    rbar[0] = 0.0

    for l in range(0, num_layers - 1):
        d_l = layers[l]['thickness']
        Q_l = Q[l]

        if J[l] != 0:
            ratio_back = J[l + 1] / J[l]
        else:
            ratio_back = 0.0

        m_plus = 1.0 + ratio_back
        m_minus = 1.0 - ratio_back

        exp_factor = np.exp(-Q_l * d_l) if d_l > 0 else 1.0
        exp_rbar = exp_factor * rbar[l] * exp_factor

        denom = m_plus + m_minus * exp_rbar
        if abs(denom) > 1e-30:
            t_l = 1.0 / denom
        else:
            t_l = 0.0

        rbar[l + 1] = (m_minus + m_plus * exp_rbar) * t_l

    return R, rbar


def compute_Q_per_layer(layers, kn_sq, k0):
    """
    Compute Q_l = sqrt(k_n^2 - epsilon_l * k0^2) for each layer.

    Parameters
    ----------
    layers : list of dict
        Layer specifications with 'epsilon' key.
    kn_sq : float
        |k_n|^2, magnitude squared of in-plane wavevector.
    k0 : float
        Free-space wavenumber 2*pi/lambda.

    Returns
    -------
    layers_with_Q : list of dict
        Layers with 'Q' field added.
    """
    result = []
    for layer in layers:
        layer_copy = dict(layer)
        eps = layer['epsilon']
        q_sq = kn_sq - eps * k0**2
        # Choose branch with Re(Q) > 0 for evanescent decay
        Q_val = np.sqrt(q_sq + 0j)
        if Q_val.real < 0:
            Q_val = -Q_val
        layer_copy['Q'] = Q_val
        result.append(layer_copy)
    return result
