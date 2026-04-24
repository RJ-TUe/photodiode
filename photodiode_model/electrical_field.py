import numpy as np
import constants
import materials
from device import PINJunction

def electric_field(z: np.ndarray, device: PINJunction, V_bias: float = 0.0, model="depletion") -> np.ndarray:
    """
    Compute electric field profile across P-i-N junction.

    Parameters
    ----------
    z : ndarray
        Spatial grid [m], from 0 to device.d.
    device : PINJunction
        Device geometry and doping.
    V_bias : float
        Applied bias [V]. 0 = equilibrium, negative = reverse bias.
    model : str
        "depletion" — analytical depletion approximation.

    Returns
    -------
    ndarray
        Electric field E(z) [V/m]. Negative = points toward n-region.
    """
    if model == "depletion":
        eps = materials.eps_r * constants.eps_0
        kT  = constants.kb_j * 300

        # built-in voltage
        V_bi = (kT / constants.q) * np.log(device.N_A * device.N_D / materials.n_i**2)
        V_total = V_bi - V_bias   # reverse bias increases total voltage

        # E_max from voltage balance across all three regions:
        # V_total = (eps * E_max^2) / (2q) * (1/N_A + 1/N_D) + |E_max| * d_i
        # rearranges to quadratic in E_max:
        # a*E_max^2 + b*E_max + c = 0
        a = (eps / (2 * constants.q)) * (1/device.N_A + 1/device.N_D)
        b = device.d_i
        c = -V_total

        E_max_magnitude = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)  # positive, [V/m]
        E_max = -E_max_magnitude   
        # depletion widths from E_max
        x_p = -(eps * E_max) / (constants.q * device.N_A)
        x_n = -(eps * E_max) / (constants.q * device.N_D)   # E_max negative so this is positive

        x_p = min(x_p, device.d_p)
        x_n = min(x_n, device.d_n)

        # depletion boundaries
        z_dep_p = device.z_p - x_p
        z_dep_n = device.z_i + x_n

        E = np.zeros_like(z, dtype=float)

        for idx, zi in enumerate(z):
            if zi < z_dep_p:
                E[idx] = 0.0
            elif zi < device.z_p:
                E[idx] = -(constants.q * device.N_A / eps) * (zi - z_dep_p)
            elif zi < device.z_i:
                E[idx] = E_max
            elif zi < z_dep_n:
                E[idx] = E_max + (constants.q * device.N_D / eps) * (zi - device.z_i)
            else:
                E[idx] = 0.0

        print(f"V_bi   = {V_bi:.4f} V")
        print(f"E_max  = {E_max:.2f} V/m")
        print(f"x_p    = {x_p*1e9:.2f} nm  (d_p = {device.d_p*1e9:.2f} nm)")
        print(f"x_n    = {x_n*1e9:.2f} nm  (d_n = {device.d_n*1e9:.2f} nm)")
        print(f"z_dep_p = {z_dep_p*1e9:.2f} nm")
        print(f"z_dep_n = {z_dep_n*1e9:.2f} nm")

        return E

    else:
        raise ValueError(f"Unknown electric field model: {model}")