import numpy as np
import constants
import materials
from device import PINJunction, PNJunction
from scipy.integrate import cumulative_trapezoid

def electric_field(device, V_bias: float = 0.0, model="depletion") -> np.ndarray:
    """
    Compute electric field profile across P-i-N junction.

    Parameters
    ----------
    device : PINJunction
        Device geometry and doping.
    V_bias : float
        Applied bias [V]. 0 = equilibrium, negative = reverse bias.
    model : str
        "depletion" — analytical depletion approximation.

    Returns
    -------
    ndarray
        Electric field E(z) [V/m]
        Charge density Rho(z)
        Voltage V(z)
    """
    if model == 'depletion':
        eps = materials.eps_r * constants.eps_0
        kT  = constants.kb_j * 300
        z = device.mesh

        T = 300 # Temperature (room for now)
        V_t = constants.kb_j * T / constants.q# Thermal voltage
        V_a = V_bias ## aplied bias, negative is reverse
        V_bi = V_t * np.log((device.N_A*device.N_D)/(materials.n_i**2)) # computes built-in voltage
        Vtotal = V_bi - V_a

        if Vtotal <= 0:
            raise ValueError(
                f"Va={V_a:.3f} V exceeds V_bi={V_bi:.4f} V — "
                "depletion approximation breaks down.")

        dep_w = np.sqrt(2 * eps / constants.q * (device.N_A+device.N_D)/(device.N_A*device.N_D)*(Vtotal)) # computes depletion width
        x_n = dep_w * device.N_A / (device.N_A+device.N_D)
        x_p = dep_w * device.N_D / (device.N_A+device.N_D)

        Emax_p = -constants.q * device.N_A * x_p / eps
        Emax_n = -constants.q * device.N_D * x_n / eps   # should e

        # Check for matching maximum field
        if Emax_p != Emax_p:
            raise ValueError(
                f"Emax_p = {Emax_p:.3f}, which is not equal to Emax_n = {Emax_n:.3f}"
            )
        
        # Check for xp
        if x_p > device.d_p:
            raise ValueError(f"Depletion width xp={x_p*1e9:.1f} nm exceeds p-side length Lp={device.d_p*1e9:.1f} nm.")
        if x_n > device.d_p:
            raise ValueError(f"Depletion width xn={x_n*1e9:.1f} nm exceeds n-side length Ln={device.d_n*1e9:.1f} nm.")
        
        rho = np.zeros(len(z))
        rho[(z >= -x_p) & (z <  0)] = -constants.q * device.N_A     # ionised acceptors [C/m^3]
        rho[(z >= 0)  & (z <= x_n)] =  constants.q * device.N_D    # ionised donors    [C/m^3]

        dEdz = rho / eps                                        # [V/m^2]
        E = np.concatenate(([0.0], cumulative_trapezoid(dEdz, z))) # [V/m]
        E[z < -x_p] = 0.0
        E[z >  x_n] = 0.0

        V = np.concatenate(([0.0], cumulative_trapezoid(-E, z)))   # [V]
        V[z < -x_p] = 0.0
        V[z >  x_n] = V[np.searchsorted(z, x_n)]   # hold the value reached at xn

        return E, rho, V
    else:
        raise ValueError(f"Unknown electric field model: {model}")

