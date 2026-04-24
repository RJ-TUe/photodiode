"""
Photodetector Physics Model for calculating the generated carriers
==================

Calculate the generation of carriers based on the wavelength, intensity, 
and bandwidth of incident light.

The model is a P-i-N diode. Try to define the photodiode in a structurally
agnostic way, so no difference between horizontal, vertical, or other layers.

Assumption: Each photon absorbed in the active region forms an eh pair
Input:
    - Wavelength λ [m]
    - Light intensity
    - Light bandwitdh []
    - Material properties

Output
    - Carrier generation rate, G
    - Quantum efficiency η_int (internal, after absorption)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import warnings

# ── Physical constants ────────────────────────────────────────────────────────
q   = 1.602e-19   # Elementary charge [C]
eps0= 8.854e-12   # Permittivity of free space [F/m]
hc  = 1.986e-25   # h·c  [J·m]
kB  = 1.381e-23   # Boltzmann constant [J/K]


# All parameters at 300 K unless noted. For now, use silicon
# to build a working model
MATERIALS = {
    "Si": {
        "description"   : "Silicon",
        "bandgap_eV"    : 1.12,         # E_g [eV]
        "epsilon_r"     : 11.7,         # Relative permittivity
        "mu_n"          : 0.1350,       # Electron mobility [m²/V·s]
        "mu_p"          : 0.0480,       # Hole mobility [m²/V·s]
        "tau_n"         : 1e-6,         # Electron lifetime [s]  (lightly doped)
        "tau_p"         : 1e-6,         # Hole lifetime [s]
        "n_i"           : 9.65e9 * 1e6, # Intrinsic carrier density [m⁻³]
        "alpha_coeff"   : (1e7, 1.1e-6),# (α₀ [m⁻¹], λ_gap [m]) for Urbach model
        "D_n"           : 0.0025,       # Electron diffusivity [m²/s]  (D=μkT/q)
        "D_p"           : 0.00125,      # Hole diffusivity [m²/s]  (D=μkT/q)
        "EU"            : 0.007,        # Urbach Energy factor
        "alpha0"        : 1e7,          # prefactor for urbach
    },
}

## Returns photon energy for a particular wavelength
def photon_energy(wavelength: float) -> float:
    """Photon energy E_ph = hc/λ [J]."""
    return hc / wavelength

def calculate_alpha(wavelength      : float, # wavelength in m
                    material_dict   : dict, # material parameters
                    A               : float=1e5,          # scaling factor (cm^-1 eV^-1/2)
                    ):  
    """
    Absorption coefficient for InP using a simplified Adachi-like model.

    Parameters:
        lambda : float or np.array
            Wavelength in m
        Eg : Get from material database
            Bandgap energy (eV)
        A : float
            Above-bandgap scaling constant
        EU : float
            Urbach energy (eV)
        alpha0 : float
            Tail prefactor (cm^-1)

    Returns:
        alpha : float or np.array
            Absorption coefficient (cm^-1)
    """
    ## Get material parameters
    Eg = material_dict['bandgap_eV'] * q # bandgap in Joule
    EU = material_dict['EU']*q     # Urbach energy
    alpha0 = material_dict['alpha0'] # prefactor scaling factor
    # convert wavelength to energy (eV)
    E = photon_energy(wavelength)

    # convert to array
    alpha = np.zeros_like(E)

    # above bandgap
    mask_above = E > Eg
    alpha[mask_above] = A * np.sqrt(E[mask_above] - Eg)

    # below bandgap (Urbach tail)
    mask_below = ~mask_above
    alpha[mask_below] = alpha0 * np.exp((E[mask_below] - Eg) / EU)

    return alpha

wavelength = np.linspace(100e-9, 1550e-9, 100)
Si = MATERIALS['Si']

E_photon = photon_energy(wavelength)
alpha = calculate_alpha(wavelength, Si)

### plot
plt.figure()
plt.plot(E_photon, alpha)
plt.show()

print(E_photon)