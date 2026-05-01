"""
materials.py — Material property containers.

Design rules:
- Material is an abstract base dataclass defining the interface every material must implement.
- Concrete subclasses (Silicon, InGaAs, ...) provide actual parameter values.
- Properties that depend on temperature take T [K] as an argument — materials are
  stateless with respect to temperature so the same instance works for T sweeps.
- All units are SI. Comments state units on every field.
- Module-level bare variables (the old materials.py style) are removed. Everything
  lives on a class instance so it can be passed around, inspected, and extended.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import constants


# ── Abstract base ──────────────────────────────────────────────────────────────

@dataclass
class Material:
    """
    Abstract base for semiconductor materials.

    Subclasses must override every method marked NotImplementedError.
    Fields that are truly constant (eps_r) are dataclass fields with defaults.
    Fields that depend on T are methods.
    """
    name: str
    eps_r: float     # relative permittivity [-]

    # ── Intrinsic carrier density ──────────────────────────────────────────────

    def ni(self, T: float) -> float:
        """Intrinsic carrier concentration [m^-3] at temperature T [K]."""
        raise NotImplementedError

    # ── Mobilities ────────────────────────────────────────────────────────────

    def mu_n(self, T: float) -> float:
        """Electron mobility [m^2/Vs] at temperature T [K]."""
        raise NotImplementedError

    def mu_p(self, T: float) -> float:
        """Hole mobility [m^2/Vs] at temperature T [K]."""
        raise NotImplementedError

    # ── Diffusion coefficients (Einstein relation) ─────────────────────────────
    # D = mu * kB*T/q  — not overridden by subclasses unless Einstein breaks down.

    def D_n(self, T: float) -> float:
        """Electron diffusion coefficient [m^2/s] at temperature T [K]."""
        return self.mu_n(T) * constants.kB * T / constants.q

    def D_p(self, T: float) -> float:
        """Hole diffusion coefficient [m^2/s] at temperature T [K]."""
        return self.mu_p(T) * constants.kB * T / constants.q

    # ── Optical absorption ────────────────────────────────────────────────────

    def alpha(self, lam: float, T: float = 300.0) -> float:
        """
        Absorption coefficient [m^-1] at wavelength lam [m] and temperature T [K].
        Returns 0.0 by default (transparent). Override for absorbing materials.
        """
        return 0.0

    # ── Convenience ───────────────────────────────────────────────────────────

    def eps_s(self) -> float:
        """Absolute permittivity [F/m]."""
        return self.eps_r * constants.eps0

    def Vt(self, T: float) -> float:
        """Thermal voltage kB*T/q [V]."""
        return constants.kB * T / constants.q

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps_r={self.eps_r})"


# ── Silicon ────────────────────────────────────────────────────────────────────

@dataclass
class Silicon(Material):
    """
    Bulk silicon material parameters.

    ni:   Fitted to Morin & Maita (1954) / Green (1990) data.
          ni(300K) ≈ 1.5e16 m^-3.
    mu_n: Constant 300 K value — 0.135 m^2/Vs.
    mu_p: Constant 300 K value — 0.048 m^2/Vs.

    TODO: Replace mu with Caughey-Thomas T-dependent model when drift-diffusion
    solver is added.
    """
    name:  str   = "Silicon"
    eps_r: float = 11.7

    # ── ni: empirical fit valid ~200–500 K ────────────────────────────────────
    def ni(self, T: float) -> float:
        """
        Intrinsic carrier concentration [m^-3]. Valid ~200-500 K.

        Uses Varshni bandgap with prefactor calibrated to Green (1990):
          ni(300 K) = 9.65e9 cm^-3 = 9.65e15 m^-3.

        Note: older literature uses 1.5e10 cm^-3 (1.5e16 m^-3), which is
        a measurement artefact from indirect recombination. 9.65e15 m^-3
        is the currently accepted value for pure Si at 300 K.
        """
        Eg  = 1.1242 - (4.73e-4 * T**2) / (T + 636)   # Varshni bandgap [eV]
        kT  = constants.kB * T / constants.q            # [eV]
        # Prefactor C in m^-6 K^-3, calibrated to ni(300K) = 9.65e15 m^-3
        C   = 4.5643e42
        ni2 = C * T**3 * np.exp(-Eg / kT)
        return np.sqrt(ni2) # [m^-6]

    # ── Mobilities (300 K, constant) ─────────────────────────────────────────

    def mu_n(self, T: float) -> float:
        """Electron mobility [m^2/Vs]. Currently constant at 300 K value."""
        # TODO: Caughey-Thomas: mu = mu_min + (mu_max - mu_min) / (1 + (T/T0)^gamma)
        _ = T   # suppress unused-arg warning until T-dependence is added
        return 0.1350   # m^2/Vs

    def mu_p(self, T: float) -> float:
        """Hole mobility [m^2/Vs]. Currently constant at 300 K value."""
        _ = T
        return 0.0480   # m^2/Vs

    def alpha(self, lam: float, T: float = 300.0) -> float:
        """
        Absorption coefficient [m^-1] at wavelength lam [m].
        Simple above-bandgap step model. Replace with tabulated data or
        more detailed model when optics module is added.
        """
        Eg_eV = 1.1242 - (4.73e-4 * T**2) / (T + 636)
        E_photon_eV = constants.h * constants.c / (lam * constants.q)
        if E_photon_eV > Eg_eV:
            # Rough indirect-gap fit: alpha ~ A * sqrt(E - Eg)
            A = 1.4e8   # m^-1 eV^-0.5, empirical for Si
            return A * np.sqrt(E_photon_eV - Eg_eV)
        return 0.0


# ── Registry ───────────────────────────────────────────────────────────────────
# Add new materials here as they are implemented.
MATERIALS: dict[str, type[Material]] = {
    "silicon": Silicon,
}