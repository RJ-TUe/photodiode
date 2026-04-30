"""
device.py — Device geometry and layer stack.

Design rules:
- Layer1D is a single physical layer: material, thickness, doping.
- Device1D is an ordered stack of Layer1D instances.
- Neither class contains solver logic, mesh arrays, or computed physics.
- make_mesh() lives here because the spatial extent is a geometric property.
  Nz is passed in as a numerical parameter, not stored on the device.
- __post_init__ validation runs at construction time to catch unit errors
  and nonsensical geometry before any solver sees the device.

Coordinate convention (established by make_mesh):
  z = 0 at the start of the first layer (left contact).
  z increases toward the right contact.
  Total device spans [0, Device1D.thickness].
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from materials import Material


# ── Single layer ───────────────────────────────────────────────────────────────

@dataclass
class Layer1D:
    """
    One physical layer in a 1D device stack.

    Parameters
    ----------
    material : Material
        Material instance (e.g. Silicon()). Carries eps_r, ni(T), mu, alpha.
    thickness : float
        Layer thickness [m]. Must be > 0.
    NA : float
        Acceptor doping concentration [m^-3]. Default 0 (undoped).
    ND : float
        Donor doping concentration [m^-3]. Default 0 (undoped).
    label : str
        Human-readable region name ('p', 'i', 'n', 'well', 'barrier', ...).
        Used for output and debugging only — not used in calculations.
    """
    material:  Material
    thickness: float
    NA:        float = 0.0
    ND:        float = 0.0
    label:     str   = ""

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(
                f"Layer '{self.label}': thickness must be > 0, got {self.thickness} m."
            )
        if self.NA < 0 or self.ND < 0:
            raise ValueError(
                f"Layer '{self.label}': doping concentrations must be >= 0. "
                f"Got NA={self.NA:.2e}, ND={self.ND:.2e}."
            )
        # Unit guard: doping above zero but below ~1e17 m^-3 is almost certainly
        # a cm^-3 value. Lowest intentional doping in Si is ~1e20 m^-3 (1e14 cm^-3).
        _DOPING_MIN_SI = 1e14   # m^-3 — below this is sub-intrinsic, never intentional doping
        if 0 < self.NA < _DOPING_MIN_SI:
            raise ValueError(
                f"Layer '{self.label}': NA={self.NA:.2e} m^-3 looks like cm^-3. "
                f"Multiply by 1e6 to convert to m^-3."
            )
        if 0 < self.ND < _DOPING_MIN_SI:
            raise ValueError(
                f"Layer '{self.label}': ND={self.ND:.2e} m^-3 looks like cm^-3. "
                f"Multiply by 1e6 to convert to m^-3."
            )

    @property
    def net_doping(self) -> float:
        """Net doping ND - NA [m^-3]. Positive = n-type, negative = p-type."""
        return self.ND - self.NA

    @property
    def is_intrinsic(self) -> bool:
        """True if both NA and ND are zero (undoped)."""
        return self.NA == 0.0 and self.ND == 0.0

    def __repr__(self) -> str:
        label = f"'{self.label}' " if self.label else ""
        return (
            f"Layer1D({label}{self.material.name}, "
            f"d={self.thickness*1e9:.1f} nm, "
            f"NA={self.NA:.1e}, ND={self.ND:.1e})"
        )


# ── Device stack ───────────────────────────────────────────────────────────────

@dataclass
class Device1D:
    """
    1D semiconductor device as an ordered stack of Layer1D objects.

    The stack runs left to right: layers[0] is the leftmost layer (z=0),
    layers[-1] is the rightmost layer (z=thickness).

    Parameters
    ----------
    layers : list[Layer1D]
        Ordered list of layers. Must contain at least one layer.
    Va : float
        Applied bias [V]. Positive = forward bias. Default 0.
    T : float
        Device temperature [K]. Default 300.

    Examples
    --------
    P-N junction:
        Device1D(layers=[
            Layer1D(Silicon(), thickness=500e-9, NA=1e23, label='p'),
            Layer1D(Silicon(), thickness=500e-9, ND=1e22, label='n'),
        ])

    P-i-N junction:
        Device1D(layers=[
            Layer1D(Silicon(), thickness=300e-9, NA=1e23, label='p'),
            Layer1D(Silicon(), thickness=200e-9,          label='i'),
            Layer1D(Silicon(), thickness=300e-9, ND=1e22, label='n'),
        ])
    """
    layers: list[Layer1D]
    Va:     float = 0.0
    T:      float = 300.0

    def __post_init__(self):
        if not self.layers:
            raise ValueError("Device1D must have at least one layer.")
        if self.T <= 0:
            raise ValueError(f"Temperature must be > 0 K, got T={self.T}.")

    # ── Geometry ──────────────────────────────────────────────────────────────

    @property
    def thickness(self) -> float:
        """Total device thickness [m]."""
        return sum(layer.thickness for layer in self.layers)

    @property
    def z_interfaces(self) -> list[float]:
        """
        z-positions of all layer interfaces [m], including both contacts.
        Length = len(layers) + 1.
        e.g. [0.0, 300e-9, 500e-9, 800e-9] for a 3-layer device.
        """
        boundaries = [0.0]
        z = 0.0
        for layer in self.layers:
            z += layer.thickness
            boundaries.append(z)
        return boundaries

    def layer_at(self, z: float) -> Layer1D:
        """Return the Layer1D that contains position z [m]."""
        boundaries = self.z_interfaces
        for i, layer in enumerate(self.layers):
            if boundaries[i] <= z < boundaries[i + 1]:
                return layer
        # z exactly at right contact belongs to last layer
        if z == boundaries[-1]:
            return self.layers[-1]
        raise ValueError(
            f"z={z:.4e} m is outside device extent [0, {self.thickness:.4e} m]."
        )

    # ── Doping profile ────────────────────────────────────────────────────────

    def NA_at(self, z: float) -> float:
        """Acceptor concentration at position z [m^-3]."""
        return self.layer_at(z).NA

    def ND_at(self, z: float) -> float:
        """Donor concentration at position z [m^-3]."""
        return self.layer_at(z).ND

    def doping_profile(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (NA, ND) arrays [m^-3] evaluated on a z-grid [m].
        Vectorised over the mesh — used by solvers to build rho(z).
        """
        boundaries = np.array(self.z_interfaces)
        NA = np.zeros_like(z)
        ND = np.zeros_like(z)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                mask = (z >= boundaries[i]) & (z < boundaries[i + 1])
            else:
                mask = (z >= boundaries[i]) & (z <= boundaries[i + 1])
            NA[mask] = layer.NA
            ND[mask] = layer.ND
        return NA, ND

    # ── Material profile ──────────────────────────────────────────────────────
    def eps_profile(self, z: np.ndarray) -> np.ndarray:
        """Absolute permittivity eps_s [F/m] evaluated on mesh z."""
        boundaries = np.array(self.z_interfaces)
        eps = np.zeros_like(z)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                mask = (z >= boundaries[i]) & (z < boundaries[i + 1])
            else:
                mask = (z >= boundaries[i]) & (z <= boundaries[i + 1])
            eps[mask] = layer.material.eps_r * 8.854187817e-12
        return eps

    # ── Mesh ──────────────────────────────────────────────────────────────────
    def make_mesh(self, Nz: int = 20000) -> np.ndarray:
        """
        Uniform mesh over [0, thickness] with Nz points.

        Nz is a numerical parameter (solver concern), not a device property,
        which is why it is passed here rather than stored on the device.

        For non-uniform meshing with refinement at interfaces, use
        make_mesh_refined() below.
        """
        return np.linspace(0.0, self.thickness, Nz)

    def make_mesh_refined(self, Nz_bulk: int = 1000, Nz_interface: int = 200,
                          interface_width: float = 5e-9) -> np.ndarray:
        """
        Non-uniform mesh: fine near layer interfaces, coarse in bulk.

        Parameters
        ----------
        Nz_bulk : int
            Points per layer in the bulk region.
        Nz_interface : int
            Points in the refined zone on each side of each interface.
        interface_width : float
            Half-width of the refined zone around each interface [m].
        """
        boundaries = self.z_interfaces
        segments = []
        for i, layer in enumerate(self.layers):
            z0 = boundaries[i]
            z1 = boundaries[i + 1]
            segments.append(np.linspace(z0, z1, Nz_bulk))
            # Refine near right interface (except right contact)
            if i < len(self.layers) - 1:
                zi = boundaries[i + 1]
                z_left  = max(z0, zi - interface_width)
                z_right = min(z1 + self.layers[i+1].thickness, zi + interface_width)
                segments.append(np.linspace(z_left, z_right, Nz_interface))
        z = np.unique(np.concatenate(segments))
        return z

    # ── Visualizaion ───────────────────────────────────────────────────────────
    def display(self):
        """Display a figure of the current stack, using color coding and squares."""
        fig, ax = plt.subplots()
        width = 1
        len_hold = self.thickness
        for layer in self.layers:
            length = layer.thickness
            if layer.net_doping > 0: # n-doping is green
                facecolor = 'green'
                label = 'n'
            elif layer.net_doping < 0: # p-doping is orange
                facecolor = 'orange'
                label = 'p'
            else:
                facecolor = 'yellow'
                label = 'i'

            region = patches.Rectangle((0, len_hold), width, -length,
                           linewidth=1, edgecolor='black',
                            facecolor=facecolor, label = label)
            # Add the patch to the Axes
            ax.add_patch(region)
            # update x position
            len_hold = len_hold - length
        
        ax.set_xlim(0, width + 0)
        ax.set_ylim(0, self.thickness+500e-9)
        ax.legend()
        plt.show()

    # ── Convenience ───────────────────────────────────────────────────────────
    def summary(self) -> str:
        """Human-readable layer stack summary."""
        lines = [
            f"Device1D  T={self.T} K  Va={self.Va} V  "
            f"total={self.thickness*1e9:.1f} nm",
            f"  {'#':<4} {'Label':<10} {'Material':<12} "
            f"{'d [nm]':>10} {'NA [m^-3]':>12} {'ND [m^-3]':>12}",
            "  " + "-" * 62,
        ]
        for i, layer in enumerate(self.layers):
            lines.append(
                f"  {i:<4} {layer.label:<10} {layer.material.name:<12} "
                f"{layer.thickness*1e9:>10.1f} {layer.NA:>12.2e} {layer.ND:>12.2e}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Device1D({len(self.layers)} layers, "
            f"d={self.thickness*1e9:.1f} nm, "
            f"Va={self.Va} V, T={self.T} K)"
        )