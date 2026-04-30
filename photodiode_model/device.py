# Define as layer stack, not as a structured device.

from dataclasses import dataclass
import materials
import numpy as np

@dataclass
class PINJunction:
    """
    Defines the geometry and doping of a P-i-N junction.

    Parameters
    ----------
    d_p : float
        Thickness of p-region [m].
    d_i : float
        Thickness of intrinsic region [m].
    d_n : float
        Thickness of n-region [m].
    N_A : float
        Acceptor doping concentration in p-region [m⁻³].
    N_D : float
        Donor doping concentration in n-region [m⁻³].
    """
    d_p : float
    d_i : float
    d_n : float
    N_A : float
    N_D : float

    ## Region commands
    @property
    def d(self) -> float:
        """Total device thickness [m]."""
        return self.d_p + self.d_i + self.d_n

    @property
    def z_p(self) -> float:
        """End of p-region [m]."""
        return self.d_p

    @property
    def z_i(self) -> float:
        """End of i-region [m]."""
        return self.d_p + self.d_i

    def region(self, z: float) -> str:
        """Return which region z falls in: 'p', 'i', or 'n'."""
        if z < self.z_p:
            return 'p'
        elif z < self.z_i:
            return 'i'
        else:
            return 'n'
        
@dataclass
class PNJunction:
    """
    Defines the geometry and doping of a P-N junction.

    Parameters
    ----------
    d_p : float
        Thickness of p-region [m].
    d_n : float
        Thickness of n-region [m].
    N_A : float
        Acceptor doping concentration in p-region [m⁻³].
    N_D : float
        Donor doping concentration in n-region [m⁻³].
    """
    d_p : float
    d_n : float
    N_A : float
    N_D : float
    n_points: int = 20000  # default resolution

    ## Region properties
    @property
    def d(self) -> float:
        """Total device thickness [m]."""
        return self.d_p + self.d_n

    @property
    def z_p(self) -> float:
        """boundary of pn junction [m]."""
        return self.d_p
        
    @property
    def mesh(self):
        """1D spatial mesh from 0 to device length."""
        return np.linspace(-self.d_p, self.d_n, self.n_points)
    
    def region(self, z: float) -> str:
        """Return which region z falls in: 'p', or 'n'."""
        if z < self.z_p:
            return 'p'
        else:
            return 'n'