import numpy as np
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class LightSource:
    """
    Defines the incident light signal.

    Parameters
    ----------
    P : float
        Peak optical power [W].
    lam : float
        Center wavelength [m].
    bandwidth : float
        Spectral 1-sigma bandwidth [m]. 0 for monochromatic.
    area : float
        Illuminated area [m²].
    mode : str
        'cw' for constant wave, 'pulsed' for time-varying envelope.
    pulse_t : ndarray, optional
        Time axis of pulse envelope [s]. Required if mode='pulsed'.
    pulse_env : ndarray, optional
        Normalized pulse envelope (0-1). Required if mode='pulsed'.
    """
    P          : float
    lam        : float
    bandwidth  : float
    area       : float
    mode       : str = 'cw'
    pulse_t    : np.ndarray = field(default=None, repr=False)
    pulse_env  : np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.mode == 'pulsed':
            if self.pulse_t is None or self.pulse_env is None:
                raise ValueError("pulse_t and pulse_env required for pulsed mode")
            # build interpolator once, evaluated on demand by solver
            self._interp = lambda t: np.interp(t, self.pulse_t, self.pulse_env, left=0.0, right=0.0)

    def power(self, t: float) -> float:
        """Return power at time t [W]."""
        if self.mode == 'cw':
            return self.P
        elif self.mode == 'pulsed':
            return self.P * float(self._interp(t))