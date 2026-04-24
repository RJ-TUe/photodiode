# function that computes photon flux
# todo: Add in polylchromatic model.
import constants
from light_source import LightSource

def flux(source: LightSource, t: float, model='monochromatic') -> float:
    """
    Compute photon flux based on the incident light

    Parameters
    ----------
    P : float
        Power [W]
    lam : float or ndarray
        Wavelength of incident light [m]
    A : float
        Area where light is shone [m^2]
    **kwargs
        Model-specific parameters.
    Returns
    -----
    float or ndarray
        photon flux, [m^2/s]
    Notes
    -----
    """
    if model == 'monochromatic':
        P = source.power(t)
        return (P / source.area) * source.lam / (constants.h * constants.c)
    else:
        raise ValueError(f"Unknown absorbance model: {model}")