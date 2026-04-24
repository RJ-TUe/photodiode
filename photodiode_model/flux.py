# function that computes photon flux
# todo: Add in polylchromatic model.
import constants

def flux(P, lam, A, model = 'monochromatic'):
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
        return (P / A) * lam / (constants.h * constants.c)   # photons / (m^2 s)
    else:
        raise ValueError(f"Unknown absorbance model: {model}")