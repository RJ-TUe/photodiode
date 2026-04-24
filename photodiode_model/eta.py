import numpy as np
def eta(lam, model, **kwargs):
    """
    Compute wavelength-dependent quantum efficiency

    Parameters
    ----------
    lam : float or ndarray
        Wavelength in meters.
    model : str
        Absorbance model ("unity", "gaussian", etc.).
    **kwargs
        Model-specific parameters.
    Returns
    -----
    float or ndarray
        Absorbance value(s), dimensionless.
    Notes
    -----
    Internal quantum efficiency
    """
    if model == 'unity':
        return 1.0

    elif model == "gaussian":
        lam0 = kwargs.get("lam0", 800e-9)
        sigma = kwargs.get(   "sigma", 100e-9)
        return 0.9 * np.exp(-((lam - lam0)**2) / (2 * sigma**2))
    
    else:
        raise ValueError(f"Unknown absorbance model: {model}")