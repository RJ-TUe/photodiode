import numpy as np
## TD: Split functions, or in some way make sure that units are respected.
def absorbance(lam, model="gaussian", **kwargs):
    """
    Compute wavelength-dependent absorbance.

    Parameters
    ----------
    lam : float or ndarray
        Wavelength in meters.
    model : str
        Absorbance model ("gaussian", "bandgap", "beer_lambert).
    **kwargs
        Model-specific parameters.
    Returns
    -----
    float or ndarray
        Absorbance value(s), dimension depends on method
    Notes
    -----
    Gaussian model assumes symmetric spectral broadening.
    """
    if model == "gaussian":
        lam0 = kwargs.get("lam0", 800e-9)
        sigma = kwargs.get(   "sigma", 100e-9)
        return np.exp(-((lam - lam0)**2) / (2 * sigma**2)) # [dimensionless]
    
    elif model == "bandgap":
        Eg = kwargs.get("Eg", 1.5)  # eV
        hc = 1240e-9  # eV*m
        lam_cutoff = hc / Eg
        return np.where(lam < lam_cutoff, 1.0, 0.0) # [dimensionless]
    # Beer-lambert Law: https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law
    elif model == "beer_lambert":
        alpha_0 = kwargs.get("alpha_0", 1e6)  # m⁻¹, peak absorption coefficient
        lam0    = kwargs.get("lam0", 800e-9)
        sigma   = kwargs.get("sigma", 100e-9)
        # spectral dependence of alpha
        return alpha_0 * np.exp(-((lam - lam0)**2) / (2 * sigma**2)) # [m⁻¹]
    
    else:
        raise ValueError(f"Unknown absorbance model: {model}")