import numpy as np

def generation_rate(z, eta, alpha, phi, model="uniform"):
    """
    Compute volumetric carrier generation rate.

    Parameters
    ----------
    z : float or ndarray
        Depth into absorber [m].
    eta : float or ndarray
        Quantum efficiency, dimensionless.
    alpha : float or ndarray
        Absorption coefficient [m⁻¹]. For Beer-Lambert model.
    phi : float
        Incident photon flux at surface [m⁻² s⁻¹].
    model : str
        "uniform"      — spatially uniform G (original 0D model)
        "beer_lambert" — exponential depth profile

    Returns
    -------
    float or ndarray
        Generation rate [m⁻³ s⁻¹].
    """
    if model == "uniform":
        # alpha here is dimensionless absorptance, d must be divided externally
        return eta * alpha * phi  # caller divides by d

    elif model == "beer_lambert":
        return eta * alpha * phi * np.exp(-alpha * z)

    else:
        raise ValueError(f"Unknown generation rate model: {model}")