""" Useful functions for reservoir simulation tasks """

import numpy

def linear_mobility(s, vw, vo, swir, soir, deriv=False):
    """ Linear mobility model

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    vw : float
        Viscosity of water

    vo : float
        Viscosity of oil

    swir : float
        Irreducible water saturation

    soir : float
        Irreducible oil saturation

    deriv : bool
        If True, also return derivatives

    Returns
    -------
    if deriv=False,
    lamb_w, lamb_o : (2x) ndarray, shape (ny, nx) | (ny*nx,)
        lamb_w : water mobility
        lamb_o : oil mobility

    if deriv=True,
    lamb_w, lamb_o, dlamb_w, dlamb_o : (4x) ndarray, shape (ny, nx) | (ny*nx,)
        lamb_w : water mobility
        lamb_o : oil mobility
        dlamb_w : derivative of water mobility
        dlamb_o : derivative of oil mobility
    """
    vw, vo, swir, soir = float(vw), float(vo), float(swir), float(soir)
    _s = (s-swir)/(1.0-swir-soir)
    lamb_w = _s/vw
    lamb_o = (1.0-_s)/vo
    lamb_w, lamb_o = numpy.clip(lamb_w, 0., 1.), numpy.clip(lamb_o, 0., 1.)  # clip to ensure within [0,1]

    if deriv:
        dlamb_w = 1.0/(vw*(1.0-swir-soir))
        dlamb_o = -1.0/(vo*(1.0-swir-soir))
        return lamb_w, lamb_o, dlamb_w, dlamb_o

    return lamb_w, lamb_o

def quadratic_mobility(s, vw, vo, swir, soir, deriv=False):
    """ Quadratic mobility model

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    vw : float
        Viscosity of water

    vo : float
        Viscosity of oil

    swir : float
        Irreducible water saturation

    soir : float
        Irreducible oil saturation

    deriv : bool
        If True, also return derivatives

    Returns
    -------
    if deriv=False,
    lamb_w, lamb_o : (2x) ndarray, shape (ny, nx) | (ny*nx,)
        lamb_w : water mobility
        lamb_o : oil mobility

    if deriv=True,
    lamb_w, lamb_o, dlamb_w, dlamb_o : (4x) ndarray, shape (ny, nx) | (ny*nx,)
        lamb_w : water mobility
        lamb_o : oil mobility
        dlamb_w : derivative of water mobility
        dlamb_o : derivative of oil mobility
    """

    vw, vo, swir, soir = float(vw), float(vo), float(swir), float(soir)
    _s = (s-swir)/(1.0-swir-soir)
    lamb_w = _s**2/vw
    lamb_o = (1.0-_s)**2/vo
    lamb_w, lamb_o = numpy.clip(lamb_w, 0., 1.), numpy.clip(lamb_o, 0., 1.)  # clip to ensure within [0,1]

    if deriv:
        dlamb_w = 2.0*_s/(vw*(1.0-swir-soir))
        dlamb_o = -2.0*(1.0-_s)/(vo*(1.0-swir-soir))
        return lamb_w, lamb_o, dlamb_w, dlamb_o

    return lamb_w, lamb_o

def f_fn(s, mobi_fn):
    """ Water fractional flow

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    mobi_fn : callable
        Mobility function lamb_w, lamb_o = mobi_fn(s) where:
            lamb_w : water mobility
            lamb_o : oil mobility
    """
    lamb_w, lamb_o = mobi_fn(s)
    return lamb_w / (lamb_w + lamb_o)

def df_fn(s, mobi_fn):
    """ Derivative (element-wise) of water fractional flow

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    mobi_fn : callable
        Mobility function lamb_w, lamb_o, dlamb_w, dlamb_o = mobi_fn(s, deriv=True) where:
            lamb_w : water mobility
            lamb_o : oil mobility
            dlamb_w : derivative of water mobility
            dlamb_o : derivative of oil mobility
    """
    lamb_w, lamb_o, dlamb_w, dlamb_o = mobi_fn(s, deriv=True)
    return dlamb_w / (lamb_w + lamb_o) - lamb_w * (dlamb_w + dlamb_o) / (lamb_w + lamb_o)**2

def lamb_fn(s, mobi_fn):
    """ Total mobility

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    mobi_fn : callable
        Mobility function lamb_w, lamb_o = mobi_fn(s) where:
            lamb_w : water mobility
            lamb_o : oil mobility
    """
    lamb_w, lamb_o = mobi_fn(s)
    return lamb_w + lamb_o
