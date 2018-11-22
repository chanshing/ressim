""" Useful functions for reservoir simulation tasks """

import numpy

def linear_mobility(s, mu_w, mu_o, s_wir, s_oir, deriv=False):
    """ Linear mobility model

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    mu_w : float
        Viscosity of water

    mu_o : float
        Viscosity of oil

    s_wir : float
        Irreducible water saturation

    s_oir : float
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
    mu_w, mu_o, s_wir, s_oir = float(mu_w), float(mu_o), float(s_wir), float(s_oir)
    _s = (s-s_wir)/(1.0-s_wir-s_oir)
    lamb_w = _s/mu_w
    lamb_o = (1.0-_s)/mu_o

    if deriv:
        dlamb_w = 1.0/(mu_w*(1.0-s_wir-s_oir))
        dlamb_o = -1.0/(mu_o*(1.0-s_wir-s_oir))
        return lamb_w, lamb_o, dlamb_w, dlamb_o

    return lamb_w, lamb_o

def quadratic_mobility(s, mu_w, mu_o, s_wir, s_oir, deriv=False):
    """ Quadratic mobility model

    Parameters
    ----------
    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    mu_w : float
        Viscosity of water

    mu_o : float
        Viscosity of oil

    s_wir : float
        Irreducible water saturation

    s_oir : float
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

    mu_w, mu_o, s_wir, s_oir = float(mu_w), float(mu_o), float(s_wir), float(s_oir)
    _s = (s-s_wir)/(1.0-s_wir-s_oir)
    lamb_w = _s**2/mu_w
    lamb_o = (1.0-_s)**2/mu_o

    if deriv:
        dlamb_w = 2.0*_s/(mu_w*(1.0-s_wir-s_oir))
        dlamb_o = -2.0*(1.0-_s)/(mu_o*(1.0-s_wir-s_oir))
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
