import numpy as np
import healpy as hp
import scipy
from scipy.special import sph_harm
import os
import sys


def real_sph_harm(m, ell, theta, phi):
    """
    Return a real basis of the spherical harmonics given `ell`, `m`, and a set of angles `(theta,phi)`.
    """
    if m < 0:
        res = (1j * np.sqrt(1/2) * \
               (sph_harm(-np.abs(m), ell, phi, theta) - (-1)**m * sph_harm(np.abs(m), ell, phi, theta))).real
    elif m > 0:
        res = np.sqrt(1/2) * \
                (sph_harm(-np.abs(m), ell, phi, theta) + (-1)**m * sph_harm(np.abs(m), ell, phi, theta)).real
    else:
        assert m == 0
        res = sph_harm(m, ell, phi, theta).real
    return res


def multipole_map(amps, NSIDE=64):
    """
    Return a healpix map of a multipole given input amplitude of each component `amps`.
    The input `amps` must be ordered in increasing m, i.e. -ell to ell.
    """
    assert amps.ndim <= 1
    if amps.ndim == 0:
        amps = amps[...,np.newaxis]
    assert len(amps) % 2 == 1
    ell = (len(amps) - 1) // 2
    ms = np.linspace(-ell, ell, len(amps))
    NPIX = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, ipix=np.arange(NPIX))
    hpmap = np.zeros(NPIX)
    for i, m in enumerate(ms):
        comp = real_sph_harm(m, ell, theta, phi)
        comp /= np.max(comp)
        hpmap += amps[i] * comp
    return hpmap