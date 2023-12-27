import numpy as np
import healpy as hp
import scipy
from scipy.special import sph_harm
import os
import sys
import tools


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
        hpmap += amps[i] * comp
    return hpmap


def construct_templates(ells, NSIDE=64):
    """
    Returns a (n,npix) array of Y_lm templates; the design matrix used to fit multipoles to a healpix map.
    
    Parameters
    ----------
    ells : int or array-like
        The degrees to construct.
    NSIDE : int, optional
        The healpix resolution.
    
    Returns
    -------
    templatess : (n,npix) array
        The design matrix: each column corresponds to a Ylm template. n is 2ell+1 summed over the input ells.
        
    """
    # check/adjust input ells
    ells = np.array(ells).astype(int)
    assert ells.ndim <= 1
    # if input is a single value
    if ells.ndim == 0:
        ells = ells[...,np.newaxis]
    
    # construct templates for each ell and append to 
    n = np.sum([2 * ell + 1 for ell in ells])
    templatess = np.empty((n,hp.nside2npix(NSIDE)))
    it = 0  # keep track of the column index
    for ell in ells:
        templates = np.array([
            multipole_map(amps, NSIDE=NSIDE) for amps in np.identity(2 * ell + 1)
        ])
        templatess[it:it + 2 * ell + 1] = templates
        it += 2 * ell + 1
    
    return templatess


def fit_multipole(map_to_fit, template_maps, Cinv=None, fit_zeros=False, idx=None):
    """
    Fits multipole amplitudes to an input healpix density map.
    
    Parameters
    ----------
    map_to_fit : 1D array-like, length npix
        Input healpix map.
    template_maps : 2D array-like, shape (n,npix)
        Y_lm templates to fit.
    Cinv : array-like, optional
        Inverse covariance matrix. If 1D, taken to be the diagonal terms.
    fit_zeros : bool, optional
        Whether to fit zero-valued pixels in `map_to_fit`. The default is False.
    idx : array-like, optional
        Pixel indices to fit.
    
    Returns
    -------
    bestfit_pars :
        The 2 * ell + 1 best-fit amplitudes corresponding to each template map.
    bestfit_stderr :
        The standard error on the fit.
    
    """
    assert map_to_fit.ndim == 1, "input map must be 1-dimensional"
    assert len(map_to_fit) == template_maps.shape[1], "input map and template maps must have the same NPIX"
    
    NPIX = len(map_to_fit)
    # design matrix
    A = template_maps.T
    # covariances: identity for now
    if Cinv is None:
        Cinv = np.ones(NPIX)
    else:
        assert len(Cinv) == NPIX, "input Cinv and input map must have the same length"

    # indices to fit
    idx_to_fit = np.full(NPIX, True)
    if fit_zeros is False:
        idx_to_fit = idx_to_fit & (map_to_fit!=0.)
    if idx is not None:
        assert len(idx) == NPIX, "input idx and input map must have the same length"
        idx_to_fit = idx_to_fit & idx
    map_to_fit, A, Cinv = map_to_fit[idx_to_fit], A[idx_to_fit], Cinv[idx_to_fit]

    # perform the regression
    bestfit_pars, bestfit_Cinv = tools.lstsq(map_to_fit, A, Cinv)

    # uncertainties on the best-fit pars
    bestfit_stderr = np.sqrt(np.diag(np.linalg.inv(bestfit_Cinv)))

    return bestfit_pars, bestfit_stderr


def compute_Cells(amps):
    """
    Returns the power C(ell) for several ells given a list of amplitudes corresponding to the a_lm coefficients
    for each ell, increasing from ell=0.
    """
    ell = 0
    i1 = 0
    Cells = np.array([])
    while i1 < len(amps):
        i2 = i1 + 2 * ell + 1
        assert i2 <= len(amps)
        Cell = compute_Cell(amps[i1:i2])
        Cells = np.append(Cells, Cell)
        ell += 1
        i1 = i2
    return Cells


def compute_Cell(alms):
    """
    Returns the power C(ell) given a list of coefficients a_lm for a single ell.
    """
    assert alms.ndim <= 1
    # pad if aellems is a scalar:
    if alms.ndim == 0:
        alms = alms[..., np.newaxis]
    # infer ell from the number of moments 2ell+1
    ell = (len(alms) - 1) // 2
    assert np.mean(alms**2) == np.sum(alms**2)/(2*ell+1)
    return np.mean(alms**2)