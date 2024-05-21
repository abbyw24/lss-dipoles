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

def reconstruct_map(alms, NSIDE=64):
    """
    Reconstruct a healpix map from a list of spherical harmonic coefficients a_lm.
    """
    # iterative over ells to correctly assign the alms to their respective multipoles
    reconstructed_map = np.zeros(hp.nside2npix(NSIDE))
    ell = 0
    i1 = 0  # this will give us the starting index to pull from amps for each ell
    while i1 < len(alms):
        i2 = i1 + 2 * ell + 1  # stopping index to pull from amps
        assert i2 <= len(alms)  # make sure we aren't trying to pull more amplitudes than we input!
        # construct the 2ell+1 templates
        ells = np.arange(ell+1)
        templates = construct_templates(ells, NSIDE=NSIDE)
        # add the map for this ell to the overall reconstructed map
        map_thisell = np.zeros_like(reconstructed_map)
        for im, alm_ in enumerate(alms[i1:i2]):
            map_thisell += alm_ * templates[i1:i2][im]
        reconstructed_map += map_thisell
        ell += 1
        i1 = i2
        
    return reconstructed_map


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


def compute_Cells_from_alms_fit(datamap, Cinv, max_ell, idx_to_fit=None, return_alms=False):
    """
    Performs a linear least-squares fit to a healpix density map to get best-fit spherical harmonic amplitudes alm.
    Automatically excludes NaN and `hp.UNSEEN` pixels in `datamap` from the fit.
    Returns the Cells as computed from the alms (sum of |alms|^2).
    
    """
    assert datamap.shape == Cinv.shape
    assert datamap.ndim == Cinv.ndim == 1
    
    # get number of pixels from input data map
    NPIX = len(datamap)
    
    # construct Ylm templates -> design matrix
    ells = np.arange(0, max_ell+1)
    templates = construct_templates(ells, hp.npix2nside(NPIX))
    A = templates.T
    
    # indices to fit: non-NaN, non-hp.UNSEEN, non-zero
    if idx_to_fit is None:
        idx_to_fit = np.full(NPIX, True).astype(bool)
    idx_to_fit = idx_to_fit & (~np.isnan(datamap)) & (datamap != hp.UNSEEN)
    map_to_fit, A_fit, Cinv_fit = datamap.copy(), A.copy(), Cinv.copy()
    
    # perform the regression: bestfit_pars are the alms
    bestfit_pars, bestfit_Cinv = tools.lstsq(map_to_fit[idx_to_fit], A_fit[idx_to_fit], Cinv_fit[idx_to_fit])
    Cells = compute_Cells(bestfit_pars)
    
    if return_alms == True:
        return ells, Cells, bestfit_pars
    else:
        return ells, Cells


def compute_Cells_in_overdensity_map(overdensity_map, Wmask, max_ell, return_alms=False, selfunc=None,
                                        idx_to_fit=None):
    # wrapper for compute_Cells_from_alms_fit() but taking an overdensity map as input,
    #  to replace all NaN pixels with 0 data and Wmask Cinv
    map_to_fit = overdensity_map.copy()
    idx_masked = np.isnan(map_to_fit)
    map_to_fit[idx_masked] = 0.
    Cinv = np.ones_like(map_to_fit) if np.all(selfunc == None) else selfunc.copy()
    Cinv[idx_masked] = Wmask
    return compute_Cells_from_alms_fit(map_to_fit, Cinv, max_ell, idx_to_fit=idx_to_fit, return_alms=return_alms)


def compute_Cells(amps):
    """
    Returns the power C(ell) for several ells given a list of amplitudes corresponding to the a_lm coefficients
    for each ell, increasing from ell=0.
    """
    ell = 0
    i1 = 0  # this will give us the starting index to pull from amps for each ell
    Cells = np.array([])
    while i1 < len(amps):
        i2 = i1 + 2 * ell + 1  # stopping index to pull from amps
        assert i2 <= len(amps)  # make sure we aren't trying to pull more amplitudes than we input!
        Cell = compute_Cell(amps[i1:i2])  # power for this ell: mean of amps squared
        Cells = np.append(Cells, Cell)  # add the power for this ell to the list
        ell += 1
        i1 = i2
    return Cells


def compute_Cell(alms):
    """
    Returns the power C(ell) given a list of coefficients a_lm for a single ell.
    """
    assert alms.ndim <= 1
    assert np.sum(np.isnan(alms)) == 0, "NaNs in alms!"
    # pad if alms is a scalar:
    if alms.ndim == 0:
        alms = alms[..., np.newaxis]
    # infer ell from the number of moments 2ell+1
    ell = (len(alms) - 1) // 2
    assert np.mean(alms**2) == np.sum(alms**2)/(2*ell+1), f"{np.mean(alms**2):.4f} != {np.sum(alms**2)/(2*ell+1):.4f}!"
    return np.mean(alms**2)