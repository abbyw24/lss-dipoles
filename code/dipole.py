import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord

import tools


### DIPOLE CONTRIBUTIONS

def dipole(theta, phi, dipole_x, dipole_y, dipole_z):
    """
    Return the signal contribution from the dipole at a certain sky location (theta,phi).
    """
    return dipole_x*np.sin(theta)*np.cos(phi) + dipole_y*np.sin(theta)*np.sin(phi) + dipole_z*np.cos(theta)


def dipole_map(amps, NSIDE=64):
    """
    Generate a healpix dipole map (equatorial coordinates) given four parameters:
       the monopole plus three dipole amplitudes.
       
    Bugs/Comments:
    - amps is a 4-vector because we suck.
    """
    NPIX = hp.nside2npix(NSIDE)  # number of pixels
    theta, phi = hp.pix2ang(NSIDE, ipix=np.arange(NPIX))  # get (theta,phi) coords of each pixel
    dip = dipole(theta, phi, *amps[1:])  # expected dipole: shape==(NPIX,)
    return amps[0] + dip


def fit_dipole(map_to_fit, Cinv=None, fit_zeros=False, idx=None, Lambda=0):
    """
    Perform a least-squares fit to the dipole in a healpix map.

    Parameters
    ----------
    map_to_fit : healpix map
        Map of pixel values to fit.
    Cinv : array-like, optional
        Diagonal elements of the inverse covariance matrix---or really data weights---to
        use in the fit. If None, identity is used.
    fit_zeros : bool
        Whether to fit pixels in the map with value zero.
    idx : array-like, optional
        The indices to fit.

    Returns
    -------
    bestfit_pars : array-like
        Monopole + three dipole components.
    bestfit_stderr : float
        Uncertainties on the best-fit amplitudes.
    """

    assert map_to_fit.ndim == 1, "input map must be 1-dimensional"

    NPIX = len(map_to_fit)

    # 3 orthogonal dipole template maps
    template_amps = np.column_stack((np.zeros(3), np.diag(np.ones(3))))  # first column for the monopole
    template_maps = np.array([dipole_map(amps, NSIDE=hp.npix2nside(NPIX)) for amps in template_amps])

    # design matrix
    A = np.column_stack((np.ones(NPIX), template_maps.T))
    # covariances: identity for now
    if Cinv is None:
        Cinv = np.ones(NPIX)
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
    bestfit_pars, bestfit_Cinv = tools.lstsq(map_to_fit, A, Cinv, Lambda=Lambda)

    # uncertainties on the best-fit pars
    bestfit_stderr = np.sqrt(np.diag(np.linalg.inv(bestfit_Cinv)))

    return bestfit_pars, bestfit_stderr

def overdensity_map(qmap, selfunc, min_selfunc=0.5):
    # turn the input into an overdensity map
    qmap_corrected = qmap / selfunc # Hogg is dying here
    good = selfunc > min_selfunc # This has no meaning, which is great!
    odmap = qmap_corrected * np.nanmean(selfunc[good]) / np.nanmean(qmap[good]) - 1.
    odmap[np.logical_not(good)] = np.NaN # Every line of this code is making Hogg die a little.
    return odmap

def measure_overdensity_dipole_Lambda(sample, Lambda, selfunc=None, fit_zeros=True, verbose=False):
    """
    Wrapper for `dipole.fit_dipole()`. The input `sample` should be an overdensity map.
    """
    map_to_fit = sample.copy()
    idx_masked = np.isnan(map_to_fit)
    map_to_fit[idx_masked] = 0.
    if np.all(selfunc == None):
        if verbose:
            print("selection function not provided; assuming completeness = 1 everywhere")
        Cinv = np.ones_like(sample)
    else:
        Cinv = selfunc.copy()
    Cinv[idx_masked] = 0. # for Lambda regularization, Cinv is zero in the masked pixels
    comps, stderr = fit_dipole(map_to_fit, Cinv=Cinv, fit_zeros=fit_zeros, Lambda=Lambda)
    if verbose:
        amplitude, direction = get_dipole(comps[1:])
        print(f"best-fit dipole amp. =\t{amplitude:.5f}")
        print(f"best-fit dipole dir.: ", direction)
    return comps[1:] # since we're fitting overdensities

def getDipoleVectors_healpy(densitymap, mask=[None], galcut=0, verbose=False) :
	"""
    ! COPIED FROM SECREST !
	Computes the preferred direction and the estimated dipole amplitude from a density map
	
	This is a wrapper for the healpy routine fit_dipole
	"""
	
	if mask[0] != None :
		densitymap[(mask == 0)] = np.nan
	
	residual,monopole,dipole = hp.remove_dipole(densitymap,bad=np.nan,fitval=True,gal_cut=galcut,verbose=verbose)
	norm = np.sqrt(np.dot(dipole,dipole))
	dipole_norm = dipole/norm
	
	d = norm/monopole
	
	return dipole_norm,d


def cmb_dipole(frame='icrs', amplitude=0.007, return_comps=False):
    """
    Return the orthogonal (x,y,z) CMB dipole components.
    """
    cmb_dipdir = SkyCoord(264, 48, unit=u.deg, frame='galactic')
    if frame=='icrs':
        comps = tools.spherical_to_cartesian(r=amplitude,
                                             theta=np.pi/2-cmb_dipdir.icrs.dec.rad,
                                             phi=cmb_dipdir.icrs.ra.rad)
    elif frame=='galactic':
        comps = tools.spherical_to_cartesian(r=amplitude,
                                             theta=np.pi/2-cmb_dipdir.b.rad,
                                             phi=cmb_dipdir.l.rad)
    else:
        assert False, "unknown frame"
    if return_comps is True:
        return comps
    else:
        return get_dipole(comps, frame=frame)

def get_dipole(comps, frame='icrs', from_alms=False, verbose=False):
    """
    Return the amplitude and direction of a dipole given its three amplitudes.

    Parameters
    ----------
    comps : array-like
        3 orthogonal dipole amplitudes
    frame : SkyCoord-compatible coordinate frame
        frame of the input amplitudes
    
    Returns
    -------
    amp : float
        amplitude (norm) of the dipole components
    direction : SkyCoord
        direction of the dipole
    """
    assert len(comps) == 3
    if from_alms:
        comps = tools.a1ms_to_dipole_comps(comps)
    amp = np.linalg.norm(comps)
    direction = hp.vec2dir(comps)
    direction = SkyCoord(direction[1], np.pi/2 - direction[0], frame=frame, unit='rad')
    if verbose:
        print(f"amp = {amp:.6f}")
        print("direction: ", direction.galactic)
    return amp, direction