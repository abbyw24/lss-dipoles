import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord

import tools


### COORDINATE TRANSFORMATIONS

def xyz_to_thetaphi(xyz):
    """
    Given (x,y,z), return (theta,phi) coordinates on the sphere, where phi=LON and theta=LAT.
    """
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    return theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, y, z])


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
    cov : array-like, optional
        Covariance matrix to use in the fit. If None, identity is used.
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
    bestfit_pars, bestfit_Cinv = tools.lstsq(map_to_fit, A, Cinv, Lambda=Lambda)

    # uncertainties on the best-fit pars
    bestfit_stderr = np.sqrt(np.diag(np.linalg.inv(bestfit_Cinv)))

    return bestfit_pars, bestfit_stderr


def measure_dipole_in_overdensity_map(sample, selfunc=None, Wmask=0.1):
    map_to_fit = sample.copy()
    idx_masked = np.isnan(map_to_fit)
    map_to_fit[idx_masked] = 0.
    Cinv = np.ones_like(sample) if np.all(selfunc == None) else selfunc.copy()
    Cinv[idx_masked] = Wmask
    amps, stderr = fit_dipole(map_to_fit, Cinv=Cinv, fit_zeros=True)
    return amps[1:] # since we're fitting overdensities


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


def cmb_dipole(frame='icrs', amplitude=0.007, return_amps=False):
    """
    Return the orthogonal (x,y,z) CMB dipole components.
    """
    cmb_dipdir = SkyCoord(264, 48, unit=u.deg, frame='galactic')
    if frame=='icrs':
        amps = spherical_to_cartesian(r=amplitude,
                                             theta=np.pi/2-cmb_dipdir.icrs.dec.rad,
                                             phi=cmb_dipdir.icrs.ra.rad)
    elif frame=='galactic':
        amps = spherical_to_cartesian(r=amplitude,
                                             theta=np.pi/2-cmb_dipdir.b.rad,
                                             phi=cmb_dipdir.l.rad)
    else:
        assert False, "unknown frame"
    if return_amps is True:
        return amps
    else:
        return get_dipole(amps, frame=frame)


def get_dipole(amps, frame='icrs', verbose=False):
    """
    Return the amplitude and direction of a dipole given its three amplitudes.

    Parameters
    ----------
    amps : array-like
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
    assert len(amps) == 3
    amp = np.linalg.norm(amps)
    direction = hp.vec2dir(amps)
    direction = SkyCoord(direction[1], np.pi/2 - direction[0], frame=frame, unit='rad')
    if verbose:
        print(f"amp = {amp:.6f}")
        print("direction: ", direction.galactic)
    return amp, direction