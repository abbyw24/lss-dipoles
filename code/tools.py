"""
Helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import os
import healpy as hp
from healpy.newvisufunc import projview
import pyabc
import pandas as pd


"""
HEALPIX FUNCTIONS
"""
def load_catalog_as_map(catalog, frame='icrs', nside=64, dtype=float):
    if type(catalog)==str:
        tab = Table.read(catalog, format='fits')
    elif type(catalog)==astropy.table.table.Table:
        tab = catalog
    else:
        raise TypeError("invalid input catalog type (filename or astropy table)")
    if frame=='galactic':
        try:
            lon, lat = tab['l'], tab['b']
        except KeyError:
            print("galactic coordinates not found, defaulting to ICRS")
            frame = 'icrs'
    else:
        assert frame=='icrs', "invalid 'frame'"
        lon, lat = tab['ra'], tab['dec']
    # format into healpy map
    pix_idx = hp.ang2pix(nside, lon, lat, lonlat=True)
    hpmap = np.bincount(pix_idx, minlength=hp.nside2npix(nside))
    return hpmap.astype(dtype)

def get_galactic_plane_mask(blim, nside=64, frame='icrs'):
    """Returns a HEALPix mask (1s and 0s) around the galactic plane given an absolute b (latitude) limit."""
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    b = SkyCoord(lon * u.deg, lat * u.deg, frame='icrs').galactic.b
    gal_plane_mask = np.zeros(hp.nside2npix(nside))
    gal_plane_mask[np.abs(b.deg) >= blim] = 1
    return gal_plane_mask

def flatten_map(hpmap):
    newarr = np.array([row[0] for row in hpmap])
    return np.reshape(newarr, (newarr.size,))

def plot_map(map, projection_type='mollweide', coord=['C'],
             graticule=True, graticule_labels=True, **kwargs):
    projview(map, projection_type=projection_type, coord=coord,
             graticule=graticule, graticule_labels=graticule_labels, **kwargs)

def mollview(map, coord=['C'], graticule=True, graticule_coord=None, graticule_labels=True,
            title=None, unit='number density per healpixel', figsize=None, **kwargs):
    hp.mollview(map, coord=coord, title=title, unit=unit, **kwargs)
    if graticule:
        gratcoord = graticule_coord if graticule_coord else None
        hp.graticule(coord=gratcoord)

def plot_marker(lon, lat, **kwargs):
    theta, phi = lonlat_to_thetaphi(lon, lat)
    hp.newprojplot(theta, phi.wrap_at(np.pi * u.rad), **kwargs)

def label_coord(coordsysstr, ax=None, fs=14):
    if ax is None:
        ax = plt.gca()
    ax.text(0.86,
            0.05,
            coordsysstr,
            fontsize=fs,
            fontweight="bold",
            transform=ax.transAxes)

def smooth_map(density_map, sr=1, verbose=True):

    theta = omega_to_theta(sr)  # convert steradians to angle on the sky

    NPIX = len(density_map)
    lon, lat = hp.pix2ang(hp.npix2nside(NPIX), np.arange(NPIX), lonlat=True)
    sc = SkyCoord(lon * u.deg, lat * u.deg, frame='icrs')

    # initial column
    smoothed_map = -1 * np.ones(NPIX)
    for i, denspix in enumerate(density_map):
        if np.isnan(denspix):
            smoothed_map[i] = np.nan
        else:
            d2d = sc[i].separation(sc)
            mask = d2d < theta
            smoothed_map[i] = np.nanmean(density_map[mask])
        if verbose:
            print("%.1f%%" % ((i + 1) / NPIX * 100), end='\r')

    return smoothed_map

def generate_noise_map(mu, nside):
    # Poisson noise healpix map given a mean density and pixel resolution
    return np.random.poisson(mu, hp.nside2npix(int(nside))).astype(float)


"""
LINEAR LEAST-SQUARES FIT
"""
def lstsq(Y, A, Cinv, Lambda=0):
    """
    Return the least-squares solution to a linear matrix equation,
    given data Y, design matrix A, inverse covariance Cinv, and
    optional regularization term Lambda.
    BUG:
    - This should catch warnings and errors in the `res` object.
    Theta = [ A.T @ C^{-1} @ A ]^{-1} @ [ A.T @ C^{-1} @ Y ]
    """
    if len(Cinv.shape)==1:
        a = A.T @ (Cinv[:,None] * A)
        b = A.T @ (Cinv * Y)
    else:
        a = A.T @ Cinv @ A
        b = A.T @ Cinv @ Y
    # add regularization term
    a += Lambda * len(Y) * np.identity(len(a))
    res = np.linalg.lstsq(a, b, rcond=None)
    return res[0], a


"""
COORDINATE TRANSFORMATIONS
"""
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

def omega_to_theta(omega):
    """
    Convert solid angle omega in steradians to theta in radians for
    a cone section of a sphere.
    """
    return np.arccos(1 - omega / (2 * np.pi)) * u.rad

def lonlat_to_thetaphi(lon, lat):
    lon = lon.to(u.rad) if isinstance(lon, u.Quantity) else (lon * u.deg).to(u.rad)
    lat = lat.to(u.rad) if isinstance(lat, u.Quantity) else (lat * u.deg).to(u.rad)
    theta = Angle((np.pi/2 * u.rad) - lat)
    phi = Angle(lon)
    return theta, phi

def thetaphi_to_lonlat(theta, phi):
    phi = lon.to(u.rad) if isinstance(phi, u.Quantity) else (phi * u.rad)
    theta = lat.to(u.rad) if isinstance(theta, u.Quantity) else (theta * u.rad)
    lon = Angle(phi).to(u.deg)
    lat = Angle((np.pi/2 * u.rad) - theta).to(u.deg)
    return lon, lat

"""
DIPOLE-Y THINGS
"""
def dipole_dir_to_comps(direction):
    return hp.dir2vec(np.pi/2 - direction.dec.rad, direction.ra.rad)

def dipole_comps_to_dir(comps):
    return hp.vec2dir()

def a1ms_to_dipole_comps(a1ms):
    """
    Inputs assumed to be in order (a_{1,-1}, a_{1,0}, a_{1,1})
    (like as output by multipoles fitting functions).
    """
    assert len(a1ms) == 3
    Dx, Dy, Dz = [
        np.sqrt(3 / 4 * np.pi) * a1ms[i] for i in [2, 0, 1]     # a10 corresponds to z
    ]
    return np.array([Dx, Dy, Dz])

def dipole_comps_to_a1ms(comps):
    """
    Inputs assumed to be in order (D_x, D_y, D_z)
    (like as output by dipole fitting functions including `healpy.fit_dipole()`)
    """
    assert len(comps) == 3
    a1ms = [
        2 * np.sqrt(np.pi / 3) * Di for Di in comps
    ]
    return a1ms

def C1_from_D(D):   # from Gibelyou & Huterer (2012)
    return 4 * np.pi / 9 * D**2

def D_from_C1(C1):
    return np.sqrt(C1 * 9 / (4 * np.pi))

"""
FILE MANAGEMENT
"""
def filter_max_mocks(fns, max_mocks=None):
    """
    Returns a list of mock files, given a mock `case_dict` and parent directory `dir_mocks`.
    If `max_mocks` is not `None`, only returns the first `max_mocks` trials that match the
    case_dict.

    """
    if max_mocks is not None:
        fns_to_return = []
        itrial = 0
        nmocks = 0
        while nmocks < max_mocks:
            matches = [x for x in fns if f'trial{itrial:03d}' in x]
            if len(matches) == 1:
                fns_to_return.append(matches[0])
                nmocks += 1
            else:
                assert not matches, f"error: need 1 match but found {len(matches)}"
            itrial += 1
            if itrial > len(fns):
                break
    else:
        fns_to_return = fns

    return fns_to_return


"""
ABC
"""
def get_kde_1d(history, prior, parameter):

    df, w = history if type(history) == list else history.get_distribution()
    return pyabc.visualization.kde.kde_1d(pd.concat((df[parameter],), axis=1), w, df[parameter].name,
                           xmin=prior[parameter][0],
                           xmax=prior[parameter][0] + prior[parameter][1])


def get_kde_2d(history, prior, parameter1, parameter2):

    df, w = history if type(history) == list else history.get_distribution()
    return pyabc.visualization.kde.kde_2d(pd.concat((df[parameter1], df[parameter2]), axis=1), w, df[parameter1].name, df[parameter2].name,
                            xmin=prior[parameter1][0],
                            xmax=prior[parameter1][0] + prior[parameter1][1],
                            ymin=prior[parameter2][0],
                            ymax=prior[parameter2][0] + prior[parameter2][1])


def scatter(history, prior, parameter1, parameter2, ax, **kwargs):

    df, w = history if type(history) == list else history.get_distribution()
    ax.scatter(df[parameter1], df[parameter2], **kwargs)
    ax.set_xlim(prior[parameter1][0],
                prior[parameter1][0] + prior[parameter1][1])
    ax.set_ylim(prior[parameter2][0],
                prior[parameter2][0] + prior[parameter2][1])


def plot_posterior(history, prior, title=None, true_dipamp=0.0052, true_log_excess=None):

    # plot (copied and adjusted from the pyabc.visualization source code)
    par_ids = [x for x in prior.keys()]
    fig, axs = plt.subplots(len(par_ids), len(par_ids), figsize=(10,9), tight_layout=True)

    for i, par_id in enumerate(par_ids):

        # diagonal
        ax = axs[i, i]
        x, pdf = get_kde_1d(history, prior, par_id)
        ax.plot(x, pdf, c='k')
        if par_id == 'log_excess' and true_log_excess is not None:
            ax.axvline(true_log_excess, c='b', alpha=0.5)
        ax.grid(alpha=0.5, lw=0.5)
        if par_id == 'dipole_amp' and true_dipamp is not None:
            ax.axvline(true_dipamp, c='b', alpha=0.5)

        axs[i,0].set_ylabel(par_id)
        axs[len(par_ids)-1,i].set_xlabel(par_id)

        for j in range(0, i):

            # lower
            ax = axs[i, j]
            x, y, pdf = get_kde_2d(history, prior, par_ids[j], par_id)
            mesh = ax.pcolormesh(x, y, pdf, shading='auto')

            # upper
            ax = axs[j, i]
            scatter(history, prior, par_id, par_ids[j], ax, color='k', alpha=0.8, marker='.', s=7)
            ax.grid(alpha=0.5, lw=0.5)
        
    title = 'ABC posteriors' if title is None else title
    fig.suptitle(title)


def generate_mocks_from_prior(prior, model, nmocks, nside, **model_args):

    # draw mocks from the prior
    mocks_prior = []
    for i in range(nmocks):
        # randomly draw from the prior
        pars = {}
        for key in prior.keys():
            pars[key] = np.random.uniform(prior[key][0], prior[key][0]+prior[key][1])
        mocks_prior.append(model(pars, **model_args)['data'])
    
    return mocks_prior


def compare_mocks(posterior_dir, prior, nmocks, qmap, selfunc, nside, resdir='/scratch/aew492/lss-dipoles_results/results'):

    mock_dict = get_post_prior_mocks(posterior_dir, prior, nmocks, selfunc, nside, resdir)

    # downgrade data
    qmap_dg = hp.ud_grade(qmap, nside, power=-2)

    # differences
    res_post = [
        mock - qmap for mock in mock_dict['mocks_post']
    ]
    res_prior = [
        mock - qmap for mock in mock_dict['mocks_prior']
    ]
    res_post_dg = [
        mock - qmap_dg for mock in mock_dict['mocks_post_dg']
    ]
    res_prior_dg = [
        mock - qmap_dg for mock in mock_dict['mocks_prior_dg']
    ]

    return dict(res_post=res_post, res_prior=res_prior, res_post_dg=res_post_dg, res_prior_dg=res_prior_dg)


"""QUANTILES"""
# copied from corner.py — thank you!!
def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()