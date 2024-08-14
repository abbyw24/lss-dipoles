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

"""
HEALPIX FUNCTIONS
"""
def load_catalog_as_map(catalog, frame='icrs', NSIDE=64, dtype=float):
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
    pix_idx = hp.ang2pix(NSIDE, lon, lat, lonlat=True)
    hpmap = np.bincount(pix_idx, minlength=hp.nside2npix(NSIDE))
    return hpmap.astype(dtype)

def get_galactic_plane_mask(blim, NSIDE=64, frame='icrs'):
    """Returns a HEALPix mask (1s and 0s) around the galactic plane given an absolute b (latitude) limit."""
    lon, lat = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    b = SkyCoord(lon * u.deg, lat * u.deg, frame='icrs').galactic.b
    gal_plane_mask = np.zeros(hp.nside2npix(NSIDE))
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
    lon = lon.to(u.rad) if isinstance(lon, u.Quantity) else (lon * u.deg).to(u.rad)
    lat = lat.to(u.rad) if isinstance(lat, u.Quantity) else (lat * u.deg).to(u.rad)
    theta = Angle((np.pi/2 * u.rad) - lat)
    phi = Angle(lon)
    hp.newprojplot(theta, phi.wrap_at(np.pi * u.rad), **kwargs)

def label_coord(ax, coordsysstr):
    ax.text(0.86,
            0.05,
            coordsysstr,
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes)

def smooth_map(density_map, verbose=True):

    theta = omega_to_theta(1)  # 1 steradian

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

"""
DIPOLE-Y THINGS
"""
def dipole_dir_to_comps(direction):
    return hp.dir2vec(np.pi/2 - direction.dec.rad, direction.ra.rad)

def a1ms_to_dipole_comps(a1ms):
    """
    Inputs assumed to be in order (a_{1,-1}, a_{1,0}, a_{1,1})
    (like as output by multipoles fitting functions).
    """
    assert len(a1ms) == 3
    a1x = np.sqrt(3 / (4 * np.pi)) * a1ms[2]
    a1y = np.sqrt(3 / (4 * np.pi)) * a1ms[0]
    a1z = np.sqrt(3 / (4 * np.pi)) * a1ms[1]
    return np.array([a1x, a1y, a1z])

def C1_from_D(D):   # from Gibelyou & Huterer (2012)
    return 4 * np.pi / 9 * D**2

def D_from_C1(C1):
    return np.sqrt(C1 * 9 / (4 * np.pi))