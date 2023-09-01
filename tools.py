"""
Helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.table import Table
from astropy.coordinates import SkyCoord
import os
import healpy as hp
from healpy.newvisufunc import projview


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


def get_galactic_mask(blim, NSIDE=64, frame='icrs'):
    """Returns a HEALPix mask around the galactic plane given an absolute b (latitude) limit."""
    NPIX = hp.nside2npix(NSIDE)
    lon, lat = hp.pix2ang(NSIDE, ipix=np.arange(NPIX), lonlat=True)
    coords = SkyCoord(lon, lat, frame=frame, unit='deg')
    idx_to_cut = (np.abs(coords.galactic.b.deg) < blim)
    galactic_mask = np.full(NPIX, True)
    galactic_mask[idx_to_cut] = False
    return galactic_mask


def flatten_map(sf_map):
    newarr = np.array([row[0] for row in sf_map])
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


def lstsq(Y, A, Cinv):
    """
    Return the least-squares solution to a linear matrix equation,
    given data Y, design matrix A, and inverse covariance Cinv.
    BUG:
    - This should catch warnings and errors in the `res` object.
    """
    if len(Cinv.shape)==1:
        a = A.T @ (Cinv[:,None] * A)
        b = A.T @ (Cinv * Y)
    else:
        a = A.T @ Cinv @ A
        b = A.T @ Cinv @ Y
    res = np.linalg.lstsq(a, b, rcond=None)
    return res[0], a