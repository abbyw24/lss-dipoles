"""

MEASURE THE DIPOLE UNCERTAINTY IN THE CATWISE2020 AGN / QUAIA SAMPLES
using a jackknife approach.

Measure the dipole n times, dividing the sample into n longitudinal wedges
and leaving out one wedge for each measurement.

"""
import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import random
import os
import sys
sys.path.insert(0, '/home/aew492/lss-dipoles')
from dipole import fit_dipole


def covar_jackknife(Y):
    """
    Return the jackknife variance of a data set `Y`, where each row in `Y` is a jackknife replicant.
    Implemented from Eq. 38 of Hogg & Villar (2021) (2101.07256).
    """
    n = len(Y)
    if Y.ndim == 1:
        Y = Y[:,np.newaxis]
    # mean across the subsamples
    Y_avg = np.nanmean(Y, axis=0)
    prefactor = (n-1) / n
    # need to pad (Y[i]-Y_avg) with an extra dimension to make 2D, otherwise @ just performs dot product
    X = np.array([
        (Y[i] - Y_avg)[...,None] @ (Y[i] - Y_avg)[...,None].T for i in range(n)
    ])
    return prefactor * np.sum(X, axis=0)


def compute_jackknife_uncertainty(subsamples, func, return_outputs=False, **kwargs):
    """
    Compute the uncertainty on a measurement using jackknife resampling.

    Parameters
    ----------
    subsamples : The data subsamples to use in the fit.

    func : The function that returns the measurement for which to estimate the uncertainty.
        Must take data (a subsample) followed by `kwargs` as inputs, and output an ndarray for
        which to estimate the uncertainty.

    kwargs : Keyword arguments to input to `func`.

    Returns
    -------
    std : The jackknife uncertainty on the output of `func`.

    """

    # run the function on each subsample
    outputs = np.array([
        func(subsample, **kwargs) for subsample in subsamples
    ])

    # covariance matrix
    covar = covar_jackknife(outputs)

    # uncertainty: square root of the variance (diagonal terms)
    std = np.sqrt(np.diag(covar))

    if return_outputs == True:
        return std, outputs
    else:
        return std


def get_longitude_subsamples_from_hpmap(hpmap, nsamples, input_frame='icrs'):
    """
    Return healpix subsamples of a healpix map:
    divides the sky into equal (galactic) longitude wedges and leaves out one wedge
    in each subsample.

    Parameters
    ----------
    hpmap : healpix map
        Healpix map to divide into subsamples.
    nsamples : int
        Number of subsamples to construct.
    input_frame : str, optional
        Sky coordinate system of the input map. Default is ICRS.
    
    Returns
    -------
    lonavg : The average galactic longitude of the wedge left out in each subsample.

    subsamples : A (nsamples,NPIX) array of healpix maps of the LOO subsamples.

    """
    # get NPIX from length of the input map
    NPIX = len(hpmap)

    # longitude bins
    lonedges = np.linspace(0, 360, nsamples+1) << u.deg
    lonavg = 0.5 * (lonedges[1:] + lonedges[:-1])

    # get the central sky coordinate of each healpixel
    lon, lat = hp.pix2ang(hp.npix2nside(NPIX), np.arange(NPIX), lonlat=True)
    sc = SkyCoord(lon * u.deg, lat * u.deg, frame=input_frame).galactic

    # construct subsamples
    subsamples = np.empty((nsamples, NPIX))
    # in each longitude bin, construct the LOO sample, and run the function
    for i in range(nsamples):
        # galactic coordinates: get pixel indices of each slice
        idx_to_cut = (sc.l >= lonedges[i] - 1*u.deg) & (sc.l<<u.deg < lonedges[i+1] + 1*u.deg)
        # make healpix map
        subsample = np.full(NPIX, np.nan)
        subsample[~idx_to_cut] = hpmap[~idx_to_cut]
        subsamples[i] = subsample

    return lonavg, subsamples


def get_longitude_subsamples_from_hptable(t, nsamples, NSIDE=64, density_key='elatdenscorr'):
    """
    Return healpix subsamples of a density map from a healpix table `t`:
    divides the sky into equal (galactic) longitude wedges and leaves out one wedge
    in each subsample.

    Parameters
    ----------
    t : astropy.table.Table
        Table containing the all-sky sample. Must contain columns 'l' for the
        galactic longitude and 'hpidx' for the healpix index of each entry.
    nsamples : int
        Number of subsamples to construct.
    NSIDE : int, optional
        Resolution of the healpix map.
    density_key : str, optional
        Column key for the source density in each healpixel.
    
    Returns
    -------
    lonavg : The average galactic longitude of the wedge left out in each subsample.

    subsamples : A (nsamples,NPIX) array of healpix maps of the LOO subsamples.

    """

    assert 0 <= np.all(t['l']) <= 360
    NPIX = hp.nside2npix(NSIDE)

    # longitude bins
    lonedges = np.linspace(0, 360, nsamples+1) << u.deg
    lonavg = 0.5 * (lonedges[1:] + lonedges[:-1])

    subsamples = np.empty((nsamples, NPIX))
    # in each longitude bin, construct the LOO sample, and run the function
    for i in range(nsamples):
        # galactic coordinates: get table indices of pixels in each slice
        idx_to_cut = (t['l']<<u.deg >= lonedges[i] - 1*u.deg) & (t['l']<<u.deg < lonedges[i+1] + 1*u.deg)
        # get all pixels except those in this longitude wedge
        t_subsample = t[~idx_to_cut]
        # turn into healpix map
        subsample = np.empty(NPIX)
        subsample[:] = np.nan
        subsample[t_subsample['hpidx']] = t_subsample[density_key]
        subsamples[i] = subsample

    return lonavg, subsamples