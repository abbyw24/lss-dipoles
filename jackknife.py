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
from Secrest_dipole import SecrestDipole


def var_jackknife(Y):
    """
    Return the jackknife variance of a data set `Y`, where each row in `Y` is a jackknife replicant.
    """
    n = len(Y)
    if Y.ndim == 1:
        Y = Y[:,np.newaxis]
    Y_avg = np.nanmean(Y, axis=0)
    prefactor = (n-1) / n
    X = np.array([
        (Y[i] - Y_avg) @ (Y[i] - Y_avg).T for i in range(n)
    ])
    return prefactor * np.sum(X, axis=0)


def measure_dipole(d, t, loncut=None):
    """
    Measure the dipole amplitudes from a source table and a `SecrestDipole` object.
    """
    # converts source table to a table of healpix densities with NPIX rows
    t = d._make_healpix_map(t)
    # mask healpixels with the bright galaxy + image artifacts masks from Secrest
    t = d._mask_initial_healpix_map(map_=t)
    # make extra galactic plane cut (to avoid underdensities in pixels that land on the boundary) 
    t = t[np.abs(t['b']) > (d.blim + 1)]
    if loncut is not None:
        # ** also make extra longitude cut now that we made a sharp cut at the source level
        idx_to_cut = (t['l']<<u.deg >= loncut[0] - 1*u.deg) & (t['l']<<u.deg < loncut[1] + 1*u.deg)
        t = t[~idx_to_cut]
    # COMPLETENESS CORRECT
    if d.compcorrect:
        t = d._completeness_correct(t, key='density')
    # density correct: fit a line to density vs. ecliptic latitude
    t = d._hpx_vs_direction(tab=t)
    # measure the dipole: this is a wrapper for my lstsq function and matches healpy.fit_dipole when Cinv==None
    #  * this also adds the "raw" output amps (monopole + 3 dipole amps) as an attribute
    _, _ = d.compute_dipole(t, Cinv=None)
    return d.amps


def dipole_uncertainty_jk(d, nsamples=12):
    """
    Compute the jackknife variance in the measured dipole, given an input source table `t`.
    """

    t = d.table

    # longitude bins
    lonedges = np.linspace(0, 360, nsamples+1) << u.deg
    lonavg = 0.5 * (lonedges[1:] + lonedges[:-1])

    # galactic coordinates: get table indices of sources in each slice
    assert 0 <= np.all(t['l']) <= 360
    idx_to_drop = [
        ((t['l'] >= lonedges[i]) & (t['l'] < lonedges[i+1])) for i in range(nsamples)
    ]

    # measure dipole in each LOO sample
    bestfit_pars = [
        measure_dipole(d, t[~idx], loncut=lonedges[i:i+2]) for i, idx in enumerate(idx_to_drop)
    ]

    # dipole comps: components scaled by the monopole
    dipole_comps = np.array([
        pars[1:] / pars[0] for pars in bestfit_pars
    ])

    # compute the amplitude because we want the error bar on the amplitude
    # amplitude of the dipole (dimensionless)
    dipole_amp = np.array([
        np.linalg.norm(comps) for comps in dipole_comps
    ])

    # compute the angular offsets because we want the error bar on the direction
    avg_comps = np.nanmean(dipole_comps, axis=0)
    avg_comps_norm = avg_comps / np.sqrt(avg_comps @ avg_comps)
    angular_dist = np.rad2deg(np.array([
        np.arccos((comps @ avg_comps_norm) / np.sqrt(comps @ comps)) for comps in dipole_comps
    ])) # in degrees

    # finally, return the jackknife /standard deviation/ -> sqrt(variance)
    return np.sqrt(var_jackknife(dipole_amp)), np.sqrt(var_jackknife(angular_dist))


def main():

    """
    CATWISE
    """
    # # galactic plane cuts
    # d = SecrestDipole(initial_catfn='catwise_agns_master_masks.fits', 
    #               catname='catwise_agns', mag='w1', maglim=16.4, blim=15)
    # d.cut_mag()
    # d.cut_galactic_plane()

    # print("galactic plane cuts", flush=True)
    # blims = np.arange(15, 71, 5)
    # stds_blims = np.empty((len(blims),2))
    # for i, blim in enumerate(blims):
    #     d.blim = blim
    #     d.cut_galactic_plane()
    #     std_amp, std_dir = dipole_uncertainty_jk(d)
    #     print(std_amp, std_dir)
    #     stds_blims[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((blims[:,np.newaxis], stds_blims), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_blims.npy', save_arr)

    # # masking radii
    # d = SecrestDipole(initial_catfn='catwise_agns_master_masks.fits', 
    #               catname='catwise_agns', mag='w1', maglim=16.8, blim=30)
    
    # d.cut_mag()
    # d.cut_galactic_plane()

    # print("masking radii", flush=True)
    # factors = np.array([0., 0.2, 0.5, 1., 1.5, 2., 2.5])
    # stds_masks = np.empty((len(factors),2))
    # for i, factor in enumerate(factors):
    #     print(factor, flush=True)
    #     d.mask_fn = f'/scratch/aew492/quasars/catalogs/masks/mask_master_hpx_r{factor:.1f}.fits' if factor > 0 else None
    #     std_amp, std_dir = dipole_uncertainty_jk(d)
    #     print(std_amp, std_dir)
    #     stds_masks[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((factors[:,np.newaxis], stds_masks), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_masks.npy', save_arr)

    # # magnitude limits
    # d = SecrestDipole(initial_catfn='catwise_agns_master_masks_w116.8.fits', 
    #               catname='catwise_agns', mag='w1', maglim=16.8, blim=30)
    # d.cut_mag()
    # d.cut_galactic_plane()

    # print("magnitude limit")
    # maglims = np.arange(15.7, 16.81, 0.1)[::-1]
    # stds_maglims = np.empty((len(maglims),2))
    # for i, maglim in enumerate(maglims):
    #     print(maglim)
    #     d.maglim = maglim
    #     d.cut_mag()
    #     std_amp, std_dir = dipole_uncertainty_jk(d)
    #     print(std_amp, std_dir)
    #     stds_maglims[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((maglims[:,np.newaxis], stds_maglims), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_maglims.npy', save_arr)

    """
    QUAIA
    """

    # galactic plane cuts
    d = SecrestDipole(initial_catfn='quaia_G20.5.fits', 
                  catname='quaia', mag='G', maglim=20., blim=15, compcorrect=True)
    d.cut_mag()
    d.cut_galactic_plane()

    print("galactic plane cuts", flush=True)
    blims = np.arange(15, 71, 5)
    stds_blims = np.empty((len(blims),2))
    for i, blim in enumerate(blims):
        print(blim, flush=True)
        d.blim = blim
        d.cut_galactic_plane()
        std_amp, std_dir = dipole_uncertainty_jk(d)
        print(std_amp, std_dir)
        stds_blims[i] = (std_amp, std_dir)
    
    save_arr = np.concatenate((blims[:,np.newaxis], stds_blims), axis=1)
    np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_blims.npy', save_arr)

    # del d

    # # magnitude limit
    # d = SecrestDipole(initial_catfn='quaia_G20.5.fits', 
    #               catname='quaia', mag='G', maglim=20.5, blim=30, compcorrect=True)
    # d.cut_mag()
    # d.cut_galactic_plane()

    # print("magnitude limit", flush=True)
    # maglims = np.arange(19., 20.51, 0.1)[::-1]
    # stds_maglims = np.empty((len(maglims),2))
    # for i, maglim in enumerate(maglims):
    #     print(maglim, flush=True)
    #     d.maglim = maglim
    #     d.cut_mag()
    #     std_amp, std_dir = dipole_uncertainty_jk(d)
    #     print(std_amp, std_dir)
    #     stds_maglims[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((maglims[:,np.newaxis], stds_maglims), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_maglims.npy', save_arr)

    # del d

    # # masking radii
    # d = SecrestDipole(initial_catfn='quaia_G20.5.fits', 
    #               catname='quaia', mag='G', maglim=20., blim=30, compcorrect=True)
    # d.cut_mag()
    # d.cut_galactic_plane()

    # print("masking radii", flush=True)
    # factors = np.array([0., 0.2, 0.5, 1., 1.5, 2., 2.5])
    # stds_masks = np.empty((len(factors),2))
    # for i, factor in enumerate(factors):
    #     print(factor, flush=True)
    #     d.mask_fn = f'/scratch/aew492/quasars/catalogs/masks/mask_master_hpx_r{factor:.1f}.fits' if factor > 0 else None
    #     std_amp, std_dir = dipole_uncertainty_jk(d)
    #     print(std_amp, std_dir)
    #     stds_masks[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((factors[:,np.newaxis], stds_masks), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_masks.npy', save_arr)


if __name__=='__main__':
    main()