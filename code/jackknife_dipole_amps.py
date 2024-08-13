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
from dipole_object import DipoleObject
from dipole import fit_dipole
from jackknife import jackknife_dipole_amps


def dipole_uncertainty_jk(t, nsamples=12, density_key='elatdenscorr'):
    """
    Compute the jackknife variance in the measured dipole, given an input healpix table.
    """

    lonavg, subsamples = get_longitude_subsamples(t, nsamples, NSIDE=64, density_key=density_key)

    # measure best-fit monopole and dipole amplitudes in each LOO sample
    bestfit_pars = np.array([
        fit_dipole(hpmap, Cinv=None, fit_zeros=False, idx=~np.isnan(hpmap))[0] for hpmap in subsamples
    ])

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
    return np.sqrt(np.diag(covar_jackknife(dipole_amp))), np.sqrt(np.diag(covar_jackknife(angular_dist)))


def main():

    """
    CATWISE
    """
    # # galactic plane cuts
    # d = DipoleObject(initial_catfn='catwise_agns_master_masks.fits', 
    #               catname='catwise_agns', mag='w1', maglim=16.4, blim=15)

    # print("galactic plane cuts", flush=True)
    # blims = np.arange(15, 71, 5)
    # stds_blims = np.empty((len(blims),2))
    # for i, blim in enumerate(blims):
    #     d.blim = blim
    #     t = Table.read(os.path.join(d.catdir,
    #                     f'choices/{d.catname}_{d.mag}{d.maglim:.1f}_blim{d.blim:.0f}/hpx_masked_final_elatdenscorr.fits'))
    #     std_amp, std_dir = dipole_uncertainty_jk(t)
    #     print(std_amp, std_dir)
    #     stds_blims[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((blims[:,np.newaxis], stds_blims), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_blims_2023-12-28.npy', save_arr)

    # masking radii
    d = DipoleObject(initial_catfn='catwise_agns_master_masks.fits', 
                  catname='catwise_agns', mag='w1', maglim=16.8, blim=30)
    
    d.cut_mag()
    d.cut_galactic_plane()

    print("masking radii", flush=True)
    factors = np.array([0., 0.2, 0.5, 1., 1.5, 2., 2.5])
    stds_masks = np.empty((len(factors),2))
    for i, factor in enumerate(factors):
        print(factor, flush=True)
        d.mask_fn = f'/scratch/aew492/quasars/catalogs/masks/mask_master_hpx_r{factor:.1f}.fits' if factor > 0 else None
        
        std_amp, std_dir = dipole_uncertainty_jk(d)
        print(std_amp, std_dir)
        stds_masks[i] = (std_amp, std_dir)
    
    save_arr = np.concatenate((factors[:,np.newaxis], stds_masks), axis=1)
    np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_masks.npy', save_arr)

    # # magnitude limits
    # d = DipoleObject(initial_catfn='catwise_agns_master_masks_w116.8.fits', 
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

    # # galactic plane cuts
    # d = DipoleObject(initial_catfn='quaia_G20.5.fits', 
    #               catname='quaia', mag='G', maglim=20., blim=15, compcorrect=True)
    # d.cut_mag()
    # d.cut_galactic_plane()

    # print("galactic plane cuts", flush=True)
    # blims = np.arange(15, 71, 5)
    # stds_blims = np.empty((len(blims),2))
    # for i, blim in enumerate(blims):
    #     print(blim, flush=True)
    #     d.blim = blim
    #     d.cut_galactic_plane()
    #     std_amp, std_dir = dipole_uncertainty_jk(d)
    #     print(std_amp, std_dir)
    #     stds_blims[i] = (std_amp, std_dir)
    
    # save_arr = np.concatenate((blims[:,np.newaxis], stds_blims), axis=1)
    # np.save(f'/scratch/aew492/quasars/catalogs/{d.catname}/choices/jackknife_uncertainties_blims.npy', save_arr)

    # del d

    # # magnitude limit
    # d = DipoleObject(initial_catfn='quaia_G20.5.fits', 
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
    # d = DipoleObject(initial_catfn='quaia_G20.5.fits', 
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