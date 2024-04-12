import numpy as np
import healpy as hp
# from mpi4py import MPI
import multiprocessing as mp
import fitsio
from astropy.coordinates import SkyCoord
import astropy.units as u
import random
import time
import datetime
import os
import sys

import tools
from Secrest_dipole import SecrestDipole
from multipoles import compute_Cells_from_alms_fit
import dipole
from jackknife import get_longitude_subsamples, compute_jackknife_uncertainty

def main():

    s = time.time()

    print(mp.cpu_count(), flush=True)

    """INPUT PARAMETERS"""
    nside = 64
    max_ell = 8
    ntrials = 20
    Wmasks = np.logspace(-2,0,10)
    # catalog_dict = dict(initial_catfn='catwise_agns_master.fits',
    #                     catname='catwise_agns',
    #                     mag='w1',
    #                     blim=30,
    #                     maglim=16.4,
    #                     save_tag='_r1.0',
    #                     load_init=True,
    #                     NSIDE=nside)
    catalog_dict = dict(initial_catfn='quaia_G20.0.fits',
                        catname='quaia',
                        mag='G',
                        blim=30,
                        maglim=20.,
                        save_tag='_r1.0',
                        load_init=True,
                        compcorrect=False,
                        NSIDE=nside)

    # where to save
    save_dir = os.path.join(f'/scratch/aew492/quasars/noise_Cells', catalog_dict['catname'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = os.path.join(save_dir, f'noise_Cells_ellmax{int(max_ell)}_{ntrials}trials.npy')

    """DATA"""
    # instantiate a dipole object
    print(f"creating dipole object")
    d = SecrestDipole(**catalog_dict)
    d.cut_mag()
    # mean density
    mask, mu = get_mask_and_mean_density(d)
    print(f"mean density in the masked map = {mu:.2f} quasars per healpixel", flush=True)


    """COMPUTE ANGULAR POWER SPECTRA"""
    # measure Cells on the full sky
    print(f"computing Cells on the full sky", flush=True)
    Cells_fullsky_trials = np.empty((ntrials, max_ell))
    for i in range(ntrials):
        print(f"trial {i+1} of {ntrials}\t") #, end='\r')
        # noise realization for this trial, with the mean density, on the full sky
        noise = np.random.poisson(mu, hp.nside2npix(nside)).astype(float)
        # map to fit is the overdensities
        map_to_fit = noise / np.nanmean(noise) - 1
        # fit to spherical harmonics templates
        ells, Cells_fullsky = compute_Cells_from_alms_fit(map_to_fit, np.ones_like(map_to_fit), max_ell)
        Cells_fullsky_trials[i] = Cells_fullsky[1:]  # cut out monopole since we're fitting overdensities
    Cells_fullsky = np.mean(np.array(Cells_fullsky_trials), axis=0)

    # measure Cells on the cut sky

    # define function to compute Cells on a cut sky as a function of Wmask
    def compute_Cells_cutsky(i, Wmask, results_dict):
        Cells_this_Wmask_trials = np.empty((ntrials, max_ell))
        for i in range(ntrials):
            print(f"Wmask = {Wmask:.2e}: trial {i+1} of {ntrials}\t") #, end='\r')
            # noise realization for this trial, with the mean density, on the full sky
            noise = np.random.poisson(mu, hp.nside2npix(nside)).astype(float)
            # MASK the same pixels as in the data
            noise[~mask] = np.nan
            # map to fit is teh overdensities
            map_to_fit = noise / np.nanmean(noise) - 1
            map_to_fit[~mask] = 0.
            # Cinv: completeness (assumed perfect) in unmasked pixels, this Wmask in the masked pixels
            Cinv_ = np.ones_like(map_to_fit)
            Cinv_[~mask] = Wmask
            # fit to spherical harmonics templates
            ells, Cells_ = compute_Cells_from_alms_fit(map_to_fit, Cinv_, max_ell)
            Cells_this_Wmask_trials[i] = Cells_[1:]  # cut out monopole since we're fitting overdensities
        results_dict[i] = (Wmask, np.nanmean(Cells_this_Wmask_trials, axis=0))

    # multiprocessing pool
    manager = mp.Manager()
    procs = []
    cutsky_results = manager.dict()
    for i, Wmask in enumerate(Wmasks):
        proc = mp.Process(target=compute_Cells_cutsky, args=(i, Wmask, cutsky_results,))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()


    results_dict = dict(ells=np.arange(1, max_ell+1),
                        Cells_fullsky=Cells_fullsky,
                        Cells_cutsky=cutsky_results.values())
    np.save(save_fn, results_dict)

    total_time = time.time()-s 
    print(f"total time = {datetime.timedelta(seconds=total_time)}")


def get_mask_and_mean_density(d, nside=64, min_completeness=0.5):

    # construct map from source density table
    datamap = tools.load_catalog_as_map(d.table, NSIDE=nside)

    # delete the tables to clear up memory !
    del d.initial_cattab
    del d.table

    # mask the galactic plane (mask pixels based on their central sky coordinate)
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    b = SkyCoord(lon * u.deg, lat * u.deg, frame='icrs').galactic.b
    gal_plane_mask = np.zeros_like(datamap)
    gal_plane_mask[np.abs(b.deg) >= d.blim] = 1

    # load smaller masks (used in S21): at the pixel level
    small_masks = fitsio.read(d.mask_fn)
    assert len(small_masks) == len(gal_plane_mask) == len(datamap)

    # combine these two into a single mask
    mask = gal_plane_mask.astype(bool) & small_masks.astype(bool)

    # completeness in each pixel
    if d.catname == 'quaia':
        # and load selection function
        completeness = d.load_selfunc()
    else:
        assert d.catname == 'catwise_agns', "input `catname` must be 'quaia' or 'catwise_agns'"
        # if selfunc is not provided, assume perfect completeness
        completeness = np.ones_like(datamap)

    # mask the data: populate a nan map with the data values in unmasked pixels
    masked_datamap = np.multiply(datamap, mask, where=(mask!=0), out=np.full_like(datamap, np.nan))

    # mean density of the map
    final_mask = (completeness > min_completeness) & (mask != 0.)
    mu = np.nansum((masked_datamap * completeness)[final_mask]) / np.nansum((completeness * completeness)[final_mask])

    return final_mask, mu


if __name__ == '__main__':
    main()