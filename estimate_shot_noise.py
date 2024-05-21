import numpy as np
import healpy as hp
# from mpi4py import MPI
import multiprocessing as mp
import astropy.units as u
import random
import time
import datetime
import os
import sys

import tools
from qso_sample import QSOSample
from multipoles import compute_Cells_in_overdensity_map

def main():

    s = time.time()

    """INPUT PARAMETERS"""
    nside = 64
    max_ell = 5
    ntrials = 20
    Wmasks = np.logspace(-2,0,10)

    # instantiate a dipole object
    print(f"creating dipole object")
    catdir = '/scratch/aew492/quasars/catalogs'
    # d = QSOSample(initial_catfn=os.path.join(catdir, 'catwise_agns/catwise_agns_master.fits'),
    #                     mask_fn=os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits'),
    #                     mag='W1', maglim=16.4,
    #                     blim=30,
    #                     NSIDE=nside)
    d = QSOSample(initial_catfn=os.path.join(catdir, 'quaia/quaia_G20.5.fits'),
                    mask_fn=os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits'),
                    mag='G', maglim=20.,
                    blim=30,
                    NSIDE=nside)
    d.cut_galactic_plane()
    d.cut_mag()

    # selection function
    selfunc_fn = os.path.join(catdir, f'quaia/selfuncs/selection_function_NSIDE{nside}_G20.0_blim15.fits')
    # selfunc_fn = os.path.join(catdir, f'catwise_agns/selfuncs/selection_function_NSIDE{nside}_catwise_pluszodis.fits')
    selfunc = d.get_selfunc(selfunc=selfunc_fn)

    # where to save
    save_dir = os.path.join(f'/scratch/aew492/quasars/noise_Cells/quaia')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    def save_fn(tag):  # tag to specify either full or cut sky
        return os.path.join(save_dir, f'noise_Cells_{tag}_ellmax{int(max_ell)}_{ntrials}trials.npy')

    # MULTIPROCESSING
    nprocesses = 10 # os.cpu_count()
    print(f"using {nprocesses} processes", flush=True)

    """GET MASK AND MEAN SOURCE DENSITY"""
    # get masked datamap
    masked_datamap = d.construct_masked_datamap(selfunc=selfunc, return_map=True)
    # mask and mean density
    mask = d.mask
    mu = np.nanmean(masked_datamap)
    print(f"mean density in the masked map = {mu:.2f} quasars per healpixel", flush=True)

    # comment to save along with results
    comment = 'Quaia G<20.0, masking completeness < 0.5; ' + \
                'correcting by completeness to estimate mean source density mu, but NOT weighting by completeness in lstsq'

    """
    COMPUTE ANGULAR POWER SPECTRA
    """
    # """FULL SKY"""
    # print(f"computing Cells on the full sky", flush=True)
    # pool = mp.Pool(processes=nprocesses)
    # Cells_fullsky_trials = np.array([
    #     pool.apply(compute_Cells_fullsky, args=(i, mu, nside, max_ell,)) for i in range(ntrials)
    # ])
    # Cells_fullsky = np.nanmean(Cells_fullsky_trials, axis=0)
    # print(Cells_fullsky.shape, Cells_fullsky)

    # results_dict = dict(ells=np.arange(1, max_ell+1),
    #                     Cells_fullsky=Cells_fullsky,
    #                     ntrials=ntrials,
    #                     selfunc_fn=selfunc_fn,
    #                     mu=mu,
    #                     comment=comment)
    # np.save(save_fn('fullsky'), results_dict)
    # print(f"saved to {save_fn('fullsky')}", flush=True)


    """CUT SKY"""
    # define function (for multiprocessing) to compute Cells on a cut sky as a function of Wmask
    def compute_Cells_cutsky(index, Wmask, results_dict):
        Cells_this_Wmask_trials = np.empty((ntrials, max_ell))
        for i in range(ntrials):
            print(f"Wmask = {Wmask:.2e}: trial {i+1} of {ntrials}")
            # noise realization for this trial, MASKED, with the mean density, on the full sky
            noise = np.random.poisson(mu, hp.nside2npix(nside)).astype(float)
            noise[~mask] = np.nan
            # map to fit is the overdensities
            map_to_fit = noise / np.nanmean(noise) - 1
            # fit to spherical harmonics templates
            ells, Cells_ = compute_Cells_in_overdensity_map(map_to_fit, Wmask=Wmask, max_ell=int(max_ell))
            Cells_this_Wmask_trials[i] = Cells_[1:]  # cut out monopole since we're fitting overdensities
        results_dict[index] = (Wmask, np.nanmean(Cells_this_Wmask_trials, axis=0))

    # multiprocessing Process
    print(f"computing Cells on the cut sky...", flush=True)
    s = time.time()
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
                        Cells_cutsky=cutsky_results.values(),
                        selfunc_fn=selfunc_fn,
                        mu=mu,
                        comment=comment)
    np.save(save_fn('cutsky'), results_dict)
    print(f"saved to {save_fn('cutsky')}", flush=True)

    total_time = time.time()-s 
    print(f"total time = {datetime.timedelta(seconds=total_time)}")


def compute_Cells_fullsky(itrial, mu, nside, max_ell, verbose=True):
    if verbose:
        print(f"starting trial {int(itrial+1)}", flush=True)
    # noise realization for this trial, with the mean density, on the full sky
    noise = np.random.poisson(mu, hp.nside2npix(int(nside))).astype(float)
    # map to fit is the overdensities
    map_to_fit = noise / np.mean(noise) - 1
    assert np.sum(np.isnan(map_to_fit)) == 0.  # in this case Wmask is not used !
    # fit to spherical harmonics templates
    ells, Cells_fullsky = compute_Cells_in_overdensity_map(map_to_fit, Wmask=0., max_ell=int(max_ell))
    return Cells_fullsky[1:]  # cut out monopole since we're fitting overdensities


def compute_Cells_fullsky_wrapper(result_list, itrial, mu, nside, max_ell, verbose=True):

    result_list.append(compute_Cells_fullsky(itrial, mu, nside, max_ell, verbose))


if __name__ == '__main__':
    main()