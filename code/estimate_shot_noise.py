import numpy as np
import healpy as hp
# from mpi4py import MPI
# import multiprocessing as mp
import astropy.units as u
import random
import time
import datetime
import os
import sys

import tools
from qso_sample import QSOSample
from multipoles import compute_Cells_in_overdensity_map_Lambda
from dipole import measure_dipole_in_overdensity_map_Lambda

def main():

    s = time.time()

    """ INPUT PARAMETERS """
    nside = 64
    max_ell = 1
    ntrials = 10000
    Lambdas = np.logspace(-3, 0, 10)

    # instantiate a dipole object
    print(f"creating dipole object", flush=True)
    catdir = '/scratch/aew492/quasars/catalogs'

    """ which sample? """
    sample = 'quaia'

    if sample == 'catwise_agns':
        d = QSOSample(initial_catfn=os.path.join(catdir, 'catwise_agns/catwise_agns_master.fits'),
                            mask_fn=os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits'),
                            mag='W1', maglim=16.4,
                            blim=30,
                            NSIDE=nside)
        selfunc_fn = os.path.join(catdir, f'catwise_agns/selfuncs/selection_function_NSIDE{nside}_catwise_pluszodis.fits')
        # comment to save along with results
        comment = f'CatWISE AGNs, masking completeness < 0.5; ' + \
                'correcting by completeness to estimate mean source density mu, but NOT weighting by completeness in lstsq'
    else:
        assert sample == 'quaia'
        d = QSOSample(initial_catfn=os.path.join(catdir, 'quaia/quaia_G20.5.fits'),
                        mask_fn=os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits'),
                        mag='G', maglim=20.,
                        blim=30,
                        NSIDE=nside)
        selfunc_fn = os.path.join(catdir, f'quaia/selfuncs/selection_function_NSIDE{nside}_G20.0_blim15.fits')
        # comment to save along with results
        comment = f'Quaia G<{d.maglim}, masking completeness < 0.5; ' + \
                'correcting by completeness to estimate mean source density mu, but NOT weighting by completeness in lstsq'
    # cut galactic plane, magnitude, and load selection function
    d.cut_mag()
    d.cut_galactic_plane_hpx()
    selfunc = d.get_selfunc(selfunc=selfunc_fn)

    # where to save
    save_dir = os.path.join(f'/scratch/aew492/quasars/noise_Cells', sample)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    """ GET MASK AND MEAN SOURCE DENSITY """
    # get masked datamap, uncorrected
    masked_datamap = d.construct_masked_datamap(selfunc=None, return_map=True)
    print(f"mean density in masked map, uncorrected = {np.nanmean(masked_datamap):.2f} quasars per healpixel", flush=True)
    masked_datamap_corr = masked_datamap / selfunc
    print(f"mean density in masked map / selfunc = {np.nanmean(masked_datamap_corr):.2f} quasars per healpixel", flush=True)
    # get masked datamap
    masked_datamap = d.construct_masked_datamap(selfunc=selfunc, return_map=True)
    # mask and mean density
    mask = d.mask
    print(f"mean density in masked map, corrected = {np.nanmean(masked_datamap):.2f} quasars per healpixel", flush=True)
    # expected number of quasars in each pixel
    expected_map = np.nanmean(masked_datamap) * selfunc
    print(f"mean density in expected map = {np.nanmean(expected_map):.2f} quasars per healpixel", flush=True)
    # mu: which mean to use for the Poisson draw
    mu = np.nanmean(expected_map)
    print(f"mu = {mu:.2f} quasars per healpixel", flush=True)
    assert False
    # print(mu.shape, np.sum(np.isnan(mu)))

    """ RUN """
    # which function to actually run:
    res = run_Cells_convergence(mu, ntrials, 1, comment=comment, compare_to_D=True, verbose=True)
    # res = run_dipole_convergence(mu, ntrials, comment=comment, compare_to_C1=False, verbose=True)

    """ SAVE """
    save_fn = os.path.join(save_dir, f'noise_Cells_fullsky_ellmax1_{ntrials}trials.npy')
    np.save(save_fn, res)

    total_time = time.time()-s 
    print(f"total time = {datetime.timedelta(seconds=total_time)}", flush=True)


"""
DIPOLE FUNCTIONS
"""
def run_dipole_convergence(mu, ntrials, nside=64, comment=None, verbose=False,
                            compare_to_C1=False):
    # final results to save: the dipole amplitude (norm of dipole vector) in each realization
    dipole_amps = np.empty(ntrials)  
    for i in range(ntrials):
        if i % 100 == 0 and verbose:
            print(f"trial {i} of {ntrials}", flush=True)
        dipole_comps = compute_dipole_in_noise_map(mu, nside, Lambda=0., compare_to_C1=compare_to_C1)
        dipole_amps[i] = np.linalg.norm(dipole_comps)
    results_dict = dict(mu=mu,
                        ntrials=ntrials,
                        dipole_amps=dipole_amps)
    if comment is not None:
        results_dict['comment'] = comment
    return results_dict

def run_dipole_cutsky(mu, mask, Lambdas, ntrials, nside=64, comment=None, verbose=False):
    # final results to save: the average dipole amplitude (norm of dipole vector) in each of 
    #   ntrials realizations, as well as the standard deviation across the realizations
    dipole_amps_Lambdas = np.empty(len(Lambdas))
    stds_Lambdas = np.empty(len(Lambdas))
    for j, Lambda in enumerate(Lambdas):
        amps_trials_ = np.empty(ntrials)
        for i in range(ntrials):
            if i % 10 == 0 and verbose:
                print(f"Lambda = {Lambda:.2e}: trial {i} of {ntrials}", flush=True)
            dipole_comps = compute_dipole_in_noise_map(mu, nside, Lambda=Lambda, mask=mask)
            amps_trials_[i] = np.linalg.norm(dipole_comps)
        dipole_amps_Lambdas[j] = np.nanmean(amps_trials_)
        stds_Lambdas[j] = np.nanstd(amps_trials_)
    results_dict = dict(mu=mu,
                        mask=mask,
                        Lambdas=Lambdas,
                        ntrials=ntrials,
                        dipole_amps=dipole_amps_Lambdas,
                        stds=stds_Lambdas)
    if comment is not None:
        results_dict['comment'] = comment
    return results_dict

def compute_dipole_in_noise_map(mu, nside, Lambda, mask=None, compare_to_C1=False):
    # noise realization
    map_to_fit = noise_overdensity_map(mu, nside, mask=mask)
    if compare_to_C1 == True:
        dipole_comps = measure_dipole_in_overdensity_map_Lambda(map_to_fit, Lambda=Lambda)
        ells, Cells, alms = compute_Cells_in_overdensity_map_Lambda(map_to_fit,
                                                                    Lambda=Lambda, max_ell=1, return_alms=True)
        # do the dipole components convert to the a1ms like we expect?
        assert np.allclose(dipole_comps, tools.a1ms_to_dipole_comps(alms[1:]))
        # do the sum of the squares of the a1ms match C1?
        assert np.allclose(np.sum(alms[1:]**2) / 3, Cells[1])
        # does the C1_from_D( norm of the dipole components ) match C1 ?
        assert np.allclose(Cells[1], tools.C1_from_D(np.linalg.norm(dipole_comps)))
    # fit dipole components
    return measure_dipole_in_overdensity_map_Lambda(map_to_fit, Lambda=Lambda)


"""
Cells FUNCTIONS
"""
def run_Cells_convergence(mu, ntrials, max_ell, nside=64, comment=None, verbose=False,
                            compare_to_D=False):
    # final results to save: the Cells measured in each noise realization
    Cells = np.empty((ntrials, max_ell)) 
    for i in range(ntrials):
        if i % 100 == 0 and verbose:
            print(f"trial {i} of {ntrials}", flush=True)
        Cells[i] = compute_Cells_in_noise_map(mu, nside, max_ell, Lambda=0., compare_to_D=compare_to_D)
    results_dict = dict(mu=mu,
                        ntrials=ntrials,
                        max_ell=max_ell,
                        Cells=Cells)
    if comment is not None:
        results_dict['comment'] = comment
    return results_dict

def run_Cells_cutsky(mu, mask, Lambdas, ntrials, max_ell, nside=64, comment=None, verbose=False):
    # final results to save: the average Cells in each of ntrials realizations,
    #   as well as the standard deviation across the realizations
    Cells_Lambdas = np.empty(len(Lambdas))
    stds_Lambdas = np.empty(len(Lambdas))
    for j, Lambda in enumerate(Lambdas):
        Cells_trials_ = np.empty(ntrials)
        for i in range(ntrials):
            if i % 10 == 0 and verbose:
                print(f"Lambda = {Lambda:.2e}: trial {i} of {ntrials}", flush=True)
            Cells_trials[i] = compute_Cells_in_noise_map(mu, nside, max_ell, Lambda=Lambda, mask=mask)
        Cells_Lambdas[j] = np.nanmean(Cells_trials_)
        stds_Lambdas[j] = np.nanstd(Cells_trials_)
    results_dict = dict(mu=mu,
                        mask=mask,
                        Lambdas=Lambdas,
                        ntrials=ntrials,
                        Cells=Cells_Lambdas,
                        stds=stds_Lambdas)
    if comment is not None:
        results_dict['comment'] = comment
    return results_dict

def compute_Cells_in_noise_map(mu, nside, max_ell, Lambda, mask=None, compare_to_D=False):
    # noise realization
    map_to_fit = noise_overdensity_map(mu, nside, mask=mask)
    # fit to spherical harmonics templates
    ells, Cells = compute_Cells_in_overdensity_map_Lambda(map_to_fit,
                                                                    Lambda=Lambda, max_ell=int(max_ell))
    if compare_to_D == True:
        dipole_comps = measure_dipole_in_overdensity_map_Lambda(map_to_fit, Lambda=Lambda)
        # does the C1_from_D( norm of the dipole components ) match C1 ?
        assert np.allclose(Cells[1], tools.C1_from_D(np.linalg.norm(dipole_comps)))
        # does the D_from_C1 match D ?
        assert np.allclose(np.linalg.norm(dipole_comps), tools.D_from_C1(Cells[1]))
    return Cells[1:]  # cut out monopole since we're fitting overdensities


"""
GENERATE NOISE MAP
"""
def noise_overdensity_map(mu, nside=64, mask=None):
    # healpix Poisson noise map with uniform mean density mu,
    #   resolution nside, and optional mask (should be 1 in unmasked and 0 in masked)
    noise = tools.generate_noise_map(mu, nside)
    if mask is not None:
        noise[~mask] = np.nan
    return noise / np.nanmean(noise) - 1


if __name__ == '__main__':
    main()



# def run_Cells_convergence():
#     # sort the trials increasing
#     ntrialss.sort()
#     # final resultsl to save: the average Cells at each "checkpoint" ntrials
#     Cells = np.empty((len(ntrialss), max_ell))  
#     itrial = 0
#     intrial = 0
#     Cells_ = []  # running list of the result from each trial up to max(ntrialss)
#     while itrial < max(ntrialss):
#         Cells_.append(compute_Cells_fullsky(itrial, mu, nside, max_ell, verbose=False))
#         if itrial+1 == ntrialss[intrial]:
#             assert ntrialss[intrial] == len(Cells_)
#             Cells[intrial] = np.mean(np.array(Cells_), axis=0)
#             intrial += 1
#             print(f"saving mean Cells after {itrial+1} trials", flush=True)
#         itrial += 1
#     results_dict = dict(ells=np.arange(1, max_ell+1),
#                             ntrials=ntrialss,
#                             Cells_fullsky=Cells,
#                             selfunc_fn=selfunc_fn,
#                             mu=mu,
#                             comment=comment)
#     save_fn = os.path.join(save_dir, f'noise_Cells_fullsky_ellmax{int(max_ell)}_{max(ntrialss)}trials.npy')
#     np.save(save_fn, results_dict)
#     print(f"saved to {save_fn}", flush=True)

# def run_Cells_cut_sky():
#     for i, max_ell in enumerate(max_ells):
#         # define function (for multiprocessing) to compute Cells on a cut sky as a function of Lambda
#         def compute_Cells_cutsky(index, Lambda, results_dict):
#             Cells_this_Lambda_trials = np.empty((ntrials, max_ell))
#             for i in range(ntrials):
#                 if i % 10 == 0:
#                     print(f"max_ell = {int(max_ell)}, Lambda = {Lambda:.2e}: trial {i+1} of {ntrials}", flush=True)
#                 # noise realization for this trial, MASKED, with the mean density, on the full sky
#                 map_to_fit = noise_overdensity_map(mu, nside, mask=mask)
#                 # fit to spherical harmonics templates
#                 ells, Cells_ = compute_Cells_in_overdensity_map_Lambda(map_to_fit, Lambda, max_ell=int(max_ell))
#                 Cells_this_Lambda_trials[i] = Cells_[1:]  # cut out monopole since we're fitting overdensities
#             Cells_this_Lambda = np.nanmean(Cells_this_Lambda_trials, axis=0)
#             std_this_Lambda = np.nanstd(Cells_this_Lambda_trials, axis=0)
#             results_dict[index] = (Lambda, Cells_this_Lambda, std_this_Lambda)

#         # multiprocessing Process
#         print(f"computing Cells on the cut sky...", flush=True)
#         s = time.time()
#         manager = mp.Manager()
#         procs = []
#         cutsky_results = manager.dict()
#         for i, Lambda in enumerate(Lambdas):
#             proc = mp.Process(target=compute_Cells_cutsky, args=(i, Lambda, cutsky_results,))
#             procs.append(proc)
#             proc.start()
        
#         for proc in procs:
#             proc.join()

#         results_dict = dict(ells=np.arange(1, max_ell+1),
#                             Cells_cutsky=cutsky_results.values(),
#                             selfunc_fn=selfunc_fn,
#                             mu=mu,
#                             comment=comment)
#         save_fn = os.path.join(save_dir, f'noise_Cells_cutsky_ellmax{int(max_ell)}_{ntrials}trials_Lambda.npy')
#         np.save(save_fn, results_dict)
#         print(f"saved to {save_fn}", flush=True)