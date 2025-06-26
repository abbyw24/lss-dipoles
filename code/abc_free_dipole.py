import numpy as np
import os
import sys
import tempfile
import pyabc

import healpy as hp
from scipy.special import sph_harm
import fitsio
from astropy.coordinates import SkyCoord
import astropy.units as u

import tools
import dipole
import multipoles
import generate_mocks as gm
from abc_for_qso import get_catalog_info, distance, save_accepted_mocks, model_dipole_excess, get_observation

RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def main():

    """ MAIN INPUTS """

    catname = 'catwise_elatcorr'

    distance_nside = 2
    nside = 64
    blim = 30

    population_size = 500
    minimum_epsilon = 1e-10
    ngens = 18

    continue_run = True        # continue a run where we left off, if one exists but stopped (probably due to time limit issues)

    # run the ABC for this catalog and model:
    #   saves the posteriors, history, and the accepted maps from the final generation
    run_abc(catname, distance_nside, population_size, ngens,
            minimum_epsilon=minimum_epsilon, nside=nside, blim=blim, continue_run=continue_run)


def run_abc(catname, distance_nside, population_size, ngens,
            minimum_epsilon=1e-10, nside=64, blim=30, poisson=True, continue_run=True):

    """ DATA & SELECTION FUNCTION """
    catalog_info = get_catalog_info(catname)

    # where to store results
    catname_ = catalog_info['selfunc_str']
    save_dir = os.path.join(RESULTDIR, 'results/ABC', f'{catname_}_free_dipole_nside{distance_nside}_' +
                                                        f'{population_size}mocks_{ngens}iters_base-rate-{catalog_info["base_rate"]:.4f}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    odmap, qmap_masked = get_observation(catalog_info['fn_cat'], nside, blim)

    # selection function: this isn't applied to the data but used to generate the mock skies
    selfunc = gm.get_selfunc_map(catalog_info['selfunc_str'], nside=nside, blim=blim)

    # (theta, phi) in each healpixel
    theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))

    """ PRIOR """
    # bounds for prior:
    #   first is lower bound, second entry is WIDTH (not upper bound)
    dipole_x_bounds = (-.015, .03)
    dipole_y_bounds = (-.015, .03)
    dipole_z_bounds = (-.015, .03)

    prior = {}
    prior['dipole_x'] = pyabc.RV("uniform", *dipole_x_bounds)
    prior['dipole_y'] = pyabc.RV("uniform", *dipole_y_bounds)
    prior['dipole_z'] = pyabc.RV("uniform", *dipole_z_bounds)
    
    prior_abc = pyabc.Distribution(prior)

    """ WRAPPERS """
    def model_wrapper(parameters):
        return model_free_dipole(parameters, selfunc, catalog_info['base_rate'], theta, phi)

    def distance_wrapper(x, x0):
        return distance(x, x0, distance_nside)

    """ PERFORM THE INFERENCE """
    abc = pyabc.ABCSMC(model_wrapper, prior_abc, distance_wrapper, population_size=population_size)

    # store the history at this tempfile
    db_path = os.path.join(tempfile.gettempdir(), save_dir, f'history.db')
    if os.path.exists(db_path) and continue_run == True:
        print(f"continuing run found at {db_path}")
        # load old history to get info
        history = pyabc.History("sqlite:///" + db_path)
        abc.load("sqlite:///" + db_path, history.id)  # second argument is the run ID which is always 1 unless I do something fancy
        # max_nr_populations is the number we actually want _minus_ how many have already run
        max_nr_populations = ngens - history.max_t - 1
    else:
        print(f"starting a new run for this case: {db_path}")
        abc.new("sqlite:///" + db_path, {"data": odmap})
        max_nr_populations = ngens

    # start the sampling!
    if max_nr_populations > 0:
        history = abc.run(minimum_epsilon=minimum_epsilon, max_nr_populations=max_nr_populations)

    """SAVE RESULTS"""
    # save dictionary of results
    #   (and save some of the key history info since I've run into weird bugs trying to load the history object)
    res = {
        'history' : history,
        'prior' : prior,
        'observation' : odmap,
        'selfunc' : selfunc,
        'expected_dipole_amp' : catalog_info['expected_dipole_amp'],
        'qmap' : qmap_masked,
        'posterior' : history.get_distribution(),
        'max_t' : history.max_t,
        'old_posteriors' : [history.get_distribution(t=t) for t in range(history.max_t + 1)]
    }

    np.save(os.path.join(save_dir, f'results.npy'), res)
    print(f"history saved at {save_dir}", flush=True)

    save_accepted_mocks(save_dir, history)
    print(f"saved accepted mocks from final generation", flush=True)


def model_free_dipole(parameters, selfunc, base_rate, theta, phi, poisson=True):

    nside = hp.npix2nside(len(selfunc))

    # expected dipole map
    dipole_map = dipole.dipole(theta, phi, parameters['dipole_x'],
                                            parameters['dipole_y'],
                                            parameters['dipole_z'])

    # poisson sample, including the base rate and the selfunc map
    number_map = (1. + dipole_map) * base_rate * selfunc
    if poisson == True:
        rng = np.random.default_rng(seed=None) # should I put a seed in here??
        number_map = rng.poisson(number_map)

    return { "data" : number_map }


def model_free_dipole_excess(parameters, selfunc, base_rate, theta, phi, ell_max=8, poisson=True, return_alms=False):

    nside = hp.npix2nside(len(selfunc))

    # expected dipole map
    dipole_map = dipole.dipole(theta, phi, parameters['dipole_x'],
                                            parameters['dipole_y'],
                                            parameters['dipole_z'])

    # add Cells
    # Cells: flat, determined by input log_excess
    if parameters["log_excess"] < -20:  # magic, kind of hacky but I want a way to have literally zero excess power
        excess_map = np.zeros_like(dipole_map)
        alms = np.zeros(np.sum([2 * ell + 1 for ell in range(ell_max + 1)]))
    else:
        Cells = np.zeros(ell_max + 1)
        Cells[1:] += 10**parameters["log_excess"]   # because we don't want excess power in the monopole
        excess_map, alms = hp.sphtfunc.synfast(Cells, nside, alm=True)

    # smooth overdensity map
    smooth_overdensity_map = dipole_map + excess_map

    # poisson sample, including the base rate and the selfunc map
    number_map = (1. + smooth_overdensity_map) * base_rate * selfunc
    if poisson == True:
        rng = np.random.default_rng(seed=None) # should I put a seed in here??
        number_map = rng.poisson(number_map)

    if return_alms:
        return { "data" : number_map, "alms" : alms}
    else:
        return { "data" : number_map }


if __name__=='__main__':
    main()