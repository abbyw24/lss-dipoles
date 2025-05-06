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
from abc_for_qso import get_catalog_info, distance, save_accepted_mocks

RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def main():

    """ MAIN INPUTS """

    catname = 'quaia_G20.0'

    distance_nside = 2
    nside = 64

    population_size = 500
    minimum_epsilon = 1e-10
    ngens = 14

    ell_max = 8

    continue_run = True        # continue a run where we left off, if one exists but stopped (probably due to time limit issues)

    # run the ABC for this catalog and model:
    #   saves the posteriors, history, and the accepted maps from the final generation
    run_abc(catname, distance_nside, population_size, ngens,
            minimum_epsilon=minimum_epsilon, nside=nside, ell_max=ell_max, continue_run=continue_run)


def run_abc(catname, distance_nside, population_size, ngens,
            minimum_epsilon=1e-10, nside=64, ell_max=8, continue_run=True):

    """ DATA & SELECTION FUNCTION """
    catalog_info = get_catalog_info(catname)    # in this case we just need the base rate

    # where to store results
    save_dir = os.path.join(RESULTDIR, 'results/ABC',
                            f'{catname}_shot-noise_nside{distance_nside}_{population_size}mocks_{ngens}iters_base-rate-{catalog_info["base_rate"]:.4f}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate shot-noise-only data
    rng = np.random.default_rng(seed=None) # should I put a seed in here??
    snmap = rng.poisson(np.ones(hp.nside2npix(nside)) * catalog_info["base_rate"])
    # convert to overdensities
    odmap = snmap / np.nanmean(snmap) - 1.

    """ PRIOR """
    log_excess_bounds = (-10, 7)

    prior = {
        'log_excess' : pyabc.RV("uniform", *log_excess_bounds)
    }
    
    prior_abc = pyabc.Distribution(prior)

    """ WRAPPERS """
    # need these wrapper functions to match required format for pyabc ; selfunc and nside defined above
    def model_wrapper(parameters):
        return model_excess_only(parameters, catalog_info["base_rate"], ell_max=ell_max, nside=nside)

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
    
    # save dictionary of results
    #   (and save some of the key history info since I've run into weird bugs trying to load the history object)
    res = {
        'history' : history,
        'prior' : prior,
        'observation' : odmap,
        'shot_noise_map' : snmap,
        'ell_max' : ell_max,
        'base_rate' : catalog_info["base_rate"],
        'posterior' : history.get_distribution(),
        'max_t' : history.max_t,
        'old_posteriors' : [history.get_distribution(t=t) for t in range(history.max_t + 1)]
    }

    np.save(os.path.join(save_dir, f'results.npy'), res)
    print(f"history saved at {save_dir}", flush=True)

    save_accepted_mocks(save_dir, history)
    print(f"saved accepted mocks from final generation", flush=True)


"""
MODEL
"""
def model_excess_only(parameters, base_rate, ell_max=8, nside=64):
    """
    Generates a healpix density map with dipole in fixed CMB dipole direction and excess angular power.

    Parameters
    ----------
    parameters : dict
        keys:
            "log_excess" = log of the excess power (flat in Cell)

    Returns
    -------
    Quasar number map.
    
    """

    # Cells: flat, determined by input log_excess
    Cells = np.zeros(ell_max)
    Cells[1:] += 10**parameters["log_excess"]   # because we don't want excess power in the monopole
    excess_map = hp.sphtfunc.synfast(Cells, nside)

    # smooth overdensity map
    smooth_overdensity_map = excess_map

    # turn into a number map (adding ones here so we can use the same distance metric, which converts to overdensities)
    number_map = (1. + smooth_overdensity_map) * base_rate

    return { "data" : number_map }


if __name__=='__main__':
    main()