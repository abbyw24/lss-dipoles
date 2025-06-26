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
from abc_for_qso import get_catalog_info, distance, save_accepted_mocks, model_dipole_excess
from abc_free_dipole import model_free_dipole, model_free_dipole_excess

RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def main():

    """ MAIN INPUTS """

    model = 'free_dipole_excess'

    catname = 'quaia_G20.0'

    # which catalog are we trying to emulate? get the info here if we want the expected dipole amp
    catalog_info = get_catalog_info(catname)

    # fake data parameters
    data_dipole_amp = catalog_info['expected_dipole_amp']
    data_log_excess = -4
    base_rate = catalog_info['base_rate']

    distance_nside = 2
    nside = 64
    blim = 30

    population_size = 500
    minimum_epsilon = 1e-10
    ngens = 10

    ell_max_data = 2     # used to inject any excess power into the fake data
    ell_max_abc = 2                 # used in the ABC only if 'excess' in model

    selfunc = False     # use the catalog's selfunc? if False, uses 'ones' i.e. perfect completeness

    # include shot noise?
    poisson_data = False
    poisson_abc = False

    continue_run = True        # continue a run where we left off, if one exists but stopped (probably due to time limit issues)

    # check if we've already started a run for this case
    save_dir = fake_data_dir(catalog_info['selfunc_str'], base_rate, data_dipole_amp, data_log_excess, catalog_info['expected_dipole_amp'],
                                poisson=poisson_data, ell_max=ell_max_data, selfunc=selfunc)
    data_fn = os.path.join(save_dir, f'fake_data.npy')
    if continue_run == True and os.path.exists(data_fn):
        print(f"loading data from a previous run")
        fake_data_dict = np.load(data_fn, allow_pickle=True).item()
        continue_run_ = True    # tell run_abc() that we want to continue the previous run using this data
    else:
        print(f"generating new fake data")
        # generate fake data
        fake_data_dict = generate_fake_data(catname, data_dipole_amp, data_log_excess, base_rate,
                                            ell_max=ell_max_data, poisson=poisson_data, selfunc=selfunc)
        # save the fake data here in case we need to continue the run later (otherwise a continued run would generate new fake data!)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(data_fn, fake_data_dict)
        continue_run_ = False  # tell run_abc() that we actually want to start a new run since we don't have the old data

    # run the ABC for this catalog and model:
    #   saves the posteriors, history, and the accepted maps from the final generation
    run_abc(fake_data_dict, save_dir, model, distance_nside, population_size, ngens,
            minimum_epsilon=minimum_epsilon, nside=nside, blim=blim, ell_max=ell_max_abc, poisson=poisson_abc,
            continue_run=continue_run_)


def generate_fake_data(catname, dipole_amp, log_excess, base_rate, nside=64, blim=30, ell_max=8,
                        poisson=True, selfunc=True):

    data_pars = dict(dipole_amp=dipole_amp, log_excess=log_excess)

    # expected dipole direction
    cmb_dipdir = SkyCoord(264, 48, unit=u.deg, frame='galactic')

    # (theta, phi) in each healpixel
    theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))

    # which catalog are we trying to emulate?
    catalog_info = get_catalog_info(catname)

    # selection function
    selfunc_str = catalog_info['selfunc_str'] if selfunc else 'ones'
    selfunc = gm.get_selfunc_map(selfunc_str, nside=nside, blim=blim)

    # call the model
    data = model_dipole_excess(data_pars, selfunc, base_rate, cmb_dipdir, theta, phi, ell_max,
                                poisson=poisson, return_alms=True)

    # convert to overdensity
    odmap = data['data'] / np.nanmean(data['data']) - 1.

    fake_data_dict = {
        'catname' : catname,
        'selfunc_str' : selfunc_str,
        'selfunc' : selfunc,
        'expected_dipole_amp' : catalog_info['expected_dipole_amp'],
        'odmap' : odmap,
        'data' : data,
        'alms' : data['alms'],
        'input_dipole_amp' : dipole_amp,
        'input_log_excess' : log_excess,
        'base_rate' : base_rate,
        'poisson' : poisson
    }
    return fake_data_dict

def run_abc(fake_data_dict, fake_data_dir, model, distance_nside, population_size, ngens,
            minimum_epsilon=1e-10, nside=64, blim=30, ell_max=8, poisson=True, continue_run=True):

    assert model.lower() in ['free_dipole', 'free_dipole_excess'], "unknown model name"
    model = model.lower()

    """ DATA & SELECTION FUNCTION """

    # unpack what we need from the fake data dictionary
    selfunc = fake_data_dict['selfunc']
    base_rate = fake_data_dict['base_rate']
    input_dipole_amp = fake_data_dict['input_dipole_amp']
    odmap = fake_data_dict['odmap'] # the data overdensity map
    expected_dipole_amp = fake_data_dict['expected_dipole_amp']

    # (theta, phi) in each healpixel
    theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))

    """ PRIOR """
    # bounds for prior:
    #   first is lower bound, second entry is WIDTH (not upper bound)
    dipole_x_bounds = (-.01, .02)
    dipole_y_bounds = (-.01, .02)
    dipole_z_bounds = (-.01, .02)
    log_excess_bounds = (-10, 7)        # only used if excess power is a free parameter in the model

    prior = {}
    prior['dipole_x'] = pyabc.RV("uniform", *dipole_x_bounds)
    prior['dipole_y'] = pyabc.RV("uniform", *dipole_y_bounds)
    prior['dipole_z'] = pyabc.RV("uniform", *dipole_z_bounds)
    if 'excess' in model:
        prior['log_excess'] = pyabc.RV("uniform", *log_excess_bounds)
    
    prior_abc = pyabc.Distribution(prior)

    """ WRAPPERS """
    if model == 'free_dipole':
        def model_wrapper(parameters):
            return model_free_dipole(parameters, selfunc, base_rate, theta, phi, poisson=poisson)
    
    else:
        assert model == 'free_dipole_excess'
        def model_wrapper(parameters):
            return model_free_dipole_excess(parameters, selfunc, base_rate, theta, phi, ell_max=ell_max, poisson=poisson)

    def distance_wrapper(x, x0):
        return distance(x, x0, distance_nside)

    """ SAVE DIRECTORY """
    # where to store results
    noise_tag = f'_no-SN' if poisson == False else ''
    res_dir = os.path.join(fake_data_dir,
                f'{model}_nside{distance_nside}_{population_size}mocks_{ngens}iters')
    # # !! trying a different epsilon:
    # res_dir = os.path.join(res_dir, f'QuantileEpsilon_alpha-0.2')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    """ PERFORM THE INFERENCE """
    abc = pyabc.ABCSMC(model_wrapper, prior_abc, distance_wrapper, population_size=population_size)
                        # eps=pyabc.SilkOptimalEpsilon(k=10)) # !!
                        # eps=pyabc.QuantileEpsilon(alpha=0.2))

    # store the history at this tempfile
    db_path = os.path.join(tempfile.gettempdir(), res_dir, f'history.db')
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
        'expected_dipole_amp' : expected_dipole_amp,
        'posterior' : history.get_distribution(),
        'max_t' : history.max_t,
        'old_posteriors' : [history.get_distribution(t=t) for t in range(history.max_t + 1)],
        'fake_data_dict' : fake_data_dict,
    }

    np.save(os.path.join(res_dir, f'results.npy'), res)
    print(f"history saved at {res_dir}", flush=True)

    save_accepted_mocks(res_dir, history)
    print(f"saved accepted mocks from final generation", flush=True)


def fake_data_dir(catname_, base_rate, input_dipole_amp, input_log_excess, expected_dipole_amp,
                    poisson=True, ell_max=8, selfunc=True):
    # helper function to get directory for this fake data set
    dipamp_tag = f'_dipamp-{input_dipole_amp / expected_dipole_amp:.1f}x'
    excess_tag = f'_excess-zero' if input_log_excess < -20 else f'_excess-{input_log_excess:.1f}'
    noise_tag = f'_no-SN' if poisson == False else ''
    ell_max_tag = '' if ell_max == 8 else f'_ellmax-{int(ell_max)}'
    selfunc_tag = '' if selfunc else '_sf-ones'
    save_dir = os.path.join(RESULTDIR, 'results/ABC/fake_data',  # same as the real data case except now in the extra fake_data/ dir.
                        catname_ + f'_base-rate-{base_rate:.4f}{dipamp_tag}{excess_tag}{noise_tag}{ell_max_tag}{selfunc_tag}')

    return save_dir


if __name__=='__main__':
    main()