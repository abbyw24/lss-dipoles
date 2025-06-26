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

RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def main():

    """ MAIN INPUTS """
    """
    Which model are we using? Options are:
    - "dipole_excess" : two free parameters, dipole amplitude (fixed dir.) and log excess power
    - "dipole_only" : one free parameter, dipole amplitude (fixed dir.)
    - "dipole_excess_free-base" : three free parameters, dipole amplitude (fixed dir.), log excess power, and base rate
    """
    model = 'dipole_excess'

    catname = 'quaia_G20.0_zsplit2bin1'

    distance_nside = 2
    nside = 64
    blim = 30

    population_size = 500
    minimum_epsilon = 1e-10
    ngens = 18

    ell_max = 8     # only used if 'excess' is in model

    continue_run = True        # continue a run where we left off, if one exists but stopped (probably due to time limit issues)

    # run the ABC for this catalog and model:
    #   saves the posteriors, history, and the accepted maps from the final generation
    run_abc(catname, model, distance_nside, population_size, ngens,
            minimum_epsilon=minimum_epsilon, nside=nside, blim=blim, ell_max=ell_max, continue_run=continue_run)


def run_abc(catname, model, distance_nside, population_size, ngens,
            minimum_epsilon=1e-10, nside=64, blim=30, ell_max=8, continue_run=True):

    assert model.lower() in ['dipole_excess', 'dipole_only', 'dipole_excess_free-base'], "unknown model name"
    model = model.lower()

    """ DATA & SELECTION FUNCTION """
    catalog_info = get_catalog_info(catname)

    # where to store results
    catname_ = catalog_info['selfunc_str']
    base_rate_tag = '' if 'base' in model else f'_base-rate-{catalog_info["base_rate"]:.4f}'
    save_dir = os.path.join(RESULTDIR, 'results/ABC',
                            f'{catname_}_{model}_nside{distance_nside}_{population_size}mocks_{ngens}iters{base_rate_tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    odmap, qmap_masked = get_observation(catalog_info['fn_cat'], nside, blim)

    # selection function: this isn't applied to the data but used to generate the mock skies
    selfunc = gm.get_selfunc_map(catalog_info['selfunc_str'], nside=nside, blim=blim)

    # expected dipole direction
    cmb_dipdir = SkyCoord(264, 48, unit=u.deg, frame='galactic')

    # (theta, phi) in each healpixel
    theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))

    """ PRIOR """
    # bounds for prior: these are the same for all catalogs aside from the dependence on expected dipole amp and base rate
    #   note that these bounds are not necessarily used if the corresponding parameter is fixed in the model.
    #   first is lower bound, second entry is WIDTH (not upper bound)
    dipole_amp_bounds = (-1. * catalog_info['expected_dipole_amp'], 5 * catalog_info['expected_dipole_amp'])
    log_excess_bounds = (-10, 7)
    base_rate_bounds = (catalog_info['base_rate'] - 10, 20)

    prior = {}
    if 'dipole' in model:
        prior['dipole_amp'] = pyabc.RV("uniform", *dipole_amp_bounds)
    if 'excess' in model:
        prior['log_excess'] = pyabc.RV("uniform", *log_excess_bounds)
    if 'base' in model:
        prior['base_rate'] = pyabc.RV("uniform", *base_rate_bounds)
    
    prior_abc = pyabc.Distribution(prior)

    """ WRAPPERS """
    # need these wrapper functions to match required format for pyabc ; selfunc and nside defined above
    if model == 'dipole_excess':
        def model_wrapper(parameters):
            return model_dipole_excess(parameters, selfunc, catalog_info['base_rate'], cmb_dipdir, theta, phi, ell_max=ell_max)

    elif model == 'dipole_only':
        def model_wrapper(parameters):
            return model_dipole_only(parameters, selfunc, catalog_info['base_rate'], cmb_dipdir, theta, phi)

    else:
        assert model == 'dipole_excess_free-base'
        def model_wrapper(parameters):
            return model_dipole_excess_base(parameters, selfunc, cmb_dipdir, theta, phi)

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
        'selfunc' : selfunc,
        'expected_dipole_amp' : catalog_info['expected_dipole_amp'],
        'qmap' : qmap_masked,
        'posterior' : history.get_distribution(),
        'max_t' : history.max_t,
        'old_posteriors' : [history.get_distribution(t=t) for t in range(history.max_t + 1)]
    }
    # model-specific info
    if 'excess' in model:
        res['ell_max'] = ell_max
    if 'base' not in model:
        res['base_rate'] = catalog_info['base_rate']

    np.save(os.path.join(save_dir, f'results.npy'), res)
    print(f"history saved at {save_dir}", flush=True)

    save_accepted_mocks(save_dir, history)
    print(f"saved accepted mocks from final generation", flush=True)


def get_catalog_info(catname):
    # catalog-specific inputs
    if catname == 'quaia_G20.0':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/quaia/{catname}.fits')
        selfunc_str = 'quaia_G20.0_zodi'
        expected_dipole_amp = 0.0052
        base_rate = 33.6330 # mean base rate of the final 100 accepted samples for Quaia, 14 generations
    
    elif catname == 'quaia_G20.0_orig':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/quaia/quaia_G20.0.fits')
        selfunc_str = 'quaia_G20.0_orig'
        expected_dipole_amp = 0.0052
        base_rate = 33.6330 # mean base rate of the final 100 accepted samples for Quaia, 14 generations

    elif catname == 'quaia_G20.5':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/quaia/{catname}.fits')
        selfunc_str = f'quaia_G20.5_orig'
        expected_dipole_amp = 0.0047
        base_rate = 41.356  # mean of (selfunc corrected) unmasked pixels in G<20.5 with 'quaia_G20.5_orig'

    elif catname == 'quaia_G20.0_zsplit2bin0':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/quaia/{catname}.fits')
        selfunc_str = 'quaia_G20.0_zsplit2bin0'
        expected_dipole_amp = 0.0050
        base_rate = 18.081 # mean of (selfunc corrected) unmasked pixels in 'quaia_G20.0_zsplit2bin0'

    elif catname == 'quaia_G20.0_zsplit2bin1':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/quaia/{catname}.fits')
        selfunc_str = 'quaia_G20.0_zsplit2bin1'
        expected_dipole_amp = 0.0055
        base_rate = 18.840 # mean of (selfunc corrected) unmasked pixels in 'quaia_G20.0_zsplit2bin1'
    
    elif catname == 'catwise':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/catwise_agns/catwise_agns_master.fits')
        selfunc_str = 'catwise_zodi'
        expected_dipole_amp = 0.0074
        base_rate = 77.4495 # mean base rate of the final 100 accepted samples for CatWISE, 13 generations

    elif catname == 'catwise_elatcorr':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/catwise_agns/catwise_agns_master.fits')
        selfunc_str = 'catwise_elatcorr'
        expected_dipole_amp = 0.0074
        base_rate = 77.4495 # mean base rate of the final 100 accepted samples for CatWISE, 13 generations

    else:
        raise ValueError("unknown catname")

    return dict(fn_cat=fn_cat, selfunc_str=selfunc_str, expected_dipole_amp=expected_dipole_amp, base_rate=base_rate)


def get_observation(fn_cat, nside, blim):
    # load catalog
    qmap_raw = tools.load_catalog_as_map(fn_cat, frame='icrs', nside=nside)
    # mask
    small_masks = fitsio.read(os.path.join(RESULTDIR, f'data/catalogs/masks/mask_master_hpx_r1.0.fits'))
    qmap_masked = qmap_raw * small_masks * tools.get_galactic_plane_mask(blim, nside=nside, frame='icrs')
        # at this point, any masked pixels are ZERO (rather than NaN, e.g.); ~half the map is masked

    # convert to overdensity
    odmap = qmap_masked / np.nanmean(qmap_masked) - 1.
        # mean of odmap = 0. ; any masked pixels are zero
    return odmap, qmap_masked


def save_accepted_mocks(save_dir, history):
    mock_dir = os.path.join(save_dir, 'accepted_samples')
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir)
    # get the accepted mocks from the final generation
    pop = history.get_population()
    for i, particle in enumerate(pop.particles):
        np.save(os.path.join(mock_dir, f'mock{i}.npy'), particle.sum_stat['data'])


def distance(x, x0, nside):
    """
    Distance function to evaluate acceptance or rejection of mocks.

    Parameters
    ----------
    x : shape (npix,)
        Mock quasar map.
    x0 : shape (npix,)
        Data; real quasar map.
    
    Returns
    -------
    rho : float
        Sum of the difference in the _overdensity_ pixel values squared.
    """

    # # convert any zero pixels to NaN
    # masked_mock = x['data'].astype(float)  # converting to float here so that the line below doesn't throw an error
    # masked_mock[x['data'] == 0.] = np.nan

    # convert mock counts to overdensity
    odmap_mock = x['data'] / np.nanmean(x['data']) - 1.

    return np.sum(hp.ud_grade(odmap_mock - x0['data'], nside)**2)
                                                            # power = -2 preserves the sum of the map => preserves total number of quasars

"""
MODELS
"""

def model_dipole_excess(parameters, selfunc, base_rate, dipdir, theta, phi, ell_max=8, poisson=True, return_alms=False):
    """
    Generates a healpix density map with dipole in fixed CMB dipole direction and excess angular power.

    Parameters
    ----------
    parameters : dict
        keys:
            "dipole_amp" = dipole amplitude
            "log_excess" = log of the excess power (flat in Cell)
    selfunc : ndarray
        Selection function map. The map is generated with the same npix.
    base_rate : float
        Base rate (used to be a parameter, now hard-coded).

    Returns
    -------
    Quasar number map.
    
    """

    nside = hp.npix2nside(len(selfunc))

    # expected dipole map
    # amps = np.zeros(4)
    amps = tools.spherical_to_cartesian(r=parameters["dipole_amp"],
                                        theta=np.pi/2-dipdir.icrs.dec.rad,
                                        phi=dipdir.icrs.ra.rad)
    expected_dipole_map = dipole.dipole(theta, phi, *amps)

    # add Cells
    # Cells: flat, determined by input log_excess
    if parameters["log_excess"] < -20:  # magic, kind of hacky but I want a way to have literally zero excess power
        excess_map = np.zeros_like(expected_dipole_map)
        alms = np.zeros(np.sum([2 * ell + 1 for ell in range(ell_max + 1)]))
    else:
        Cells = np.zeros(ell_max + 1)
        Cells[1:] += 10**parameters["log_excess"]   # because we don't want excess power in the monopole
        excess_map, alms = hp.sphtfunc.synfast(Cells, nside, alm=True)

    # smooth overdensity map
    smooth_overdensity_map = expected_dipole_map + excess_map

    # poisson sample, including the base rate and the selfunc map
    number_map = (1. + smooth_overdensity_map) * base_rate * selfunc
    if poisson == True:
        rng = np.random.default_rng(seed=None) # should I put a seed in here??
        number_map = rng.poisson(number_map)

    if return_alms:
        return { "data" : number_map, "alms" : alms}
    else:
        return { "data" : number_map }


def model_dipole_only(parameters, selfunc, base_rate, dipdir, theta, phi, poisson=True):
    """
    Generates a healpix density map with dipole in fixed CMB dipole direction.

    Parameters
    ----------
    parameters : dict
        keys:
            "dipole_amp" = dipole amplitude
    selfunc : ndarray
        Selection function map. The map is generated with the same npix.
    base_rate : float
        Base rate, number of quasars per healpixel.

    Returns
    -------
    Quasar number map.
    
    """

    nside = hp.npix2nside(len(selfunc))

    # expected dipole map
    # amps = np.zeros(4)
    amps = tools.spherical_to_cartesian(r=parameters["dipole_amp"],
                                        theta=np.pi/2-dipdir.icrs.dec.rad,
                                        phi=dipdir.icrs.ra.rad)
    expected_dipole_map = dipole.dipole(theta, phi, *amps)

    # smooth overdensity map
    smooth_overdensity_map = expected_dipole_map

    # poisson sample, including the base rate and the selfunc map
    number_map = (1. + smooth_overdensity_map) * base_rate * selfunc
    if poisson == True:
        rng = np.random.default_rng(seed=None) # should I put a seed in here??
        number_map = rng.poisson(number_map)

    return { "data" : number_map }


def model_dipole_excess_base(parameters, selfunc, dipdir, theta, phi, ell_max=8, poisson=True):
    """
    Generates a healpix density map with dipole in fixed CMB dipole direction and excess angular power.

    Parameters
    ----------
    parameters : dict
        keys:
            "dipole_amp" = dipole amplitude
            "log_excess" = log of the excess power (flat in Cell)
            "base_rate" = base rate of quasars per healpixel
    selfunc : ndarray
        Selection function map. The map is generated with the same npix.

    Returns
    -------
    Quasar number map.
    
    """

    nside = hp.npix2nside(len(selfunc))

    # expected dipole map
    # amps = np.zeros(4)
    amps = tools.spherical_to_cartesian(r=parameters["dipole_amp"],
                                        theta=np.pi/2-dipdir.icrs.dec.rad,
                                        phi=dipdir.icrs.ra.rad)
    expected_dipole_map = dipole.dipole(theta, phi, *amps)

    # add Cells
    # Cells: flat, determined by input log_excess
    if parameters["log_excess"] < -20:  # magic, kind of hacky but I want a way to have literally zero excess power
        excess_map = np.zeros_like(expected_dipole_map)
    else:
        Cells = np.zeros(ell_max + 1)
        Cells[1:] += 10**parameters["log_excess"]   # because we don't want excess power in the monopole
        excess_map = hp.sphtfunc.synfast(Cells, nside)

    # smooth overdensity map
    smooth_overdensity_map = expected_dipole_map + excess_map

    # poisson sample, including the base rate and the selfunc map
    number_map = (1. + smooth_overdensity_map) * parameters["base_rate"] * selfunc
    if poisson == True:
        rng = np.random.default_rng(seed=None) # should I put a seed in here??
        number_map = rng.poisson(number_map)

    return { "data" : number_map }


if __name__=='__main__':
    main()