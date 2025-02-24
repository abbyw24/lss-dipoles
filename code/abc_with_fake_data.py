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

    distance_nside = 1
    nside = 64
    blim = 30
    ell_max = 8

    # parameters in the fit
    expected_dipole_amp = 0.0052
    log_excess = -6 #np.log10(3e-5)
        # I did a hacky thing where if the input log_excess < -20, it gets set to exactly zero

    # priors
    dipole_amp_bounds = (0., 3. * expected_dipole_amp)
    log_excess_bounds = (-7, 4)
    
    # other parameters for constructing the fake observation
    base_rate = 33.6330  # mean base rate of the final 100 accepted samples for Quaia, 14 generations
    poisson = True     # whether to include shot noise
    selfunc_str = 'quaia_G20.0_orig'
    selfunc = gm.get_selfunc_map(selfunc_str, nside=nside, blim=blim) #np.ones(hp.nside2npix(nside))

    population_size = 500
    minimum_epsilon = 1e-8
    ngens = 14

    continue_run = False     # continue a run where we left off, if one exists but stopped (probably due to time limit issues?)

    """ FAKE DATA """
    data_pars = dict(dipole_amp=expected_dipole_amp, log_excess=log_excess)

    # expected dipole direction
    cmb_dipdir = SkyCoord(264, 48, unit=u.deg, frame='galactic')

    # (theta, phi) in each healpixel
    theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))

    # orthogonal dipole comps (not alms: this is a 3-vector in (x,y,z) with norm dipole_amp)
    comps = tools.spherical_to_cartesian(r=expected_dipole_amp,
                                        theta=np.pi/2-cmb_dipdir.icrs.dec.rad,
                                        phi=cmb_dipdir.icrs.ra.rad)

    # # antiparallel dipole amps
    # anti_comps = -comps
    # # make the amplitude match the excess power
    # anti_comps /= np.linalg.norm(anti_comps)
    # anti_comps *= tools.D_from_C1(10**log_excess)
    # # perpendicular dipole amps

    # generate the data using the same model as the mocks
    data = model(data_pars, selfunc, base_rate, cmb_dipdir, theta, phi, ell_max, poisson=poisson)#,
                    #excess_dipole_comps=[0.,0.,0.])

    # where to store results
    excess_tag = f"_no_excess" if data_pars['log_excess'] < -20 else f"_excess-1e{data_pars['log_excess']:.1f}" # !
    case_name = f"dipole-{data_pars['dipole_amp']}{excess_tag}_base-rate-{base_rate:.4f}"
    if poisson == False:
        case_name += "_no-SN"
    # # !!
    # case_name += "_low-excess-prior"
    # case_name += "_no-excess-dipole"
    save_dir = os.path.join(RESULTDIR, 'results/ABC', 'fake_data', case_name, f'{population_size}mocks_{ngens}gens')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # convert to overdensity
    odmap = data['data'] / np.nanmean(data['data']) - 1.
        # mean of odmap = 0. ; any masked pixels are zero # !!

    """ PRIOR """
    prior = pyabc.Distribution(dipole_amp = pyabc.RV("uniform", *dipole_amp_bounds),
                           log_excess = pyabc.RV("uniform", *log_excess_bounds))

    """ WRAPPERS """
    # need these wrapper functions to match required format for pyabc ; selfunc and nside defined above
    def model_wrapper(parameters):
        
        return model(parameters, selfunc, base_rate, cmb_dipdir, theta, phi, ell_max)

    def distance_wrapper(x, x0):

        return distance(x, x0, distance_nside)

    """ PERFORM THE INFERENCE """
    abc = pyabc.ABCSMC(model_wrapper, prior, distance_wrapper, population_size=population_size)
    observation = odmap   # the true data _overdensity_

    # store the history at this tempfile
    db_path = os.path.join(tempfile.gettempdir(), save_dir, f'history.db')
    if os.path.exists(db_path) and continue_run == True:
        print(f"continuing run found at {db_path}")
        # load old history to get info
        history = pyabc.History("sqlite:///" + db_path)
        abc.load("sqlite:///" + db_path, history.id)  # second argument is the run ID which is always 1 unless I do something fancy
        # max_nr_populations is the number we actually want _minus_ how many have already run
        max_nr_populations = ngens - history.max_t
    else:
        print(f"starting a new run for this case")
        abc.new("sqlite:///" + db_path, {"data": observation})
        max_nr_populations = ngens

    # start the sampling!
    if max_nr_populations > 0:
        history = abc.run(minimum_epsilon=minimum_epsilon, max_nr_populations=max_nr_populations)

    # save
    prior = dict(dipole_amp=dipole_amp_bounds, log_excess=log_excess_bounds)
    # result dictionary
    #   (and save some of the key history info since I've run into weird bugs trying to load the history object)
    res = {
        'data' : data,
        'selfunc' : selfunc,
        'data_pars' : data_pars,
        'history' : history,
        'prior' : prior,
        'observation' : observation,
        'posterior' : history.get_distribution(),
        'ell_max' : ell_max,
        'max_t' : history.max_t,
        'old_posteriors' : [history.get_distribution(t=t) for t in range(history.max_t + 1)]
    }
    np.save(os.path.join(save_dir, f'results.npy'), res)

    print(f"results saved at {save_dir}", flush=True)

    print(f"saving accepted mocks from final generation...", flush=True)

    mock_dir = os.path.join(save_dir, 'accepted_samples')
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir)
    # get the accepted mocks from the final generation
    pop = history.get_population()
    for i, particle in enumerate(pop.particles):
        np.save(os.path.join(mock_dir, f'mock{i}.npy'), particle.sum_stat['data'])
    print(f"done!", flush=True)
    

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


def model(parameters, selfunc, base_rate, dipdir, theta, phi, ell_max=8, poisson=True, excess_dipole_comps=None):
    """
    Generates a healpix density map with dipole in fixed CMB dipole direction and excess angular power.
    Uses fiducial selection function (Quaia + galactic plane mask + smaller masks from S21).

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
    excess_dipole_comps : array-like (len 3) or None, optional
        Optionally input a set of three orthogonal dipole amplitudes to specify the amplitude and direction of
        the "excess dipole." If `None`, uses `parameters['log_excess']` to draw random a1ms (like the other alms).

    Returns
    -------
    Quasar number map.
    
    """

    nside = hp.npix2nside(len(selfunc))

    # expected dipole map
    amps = tools.spherical_to_cartesian(r=parameters["dipole_amp"],
                                        theta=np.pi/2-dipdir.icrs.dec.rad,
                                        phi=dipdir.icrs.ra.rad)
    expected_dipole_map = dipole.dipole(theta, phi, *amps)

    # excess power:
    if parameters["log_excess"] < -20:  # magic, kind of hacky but I want a way to have literally zero excess power
        excess_map = np.zeros_like(expected_dipole_map)
    else:
        if excess_dipole_comps is not None:
            assert len(excess_dipole_comps) == 3, "must input three dipole amplitudes (m=-1, 0, 1)"
            excess_map = dipole.dipole(theta, phi, *excess_dipole_comps)

            # then just add power in Cells starting at ell=2
            Cells = np.zeros(ell_max)
            Cells[2:] += 10**parameters["log_excess"]
            excess_map += hp.sphtfunc.synfast(Cells, nside)
        else:
            # otherwise just add Cells like normal: flat, determined by input log_excess (and maybe input_a1ms)
            Cells = np.zeros(ell_max)
            Cells[1:] += 10**parameters["log_excess"]   # because we don't want excess power in the monopole
            excess_map = hp.sphtfunc.synfast(Cells, nside)

    # smooth overdensity map
    smooth_overdensity_map = expected_dipole_map + excess_map

    # poisson sample, including the base rate and the selfunc map
    number_map = (1. + smooth_overdensity_map) * base_rate * selfunc
    if poisson == True:
        rng = np.random.default_rng(seed=None) # should I put a seed in here??
        number_map = rng.poisson(number_map)

    return { "data" : number_map }


if __name__=='__main__':
    main()