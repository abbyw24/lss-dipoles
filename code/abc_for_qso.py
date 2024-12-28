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

RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def main():

    """ MAIN INPUTS """

    catname = 'catwise'

    distance_nside = 1
    nside = 64
    blim = 30

    population_size = 50
    minimum_epsilon = 100
    max_nr_populations = 8

    # where to store results
    save_dir = os.path.join(RESULTDIR, 'results/ABC',
                            f'{catname}_dipole_excess_nside{distance_nside}_{population_size}mocks_{max_nr_populations}iters')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # catalog-specific inputs
    if catname == 'quaia':
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/quaia/quaia_G20.0.fits')
        selfunc_fn = os.path.join(RESULTDIR, f'data/catalogs/quaia/selfuncs/selection_function_NSIDE64_G20.0.fits')
        expected_dipole_amp = 0.0052
        dipole_amp_bounds = (0., 3 * expected_dipole_amp) # first is lower bound, second entry is WIDTH (not upper bound)
        log_excess_bounds = (-6, 3)
        if nside == 1:
            base_rate_bounds = (1.35e5, 1.5e4)  # note much higher base rate since healpixels are much bigger; depends on nside
        elif nside == 2:
            base_rate_bounds = (3.4e4, 3e3)
        elif nside == 4:
            base_rate_bounds = (8e3, 1e3) #(8.6e3, 5e2)
        elif nside == 64:
            base_rate_bounds = (30., 35.)
        else:
            assert False, f"need to add base rate bounds for this nside"
    else:
        assert catname == 'catwise', "catname must be quaia or catwise"
        fn_cat = os.path.join(RESULTDIR, f'data/catalogs/catwise_agns/catwise_agns_master.fits')
        selfunc_fn = os.path.join(RESULTDIR, f'data/catalogs/catwise_agns/selfuncs/selection_function_NSIDE64_catwise_pluszodis.fits')
        expected_dipole_amp = 0.0074
        dipole_amp_bounds = (0., 3 * expected_dipole_amp)
        log_excess_bounds = (-6, 3)
        if nside == 1:
            base_rate_bounds = (3e5, 2e4)
        elif nside == 2:
            base_rate_bounds = (7.5e4, 5e3)
        elif nside == 4:
            base_rate_bounds = (1.8e4, 2e3) #(1.85e4, 1.5e3) before mask fix
        elif nside == 64:
            base_rate_bounds = (65., 75.)
        else:
            assert False, f"need to add base rate bounds for this nside"

    """ DATA & SELECTION FUNCTION """
    # selection function: this isn't applied to the data but used to generate the mock skies
    small_masks = fitsio.read(os.path.join(RESULTDIR, f'data/catalogs/masks/mask_master_hpx_r1.0.fits'))
    selfunc = hp.read_map(selfunc_fn) * small_masks * tools.get_galactic_plane_mask(blim, nside=64, frame='icrs')

    # load catalog
    qmap_raw = tools.load_catalog_as_map(fn_cat, frame='icrs', nside=64)
    # add galactic plane mask
    qmap = qmap_raw * tools.get_galactic_plane_mask(blim, nside=64, frame='icrs') * small_masks

    """ PRIOR """
    prior = pyabc.Distribution(dipole_amp = pyabc.RV("uniform", *dipole_amp_bounds),
                           log_excess = pyabc.RV("uniform", *log_excess_bounds),
                           base_rate = pyabc.RV("uniform", *base_rate_bounds))

    """ WRAPPERS """
    # need these wrapper functions to match required format for pyabc ; selfunc and nside defined above
    def model_wrapper(parameters):
        
        return model(parameters, selfunc)

    def distance_wrapper(x, x0):

        return distance(x, x0, distance_nside)

    """ PERFORM THE INFERENCE """
    abc = pyabc.ABCSMC(model_wrapper, prior, distance_wrapper, population_size=population_size)
    observation = qmap   # the true data

    # store the history at this tempfile
    db_path = os.path.join(tempfile.gettempdir(), save_dir, f'history.db')
    abc.new("sqlite:///" + db_path, {"data": observation})

    # start the sampling!
    history = abc.run(minimum_epsilon=minimum_epsilon, max_nr_populations=max_nr_populations)

    # save
    prior = dict(dipole_amp=dipole_amp_bounds, log_excess=log_excess_bounds, base_rate=base_rate_bounds)
    res = dict(history=history, prior=prior, observation=observation)
    np.save(os.path.join(save_dir, f'history.npy'), res)

    print(f"history saved at {save_dir}", flush=True)

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
        Data; real quasar map.
    x0 : shape (npix,)
        Mock quasar map.
    
    Returns
    -------
    rho : float
        Sum of the difference in the pixel values squared.
    """

    return sum(hp.ud_grade(x['data'] - x0['data'], nside, power=-2)**2)
                                                            # power = -2 preserves the sum of the map => preserves total number of quasars


def model(parameters, selfunc):
    """
    Generates a healpix density map with dipole in fixed CMB dipole direction and excess angular power.
    Uses fiducial selection function (Quaia + galactic plane mask + smaller masks from S21).

    Parameters
    ----------
    parameters : dict
        keys:
            "dipole_amp" = dipole amplitude
            "log_excess" = log of the excess power (flat in Cell)
            "base_rate" = quasar density base rate
    nside : int

    Returns
    -------
    Quasar number map.
    
    """

    nside = hp.npix2nside(len(selfunc))
    
    rng = np.random.default_rng(seed=None) # should I put a seed in here??

    # expected dipole map
    amps = np.zeros(4)
    amps[1:] = dipole.cmb_dipole(amplitude=parameters["dipole_amp"], return_amps=True)
    expected_dipole_map = dipole.dipole_map(amps, NSIDE=nside)

    # add Cells
    # Cells: flat, determined by input log_excess
    Cells = np.zeros(8) + 10**parameters["log_excess"]
    # draw alms from a Gaussian
    sph_harm_amp_dict = {}
    for ell in range(1, len(Cells)+1):
        sph_harm_amp_dict[ell] = np.sqrt(Cells[ell-1]) * rng.normal(size=2 * ell + 1)
    # then make map from the alms
    excess_map = np.zeros((hp.nside2npix(nside)))
    for ell in sph_harm_amp_dict.keys():
        alms = sph_harm_amp_dict[ell]
        assert len(alms) == 2 * ell + 1, \
            f"incorrect number of coefficients for ell={ell} ({len(alms)}, expected {2 * ell + 1}"
        excess_map += multipoles.multipole_map(alms, NSIDE=nside)

    # smooth overdensity map
    smooth_overdensity_map = expected_dipole_map + excess_map

    # poisson sample, including the base rate and the selfunc map
    number_map = rng.poisson((1. + smooth_overdensity_map) * parameters["base_rate"] * selfunc)

    return { "data" : number_map }


if __name__=='__main__':
    main()