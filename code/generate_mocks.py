"""
# generate mocks

## License
Copyright 2024 The authors.
This code is released for re-use under the open-source MIT License.

## Authors:
- **Abby Williams** (Chicago)
- **David W. Hogg** (NYU)
- **Kate Storey-Fisher** (DIPC)

## To-do / bugs / projects
- Assumes the relevant files exist in the right places.
- Replace the `for` loop with `map()`.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import fitsio
from pathlib import Path
import os
import time
import datetime
import urllib.request 

import dipole
from multipoles import multipole_map
from tools import get_galactic_plane_mask

NSIDE = 64
RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def generate_mocks_from_cases():
    """
    main loop
    """
    set_name = 'binary_quaia'
    dir_mocks = os.path.join(RESULTDIR, 'data/mocks', set_name)
    Path.mkdir(Path(dir_mocks), exist_ok=True, parents=True)

    case_dicts = case_set(set_name=set_name, excess=1e-5)
    n_trials_per_case = 12 # magic

    overwrite = False

    for case_dict in case_dicts:
        payload = get_payload(case_dict) 
        for i in range(n_trials_per_case):
            # print(f"making hash {hash((case_dict['tag'], i)) % 2**32}")
            rng = np.random.default_rng(hash((case_dict['tag'], i)) % 2**16)
            trial_name = f"mock{case_dict['tag']}_trial{i:03d}"
            fn_mock = os.path.join(dir_mocks, trial_name)
            if not overwrite and os.path.exists(f"{fn_mock}.npy"):
                # print(f"trial {i} already exists at {fn_mock} and overwrite is False")
                continue
            mock = generate_mock(payload, rng, trial=i) 
            print(f"writing file {fn_mock}")
            np.save(fn_mock, mock)
            fig = plt.figure()
            hp.mollview(dipole.overdensity_map(mock, payload['selfunc']),
                        coord=['C','G'], title=trial_name, fig=fig,
                        min=-0.25, max=0.25)
            plt.savefig(f"{fn_mock}.png")
            plt.close(fig)

def generate_mocks_from_grid():

    set_name = 'grid_catwise'

    dir_mocks = os.path.join(RESULTDIR, 'data/mocks', set_name)
    Path.mkdir(Path(dir_mocks), exist_ok=True, parents=True)

    case_dicts = grid_case_set(set_name, n_amps=20, n_excess=10)

    n_trials_per_case = 12
    overwrite = False

    for j, case_dict in enumerate(case_dicts):
        payload = get_payload(case_dict)
        for i in range(n_trials_per_case):
            rng = np.random.default_rng(hash((case_dict['tag'], i)) % 2**16)
            trial_name = f"mock{case_dict['tag']}_trial{i:03d}"
            fn_mock = os.path.join(dir_mocks, trial_name)
            if not overwrite and os.path.exists(fn_mock+'.npy'):
                print(f"trial {i} already exists at {fn_mock} and overwrite is False", flush=True)
                continue
            mock = generate_mock(payload, rng, trial=i) 
            print(f"{j+1} of {len(case_dicts)}:\twriting file {fn_mock}", flush=True)
            np.save(fn_mock, mock)


def case_set(set_name='full', excess=1e-5):
    """
    Define cases combinatorially from choices.

    Parameters
    ----------
    set_name : str, optional

    excess : float, optional
        Amount of excess power to add (flat in Cell space) when the case includes excess power.

    Returns
    -------
    List of dicts; each dict is a description of one case.

    Bugs/Comments:
    - base_rates is hard-coded.
    """
    quaia_base_rate = 33.64 # magic: * calculated from average of (old_base_rate. (=35.) * (data_mean / mock_mean)) where mocks
                            #           have been generated using old_base_rate (dipole amplitude 0.0052)
                            #   -> criterion that the mean masked mean quasar density of mocks generated with new_base_rate
                            #       be within 0.1% discrepancy of mean masked quasar density of the data
    catwise_base_rate = 72.42 # magic * calculated from old_base_rate = 70.
    quaia_dipole_amp = 0.0052
    catwise_dipole_amp = 0.0074
    if set_name == 'full':
        Cell_modes = ['excess', 'zeros', ] # 'datalike']
        selfunc_modes = ['quaia_G20.0_orig', 'catwise_zodi', 'ones', 'binary', ]
        #0.0052 is expected for Quaia; 0.0074 for catwise. pulled from a random notebook, go do this properly!
        dipole_amps = [0.,
                        quaia_dipole_amp, quaia_dipole_amp * 2,
                        catwise_dipole_amp, catwise_dipole_amp * 2] # magic 
        base_rates = [quaia_base_rate, catwise_base_rate]
    elif set_name == 'excess_quaia':
        Cell_modes = ['excess']
        selfunc_modes = ['quaia_G20.0_orig'] # 'ones', 'binary', 
        dipole_amps = [quaia_dipole_amp] # 0., 
        base_rates = [quaia_base_rate]
    elif set_name == 'excess_catwise':
        Cell_modes = ['excess']
        selfunc_modes = ['binary', 'ones'] # 'catwise_zodi', 
        dipole_amps = [catwise_dipole_amp] # 0., 
        base_rates = [catwise_base_rate]
    elif set_name == 'ideal_quaia':
        Cell_modes = ['zeros']
        selfunc_modes = ['ones']
        dipole_amps = [quaia_dipole_amp]
        base_rates = [0., quaia_base_rate]
    elif set_name == 'ideal_catwise':
        Cell_modes = ['zeros']
        selfunc_modes = ['ones'] # !! 'binary' instead of 'ones' (the true "ideal" case) to generate mocks matching S21 method
        dipole_amps = [catwise_dipole_amp]
        base_rates = [0., catwise_base_rate]
    elif set_name == 'binary_quaia':  # to generate mocks matching S21 method: same as "ideal" except binary mask instead of ones everywhere
        Cell_modes = ['zeros']
        selfunc_modes = ['binary']
        dipole_amps = [0., quaia_dipole_amp]
        base_rates = [quaia_base_rate]
    elif set_name == 'binary_catwise':  # to generate mocks matching S21 method: same as "ideal" except binary mask instead of ones everywhere
        Cell_modes = ['zeros']
        selfunc_modes = ['binary']
        dipole_amps = [catwise_dipole_amp]
        base_rates = [0., catwise_base_rate]
    elif set_name == 'shot_noise_quaia':
        Cell_modes = ['zeros']
        selfunc_modes = ['ones', 'binary']
        dipole_amps = [0.]
        base_rates = [quaia_base_rate]
    elif set_name == 'shot_noise_catwise':
        Cell_modes = ['zeros']
        selfunc_modes = ['ones'] #, 'binary']
        dipole_amps = [0.]
        base_rates = [catwise_base_rate]
    elif set_name == 'excess':
        Cell_modes = ['excess']
        selfunc_modes = ['ones', 'binary']
        dipole_amps = [0.]
        base_rates = [0.]
    elif set_name == 'zeros':
        Cell_modes = ['zeros']
        selfunc_modes = ['ones']
        dipole_amps = [0.]
        base_rates = [0.]
    elif set_name == 'full_quaia':
        Cell_modes = ['excess', 'zeros']
        selfunc_modes = ['quaia_G20.0_orig', 'ones', 'binary']
        dipole_amps = [0., quaia_dipole_amp, quaia_dipole_amp * 2]
        base_rates = [quaia_base_rate]
    elif set_name == 'full_catwise':
        Cell_modes = ['excess', 'zeros']
        selfunc_modes = ['catwise_zodi', 'ones', 'binary']
        dipole_amps = [0., catwise_dipole_amp, catwise_dipole_amp * 2]
        base_rates = [catwise_base_rate]
    cases = list(itertools.product(Cell_modes,
                                   selfunc_modes,
                                   dipole_amps,
                                   base_rates))
    case_dicts = []
    for case in cases:
        case_dict = {
            "Cell_mode": case[0],
            "selfunc_mode": case[1],
            "dipole_amp": case[2],
            "base_rate": case[3]
        }
        if case_dict['Cell_mode'] == 'excess':
            assert (excess > 0) and (excess < 1), "excess power must be between 0 and 1"
            case_dict['excess'] = excess
            case_dict["tag"] = f"_case-{case_dict['Cell_mode']}-{case_dict['excess']:.0e}-{case_dict['selfunc_mode']}-{case_dict['dipole_amp']:.4f}-{case_dict['base_rate']:.3f}"
        else:
            case_dict['excess'] = None
            case_dict["tag"] = f"_case-{case_dict['Cell_mode']}-{case_dict['selfunc_mode']}-{case_dict['dipole_amp']:.4f}-{case_dict['base_rate']:.3f}"
        case_dicts.append(case_dict)

    return case_dicts

def grid_case_set(set_name, n_amps, n_excess):
    """
    Define case payloads combinatorially from a grid of input dipole amplitudes and excess angular power.

    Returns
    -------
    List of dicts; each dict is a payload with one combination of dipole amplitude + excess power.

    Bugs/Comments:
    - repeated code with base_rates and dipole_amps !
    - base_rates is hard-coded
    """
    # copied from case_set()...
    quaia_dipole_amp = 0.0052
    catwise_dipole_amp = 0.0074

    if set_name == 'grid_catwise':
        base_rate = 72.42 # magic: * calculated from average of (old_base_rate. (=70.) * (data_mean / mock_mean)) where mocks
                            #           have been generated using old_base_rate (dipole amplitude 0.0074)
                            #   -> criterion that the mean masked mean quasar density of mocks generated with new_base_rate
                            #       be within 0.1% discrepancy of mean masked quasar density of the data
        selfunc_mode = 'catwise_zodi'
        dipole_amps = np.linspace(catwise_dipole_amp * 0.5, catwise_dipole_amp * 3., n_amps)
        excess_power = np.logspace(-6, -4, n_excess)
    else:
        assert set_name == 'grid_quaia', "set_name must be either 'grid_catwise' or 'grid_quaia"
        base_rate = 33.64 # magic * calculated from old_base_rate = 35.
        selfunc_mode = 'quaia_G20.0_orig'
        dipole_amps = np.linspace(quaia_dipole_amp * 0.5, quaia_dipole_amp * 3., n_amps)
        excess_power = np.logspace(-6, -4, n_excess)
    grid = list(itertools.product(dipole_amps, excess_power))

    case_dicts = []
    for case in grid:
        case_dict = {
            "Cell_mode" : "excess",
            "dipole_amp" : case[0],
            "excess" : case[1],
            "selfunc_mode" : selfunc_mode,
            "base_rate" : base_rate,
            "tag" : f"_case-excess-{case[1]:.2e}-{selfunc_mode}-{case[0]:.4f}-{base_rate:.3f}"
        }
        case_dicts.append(case_dict)
    return case_dicts


def get_payload(case_dict):
    """
    expand case choices into useful variables for data generation.
    """
    payload_dict = {
        "Cells": get_cells(case_dict['Cell_mode'], excess=case_dict['excess']), 
        "selfunc": get_selfunc_map(case_dict['selfunc_mode']),
        "dipole_amp": case_dict['dipole_amp'],
        "base_rate": case_dict['base_rate']
    }
    return payload_dict

def get_cells(cell_str, excess=1e-5):
    if cell_str == 'zeros':
        Cells = np.array([])
    elif cell_str == 'excess': # take this out to ell_max = 16 for no reason but hey. # magic
        assert excess is not None
        Cells = np.zeros(16) + excess  # magic # !! different excess Cell for CatWISE ??
#    elif cell_str == 'datalike':
#        Cells = np.array([0.007, 0.0014, 0.0021, 0., 0., 0., 0., 0.]) # magic
    else:
        raise ValueError("unknown cell_str")
    return Cells

def get_selfunc_map(selfunc_str, nside=NSIDE, blim=30):
    mask_fn = os.path.join(RESULTDIR, 'data/catalogs/masks/mask_master_hpx_r1.0.fits')
    # galactic plane cut: used in every case except 'ones'
    gal_plane_mask = get_galactic_plane_mask(blim, nside=NSIDE, frame='icrs') # 1 in unmasked, 0 in masked
    if selfunc_str == 'ones':
        selfunc_map = np.ones(hp.nside2npix(nside))
    elif selfunc_str == 'binary':
        selfunc_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'quaia_G20.0_orig':
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G20.0.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map # TODO check that this is right
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'quaia_G20.0_zodi':
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G20.0_pluszodis.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    elif 'zsplit2bin' in selfunc_str:
        Glim = selfunc_str.split('G')[1].split('_zsplit')[0] # gross!
        assert Glim in ['20.0', '20.5'], f"G lim {Glim} not 20.0 or 20.5"
        ibin = selfunc_str.split('_zsplit2bin')[1]
        assert ibin in ['0', '1'], f"bin {ibin} not 0 or 1"
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G{Glim}_zsplit2bin{ibin}.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'quaia_G20.5_orig':
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G20.5.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'quaia_G20.5_zodi':
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G20.5_pluszodis.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'catwise':
        fn_selfunc = f'../data/catalogs/catwise_agns/selfuncs/selection_function_NSIDE{nside}_catwise.fits'
        selfunc_map = hp.read_map(fn_selfunc)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'catwise_zodi':
        fn_selfunc = f'../data/catalogs/catwise_agns/selfuncs/selection_function_NSIDE{nside}_catwise_pluszodis.fits'
        selfunc_map = hp.read_map(fn_selfunc)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    elif selfunc_str == 'catwise_elatcorr':
        # note that catwise fiducial selfunc includes z
        fn_selfunc = f'../data/catalogs/catwise_agns/selfuncs/selection_function_NSIDE{nside}_catwise_elatcorr.fits'
        selfunc_map = hp.read_map(fn_selfunc)
        mask_map = fitsio.read(mask_fn) # mask saved in fits, not healpy save convention
        selfunc_map *= mask_map
        selfunc_map *= gal_plane_mask
    else:
        raise ValueError("unknown selfunc_str")
    return selfunc_map

def generate_mock(payload, rng, trial=0):
    """
    Parameters
    ----------
    payload : dict
        Cells, selfunc, and dipole amplitude.
    rng : numpy random number generator
    trial : int, optional
    
    Bugs/Comments:
    - Possible nside conflict between magic NSIDE and payload nside.
    """
    expected_dipole_map = generate_expected_dipole_map(payload['dipole_amp'])
    sph_harm_amp_dict = get_sph_harm_amp_dict(payload['Cells'], rng)
    smooth_overdensity_map = generate_smooth_overdensity_map(sph_harm_amp_dict)
    selfunc_map = payload['selfunc']
    base_rate = payload['base_rate']
    mock = generate_map(expected_dipole_map + smooth_overdensity_map, base_rate, selfunc_map, rng)
    return mock

def generate_expected_dipole_map(dipole_amplitude, nside=NSIDE):
    """
    Parameters
    ----------
    dipole_amplitude : float
        The amplitude of the dipole in the normal convention (not the alm convention),
        depends on the number counts.
    """
    amps = np.zeros(4)
    amps[1:] = dipole.cmb_dipole(amplitude=dipole_amplitude, return_amps=True)
    return dipole.dipole_map(amps, NSIDE=nside)

def get_sph_harm_amp_dict(Cells, rng):
    """
    Parameters
    ----------
    Cells : ndarray, type float, shape (ellmax-1,)
        Cells for ell = 1, 2, ..., ellmax
        Note: 1-indexed
    """
    sph_harm_amp_dict = {}
    for ell in range(1, len(Cells)+1):
        sph_harm_amp_dict[ell] = np.sqrt(Cells[ell-1]) * rng.normal(size=2 * ell + 1)
    return sph_harm_amp_dict

def generate_smooth_overdensity_map(sph_harm_amp_dict, nside=NSIDE):
    """
    Parameters
    ----------
    sph_harm_amp_dict : dict
        Generated by get_sph_harm_amp_dict().
    nside : int, optional
    """
    mock_map = np.zeros((hp.nside2npix(nside)))
    for ell in sph_harm_amp_dict.keys():
        alms = sph_harm_amp_dict[ell]
        assert len(alms) == 2 * ell + 1, \
            f"incorrect number of coefficients for ell={ell} ({len(alms)}, expected {2 * ell + 1}"
        mock_map += multipole_map(alms)
    return mock_map

def generate_map(overdensity_map, base_rate, selfunc_map, rng):
    """
    Parameters
    ----------
    overdensity_map : healpix map
        Sum of expected dipole map and smooth_overdensity_map().
    base_rate : float
        Quasar rate per pixel in the overdensity=0, selection function=1 regions of the sky.
    selfunc_map : healpix map
        Selection function (mask or continuous).
    rng : numpy random number generator
    
    Bugs/Comments
    --------------
    - This function will only operate if there's a global variable called NSIDE.
    - If overdensity_map < -1 anywhere, this code will fail.
    - If base_rate == 0, the map is 0 everywhere.
    """
    if base_rate == 0:
        return (1. + overdensity_map) * selfunc_map
    else:
        assert base_rate > 0, "base_rate cannot be negative"
        return rng.poisson((1. + overdensity_map) * base_rate * selfunc_map)

if __name__ == "__main__":
    s = time.time()
    generate_mocks_from_cases()
    total_time = time.time() - s
    print(f"total time: {datetime.timedelta(seconds=total_time)}", flush=True)