import itertools
import numpy as np
import healpy as hp
from pathlib import Path

import dipole
from multipoles import multipole_map

NSIDE = 64

def main():
    generate_mocks_from_cases()


def generate_mocks_from_cases():

    dir_mocks = '../data/mocks'
    Path.mkdir(Path(dir_mocks), exist_ok=True, parents=True)

    case_dicts = case_set()
    n_trials_per_case = 12

    for case_dict in case_dicts:
        payload = get_payload(case_dict) 
        for i in range(n_trials_per_case):
            mock = generate_mock(payload, trial=i) 
            fn_mock = f"{dir_mocks}/mock{case_dict['tag']}_trial{i}.npy"
            print(f"writing file {fn_mock}")
            np.save(fn_mock, mock)


def case_set():

    Cell_modes = ['zeros', 'flat']#, 'datalike']
    selfunc_modes = ['ones', 'binary', 'quaia_G20.0_orig']
    #0.0052 is expected for Quaia; 0.0074 for catwise. pulled from a random notebook, go do this properly!
    dipole_amps = [0.0, 0.0052, 0.0052*2] #magic 

    arrs = [Cell_modes, selfunc_modes, dipole_amps]
    cases = list(itertools.product(*arrs))

    case_dicts = []
    for case in cases:
        case_dict = {
            "Cell_mode": case[0],
            "selfunc_mode": case[1],
            "dipole_amp": case[2]
        }
        case_dict["tag"] = f"_case-{case_dict['Cell_mode']}-{case_dict['selfunc_mode']}-{case_dict['dipole_amp']:.5f}"
        case_dicts.append(case_dict)

    return case_dicts


def get_payload(case_dict):
    payload_dict = {
        "Cells": get_cells(case_dict['Cell_mode']), # write this function!
        "selfunc": get_selfunc_map(case_dict['selfunc_mode']), # write this function!
        "dipole_amp": case_dict['dipole_amp']
    }
    return payload_dict


def get_cells(cell_str):
    if cell_str == 'zeros':
        Cells = np.zeros(8)
    elif cell_str == 'flat':
        Cells = np.zeros(8) + 1e-5  # magic
    elif cell_str == 'datalike':
        Cells = np.array([0.007, 0.014, 0.021, 0., 0., 0., 0., 0.]) # magic
    else:
        raise ValueError("unknown cell_str")
    return Cells


def get_selfunc_map(selfunc_str, nside=NSIDE):
    mask_fn = '../data/catalogs/masks/mask_master_hpx_r1.0.fits'
    if selfunc_str == 'ones':
        selfunc_map = np.ones(hp.nside2npix(nside))
    elif selfunc_str == 'binary':
        selfunc_map = hp.read_map(mask_fn)
    elif selfunc_str == 'quaia_G20.0_orig':
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G20.0.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = hp.read_map(mask_fn)
        selfunc_map *= mask_map # TODO check that this is right
    elif selfunc_str == 'quaia_G20.0_zodi':
        fn_selfunc_quaia = f'../data/catalogs/quaia/selfuncs/selection_function_NSIDE{nside}_G20.0_pluszodis.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = hp.read_map(mask_fn)
        selfunc_map *= mask_map 
    elif selfunc_str == 'catwise_zodi':
        # note that catwise fiducial selfunc includes z
        fn_selfunc_quaia = f'../data/catalogs/catwise_agns/selfuncs/selection_function_NSIDE{nside}_catwise_pluszodis.fits'
        selfunc_map = hp.read_map(fn_selfunc_quaia)
        mask_map = hp.read_map(mask_fn)
        selfunc_map *= mask_map
    else:
        raise ValueError("unknown selfunc_str")
    return selfunc_map


def generate_mock(payload, rng=None, trial=0):
    """
    Parameters
    ----------
    payload : dict
        Cells, selfunc, and dipole amplitude.
    rng : numpy random number generator, optional
    trial : int, optional
    
    Bugs/Comments:
    - Possible nside conflict between magic NSIDE and payload nside.
    - base_rate is hard-coded.
    """
    if rng is None:
        rng = np.random.default_rng(17)
    expected_dipole_map = generate_expected_dipole_map(payload['dipole_amp'])
    sph_harm_amp_dict = get_sph_harm_amp_dict(payload['Cells'], rng)
    smooth_overdensity_map = generate_smooth_overdensity_map(sph_harm_amp_dict)
    selfunc_map = payload['selfunc']
    base_rate = 35.0    # magic
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
    """
    return rng.poisson((1. + overdensity_map) * base_rate * selfunc_map)


if __name__ == "__main__":
    main()