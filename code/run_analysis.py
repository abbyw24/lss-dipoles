"""
# run multiple analyses of mocks and real data

## License
Copyright 2024 The authors.
This code is released for re-use under the open-source MIT License.

## Authors:
- **Abby Williams** (Chicago)
- **David W. Hogg** (NYU)
- **Kate Storey-Fisher** (DIPC)

## To-do / bugs / projects / comments
- The `for` loops here should me `map()`.
- This code's clobber mode wouldn't work if we switch to `map()`.
- Name synchronization between the analyses on the mocks and the real data (which "deeply upsets" Hogg).
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import healpy as hp

import generate_mocks as gm
import dipole
import multipoles
import tools

RESULTDIR = '/scratch/aew492/lss-dipoles_results'

def main():

    set_name = 'binary_quaia'

    dir_mocks = os.path.join(RESULTDIR, 'data/mocks', set_name)
    dir_results = os.path.join(RESULTDIR, 'results/results_mocks', set_name)
    Path.mkdir(Path(dir_results), exist_ok=True, parents=True)

    # case_dicts = gm.grid_case_set(set_name=set_name, n_amps=20, n_excess=10)
    case_dicts = gm.case_set(set_name, excess=1e-5)
    print(f"got {len(case_dicts)} cases", flush=True)   

    analyze_mocks(case_dicts, dir_mocks, dir_results, Lambdas=[0., 1e-3, 1e-2],
                     overwrite=False, compute_Cells=True, max_mocks_for_Cells=12)
    # analyze_data(overwrite=False)


def get_fns_to_analyze(dir_mocks, case_dict, max_mocks=None):
    """
    Returns a list of mock files, given a mock `case_dict` and parent directory `dir_mocks`.
    If `max_mocks` is not `None`, only returns the first `max_mocks` trials that match the
    case_dict.

    """
    pattern = f"{dir_mocks}/*{case_dict['tag']}*.npy"
    fns_mock = glob.glob(pattern)
    print(f"found {len(fns_mock)} files for mock case {pattern}", flush=True)
    # optionally, only analyze a certain number of mocks
    fns_mock = tools.filter_max_mocks(fns_mock, max_mocks)
    print(f"analyzing the first {len(fns_mock)}", flush=True)

    return fns_mock

def analyze_mocks(case_dicts, dir_mocks, dir_results, Lambdas=[0.],
                    overwrite=False, compute_Cells=True, max_mocks_for_Cells=None):
    """
    Analyzes the mock data generated by the `generate_mocks` module.

    This function is a wrapper of the analysis function 'analyze_dipole' and 'compute_Cells' for the mock data;
    it reads the mock data from the specified directory for the set of cases in
    'case_set()' and loops over them.

    Parameters:
        case_dicts : dictionary
        dir_mocks : str
        dir_results : str
        Lambdas : list of float, optional
        overwrite : bool, optional
        compute_Cells : bool, optional
        max_mocks_for_Cells : int or None, optional

    Returns:
        None

    Bugs/Comments:
    - Added `compute_Cells` toggle in the case of many mocks for dipole vs. Lambda analysis
    """

    for j, case_dict in enumerate(case_dicts):

        fns_to_analyze = get_fns_to_analyze(dir_mocks, case_dict, max_mocks=None)
        fns_to_analyze_Cells = get_fns_to_analyze(dir_mocks, case_dict, max_mocks=max_mocks_for_Cells)

        for i, fn_mock in enumerate(fns_to_analyze):
            print(f"case {j+1} of {len(case_dicts)}: mock {i+1} of {len(fns_to_analyze)}", flush=True)
            mock = np.load(fn_mock, allow_pickle=True)
            # print(f"analyze_mocks(): loaded file {fn_mock}", flush=True)

            fn_res = os.path.join(dir_results, f"dipole_comps_lambdas_" + fn_mock.split('/')[-1])
            if not os.path.exists(fn_res) or overwrite:
                Lambda_grid, comps = analyze_dipole(mock, case_dict)
                result_dict = {
                    "Lambdas" : Lambda_grid,
                    "dipole_comps" : comps
                }
                np.save(fn_res, result_dict)
                print(f"analyze_dipole(): wrote {fn_res}", flush=True)

            if compute_Cells and fn_mock in fns_to_analyze_Cells:

                for Lambda in Lambdas:
                    fn_res = os.path.join(dir_results, f"Cells_Lambda-{Lambda:.1e}_" + fn_mock.split('/')[-1])
                    if os.path.exists(fn_res) and not overwrite:
                        print(f"{fn_res} exists and overwrite is {overwrite}")
                        # wouldn't work if we switch to map
                        continue
                    ells, Cells, alms = analyze_Cells(mock, case_dict, Lambda)
                    result_dict = {
                        "ells" : ells,
                        "Cells" : Cells,
                        "alms" : alms,
                        "Lambda" : Lambda
                    }
                    np.save(fn_res, result_dict)
                    print(f"analyze_Cells(): wrote {fn_res}", flush=True)


def analyze_data(overwrite=False):
    """
    Analyzes the data from the specified catalog file and saves the results.

    This function is a wrapper of the analysis functions 'analyze_dipole' and 'analyze_Cells' for the data;
    there are cases for both quaia and catwise to pull the proper paths.

    Parameters:
        overwrite (bool)

    Returns:
        None

    Bugs/Comments:
    - current analyzes both Quaia and CatWISE; do we want this or a toggle?
    """
    # quaia settings
    catalog_name = 'quaia_G20.0'
    fn_cat = os.path.join(RESULTDIR, 'data/catalogs/quaia/quaia_G20.0.fits')
    selfunc_mode = 'quaia_G20.0_orig'
    quaia_settings = [catalog_name, fn_cat, selfunc_mode]

    # catwise settings
    catalog_name = 'catwise'
    fn_cat = os.path.join(RESULTDIR, f'data/catalogs/catwise_agns/catwise_agns_master.fits')
    selfunc_mode = 'catwise_zodi'
    catwise_settings = [catalog_name, fn_cat, selfunc_mode]

    nside = 64  # magic
    dir_results = os.path.join(RESULTDIR, 'results/results_data')
    Path.mkdir(Path(dir_results), exist_ok=True, parents=True)

    for [catalog_name, fn_cat, selfunc_mode] in [quaia_settings, catwise_settings]:

        print(f"analyzing {catalog_name}", flush=True)

        # turn the source table into a healpix map (*NO masks or plane cuts yet!)
        qmap = tools.load_catalog_as_map(fn_cat, frame='icrs', nside=nside)

        case_dict = {
            "catalog_name": catalog_name, #maybe we shouldnt need this here...? think about it!
            "selfunc_mode": selfunc_mode, #this also multiplies in the mask
            "tag": f"_case-{selfunc_mode}"
        }
        fn_res = os.path.join(dir_results, f"dipole_comps_lambdas_{case_dict['catalog_name']}{case_dict['tag']}.npy")
        
        fig = plt.figure()
        hp.mollview(dipole.overdensity_map(qmap, selfunc=gm.get_selfunc_map(selfunc_mode)), 
                    coord=['C','G'], title=f"{case_dict['catalog_name']}{case_dict['tag']}", 
                    min=-0.5, max=0.5,
                    cmap='RdBu',
                    fig=fig)
        plt.savefig(f"{fn_res[:-4]}.png")
        print(f"Saved figure to {fn_res[:-4]}.png", flush=True)
        plt.close(fig)
            
        if os.path.exists(fn_res) and not overwrite:
            return
        
        Lambdas, comps = analyze_dipole(qmap, case_dict)
        result_dict = {
            "Lambdas" : Lambdas,
            "dipole_comps" : comps
        }
        np.save(fn_res, result_dict)
        print("Saved dipole results to", fn_res, flush=True)

        for Lambda in [0., 1e-4, 1e-3, 1e-2, 1e-1]: # magic
            fn_res = os.path.join(dir_results, f"Cells_Lambda-{Lambda:.1e}_{case_dict['catalog_name']}{case_dict['tag']}.npy")
            if os.path.exists(fn_res) and not overwrite:
                continue
            ells, Cells, alms = analyze_Cells(qmap, case_dict, Lambda)
            result_dict = {
                "ells" : ells,
                "Cells" : Cells,
                "alms" : alms,
                "Lambda" : Lambda
            }
            np.save(fn_res, result_dict)
            print(f"Saved Cell results to {fn_res}", flush=True)

def analyze_dipole(qmap, case_dict):
    Lambdas = np.geomspace(1e-3, 1e0, 33) # magic
    selfunc = gm.get_selfunc_map(case_dict['selfunc_mode'])
    odmap = dipole.overdensity_map(qmap, selfunc)
    comps = np.zeros((len(Lambdas), 3))
    for i, Lambda in enumerate(Lambdas):
        comps[i] = dipole.measure_overdensity_dipole_Lambda(odmap,
                                                            Lambda=Lambda,
                                                            selfunc=selfunc)
    return Lambdas, comps

def analyze_Cells(qmap, case_dict, Lambda):
    selfunc = gm.get_selfunc_map(case_dict['selfunc_mode'])
    odmap = dipole.overdensity_map(qmap, selfunc)
    ells, Cells, alms = multipoles.compute_Cells_in_overdensity_map_Lambda(odmap,
                                                                Lambda=Lambda,
                                                                max_ell=8, # magic
                                                                selfunc=selfunc,
                                                                return_alms=True)
    return ells, Cells, alms

if __name__ == "__main__":
    main()
