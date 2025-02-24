import numpy as np
import os
import sys
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp

import tools
from abc_with_fake_data import model
import multipoles
import dipole

def main():

    """
    Run some function on a list of mocks.
    """

    # inputs
    # dipamp = 0.0052
    # excess = '1e-4'

    base_rate = 77.4495 # for Catwise, 33.633 # for Quaia
    nmocks = 500
    ngens = 20
    ell_max = 8

    # regularization for the Cells
    Lambda = 1e-3

    n = 500  # number to actually compute (will take the first n)
    overwrite = False

    # info to load
    result_dir = os.path.join('/scratch/aew492/lss-dipoles_results/results/ABC/',
                                f'catwise_dipole_excess_nside1_{nmocks}mocks_{ngens}iters_base-rate-{base_rate:.4f}')
                                    # f'fake_data/dipole-{dipamp:.4f}_excess-{excess}_base-rate-{base_rate:.4f}',
                                    # f'{nmocks}mocks_{ngens}gens')
    posterior_dir = os.path.join(result_dir, 'accepted_samples')
    fn_list = [
        os.path.join(posterior_dir, f'mock{i}.npy') for i in range(n)
    ]

    # where to save
    save_dir = os.path.join(result_dir, 'accepted_samples_Cells', f'Lambda-{Lambda:.0e}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn_list = [
        os.path.join(save_dir, f'Cells_mock{i}.npy') for i in range(n)
    ]

    # load the ABC results
    resdict = np.load(os.path.join(result_dir, f'results.npy'), allow_pickle=True).item()
    try:
        selfunc = resdict['selfunc']
    except KeyError:
        print(f"'selfunc' key not found; assuming perfect completeness")
        selfunc = np.ones_like(np.load(fn_list[i])).astype(float)

    """
    loop through the mocks
    """

    for i, fn in enumerate(fn_list):
        print(f"mock {i+1:01d} of {len(fn_list)}, {(i+1)/len(fn_list) * 100:.0f}%", end='\r', flush=True)
        # does this save file already exist?
        if os.path.exists(save_fn_list[i]) and overwrite == False:
            continue

        # load the mock
        mock = np.load(fn).astype(float)

        # compute the Cells on the mock:
        # convert mock to overdensities
        odmap = dipole.overdensity_map(mock, selfunc)
        ells, Cells, alms = multipoles.compute_Cells_in_overdensity_map_Lambda(odmap,
                                                                Lambda=Lambda,
                                                                max_ell=ell_max,
                                                                selfunc=selfunc,
                                                                return_alms=True)

        # package results
        res = dict(ells=ells, Cells=Cells, alms=alms, Lambda=Lambda)

        # save
        np.save(save_fn_list[i], res)

    print("done!")


if __name__ == '__main__':
    main()