import numpy as np
import os
import sys
import healpy as hp

import tools
import multipoles

def main():

    """
    Smooth a set of posterior and prior mock quasar maps from an ABC result.
    """

    # inputs
    dipamp = 0.0
    base_rate = 33.633
    nmocks = 500
    ngens = 15
    nside = 64

    selfunc = np.ones(hp.nside2npix(nside))

    Lambda = 0.

    overwrite = False

    n_to_compute = 500  # will take the first n mocks from the accepted samples, and generate(&smooth) n mocks from the prior

    # directory things
    result_dir = os.path.join('/scratch/aew492/lss-dipoles_results/results/ABC/fake_data',
                                    f'dipole-{dipamp:.1f}_no_excess_base-rate-{base_rate:.4f}_ones_shot-noise-only',
                                    f'{nmocks}mocks_{ngens}gens_2025-04-02')
    posterior_dir = os.path.join(result_dir, 'accepted_samples')
    post_mocks_fn_list = [
        os.path.join(posterior_dir, f'mock{i}') for i in range(n_to_compute)
    ]

    Cells_dir = os.path.join(posterior_dir, 'Cells')
    if not os.path.exists(Cells_dir):
        os.makedirs(Cells_dir)
    save_fn_list = [
        os.path.join(Cells_dir, f'Cells_Lambda-{Lambda:.1f}_mock{i}') for i in range(n_to_compute)
    ]

    fns_to_compute = post_mocks_fn_list.copy()
    save_fn_list_to_compute = save_fn_list.copy()
    if overwrite == False:
        for i in range(n_to_compute):
            if os.path.exists(save_fn_list[i]):
                fns_to_compute.remove(post_mocks_fn_list[i])
                save_fn_list_to_compute.remove(save_fn_list_to_compute[i])
    print(f"computing {len(fns_to_compute)} Cells", flush=True)

    """
    compute Cells from the posterior mocks
    """

    for i, mock_fn in enumerate(fns_to_compute):
        print(f"mock {i}", flush=True)
        mock = np.load(mock_fn + '.npy')
        odmap = mock / np.nanmean(mock) - 1
        ells, Cells, alms = multipoles.compute_Cells_in_overdensity_map_Lambda(odmap,
                                                                Lambda=Lambda,
                                                                max_ell=8, # magic
                                                                selfunc=selfunc,
                                                                return_alms=True)
        np.save(save_fn_list_to_compute[i], {'ells' : ells, 'Cells' : Cells, 'alms' : alms})

    print("done!", flush=True)


if __name__ == '__main__':
    main()