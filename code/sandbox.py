import numpy as np
import os
import sys
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp

import tools
from abc_with_fake_data import model

def main():

    """
    Smooth a set of posterior and prior mock quasar maps from an ABC result.
    """

    # inputs
    dipamp = 0.0052
    excess = '1e-4'
    base_rate = 33.633
    nmocks = 500
    ngens = 15
    nside = 64
    sr = 1

    n_to_smooth = 500  # will take the first n mocks from the accepted samples, and generate(&smooth) n mocks from the prior
    overwrite = True

    # directory things
    result_dir = os.path.join('/scratch/aew492/lss-dipoles_results/results/ABC/fake_data',
                                    f'dipole-{dipamp:.4f}_excess-{excess}_base-rate-{base_rate:.4f}',
                                    f'{nmocks}mocks_{ngens}gens')
    posterior_dir = os.path.join(result_dir, 'accepted_samples')
    prior_dir = os.path.join(result_dir, 'prior_samples')
    if not os.path.exists(prior_dir):
        os.makedirs(prior_dir)
    post_mocks_fn_list = [
        os.path.join(posterior_dir, f'mock{i}') for i in range(n_to_smooth)
    ]
    prior_mocks_fn_list = [
        os.path.join(prior_dir, f'mock{i}') for i in range(n_to_smooth)
    ]

    post_mocks_to_generate = post_mocks_fn_list.copy()
    prior_mocks_to_generate = prior_mocks_fn_list.copy()

    # do the smoothed mocks exist?
    if overwrite == False:
        for i in range(n_to_smooth):
            if os.path.exists(post_mocks_fn_list[i]+'_smoothed.npy'):
                post_mocks_to_generate.remove(post_mocks_fn_list[i])
            if os.path.exists(prior_mocks_fn_list[i]+'_smoothed.npy'):
                prior_mocks_to_generate.remove(prior_mocks_fn_list[i])

    # load the ABC results
    resdict = np.load(os.path.join(result_dir, f'results.npy'), allow_pickle=True).item()

    """
    get mocks from the prior and posterior
    """

    # ## POSTERIOR ##
    # print("smoothing the posterior mocks")
    # for i, fn in enumerate(post_mocks_to_generate):
    #     print(f"{i+1} of {len(post_mocks_to_generate)}")
    #     mock = np.load(fn+'.npy')
    #     smoothed_mock = tools.smooth_map(mock, sr=sr)
    #     np.save(fn+'_smoothed.npy', smoothed_mock)


    ## PRIOR ##

    # required inputs for the sky model
    selfunc = np.ones(hp.nside2npix(nside))
    dipdir = SkyCoord(264, 48, unit=u.deg, frame='galactic')
    theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))
    model_args = dict(base_rate=base_rate, selfunc=selfunc, dipdir=dipdir, theta=theta, phi=phi, ell_max=resdict['ell_max'])

    prior_mocks = tools.generate_mocks_from_prior(resdict['prior'], model, len(prior_mocks_to_generate), nside, **model_args)
    for i, fn in enumerate(prior_mocks_to_generate):
        np.save(fn, prior_mocks[i])
    # for i, fn in enumerate(prior_mocks_to_generate):
    #     print(f"{i+1} of {len(prior_mocks_to_generate)}")
    #     smoothed_mock = tools.smooth_map(prior_mocks[i], sr=sr)
    #     np.save(fn+'_smoothed.npy', smoothed_mock)

    print("done!")


if __name__ == '__main__':
    main()