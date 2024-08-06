import numpy as np
import healpy as hp
import astropy.units as u
import random
import time
import datetime
import os
import sys

import tools
import ellisbaldwin
import dipole
from qso_sample import QSOSample
from multipoles import compute_Cells_in_overdensity_map_Lambda, reconstruct_map, multipole_map


def main():

    s = time.time()

    """
    MAIN INPUTS
    """
    sample = 'quaia'

    """
    CONSTRUCT SAMPLE AND CALCULATE EXPECTED DIPOLE AMPLITUDE
    """
    catdir = '/scratch/aew492/quasars/catalogs'
    mask_fn = os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits')

    if sample == 'quaia':
        d = QSOSample(initial_catfn=os.path.join(catdir, 'quaia/quaia_G20.5.fits'),
                    mask_fn=mask_fn,
                    mag='g', maglim=20.,
                    blim=30)
        selfunc_fn = os.path.join(catdir, f'quaia/selfuncs/selection_function_NSIDE{d.NSIDE}_G20.0_blim15.fits')
        expected_dipamp = ellisbaldwin.compute_expected_dipole_gaia(d.table, maglimit=d.maglim,
                                                            min_g=19.5, max_g=20.5)
    else:
        assert sample == 'catwise_agns'
        d = QSOSample(initial_catfn=os.path.join(catdir, 'catwise_agns/catwise_agns_master.fits'),
                    mask_fn=mask_fn,
                    mag='w1', maglim=16.4,
                    blim=30)
        selfunc_fn = os.path.join(catdir, f'catwise_agns/selfuncs/selection_function_NSIDE{d.NSIDE}_catwise_pluszodis.fits')
        expected_dipamp = ellisbaldwin.compute_expected_dipole_wise(d.table, maglimit=d.maglim,
                                                                min_w1=16., max_w1=16.5)

    d.cut_mag()  # cut all sources fainter than the input magnitude limit
    d.cut_galactic_plane_hpx()  # cut all sources with |b|<blim from the working source table
    selfunc = d.get_selfunc(selfunc=selfunc_fn) # load selection function

    # get masked datamap
    masked_datamap = d.construct_masked_datamap(selfunc=selfunc, return_map=True)
    # mean density in the masked datamap
    mu = np.nanmean(masked_datamap)

    print(f"\nmean density in masked datamap = {mu:.2f} quasars per healpixel\n" + \
            f"expected dipole amplitude = {expected_dipamp:.6f}", flush=True)

    """
    INPUT DIPOLE COMPONENTS
    """
    # CMB dipole direction
    dipdir = dipole.cmb_dipole()[1]
    input_dipole_comps = hp.dir2vec(np.pi/2 - dipdir.dec.rad, dipdir.ra.rad)  # norm = 1.
    input_dipole_comps *= expected_dipamp
    print("input dipole components = ", input_dipole_comps, flush=True)

    """
    GENERATE MOCK HEALPIX MAPS AND MEASURE SPHERICAL HARMONIC COEFFICIENTS
    """
    # inputs
    max_ell = 1
    sph_harm_dict = {
        1 : input_dipole_comps
    }
    if max_ell > 1:
        for i, ell in enumerate(range(2, max_ell+1)):
            sph_harm_dict[ell] = np.zeros(2 * ell + 1)
    noise = True
    ntrials = 10
    regularize = True

    if regularize:
        if sample == 'quaia':
            Lambda_dict = np.load(f'/scratch/aew492/quasars/regularization/Lambdas_quaia_noise-matched.npy',
                                    allow_pickle=True).item()
        else:
            assert sample == 'catwise_agns'
            Lambda_dict = np.load(f'/scratch/aew492/quasars/regularization/Lambdas_catwise_noise-matched.npy',
                                    allow_pickle=True).item()
        Lambda = Lambda_dict[max_ell]
    else:
        Lambda = 0.
    print(f"Lambda = {Lambda:.3e}", flush=True)

    Cells_trials = np.empty((ntrials, max_ell))
    dipole_comps_trials = np.empty((ntrials, 3))
    for itrial in range(ntrials):
        print(f"trial {itrial}", flush=True)
        # generate mock healpix density map
        mock_overdensity_map = generate_mock_overdensity_map(sph_harm_dict, noise=noise, mu=mu, verbose=False)
        # mask
        mock_overdensity_map[~d.mask] = np.nan
        print(f"\tmean overdensity = {np.nanmean(mock_overdensity_map):.3e}", flush=True)

        ells, Cells, alms = compute_Cells_in_overdensity_map_Lambda(mock_overdensity_map, Lambda=Lambda,
                                                                    max_ell=max_ell, return_alms=True)
        # add this trial
        Cells_trials[itrial] = Cells[1:]
        dipole_comps_trials[itrial] = alms[1:4]
        print(f"\tbest-fit dipole components = ", alms[1:4], f"\n\tCells = ", Cells[1:],
                flush=True)

    Cells = np.nanmean(Cells_trials, axis=0)
    dipole_comps = np.nanmean(dipole_comps_trials, axis=0)
    print("Cells = ", Cells, flush=True)
    print("best-fit dipole components = ", dipole_comps, flush=True)

    """
    SAVE
    """
    res = dict(Cells=Cells, dipole_comps=dipole_comps, ntrials=ntrials, max_ell=max_ell, Lambda=Lambda,
                input_dipole_comps=input_dipole_comps)
    save_dir = f'/scratch/aew492/quasars/mock_map_results/{sample}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    noise_tag = '_noise' if noise else ''
    reg_tag = '_no-reg' if regularize == False else ''
    save_fn = os.path.join(save_dir, f'mock_maps{noise_tag}{reg_tag}_ellmax{max_ell}_{ntrials}trials.npy')
    np.save(save_fn, res)
    print(f"saved to {save_fn}", flush=True)
    total_time = time.time()-s
    print(f"total time = {datetime.timedelta(seconds=total_time)}", flush=True)

def generate_mock_overdensity_map(sph_harm_dict, noise=True, mu=None, nside=64, verbose=False):

    mock_map = np.zeros((hp.nside2npix(nside)))

    # noise
    if noise == True:
        assert float(mu) > 0, "mu must be a non-zero scalar if noise == True"
        if verbose:
            print(f"adding Poisson noise with amplitude {mu:.2f}", flush=True)
        noise_map = tools.generate_noise_map(mu, nside)
        mock_map += noise_map / np.nanmean(noise_map) - 1
    
    # add the spherical harmonics that we want
    if verbose:
        print(f"adding ells {sph_harm_dict.keys()}", flush=True)
    for i, ell in enumerate(sph_harm_dict.keys()):
        alms = sph_harm_dict[ell]
        assert len(alms) == 2 * ell + 1, \
            f"incorrect number of coefficients for ell={ell} ({len(alms)}, expected {2 * ell + 1}"
        mock_map += multipole_map(alms)

    return mock_map

def generate_mock_dipole_map(dipole_comps, noise=True, mu=None, nside=64, verbose=False):

    assert len(dipole_comps) == 3

    mock_map = np.zeros((hp.nside2npix(nside)))

    # noise
    if noise == True:
        assert float(mu) > 0, "mu must be a non-zero scalar if noise == True"
        if verbose:
            print(f"adding Poisson noise with amplitude {mu:.2f}", flush=True)
        noise_map = tools.generate_noise_map(mu, nside)
        mock_map += noise_map / np.nanmean(noise_map) - 1
    
    # add the dipole
    mock_map += dipole.dipole_map(np.append(0, dipole_comps))

    return mock_map


if __name__ == '__main__':
    main()