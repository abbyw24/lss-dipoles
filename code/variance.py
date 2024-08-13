import numpy as np
import healpy as hp
import astropy.units as u
import random
import time
import datetime
import os
import sys

import tools
from qso_sample import QSOSample
from multipoles import compute_Cells_in_overdensity_map_Lambda, reconstruct_map

def main():

    s = time.time()

    """
    MAIN INPUTS
    """
    sample = 'catwise_agns'
    Lambdas = np.logspace(-3, 1, 15)
    max_ells = [1,2,3,4,5,6,7,9]
    save_fns = [
        f'/scratch/aew492/quasars/regularization/variance_{sample}_ellmax{max_ell}.npy' for max_ell in max_ells
    ]

    """
    CONSTRUCT SAMPLE
    """
    catdir = '/scratch/aew492/quasars/catalogs'
    mask_fn = os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits')

    if sample == 'quaia':
        d = QSOSample(initial_catfn=os.path.join(catdir, 'quaia/quaia_G20.5.fits'),
                    mask_fn=mask_fn,
                    mag='g', maglim=20.,
                    blim=30)
        selfunc_fn = os.path.join(catdir, f'quaia/selfuncs/selection_function_NSIDE{d.NSIDE}_G20.0_blim15.fits')
    else:
        assert sample == 'catwise_agns'
        d = QSOSample(initial_catfn=os.path.join(catdir, 'catwise_agns/catwise_agns_master.fits'),
                    mask_fn=mask_fn,
                    mag='w1', maglim=16.4,
                    blim=30)
        selfunc_fn = os.path.join(catdir, f'catwise_agns/selfuncs/selection_function_NSIDE{d.NSIDE}_catwise_pluszodis.fits')

    d.cut_mag()  # cut all sources fainter than the input magnitude limit
    d.cut_galactic_plane_hpx()  # cut all sources with |b|<blim from the working source table
    selfunc = d.get_selfunc(selfunc=selfunc_fn) # load selection function

    # get masked datamap
    masked_datamap = d.construct_masked_datamap(selfunc=selfunc, return_map=True)

    # overdensity map, corrected by selection function
    overdensity_map = d.construct_overdensity_map(selfunc=selfunc_fn, min_completeness=0.)
    
    for j, max_ell in enumerate(max_ells):
        """
        FIT TO SPHERICAL HARMONICS TEMPLATES
        """
        # for each Lambda, do the fit, and compare the variance in the masked vs unmasked pixels
        #  also save the reconstructed maps to plot
        var_unmasked = np.empty(len(Lambdas))
        var_masked = np.empty_like(var_unmasked)
        reconstructed_maps_wo_dipole = np.empty((len(Lambdas), d.NPIX))
        for i, Lambda in enumerate(Lambdas):
            print(f"{i+1} of {len(Lambdas)}", flush=True)
            _, _, alms = compute_Cells_in_overdensity_map_Lambda(overdensity_map, Lambda=Lambda,
                                                                    max_ell=max_ell, selfunc=selfunc, return_alms=True)
            # save the reconstructed map _without_ the dipole
            reconstructed_maps_wo_dipole[i] = reconstruct_map(alms) - reconstruct_map(alms[:4])
            var_unmasked[i] = np.nanvar(reconstructed_maps_wo_dipole[i][np.where(d.mask)]) # since d.mask == 1 in the unmasked pixels
            var_masked[i] = np.nanvar(reconstructed_maps_wo_dipole[i][np.where(~d.mask)])

        """
        SAVE VARIANCE IN THE MASKED/UNMASKED PIXELS
        """
        save_dict = dict(max_ell=max_ell, Lambdas=Lambdas,
                        var_unmasked=var_unmasked, var_masked=var_masked,
                        overdensity_map=overdensity_map, reconstructed_maps_wo_dipole=reconstructed_maps_wo_dipole)
        np.save(save_fns[j], save_dict)
        print(f"saved results to {save_fns[j]}", flush=True)
        print(f"current time = {datetime.timedelta(seconds=time.time()-s)}", flush=True)

    total_time = time.time()-s 
    print(f"total time = {datetime.timedelta(seconds=total_time)}", flush=True)

if __name__=='__main__':
    main()