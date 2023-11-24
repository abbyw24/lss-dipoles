"""
Measure the dipole on a fine grid of construction parameter choices, using the S21 method.
"""

import numpy as np
from astropy.table import Table, Column, hstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
from healpy.visufunc import projplot
from scipy.stats import sem
from datetime import datetime
import os
import sys

from Secrest.hpx_vs_direction import linreg, omega_to_theta
from Secrest.get_colors import synthmagAB, get_passband
from dipole import fit_dipole, get_dipole
import tools
from Secrest_dipole import SecrestDipole


def main():

    # galactic plane cuts
    bgrid = np.arange(15,70.1,5)

    # # magnitude cuts
    # w1grid = np.arange(15., 16.81, .05)

    catalog = 'quaia'

    # instantiate dipole object
    if catalog == 'catwise_agns':
        d = SecrestDipole(initial_catfn='catwise_agns_master.fits',
                            catname='catwise_agns',
                            mask_fn='/scratch/aew492/quasars/catalogs/masks/mask_master_hpx_r1.0.fits',
                            mag='w1',
                            maglim=16.4,
                            blim=30,
                            log=False,
                            load_init=False)
    elif catalog == 'quaia':
        d = SecrestDipole(initial_catfn='quaia_G20.5.fits',
                                        catname='quaia',
                                        mask_fn='/scratch/aew492/quasars/catalogs/masks/mask_master_hpx_r1.0.fits',
                                        mag='G',
                                        maglim=20.5,
                                        blim=30,
                                        compcorrect=True,
                                        log=False,
                                        load_init=False)
    else:
        assert False, "unknown catalog"
    
    # load initial catalog: this is the same for all plane cuts
    d.load_initial_cattab()
    # make cut: again, same for all plane cuts
    d.cut_mag()
    # d.cut_galactic_plane()

    initial_table = d.table

    res = []

    for i, blim in enumerate(bgrid):
        # make plane cut
        t = initial_table[np.abs(initial_table['b']) > blim]
        print(f"cut |b| <= {blim} -> {len(t)} sources left.", flush=True)

        # turn into HealPIX map
        t = d._make_healpix_map(t)

        # mask initial map
        t = d._mask_initial_healpix_map(map_=t)

        # make extra galactic plane cut
        t = t[np.abs(t['b']) > (blim + 1)]

        # completeness correct for quaia
        if d.compcorrect is True:
            t = d._completeness_correct(tab=t, key='density',
                                selfunc_fn=os.path.join(d.catdir,
                                            f'selection_function_NSIDE{d.NSIDE}_{d.mag}20.5.fits'))

        # density correct
        t = d._hpx_vs_direction(tab=t)

        # measure the dipole
        dipole_amp, dipole_dir = d.compute_dipole(t, Cinv=None, out_frame='galactic', logoutput=False)
        res.append([blim, dipole_amp, dipole_dir])
        del t
        print("")
    
    res = np.array(res)
    res[:,0] = res[:,0].astype(float) # cut associated with each result
    res[:,1] = res[:,1].astype(float) # recovered dipole amplitudes
    
    np.save(f'/scratch/aew492/quasars/catalogs/{catalog}/dipoles_blimgrid_G20.5.npy', res)
    print("done")


if __name__=='__main__':
    main()