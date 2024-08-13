import numpy as np
import os
import sys

import dipole
from qso_sample import QSOSample
import tools


def main():

    # instantiate QSO sample
    catdir = '/scratch/aew492/quasars/catalogs'
    nside = 64
    d = QSOSample(initial_catfn=os.path.join(catdir, 'quaia/quaia_G20.5.fits'),
                    mask_fn=os.path.join(catdir, 'masks/mask_master_hpx_r1.0.fits'),
                    mag='G', maglim=20.,
                    blim=30,
                    NSIDE=nside)
    selfunc_fn = os.path.join(catdir, f'quaia/selfuncs/selection_function_NSIDE{nside}_G20.0_blim15.fits')

    # test new galactic plane cut
    d.cut_galactic_plane_hpx()

if __name__ == '__main__':
    main()