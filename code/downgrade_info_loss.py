"""
Calculate the Fisher information loss in the downgraded Healpix map.
"""

import os
import sys
import numpy as np
import healpy as hp
from scipy.linalg import eigh
import dipole
from abc_for_qso import get_catalog_info
import generate_mocks as gm

def main():

    nside_hi = 64
    catname = 'quaia_G20.5'
    catalog_info = get_catalog_info(catname)
    print(f"Fisher information loss for downgraded {catname} (nside={nside_hi}, galactic frame)\n"+"=" * 100)

    # get the completeness from the selection function
    c = gm.get_selfunc_map(catalog_info['selfunc_str'],
                                 nside=nside_hi, blim=30)
    # some stats
    print(f"{np.sum(c == 0)} of {len(c)} pixels with zero completeness")
    
    # get the dipole templates
    comps = np.column_stack((np.zeros(3), np.diag([1, 1, 1])))
    D = np.array([
        dipole.dipole_map(comp, NSIDE=64) for comp in comps
    ]).T    # shape (nside, 3)

    for dg in [1, 2, 4, 8, 16]:
        I1, I2 = fisher_info_pair(c, D, nside_dg=dg, nbar=catalog_info['base_rate'])
        # rotate from celestial to galactic
        R = hp.Rotator(coord=['C', 'G']).mat          # 3x3, ICRS (celestial) -> Galactic
        I1g = R @ I1 @ R.T
        I2g = R @ I2 @ R.T
        s1 = np.sqrt(np.diag(np.linalg.inv(I1g)))
        s2 = np.sqrt(np.diag(np.linalg.inv(I2g)))
        if dg == 1:
            print(f"info at full resolution: ", I1g)
        print(f"info at nside = {dg}: ", I2g)
        print(f"loss = {I1g - I2g}")
        print("info retained:", np.diag(I2g) / np.diag(I1g))
        print("error inflation:", s2 / s1)

        # trace loss: fraction of the total information lost
        trace_loss = np.trace(I1g - I2g) / np.trace(I1g)

        # maximum error inflation over all directions:
        # sqrt of the largest generalized eigenvalue of I1 x = lambda I2 x
        lam = eigh(I1g, I2g, eigvals_only=True)
        max_inflation = np.sqrt(lam.max())

        print(f"trace loss = {trace_loss * 100:.2f}%")
        print(f"max error inflation (any direction) = {max_inflation:.4f}\n")


def fisher_info_pair(c, D, nside_dg=2, nbar=1.):
    """
    Returns the Fisher information for two healpixel resolutions using the dipole model.
    Makes the approximation d -> 0.

    Parameters
    ----------
    c : ndarray
        Completeness (0 where masked), shape (npix,), NESTED ordering
    D : ndarray
        Templates D_i(p), shape (npix,3), NESTED ordering
    nside_hi : int, optional
    nside_dg : int, optional
    nbar : float, optional
        Mean number density.

    Returns
    -------
    I1 : Fisher information at `nside_hi`
    I2 : Fisher information at `nside_dg`

    """
    # define a downgrade function
    ud = lambda m: hp.ud_grade(m, nside_dg, power=-2)   # power=-2 preserves the sum of the counts

    # completeness of each superpixel
    c = np.asarray(c, dtype=np.float64)   # avoid float32 round-off
    C = ud(c)
    print(f"{(C > 0).sum()} of {len(C)} unmasked super-pixels at nside={nside_dg}")

    # a quick check
    total_hi, total_dg = c.sum(), C.sum()
    assert np.isclose(total_dg, total_hi, rtol=1e-8), \
        f"downgrade changes the sum: {total_hi} vs {total_dg}"

    # completeness-weighted template sums for each superpixel
    Ms = [ud(c * D[:,i]) for i in range(3)]
    M = np.stack(Ms, 1)

    # full resolution
    I1 = nbar * np.einsum('p,pi,pj->ij', c, D, D)

    # downgraded
    mask = (C > 0)  # only include pixels with nonzero completeness in the sum
    I2 = nbar * np.einsum('pi,pj,p->ij', M[mask], M[mask], 1 / C[mask])

    return I1, I2

if __name__=='__main__':
    main()