#!/usr/bin/env python
import numpy as np
from astropy.table import Table, Column

# Speed of light in Angstrom/s
c = 299792458 * 1e10

def get_passband(rspfile):
    t = Table.read(rspfile, format='ascii')
    t['nu'] = c / t['Angstrom']

    # Frequency must be monotonically increasing and unique
    t.sort('nu')
    idx = np.unique(t['nu'].data, return_index=True)[1]
    t = t[idx]

    nu, Snu = t['nu'], t['Fraction'].data

    return nu, Snu


def intg(Fx_matrix, x):
    """Function to integrate a matrix of functions with respect to x.
    Performs trapezoidal integration. Matrix have shape (m, n) where
    m is the length of x and n is the number of functions. Returns an
    array with shape (n,). If the matrix is a single array
    (single function), returns the same output as np.trapz.
    """

    Fx_matrix = np.array(Fx_matrix, dtype=float)
    x = np.array(x, dtype=float)
    dx = np.diff(x)
    dFx_matrix = (Fx_matrix[:][1:] + Fx_matrix[:][:-1]) / 2

    return np.matmul(dx, dFx_matrix)


def synthmagAB(nu, fnu, Snu, zpAB=48.60):
    """Returns a synthetic AB magnitude as per
    Fukugita et al. (1996, AJ, 111, 1748), Eq. 7.
    """

    ln_nu = np.log(nu)
    m = (
        -2.5 * np.log10(
            intg(fnu * Snu, ln_nu) / intg(Snu, ln_nu)
            )
        - zpAB
        )

    return m


def nu_iso_pow(nu, alpha, Snu, alpha0=1e-5):
    """nu_iso_pow(nu, alpha, Snu, alpha0=1e-5)

    Returns isophotal frequency, which is defined as the frequency at
    which the flux density function fnu is equal to its mean value
    over a passband given by Snu, of a power-law flux density spectrum.

    See Bessell & Murphy (2012, PASP, 124, 140), Equation A19.

    Parameters
    ----------
    nu : numpy.array (float)
        frequency
    alpha : float
        spectral index such that fnu = nu**alpha
    Snu : numpy.array (float)
        Passband (photonic)
    alpha0 : float
        Number to replace alpha=0 with (there is not isophotal frequency
        for a constant spectrum; however the isophotal frequency just to
        the negative and positive of zero converge to the same value.

    Returns
    -------
    nu_iso : float
        Isophotal frequency
    """

    if alpha==0.0:
        ln_fnu = alpha0 * np.log(nu)
    else:
        ln_fnu = alpha * np.log(nu)

    ln_fnu -= ln_fnu.max()  # Get rid of small/huge floats
    ln_nu = np.log(nu)
    p = np.polyfit(ln_fnu, ln_nu, deg=1)

    fnu = np.exp(ln_fnu)
    fnu_iso = 10**( -synthmagAB(nu, fnu, Snu, zpAB=0.0) / 2.5 )
    ln_fnu_iso = np.log( fnu_iso )

    ln_nu_iso = np.polyval(p, ln_fnu_iso)
    nu_iso = np.exp( ln_nu_iso )

    return nu_iso



def fnu_norm(alpha, nu, Snu, zpAB=48.60):
    """fnu_norm(alpha, nu, Snu, zpAB=48.60)

    Returns normalization k such that

    fnu = k * 10**(-mAB / 2.5) * nu**alpha

    where mAB is the apparent magnitude in the AB system defined by
    zeropoint zpAB. This provides the flux density normalization of a
    power-law source with spectral index alpha and observed magnitue mAB.

    Parameters
    ----------
    alpha : numpy.array(dtype=float)
        spectral index such that fnu = k * nu**alpha
    Snu : numpy.array(dtype=float)
        Passband (photonic)
    zpAB : float
        AB zeropoint as in Fukugita+96, Eq. 7. Default is 48.60

    Returns
    -------
    k : float
        k such that fnu = k * 10**(-mAB / 2.5) * nu**alpha
    """

    ln_nu = np.log(nu)
    k = intg(Snu, ln_nu) / intg(nu**alpha * Snu, ln_nu) * 10**(-zpAB/2.5)

    return k


def main():
    # Bandpasses
    nu_G,  Snu_G  = get_passband('G_gaiaDR2_published.txt')
    nu_BP, Snu_BP = get_passband('BP_gaiaDR2_published.txt')
    nu_RP, Snu_RP = get_passband('RP_gaiaDR2_published.txt')
    nu_W1, Snu_W1 = get_passband('RSR-W1.txt')
    nu_W2, Snu_W2 = get_passband('RSR-W2.txt')

    # Take the natural log of the passband nu to pull it out of the loop.
    ln_nu_G  = np.log( nu_G  )
    ln_nu_BP = np.log( nu_BP )
    ln_nu_RP = np.log( nu_RP )
    ln_nu_W1 = np.log( nu_W1 )
    ln_nu_W2 = np.log( nu_W2 )

    # Spectral indices
    alphas = np.around(np.arange(-30, 30.001, step=0.001), 3)
    Nalpha = alphas.size

    # Vectors to store flux factors
    k_G  = np.nan * np.empty( Nalpha, dtype=float )
    k_W1 = np.nan * np.empty( Nalpha, dtype=float )

    # Isophotal frequencies
    nu_G_iso  = np.nan * np.empty( Nalpha, dtype=float )
    nu_W1_iso = np.nan * np.empty( Nalpha, dtype=float )

    # Colors
    BP_G  = np.nan * np.empty( Nalpha, dtype=float )
    BP_RP = np.nan * np.empty( Nalpha, dtype=float )
    G_RP  = np.nan * np.empty( Nalpha, dtype=float )
    G_W1  = np.nan * np.empty( Nalpha, dtype=float )
    W1_W2 = np.nan * np.empty( Nalpha, dtype=float )

    for i in range(Nalpha):
        # Get power law normalizations first
        k_G[i]  = fnu_norm( alphas[i], nu_G,  Snu_G )
        k_W1[i] = fnu_norm( alphas[i], nu_W1, Snu_W1 )

        # Small/large float problems: normalize in log space.
        # np.log is the natural log, not that it matters.
        ln_fnu_G  = alphas[i] * ln_nu_G
        ln_fnu_BP = alphas[i] * ln_nu_BP
        ln_fnu_RP = alphas[i] * ln_nu_RP
        ln_fnu_W1 = alphas[i] * ln_nu_W1
        ln_fnu_W2 = alphas[i] * ln_nu_W2

        ln_norm = ln_fnu_G.max()

        ln_fnu_G  -= ln_norm
        ln_fnu_BP -= ln_norm
        ln_fnu_RP -= ln_norm
        ln_fnu_W1 -= ln_norm
        ln_fnu_W2 -= ln_norm

        fnu_G  = np.exp( ln_fnu_G  )
        fnu_BP = np.exp( ln_fnu_BP )
        fnu_RP = np.exp( ln_fnu_RP )
        fnu_W1 = np.exp( ln_fnu_W1 )
        fnu_W2 = np.exp( ln_fnu_W2 )

        # Get effective frequencies for fnu
        nu_G_iso[i]  = nu_iso_pow(nu_G,  alphas[i], Snu_G)
        nu_W1_iso[i] = nu_iso_pow(nu_W1, alphas[i], Snu_W1)

        # The catalogs given magnitudes in their native Vega system, so
        # convert. synthmagAB gives Oke & Gunn / Fukugita AB mags (48.60),
        # so use the modified AB mag offsets for WISE.
        G  = synthmagAB( nu_G,  fnu_G,  Snu_G  ) - 0.105
        BP = synthmagAB( nu_BP, fnu_BP, Snu_BP ) - 0.029
        RP = synthmagAB( nu_RP, fnu_RP, Snu_RP ) - 0.354
        W1 = synthmagAB( nu_W1, fnu_W1, Snu_W1 ) - 2.673
        W2 = synthmagAB( nu_W2, fnu_W2, Snu_W2 ) - 3.313

        # These are now in the Vega system.
        BP_G[i]  = BP - G
        BP_RP[i] = BP - RP
        G_RP[i]  = G - RP
        G_W1[i]  = G - W1
        W1_W2[i] = W1 - W2

    tbl = Table()
    tbl['alpha']     = alphas
    tbl['k_G']       = k_G
    tbl['k_W1']      = k_W1
    tbl['nu_G_iso']  = nu_G_iso
    tbl['nu_W1_iso'] = nu_W1_iso
    tbl['BP_G']      = BP_G
    tbl['BP_RP']     = BP_RP
    tbl['G_RP']      = G_RP
    tbl['G_W1']      = G_W1
    tbl['W1_W2']     = W1_W2
    tbl.write('alpha_colors_extension.fits', overwrite=True)

    return None


if __name__=="__main__":
    main()
