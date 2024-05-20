"""
Functions to calculate the expected number-count dipole in a quasar catalog,
from the Ellis & Baldwin (1984) formula.
"""
import numpy as np
import astropy.units as u
import astropy.constants as const


"""
FUNCTIONS APPLICABLE TO BOTH GAIA AND WISE
"""
def EllisBaldwin(x, alpha, v=369.825*u.km/u.s):
    """
    Return the expected dipole amplitude from Ellis & Baldwin (1984), given
        x = number-count slope at flux density limit
        alpha = source spectral index assuming power law spectra
    """
    return v / const.c.to(u.km/u.s) * (2 + x * (1+alpha))

def compute_number_counts(mags, min_mag, max_mag, width=0.1):
    """
    Compute the source counts on a grid of magnitudes.

    Parameters
    ----------
    mags : array-like
        Magnitudes of the sources.
    min_mag : float
        Minimum magnitude at which to compute the source counts.
    max_mag : float
        Maximum magnitude at which to compute the source counts.

    Returns
    -------
    mag_grid : 1D array
        Grid of magnitude limits.
    counts : 1D array
        Number of sources with magnitude brighter than each magnitude
        limit in the grid.
    """
    # number of magnitude limits
    nmags = int((max_mag-min_mag) / width)+1
    # grid of magnitude limits
    mag_grid = np.linspace(min_mag, max_mag, nmags)
    # number of sources with mag brighter than each magnitude limit
    counts = np.array([
        np.sum(mags<=mag) for mag in mag_grid
    ])
    return mag_grid, counts

def compute_x(mag_grid, counts, mag_limit):
    """
    Compute the number-count slope at the magnitude limit of the sample.
    """
    # index of the magnitude limit of the sample
    mag_idx = np.where(np.isclose(mag_grid, mag_limit))[0][0]
    # slope of interest: from the neighboring magnitudes in the grid
    dlogN = np.log10(counts[mag_idx-1])-np.log10(counts[mag_idx+1])
    dmags = mag_grid[mag_idx-1]-mag_grid[mag_idx+1]
    x = 2.5 * dlogN / dmags
    return x


"""
GAIA
"""
def BR_Vega_to_AB(BR, offset=-0.3250):
    """
    Convert Gaia (BP-RP) magnitudes in the Vega system to the AB system.
    Offset computed from Gaia photometric zero points
    (Tables 5.2-5.3 in \
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html)

    """  
    return BR + offset

def compute_alpha_gaia(BR, lambda_B=505.15, lambda_R=772.62):
    """
    Compute the spectral slope alpha for a source with given (BP-RP) magnitude.
    Assumes the source SED follows a power law, S(nu) ~ nu^{-alpha}

    Parameters
    ----------
    BR : float
        BP - RP magnitude of the source (Vega system)
    lambda_B : float, optional
        Wavelength of blue passband (default is the "pivot wavelength" given by Gaia)
    lambda_R : float, optional
        Wavelength of red passband (")
    Note that the unit of wavelength doesn't matter here as long as it's the same in both bands.

    Returns
    -------
    alpha : float
        Estimated spectral slope of the source.
    """
    # convert given (B-R)_Vega to AB color (B-R)_AB
    BR_AB = BR_Vega_to_AB(BR)
    
    alpha = BR_AB / (2.5 * np.log10(lambda_R / lambda_B))

    return alpha

def compute_expected_dipole_gaia(table, maglimit=20.0, min_g=19.5, max_g=20.5,
                                    verbose=True, return_full_results=False):
    """
    Compute the expected dipole from the Ellis-Baldwin formula given an input source table.
    """
    if verbose:
        print("computing expected dipole from Ellis-Baldwin:")
    
    # X: number-count slope at magnitude limit
    mags = table['phot_g_mean_mag']
    mag_grid, counts = compute_number_counts(mags, min_g, max_g)
    x = compute_x(mag_grid, counts, maglimit)
    if verbose:
        print(f"\tnumber-count slope x = {x:.3f}")

    # ALPHA: effective spectral index
    # only compute alpha from sources within the magnitude limit
    idx_mag = (mags <= maglimit)
    # (BP-RP) magnitudes (Vega system)
    bprp = table[idx_mag]['phot_bp_mean_mag'] - table[idx_mag]['phot_rp_mean_mag']
    # compute alpha for each source in the table
    alphas = [
        compute_alpha_gaia(bprp[i]) for i in range(len(bprp))
    ]
    alpha = np.mean(alphas)
    if verbose:
        print(f"\teffective alpha = {alpha:.3f}")
    
    # put it all together
    expected_dipole_amplitude = EllisBaldwin(x, alpha)
    if verbose:
        print(f"\texpected dipole amplitude = {expected_dipole_amplitude:.4f}")
    
    if return_full_results:
        res = dict(alpha=alpha, alphas=alphas,
                    x=x, mag_grid=mag_grid, counts=counts,
                    expected_dipamp=expected_dipole_amplitude)
        return res
    else:
        return expected_dipole_amplitude


"""
WISE
"""
def W12_Vega_to_AB(W12, offset=-0.6400):
    return W12 + offset

def compute_alpha_wise(W12, lambda_W1=3.368, lambda_W2=4.618):
    """
    Compute the spectral slope alpha for a source with given (W1-W2) magnitudes.

    Parameters
    ----------
    W12 : float
        W1 - W2 magnitude of the source (Vega system)
    lambda_W1 : float, optional
        Wavelength of the W1 passband
        (default is the "effective wavelength" of the band for a
        signal with nu * F_nu = const. from https://www.astro.ucla.edu/~wright/WISE/passbands.html)
    lambda_W2 : float, optional
        Wavelength of the W2 passband (")
    Note that the unit of wavelength doesn't matter here as long as it's the same in both bands.

    Returns
    -------
    alpha : float
        Estimated spectral slope of the source.
    """
    # convert given (W1-W2)_Vega to AB color (W1-W2)_AB
    W12_AB = W12_Vega_to_AB(W12)
    
    alpha = W12_AB / (2.5 * np.log10(lambda_W2 / lambda_W1))

    return alpha

def compute_expected_dipole_wise(table, maglimit=16.4, min_w1=16., max_w1=16.5,
                                    verbose=True, return_full_results=False):
    """
    Compute the expected dipole from the Ellis-Baldwin formula given an input source table.
    """

    # X: number-count slope at magnitude limit
    mags = table['w1']
    mag_grid, counts = compute_number_counts(mags, min_w1, max_w1)
    x = compute_x(mag_grid, counts, maglimit)
    if verbose:
        print(f"number-count slope x = {x:.3f}")

    # ALPHA: effective spectral index
    # only compute alpha from sources within the magnitude limit
    idx_mag = (mags <= maglimit)
    # (W1-W2) magnitudes (Vega system)
    w12 = table[idx_mag]['w12']
    # compute alpha for each source in the table
    alphas = [
        compute_alpha_wise(w12[i]) for i in range(len(w12))
    ]
    alpha = np.mean(alphas)
    if verbose:
        print(f"effective alpha = {alpha:.3f}")
    
    # put it all together
    expected_dipole_amplitude = EllisBaldwin(x, alpha)
    if verbose:
        print(f"expected dipole amplitude = {expected_dipole_amplitude:.4f}")
    
    if return_full_results:
        res = dict(alpha=alpha, alphas=alphas,
                    x=x, mag_grid=mag_grid, counts=counts,
                    expected_dipamp=expected_dipole_amplitude)
        return res
    else:
        return expected_dipole_amplitude