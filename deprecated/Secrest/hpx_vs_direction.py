"""
Edits from Abby:
- script moved from export/ to access functions
- created main() to avoid running upon import
"""
#!/usr/bin/env python
import numpy as np
from scipy.stats import pearsonr, sem
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import os


def main():

    catdir = '/scratch/aew492/quasars/catalogs/catwise2020'

    # t = Table.read(os.path.join(catdir, 'catwise_agns_masked_final_alpha_hpx.fits')) # !!
    t = Table.read(os.path.join(catdir, 'hpx_final_masked_bcut.fits'))

    print(f"{len(t)} pixels in table") # !!
    print("splitting t on masked and unmasked")
    # Split t on masked and unmasked
    msk = t['density'] < 0
    masked = t[msk]
    t = t[~msk]

    binsize = 1

    # Make linear regression to "correct" density and see if there is an
    # additional component due to the Galactic plane.
    p = np.polyfit(np.abs(t['elat']), t['density'], deg=1)
    print("Equation of raw fit: y = %.3f * x + %.1f" % (p[0], p[1]))

    print("Bin size: %.1f deg" % binsize)
    x, estat = getstat('elat', 'density', absx=True)
    p, pcov, fx, z, chi2, dof = linreg(np.abs(x), estat[:,0], 1 / estat[:,1])

    #jointfit(x, estat[:,0], 1 / estat[:,1])

    prand = np.random.multivariate_normal(p, pcov, 1000)
    # !!
    """
    for pr in prand:
        plt.plot(x, np.polyval(pr, x), c='k', alpha=0.01, zorder=0)


    pltstat(x, estat)
    #plt.plot(x, np.polyval(p,x))
    plt.xlabel('absolute ecliptic latitude')
    plt.ylabel('deg$^{-2}$')
    plt.title("y = %.3f * x + %.1f" % (p[0], p[1]))
    plt.show()
    """

    # Save p and its covariance
    np.save(os.path.join(catdir, 'p.npy'), p)
    np.save(os.path.join(catdir, 'pcov.npy'), pcov)

    # Sebastian's values:
    p[0] = -0.05126576725374681
    p[1] = 68.89130135046557

    t['denscorr'] = t['density'] - np.polyval(p, np.abs(t['elat'])) + p[1]


    # Now look at Galactic latitude

    #t = t[(t['l'] > 130) & (t['l']<150)]
    x, bstat = getstat('b', 'denscorr', absx=False)

    # Make smoothed average for map
    print("Calculating smoothed corrected density...")

    theta = omega_to_theta(1)
    #theta = 1 * u.rad

    # Alternatively, smooth on the scales of a multipole component
    #l = 5
    #theta = omega_to_theta(2 * np.pi / l)
    lent = len(t)
    sc = SkyCoord(t['ra'], t['dec'], frame='icrs')
    t['smoothed'] = -1 * np.ones(lent)
    t['sterr'] = -1 * np.ones(lent)
    t['Nsmooth'] = -1 * np.ones(lent)
    t['alphasmoothed'] = np.nan * np.ones(lent) # New
    t['alphasterr'] = np.nan * np.ones(lent)
    t['smoothed_uncorrected'] = -1 * np.ones(lent)
    t['sterr_uncorrected'] = -1 * np.ones(lent)
    for i in range(lent):
        d2d = sc[i].separation(sc)
        msk = d2d < theta
        sample = t[msk]['denscorr']
        t['smoothed'][i] = sample.mean()
        t['sterr'][i] = sem(sample)
        t['Nsmooth'][i] = sample.size
        sample = t[msk]['alpha']
        t['alphasmoothed'][i] = sample.mean()
        t['alphasterr'][i] = sem(sample)
        sample = t[msk]['density']
        t['smoothed_uncorrected'][i] = sample.mean()
        t['sterr_uncorrected'][i] = sem(sample)
        print("\t%.1f%%" % ((i + 1) / lent * 100), end='\r')

    # !! `hpx_final_masked_bcut.fits` has no masked values so the masked table is empty
    """
    masked['density'] = np.nan
    masked['denscorr'] = np.nan
    t = vstack((t, masked))
    """
    # !!
    t.sort('hpidx')

    t.write(os.path.join(catdir, 'steradian_smoothed.fits'), overwrite=True)
    print("wrote steradian_smoothed")
    # !!
    """
    pltstat(x, bstat)
    #plt.plot(x, y)
    plt.xlabel('Galactic latitude (ecliptic bias-corrected)')
    plt.ylabel('HEALPix density')
    plt.show()
    """


def getstat(xkey, ykey, absx=True, xtyp='lat'):
    if absx == True and xtyp=='lat':
        xs = np.abs(t[xkey].data)
        bins = np.arange(0, 91, binsize)


    elif absx == True and xtyp=='lon':
        xs = t[xkey].data % 180
        bins = np.arange(0, 181, binsize)

    elif absx == False and xtyp=='lat':
        xs = t[xkey].data
        bins = np.arange(-90, 91, binsize)
    elif absx == False and xtyp=='lon':
        xs = t[xkey].data
        bins = np.arange(0, 361, binsize)
    else:
        raise NameError("unrecognized absx or xtyp.")

    binx = bins[0:-1] + binsize / 2
    idx = np.digitize(xs, bins, right=False)

    stat = np.empty((binx.size, 3), dtype=float)
    for i in range(1, binx.size + 1):
        density = t[idx==i][ykey].data
        if density.size < 10:
            stat[i-1, 0] = np.nan
            stat[i-1, 1] = np.nan
        else:
            stat[i-1, 0] = density.mean()
            stat[i-1, 1] = sem(density)

        stat[i-1, 2] = density.size

    msk = np.isfinite(stat[:,1])

    return binx[msk], stat[msk]


def residual(fx, x, y, w, prints=False):
    r = y - fx
    z = r * w   # z-score
    chi2 = np.sum(z**2)
    dof = x.size - 2
    if prints:
        print("z-score stdev: %.2f" % z.std())
        print("chi2/dof: %.2f/%i = %.2f" % (chi2, dof, chi2/dof))

    return r, z, chi2, dof


def linreg(x, y, w, prints=False):
    if prints:
        print("Pearson r: %.2f" % pearsonr(x, y)[0])
    p, pcov = np.polyfit(x, y, deg=1, w=w, cov=True)
    perr = np.sqrt(np.diag(pcov))
    print("Equation of fit: y = %.3f(%.3f) * x + %.1f(%.1f)" % (p[0], perr[0],
                                                                p[1], perr[1]))
    fx = np.polyval(p, x)
    r, z, chi2, dof = residual(fx, x, y, w)

    return p, pcov, fx, z, chi2, dof


def pltstat(x, stat):
    plt.errorbar(x, stat[:,0], xerr=binsize/2, yerr=stat[:,1],
                 linestyle='')

def omega_to_theta(omega):
    """Convert solid angle omega in steradians to theta in radians for
    a cone section of a sphere."""
    return np.arccos(1 - omega / (2 * np.pi)) * u.rad


if __name__=='__main__':
    main()