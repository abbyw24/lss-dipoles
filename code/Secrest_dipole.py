"""

MEASURE THE DIPOLE IN THE CATWISE2020 QUASAR CATALOG
using the steps provided in Secrest/export/README .

with the possibility to turn knobs.

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

"""
STEP 1. Download catalogs from IPAC.

STEP 2. Run `correct_catwise.py` to correct for extinction, proper motions, positions, and make magnitude cuts.

STEP 3.
Step 3a. Mask in TOPCAT with `MASKS_exclude_master_final.fits`:
"
    -   Add "dummy" primary and secondary radii, and position angles to the output from step 2, \
        with values of 0.0 degrees.
    -   The mask will have a radius and position angle already, but add a \
        secondary radius as radius * ba. This is in degrees.
    -   Using TOPCAT's Sky Ellipses function with a Scale of 1.0 degrees, do an "All Matches", \
        Join Type 1 not 2, where 1 is the output from step 2 and 2 is the mask file.
"
    -   Cut any sources with w1cov < 80 (417) (these have high w12 color).
    --> `catwise_agns_master.fits`

Step 3b. Make W1 and galactic plane cuts.
        Secrest makes W1 <= 16.4 and |b| < 30deg. --> `catwise_agns_masked_final.fits`

STEP 4. Make HEALPix map with `mk_hpx.py`.

STEP 5. Ecliptic latitude correction with `hpx_vs_direction.py`.

STEP 6. Get spectral indices with `lookup_alpha_catwise.py`.

"""

def main():

    # Rv = 3.1
    # delRW = 0.
    # Rv_tag = f'_Rv{Rv:.1f}' if Rv != 3.1 else ''
    # delRW_tag = f'_RW{delRW:.3f}' if delRW != 0. else ''
    # initial_catfn = f'catwise_agns_master{Rv_tag}{delRW_tag}.fits'

    mask_factor = 1
    mask_fn = f'/scratch/aew492/lss-dipoles_results/catalogs/masks/mask_master_hpx_r{mask_factor:.1f}.fits' if mask_factor > 0. else None
    save_tag = f'_r{mask_factor:.1f}'

    catname = 'quaia'

    # blims = np.arange(30, 61, 10)
    # maglims = np.arange(15.8, 16.81, 0.2)


    if catname == 'catwise_agns':
        dipoleset = SecrestDipole(initial_catfn=initial_catfn,
                                    catname='catwise_agns',
                                    mask_fn=mask_fn,
                                    mag='w1',
                                    maglim=16.8,
                                    blim=20,
                                    Rv=Rv,
                                    delRW=delRW,
                                    save_tag=save_tag)
    elif catname == 'quaia':
        dipoleset = SecrestDipole(initial_catfn='quaia_G20.0.fits',
                                    catname='quaia',
                                    mask_fn=mask_fn,
                                    mag='G',
                                    maglim=20.,
                                    blim=10,
                                    compcorrect=True,
                                    save_tag=save_tag)
    else:
        assert False, "unknown catname"

    # Steps 1-3a already done; start with `initial_catfn`.

    # log
    dipoleset.make_new_log()

    # Step 3b. Make W1 and galactic plane cuts.
    dipoleset.cut_mag()
    dipoleset.cut_galactic_plane()

    # Step 4. Make HEALPix map and perform additional masking.
    # 4a. Make HEALPix map.
    dipoleset.make_healpix_map(save=True, overwrite=True)

    dipoleset.mask_initial_healpix_map()

    # 4b. Prepare additional masking catalog.
    # -> already done; we can use the same modified masking file for all catalogs
    # dipoleset.prepare_additional_masking_catalog(save=True, overwrite=True)

    # # 4c. (in TOPCAT) Perform exclusionary sky ellipses match.
    # dipoleset.log("(TOPCAT: performed exclusionary sky ellipses match)")

    # # 4d. Load in masked HEALPix map and make an extra galactic plane cut.
    # dipoleset.load_masked_healpix_map()

    dipoleset.make_extra_galactic_plane_cut(save=True, overwrite=True)

    # Step 5. Determine the bias correction for ecliptic latitude and make corrected density.
    dipoleset.make_elat_correction(tab_fn=None, sebsvals=False, save=True, overwrite=True)

    # compute dipole?
    dipole_amp, dipole_dir = dipoleset.compute_dipole(dipoleset.hpxelatcorr,
                                                        Cinv=None, logoutput=True)
    
    del dipoleset

    # Step 6. Get spectral indices from the alpha lookup table.
    # dipoleset.get_alphas(tab_fn=None, save=True, overwrite=True)



class SecrestDipole():

    """
    STEPS 1, 2 & 3a are complete --> master catalog `catwise_agns_master.fits` as a starting point.
    """

    def __init__(self,
                    catname,
                    initial_catfn,
                    mag, maglim,
                    mask_fn='/home/aew492/lss-dipoles/data/catalogs/masks/mask_master_hpx_r1.0.fits',
                    blim=30,
                    Rv=3.1,
                    delRW=0,
                    badd=1,
                    compcorrect=False,
                    NSIDE=64,
                    load_init=True,
                    log=True,
                    newlog=True,
                    basedir='/home/aew492/lss-dipoles/data/catalogs',
                    save_tag=''):

        # asserts
        assert initial_catfn.endswith('.fits'), "initial catalog must be a fits file"

        # input parameters
        self.catname = catname
        self.initial_catfn = initial_catfn
        self.mag = mag
        self.maglim = maglim
        self.mask_fn = mask_fn
        self.basedir = basedir
        self.catdir = os.path.join(self.basedir, self.catname)
        if not os.path.exists(self.catdir):
            os.makedirs(self.catdir)
        self.blim = blim
        self.Rv = Rv
        self.delRW = delRW
        self.badd = badd
        # self.r1 = r1.to(u.deg) if isinstance(r1, u.Quantity) else r1 * u.deg
        self.compcorrect = compcorrect if catname == 'quaia' else False
        self.NSIDE = NSIDE
        self.NPIX = hp.nside2npix(NSIDE)

        # tags
        self.Rv_tag = f'_Rv{Rv:.1f}' if Rv != 3.1 else ''
        self.delRW_tag = f'_RW{delRW:.3f}' if delRW != 0. else ''
        self.comp_tag = '_compcorr' if self.compcorrect else ''
        self.save_tag = save_tag

        # save directory
        self.savedir = os.path.join(self.catdir,
                        f'choices/{catname}{self.Rv_tag}{self.delRW_tag}_{mag}{maglim:.1f}_blim{blim:.0f}{save_tag}')
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        if self.compcorrect:
            self.savedir_ = os.path.join(self.savedir, 'comp-corrected')
            if not os.path.exists(self.savedir_):
                os.makedirs(self.savedir_)
        else:
            self.savedir_ = self.savedir

        # optionally, create a file to log results
        if log:
            self.log_fn = os.path.join(self.savedir, 'log.txt')

        # load initial catalog as an astropy table
        if load_init:
            self.load_initial_cattab()


    def make_new_log(self):
        with open(self.log_fn, 'w') as f:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"log created at {now}\n")

    def log(self, line):
        try:
            with open(self.log_fn, 'a') as f:
                f.write(line+'\n')
        except AttributeError:
            pass
        finally:
            print(line, flush=True)
    

    def load_initial_cattab(self):
        self.log(f"loading initial catalog, {self.initial_catfn}")
        self.initial_cattab = Table.read(os.path.join(self.catdir, self.initial_catfn), format='fits')
        if not hasattr(self, 'cattab'):
            self._update_working(self.initial_cattab)
        self.log(f"{len(self.initial_cattab)} sources in initial catalog.")
    

    def _update_working(self, table):
        self.table = table
    

    def _step_handler(self, func, save=False, save_fn=None, overwrite=True, **kwargs):
        """
        Helper function to check if step has already been performed and deal with overwrite and saving.
        We want to do the step if:
        - save is False
        - save is True and save_fn doesn't exist
        - save is True and save_fn exists but overwrite is True
        """
        do_step = True
        saved = False
        if save is True:
            assert save_fn.endswith('.fits'), "must provide a fits save filename"
            if os.path.exists(save_fn) and overwrite is False:
                self.log(f"{save_fn} exists and overwrite is False! continuing.")
                do_step = False 
        if do_step is True:
            output = func(**kwargs)
            if save is True:
                output.write(save_fn, format='fits', overwrite=overwrite)
                saved = True
        else:
            assert do_step is False, "do_step is not False, something went wrong"
            output = []
        did_step = do_step
        return output, did_step, saved

    """
    STEP 3b. Magnitude and galactic plane cuts.
    """
    def cut_mag(self):
        if self.mag == 'G':
            key = 'phot_g_mean_mag'
        else:
            key = self.mag
        self._update_working(self.table[self.table[key] <= self.maglim])
        self.log(f"cut {self.mag} > {self.maglim} -> {len(self.table)} sources left.")
    

    def cut_galactic_plane(self):
        self._update_working(self.table[np.abs(self.table['b']) > self.blim])
        self.log(f"cut |b| <= {self.blim} -> {len(self.table)} sources left.")
    

    """
    STEP 4. Make HEALPix map and perform additional masking.
    """
    def _make_healpix_map(self, t):
        """
        Convert the initial catalog, with mag and |b| cuts, to a HEALPix map.
        Copied / adapted from Secrest's `mk_hpx.py`.
        """

        sc = SkyCoord(t['ra'], t['dec'], frame='icrs')
        theta = -(sc.dec - 90 * u.deg).radian
        phi = sc.ra.radian

        hpidx = hp.ang2pix(self.NSIDE, theta, phi)

        # TODO: change this to np.bincount()
        hpmap = np.zeros(self.NPIX, dtype=float)
        for i in range(len(t)):
            hpmap[hpidx[i]] += 1
        
        # hpxmap is total per pixel. Convert to per deg^2
        skyarea = 4 * np.pi * (180 / np.pi)**2
        hpmap *= self.NPIX / skyarea

        # Set neighbors of 0 pixel count to UNSEEN, as these are masked region
        # bordering pixels.
        idx0 = np.where(hpmap==0)[0]
        indices = np.empty((idx0.size, 8), dtype=int)
        for i in range(idx0.size):
            indices[i] = hp.pixelfunc.get_all_neighbours(self.NSIDE, idx0[i])

        indices = np.unique(indices.flatten())

        hpmap[idx0] = hp.pixelfunc.UNSEEN
        hpmap[indices] = hp.pixelfunc.UNSEEN

        # Get ra, dec of pixels
        hpidx = np.arange(self.NPIX)
        lon,lat = hp.pix2ang(self.NSIDE, hpidx, lonlat=True)
        sc = SkyCoord(lon * u.deg, lat * u.deg, frame='icrs')
        lb  = sc.galactic
        ec = sc.barycentricmeanecliptic

        hpx = Table()
        hpx['hpidx'] = hpidx
        hpx['ra'] = sc.ra
        hpx['dec'] = sc.dec
        hpx['l'] = lb.l
        hpx['b'] = lb.b
        hpx['elon'] = ec.lon
        hpx['elat'] = ec.lat
        hpx['density'] = hpmap

        # Make dummy radii for masking
        hpx['primrad'] = 0.0 * u.deg
        hpx['secrad'] = 0.0 * u.deg
        hpx['pa'] = 0.0 * u.deg

        return hpx

    def make_healpix_map(self, save=True, overwrite=True):
        save_fn = os.path.join(self.savedir, 'first_hpx_map.fits')
        self.first_hpx_map, did_step, saved = self._step_handler(self._make_healpix_map, t=self.table,
                                                    save=save, save_fn=save_fn, overwrite=overwrite)
        if saved:
            self.log(f"saved HEALPix map to {save_fn}")
        self._update_working(self.first_hpx_map)
    

    # # ! no longer needed, since we can use the same modified mask file for all catalogs
    # def _prepare_additional_masking_catalog(self):

    #     # load initial masks file
    #     masktab = Table.read(self.mask_fn)
    #     print(f"loaded {len(masktab)} masks from {self.mask_fn}.", flush=True)

    #     # increase radii "to avoid edge effects when excluding HEALPix pixels near the edge of masks":
    #     # add r1 to the primary radius (degrees)
    #     masktab['radius'] += self.r1.value
    #     print(f"added {self.r1.value} degrees to the primary radius", flush=True)
    #     # make a secondary radius as (radius + 1) * ba
    #     masktab['radius2'] = (masktab['radius'] + 1) * masktab['ba']
    #     print(f"made secondary radius as (radius + 1) * ba", flush=True)

    #     return masktab
    
    # def prepare_additional_masking_catalog(self, save=True, overwrite=True):
    #     save_fn = os.path.join(self.savedir, self.mask_fn.replace('.fits', '_mod.fits'))
    #     masktab, did_step, saved = self._step_handler(self._prepare_additional_masking_catalog,
    #                                                 save=save, save_fn=save_fn, overwrite=overwrite)
    #     if saved:
    #         print(f"saved modified mask file to {save_fn}", flush=True)
    

    def _mask_initial_healpix_map(self, map_=None):
        # initial healpix map
        if map_ is None:
            try:
                map_ = self.first_hpx_map
            except AttributeError:
                map_ = Table.read(os.path.join(self.savedir, 'first_hpx_map.fits'))
        if self.mask_fn is not None:
            with fits.open(self.mask_fn) as hdu:
                mask = hdu[0].data
            assert map_['density'].shape == mask.shape
            map_['density'][mask==0.] = hp.UNSEEN
        return map_
    
    def mask_initial_healpix_map(self, save=True, overwrite=True):
        save_fn = os.path.join(self.savedir, 'hpx_masked.fits')
        self.hpxmasked, did_step, saved = self._step_handler(self._mask_initial_healpix_map,
                                                        save=save, save_fn=save_fn, overwrite=overwrite)
        if saved:
            self.log(f"saved masked healpix map to {save_fn}: \
                        {sum(self.hpxmasked['density']!=hp.UNSEEN)} unmasked healpixels")
        self._update_working(self.hpxmasked)
    

    def load_masked_healpix_map(self):
        map_fn = os.path.join(self.savedir, 'hpx_masked.fits')
        try:
            self.hpxmasked = Table.read(map_fn, format='fits')
        except FileNotFoundError:
            raise FileNotFoundError("map not found; perform additional masking in TOPCAT")
        self.log(f"successfully loaded masked healpix map, {map_fn}: {len(self.hpxmasked)} healpixels")
        self._update_working(self.hpxmasked)
    

    def _make_extra_galactic_plane_cut(self):
        blim = self.blim + self.badd
        return self.table[np.abs(self.table['b']) > blim]

    def make_extra_galactic_plane_cut(self, save=True, overwrite=True):
        save_fn = os.path.join(self.savedir, 'hpx_masked_final.fits')
        self.hpxmaskedbcut, did_step, saved = self._step_handler(self._make_extra_galactic_plane_cut,
                                                                save=save, save_fn=save_fn, overwrite=overwrite)
        if did_step:
            self.log(f"made an additional galactic plane cut |b| > {self.blim + self.badd} degrees: {len(self.hpxmaskedbcut)} healpixels left")
        if saved:
            self.log(f"saved HEALPix map with extra masking to {save_fn}")
        
        if self.compcorrect:
            self.completeness_correct()
        self._update_working(self.hpxmaskedbcut)
    

    def _completeness_correct(self, tab, key, selfunc_fn=None):
        """ Correct the density in each healpixel by the completeness in that pixel. """
        assert self.catname == 'quaia', "catalog must be Quaia for completeness correction"
        # if a table is provided, use that; otherwise try to use the class attribute
        if tab is None:
            try:
                t = self.hpxmaskedbcut
            except AttributeError:
                t = Table.read(os.path.join(self.savedir, 'hpx_masked_final.fits'))
                print("loaded hpx_masked_final")
        else:
            if type(tab)==str:
                t = Table.read(tab)
            else:
                t = tab
        hpmap = t[key]
        # which selection function to use
        maglim = 20.0 if self.maglim <= 20.25 else 20.5
        selfunc = self.load_selfunc(maglim=maglim)
        hpmap_corr = np.full(len(hpmap), hp.UNSEEN)
        hpmap_corr = np.divide(hpmap, selfunc[t['hpidx']], out=hpmap_corr, where=((hpmap!=hp.UNSEEN) & (selfunc[t['hpidx']]!=0.)))
        t[key] = hpmap_corr
        return t
    
    def completeness_correct(self, tab=None, key='density', save=True, overwrite=True):
        save_fn = os.path.join(self.savedir_, 'hpx_masked_final.fits')
        self.hpxmaskedbcut, did_step, saved = self._step_handler(self._completeness_correct, tab=tab, key=key,
                                                        save=save, save_fn=save_fn, overwrite=overwrite)
        if saved:
            self.log(f"saved completeness-corrected HEALPix map to {save_fn}")
    

    """
    STEP 5. Correct for trend with absolute ecliptic latitude.
    """
    def _hpx_vs_direction(self, key='density', tab=None, sebsvals=False):
        # if a table is provided, use that; otherwise try to use the class attribute
        if tab is None:
            try:
                t = self.hpxmaskedbcut
            except AttributeError:
                t = Table.read(os.path.join(self.savedir_, 'hpx_masked_final.fits'))
                print(f"loaded {self.comp_tag} hpx_masked_final")
        else:
            if type(tab)==str:
                t = Table.read(tab)
            else:
                assert isinstance(tab, Table)
                t = tab


        # Split t on masked and unmasked
        msk = t[key] < 0
        masked = t[msk]
        t = t[~msk]

        binsize = 1

        # Make linear regression to "correct" density and see if there is an
        # additional component due to the Galactic plane.
        p = np.polyfit(np.abs(t['elat']), t[key], deg=1)

        # get stat
        xs = np.abs(t['elat'].data)
        bins = np.arange(0, 91, binsize)

        binx = bins[0:-1] + binsize / 2
        idx = np.digitize(xs, bins, right=False)

        stat = np.empty((binx.size, 3), dtype=float)
        for i in range(1, binx.size + 1):
            density = t[idx==i][key].data
            if density.size < 10:
                stat[i-1, 0] = np.nan
                stat[i-1, 1] = np.nan
            else:
                stat[i-1, 0] = density.mean()
                stat[i-1, 1] = sem(density)
            stat[i-1, 2] = density.size

        msk = np.isfinite(stat[:,1])
        x, estat = binx[msk], stat[msk]

        # perform linear regression
        p, pcov, fx, z, chi2, dof = linreg(np.abs(x), estat[:,0], 1 / estat[:,1])

        if sebsvals:
            # these values are used by Secrest (he seems to ignore the actual fit result...)
            # Sebastian's values:
            p[0] = -0.05126576725374681
            p[1] = 68.89130135046557

        t['elatdenscorr'] = t[key] - np.polyval(p, np.abs(t['elat'])) + p[1]
        t.sort('hpidx')

        return t
    
    def make_elat_correction(self, tab_fn=None, sebsvals=False, save=True, overwrite=True):
        save_fn = os.path.join(self.savedir_, 'hpx_masked_final_elatdenscorr.fits')
        self.log("making elat density correction...")
        self.hpxelatcorr, did_step, saved = self._step_handler(self._hpx_vs_direction, tab=tab_fn, sebsvals=sebsvals,
                                                                save=save, save_fn=save_fn, overwrite=overwrite)
        if did_step:
            self.log(f"corrected for ecliptic latitude trend")
        if saved:
            self.log(f"saved HEALPix map with elat correction to {save_fn}")
        self._update_working(self.hpxelatcorr)
         

    """
    STEP 6. Get spectral indices.
    """
    # ! BUG: only works with WISE catalogs i.e. W1 and W2
    def _get_alphas_wise(self, tab_fn=None):
        assert self.mag.upper() == 'W1', "W1 magnitudes not found"
        # if a table is provided, use that; otherwise try to use the class attribute
        if tab_fn is None:
            t = self.table
        else:
            t = Table.read(tab_fn)

        # Make vector of needed color
        W1_W2 = (t['w1'] - t['w2']).data

        # Read in lookup table
        a = Table.read('/home/aew492/lss-dipoles/Secrest/alpha_colors.fits')
        a_alpha     = a['alpha'].data
        a_k_W1      = a['k_W1'].data    # Flux conversion factor
        a_nu_W1_iso = a['nu_W1_iso'].data
        a_W1_W2     = a['W1_W2'].data

        N = len(t)
        alpha_W1  = np.nan * np.empty(N, dtype=float)
        k_W1      = np.nan * np.empty(N, dtype=float)
        nu_W1_iso = np.nan * np.empty(N, dtype=float)
        for i in range(N):
            idx_W1_W2    = np.argmin(np.abs(a_W1_W2 - W1_W2[i]))
            k_W1[i]      = a_k_W1[idx_W1_W2]
            alpha_W1[i]  = a_alpha[idx_W1_W2]
            nu_W1_iso[i] = a_nu_W1_iso[idx_W1_W2]
            print("\t%.1f%%" % ((i + 1) / N * 100), end='\r')
        
        # Calculate k such that fnu = k * nu**alpha
        # We're using the Oke & Gunn / Fukugita AB magnitude, which has a
        # zeropoint of 48.60, so the AB - Vega offset for W1 is 2.673.

        W1_AB = t['w1'].data + 2.673
        k = k_W1 * 10**( -W1_AB / 2.5 )

        # Double check to ensure that fnu = k * nu**alpha gives the right mag
        nu, Snu = get_passband('/home/aew492/lss-dipoles/Secrest/passbands/RSR-W1.txt') # !! passbands/ directory
        W1_AB_check = np.empty(len(t), dtype=float)
        for i in range(len(t)):
            fnu = k[i] * nu**alpha_W1[i]
            W1_AB_check[i] = synthmagAB(nu, fnu, Snu)

        abs_dmag = np.abs(W1_AB_check - W1_AB)
        if abs_dmag.max() > 1e-12:
            print("WARNING: Measured and predicted magnitudes differ!")
        
        # Write out table
        out = Table()

        # Even though k is multiplied by nu**alpha, nu**alpha also appears in
        # the denominator of the flux factor so the only remaining Hz is
        # implicit in the cgs unit of the AB zeropoint.
        out['k'] = Column(k, unit = u.erg / u.cm**2 / u.second / u.Hz)
        out['alpha_W1'] = alpha_W1
        out['nu_W1_iso'] = Column(nu_W1_iso, unit = u.Hz)
        # add these columns to the table
        out = hstack((t, out))

        return out
    
    def _get_alphas_gaia(self, tab_fn=None):
        assert self.mag.upper() == 'W1', "W1 magnitudes not found"
    
    def get_alphas(self, tab_fn=None, save=True, overwrite=True):
        self.log("getting spectral indices (alphas)...")
        save_fn = os.path.join(self.savedir, f'{self.catname}_final_alphas.fits')
        self.table_with_alphas, did_step, saved = self._step_handler(self._get_alphas, tab_fn=tab_fn,
                                                                save=save, save_fn=save_fn, overwrite=overwrite)
        if saved:
            self.log(f"saved final catalog with alphas to {save_fn}")
    
    """
    SMOOTHED MAP
    """
    def _smooth_map(self, tab=None):
        """
        Modified from Secrest's `hpx_vs_direction.py`.
        """
        # if a table is provided, use that; otherwise try to use the class attribute
        if tab is None:
            try:
                t = self.hpxelatcorr
            except AttributeError:
                t = self.load_hpxelatcorr()
                print("loaded hpxelatcorr")
        else:
            if type(tab)==str:
                t = Table.read(tab)
            else:
                assert isinstance(tab, Table)
                t = tab

        theta = omega_to_theta(1)
        lent = len(t)
        sc = SkyCoord(t['ra'], t['dec'], frame='icrs')
        t['smoothed'] = -1 * np.ones(lent)
        t['sterr'] = -1 * np.ones(lent)
        t['Nsmooth'] = -1 * np.ones(lent)
        # t['alphasmoothed'] = np.nan * np.ones(lent) # New
        # t['alphasterr'] = np.nan * np.ones(lent)
        t['smoothed_uncorrected'] = -1 * np.ones(lent)
        t['sterr_uncorrected'] = -1 * np.ones(lent)
        for i in range(lent):
            d2d = sc[i].separation(sc)
            msk = d2d < theta
            sample = t[msk]['elatdenscorr']
            t['smoothed'][i] = sample.mean()
            t['sterr'][i] = sem(sample)
            t['Nsmooth'][i] = sample.size
            # sample = t[msk]['alpha']
            # t['alphasmoothed'][i] = sample.mean()
            # t['alphasterr'][i] = sem(sample)
            sample = t[msk]['density']
            t['smoothed_uncorrected'][i] = sample.mean()
            t['sterr_uncorrected'][i] = sem(sample)
            print("\t%.1f%%" % ((i + 1) / lent * 100), end='\r')
        return t 
    
    def smooth_map(self, tab=None, save=True, overwrite=True):
        self.log("smoothing density-corrected map...")
        save_fn = os.path.join(self.savedir_, f'{self.catname}_hpx_smoothed.fits')
        if os.path.exists(save_fn):
            self.hpx_smoothed = Table.read(save_fn, format='fits')
            self.log(f"loaded smoothed map from {save_fn}")
        else:
            self.hpx_smoothed, did_step, saved = self._step_handler(self._smooth_map, tab=tab,
                                                                    save=save, save_fn=save_fn, overwrite=overwrite)
            if saved:
                self.log(f"saved smoothed map to {save_fn}")


    """
    COMPUTE DIPOLE
    """
    def compute_dipole(self, hpxmap, key='elatdenscorr', Cinv=None, logoutput=False, verbose=False):
        """
        Wrapper for `dipole.fit_dipole()`.
        """
        # TODO: confirm that we get the same result with keeping zeros in the map (i.e. no nans)
        map_to_fit = np.zeros(self.NPIX)
        try:
            map_to_fit[hpxmap['hpidx']] = hpxmap[key]
        except KeyError:
            map_to_fit[hpxmap['hpidx']] = hpxmap['denscorr']

        self.amps, self.stderr = fit_dipole(map_to_fit, Cinv=Cinv, fit_zeros=False)
        dipole_amp, dipole_dir = get_dipole(self.amps, frame='icrs', verbose=verbose)
        if logoutput:
            self.log(f"dipole amp = {dipole_amp:.6f}")
            self.log("direction: "+ str(dipole_dir.galactic))
        return dipole_amp, dipole_dir.galactic
    

    """
    HELPERS
    """
    def load_hpxelatcorr(self):
        try:
            return Table.read(os.path.join(self.savedir_, 'hpx_masked_final_elatdenscorr.fits'))
        except FileNotFoundError:
            return Table.read(os.path.join(self.savedir_, 'hpx_masked_final_denscorr.fits'))
    
    def load_smoothed_map(self):
        return Table.read(os.path.join(self.savedir_, f'{self.catname}_hpx_smoothed.fits'))
    
    def load_selfunc(self, maglim=20., selfunc_fn=None):
        selfunc_fn = os.path.join(self.catdir, 'selfuncs',
                                f'selection_function_NSIDE{self.NSIDE}_{self.mag}{maglim:.1f}.fits') \
                    if selfunc_fn is None else selfunc_fn
        return tools.flatten_map(Table.read(selfunc_fn))
    
    def plot_map(self, maptab, key='elatdenscorr', coord=['C','G'], badcolor='white', **kwargs):
        """
        Wrapper for `tools.mollview()`
        """
        map_to_plot = np.zeros(self.NPIX)
        try:
            map_to_plot[maptab['hpidx']] = maptab[key]
        except KeyError:
            map_to_plot[maptab['hpidx']] = maptab['denscorr']
        map_to_plot[map_to_plot==0.] = np.nan

        tools.mollview(map_to_plot, coord=coord, badcolor=badcolor, **kwargs)
    
    def plot_dipole(self, dipdir, **kwargs):
        """
        Wrapper for `hp.newvisufunc.projplot()`
        """
        try:
            projplot(np.pi/2 * u.rad - dipdir.galactic.b.to(u.rad).wrap_at(np.pi * u.rad),
                        dipdir.galactic.l.to(u.rad).wrap_at(np.pi * u.rad), **kwargs)
        except AttributeError:
            projplot(np.pi/2 - dipdir[1], dipdir[0], **kwargs)


if __name__=='__main__':
    main()