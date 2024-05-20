"""

CLASS FOR HANDLING QUASAR CATALOGS FOR DIPOLE (and other multipoles)

"""

import numpy as np
from astropy.table import Table, Column, hstack
import fitsio
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
from healpy.visufunc import projplot
from scipy.stats import sem
from datetime import datetime
import os
import sys

from dipole import fit_dipole, get_dipole
from multipoles import compute_Cells_from_alms_fit
import tools


class QSOSample():

    def __init__(self,
                    catname,
                    initial_catfn,
                    mag, maglim,
                    mask_fn='/scratch/aew492/quasars/catalogs/masks/mask_master_hpx_r1.0.fits',
                    blim=30,
                    badd=1,
                    NSIDE=64,
                    load_init=True,
                    basedir='/scratch/aew492/quasars/catalogs',
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
        self.badd = badd
        self.NSIDE = NSIDE
        self.NPIX = hp.nside2npix(NSIDE)

        # load initial catalog as an astropy table
        if load_init:
            self.load_initial_cattab()

    
    """
    Source tables.
    """
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
    Magnitude and galactic plane cuts.
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
    Selection functions.
    """
    def load_selfunc(self, selfunc=None, return_selfunc=False):
        """
        Sets the `selfunc` attribute.
        """
        if selfunc is None:
            selfunc = os.path.join(self.catdir, 'selfuncs',
                                    f'selection_function_NSIDE{self.NSIDE}_{self.mag}{self.maglim:.1f}.fits')
        if type(selfunc) == str:
            selfunc = tools.flatten_map(Table.read(selfunc))
        assert len(selfunc) == self.NPIX, f"selection function has {len(selfunc)} pixels but should have {self.NPIX}"
        self.selfunc = selfunc
        if return_selfunc == True:
            return selfunc

    def get_selfunc(self, selfunc):
        """
        Helper that returns the completeness in each pixel, based on the current (or lack of) `selfunc`
        attribute. If input `selfunc` is `None`, assumes completeness = 1 in every pixel.
        """
        if selfunc is not None:
            selfunc = self.load_selfunc(selfunc, return_selfunc=True)
        else:
            try:
                selfunc = self.selfunc
            except AttributeError:
                self.log("selection function not provided; assuming completeness = 1 in every pixel")
                selfunc = np.ones(self.NPIX)
        return selfunc

    """
    Healpix maps.
    """

    def define_healpix_mask(self, selfunc=None, min_completeness=0.5):
        """
        Define a healpix mask based on a galactic plane cut, mask file, and
        minimum completeness. Returns 1s for unmasked pixels and NaNs for masked pixels.
        """
        # mask the galactic plane (mask pixels based on their central sky coordinate)
        gal_plane_mask = tools.get_galactic_plane_mask(self.blim, self.NSIDE, frame='icrs')

        # load smaller masks (used in S21): at the pixel level
        small_masks = fitsio.read(self.mask_fn)
        assert len(small_masks) == len(gal_plane_mask) == self.NPIX

        # apply the minimum completeness criterion
        #  (if no selfunc has been provided, assume perfect completeness everywhere)
        selfunc = self.get_selfunc(selfunc)

        self.log(f"masked pixels |b|<{self.blim}deg, from mask_fn, and where completeness < {min_completeness}")
        
        # total mask: galactic plane, smaller masks, plus minimum completeness criterion
        return gal_plane_mask.astype(bool) & small_masks.astype(bool) & (selfunc > min_completeness)

    def construct_overdensity_map(self, mask=None, selfunc=None, min_completeness=0.5):
        """
        Construct a healpix map of source overdensities from the working source table.
        """
        # turn astropy table into healpix map
        datamap = tools.load_catalog_as_map(self.table, NSIDE=self.NSIDE)

        # define the mask
        if mask is None:
            mask = self.define_healpix_mask(selfunc, min_completeness)
        assert len(mask) == self.NPIX

        # selection function
        selfunc = self.get_selfunc(selfunc)

        # mask the data: populate a nan map with the data values in unmasked pixels
        masked_datamap = np.multiply(datamap, mask, where=(mask!=0), out=np.full_like(datamap, np.nan))

        # mean density of the map: mean of the _expected_ counts
        mean_density = np.nanmean(masked_datamap / selfunc)

        # then the overdensity map is
        overdensity_map = np.divide(masked_datamap, selfunc, where=mask,
                                    out=np.full_like(masked_datamap, np.nan)) / mean_density - 1
                            # (fancy divide to avoid RuntimeWarnings where completeness goes to zero)

        return overdensity_map

    """
    MEASURE DIPOLE + OTHER MULTIPOLES
    """
    def measure_dipole_in_overdensity_map(self, sample, selfunc=None, Wmask=0.1, verbose=False):
        """
        Wrapper for `dipole.fit_dipole()`.
        """
        map_to_fit = sample.copy()
        idx_masked = np.isnan(map_to_fit)
        map_to_fit[idx_masked] = 0.
        Cinv = self.get_selfunc(selfunc)
        Cinv[idx_masked] = Wmask
        comps, stderr = dipole.fit_dipole(map_to_fit, Cinv=Cinv, fit_zeros=True)
        if verbose:
            amplitude, direction = dipole.get_dipole(comps)
            self.log(f"best-fit dipole amp. =\t{dipole_amp:.5f}")
            self.log(f"best-fit dipole dir.: ", direction)
        return comps[1:] # since we're fitting overdensities
    
    def Cells_from_alms_fit(self, sample, max_ell, Wmask=0.1, return_alms=False, selfunc=None):
        # wrapper for compute_Cells_from_alms_fit() but taking an overdensity map as input,
        #  to replace all NaN pixels with 0 data and Wmask Cinv
        map_to_fit = sample.copy()
        idx_masked = np.isnan(map_to_fit)
        map_to_fit[idx_masked] = 0.
        Cinv = self.get_selfunc(selfunc)
        Cinv[idx_masked] = Wmask
        return compute_Cells_from_alms_fit(map_to_fit, Cinv, max_ell, return_alms=return_alms)

    """
    Logger.
    """
    def log(self, line):
        print(line, flush=True)