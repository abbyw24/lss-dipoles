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

import dipole
from multipoles import compute_Cells_from_alms_fit
import tools


class QSOSample():

    def __init__(self,
                    initial_catfn,
                    mag, maglim,
                    mask_fn=None,
                    blim=30,
                    NSIDE=64,
                    load_init=True,
                    delete_init=True,
                    basedir='/scratch1/08811/aew492/quasars/catalogs'):

        # asserts
        assert initial_catfn.endswith('.fits'), "initial catalog must be a fits file"

        # input parameters
        self.initial_catfn = initial_catfn
        self.mag = mag.upper()
        self.maglim = maglim
        self.mask_fn = mask_fn
        self.blim = blim
        self.NSIDE = NSIDE
        self.NPIX = hp.nside2npix(NSIDE)

        # load initial catalog as an astropy table
        if load_init:
            self.load_initial_cattab()
            if delete_init:
                del self.initial_cattab

    
    """
    Source tables.
    """
    def load_initial_cattab(self):
        self.log(f"loading initial catalog, {self.initial_catfn}")
        self.initial_cattab = Table.read(self.initial_catfn, format='fits')
        if not hasattr(self, 'cattab'):
            self._update_working(self.initial_cattab)
        self.log(f"{len(self.initial_cattab)} sources in initial catalog.")

    def _update_working(self, table):
        self.table = table
    

    """
    Magnitude and galactic plane cuts.
    """
    def cut_mag(self):
        if self.mag == 'G':
            key = 'phot_g_mean_mag'
        else:
            key = self.mag.lower()
        self._update_working(self.table[self.table[key] <= self.maglim])
        self.log(f"cut {self.mag} > {self.maglim} -> {len(self.table)} sources left.")

    def cut_galactic_plane(self):
        self._update_working(self.table[np.abs(self.table['b']) > self.blim])
        self.log(f"cut |b| <= {self.blim} -> {len(self.table)} sources left.")
    

    """
    Selection functions.
    """
    def set_selfunc(self, selfunc):
        """
        Sets the `selfunc` attribute. If input `selfunc` is `'None'`, assumes completeness = 1 in every pixel.
        """
        if np.all(selfunc == 'None'):
            selfunc = np.ones(self.NPIX)
        elif type(selfunc) == np.ndarray:
            assert len(selfunc) == self.NPIX, f"length of input selfunc ({len(selfunc)}) must match NPIX ({self.NPIX})"
            selfunc = selfunc
        else:
            assert type(selfunc) == str, "input selfunc must be a file string, 'None', or a numpy array"
        if type(selfunc) == str:
            selfunc = tools.flatten_map(Table.read(selfunc))
        self.selfunc = selfunc

    def get_selfunc(self, selfunc):
        """
        Helper function to get the selection function. If input `selfunc` is `None`,
        assumes completeness = 1 in every pixel.
        """
        # if we don't input a selfunc, try to load the current attribute
        if selfunc is None:
            try:
                return self.selfunc
            except AttributeError:
                self.log("selection function not provided; assuming completeness = 1 in every pixel")
                return np.ones(self.NPIX)
        # otherwise, use the input selfunc:
        else:
            if type(selfunc) == str:
                if selfunc == 'None':
                    selfunc = np.ones(self.NPIX)
                else:
                    selfunc = tools.flatten_map(Table.read(selfunc))
            assert type(selfunc) == np.ndarray
            return selfunc


    """
    Healpix maps.
    """
    def define_healpix_mask(self, selfunc=None, min_completeness=0.5):
        """
        Define a healpix mask based on a galactic plane cut, mask file, and
        minimum completeness. Returns 1s for unmasked pixels and NaNs for masked pixels.
        """
        self.log("defining healpix mask...")
        # mask the galactic plane (mask pixels based on their central sky coordinate)
        gal_plane_mask = tools.get_galactic_plane_mask(self.blim, self.NSIDE, frame='icrs')

        # load smaller masks (used in S21): at the pixel level
        small_masks = fitsio.read(self.mask_fn)
        assert len(small_masks) == len(gal_plane_mask) == self.NPIX

        # apply the minimum completeness criterion
        #  (if no selfunc has been provided, assume perfect completeness everywhere)
        selfunc = self.get_selfunc(selfunc)

        self.log(f"\tmasked pixels |b|<{self.blim}deg, from mask_fn, and where completeness < {min_completeness}")

        # total mask: galactic plane, smaller masks, plus minimum completeness criterion
        self.mask = gal_plane_mask.astype(bool) & small_masks.astype(bool) & (selfunc > min_completeness)

        # also save min completeness used
        self.min_completeness = min_completeness

    def load_datamap(self):
        """
        Loads the working source table as a healpix map with resolution `NSIDE`.
        """
        self.datamap = tools.load_catalog_as_map(self.table, NSIDE=self.NSIDE)

    def construct_masked_datamap(self, selfunc=None, min_completeness=0.5, return_map=False):
        """
        Mask the catalog at the healpix level, and correct the densities by a selection function.
        """
        if not hasattr(self, 'datamap'):
            self.load_datamap()
        selfunc = self.get_selfunc(selfunc)
        if not hasattr(self, 'mask') or min_completeness != self.min_completeness:
            self.define_healpix_mask(selfunc, min_completeness)
        masked_datamap_uncorr = np.multiply(self.datamap, self.mask, where=(self.mask!=0), out=np.full_like(self.datamap, np.nan))
        # correct by selection function
        self.masked_datamap = masked_datamap_uncorr / selfunc
        if return_map:
            return self.masked_datamap

    def construct_overdensity_map(self, selfunc=None, min_completeness=0.5):
        """
        Construct a healpix map of source overdensities from the working source table.
        """

        self.log("constructing overdensity map")

        # get selection function
        selfunc = self.get_selfunc(selfunc)

        # construct the masked datamap from self.table
        self.construct_masked_datamap(selfunc=selfunc, min_completeness=min_completeness)

        # then the overdensity map is
        overdensity_map = self.masked_datamap / np.nanmean(self.masked_datamap) - 1

        return overdensity_map


    """
    Logger.
    """
    def log(self, line):
        print(line, flush=True)