import numpy as np
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import healpy as hp
import os
import sys


def main():

    nside_lo = 64
    factor = 2.5
    nside_hi = 1024
    galcut = 0
    overwrite = True

    # load masks
    masktab = Table.read('/scratch/aew492/quasars/catalogs/masks/MASKS_exclude_master_final.fits')

    # select specific objects to mask
    obj_list = None # ['LMC','SMC']

    # save file
    save_dir = f'/scratch/aew492/quasars/catalogs/masks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = os.path.join(save_dir, f'mask_master_hpx_r{factor:.1f}.fits')

    if obj_list is not None:
        obj_ids = np.array([x.strip(' ') for x in masktab['obj']])  # remove extra spaces from object IDs
        idx = np.array([np.argwhere(obj_ids==obj) for obj in obj_list]).flatten()
        masks = masktab[idx]
    else:
        masks = masktab
    # masks = masktab[masktab['radius'] > 5]  # this includes LMC, SMC, and galactic center
    
    mask = construct_hpx_mask(masks, factor=factor, nside_lo=nside_lo, nside_hi=nside_hi, galcut=galcut)

    # save
    hdu = fits.PrimaryHDU()
    hdu.data = mask
    if os.path.exists(save_fn) and overwrite is True:
        os.remove(save_fn)
    hdu.writeto(save_fn)
    print(f"saved to {save_fn}")


def construct_hpx_mask(masks,
                        factor=1,
                        nside_lo=64,
                        nside_hi=1024,
                        galcut=0):
    
    # start with a high resolution mask and then downgrade to the resolution we'll use in the other maps
    hires_mask = makeMask(masks, galcut=galcut, nside=nside_hi, masking='onesided', factor=factor)
    lores_mask = hp.ud_grade(hires_mask, nside_out=nside_lo)
    lores_mask[(lores_mask!=1)] = 0  # when we downgrade, set any pixels not equal to 1 as 0

    # convert to equatorial coords
    ang = hp.pix2ang(nside_lo, np.arange(hp.nside2npix(nside_lo)))
    rot = hp.Rotator(coord=['C','G'])
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside_lo, *new_ang)
    lores_mask_eq = lores_mask[new_pix]

    return lores_mask_eq


def EquatorialtoGalactic(ra, dec):
    """ Returns Galactic coordinates for RA and Dec (in Equatorial Coordinates) """
    skc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    return skc.galactic.l.value, skc.galactic.b.value


def make_galmask(nside=256, planecut=30) :
    """
    Computes a Galactic plane mask
    """
    mask = np.ones(hp.nside2npix(nside))
    vector = hp.ang2vec(0, 90, lonlat=1) # (l,b)=(0,0)
    indices = hp.query_disc(nside, vector, np.deg2rad(90+planecut))
    mask[indices] = 0
    indices = hp.query_disc(nside, vector, np.deg2rad(90-planecut))
    mask[indices] = 1
    
    return mask

def offset_by(lon, lat, posang, distance):
	"""
	Point with the given offset from the given point.
	Parameters
	----------
	lon, lat, posang, distance : `Angle`, `~astropy.units.Quantity` or float
		Longitude and latitude of the starting point,
		position angle and distance to the final point.
		Quantities should be in angular units; floats in radians.
		Polar points at lat= +/-90 are treated as limit of +/-(90-epsilon) and same lon.
	Returns
	-------
	lon, lat : `~astropy.coordinates.Angle`
		The position of the final point.  If any of the angles are arrays,
		these will contain arrays following the appropriate `numpy` broadcasting rules.
		0 <= lon < 2pi.
	Notes
	-----
	"""
	
	# Calculations are done using the spherical trigonometry sine and cosine rules
	# of the triangle A at North Pole,   B at starting point,   C at final point
	# with angles	 A (change in lon), B (posang),			C (not used, but negative reciprocal posang)
	# with sides	  a (distance),	  b (final co-latitude), c (starting colatitude)
	# B, a, c are knowns; A and b are unknowns
	# https://en.wikipedia.org/wiki/Spherical_trigonometry
	
	cos_a = np.cos(np.deg2rad(distance))
	sin_a = np.sin(np.deg2rad(distance))
	cos_c = np.sin(np.deg2rad(lat))
	sin_c = np.cos(np.deg2rad(lat))
	cos_B = np.cos(np.deg2rad(posang))
	sin_B = np.sin(np.deg2rad(posang))
	
	# cosine rule: Know two sides: a,c and included angle: B; get unknown side b
	cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
	
	# sin_b = np.sqrt(1 - cos_b**2)
	# sine rule and cosine rule for A (using both lets arctan2 pick quadrant).
	# multiplying both sin_A and cos_A by x=sin_b * sin_c prevents /0 errors
	# at poles.  Correct for the x=0 multiplication a few lines down.
	
	# sin_A/sin_a == sin_B/sin_b	# Sine rule
	xsin_A = sin_a * sin_B * sin_c
	
	# cos_a == cos_b * cos_c + sin_b * sin_c * cos_A  # cosine rule
	xcos_A = cos_a - cos_b * cos_c
	
	A = Angle(np.arctan2(xsin_A, xcos_A), u.radian)
	# Treat the poles as if they are infinitesimally far from pole but at given lon
	# The +0*xsin_A is to broadcast a scalar to vector as necessary
	w_pole = np.argwhere((sin_c + 0*xsin_A) < 1e-12)
	
	if len(w_pole) > 0:
	
		# For south pole (cos_c = -1), A = posang; for North pole, A=180 deg - posang
		A_pole = (90*u.deg + cos_c*(90*u.deg-Angle(posang, u.radian))).to(u.rad)
		try:
			A[w_pole] = A_pole[w_pole]
		except TypeError as e: # scalar
			A = A_pole
	
	outlon = (Angle(lon, u.deg) + A).wrap_at(360.0*u.deg).to(u.deg)
	outlat = Angle(np.arcsin(cos_b), u.radian).to(u.deg)

	return outlon.value, outlat.value

def evaluateEllipse(lon, lat, lon_0, lat_0, semi_major, e, theta, verbose=False):
    """Evaluate the model (static function)."""
    # find the foci of the ellipse
    c = semi_major * e
    lon_1, lat_1 = offset_by(lon_0, lat_0, 90 - theta, c)
    lon_2, lat_2 = offset_by(lon_0, lat_0, 270 - theta, c)
    
    if verbose:
        print(lon_1, lat_1, lon_2, lat_2, semi_major, e, theta)

    sep_1 = np.rad2deg(hp.rotator.angdist(hp.dir2vec(lon, lat, lonlat=True), hp.dir2vec(lon_1, lat_1, lonlat=True)))
    sep_2 = np.rad2deg(hp.rotator.angdist(hp.dir2vec(lon, lat, lonlat=True), hp.dir2vec(lon_2, lat_2, lonlat=True)))
    in_ellipse = sep_1 + sep_2 <= 2 * semi_major
    
    if verbose:
        print(len(sep_1), len(sep_1[in_ellipse]))

    return in_ellipse

def makeMask(psmasks, nside=256, galcut=0, masking='symmetric', factor=1.):
    """
    Computes a mask given a file that specifies locations and extent of point sources
    """
    
    mask = np.ones(hp.nside2npix(nside))
    pixels = np.arange(hp.nside2npix(nside))
    mask_lon,mask_lat = hp.pix2ang(nside,pixels,lonlat=True)

    cmasks = psmasks[(psmasks['pa']<=2) * (psmasks['radius']<15)]  # the only row with radius >15 is the galactic center 'GC'
    emasks = psmasks[(psmasks['pa']>2)]
    
    cmask_lon, cmask_lat, cmask_rad = *EquatorialtoGalactic(cmasks['ra'], cmasks['dec']), cmasks['radius']
    
    for lon, lat, radius in zip(cmask_lon, cmask_lat, cmask_rad):
        vector = hp.ang2vec(lon, lat, lonlat=True)
        indices = hp.query_disc(nside, vector, factor * np.deg2rad(radius))
        mask[indices] = 0
        if masking=='symmetric':
            indices = hp.query_disc(nside, -vector, factor * np.deg2rad(radius))
            mask[indices] = 0
    
    emask_lon, emask_lat, emask_rad, emask_ba, emask_pa = \
                *EquatorialtoGalactic(emasks['ra'], emasks['dec']), emasks['radius'], emasks['ba'], emasks['pa']

    for lon, lat, rad, ba, pa in zip(emask_lon, emask_lat, emask_rad, emask_ba, emask_pa):
        ell = evaluateEllipse(mask_lon, mask_lat, lon, lat, factor * rad, ba, pa)
        mask[ell] = 0
        if masking=='symmetric':
            ell = evaluateEllipse(mask_lon, mask_lat, lon+180., -1. * lat, factor * rad, ba, -1. * pa)
            mask[ell] = 0
    
    if galcut: 
        planemask = make_galmask(nside=nside, planecut=galcut)
    else:
        planemask = np.ones_like(mask)
    
    return mask * planemask


if __name__=='__main__':
    main()