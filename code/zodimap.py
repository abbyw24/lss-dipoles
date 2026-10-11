import multiprocessing as mp
import time
import os
import sys
import datetime
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body, Angle
import zodipy

def main():

    s = time.time()

    # we want a full sky zodi map from solar elongation angles ~ 90deg
    """INPUTS"""
    nside = 64
    npix = hp.nside2npix(nside)
    frame = 'geocentrictrueecliptic'
    model_name = 'dirbe'	# 'dirbe' is the zodipy default

    # OBSERVATION TIMES
    # start and end obstimes
    start_obstime = Time(datetime.datetime.now().isoformat(),
                format='isot', scale='utc')
    end_obstime = start_obstime + datetime.timedelta(days=365)
    # grid of obstimes
    obstimes = np.linspace(start_obstime.jd, end_obstime.jd, 360 * 4)  # 360 degrees * 4 per degree

    # range of elongation angles
    min_epsilon = 45 << u.deg
    max_epsilon = 135 << u.deg

    # wavelengths
    wl_w1 = 3.4 << u.um
    wl_w2 = 4.6 << u.um
    wl_min = 1.25 << u.um
    wls = [wl_min]

    # get observation coordinates: center of each healpixel with the input frame
    theta, phi = hp.pix2ang(nside,np.arange(npix))
    lat = 90.0 - theta * 180. / np.pi
    lon = phi * 180. / np.pi

    # for each wavelength:
    for wl in wls:
        print(f"starting wavelength {wl}", flush=True)
        """GENERATE TIME-AVERAGED ZODI MAP"""
        # where to store the computed zodi brightness in each pixel
        zodi_sum = np.zeros(npix)
        # count the number of observations in each pixel
        counts = np.zeros(npix, dtype=int)
        model = zodipy.Model(wl, name=model_name) # DIRBE model by default
        for i, obstime in enumerate(obstimes):
            print(f'{i+1} / {len(obstimes)}', end='\r')
            # turn obstime back into Time object
            obstime = Time(obstime, format='jd')
            obscrd_hpx = SkyCoord(lon, lat, obstime=obstime,
                                  unit='deg', frame='geocentrictrueecliptic')
            # get positions relative to the sun
            sol_elons = get_solelons(obscrd_hpx, obstime)
            # mask solar elongation angles
            mask = (sol_elons > min_epsilon) & (sol_elons < max_epsilon)

            # if all angles are masked, move to the next time
            if not np.any(mask):
                continue
	   
            # evaluate only on unmasked pix
            zodi_brightness = model.evaluate(obscrd_hpx[mask], obspos='earth', nprocesses=mp.cpu_count())

            # add to the sum
            zodi_sum[mask] += zodi_brightness.value	# MJy / sr
            counts[mask] += 1 

        # average these maps over obstime and append to the list
        zodimap = np.full(npix, np.nan)
        # which pixels actually have zodi observations
        observed = (counts > 0)
        zodimap[observed] = zodi_sum[observed] / counts[observed]

        """SAVE MAP"""
        save_fn = os.path.join('/work2/08811/aew492/frontera/lss-dipoles/data/maps/zodi',
                                f'zodimap_{min_epsilon}-{max_epsilon}degfromSun_oneyear_{wl.value:.2f}{wl.unit}.fits')

        hdu = fits.PrimaryHDU()
        hdu.data = zodimap
        hdu.header['UNIT'] = ('MJy / sr', 'Zodi intensity unit')
        hdu.header['WL'] = (wl.value, f'Observation wavelength ({wl.unit})')
        hdu.header['MINEPS'] = (min_epsilon.value, f'Min. solar elongation angle observed ({min_epsilon.unit})')
        hdu.header['MAXEPS'] = (max_epsilon.value, f'Max. solar elongation angle observed ({min_epsilon.unit})')
        hdu.header['NSIDE'] = (nside, 'Healpix resolution (NSIDE)')
        hdu.header['MODEL'] = (model_name, 'ZodiPy model')
        hdu.header['COORD'] = (frame, 'Sky coordinate system')
        hdu.header['COMMENT'] = f'Zodi intensity averaged over one year ({len(obstimes)} observations)'
        hdu.writeto(save_fn, overwrite=True)

    total_time = time.time()-s 
    print(f"total time = {datetime.timedelta(seconds=total_time)}", flush=True)


def get_solelons(obscrd, obstime, unit='deg'):
    """
    Return the angle between the observation LOS and the Sun, given input observation coordinate
    (~astropy.coordinates.SkyCoord) and observation time (~astropy.time.Time).
    """
    losuv = obscrd.geocentrictrueecliptic.cartesian
    obspos = get_body('earth', obstime).heliocentrictrueecliptic
    return Angle(np.arccos(losuv.dot(-obspos.cartesian) / obspos.distance)).to(unit)


def get_zodi_slice(model, obstime, wl, min_epsilon, max_epsilon,
                    frame='geocentrictrueecliptic', nside=64):
    """
    Return the zodiacal light emission at a single observation time (~astropy.time.Time)
    in every healpixel at solar elongation angles between `min_epsilon` and `max_epsilon`.
    """
    # get observation coordinates: center of each healpixel with the input frame
    theta, phi = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)))
    lat = 90.0 - theta * 180. / np.pi
    lon = phi * 180. / np.pi
    obscrd_hpx = SkyCoord(lon, lat, obstime=obstime, unit='deg', frame=frame)
    # get positions relative to the sun
    sol_elons = get_solelons(obscrd_hpx, obstime)
    # we want to take the zodi only in pixels ~90deg from the Sun
    #  (where WISE actually observes)
    observing_mask = (sol_elons > min_epsilon) & (sol_elons < max_epsilon)
    # get map (full sky)
    zodimap = np.full(hp.nside2npix(nside), np.nan)
    zodimap[observing_mask] = model.get_emission_pix(
        wl,
        pixels = np.arange(hp.nside2npix(nside)),
        nside = nside,
        obs_time = obstime,
        obs = 'earth',
        coord_in = 'E'  # ecliptic coordinates
    )[observing_mask]
    return zodimap


if __name__=='__main__':
    main()
