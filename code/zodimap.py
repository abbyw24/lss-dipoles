import numpy as np
import healpy as hp
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body, Angle
from zodipy import Zodipy
import time
import datetime
import os
import sys

def main():

    s = time.time()

    # we want a full sky zodi map from solar elongation angles ~ 90deg
    """INPUTS"""
    model_name = 'dirbe'
    model = Zodipy(model_name)
    nside = 64
    frame = 'geocentrictrueecliptic'

    # OBSERVATION TIMES
    # start and end obstimes
    start_obstime = Time(datetime.datetime.now().isoformat(),
                format='isot', scale='utc')
    end_obstime = start_obstime + datetime.timedelta(days=365)
    # grid of obstimes
    obstimes = np.linspace(start_obstime.jd, end_obstime.jd, 360 * 4)  # 360 degrees * 4 per degree

    # range of elongation angles
    min_epsilon = 88 << u.deg
    max_epsilon = 92 << u.deg

    # wavelengths
    wl_w1 = 3.4 << u.um
    wl_w2 = 4.6 << u.um
    wl_min = 1.25 << u.um
    wls = [wl_min, wl_w1, wl_w2]

    # for each wavelength:
    for wl in wls:
        print(f"starting wavelength {wl}", flush=True)
        """GENERATE TIME-AVERAGED ZODI MAP"""
        maps = np.full((len(obstimes), hp.nside2npix(nside)), np.nan)
        for i, obstime in enumerate(obstimes):
            print(f'{i+1} / {len(obstimes)}', end='\r')
            maps[i] = get_zodi_slice(model, Time(obstime, format='jd'),
                                        wl=wl, frame=frame,
                                        min_epsilon=min_epsilon,
                                        max_epsilon=max_epsilon)
        # average these maps over obstime and append to the list
        zodimap = np.nanmean(maps, axis=0)

        """SAVE MAP"""
        save_fn = os.path.join('/scratch/aew492/quasars/maps/zodi',
                                f'script_zodimap_90degfromSun_oneyear_{wl.value:.2f}{wl.unit}.fits')

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
    obscrd_hpx = SkyCoord(lon, lat, unit='deg', frame=frame)
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