"""
Get the expected number-count dipole in a healpy map.
"""

import numpy as np
import healpy as hp

def radec_to_thetaphi(ra, dec):
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180
    return theta, phi

def thetaphi_to_radec(theta, phi):
    ra = theta * 180/np.pi
    dec = 90 - phi * 180/np.pi
    return ra, dec

def dipole(theta, phi, dipole_x, dipole_y, dipole_z):
    """Return the signal contribution from the dipole at a certain sky location (theta,phi)."""
    return dipole_x*np.sin(theta)*np.cos(phi) + dipole_y*np.sin(theta)*np.sin(phi) + dipole_z*np.cos(theta)

def dipole_map(amps, NSIDE=64):
    """Generate a healpy dipole map (equatorial coordinates) given four parameters:
       the monopole plus three dipole amplitudes."""
    NPIX = hp.nside2npix(NSIDE)  # number of pixels
    theta, phi = hp.pix2ang(NSIDE, ipix=np.arange(NPIX))  # get (theta,phi) coords of each pixel
    dip = dipole(theta, phi, **amps[1:])  # expected dipole: shape==(NPIX,)
    return amps[0] + dip