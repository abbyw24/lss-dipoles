"""
Get the expected number-count dipole in a healpy map.
"""

import numpy as np

def radec_to_thetaphi(ra, dec):
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180
    return theta, phi

def thetaphi_to_radec(theta, phi):
    ra = theta * 180/np.pi
    dec = 90 - phi * 180/np.pi
    return ra, dec

def dipole(ra, dec, dipamps):
    """Return the signal contribution from the dipole at a certain sky location (ra,dec)."""
    dx, dy, dz = dipamps
    theta, phi = radec_to_thetaphi(ra, dec)
    return dx*np.sin(theta)*np.cos(phi) + dy*np.sin(theta)*np.sin(phi) + dz*np.cos(theta)