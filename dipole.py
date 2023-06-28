"""
Get the expected number-count dipole in a healpy map.
"""

import numpy as np
import healpy as hp

## COORDINATE TRANSFORMATIONS ##
def xyz_to_phitheta(xyz):
    """Given (x,y,z), return (phi,theta) coordinates on the sphere, where phi=LON and theta=LAT."""
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    phi = np.arctan2(y,x)
    theta = np.arccos(z/r)
    return phi, theta


## DIPOLE CONTRIBUTIONS ##
def dipole(theta, phi, dipole_x, dipole_y, dipole_z):
    """Return the signal contribution from the dipole at a certain sky location (theta,phi)."""
    return dipole_x*np.sin(theta)*np.cos(phi) + dipole_y*np.sin(theta)*np.sin(phi) + dipole_z*np.cos(theta)

def dipole_map(amps, NSIDE=64):
    """Generate a healpy dipole map (equatorial coordinates) given four parameters:
       the monopole plus three dipole amplitudes."""
    NPIX = hp.nside2npix(NSIDE)  # number of pixels
    theta, phi = hp.pix2ang(NSIDE, ipix=np.arange(NPIX))  # get (theta,phi) coords of each pixel
    dip = dipole(theta, phi, *amps[1:])  # expected dipole: shape==(NPIX,)
    return amps[0] + dip