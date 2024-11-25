#!/usr/bin/env python
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# sample = 'catwise_agns_masked_final_alpha.fits' # !!
sample = '/scratch/aew492/quasars/catalogs/catwise2020/catwise_agns_masked_final_alpha.fits'
#sample = 'randsky.fits'

t = Table.read(sample)
sc = SkyCoord(t['ra'], t['dec'], frame='icrs')
theta = -(sc.dec - 90 * u.deg).radian
phi = sc.ra.radian

print("Read in %i sources." % len(t))

nside = 64
npix = hp.nside2npix(nside)

hpidx = hp.ang2pix(nside, theta, phi)

hpmap = np.zeros(npix, dtype=float)
alphamap = np.zeros(npix, dtype=float)
alpha = t['alpha_W1'].data
for i in range(len(t)):
    hpmap[hpidx[i]] += 1
    alphamap[hpidx[i]] += alpha[i]

# Turn map of alphas into mean
idx = np.where(hpmap > 0)
alphamap[idx] /= hpmap[idx]
alphamap[alphamap==0] = np.nan

# hpxmap is total per pixel. Convert to per deg^2
skyarea = 4 * np.pi * (180 / np.pi)**2
hpmap *= npix / skyarea

# Set neighbors of 0 pixel count to UNSEEN, as these are masked region
# bordering pixels.

# There are masks smaller than the HEALPix pixel size, which leaves
# under-dense, but not zero, pixels. Optionally, mask these.
#m = Table.read('exclude_master_final.fits')
#midx = hp.ang2pix(nside, m['ra'].data, m['dec'].data, lonlat=True)
#hpmap[midx] = 0

idx0 = np.where(hpmap==0)[0]
indices = np.empty((idx0.size, 8), dtype=int)
for i in range(idx0.size):
    indices[i] = hp.pixelfunc.get_all_neighbours(nside, idx0[i])

indices = np.unique(indices.flatten())

hpmap[idx0] = hp.pixelfunc.UNSEEN
hpmap[indices] = hp.pixelfunc.UNSEEN


# Get ra, dec of pixels
hpidx = np.arange(npix)
lon,lat = hp.pix2ang(nside, hpidx, lonlat=True)
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
hpx['alpha'] = alphamap

# Make dummy radii for masking
hpx['primrad'] = 0.0 * u.deg
hpx['secrad'] = 0.0 * u.deg
hpx['pa'] = 0.0 * u.deg

hpx.write(sample.replace('.fits', '_hpx.fits'), overwrite=True)