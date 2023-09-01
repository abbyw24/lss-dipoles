#!/usr/bin/env python
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack, join
from astropy.coordinates import SkyCoord
from dustmaps.config import config
from dustmaps.planck import PlanckQuery as query
from glob import glob
import time
import datetime
import os
from os import path

s = time.time()

query = query()

# See Figure 9 in Wang & Chen (2019ApJ...877..116W):
Rv = 3.1

# Wang & Chen (2019ApJ...877..116W):
RW1 = 0.039 * Rv
RW2 = 0.026 * Rv

def main(tbl, out):
    print("Reading in %s..." % tbl)
    t = Table.read(tbl)

    print("Grabbing sky coordinates...")
    sc = SkyCoord(t['ra'], t['dec'], unit='deg', frame='icrs')

    print("Querying dust map...")
    t['ebv'] = query(sc) * u.mag

    print("Calculating corrected photometry...")
    t['w1'] = t['w1mpro'] - RW1 * t['ebv']
    t['w2'] = t['w2mpro'] - RW2 * t['ebv']

    # Propagate EBV error. Note that the uncertainties are scaled according
    # the chi-squared value for PSF fitting in CatWISE (see 3.4.1 in
    # Eisenhardt+20). Therefore the additional error terms calculated in
    # Jarrett+11 are probably not safe to use. So we don't.
    t['w1e'] = np.hypot(t['w1sigmpro'], RW1**2 * 0.004**2)
    t['w2e'] = np.hypot(t['w2sigmpro'], RW2**2 * 0.004**2)

    t['w12'] = t['w1'] - t['w2']
    t['w12e'] = np.hypot(t['w1e'], t['w2e'])

    # Remove unneeded keys. Keep coordinate errors for NWAY use on Stripe 82
    # data.
    print("Cleaning up unneeded columns...")
    needed = ['source_id', 'ra', 'dec', 'sigra', 'sigdec', 'sigradec',
              'pmra', 'pmdec', 'sigpmra', 'sigpmdec', 'w1cov', 'w2cov',
              'w1', 'w1e', 'w2', 'w2e', 'w12', 'meanobsmjd',
              'w12e', 'ebv']
    for key in t.keys():
        if key not in needed:
            print(key, " column is not needed")
            t.remove_column(key)

    # Remove objects with w12 < 0.8, and remove objects with w1 >= 17
    # reduce the file size dramatically. Note that 17 is chosen to allow
    # flexibility for flux cut and simulations.
    Nt = len(t)
    print("Applying W1-W2, W1 cuts...")
    t = t[t['w12'] >= 0.8]
    t = t[t['w1'] < 17]
    t = t[t['w1'] > 9]

    print("Saving as %s..." % out)
    t.write(out, overwrite=True)
    print("Complete.")

    del t

    return Nt

tab_dir = 'CatWISE2020/fits' # !!
g = glob(os.path.join(tab_dir, '*.fits')) # !!

cntr = 0            # counts the number of rows collected
for tbl in g:
    if '_corr.fits' not in tbl:
        out = tbl.replace('.fits', '_corr.fits')
        if path.exists(out) == False:
            cntr += main(tbl, out)

print("%i total objects ingested." % cntr)

# Now, join corrected tables
print("Grabbing corrected table data...")
g = glob(os.path.join(tab_dir, '*_corr.fits'))
tables = []
cntr2 = 0
for f in g:
    tab = Table.read(f)
    cntr2 += len(tab)
    tables.append(tab)

print(f"grabbed {len(tables)} tables,") # !!
print("%i total objects ingested." % cntr2) # !!

print("Concatenating...")
t = vstack(tables, metadata_conflicts='silent')

print("%i objects in corrected table." % len(t))

# Correct positions and proper motions according to tile, using file
# provided by Federico Marocco.
print("Reading in position/proper motion corrections...")
r = Table.read('CatWISE2020/CatWISE2020_Table1_20201012.tbl', format='ipac') # !! CatWISE2020/ dir
print(f"{len(r)} rows (tiles) in the correction table.")

t['tile'] = [s.split('_')[0] for s in t['source_id']]
t = join(t, r, keys='tile')

print("Correcting...")
t['ra'] += t['offsetra']
t['dec'] += t['offsetdec']
t['pmra'] += t['offsetpmra']
t['pmdec'] += t['offsetpmdec']

# Empirical correction based on distribution
t['sigpmra'] *= 1.23
t['sigpmdec'] *= 1.18

t.remove_column('offsetra_ta')
t.remove_column('offsetra')
t.remove_column('offsetdec')
t.remove_column('offsetpmra')
t.remove_column('offsetpmdec')

print("Calculating Galactic and ecliptic coordinates..")
sc = SkyCoord(t['ra'], t['dec'], frame='icrs')
lb = sc.galactic
t['l'] = lb.l
t['b'] = lb.b
ec = sc.barycentricmeanecliptic
t['elon'] = ec.lon
t['elat'] = ec.lat

print("Saving as catwise_agns_corr.fits...")
t.write('catwise_agns_corr.fits', overwrite=True)
print("Complete.")

total_time = time.time()-s 
print(f"total time = {datetime.timedelta(seconds=total_time)}")