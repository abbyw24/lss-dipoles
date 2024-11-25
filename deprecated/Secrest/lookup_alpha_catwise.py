#!/usr/bin/env python
import numpy as np
from astropy.table import Table, Column, hstack
import astropy.units as u
from get_colors import synthmagAB, get_passband


# Input table
# tbl = 'catwise_agns_masked.fits' # !!
tbl = '/scratch/aew492/quasars/catalogs/catwise2020/catwise_agns_masked_final.fits' # !!

print("Loading in table...")
t = Table.read(tbl)

# Make vector of needed color
W1_W2 = (t['w1'] - t['w2']).data

# Read in lookup table
a = Table.read('/home/aew492/lss-dipoles/Secrest/alpha_colors.fits')

a_alpha     = a['alpha'].data
a_k_W1      = a['k_W1'].data    # Flux conversion factor
a_nu_W1_iso = a['nu_W1_iso'].data
a_W1_W2     = a['W1_W2'].data

# Handy shorthand for a bit of extra speed
def closest(dx):
    return np.argmin(np.abs(dx))

print("Calculating...")
N = len(t)
alpha_W1  = np.nan * np.empty( N, dtype=float )
k_W1      = np.nan * np.empty( N, dtype=float )
nu_W1_iso = np.nan * np.empty( N, dtype=float )
for i in range(N):
    idx_W1_W2    = closest(a_W1_W2 - W1_W2[i])
    k_W1[i]      = a_k_W1[idx_W1_W2]
    alpha_W1[i]  = a_alpha[idx_W1_W2]
    nu_W1_iso[i] = a_nu_W1_iso[idx_W1_W2]

    print("\t%.1f%%" % ((i + 1) / N * 100), end='\r')


# Calculate k such that fnu = k * nu**alpha
# We're using the Oke & Gunn / Fukugita AB magnitude, which has a
# zeropoint of 48.60, so the AB - Vega offset for W1 is 2.673.

W1_AB = t['w1'].data + 2.673
k = k_W1 * 10**( -W1_AB / 2.5 )

# Double check to ensure that fnu = k * nu**alpha gives the right mag
nu, Snu = get_passband('/home/aew492/lss-dipoles/Secrest/passbands/RSR-W1.txt') # !! passbands/ directory
W1_AB_check = np.empty(len(t), dtype=float)
for i in range(len(t)):
    fnu = k[i] * nu**alpha_W1[i]
    W1_AB_check[i] = synthmagAB(nu, fnu, Snu)

abs_dmag = np.abs(W1_AB_check - W1_AB)
if abs_dmag.max() > 1e-12:
    print("WARNING: Measured and predicted magnitudes differ!")


# Write out table
out = Table()

# Even though k is multiplied by nu**alpha, nu**alpha also appears in
# the denominator of the flux factor so the only remaining Hz is
# implicit in the cgs unit of the AB zeropoint.
out['k'] = Column(k, unit = u.erg / u.cm**2 / u.second / u.Hz)

out['alpha_W1'] = alpha_W1
out['nu_W1_iso'] = Column(nu_W1_iso, unit = u.Hz)
out = hstack((t, out))

# !!
print("success")
# !!

out.write(tbl.replace('.fits', '_alpha.fits'), overwrite=True)
