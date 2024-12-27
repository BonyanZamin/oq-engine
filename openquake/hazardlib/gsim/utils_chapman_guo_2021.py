# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2024 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Utilities for implementation of the Chapman and Guo (2021) Coastal Plain
site amplification model as required for the 2023 Conterminous US NSHMP.

Chapman, M.C. and Guo, Z., 2021, A response spectral ratio model to account
for amplification and attenuation effects in the Atlantic and Gulf coastal
plain: Bulletin of the Seismological Society of America, 111 (4), pp.1849-1867.

The majority of the USGS java code for implementing this model is available here:
https://code.usgs.gov/ghsc/nshmp/nshmp-lib/-/blob/main/src/main/java/gov/usgs/earthquake/nshmp/gmm/ChapmanGuo_2021.java?ref_type=heads

The code for obtaining the f_cpa parameter is available within
the USGS java code for the NGAEast GMMs which is taken from here:
https://code.usgs.gov/ghsc/nshmp/nshmp-lib/-/blob/main/src/main/java/gov/usgs/earthquake/nshmp/gmm/NgaEast.java
"""
import numpy as np

# z_sed
Z = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5,
              3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])

# Mw
M = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.2])

# rrup
R = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0,
              400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
              1100.0, 1200.0, 1350.0, 1500.0])


def get_zscale(z_sed):
    """
    Provide the depth scaling factor for application of reference site
    scaling.

    :param z_sed: Depth to sediment site parameter considered in the
                  Chapman and Guo (2021) Coastal Plains site amplification
                  model as part of the 2023 Conterminous US NSHMP.
    """
    Z_CUT = 2.
    s = 1. - np.exp((-1 * z_sed) / Z_CUT)
    return s ** 4


def get_fcpa(ctx, z_scale, psa_df):
    """
    Get f_cpa param for the given sediment depth, Mw and rjb.

    This function returns both f_cpa and z_scale within a dictionary which is
    passed into the nga_east functions for computation of mean ground-motions.
    """
    # Set default f_cpa of zero for each site
    f_cpa = np.full(len(ctx.vs30), 0.)

    # Get sites with sed. depth scaling greater than 0
    mask_z = z_scale > 0.

    # For these sites recompute f_cpa parameter
    if np.any(mask_z):
        f_cpa[mask_z] = get_psa_ratio(ctx, psa_df)
        
    # Put Coastal Plain params into a dict for passing into nga_east functions
    coastal = {'f_cpa': f_cpa, 'z_scale': z_scale}

    return coastal


def get_fraction(lo, hi, value):
    """
    Get fraction of admitted value between lower and upper values.
    """
    if value < lo:
        return 0.0
    elif value > hi:
        return 1.0
    else:
        return (value - lo) / (hi - lo)


def get_data(psa_df):
    """
    Get the z_sed for each z_sed, mag and rrup combination within
    an ndarray.
    """
    # Create empty ndarray
    data = np.zeros((len(Z), len(M), len(R)))

    # Per z_sed, mag, rrup
    for zi in range(len(Z)):
        for mi in range(len(M)):
            for ri in range(len(R)):
                # Get the row associated with the combination
                row = (psa_df[(psa_df['zsed'] == Z[zi]) &
                      (psa_df['magnitude'] == M[mi]) &
                      (psa_df['distance'] == R[ri])])
                # Store within the ndarray
                data[zi][mi][ri] = row.psa_ratio.values[0]

    return data


def interpolate(lo, hi, fraction):
    """
    Perform weighted interpolation between the lower and upper values.
    """
    return lo + fraction * (hi - lo)


def get_psa_ratio(ctx, psa_df):
    """
    Get the PSA ratio for each ctx's sediment depth, M and rrup for the
    given IMT.
    """
    # Get psa data into ndarray first
    data = get_data(psa_df)

    # Get values per ctx into arrays
    z = np.array([cx.z_sed for cx in ctx])
    m = np.array([cx.mag for cx in ctx])
    r = np.array([cx.rrup for cx in ctx])

    # Get nearest idx per ctx
    i = np.searchsorted(Z, z) - 1
    j = np.searchsorted(M, m) - 1
    k = np.searchsorted(R, r) - 1

    # Get fractions between neighbouring idx
    zf = (z - Z[i]) / (Z[i + 1] - Z[i])
    mf = (m - M[j]) / (M[j + 1] - M[j])
    rf = (r - R[k]) / (R[k + 1] - R[k])

    # Get PSA ratios from data ndarray
    z1m1r1 = data[i, j, k]
    z1m1r2 = data[i, j, k + 1]
    z1m2r1 = data[i, j + 1, k]
    z1m2r2 = data[i, j + 1, k + 1]
    z2m1r1 = data[i + 1, j, k]
    z2m1r2 = data[i + 1, j, k + 1]
    z2m2r1 = data[i + 1, j + 1, k]
    z2m2r2 = data[i + 1, j + 1, k + 1]

    z1m1 = interpolate(z1m1r1, z1m1r2, rf)
    z1m2 = interpolate(z1m2r1, z1m2r2, rf)
    z2m1 = interpolate(z2m1r1, z2m1r2, rf)
    z2m2 = interpolate(z2m2r1, z2m2r2, rf)

    z1 = interpolate(z1m1, z1m2, mf)
    z2 = interpolate(z2m1, z2m2, mf)

    # Get the interpolated PSA ratios
    psa_ratios = interpolate(z1, z2, zf)

    return np.log(psa_ratios) # Into log space