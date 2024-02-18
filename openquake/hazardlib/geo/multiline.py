# The Hazard Library
# Copyright (C) 2021 GEM Foundation
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
Module :mod:`openquake.hazardlib.geo.multiline` defines
:class:`openquake.hazardlib.geo.multiline.Multiline`.
"""

import numpy as np
import pandas as pd
from openquake.baselib.performance import compile
from openquake.hazardlib.geo import utils, geodetic
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.geodetic import geodetic_distance, azimuth


def get_endpoints(coos):
    """
    :returns a mesh of shape 2L
    """
    lons = np.concatenate([coo[[0, -1], 0] for coo in coos])  # shape 2L
    lats = np.concatenate([coo[[0, -1], 1] for coo in coos])  # shape 2L
    return Mesh(lons, lats)


def get_avg_azim_flipped(lines):
    # compute the overall strike and the origin of the multiline
    # get lenghts and average azimuths
    llenghts = np.array([ln.get_length() for ln in lines])
    if all(len(ln) == 2 for ln in lines):
        # fast lane for the engine
        coos = np.array([ln.coo for ln in lines])
        azimuths = geodetic.azimuths(coos) 
    else:
        # slow lane, only for some tests in hazardlib
        azimuths = np.array([line.average_azimuth() for line in lines])

    # determine the flipped lines
    flipped = get_flipped(llenghts, azimuths)
    
    # Compute the average azimuth
    for i in np.nonzero(flipped)[0]:
        azimuths[i] = (azimuths[i] + 180) % 360  # opposite azimuth
    avg_azim = utils.angular_mean(azimuths, llenghts) % 360
    return avg_azim, flipped


class MultiLine(object):
    """
    A collection of polylines with associated methods and attributes. For the
    most part, these are used to compute distances according to the GC2
    method.
    """
    def __init__(self, lines, u_max=None):
        self.coos = [ln.coo for ln in lines]
        avg_azim, self.flipped = get_avg_azim_flipped(lines)
        ep = get_endpoints(self.coos)
        olon, olat, self.soidx = get_origin(ep, avg_azim)

        # compute the shift with respect to the origins
        origins = []
        for idx in self.soidx:
            flip = int(self.flipped[idx])
            # if the line is flipped take the point 1 instead of 0
            origins.append(lines[idx].coo[flip])
        self.shift = get_coordinate_shift(
            np.array(origins), olon, olat, avg_azim)
        self.u_max = u_max

    def set_u_max(self):
        """
        If not already computed, compute .u_max, set it and return it.
        """
        if self.u_max is None:
            _, us = self.get_tu(get_endpoints(self.coos))
            self.u_max = np.abs(us).max()
        return self.u_max

    def gen_lines(self):
        """
        :yields: (flipped) lines in the right order
        """
        for idx in self.soidx:
            coo = self.coos[idx]
            if self.flipped[idx]:
                coo = np.flipud(coo)
            yield Line.from_coo(coo)

    # used in event based too
    def get_tu(self, mesh):
        """
        Given a mesh, computes the T and U coordinates for the multiline
        """
        S = len(self.coos)  # number of lines == number of surfaces
        N = len(mesh)
        tuw = np.zeros((3, S, N), np.float32)
        for s, line in enumerate(self.gen_lines()):
            tuw[:, s] = line.get_tuw(mesh)
        return _get_tu(self.shift, tuw)

    def get_tuw_df(self, sites):
        # debug method to be called in genctxs
        sids = []
        ts = []
        us = []
        ws = []
        ls = []
        for line in self.gen_lines():
            sline = str(line)
            tu, uu, we = line.get_tuw(sites)
            for s, sid in enumerate(sites.sids):
                sids.append(sid)
                ts.append(tu[s])
                us.append(uu[s])
                ws.append(we[s])
                ls.append(sline)
        dic = dict(sid=sids, line=ls, t=ts, u=us, w=ws)
        return pd.DataFrame(dic)

    def __str__(self):
        return ';'.join(str(Line.from_coo(coo)) for coo in self.coos)


def get_flipped(llens, avgaz):
    """
    :returns: a boolean array with the flipped lines
    """
    # Find general azimuth trend
    ave = utils.angular_mean(avgaz, llens) % 360

    # Find the sections whose azimuth direction is not consistent with the
    # average one
    flipped = np.zeros((len(avgaz)), dtype=bool)
    if (ave >= 90) & (ave <= 270):
        # This is the case where the average azimuth in the second or third
        # quadrant
        idx = (avgaz >= (ave - 90) % 360) & (avgaz < (ave + 90) % 360)
    else:
        # In this case the average azimuth points toward the northern emisphere
        idx = (avgaz >= (ave - 90) % 360) | (avgaz < (ave + 90) % 360)

    delta = abs(avgaz - ave)
    scale = np.abs(np.cos(np.radians(delta)))
    ratio = np.sum(llens[idx] * scale[idx]) / np.sum(llens * scale)

    strike_to_east = ratio > 0.5
    if strike_to_east:
        flipped[~idx] = True
    else:
        flipped[idx] = True

    return flipped


def get_origin(ep: Mesh, avg_strike: float):
    """
    Compute the origin necessary to calculate the coordinate shift

    :returns:
        The longitude and latitude coordinates of the origin and an array with
        the indexes used to sort the lines according to the origin
    """

    # Project the endpoints
    proj = utils.OrthographicProjection.from_lons_lats(ep.lons, ep.lats)
    px, py = proj(ep.lons, ep.lats)

    # Find the index of the eastmost (or westmost) point depending on the
    # prevalent direction of the strike
    DELTA = 0.1
    strike_to_east = (avg_strike > 0) & (avg_strike <= 180)
    if strike_to_east or abs(avg_strike) < DELTA:
        idx = np.argmin(px)
    else:
        idx = np.argmax(px)

    # Find for each 'line' the endpoint closest to the origin
    eps = []
    for i in range(0, len(px), 2):
        eps.append(min([px[i], px[i+1]]))

    # Find the indexes needed to sort the segments according to the prevalent
    # direction of the strike
    sort_idxs = np.argsort(eps)
    if not (strike_to_east or abs(avg_strike) < DELTA):
        sort_idxs = np.flipud(sort_idxs)

    # Set the origin to be used later for the calculation of the
    # coordinate shift
    x = np.array([px[idx]])
    y = np.array([py[idx]])
    olon, olat = proj(x, y, reverse=True)

    return olon[0], olat[0], sort_idxs


def get_coordinate_shift(origins: list, olon: float, olat: float,
                         overall_strike: float) -> np.ndarray:
    """
    Computes the coordinate shift for each line in the multiline. This is
    used to compute coordinates in the GC2 system

    :returns:
        A :class:`np.ndarray`instance with cardinality equal to the number of
        sections (i.e. the length of the lines list in input)
    """
    # Distances and azimuths between the origin of the multiline and the
    # first endpoint
    distances = geodetic_distance(olon, olat, origins[:, 0], origins[:, 1])
    azimuths = azimuth(olon, olat, origins[:, 0], origins[:, 1])

    # Calculate the shift along the average strike direction
    return np.float32(np.cos(np.radians(overall_strike - azimuths)) * distances)


@compile('f4[:],f4[:,:,:]')
def _get_tu(shift, tuw):
    # `shift` has shape S and `tuw` shape (3, S, N)
    S, N = tuw.shape[1:]
    tN, uN = np.zeros(N), np.zeros(N)
    W = tuw[2].sum(axis=0)  # shape N
    for s in range(S):
        t, u, w = tuw[:, s]  # shape N
        tN += t * w
        uN += (u + shift[s]) * w
    return tN / W, uN / W
