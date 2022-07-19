# The Hazard Library
# Copyright (C) 2012-2022 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module :mod:`openquake.hazardlib.source.multi_fault`
defines :class:`MultiFaultSource`.
"""

import numpy as np
from typing import Union

from openquake.baselib.general import gen_slices
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source.rupture import (
    NonParametricProbabilisticRupture)
from openquake.hazardlib.source.non_parametric import (
    NonParametricSeismicSource as NP)
from openquake.hazardlib.geo.surface.kite_fault import (
    geom_to_kite, kite_to_geom)
from openquake.hazardlib.geo.surface.multi import MultiSurface
from openquake.hazardlib.geo.utils import angular_distance, KM_TO_DEGREES
from openquake.hazardlib.source.base import BaseSeismicSource

F32 = np.float32
BLOCKSIZE = 200
# the BLOCKSIZE has to be large to reduce the number of sources and
# therefore the redundant data transfer in the .sections attribute


class MultiFaultSource(BaseSeismicSource):
    """
    The multi-fault source is a source typology specifiically support the
    calculation of hazard using fault models with segments participating to
    multiple ruptures.

    :param source_id:
        A unique identifier for the source
    :param name:
        The name of the fault
    :param tectonic_region_type:
        A string that defines the TRT of the fault source
    :param rupture_idxs:
        A list of lists. Each element contains the IDs of the sections
        participating to a rupture. The cardinality of this list is N.
    :param occurrence_probs:
        A list of N probabilities. Each element specifies the probability
        of 0, 1 ... occurrences of a rupture in the investigation time
    :param magnitudes:
        An iterable with cardinality N containing the magnitudes of the
        ruptures
    :param rakes:
        An iterable with cardinality N containing the rake of each
        rupture
    """
    code = b'F'
    MODIFICATIONS = {}

    def __init__(self, source_id: str, name: str, tectonic_region_type: str,
                 rupture_idxs: list, occurrence_probs: Union[list, np.ndarray],
                 magnitudes: list, rakes: list, geoms: list):
        nrups = len(rupture_idxs)
        assert len(occurrence_probs) == len(magnitudes) == len(rakes) == nrups
        self.rupture_idxs = rupture_idxs
        self.probs_occur = occurrence_probs
        self.mags = magnitudes
        self.rakes = rakes
        self.geoms = geoms
        super().__init__(source_id, name, tectonic_region_type)

    def is_gridded(self):
        return True  # convertible to HDF5

    def todict(self):
        """
        :returns: dictionary of array, called when converting to HDF5
        """
        ridxs = []
        for rupture_idxs in self.rupture_idxs:
            ridxs.append(' '.join(rupture_idxs))
        # each pmf has the form [(prob0, 0), (prob1, 1), ...]
        return dict(mag=self.mags, rake=self.rakes,
                    probs_occur=self.probs_occur, rupture_idxs=ridxs)

    def set_geoms(self, sections):
        """
        :param sections: a list of N surfaces
        """
        assert sections
        # In this loop we check that all the IDs of the sections composing one
        # rupture have a object in the section list describing their geometry.
        for i in range(len(self.mags)):
            for idx in self.rupture_idxs[i]:
                sections[idx]
        self.geoms = [kite_to_geom(sec) for sec in sections]

    def iter_ruptures(self, **kwargs):
        """
        An iterator for the ruptures.
        """
        if not self.geoms:
            raise RuntimeError('You forgot to call .set_geoms!')
        # iter on the ruptures
        step = kwargs.get('step', 1)
        n = len(self.mags)
        s = [geom_to_kite(geom) for geom in self.geoms]
        for idx, sec in enumerate(s):
            sec.suid = idx
        for i in range(0, n, step**2):
            idxs = self.rupture_idxs[i]
            if len(idxs) == 1:
                sfc = s[idxs[0]]
            else:
                sfc = MultiSurface([s[idx] for idx in idxs])
            rake = self.rakes[i]
            hypo = s[idxs[0]].get_middle_point()
            data = [(p, o) for o, p in enumerate(self.probs_occur[i])]
            yield NonParametricProbabilisticRupture(
                self.mags[i], rake, self.tectonic_region_type, hypo, sfc,
                PMF(data))

    def __iter__(self):
        # splitting is tested in disaggregation/NZ in oq-risk-tests
        if len(self.mags) <= BLOCKSIZE:  # already split
            yield self
            return
        # split in blocks of BLOCKSIZE ruptures each
        for i, slc in enumerate(gen_slices(0, len(self.mags), BLOCKSIZE)):
            src = self.__class__(
                '%s:%d' % (self.source_id, i),
                self.name,
                self.tectonic_region_type,
                self.rupture_idxs[slc],
                self.probs_occur[slc],
                self.mags[slc],
                self.rakes[slc],
                self.geoms)
            src.num_ruptures = src.count_ruptures()
            yield src

    def count_ruptures(self):
        """
        :returns: the number of the ruptures in the source
        """
        return len(self.mags)

    def get_min_max_mag(self):
        return np.min(self.mags), np.max(self.mags)

    def get_one_rupture(self, ses_seed, rupture_mutex):
        raise NotImplementedError

    @property
    def data(self):  # compatibility with NonParametricSeismicSource
        for i, rup in enumerate(self.iter_ruptures()):
            yield rup, self.probs_occur[i]

    polygon = NP.polygon
    wkt = NP.wkt
    mesh_size = NP.mesh_size

    def get_bounding_box(self, maxdist):
        """
        Bounding box containing the surfaces, enlarged by the maximum distance
        """
        multi_surf = MultiSurface(list(map(geom_to_kite, self.geoms)))
        west, east, north, south = multi_surf.get_bounding_box()
        a1 = maxdist * KM_TO_DEGREES
        a2 = angular_distance(maxdist, north, south)
        return west - a2, south - a1, east + a2, north + a1
