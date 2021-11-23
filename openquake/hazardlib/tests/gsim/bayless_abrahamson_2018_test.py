# The Hazard Library
# Copyright (C) 2020 GEM Foundation
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

from openquake.hazardlib.gsim.bayless_abrahamson_2018 import \
        BaylessAbrahamson2018
from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase


class BaylessAbrahamson2018TestCase(BaseGSIMTestCase):
    GSIM_CLASS = BaylessAbrahamson2018

    # Tables computed using the matlab script included in the supplement to
    # the BSSA paper

    def test_mean(self):
        self.check('BA18/BA18_MEAN.csv',
                   max_discrep_percentage=0.1)

"""
    def test_std_intra(self):
        self.check('BA18/BA18_STD_INTRA.csv',
                   max_discrep_percentage=0.1)

    def test_std_inter(self):
        self.check('BA18/BA18_STD_INTER.csv',
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check('BA18/BA18_STD_TOTAL.csv',
                   max_discrep_percentage=0.1)
"""
