"""
:module:`openquake.hazardlib.gsim.sinter` implements
:class:`SInterCan15Mid`, :class:`SInterCan15Upp`, :class:`SInterCan15Low`
"""

import numpy as np

from openquake.hazardlib import const
from openquake.hazardlib.gsim.base import gsim_aliases
from openquake.hazardlib.gsim.can15.western import get_sigma
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006SInter
from openquake.hazardlib.gsim.atkinson_macias_2009 import AtkinsonMacias2009
from openquake.hazardlib.gsim.abrahamson_2015 import AbrahamsonEtAl2015SInter
from openquake.hazardlib.gsim.ghofrani_atkinson_2014 import \
    GhofraniAtkinson2014


def _get_mean(self, sites, rup, dists, imt, stddev_types):
    """
    See :meth:`superclass method
    <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
    for spec of input and result values.
    """
    g = self.gsims

    # Computing adjusted mean and stds
    cff = self.COEFFS_SITE[imt]

    # Zhao et al. 2006 - Vs30 + Rrup
    mean_zh06, stds1 = ZhaoEtAl2006SInter.get_mean_and_stddevs(
        self, sites, rup, dists, imt, stddev_types)
    #
    # Atkinson and Macias (2009) - Rrup
    mean_am09, stds2 = g[0].get_mean_and_stddevs(
        sites, rup, dists, imt, stddev_types)
    #
    # Abrahamson et al. (2015) - Rrup + vs30 + backarc
    mean_ab15, stds3 = g[1].get_mean_and_stddevs(
        sites, rup, dists, imt, stddev_types)
    #
    # Ghofrani and Atkinson (2014) - Rrup + vs30
    mean_ga14, stds4 = g[2].get_mean_and_stddevs(
        sites, rup, dists, imt,  stddev_types)
    mean_adj = (np.log(np.exp(mean_zh06)*cff['mf'])*0.1 +
                mean_am09*0.5 + mean_ab15*0.2 +
                np.log(np.exp(mean_ga14)*cff['mf'])*0.2)
    return mean_adj


class SInterCan15Mid(ZhaoEtAl2006SInter):
    """
    Implements the Interface backbone model used for computing hazard for t
    the 2015 version of the Canada national hazard model developed by NRCan.
    """
    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTERFACE

    #: Required site parameters
    REQUIRES_SITES_PARAMETERS = {'vs30', 'backarc'}

    #: GMPE not tested against independent implementation so raise
    #: not verified warning
    non_verified = True

    #: Shear-wave velocity for reference soil conditions in [m s-1]
    DEFINED_FOR_REFERENCE_VELOCITY = 760.

    #: Supported standard deviations
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}

    REQUIRES_ATTRIBUTES = {'sgn'}

    gsims = [AtkinsonMacias2009(), AbrahamsonEtAl2015SInter(),
             GhofraniAtkinson2014()]  # underlying GSIMs

    def __init__(self, sgn=0, **kwargs):
        super().__init__(**kwargs)
        self.sgn = sgn

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        mean = _get_mean(self, sites, rup, dists, imt, stddev_types)
        if self.sgn:
            delta = np.minimum((0.15-0.0007*dists.rrup), 0.35)
            mean += self.sgn * delta
        stddevs = [np.ones(len(dists.rrup))*get_sigma(imt)]
        return mean, stddevs

    COEFFS_SITE = CoeffsTable(sa_damping=5, table="""\
    IMT        mf
    pgv     1.000
    pga     0.500
    0.040   0.440
    0.100   0.440
    0.200   0.600
    0.300   0.810
    0.400   1.000
    1.000   1.040
    2.000   1.510
    3.000   1.200
    5.000   1.100
    10.00   1.000
    """)


gsim_aliases['SInterCan15Low'] = '[SInterCan15Mid]\nsgn = -1'
gsim_aliases['SInterCan15Upp'] = '[SInterCan15Mid]\nsgn = +1'
