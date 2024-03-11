# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation
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
Module exports :class:'SaffriEtAl2012CenteralIran'
                class:'SaffriEtAl2012Zagros'
"""
import numpy as np
from scipy.constants import g


from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib.gsim import utils
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


def _compute_distance_scaling(C, rhypo, rjb, mag):
    """
    Returns the distance scaling term 
    (second and third term of equation 3)
    """
    r = rjb if mag.any() > 6.5 else rhypo
    e = 0.5
    rscale1 = r + C["d"] * (10.0 ** (e * mag))
    return -np.log10(rscale1) - (C["b"] * r)


def _compute_magnitude_scaling(C, mag):
    """
    Returns the magnitude scaling term
    (first term of equation 3)
    """
    return C["a"] * mag

def _compute_site_amplification(C, vs30):
    """
    Computes the fourth term of the equation 3 
    """
    delt1, delt2, delt3 = _get_site_type_dummy_variables(vs30)
    return (C["c1"] * delt1 + C["c2"] * delt2 + C["c3"] * delt3)

def _get_site_type_dummy_variables(vs30):
    """
    Returns site type dummy variables, four site types are considered
    based on the Iranian Code of Practice for the Seismic Resistant
    Design of Buildings:
    class I: Vs30 > 750 m/s
    class II: 375 <= Vs30 <= 750 m/s
    class III: 175 <= Vs30 < 375 m/s
    class IV: Vs30 < 175 m/s
    """
    delt1 = np.zeros(len(vs30))
    delt2 = np.zeros(len(vs30))
    delt3 = np.zeros(len(vs30))

    # class I: Vs30 > 750 m/s
    idx = (vs30 > 750.0)
    delt1[idx] = 1.0
    # class II: 375 <= Vs30 <= 750 m/s.
    idx = (vs30 >= 375.0) & (vs30 <= 750.0)
    delt2[idx] = 1.0
    # class III: 175 <= Vs30 < 375 m/s
    idx = (vs30 >= 175.0) & (vs30 < 375.0)
    delt3[idx] = 1.0
    return delt1, delt2, delt3

def _get_mechanism(C, ctx):
    """
    Computes the fifth term of the equation 3.
    """
    SS, NF, TF = utils.get_fault_type_dummy_variables(ctx)
    fTF = 0
    return C['f'] * SS + fTF * TF

def _zone_effect(C, ctx):
    """
    Returns the sixth term of equation 3 for Zagros zone.
    """
    z = np.ones(len(ctx.vs30))
    return C["g"] * z

class SaffriEtAl2012CenteralIran(GMPE):
    """
    Implements the PGA GMPE of Saffari et al. (2012)
    Saffari, H., Kuwata, Y., Takada, S., & Mahdavian, A. (2012). 
    Updated PGA, PGV, and spectral acceleration attenuation relations for Iran. 
    Earthquake spectra, 28(1), 257-276.
    This class is the GMPE for the central Iran and Alborz zones.
    """
    #: The GMPE is derived from shallow earthquakes in Iran
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are PGA, PGV, and SA(5%)
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is the average horizontal
    #: component
    #: :attr:`openquake.hazardlib.const.IMC.GEOMETRIC_MEAN`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation types are inter-event, intra-event
    #: and total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Required site parameter is only Vs30
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters are magnitude and rake
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake'}

    #: Required distance measures are rjb and distance to hypocenter
    REQUIRES_DISTANCES = {'rhypo', 'rjb'}

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            imean = (_compute_magnitude_scaling(C, ctx.mag) +
                     _compute_distance_scaling(C, ctx.rhypo, ctx.rjb, ctx.mag) +
                     _compute_site_amplification(C, ctx.vs30) + 
                     _get_mechanism(C, ctx))
            
            # Original GMPE returns log10 PGA and SA in gal
            # and PGV in cm/s
            # Converts to natural logarithm of g
            # but only for PGA and SA (not PGV):
            if imt.string.startswith(('PGA', 'SA')):
                mean[m] = np.log((10.0 ** (imean - 2.0)) / g)
            else:
                # PGV
                mean[m] = np.log(10.0 ** imean)

            # Convert from common logarithm to natural logarithm
            sig[m] = np.log(10.0 ** C['SigmaTot'])
            tau[m] = np.log(10.0 ** C['SigmaInter'])
            phi[m] = np.log(10.0 ** C['SigmaIntra'])

    COEFFS = CoeffsTable(sa_damping=5, table="""
    IMT	    a	    b	    d	     c1	     c2	     c3	     f	     g	    SigmaIntra	SigmaInter	SigmaTot
    PGA 	0.38	0.0045	0.005	 1.30	 1.35	 1.53	 0.03	 0.02	0.23	    0.16	    0.28
    0.05	0.35	0.0052	0.005	 1.65	 1.67	 1.83	 0.05	 0.01	0.24	    0.18	    0.30
    0.10	0.32	0.0055	0.005	 2.07	 2.10	 2.23	 0.05	-0.01	0.26	    0.19	    0.32
    0.15	0.35	0.0049	0.005	 1.87	 1.95	 2.09	 0.02	 0.00	0.25	    0.19	    0.31
    0.20	0.39	0.0043	0.005	 1.61	 1.71	 1.87	 0.00	 0.02	0.24	    0.19	    0.31
    0.25	0.42	0.0039	0.005	 1.35	 1.47	 1.64	-0.01	 0.04	0.24	    0.18	    0.30
    0.30	0.45	0.0036	0.005	 1.12	 1.24	 1.43	-0.01	 0.05	0.25	    0.17	    0.30
    0.40	0.49	0.0031	0.005	 0.72	 0.83	 1.04	-0.02	 0.08	0.26	    0.17	    0.31
    0.50	0.53	0.0028	0.005	 0.37	 0.47	 0.71	-0.02	 0.10	0.25	    0.18	    0.31
    0.60	0.57	0.0025	0.005	 0.08	 0.17	 0.41	-0.02	 0.11	0.25	    0.18	    0.31
    0.70	0.59	0.0023	0.005	-0.18	-0.10	 0.15	-0.02	 0.12	0.25	    0.17	    0.30
    0.80	0.62	0.0021	0.005	-0.41	-0.34	-0.09	-0.02	 0.13	0.25	    0.15	    0.29
    0.90	0.64	0.0019	0.005	-0.62	-0.56	-0.30	-0.02	 0.14	0.25	    0.16	    0.30
    1.00	0.66	0.0018	0.005	-0.81	-0.76	-0.50	-0.02	 0.14	0.26	    0.15	    0.30
    1.50	0.75	0.0013	0.005	-1.60	-1.56	-1.31	-0.02	 0.15	0.27	    0.16	    0.31
    2.00	0.81	0.0009	0.005	-2.21	-2.17	-1.93	-0.01	 0.15	0.26	    0.16	    0.31
    2.50	0.87	0.0008	0.005	-2.70	-2.65	-2.44	 0.00	 0.14	0.26	    0.17	    0.31
    3.00	0.92	0.0008	0.009	-3.13	-3.07	-2.88	 0.01	 0.14	0.26	    0.17	    0.31
    3.50	0.97	0.0009	0.015	-3.50	-3.43	-3.26	 0.02	 0.14	0.25	    0.18	    0.31
    4.00	1.02	0.0011	0.019	-3.83	-3.75	-3.59	 0.04	 0.14	0.25	    0.17	    0.30
    5.00	1.10	0.0016	0.024	-4.39	-4.32	-4.18	 0.08	 0.15	0.25	    0.18	    0.31
    PGV	    0.61	0.0027	0.005	-1.37	-1.30	-1.08	 0.03	 0.06	0.23	    0.14	    0.27

    """)


class SaffriEtAl2012Zagros(SaffriEtAl2012CenteralIran):
    """
    This class is the GMPE for the Zagros zone.
    """
    
    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            imean = (_compute_magnitude_scaling(C, ctx.mag) +
                     _compute_distance_scaling(C, ctx.rhypo, ctx.rjb, ctx.mag) +
                     _compute_site_amplification(C, ctx.vs30) + 
                     _get_mechanism(C, ctx)+
                     _zone_effect(C, ctx))
            
            # Original GMPE returns log10 PGA and SA in gal
            # and PGV in cm/s
            # Converts to natural logarithm of g
            # but only for PGA and SA (not PGV):
            if imt.string.startswith(('PGA', 'SA')):
                mean[m] = np.log((10.0 ** (imean - 2.0)) / g)
            else:
                # PGV
                mean[m] = np.log(10.0 ** imean)

            # Convert from common logarithm to natural logarithm
            sig[m] = np.log(10.0 ** C['SigmaTot'])
            tau[m] = np.log(10.0 ** C['SigmaInter'])
            phi[m] = np.log(10.0 ** C['SigmaIntra'])

