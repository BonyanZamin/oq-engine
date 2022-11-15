# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2020, GEM Foundation
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
Module :mod:`openquake.hazardlib.mgmpe.modifiable_gmpe` implements
:class:`~openquake.hazardlib.mgmpe.ModifiableGMPE`
"""
import copy
import warnings
import numpy as np
from openquake.hazardlib.gsim.base import GMPE, registry, CoeffsTable
from openquake.hazardlib.contexts import get_mean_stds
from openquake.hazardlib.const import (
    StdDev, IMT_DEPENDENT_KEYS, OK_COMPONENTS, apply_conversion)
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.gsim.mgmpe.nrcan15_site_term import (
    NRCan15SiteTerm, BA08_AB06)

from openquake.hazardlib.gsim.nga_east import (
    TAU_EXECUTION, get_phi_ss, TAU_SETUP, PHI_SETUP, get_tau_at_quantile,
    get_phi_ss_at_quantile)
from openquake.hazardlib.gsim.usgs_ceus_2019 import get_stewart_2019_phis2s


# ################ BEGIN FUNCTIONS MODIFYING mean_stds ################## #

def sigma_model_alatik2015(ctx, imt, mean_stds,
                           ergodic, tau_model, phi_ss_coetab, tau_coetab):
    """
    This function uses the sigma model of Al Atik (2015) as the standard
    deviation of a specified GMPE
    """
    phi = get_phi_ss(imt, ctx.mag, phi_ss_coetab)
    if ergodic:
        phi_s2s = get_stewart_2019_phis2s(imt, ctx.vs30)
        phi = np.sqrt(phi ** 2. + phi_s2s ** 2.)
    tau = TAU_EXECUTION[tau_model](imt, ctx.mag, tau_coetab)
    mean_stds[1] = np.sqrt(tau ** 2. + phi ** 2.)
    mean_stds[2] = tau
    mean_stds[3] = phi


def nrcan15_site_term(ctx, imt, mean_stds, kind):
    """
    This function adds a site term to GMMs missing it
    """
    C = NRCan15SiteTerm.COEFFS_BA08[imt]
    C2 = NRCan15SiteTerm.COEFFS_AB06r[imt]
    exp_mean = np.exp(mean_stds[0])
    fa = BA08_AB06(kind, C, C2, ctx.vs30, imt, exp_mean)
    mean_stds[0] = np.log(exp_mean * fa)


def horiz_comp_to_geom_mean(ctx, imt, mean_stds, conv=None):
    """
    This function converts ground-motion obtained for a given description of
    horizontal component into ground-motion values for geometric_mean.
    The conversion equations used are from:
        - Beyer and Bommer (2006): for arithmetic mean, GMRot and random
        - Boore and Kishida (2017): for RotD50
    """
    if conv is None:
        return
    conv_median, conv_sigma, rstd = conv
    mean_stds[0] = np.log(np.exp(mean_stds[0]) / conv_median)
    mean_stds[1] = ((mean_stds[1]**2 - conv_sigma**2) / rstd**2)**0.5


def add_between_within_stds(ctx, imt, mean_stds, with_betw_ratio):
    """
    This adds the between and within standard deviations to a model which has
    only the total standatd deviation. This function requires a ratio between
    the within-event standard deviation and the between-event one.

    :param with_betw_ratio:
        The ratio between the within and between-event standard deviations
    """
    total = mean_stds[1]
    between = (total**2 / (1 + with_betw_ratio**2))**0.5
    within = with_betw_ratio * between
    mean_stds[2] = between
    mean_stds[3] = within


def apply_swiss_amplification(ctx, imt, mean_stds):
    """
    Adds amplfactor to mean
    """
    mean_stds[0] += ctx.amplfactor


def set_between_epsilon(ctx, imt, mean_stds, epsilon_tau):
    """
    :param epsilon_tau:
        the epsilon value used to constrain the between event variability
    """
    # index for the between event standard deviation
    mean_stds[0] += epsilon_tau * mean_stds[2]

    # set between event variability to 0
    mean_stds[2] = 0

    # set total variability equal to the within-event one
    mean_stds[1] = mean_stds[3]


def set_scale_median_scalar(ctx, imt, mean_stds, scaling_factor):
    """
    :param scaling_factor:
        Simple scaling factor (in linear space) to increase/decrease median
        ground motion, which applies to all IMTs
    """
    mean_stds[0] += np.log(scaling_factor)


# self is an instance of ModifiableGMPE
def set_scale_median_vector(ctx, imt, mean_stds, scaling_factor):
    """
    :param scaling_factor:
        IMT-dependent median scaling factors (in linear space) as
        a CoeffsTable
    """
    mean_stds[0] += np.log(scaling_factor[imt]["scaling_factor"])


# self is an instance of ModifiableGMPE
def set_scale_total_sigma_scalar(ctx, imt, mean_stds, scaling_factor):
    """
    Scale the total standard deviations by a constant scalar factor
    :param scaling_factor:
        Factor to scale the standard deviations
    """
    mean_stds[1] *= scaling_factor


def set_scale_total_sigma_vector(ctx, imt, mean_stds, scaling_factor):
    """
    Scale the total standard deviations by a IMT-dependent scalar factor
    :param scaling_factor:
        IMT-dependent total standard deviation scaling factors as a
        CoeffsTable
    """
    mean_stds[1] *= scaling_factor[imt]["scaling_factor"]


def set_fixed_total_sigma(ctx, imt, mean_stds, total_sigma):
    """
    Sets the total standard deviations to a fixed value per IMT
    :param total_sigma:
        IMT-dependent total standard deviation as a CoeffsTable
    """
    mean_stds[1] = total_sigma[imt]["total_sigma"]


def add_delta_std_to_total_std(ctx, imt, mean_stds, delta):
    """
    :param delta:
        A delta std e.g. a phi S2S to be removed from total
    """
    mean_stds[1] = (mean_stds[1]**2 + np.sign(delta) * delta**2)**0.5


def set_total_std_as_tau_plus_delta(ctx, imt, mean_stds, delta):
    """
    :param delta:
        A delta std e.g. a phi SS to be combined with between std, tau.
    """
    mean_stds[1] = (mean_stds[2]**2 + np.sign(delta) * delta**2)**0.5


# ################ END OF FUNCTIONS MODIFYING mean_stds ################## #


def _dict_to_coeffs_table(input_dict, name):
    """
    Transform a dictionary of parameters organised by IMT into a
    coefficient table
    """
    coeff_dict = {}
    for key in input_dict:
        coeff_dict[from_string(key)] = {name: input_dict[key]}
    return {name: CoeffsTable.fromdict(coeff_dict)}


class ModifiableGMPE(GMPE):
    """
    This is a fully configurable GMPE

    :param string gmpe_name:
        The name of a GMPE class used for the calculation.
    :param params:
        A dictionary where the key defines the required modification and the
        value is a list with the required parameters.
    """
    REQUIRES_SITES_PARAMETERS = set()
    REQUIRES_DISTANCES = set()
    REQUIRES_RUPTURE_PARAMETERS = set()
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ''
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {StdDev.TOTAL}
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''
    DEFINED_FOR_REFERENCE_VELOCITY = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mags = ()  # used in GMPETables

        # Create the original GMPE
        [(gmpe_name, kw)] = kwargs.pop('gmpe').items()
        self.params = kwargs  # non-gmpe parameters
        self.gmpe = registry[gmpe_name](**kw)
        self.set_parameters()

        if ('set_between_epsilon' in self.params or
            'set_total_std_as_tau_plus_delta' in self.params) and (
                StdDev.INTER_EVENT not in
                self.gmpe.DEFINED_FOR_STANDARD_DEVIATION_TYPES):
            raise ValueError('%s does not have between event std' % self.gmpe)

        if 'apply_swiss_amplification' in self.params:
            self.gmpe.REQUIRES_SITES_PARAMETERS = frozenset(['amplfactor'])

        if 'add_between_within_stds' in self.params:
            setattr(self, 'DEFINED_FOR_STANDARD_DEVIATION_TYPES',
                    {StdDev.TOTAL, StdDev.INTRA_EVENT, StdDev.INTER_EVENT})

        # This is required by the `sigma_model_alatik2015` function
        key = 'sigma_model_alatik2015'
        if key in self.params:

            # Phi S2SS and ergodic param
            # self.params[key]['phi_s2ss'] = None
            self.params[key]['ergodic'] = self.params[key].get("ergodic", True)

            # Tau
            tau_model = self.params[key].get("tau_model", "global")
            if "tau_model" not in self.params:
                self.params[key]['tau_model'] = tau_model
            tau_quantile = self.params[key].get("tau_quantile", None)
            self.params[key]['tau_coetab'] = get_tau_at_quantile(
                TAU_SETUP[tau_model]["MEAN"],
                TAU_SETUP[tau_model]["STD"],
                tau_quantile)

            # Phi SS
            phi_model = self.params[key].get("phi_model", "global")
            if "phi_model" in self.params:
                del self.params[key]["phi_model"]
            phi_ss_quantile = self.params[key].get("phi_ss_quantile", None)
            self.params[key]['phi_ss_coetab'] = get_phi_ss_at_quantile(
                PHI_SETUP[phi_model], phi_ss_quantile)

        # Set params
        for key in self.params:
            if key in IMT_DEPENDENT_KEYS:
                # If the modification is period-dependent
                for subkey in self.params[key]:
                    if isinstance(self.params[key][subkey], dict):
                        self.params[key] = _dict_to_coeffs_table(
                            self.params[key][subkey], subkey)

        # Apply conversion
        self.horcomp = self.gmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT
        comp = self.horcomp._name_
        self.conv = {}  # IMT -> (conv_median, conv_sigma, rstd)
        self.convertible = comp in OK_COMPONENTS
        if comp == 'GEOMETRIC_MEAN' or self.convertible:
            pass  # all okay
        else:
            warnings.warn(f'Conversion not applicable for {comp}', UserWarning)

    # called by the ContextMaker
    def set_tables(self, mags, imts):
        """
        :param mags: a list of magnitudes as strings
        :param imts: a list of IMTs as strings

        Set the .mean_table and .sig_table attributes on the underlying gmpe
        """
        if hasattr(self.gmpe, 'set_tables'):
            self.gmpe.set_tables(mags, imts)
            self.mags = mags

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        if 'nrcan15_site_term' in self.params:
            ctx_copy = copy.copy(ctx)
            ctx_copy.vs30 = np.full_like(ctx.vs30, 760.)  # rock
        else:
            ctx_copy = ctx
        g = globals()
        # Compute the original mean and standard deviations, shape (4, M, N)
        mean_stds = get_mean_stds(self.gmpe, ctx_copy, imts, mags=self.mags)

        # Apply sequentially the modifications
        for methname, kw in self.params.items():
            for m, imt in enumerate(imts):
                if methname == 'horiz_comp_to_geom_mean' and self.convertible:
                    try:
                        conv = self.conv[imt]
                    except KeyError:
                        conv = self.conv[imt] = apply_conversion(
                            self.horcomp, imt)
                    g[methname](ctx, imt, mean_stds[:, m], conv, **kw)
                else:
                    g[methname](ctx, imt, mean_stds[:, m], **kw)

        mean[:] = mean_stds[0]
        sig[:] = mean_stds[1]
        tau[:] = mean_stds[2]
        phi[:] = mean_stds[3]
