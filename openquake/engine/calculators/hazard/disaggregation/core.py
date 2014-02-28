# -*- coding: utf-8 -*-
# Copyright (c) 2010-2013, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

"""
Disaggregation calculator core functionality
"""

import sys
import numpy

import openquake.hazardlib
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.geo.geodetic import npoints_between
from openquake.hazardlib.geo.utils import get_longitudinal_extent
from openquake.hazardlib.geo.utils import get_spherical_bounding_box
from openquake.hazardlib.site import SiteCollection

from openquake.engine import logs
from openquake.engine.calculators.hazard.classical.core import \
    ClassicalHazardCalculator
from openquake.engine.db import models
from openquake.engine.input import logictree
from openquake.engine.utils import tasks, general
from openquake.engine.performance import EnginePerformanceMonitor, LightMonitor


def _collect_bins_data(mon, trt_num, sources, site, imt, iml, gsims,
                       truncation_level, n_epsilons,
                       source_site_filter, rupture_site_filter):
    """
    Extract values of magnitude, distance, closest point, tectonic region
    types and PoE distribution.

    This method processes the source model (generates ruptures) and collects
    all needed parameters to arrays. It also defines tectonic region type
    bins sequence.
    """
    mags = []
    dists = []
    lons = []
    lats = []
    tect_reg_types = []
    probs_no_exceed = []
    sitecol = SiteCollection([site])
    sitemesh = sitecol.mesh
    mon0 = LightMonitor(mon.operation, mon.job_id, mon.task)
    mon1 = mon0.copy('calc distances')
    mon2 = mon0.copy('makectxt')
    mon3 = mon0.copy('disaggregate_poe')
    sources_sites = ((source, sitecol) for source in sources)
    # here we ignore filtered site collection because either it is the same
    # as the original one (with one site), or the source/rupture is filtered
    # out and doesn't show up in the filter's output
    for src_idx, (source, s_sites) in \
            enumerate(source_site_filter(sources_sites)):
        try:
            gsim = gsims[source.tectonic_region_type]
            tect_reg = trt_num[source.tectonic_region_type]

            ruptures_sites = ((rupture, s_sites)
                              for rupture in source.iter_ruptures())
            for rupture, r_sites in rupture_site_filter(ruptures_sites):
                # extract rupture parameters of interest
                mags.append(rupture.mag)
                with mon1:
                    [jb_dist] = rupture.surface.get_joyner_boore_distance(
                        sitemesh)
                    dists.append(jb_dist)
                    [closest_point] = rupture.surface.get_closest_points(
                        sitemesh)
                lons.append(closest_point.longitude)
                lats.append(closest_point.latitude)
                tect_reg_types.append(tect_reg)

                # compute conditional probability of exceeding iml given
                # the current rupture, and different epsilon level, that is
                # ``P(IMT >= iml | rup, epsilon_bin)`` for each of epsilon bins
                with mon2:
                    sctx, rctx, dctx = gsim.make_contexts(sitecol, rupture)
                with mon3:
                    [poes_given_rup_eps] = gsim.disaggregate_poe(
                        sctx, rctx, dctx, imt, iml, truncation_level,
                        n_epsilons)

                # collect probability of a rupture causing no exceedances
                probs_no_exceed.append(
                    rupture.get_probability_no_exceedance(poes_given_rup_eps)
                )
        except Exception, err:
            etype, err, tb = sys.exc_info()
            msg = 'An error occurred with source id=%s. Error: %s'
            msg %= (source.source_id, err.message)
            raise etype, msg, tb

    mon1.flush()
    mon2.flush()
    mon3.flush()
    return mags, dists, lons, lats, tect_reg_types, trt_num, probs_no_exceed


def _define_bins(bins_data, mag_bin_width, dist_bin_width,
                 coord_bin_width, truncation_level, n_epsilons):
    """
    Define bin edges for disaggregation histograms.

    Given bins data as provided by :func:`_collect_bins_data`, this function
    finds edges of histograms, taking into account maximum and minimum values
    of magnitude, distance and coordinates as well as requested sizes/numbers
    of bins.
    """
    mags, dists, lons, lats, tect_reg_types, _ = bins_data

    mag_bins = mag_bin_width * numpy.arange(
        int(numpy.floor(mags.min() / mag_bin_width)),
        int(numpy.ceil(mags.max() / mag_bin_width) + 1)
    )

    dist_bins = dist_bin_width * numpy.arange(
        int(numpy.floor(dists.min() / dist_bin_width)),
        int(numpy.ceil(dists.max() / dist_bin_width) + 1)
    )

    west, east, north, south = get_spherical_bounding_box(lons, lats)
    west = numpy.floor(west / coord_bin_width) * coord_bin_width
    east = numpy.ceil(east / coord_bin_width) * coord_bin_width
    lon_extent = get_longitudinal_extent(west, east)
    lon_bins, _, _ = npoints_between(
        west, 0, 0, east, 0, 0,
        numpy.round(lon_extent / coord_bin_width) + 1
    )

    lat_bins = coord_bin_width * numpy.arange(
        int(numpy.floor(south / coord_bin_width)),
        int(numpy.ceil(north / coord_bin_width) + 1)
    )

    eps_bins = numpy.linspace(-truncation_level, truncation_level,
                              n_epsilons + 1)

    return mag_bins, dist_bins, lon_bins, lat_bins, eps_bins


def _arrange_data_in_bins(bins_data, bin_edges):
    """
    Given bins data, as it comes from :func:`_collect_bins_data`, and bin edges
    from :func:`_define_bins`, create a normalized 6d disaggregation matrix.
    """
    (mags, dists, lons, lats, tect_reg_types, probs_no_exceed) = bins_data
    mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins = bin_edges
    shape = (len(mag_bins) - 1, len(dist_bins) - 1, len(lon_bins) - 1,
             len(lat_bins) - 1, len(eps_bins) - 1, len(trt_bins))
    diss_matrix = numpy.zeros(shape)

    for i_mag in xrange(len(mag_bins) - 1):
        mag_idx = mags <= mag_bins[i_mag + 1]
        if i_mag != 0:
            mag_idx &= mags > mag_bins[i_mag]

        for i_dist in xrange(len(dist_bins) - 1):
            dist_idx = dists <= dist_bins[i_dist + 1]
            if i_dist != 0:
                dist_idx &= dists > dist_bins[i_dist]

            for i_lon in xrange(len(lon_bins) - 1):
                extents = get_longitudinal_extent(lons, lon_bins[i_lon + 1])
                lon_idx = extents >= 0
                if i_lon != 0:
                    extents = get_longitudinal_extent(lon_bins[i_lon], lons)
                    lon_idx &= extents > 0

                for i_lat in xrange(len(lat_bins) - 1):
                    lat_idx = lats <= lat_bins[i_lat + 1]
                    if i_lat != 0:
                        lat_idx &= lats > lat_bins[i_lat]

                    for i_eps in xrange(len(eps_bins) - 1):

                        for i_trt in xrange(len(trt_bins)):
                            trt_idx = tect_reg_types == i_trt

                            diss_idx = (i_mag, i_dist, i_lon,
                                        i_lat, i_eps, i_trt)

                            prob_idx = (mag_idx & dist_idx & lon_idx
                                        & lat_idx & trt_idx)

                            poe = numpy.prod(
                                probs_no_exceed[prob_idx, i_eps]
                                )
                            poe = 1 - poe

                            diss_matrix[diss_idx] = poe

    return diss_matrix


@tasks.oqtask
def compute_disagg(job_id, sites, sources, rlz, ltp, trt_num):
    """
    Calculate disaggregation histograms and saving the results to the database.

    Here is the basic calculation workflow:

    1. Get all sources
    2. Get IMTs
    3. Get the hazard curve for each point, IMT, and realization
    4. For each `poes_disagg`, interpolate the IML for each curve.
    5. Get GSIMs, TOM (Temporal Occurence Model), and truncation level.
    6. Get histogram bin edges.
    7. Prepare calculation args.
    8. Call the hazardlib calculator
       (see :func:`openquake.hazardlib.calc.disagg.disaggregation`
       for more info).

    :param int job_id:
        ID of the currently running :class:`openquake.engine.db.models.OqJob`
    :param list sites:
        `list` of :class:`openquake.hazardlib.site.Site` objects, which
        indicate the locations (and associated soil parameters) for which we
        need to compute disaggregation histograms.
    :param list sources:
        `list` of hazardlib source objects
    :param rlz:
        instance of :class:`openquake.engine.db.models.LtRealization` for which
        we want to compute disaggregation histograms. This realization will
        determine which hazard curve results to use as a basis for the
        calculation.
    :param ltp:
        a :class:`openquake.engine.input.LogicTreeProcessor` instance
    """
    mon = EnginePerformanceMonitor('disagg', job_id, compute_disagg)

    job = models.OqJob.objects.get(id=job_id)
    hc = job.hazard_calculation
    gsims = ltp.parse_gmpe_logictree_path(rlz.gsim_lt_path)
    f = openquake.hazardlib.calc.filters
    src_site_filter = f.source_site_distance_filter(hc.maximum_distance)
    rup_site_filter = f.rupture_site_distance_filter(hc.maximum_distance)

    [site] = sites

    result = {}
    for imt, imls in hc.intensity_measure_types_and_levels.iteritems():
        im_type, sa_period, sa_damping = imt = from_string(imt)

        imls = numpy.array(imls[::-1])

        # get curve for this point/IMT/realization
        [curve] = models.HazardCurveData.objects.filter(
            location=site.location.wkt2d,
            hazard_curve__lt_realization=rlz,
            hazard_curve__imt=im_type,
            hazard_curve__sa_period=sa_period,
            hazard_curve__sa_damping=sa_damping,
        )

        # If the hazard curve is all zeros, don't even do the
        # disagg calculation.
        if all(x == 0.0 for x in curve.poes):
            logs.LOG.debug(
                '* hazard curve contained all 0 probability values; '
                'skipping')
            continue

        for poe in hc.poes_disagg:
            iml = numpy.interp(poe, curve.poes[::-1], imls)
            with EnginePerformanceMonitor(
                    'collecting bins', job_id, compute_disagg):
                result[rlz.id, site, poe, iml, im_type, sa_period, sa_damping
                       ] = _collect_bins_data(
                    mon, trt_num, sources, site, imt, iml, gsims,
                    hc.truncation_level, hc.num_epsilon_bins,
                    src_site_filter, rup_site_filter),
    return result


_DISAGG_RES_NAME_FMT = 'disagg(%(poe)s)-rlz-%(rlz)s-%(imt)s-%(wkt)s'


def _save_disagg_matrix(job, site, bin_edges, diss_matrix, rlz,
                        investigation_time, imt, iml, poe, sa_period,
                        sa_damping):
    """
    Save a computed disaggregation matrix to `hzrdr.disagg_result` (see
    :class:`~openquake.engine.db.models.DisaggResult`).

    :param job:
        :class:`openquake.engine.db.models.OqJob` representing the current job.
    :param site:
        :class:`openquake.hazardlib.site.Site`, containing the location
        geometry for these results.
    :param bin_edges, diss_matrix
        The outputs of :func:
        `openquake.hazardlib.calc.disagg.disaggregation`.
    :param rlz:
        :class:`openquake.engine.db.models.LtRealization` to which these
        results belong.
    :param float investigation_time:
        Investigation time (years) for the calculation.
    :param imt:
        Intensity measure type (PGA, SA, etc.)
    :param float iml:
        Intensity measure level interpolated (using ``poe``) from the hazard
        curve at the ``site``.
    :param float poe:
        Disaggregation probability of exceedance value for this result.
    :param float sa_period:
        Spectral Acceleration period; only relevant when ``imt`` is 'SA'.
    :param float sa_damping:
        Spectral Acceleration damping; only relevant when ``imt`` is 'SA'.
    """
    # Silencing 'Too many arguments', 'Too many local variables'
    # pylint: disable=R0913,R0914
    disp_name = _DISAGG_RES_NAME_FMT
    disp_imt = imt
    if disp_imt == 'SA':
        disp_imt = 'SA(%s)' % sa_period

    disp_name_args = dict(poe=poe, rlz=rlz.id, imt=disp_imt,
                          wkt=site.location.wkt2d)
    disp_name %= disp_name_args

    output = models.Output.objects.create_output(
        job, disp_name, 'disagg_matrix'
    )

    mag, dist, lon, lat, eps, trts = bin_edges
    models.DisaggResult.objects.create(
        output=output,
        lt_realization=rlz,
        investigation_time=investigation_time,
        imt=imt,
        sa_period=sa_period,
        sa_damping=sa_damping,
        iml=iml,
        poe=poe,
        mag_bin_edges=mag,
        dist_bin_edges=dist,
        lon_bin_edges=lon,
        lat_bin_edges=lat,
        eps_bin_edges=eps,
        trts=trts,
        location=site.location.wkt2d,
        matrix=diss_matrix,
    )


@tasks.oqtask
def arrange_and_save_disagg_matrix(
        job_id, trt_bins, bins, rlz_id, site, poe, iml,
        im_type, sa_period, sa_damping):
    """
    """
    hc = models.OqJob.objects.get(id=job_id).hazard_calculation
    rlz = models.LtRealization.objects.get(id=rlz_id)
    mags = numpy.array(bins[0], float)
    dists = numpy.array(bins[1], float)
    lons = numpy.array(bins[2], float)
    lats = numpy.array(bins[3], float)
    tect_reg_types = numpy.array(bins[4], int)
    probs_no_exceed = numpy.array(bins[5], float)
    bdata = (mags, dists, lons, lats, tect_reg_types, probs_no_exceed)
    with EnginePerformanceMonitor(job_id, 'define bins',
                                  arrange_and_save_disagg_matrix):
        bin_edges = _define_bins(
            bdata,
            hc.mag_bin_width,
            hc.distance_bin_width,
            hc.coordinate_bin_width,
            hc.truncation_level,
            hc.num_epsilon_bins) + (trt_bins, )
    with EnginePerformanceMonitor(job_id, 'arrange data',
                                  arrange_and_save_disagg_matrix):
        diss_matrix = _arrange_data_in_bins(bdata, bin_edges)

    with EnginePerformanceMonitor(job_id, 'saving disaggregation',
                                  arrange_and_save_disagg_matrix):
        _save_disagg_matrix(
            job_id, site, bin_edges, diss_matrix, rlz,
            hc.investigation_time, im_type, iml, poe, sa_period,
            sa_damping)


class DisaggHazardCalculator(ClassicalHazardCalculator):
    """
    A calculator which performs disaggregation calculations in a distributed /
    parallelized fashion.

    See :func:`openquake.hazardlib.calc.disagg.disaggregation` for more
    details about the nature of this type of calculation.
    """

    def disagg_task_arg_gen(self):
        """
        Generate task args for the second phase of disaggregation calculations.
        This phase is concerned with computing the disaggregation histograms.
        """
        trt_num = dict((trt, i) for i, trt in enumerate(
                       self.tectonic_region_types))
        self.trt_bins = [trt for (num, trt) in sorted(
                         (num, trt) for (trt, num) in trt_num.items())]

        realizations = models.LtRealization.objects.filter(
            hazard_calculation=self.hc)

        ltp = logictree.LogicTreeProcessor.from_hc(self.hc)
        # then distribute tasks for disaggregation histogram computation
        for rlz in realizations:
            path = tuple(rlz.sm_lt_path)
            sources = general.WeightedSequence.merge(
                self.source_blocks_per_ltpath[path])
            for site in self.hc.site_collection:
                yield self.job.id, [site], sources, rlz, ltp, trt_num

    def post_execute(self):
        """
        Start the disaggregation phase after hazard curve finalization.
        """
        super(DisaggHazardCalculator, self).post_execute()

        self.result = {}  # dictionary {key: bins} where key is the tuple
        # rlz_id, site, poe, iml, im_type, sa_period, sa_damping
        self.parallelize(
            compute_disagg, self.disagg_task_arg_gen(), self.collect_result)
        arglist = [(self.job.id, self.trt_bins, bins) + key
                   for key, bins in self.result.iteritems()]
        self.parallelize(
            arrange_and_save_disagg_matrix, arglist, self.log_percent)

    def collect_result(self, result):
        """
        Collect the results coming from compute_disagg in self.results,
        a dictionary with key (rlz_id, site, poe, iml, im_type, sa_period,
        sa_damping) and values (mag_bins, dist_bins, lon_bins, lat_bins,
        eps_bins, trt_bins).

        """
        for rlz_id, site, poe, iml, im_type, sa_period, sa_damping in result:
            # mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins
            try:
                bins = self.result[
                    rlz_id, site, poe, iml, im_type, sa_period, sa_damping]
            except KeyError:
                bins = self.result[
                    rlz_id, site, poe, iml, im_type, sa_period, sa_damping] = (
                    [], [], [], [], [], [])
            bins_data = result[
                rlz_id, site, poe, iml, im_type, sa_period, sa_damping]
            for acc, ls in zip(bins, bins_data):
                acc.extend(ls)
        self.log_percent()
