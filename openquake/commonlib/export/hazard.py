# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2016 GEM Foundation
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

import re
import pickle
import logging
import operator
import collections

import numpy

from openquake.baselib.general import (
    groupby, humansize, get_array, group_array)
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.calc import disagg
from openquake.commonlib.export import export
from openquake.commonlib.writers import (
    scientificformat, floatformat, write_csv)
from openquake.commonlib import writers, hazard_writers, util, readinput
from openquake.risklib.riskinput import create
from openquake.calculators import calc, base, event_based

F32 = numpy.float32

GMF_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
GMF_WARNING = '''\
There are a lot of ground motion fields; the export will be slow.
Consider canceling the operation and accessing directly %s.'''


class SES(object):
    """
    Stochastic Event Set: A container for 1 or more ruptures associated with a
    specific investigation time span.
    """
    # the ordinal must be > 0: the reason is that it appears in the
    # exported XML file and the schema constraints the number to be
    # nonzero
    def __init__(self, ruptures, investigation_time, ordinal=1):
        self.ruptures = sorted(ruptures, key=operator.attrgetter('etag'))
        self.investigation_time = investigation_time
        self.ordinal = ordinal

    def __iter__(self):
        return iter(self.ruptures)


class SESCollection(object):
    """
    Stochastic Event Set Collection
    """
    def __init__(self, idx_ses_dict, investigation_time=None):
        self.idx_ses_dict = idx_ses_dict
        self.investigation_time = investigation_time

    def __iter__(self):
        for idx, sesruptures in sorted(self.idx_ses_dict.items()):
            yield SES(sesruptures, self.investigation_time, idx)


@export.add(('sescollection', 'xml'), ('sescollection', 'csv'))
def export_ses_xml(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    fmt = ekey[-1]
    oq = dstore['oqparam']
    mesh = dstore['sitemesh'].value
    ruptures = []
    for serial in dstore['sescollection']:
        sr = dstore['sescollection/' + serial]
        ruptures.extend(sr.export(mesh))
    ses_coll = SESCollection(
        groupby(ruptures, operator.attrgetter('ses_idx')),
        oq.investigation_time)
    dest = dstore.export_path('ses.' + fmt)
    globals()['_export_ses_' + fmt](dest, ses_coll)
    return [dest]


def _export_ses_xml(dest, ses_coll):
    writer = hazard_writers.SESXMLWriter(dest)
    writer.serialize(ses_coll)


def _export_ses_csv(dest, ses_coll):
    rows = []
    for ses in ses_coll:
        for rup in ses:
            rows.append([rup.etag])
    write_csv(dest, sorted(rows, key=operator.itemgetter(0)))


# #################### export Ground Motion fields ########################## #

class GmfSet(object):
    """
    Small wrapper around the list of Gmf objects associated to the given SES.
    """
    def __init__(self, gmfset, investigation_time, ses_idx):
        self.gmfset = gmfset
        self.investigation_time = investigation_time
        self.stochastic_event_set_id = ses_idx

    def __iter__(self):
        return iter(self.gmfset)

    def __bool__(self):
        return bool(self.gmfset)

    def __str__(self):
        return (
            'GMFsPerSES(investigation_time=%f, '
            'stochastic_event_set_id=%s,\n%s)' % (
                self.investigation_time,
                self.stochastic_event_set_id, '\n'.join(
                    sorted(str(g) for g in self.gmfset))))


class GroundMotionField(object):
    """
    The Ground Motion Field generated by the given rupture
    """
    def __init__(self, imt, sa_period, sa_damping, rupture_id, gmf_nodes):
        self.imt = imt
        self.sa_period = sa_period
        self.sa_damping = sa_damping
        self.rupture_id = rupture_id
        self.gmf_nodes = gmf_nodes

    def __iter__(self):
        return iter(self.gmf_nodes)

    def __getitem__(self, key):
        return self.gmf_nodes[key]

    def __str__(self):
        # string representation of a _GroundMotionField object showing the
        # content of the nodes (lon, lat an gmv). This is useful for debugging
        # and testing.
        mdata = ('imt=%(imt)s sa_period=%(sa_period)s '
                 'sa_damping=%(sa_damping)s rupture_id=%(rupture_id)s' %
                 vars(self))
        nodes = sorted(map(str, self.gmf_nodes))
        return 'GMF(%s\n%s)' % (mdata, '\n'.join(nodes))


class GroundMotionFieldNode(object):
    # the signature is not (gmv, x, y) because the XML writer expects
    # a location object
    def __init__(self, gmv, loc):
        self.gmv = gmv
        self.location = loc

    def __lt__(self, other):
        """
        A reproducible ordering by lon and lat; used in
        :function:`openquake.commonlib.hazard_writers.gen_gmfs`
        """
        return (self.location.x, self.location.y) < (
            other.location.x, other.location.y)

    def __str__(self):
        """Return lon, lat and gmv of the node in a compact string form"""
        return '<X=%9.5f, Y=%9.5f, GMV=%9.7f>' % (
            self.location.x, self.location.y, self.gmv)


class GmfCollection(object):
    """
    Object converting the parameters

    :param sitecol: SiteCollection
    :param ruptures: ruptures
    :param investigation_time: investigation time

    into an object with the right form for the EventBasedGMFXMLWriter.
    Iterating over a GmfCollection yields GmfSet objects.
    """
    def __init__(self, sitecol, imts, ruptures, investigation_time):
        self.sitecol = sitecol
        self.ruptures = ruptures
        self.imts = imts
        self.investigation_time = investigation_time

    def __iter__(self):
        completemesh = self.sitecol.complete.mesh
        gmfset = collections.defaultdict(list)
        for imti, imt_str in enumerate(self.imts):
            imt, sa_period, sa_damping = from_string(imt_str)
            for rupture in self.ruptures:
                mesh = completemesh[rupture.indices]
                gmf = get_array(rupture.gmfa, imti=imti)['gmv']
                assert len(mesh) == len(gmf), (len(mesh), len(gmf))
                nodes = (GroundMotionFieldNode(gmv, loc)
                         for gmv, loc in zip(gmf, mesh))
                gmfset[rupture.ses_idx].append(
                    GroundMotionField(
                        imt, sa_period, sa_damping, rupture.etag, nodes))
        for ses_idx in sorted(gmfset):
            yield GmfSet(gmfset[ses_idx], self.investigation_time, ses_idx)

# ####################### export hazard curves ############################ #

HazardCurve = collections.namedtuple('HazardCurve', 'location poes')


def export_hazard_curves_csv(key, dest, sitecol, curves_by_imt,
                             imtls, investigation_time=None):
    """
    Export the curves of the given realization into XML.

    :param key: output_type and export_type
    :param dest: name of the exported file
    :param sitecol: site collection
    :param curves_by_imt: dictionary with the curves keyed by IMT
    :param dict imtls: intensity measure types and levels
    :param investigation_time: investigation time
    """
    nsites = len(sitecol)
    # build a matrix of strings with size nsites * (num_imts + 1)
    # the + 1 is needed since the 0-th column contains lon lat
    rows = numpy.empty((nsites, len(imtls) + 1), dtype=object)
    for sid, lon, lat in zip(range(nsites), sitecol.lons, sitecol.lats):
        rows[sid, 0] = '%.5f %.5f' % (lon, lat)
    for i, imt in enumerate(curves_by_imt.dtype.names, 1):
        for sid, curve in zip(range(nsites), curves_by_imt[imt]):
            rows[sid, i] = scientificformat(curve, fmt='%11.7E')
    write_csv(dest, rows, header=('lon lat',) + curves_by_imt.dtype.names)
    return {dest: dest}


def hazard_curve_name(dstore, ekey, kind, rlzs_assoc, sampling):
    """
    :param calc_id: the calculation ID
    :param ekey: the export key
    :param kind: the kind of key
    :param rlzs_assoc: a RlzsAssoc instance
    :param sampling: if sampling is enabled or not
    """
    key, fmt = ekey
    prefix = {'hcurves': 'hazard_curve', 'hmaps': 'hazard_map',
              'uhs': 'hazard_uhs'}[key]
    if kind.startswith('rlz-'):
        rlz_no, suffix = re.match('rlz-(\d+)(.*)', kind).groups()
        rlz = rlzs_assoc.realizations[int(rlz_no)]
        fname = build_name(dstore, rlz, prefix + suffix, fmt, sampling)
    elif kind.startswith('mean'):
        fname = dstore.export_path('%s-%s.%s' % (prefix, kind, ekey[1]))
    elif kind.startswith('quantile-'):
        # strip the 7 characters 'hazard_'
        fname = dstore.export_path(
            'quantile_%s-%s.%s' % (prefix[7:], kind[9:], fmt))
    else:
        raise ValueError('Unknown kind of hazard curve: %s' % kind)
    return fname


def build_name(dstore, rlz, prefix, fmt, sampling):
    """
    Build a file name from a realization, by using prefix and extension.

    :param dstore: a DataStore instance
    :param rlz: a realization object
    :param prefix: the prefix to use
    :param fmt: the extension
    :param bool sampling: if sampling is enabled or not

    :returns: relative pathname including the extension
    """
    if hasattr(rlz, 'sm_lt_path'):  # full realization
        fname = '%s-rlz_%04d.%s' % (prefix, rlz.ordinal, fmt)
    else:  # GSIM logic tree realization used in scenario calculators
        fname = '%s_%s.%s' % (prefix, rlz.uid, fmt)
    return dstore.export_path(fname)


@export.add(('hcurves', 'csv'), ('hmaps', 'csv'), ('uhs', 'csv'))
def export_hcurves_csv(ekey, dstore):
    """
    Exports the hazard curves into several .csv files

    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    oq = dstore['oqparam']
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    sitecol = dstore['sitecol']
    sitemesh = dstore['sitemesh']
    key, fmt = ekey
    fnames = []
    items = dstore['hmaps' if key == 'uhs' else key].items()
    for kind, hcurves in sorted(items):
        fname = hazard_curve_name(
            dstore, ekey, kind, rlzs_assoc, oq.number_of_logic_tree_samples)
        if key == 'uhs':
            uhs_curves = calc.make_uhs(hcurves, oq.imtls, oq.poes)
            write_csv(fname, util.compose_arrays(sitemesh, uhs_curves))
        elif key == 'hmaps':
            write_csv(fname, util.compose_arrays(sitemesh, hcurves))
        else:
            export_hazard_curves_csv(ekey, fname, sitecol, hcurves, oq.imtls)
        fnames.append(fname)
    return sorted(fnames)

UHS = collections.namedtuple('UHS', 'imls location')


def get_metadata(realizations, kind):
    """
    :param list realizations:
        realization objects
    :param str kind:
        kind of data, i.e. a key in the datastore
    :returns:
        a dictionary with smlt_path, gsimlt_path, statistics, quantile_value
    """
    metadata = {}
    if kind.startswith('rlz-'):
        rlz = realizations[int(kind[4:])]
        metadata['smlt_path'] = '_'.join(rlz.sm_lt_path)
        metadata['gsimlt_path'] = rlz.gsim_rlz.uid
    elif kind.startswith('quantile-'):
        metadata['statistics'] = 'quantile'
        metadata['quantile_value'] = float(kind[9:])
    elif kind == 'mean':
        metadata['statistics'] = 'mean'
    return metadata


@export.add(('uhs', 'xml'))
def export_uhs_xml(ekey, dstore):
    oq = dstore['oqparam']
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    sitemesh = dstore['sitemesh'].value
    key, fmt = ekey
    fnames = []
    periods = [imt for imt in oq.imtls if imt.startswith('SA') or imt == 'PGA']
    for kind, hmaps in dstore['hmaps'].items():
        metadata = get_metadata(rlzs_assoc.realizations, kind)
        _, periods = calc.get_imts_periods(oq.imtls)
        uhs = calc.make_uhs(hmaps, oq.imtls, oq.poes)
        for poe in oq.poes:
            poe_str = 'poe~%s' % poe
            fname = hazard_curve_name(
                dstore, ekey, kind + '-%s' % poe, rlzs_assoc,
                oq.number_of_logic_tree_samples)
            writer = hazard_writers.UHSXMLWriter(
                fname, periods=periods, poe=poe,
                investigation_time=oq.investigation_time, **metadata)
            data = []
            for site, curve in zip(sitemesh, uhs[poe_str]):
                data.append(UHS(curve, Location(site)))
            writer.serialize(data)
            fnames.append(fname)
    return sorted(fnames)


# emulate a Django point
class Location(object):
    def __init__(self, xy):
        self.x, self.y = xy
        self.wkt = 'POINT(%s %s)' % tuple(xy)

HazardCurve = collections.namedtuple('HazardCurve', 'location poes')
HazardMap = collections.namedtuple('HazardMap', 'lon lat iml')


@export.add(('hcurves', 'xml'), ('hcurves', 'geojson'))
def export_hcurves_xml_json(ekey, dstore):
    export_type = ekey[1]
    len_ext = len(export_type) + 1
    oq = dstore['oqparam']
    sitemesh = dstore['sitemesh'].value
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    hcurves = dstore[ekey[0]]
    fnames = []
    writercls = (hazard_writers.HazardCurveGeoJSONWriter
                 if export_type == 'geojson' else
                 hazard_writers.HazardCurveXMLWriter)
    for kind in hcurves:
        if kind.startswith('rlz-'):
            rlz = rlzs_assoc.realizations[int(kind[4:])]
            smlt_path = '_'.join(rlz.sm_lt_path)
            gsimlt_path = rlz.gsim_rlz.uid
        else:
            smlt_path = ''
            gsimlt_path = ''
        curves = hcurves[kind]
        name = hazard_curve_name(
            dstore, ekey, kind, rlzs_assoc,
            oq.number_of_logic_tree_samples)
        for imt in oq.imtls:
            imtype, sa_period, sa_damping = from_string(imt)
            fname = name[:-len_ext] + '-' + imt + '.' + export_type
            data = [HazardCurve(Location(site), poes[imt])
                    for site, poes in zip(sitemesh, curves)]
            writer = writercls(fname,
                               investigation_time=oq.investigation_time,
                               imls=oq.imtls[imt], imt=imtype,
                               sa_period=sa_period, sa_damping=sa_damping,
                               smlt_path=smlt_path, gsimlt_path=gsimlt_path)
            writer.serialize(data)
            fnames.append(fname)
    return sorted(fnames)


@export.add(('hmaps', 'xml'), ('hmaps', 'geojson'))
def export_hmaps_xml_json(ekey, dstore):
    export_type = ekey[1]
    oq = dstore['oqparam']
    sitemesh = dstore['sitemesh'].value
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    hmaps = dstore[ekey[0]]
    fnames = []
    writercls = (hazard_writers.HazardMapGeoJSONWriter
                 if export_type == 'geojson' else
                 hazard_writers.HazardMapXMLWriter)
    for kind in hmaps:
        if kind.startswith('rlz-'):
            rlz = rlzs_assoc.realizations[int(kind[4:])]
            smlt_path = '_'.join(rlz.sm_lt_path)
            gsimlt_path = rlz.gsim_rlz.uid
        else:
            smlt_path = ''
            gsimlt_path = ''
        maps = hmaps[kind]
        for imt in oq.imtls:
            for poe in oq.poes:
                suffix = '-%s-%s' % (poe, imt)
                fname = hazard_curve_name(
                    dstore, ekey, kind + suffix, rlzs_assoc,
                    oq.number_of_logic_tree_samples)
                data = [HazardMap(site[0], site[1], hmap['%s~%s' % (imt, poe)])
                        for site, hmap in zip(sitemesh, maps)]
                writer = writercls(
                    fname, investigation_time=oq.investigation_time,
                    imt=imt, poe=poe,
                    smlt_path=smlt_path, gsimlt_path=gsimlt_path)
                writer.serialize(data)
                fnames.append(fname)
    return sorted(fnames)


@export.add(('gmf_data', 'xml'), ('gmf_data', 'txt'))
def export_gmf(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    sitecol = dstore['sitecol']
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    oq = dstore['oqparam']
    investigation_time = (None if oq.calculation_mode == 'scenario'
                          else oq.investigation_time)
    samples = oq.number_of_logic_tree_samples
    fmt = ekey[-1]
    etags = dstore['etags'].value
    gmf_data = dstore['gmf_data']
    nbytes = gmf_data.attrs['nbytes']
    logging.info('Internal size of the GMFs: %s', humansize(nbytes))
    if nbytes > GMF_MAX_SIZE:
        logging.warn(GMF_WARNING, dstore.hdf5path)
    fnames = []
    for rlz in rlzs_assoc.realizations:
        gmf_arr = gmf_data['%04d' % rlz.ordinal].value
        ruptures = []
        for eid, gmfa in group_array(gmf_arr, 'eid').items():
            rup = util.Rupture(etags[eid], sorted(set(gmfa['sid'])))
            rup.gmfa = gmfa
            ruptures.append(rup)
        ruptures.sort(key=operator.attrgetter('etag'))
        fname = build_name(dstore, rlz, 'gmf', fmt, samples)
        fnames.append(fname)
        globals()['export_gmf_%s' % fmt](
            ('gmf', fmt), fname, sitecol, oq.imtls, ruptures, rlz,
            investigation_time)
    return fnames


@export.add(('gmfs:', 'csv'))
def export_gmf_spec(ekey, dstore, spec):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    :param spec: a string specifying what to export exactly
    """
    oq = dstore['oqparam']
    eids = numpy.array([int(rid) for rid in spec.split(',')])
    sitemesh = dstore['sitemesh']
    writer = writers.CsvWriter(fmt='%.5f')
    etags = dstore['etags']
    if 'scenario' in oq.calculation_mode:
        _, gmfs_by_trt_gsim = base.get_gmfs(dstore)
        gsims = sorted(gsim for trt, gsim in gmfs_by_trt_gsim)
        imts = gmfs_by_trt_gsim[0, gsims[0]].dtype.names
        gmf_dt = numpy.dtype([(str(gsim), F32) for gsim in gsims])
        for eid in eids:
            etag = etags[eid]
            for imt in imts:
                gmfa = numpy.zeros(len(sitemesh), gmf_dt)
                for gsim in gsims:
                    gmfa[str(gsim)] = gmfs_by_trt_gsim[0, gsim][imt][:, eid]
                dest = dstore.export_path('gmf-%s-%s.csv' % (etag, imt))
                data = util.compose_arrays(sitemesh, gmfa)
                writer.save(data, dest)
    else:  # event based
        for eid in eids:
            etag = etags[eid]
            for gmfa, imt in _get_gmfs(dstore, util.get_serial(etag), eid):
                dest = dstore.export_path('gmf-%s-%s.csv' % (etag, imt))
                data = util.compose_arrays(sitemesh, gmfa)
                writer.save(data, dest)
    return writer.getsaved()


def export_gmf_xml(key, dest, sitecol, imts, ruptures, rlz,
                   investigation_time):
    """
    :param key: output_type and export_type
    :param dest: name of the exported file
    :param sitecol: the full site collection
    :param imts: the list of intensity measure types
    :param ruptures: an ordered list of ruptures
    :param rlz: a realization object
    :param investigation_time: investigation time (None for scenario)
    """
    if hasattr(rlz, 'gsim_rlz'):  # event based
        smltpath = '_'.join(rlz.sm_lt_path)
        gsimpath = rlz.gsim_rlz.uid
    else:  # scenario
        smltpath = ''
        gsimpath = rlz.uid
    writer = hazard_writers.EventBasedGMFXMLWriter(
        dest, sm_lt_path=smltpath, gsim_lt_path=gsimpath)
    writer.serialize(
        GmfCollection(sitecol, imts, ruptures, investigation_time))
    return {key: [dest]}


def export_gmf_txt(key, dest, sitecol, imts, ruptures, rlz,
                   investigation_time):
    """
    :param key: output_type and export_type
    :param dest: name of the exported file
    :param sitecol: the full site collection
    :param imts: the list of intensity measure types
    :param ruptures: an ordered list of ruptures
    :param rlz: a realization object
    :param investigation_time: investigation time (None for scenario)
    """
    # the csv file has the form
    # etag,indices,gmvs_imt_1,...,gmvs_imt_N
    rows = []
    for rupture in ruptures:
        indices = rupture.indices
        gmvs = [a['gmv'] for a in group_array(rupture.gmfa, 'imti').values()]
        row = [rupture.etag, ' '.join(map(str, indices))] + gmvs
        rows.append(row)
    write_csv(dest, rows)
    return {key: [dest]}


def get_rup_idx(ebrup, etag):
    # extract the rupture and the index of the given etag from a collection
    for etag_idx, tag in enumerate(ebrup.etags):
        if tag == etag:
            return etag_idx
    raise ValueError('event tag %s not found in the rupture collection')


def _get_gmfs(dstore, serial, eid):
    oq = dstore['oqparam']
    min_iml = event_based.fix_minimum_intensity(oq.minimum_intensity, oq.imtls)
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    sitecol = dstore['sitecol'].complete
    N = len(sitecol.complete)
    rup = dstore['sescollection/' + serial]
    correl_model = readinput.get_correl_model(oq)
    gsims = rlzs_assoc.gsims_by_trt_id[rup.trt_id]
    rlzs = [rlz for gsim in map(str, gsims)
            for rlz in rlzs_assoc[rup.trt_id, gsim]]
    gmf_dt = numpy.dtype([('%03d' % rlz.ordinal, F32) for rlz in rlzs])
    gmfadict = create(event_based.GmfColl,
                      [rup], sitecol, oq.imtls, rlzs_assoc,
                      oq.truncation_level, correl_model, min_iml).by_rlzi()
    for imti, imt in enumerate(oq.imtls):
        gmfa = numpy.zeros(N, gmf_dt)
        for rlzname in gmf_dt.names:
            rlzi = int(rlzname)
            gmvs = get_array(gmfadict[rlzi], eid=eid, imti=imti)['gmv']
            gmfa[rlzname][rup.indices] = gmvs
        yield gmfa, imt


@export.add(('gmf_data', 'csv'))
def export_gmf_scenario(ekey, dstore):
    oq = dstore['oqparam']
    if 'scenario' in oq.calculation_mode:
        fields = ['%03d' % i for i in range(len(dstore['etags']))]
        dt = numpy.dtype([(f, F32) for f in fields])
        etags, gmfs_by_trt_gsim = base.get_gmfs(dstore)
        sitemesh = dstore['sitemesh']
        writer = writers.CsvWriter(fmt='%.5f')
        for (trt, gsim), gmfs_ in gmfs_by_trt_gsim.items():
            for imt in gmfs_.dtype.names:
                gmfs = numpy.zeros(len(gmfs_), dt)
                for i in range(len(gmfs)):
                    gmfs[i] = tuple(gmfs_[imt][i])
                dest = dstore.export_path('gmf-%s-%s.csv' % (gsim, imt))
                data = util.compose_arrays(sitemesh, gmfs)
                writer.save(data, dest)
    else:  # event based
        logging.warn('Not exporting the full GMFs for event_based, but you can'
                     ' specify the rupture ordinals with gmfs:R1,...,Rn')
        return []
    return writer.getsaved()


# not used right now
def export_hazard_curves_xml(key, dest, sitecol, curves_by_imt,
                             imtls, investigation_time):
    """
    Export the curves of the given realization into XML.

    :param key: output_type and export_type
    :param dest: name of the exported file
    :param sitecol: site collection
    :param curves_by_imt: dictionary with the curves keyed by IMT
    :param imtls: dictionary with the intensity measure types and levels
    :param investigation_time: investigation time in years
    """
    mdata = []
    hcurves = []
    for imt_str, imls in sorted(imtls.items()):
        hcurves.append(
            [HazardCurve(site.location, poes)
             for site, poes in zip(sitecol, curves_by_imt[imt_str])])
        imt = from_string(imt_str)
        mdata.append({
            'quantile_value': None,
            'statistics': None,
            'smlt_path': '',
            'gsimlt_path': '',
            'investigation_time': investigation_time,
            'imt': imt[0],
            'sa_period': imt[1],
            'sa_damping': imt[2],
            'imls': imls,
        })
    writer = hazard_writers.MultiHazardCurveXMLWriter(dest, mdata)
    with floatformat('%12.8E'):
        writer.serialize(hcurves)
    return {dest: dest}


DisaggMatrix = collections.namedtuple(
    'DisaggMatrix', 'poe iml dim_labels matrix')


@export.add(('disagg', 'xml'))
def export_disagg_xml(ekey, dstore):
    oq = dstore['oqparam']
    rlzs = dstore['csm_info'].get_rlzs_assoc().realizations
    group = dstore['disagg']
    fnames = []
    writercls = hazard_writers.DisaggXMLWriter
    for key in group:
        matrix = pickle.loads(group[key].value)
        attrs = group[key].attrs
        rlz = rlzs[attrs['rlzi']]
        poe = attrs['poe']
        iml = attrs['iml']
        imt, sa_period, sa_damping = from_string(attrs['imt'])
        fname = dstore.export_path(key + '.xml')
        lon, lat = attrs['location']
        # TODO: add poe=poe below
        writer = writercls(
            fname, investigation_time=oq.investigation_time,
            imt=imt, smlt_path='_'.join(rlz.sm_lt_path),
            gsimlt_path=rlz.gsim_rlz.uid, lon=lon, lat=lat,
            sa_period=sa_period, sa_damping=sa_damping,
            mag_bin_edges=attrs['mag_bin_edges'],
            dist_bin_edges=attrs['dist_bin_edges'],
            lon_bin_edges=attrs['lon_bin_edges'],
            lat_bin_edges=attrs['lat_bin_edges'],
            eps_bin_edges=attrs['eps_bin_edges'],
            tectonic_region_types=attrs['trts'],
        )
        data = [DisaggMatrix(poe, iml, dim_labels, matrix[i])
                for i, dim_labels in enumerate(disagg.pmf_map)]
        writer.serialize(data)
        fnames.append(fname)
    return sorted(fnames)
