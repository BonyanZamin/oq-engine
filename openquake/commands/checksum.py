# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2017 GEM Foundation
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
import sys
import os.path
from openquake.baselib import sap
from openquake.commonlib import datastore, readinput


@sap.Script
def checksum(job_ini_or_job_id):
    """
    Get the checksum of a calculation from the calculation ID (if already
    done) or from the job.ini file (if not done yet).
    """
    try:
        job_id = int(job_ini_or_job_id)
        job_ini = None
    except ValueError:
        job_id = None
        job_ini = job_ini_or_job_id
        if not os.path.exists(job_ini):
            sys.exit('%s does not correspond to an existing file' % job_ini)
    if job_id:
        dstore = datastore.read(job_id)
        checksum = dstore['/'].attrs['checksum32']
    else:
        oq = readinput.get_oqparam(job_ini)
        checksum = readinput.get_checksum32(oq)
    print(checksum)

checksum.arg('job_ini_or_job_id', 'job.ini or job ID')
