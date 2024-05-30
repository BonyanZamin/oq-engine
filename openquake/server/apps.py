# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2016-2023 GEM Foundation
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

from django.apps import AppConfig
from django.conf import settings
from openquake.baselib import config


class ServerConfig(AppConfig):
    name = 'openquake.server'

    def ready(self):
        # From Django manual:
        #     Subclasses can override this method to perform initialization
        #     tasks such as registering signals. It is called as soon as the
        #     registry is fully populated.
        #     Although you can’t import models at the module-level where
        #     AppConfig classes are defined, you can import them in ready()
        import openquake.server.signals  # NOQA

        if settings.APPLICATION_MODE not in settings.APPLICATION_MODES:
            raise ValueError(
                f'Invalid application mode: "{settings.APPLICATION_MODE}".'
                f' It must be one of {settings.APPLICATION_MODES}')
        if settings.LOCKDOWN and settings.APPLICATION_MODE in (
                'AELO', 'ARISTOTLE'):
            # check essential constants are defined
            try:
                settings.EMAIL_BACKEND  # noqa
            except NameError:
                raise NameError(
                    f'If APPLICATION_MODE is {settings.APPLICATION_MODE} an'
                    f' email backend must be defined')
            if settings.EMAIL_BACKEND == 'django.core.mail.backends.smtp.EmailBackend':  # noqa
                try:
                    settings.EMAIL_HOST           # noqa
                    settings.EMAIL_PORT           # noqa
                    settings.EMAIL_USE_TLS        # noqa
                    settings.EMAIL_HOST_USER      # noqa
                    settings.EMAIL_HOST_PASSWORD  # noqa
                except NameError:
                    raise NameError(
                        f'If APPLICATION_MODE is {settings.APPLICATION_MODE}'
                        f' EMAIL_<HOST|PORT|USE_TLS|HOST_USER|HOST_PASSWORD>'
                        f' must all be defined')
            if not config.directory.mosaic_dir:
                raise NameError(
                    f'If APPLICATION_MODE is {settings.APPLICATION_MODE}, '
                    f'mosaic_dir must be specified in openquake.cfg')
