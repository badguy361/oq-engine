# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation, Chih-Yu Chang
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

import copy
import numpy as np
import pandas as pd
from scipy import interpolate
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable, add_alias
from openquake.hazardlib import const
import pickle
from openquake.hazardlib.imt import PGA, PGV, SA
import xgboost as xgb

class Chang2023(GMPE):
    import warnings
    warnings.filterwarnings('ignore')
    #: Supported tectonic region type is active shallow crust, see title!
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration, peak
    #: ground velocity and peak ground acceleration, see tables 4
    #: pages 1036
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is orientation-independent
    #: average horizontal :attr:`~openquake.hazardlib.const.IMC.RotD50`,
    #: see page 1025.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see paragraph "Equations for standard deviations", page
    #: 1046.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Required site parameters are Vs30 and Z1.0, see table 2, page 1031
    #: Unit of measure for Z1.0 is [m]
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters are magnitude, rake, dip, ztor, and width
    #: (see table 2, page 1031)
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake'}

    #: Required distance measures are Rrup, Rjb, Ry0 and Rx (see Table 2,
    #: page 1031).
    REQUIRES_DISTANCES = {'rrup'}

    #: Reference rock conditions as defined at page
    DEFINED_FOR_REFERENCE_VELOCITY = 1180

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ML_model = xgb.Booster()
        ML_model.load_model(f'/usr/src/oq-engine/openquake/hazardlib/gsim/XGB_PGA.json')
        self.ML_model = ML_model
        Sta_ID_thread = pd.read_csv(f"/usr/src/oq-engine/openquake/hazardlib/gsim/Sta_ID_info.csv")
        self.Sta_ID_thread = Sta_ID_thread

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        point = [119.5635611,21.90093889] # standard
        for m, imt in enumerate(imts):
            # cal sta_id by distance
            sta_dist = self.Sta_ID_thread['STA_DIST'].values.tolist()
            new_number = ((((ctx.lat-point[1])*110)**2 + ((ctx.lon-point[0])*101)**2)**(1/2))
            sta_id = np.searchsorted(sta_dist, new_number)

            predict = self.ML_model.predict(xgb.DMatrix(np.column_stack((np.log(ctx.vs30), ctx.mag, np.log(ctx.rrup), ctx.rake, sta_id))))
            mean[m] = np.log(np.exp(predict)/980)
            sig[m], tau[m], phi[m] = 0.35,0.12,0.34
        print(mean)
        