# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4 \
#
# Copyright (C) 2019 GEM Foundation, Chung-Han Chan
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
Module exports :class:`Lin2011hanging`

"""
import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


class Lin2011hanging(GMPE):
    """
    Implements GMPE developed by Po-Shen Lin and others and published as
    "Response spectral attenuation relations for shallow crustal earthquakes
    in Taiwan"
    (Engineering Geology, Volume 121, Issues 3–4, 10 August 2011, Pages 150-164).
    """

    #: Supported tectonic region type is active shallow crust.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration, see tables 3 and 4, pages 227 and 228.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is geometric mean
    #: of two horizontal components, see equation 10 page 226.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.HORIZONTAL

    #: Supported standard deviation types is total, see equation 10 page 226.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameter is only Vs30 (used to distinguish rock
    #: and deep soil).
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude, and focal depth, see
    #: equation 10 page 226.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'hypo_depth'))

    #: Required distance measure is hypocentral distance, see equation 10
    #: page 226.
    REQUIRES_DISTANCES = set(('rhypo', ))

    #: Vs30 threshold value between rock sites (B, C) and soil sites (C, D).
    ROCK_VS30 = 360

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
                   for stddev_type in stddev_types)

        mean = np.zeros_like(sites.vs30)
        stddevs = [np.zeros_like(sites.vs30) for _ in stddev_types]

        idx_rock = sites.vs30 >= self.ROCK_VS30
        idx_soil = sites.vs30 < self.ROCK_VS30

        if idx_rock.any():
            C = self.COEFFS_ROCK[imt]
            self._compute_mean(C, rup.mag, dists.rhypo, rup.hypo_depth, mean,
                               idx_rock)
            self._compute_std(C, stddevs, idx_rock)

        if idx_soil.any():
            C = self.COEFFS_SOIL[imt]
            self._compute_mean(C, rup.mag, dists.rhypo, rup.hypo_depth, mean,
                               idx_soil)
            self._compute_std(C, stddevs, idx_soil)

        return mean, stddevs

    def _compute_mean(self, C, mag, rhypo, hypo_depth, mean, idx):
        """
        Compute mean value according to equations 10 and 11 page 226.
        """
        mean[idx] = (C['C1'] + C['C2'] * mag + C['C3'] * np.log(rhypo[idx] +
                     C['C4'] * np.exp(C['C5'] * mag)))

    def _compute_std(self, C, stddevs, idx):
        """
        Compute total standard deviation, see tables 3 and 4, pages 227 and
        228.
        """
        for stddev in stddevs:
            stddev[idx] += C['sigma']

    #: Coefficient table for rock sites, see table 3 page 153.
    COEFFS_ROCK = CoeffsTable(sa_damping=5, table="""\
    IMT      C1       C2        C3         C4         C5         sigma
    pga     -3.2790    1.035    -1.65100    0.15200    0.62300   0.6510
    0.01    -3.2530    1.018    -1.62900    0.15900    0.61200   0.6470
    0.06    -1.7380    0.908    -1.76900    0.32700    0.50200   0.7020
    0.09    -1.2370    0.841    -1.75000    0.47800    0.40200   0.7480
    0.10    -1.1030    0.841    -1.76500    0.45500    0.41700   0.7500
    0.20    -2.7670    0.980    -1.52200    0.09700    0.62700   0.6970
    0.30    -4.4400    1.186    -1.43800    0.02700    0.82300   0.6850
    0.40    -5.6300    1.335    -1.41400    0.01400    0.93200   0.6830
    0.50    -6.7460    1.456    -1.36500    0.00600    1.05700   0.6780
    0.60    -7.6370    1.557    -1.34800    0.00330    1.14700   0.6660
    0.75    -8.6410    1.653    -1.31300    0.00150    1.25700   0.6520
    1.00    -9.9780    1.800    -1.28600    0.00080    1.37700   0.6710
    1.50   -11.6170    1.976    -1.28400    0.00040    1.50800   0.6830
    2.00   -12.6110    2.058    -1.26100    0.00050    1.49700   0.7060
    3.00   -13.3030    2.036    -1.23400    0.00130    1.30200   0.7020
    5.00   -13.9140    1.958    -1.15600    0.00120    1.24100   0.7260
    """)

     #: Coefficient table for soil sites, see table 4 page 153.
    COEFFS_SOIL = CoeffsTable(sa_damping=5, table="""\
    IMT      C1        C2         C3        C4         C5         sigma
    pga     -3.2480    0.943    -1.47100    0.10000    0.64800   0.6280
    0.01    -3.0080    0.905    -1.45100    0.11000    0.63800   0.6230
    0.06    -1.9940    0.809    -1.50000    0.25100    0.51800   0.6860
    0.09    -1.4080    0.765    -1.55100    0.28000    0.51000   0.7090 
    0.10    -1.5080    0.785    -1.55100    0.28000    0.50000   0.7130
    0.20    -3.2260    0.870    -1.21100    0.04500    0.70800   0.6870
    0.30    -4.0500    0.999    -1.20500    0.03000    0.78800   0.6570
    0.40    -5.2930    1.165    -1.16700    0.01100    0.95800   0.6550
    0.50    -6.3070    1.291    -1.13400    0.00420    1.11800   0.6530
    0.60    -7.2090    1.395    -1.09900    0.00160    1.25800   0.6420
    0.75    -8.3090    1.509    -1.04400    0.00060    1.40800   0.6510
    1.00    -9.8680    1.691    -1.00400    0.00040    1.48500   0.6770
    1.50   -11.2160    1.798    -0.96500    0.00030    1.52200   0.7220
    2.00   -12.8060    2.005    -0.97500    0.00050    1.52800   0.7590
    3.00   -13.8860    2.099    -1.07700    0.00040    1.54800   0.7870
    5.00   -14.6060    2.160    -1.11400    0.00040    1.56200   0.8200
    """)
