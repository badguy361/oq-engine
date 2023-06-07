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
Module exports :class:`AbrahamsonEtAl2014`
               :class:`AbrahamsonEtAl2014RegCHN`
               :class:`AbrahamsonEtAl2014RegJPN`
               :class:`AbrahamsonEtAl2014RegTWN`
"""
import copy
import numpy as np
# import os
# os.chdir("E:\Yu\oq-engine")
from scipy import interpolate
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable, add_alias
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA
class uuu(GMPE):
    """
    Implements GMPE by Abrahamson, Silva and Kamai developed within the
    the PEER West 2 Project. This GMPE is described in a paper
    published in 2014 on Earthquake Spectra, Volume 30, Number 3 and
    titled 'Summary of the ASK14 Ground Motion Relation for Active Crustal
    Regions'.
    """
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

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            # compute median sa on rock (vs30=1180m/s). Used for site response
            # term calculation

            # For debugging purposes
            # f1 = _get_basic_term(C, ctx)
            # f4 = _get_hanging_wall_term(C, ctx)
            # f5 = _get_site_response_term(C, imt, ctx.vs30, sa1180)
            # f6 = _get_top_of_rupture_depth_term(C, imt, ctx)
            # f7 = _get_faulting_style_term(C, ctx)
            # f10 =_get_soil_depth_term(self.region, C, ctx.z1pt0, ctx.vs30)
            # fre = _get_regional_term(self.region, C, imt, ctx.vs30, ctx.rrup)

            # get the mean value
            mean[m] = 1

            mean[m] += 2

            # get standard deviations
            sig[m], tau[m], phi[m] = 3,5,4

    #: Coefficient tables as per annex B of Abrahamson et al. (2014)
    COEFFS = CoeffsTable(sa_damping=5, table="""\
IMT     m1      vlin    b       c       c4      a1      a2      a3      a4      a5      a6      a7   a8      a10     a11     a12     a13     a14     a15     a17     a43     a44     a45     a46     a25     a28     a29     a31     a36     a37     a38     a39     a40     a41     a42     s1e     s2e     s3      s4      s1m     s2m     s5      s6
pga     6.75    660     -1.47   2.4     4.5     0.587   -0.79   0.275   -0.1    -0.41   2.154   0.0  -0.015  1.735   0       -0.1    0.6     -0.3    1.1     -0.0072 0.1     0.05    0       -0.05   -0.0015 0.0025  -0.0034 -0.1503 0.265   0.337   0.188   0       0.088   -0.196  0.044   0.754   0.52    0.47    0.36    0.741   0.501   0.54    0.6300
pgv     6.75    330     -2.02   2400    4.5     5.975   -0.919  0.275   -0.1    -0.41   2.366   0.0  -0.094  2.36    0       -0.1    0.25    0.22    0.3     -0.0005 0.28    0.15    0.09    0.07    -0.0001 0.0005  -0.0037 -0.1462 0.377   0.212   0.157   0       0.095   -0.038  0.065   0.662   0.51    0.38    0.38    0.66    0.51    0.58    0.5300
0.01    6.75    660     -1.47   2.4     4.5     0.587   -0.790  0.275   -0.1    -0.41   2.154   0.0  -0.015  1.735   0       -0.1    0.6     -0.3    1.1     -0.0072 0.1     0.05    0       -0.05   -0.0015 0.0025  -0.0034 -0.1503 0.265   0.337   0.188   0       0.088   -0.196  0.044   0.754   0.52    0.47    0.36    0.741   0.501   0.54    0.6300
0.02    6.75    680     -1.46   2.4     4.5     0.598   -0.790  0.275   -0.1    -0.41   2.146   0.0  -0.015  1.718   0       -0.1    0.6     -0.3    1.1     -0.0073 0.1     0.05    0       -0.05   -0.0015 0.0024  -0.0033 -0.1479 0.255   0.328   0.184   0       0.088   -0.194  0.061   0.76    0.52    0.47    0.36    0.747   0.501   0.54    0.6300
0.03    6.75    770     -1.39   2.4     4.5     0.602   -0.790  0.275   -0.1    -0.41   2.157   0.0  -0.015  1.615   0       -0.1    0.6     -0.3    1.1     -0.0075 0.1     0.05    0       -0.05   -0.0016 0.0023  -0.0034 -0.1447 0.249   0.32    0.18    0       0.093   -0.175  0.162   0.781   0.52    0.47    0.36    0.769   0.501   0.55    0.6300
0.05    6.75    915     -1.22   2.4     4.5     0.707   -0.790  0.275   -0.1    -0.41   2.085   0.0  -0.015  1.358   0       -0.1    0.6     -0.3    1.1     -0.008  0.1     0.05    0       -0.05   -0.002  0.0027  -0.0033 -0.1326 0.202   0.289   0.167   0       0.133   -0.09   0.451   0.81    0.53    0.47    0.36    0.798   0.512   0.56    0.6500
0.075   6.75    960     -1.15   2.4     4.5     0.973   -0.790  0.275   -0.1    -0.41   2.029   0.0  -0.015  1.258   0       -0.1    0.6     -0.3    1.1     -0.0089 0.1     0.05    0       -0.05   -0.0027 0.0032  -0.0029 -0.1353 0.126   0.275   0.173   0       0.186   0.09    0.506   0.81    0.54    0.47    0.36    0.798   0.522   0.57    0.6900
0.1     6.75    910     -1.23   2.4     4.5     1.169   -0.790  0.275   -0.1    -0.41   2.041   0.0  -0.015  1.31    0       -0.1    0.6     -0.3    1.1     -0.0095 0.1     0.05    0       -0.05   -0.0033 0.0036  -0.0025 -0.1128 0.022   0.256   0.189   0       0.16    0.006   0.335   0.81    0.55    0.47    0.36    0.795   0.527   0.57    0.7000
0.15    6.75    740     -1.59   2.4     4.5     1.442   -0.790  0.275   -0.1    -0.41   2.121   0.0  -0.022  1.66    0       -0.1    0.6     -0.3    1.1     -0.0095 0.1     0.05    0       -0.05   -0.0035 0.0033  -0.0025 0.0383  -0.136  0.162   0.108   0       0.068   -0.156  -0.084  0.801   0.56    0.47    0.36    0.773   0.519   0.58    0.7000
0.2     6.75    590     -2.01   2.4     4.5     1.637   -0.790  0.275   -0.1    -0.41   2.224   0.0  -0.03   2.22    0       -0.1    0.6     -0.3    1.1     -0.0086 0.1     0.05    0       -0.03   -0.0033 0.0027  -0.0031 0.0775  -0.078  0.224   0.115   0       0.048   -0.274  -0.178  0.789   0.565   0.47    0.36    0.753   0.514   0.59    0.7000
0.25    6.75    495     -2.41   2.4     4.5     1.701   -0.790  0.275   -0.1    -0.41   2.312   0.0  -0.038  2.77    0       -0.1    0.6     -0.24   1.1     -0.0074 0.1     0.05    0       0       -0.0029 0.0024  -0.0036 0.0741  0.037   0.248   0.122   0       0.055   -0.248  -0.187  0.77    0.57    0.47    0.36    0.729   0.513   0.61    0.7000
0.3     6.75    430     -2.76   2.4     4.5     1.712   -0.790  0.275   -0.1    -0.41   2.338   0.0  -0.045  3.25    0       -0.1    0.6     -0.19   1.03    -0.0064 0.1     0.05    0.03    0.03    -0.0027 0.002   -0.0039 0.2548  -0.091  0.203   0.096   0       0.073   -0.203  -0.159  0.74    0.58    0.47    0.36    0.693   0.519   0.63    0.7000
0.4     6.75    360     -3.28   2.4     4.5     1.662   -0.790  0.275   -0.1    -0.41   2.469   0.0  -0.055  3.99    0       -0.1    0.58    -0.11   0.92    -0.0043 0.1     0.07    0.06    0.06    -0.0023 0.001   -0.0048 0.2136  0.129   0.232   0.123   0       0.143   -0.154  -0.023  0.699   0.59    0.47    0.36    0.644   0.524   0.66    0.7000
0.5     6.75    340     -3.6    2.4     4.5     1.571   -0.790  0.275   -0.1    -0.41   2.559   0.0  -0.065  4.45    0       -0.1    0.56    -0.04   0.84    -0.0032 0.1     0.1     0.1     0.09    -0.002  0.0008  -0.005  0.1542  0.31    0.252   0.134   0       0.16    -0.159  -0.029  0.676   0.6     0.47    0.36    0.616   0.532   0.69    0.7000
0.75    6.75    330     -3.8    2.4     4.5     1.299   -0.790  0.275   -0.1    -0.41   2.682   0.0  -0.095  4.75    0       -0.1    0.53    0.07    0.68    -0.0025 0.14    0.14    0.14    0.13    -0.001  0.0007  -0.0041 0.0787  0.505   0.208   0.129   0       0.158   -0.141  0.061   0.631   0.615   0.47    0.36    0.566   0.548   0.73    0.6900
1       6.75    330     -3.5    2.4     4.5     1.043   -0.790  0.275   -0.1    -0.41   2.763   0.0  -0.11   4.3     0       -0.1    0.5     0.15    0.57    -0.0025 0.17    0.17    0.17    0.14    -0.0005 0.0007  -0.0032 0.0476  0.358   0.208   0.152   0       0.145   -0.144  0.062   0.609   0.63    0.47    0.36    0.541   0.565   0.77    0.6800
1.5     6.75    330     -2.4    2.4     4.5     0.665   -0.790  0.275   -0.1    -0.41   2.836   0.0  -0.124  2.6     0       -0.1    0.42    0.27    0.42    -0.0022 0.22    0.21    0.2     0.16    -0.0004 0.0006  -0.002  -0.0163 0.131   0.108   0.118   0       0.131   -0.126  0.037   0.578   0.64    0.47    0.36    0.506   0.576   0.8     0.6600
2       6.75    330     -1      2.4     4.5     0.329   -0.790  0.275   -0.1    -0.41   2.897   0.0  -0.138  0.55    0       -0.1    0.35    0.35    0.31    -0.0019 0.26    0.25    0.22    0.16    -0.0002 0.0003  -0.0017 -0.1203 0.123   0.068   0.119   0       0.083   -0.075  -0.143  0.555   0.65    0.47    0.36    0.48    0.587   0.8     0.6200
3       6.82    330     0       2.4     4.5     -0.060  -0.790  0.275   -0.1    -0.41   2.906   0.0  -0.172  -0.95   0       -0.1    0.2     0.46    0.16    -0.0015 0.34    0.3     0.23    0.16    0       0       -0.002  -0.2719 0.109   -0.023  0.093   0       0.07    -0.021  -0.028  0.548   0.64    0.47    0.36    0.472   0.576   0.8     0.5500
4       6.92    330     0       2.4     4.5     -0.299  -0.790  0.275   -0.1    -0.41   2.889   0.0  -0.197  -0.95   0       -0.1    0       0.54    0.05    -0.001  0.41    0.32    0.23    0.14    0       0       -0.002  -0.2958 0.135   0.028   0.084   0       0.101   0.072   -0.097  0.527   0.63    0.47    0.36    0.447   0.565   0.76    0.5200
5       7       330     0       2.4     4.5     -0.562  -0.765  0.275   -0.1    -0.41   2.898   0.0  -0.218  -0.93   0       -0.1    0       0.61    -0.04   -0.001  0.51    0.32    0.22    0.13    0       0       -0.002  -0.2718 0.189   0.031   0.058   0       0.095   0.205   0.015   0.505   0.63    0.47    0.36    0.425   0.568   0.72    0.5000
6       7.06    330     0       2.4     4.5     -0.875  -0.711  0.275   -0.1    -0.41   2.896   0.0  -0.235  -0.91   0       -0.2    0       0.65    -0.11   -0.001  0.55    0.32    0.2     0.1     0       0       -0.002  -0.2517 0.215   0.024   0.065   0       0.133   0.285   0.104   0.477   0.63    0.47    0.36    0.395   0.571   0.7     0.5000
7.5     7.15    330     0       2.4     4.5     -1.303  -0.634  0.275   -0.1    -0.41   2.870   0.0  -0.255  -0.87   0       -0.2    0       0.72    -0.19   -0.001  0.49    0.28    0.17    0.09    0       0       -0.002  -0.14   0.15    -0.07   0       0       0.151   0.329   0.299   0.457   0.63    0.47    0.36    0.378   0.575   0.67    0.5000
10      7.25    330     0       2.4     4.5     -1.928  -0.529  0.275   -0.1    -0.41   2.843   0.0  -0.285  -0.8    0       -0.2    0       0.8     -0.3    -0.001  0.42    0.22    0.14    0.08    0       0       -0.002  -0.0216 0.092   -0.159  -0.05   0       0.124   0.301   0.243   0.429   0.63    0.47    0.36    0.359   0.585   0.64    0.5000
    """)

if __name__ == "__main__":
    print("hihihi")
    test = uuu()
    print(test.DEFINED_FOR_STANDARD_DEVIATION_TYPES)
    # print(test.COEFFS[PGA])
    ctx, imts, mean, sig, tau, phi=1,(PGA(),SA(0.1)),[[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]]
    test.compute(ctx, imts, mean, sig, tau, phi)