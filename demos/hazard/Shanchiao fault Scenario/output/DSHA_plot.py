import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import pygmt
fault_data = [
    121.4185, 24.9925, 121.4169, 24.9967, 121.4152, 25.0010, 121.4130, 25.0050,
    121.4120, 25.0093, 121.4119, 25.0138, 121.4097, 25.0177, 121.4104, 25.0220,
    121.4122, 25.0262, 121.4143, 25.0302, 121.4172, 25.0339, 121.4198, 25.0377,
    121.4223, 25.0416, 121.4248, 25.0455, 121.4271, 25.0495, 121.4287, 25.0538,
    121.4304, 25.0580, 121.4314, 25.0624, 121.4317, 25.0669, 121.4322, 25.0713,
    121.4333, 25.0758, 121.4348, 25.0800, 121.4371, 25.0840, 121.4394, 25.0880,
    121.4429, 25.0911, 121.4463, 25.0944, 121.4493, 25.0980, 121.4515, 25.1020,
    121.4538, 25.1061, 121.4565, 25.1099, 121.4595, 25.1134, 121.4632, 25.1164,
    121.4657, 25.1203, 121.4667, 25.1247, 121.4694, 25.1285, 121.4729, 25.1316,
    121.4765, 25.1347, 121.4806, 25.1371, 121.4851, 25.1391, 121.4889, 25.1420,
    121.4922, 25.1453, 121.4965, 25.1471, 121.5010, 25.1490, 121.5053, 25.1513,
    121.5088, 25.1544, 121.5128, 25.1571, 121.5169, 25.1595, 121.5202, 25.1629,
    121.5241, 25.1656, 121.5281, 25.1683, 121.5316, 25.1715, 121.5350, 25.1748,
    121.5391, 25.1771, 121.5439, 25.1783, 121.5477, 25.1811, 121.5512, 25.1843,
    121.5550, 25.1871, 121.5584, 25.1904, 121.5617, 25.1937, 121.5650, 25.1971,
    121.5681, 25.2006, 121.5721, 25.2033, 121.5767, 25.2049, 121.5813, 25.2065,
    121.5861, 25.2078, 121.5901, 25.2102, 121.5939, 25.2130, 121.5978, 25.2156,
    121.6014, 25.2187, 121.6053, 25.2216, 121.6091, 25.2245, 121.6127, 25.2276,
    121.6168, 25.2302, 121.6210, 25.2325, 121.6254, 25.2346, 121.6291, 25.2374,
    121.6324, 25.2409, 121.6366, 25.2430, 121.6411, 25.2449, 121.6455, 25.2470,
    121.6503, 25.2482, 121.6552, 25.2491, 121.6600, 25.2501, 121.6647, 25.2515,
    121.6694, 25.2529, 121.6739, 25.2548, 121.6782, 25.2570, 121.6817, 25.2602,
    121.6854, 25.2630, 121.6896, 25.2654, 121.6929, 25.2687, 121.6968, 25.2715,
    121.6999, 25.2749, 121.7027, 25.2786, 121.7063, 25.2814, 121.7110, 25.2829,
    121.7155, 25.2846, 121.7189, 25.2880, 121.7220, 25.2914, 121.7247, 25.2952,
    121.7270, 25.2992, 121.7292, 25.3033, 121.7316, 25.3072, 121.7357, 25.3097,
    121.7402, 25.3117, 121.7450, 25.3126, 121.7494, 25.3145, 121.7535, 25.3170,
    121.7586, 25.3200
]

#'AbrahamsonEtAl2014','Yu2023','Lin2009','BooreAtkinson2008','Allen2022','Chang2023'
id = 92
folder = 'AbrahamsonEtAl2014'
df_site = pd.read_csv(f"100result/{folder}({id})/sitemesh_{id}.csv", skiprows=[0])
df_gmf = pd.read_csv(f"100result/{folder}({id})/gmf-data_{id}.csv", skiprows=[0])
df_total = df_gmf.merge(df_site, how='left', on='site_id')

df_total = df_total.groupby("site_id").mean()
gmv_PGA = df_total["gmv_PGA"]
x = df_total["lon"]
y = df_total["lat"]

fig = pygmt.Figure()
region = [119.5, 122.5, 21.5, 25.5]

fig.basemap(region=region,
            projection="M12c",
            frame=["af", f"WSne+tScenario Hazard Analysis gmm type: {folder}"])
fig.coast(land="gray", water="gray", shorelines="1p,black")
pygmt.makecpt(cmap="turbo", series=(0, 1.5))
fig.plot(x=fault_data[::2], y=fault_data[1::2],pen="thick,red")
fig.plot(x=x, y=y, style="c0.2c", cmap=True, color=gmv_PGA)
fig.colorbar(frame=["x+lPGA(g)"])
fig.savefig(f"100result/{folder}({id})/gmv_PGA.png",dpi=300)
fig.show()
