"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

import pydmps
import pydmps.DMP_Discrete

y_des = np.load("2.npz")["arr_0"].T
y_des -= y_des[:, 0][:, None]

# test normal run
dmp = pydmps.DMP_Discrete(nDMPs=2, nBFs=500, ay=np.ones(2) * 10.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitatePath(yDesired=y_des, isPlot=False)
y_track, dy_track, ddy_track = dmp.rollout()
plt.figure(1, figsize=(6, 6))

plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
plt.title("DMP system - draw number 2")

plt.axis("equal")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
