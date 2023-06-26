from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\plot_2d")
from plot_2d import FigPlot, clip_df




df = clip_df(pd.read_csv('./Ansys_sampled.csv'), min_x=1.0, max_x=5.0, key='x')
xs = df.loc[:, 'x']
vs = df.loc[:, 'stress_y']


fig_plot = FigPlot(figsize=(4,3), fontsize=12)
fig_plot.ax.set_xlim(0.5, 3)
fig_plot.ax.set_xlabel(r'$x/R$ [m]')
fig_plot.ax.set_ylabel(r'$\sigma_{yy}$ [Pa]')


fig_plot.ax.plot(xs_FEM, vs_FEM, label='FEM', c='tab:gray', ls=':')

fig_plot.show()
