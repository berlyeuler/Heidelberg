import pandas as pd
import numpy as np
import sys
from shattering import basic_functions as bf
sys.path.append('/Users/cdoughty/research/athena-ccd-maxbg-bssdl-fork/')
from predict_post_shock import calc_shock_speed
from matplotlib import pyplot as plt


# layer cross-section, set by sim dimensions, y by z
# For the 2D sims
sigma_layer = 16 * 1
# For the 3D sims
sigma_layer = 16 * 16

sims = pd.read_csv('2d_grid.csv')

fig, axes = plt.subplots(ncols=2)
ax = axes[0]  # mass evolution
ax1 = axes[1]  # growth rate estimate

skip = ['try1', 'try6']

for i in range(len(sims)): 
    sim = sims.iloc[i].squeeze()
    if sim['description'] in skip:
        continue
    vshock_bg, _ = calc_shock_speed(sim['rho_bg'], sim['T_bg'], sim['rho_wind'], sim['T_wind'], sim['M_wind'])
    ratios = bf.post_shock_ratios(vshock_bg/np.sqrt(5/3.*sim['T_bg']), 5/3.)
    rho_bg_post = ratios['density'] * sim['rho_bg']
    tcool = bf.tcool_dimless(ratios['temperature']*sim['T_bg'], rho_bg_post, 'sutherland')
    Mres = vshock_bg * tcool * sigma_layer * rho_bg_post  # shocked mass reservoir

    print(f"sim {sim['description']} mass reservoir estimate: {Mres:.4f}")

    times = np.linspace(0, 3, 50)
    Mest = Mres * (1-np.exp(-1.0*times/tcool))

    ax.plot(times, Mest, label=sim['description'])
    Mest_dot = Mres / tcool * np.exp(-times/tcool)
    ax1.plot(times, Mest_dot)

ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('cold mass estimate')
ax1.set_ylabel('rate growth estimate')

plt.show()

