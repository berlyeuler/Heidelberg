from scipy.stats import qmc
import numpy as np
import os
import sys
from cooling_time import calculate_cooling_time
from matplotlib import pyplot as plt


# path to athena-ccd-... directory to get the predict_post_shock code
# CHANGE THIS PATH TO LEAD TO YOUR athena-ccd-maxbg-bssdl-fork directory!!
sys.path.append("/Users/cdoughty/research/athena-ccd-maxbg-bssdl-fork")

from predict_post_shock import calc_shock_speed, post_shock_ratios

# Use Latin Hypercube to sample randomly in N-D space
nvars = 5  # five variables
nsamples = 30 # number of samples
boxlength = 32

# Make samples
sampler = qmc.LatinHypercube(d=nvars, seed=34)
Xinit = sampler.random(n=nsamples)

# Set for cooling function (constant for now)
lambda_rescale = 1.0

# Selected limits on the variable values
# rho_bg(0), rho_w(1), T_bg(2), T_w(3), Mach(4)
lower = [1.0, 1.0, 1.0, 1.0, 1.1]
upper = [50.0, 50.0, 20.0, 100.0, 5.0]

X = qmc.scale(Xinit, lower, upper)
print('Shape of test variables:', X.shape)

# Sound speeds for background and wind
cs_bg = np.sqrt(5./3*X[:,2])
cs_wind = np.sqrt(5./3*X[:,3])


# Calculate the cooling time for these samples
tcool_dict = {'bg': [], 'wind': []}
vel_dict = {'shock_bg':[], 'shock_wind':[], 'wind':[]}

test_fig, test_ax = plt.subplots()

for i in range(nsamples):
    rho_bg, rho_w, T_bg, T_w, Mwind = X[i]
    
    # Velocities in lab frame
    v_sbg, v_sw_lab = calc_shock_speed(rho_bg, T_bg, rho_w, T_w, Mwind)
    vwind = Mwind * cs_wind[i]

    # Adjust wind shock velocity into wind frame
    v_sw = vwind - v_sw_lab
    mach_dict = {'bg': v_sbg/cs_bg[i], 'wind': v_sw/cs_wind[i]}

    # I think these should all be in lab frame...
    vel_dict['shock_bg'].append(v_sbg)
    vel_dict['shock_wind'].append(v_sw_lab)
    vel_dict['wind'].append(vwind)

    tdict = {'bg': T_bg, 'wind': T_w}
    ddict = {'bg': rho_bg, 'wind': rho_w}

    test_ax.scatter(Mwind, mach_dict['bg'], color='k')
    
    for key in tcool_dict.keys():
        r = post_shock_ratios(mach_dict[key], 5/3)
        ps_density = r['density'] * ddict[key]
        ps_temperature = r['temperature'] * tdict[key]
        tcool = calculate_cooling_time(ps_density, ps_temperature, gamma=5/3, small_lambda=lambda_rescale)
        tcool_dict[key].append(tcool)
test_ax.set_xlabel(r'$M_{\rm w}$')
test_ax.set_ylabel(r'$M_{\rm sh}$')

#plt.savefig(f"{os.environ['HOME']}/Ms_vs_Mw_diff_dens_temp.png", bbox_inches='tight', dpi=200)

total_good_samples = 0
#sys.exit()
for_print = []

for i in range(nsamples):
    
    tcools = [tcool_dict['bg'][i], tcool_dict['wind'][i]]
    smallest_tcool = min(tcools)
    where = ['BG', 'WIND']

    tlim = 20 * smallest_tcool
    dt = tlim/100.0

    wind_tcross = boxlength / vel_dict['wind'][i]
    shock_wind_tcross = boxlength / vel_dict['shock_wind'][i]
    shock_bg_tcross = boxlength / vel_dict['shock_bg'][i]

    #print('tcross wind', wind_tcross)
    #print('tcross shock wind', shock_wind_tcross)
    #print('tcross shock bg', shock_bg_tcross)

    tcross_flag = 0
    if wind_tcross < 10 * dt:
        #print('Wind too fast for cooling time+box size.')
        tcross_flag = 1
    elif shock_wind_tcross < 10 * dt:
        #print('Wind shock too fast for cooling time+box size.')
        tcross_flag = 1
    elif shock_bg_tcross < 10 * dt:
        #print('Background shock too fast for cooling time+box size.')
        tcross_flag = 1

    if tcross_flag:
        print(f'\nindex #{i}\n###############')
        print(f'\nSkipping index #{i} for too-fast t_cross.\n')
        continue

    total_good_samples += 1
    for_print.append(X[i])

    print(f'\nindex #{i}\n###############')
    print(f'Smallest tcool is {smallest_tcool:.2e} (in {where[np.argmin(tcools)]})')
    print(f'<output2>')
    print('.\n.')
    print(f'dt = {dt:.4f}\n')
    
    print(f'<time>')
    print('.\n.')
    print(f'tlim = {tlim:.4f}\n')

    print('<problem>')
    rho_bg, rho_w, T_bg, T_w, Mwind = X[i]
    print(f'rho_bg = {rho_bg:.2f}')
    print(f'rho_wind = {rho_w:.2f}')
    print(fr'T_bg = {T_bg:.2f}')
    print(f'T_wind = {T_w:.2f}')
    print(fr'M_wind = {Mwind:.2f}')


print('\n###############Total good samples=', total_good_samples, '\n\n')


print('###############For job array batch code.')

for_print = np.array(for_print).T
varnames = ['rho_bg', 'rho_w', 'T_bg', 'T_w', 'Mwind']

for i in range(len(varnames)):
    print_strs = [f"{x:.3f}" for x in for_print[i]]
    print(f"{varnames[i]}=(", ' '.join(print_strs), ')')
    #print('drat_list=(', ' '.join(chi_input_flat[::-1]), ')')


# IGNORE
"""fig, ax = plt.subplots()

ax.scatter(np.arange(nsamples), tcool_dict['bg'], label='BG')
ax.scatter(np.arange(nsamples), tcool_dict['wind'], label='wind')

ax.set_xlabel('index of sample')
ax.set_ylabel(r'$t_{\rm cool}$')
ax.set_yscale('log')
ax.legend()

plt.savefig(f'{os.environ["HOME"]}/from_LHC.png', bbox_inches='tight', dpi=300)
plt.show()"""

