import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

import glob
import os 
import imageio

# Terminalden çıkan tam yolu buraya yapıştıracaksın:
#ds = yt.load('/scratch/hpc-prf-radmix/hpcbeoe/test_beril/id0/cloud.0100.vtk')
#ad= ds.all_data()


M1 = 8.9887
gamma = 5 / 3 
rho1 = 1.0
T = 10.0
small_lambda = 1.0


def get_simulation_data(ad):
    density = ad[("athena", "density")]
    pressure = ad[("athena", "pressure")]
    
    # Sıcaklık simülasyonda doğrudan yoksa: T = P / rho
    temperature = pressure / density
    
    return density, pressure, temperature
@np.vectorize
def sutherland_dopita_cooling_rate_function(temperature_dimless):
    # result is in units of 1e-23 erg cm^3 / s
    # piecewise power law fit of Sutherland & Dopita (1993) cooling function

    normalization = 1e4  # Don't change, helps to deal with units
    temperature = temperature_dimless * normalization * 8.61733e-8  # convert from K to keV
    
    if temperature < 1.0e-5:  # left of fit region
        return 5.890872e-13 * (temperature/1.0e-5)**6.0  # extrapolate
    elif temperature < 0.0017235:  # bin 1
        return 5.890872e-13 * (temperature / 1.0e-5) ** 6.0
    elif temperature < 0.02:   # bin 2
        return 15.438249 * (temperature/0.0017235)**0.6
    elif temperature < 0.13:   # bin 3
        return 66.831473 * (temperature/0.02)**-1.7
    elif temperature < 0.7:  # bin 4
        return 2.773501 * (temperature/0.13)**-0.5
    elif temperature < 5.0:  # bin 5
        return 1.195229 * (temperature/0.7)**0.22
    elif temperature < 100.0: # bin 6
        return 1.842056 * (temperature/5.0)**0.4
    else:  # right of fit region
        return 6.10541 * (temperature/100.0)**0.4
    
def calculate_cooling_time(density, temperature, gamma=5.0/3.0, small_lambda=1.0):
    lamb= sutherland_dopita_cooling_rate_function(temperature)
    print('Alert! CCD added a missing term! :O ')
    # :) Missing term for conversion to code units :)
    mu_e = 1.17
    cooling_time_array = mu_e**2 * temperature / ((gamma - 1.0) * density * small_lambda * lamb)

    return cooling_time_array


rho_wind = 1.0
T_wind = 100.0

densty=((gamma+1)*(M1**2)) / (2 + (gamma-1) * (M1**2))
press= 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
T_ratio= press/ densty

pred_rho = densty*rho_wind
pred_T = T_ratio*rho_wind

t_cool = calculate_cooling_time(pred_rho, pred_T, gamma, small_lambda=small_lambda)
print("Cooling time array'i:", t_cool)



#  simülasyondan o anki yoğunluk ve sıcaklık verilerini çekiyoruz
#sim_density, sim_pressure, sim_temperature = get_simulation_data(ad)
min_rho = float(1.0)
max_rho = float(10.0)
min_T = float(1.0)
max_T = float(500.00)

y_temperature= np.linspace(min_T, max_T, num=100)
x_density= np.linspace(min_rho, max_rho, num=100 )
rho_mesh, T_mesh = np.meshgrid(x_density, y_temperature)
t_cool_mesh = calculate_cooling_time(rho_mesh, T_mesh, gamma=gamma, small_lambda=small_lambda)

plt.figure(figsize=(8, 6))
contour_plot = plt.contourf(rho_mesh, T_mesh, np.log10(t_cool_mesh), levels=20, cmap='viridis')

plt.colorbar(contour_plot, label='Log coolin Time ($t_{cool}$)')

plt.xlabel('Density (Density - $\\rho$)')
plt.ylabel('Tempetarue (Temperature - $T$)')
plt.title('Cooling time ($t_{cool}$)')


plt.show()
    
        
    
