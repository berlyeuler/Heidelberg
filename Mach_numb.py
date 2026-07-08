
"""
Bu kod initial density ve temperature değerlerini bulmak için fsolve kullanır.
Bu değerler, belirli bir Mach sayısı ve hedef post-shock yoğunluk ve sıcaklık değerleri ile uyumlu olacak şekilde hesaplanır.
Kullanıcı, hedef post-shock yoğunluk ve sıcaklık değerlerini n_f_target ve T_f_target değişkenlerinde belirleyebilir. Kod, Rankine-Hugoniot koşullarını 
ve Riemann problemini çözerek gerekli başlangıç değerlerini bulur.  


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt


import scipy.constants as const

import numpy as np
from scipy.optimize import fsolve, root_scalar
import argparse 
import sys

n_f_target= 9.87
T_f_target= 21.0
gamma= 5/3
rho_w_fixed = 1.0
T_w_fixed = 100.0
Mach_fixed = 2.5


def post_shock_ratios(mach_number, gamma):
    """
    Calculate the post-shock density, pressure, and temperature ratios for a given Mach number and gamma.

    Parameters
    ----------
    mach_number : float
        The Mach number of the flow.

    gamma : float
        The adiabatic index of the gas.

    Returns
    -------
    post_shock_pressure_ratio : float
        The ratio of the post- and pre-shock pressures.
    """

    d2_d1 = (gamma + 1) * mach_number**2 / (2 + (gamma-1)*mach_number**2)
    p2_p1 = (2 * gamma * mach_number ** 2 - (gamma - 1)) / (gamma + 1)
    t2_t1 = ((2 * gamma * mach_number ** 2 - (gamma - 1)) * (2 + (gamma - 1) * mach_number ** 2) /
             ((gamma + 1) ** 2 * mach_number ** 2))

    return_dict = {'density': d2_d1, 'pressure': p2_p1, 'temperature': t2_t1}

    return return_dict


def sound_speed(gamma, P, rho):
    return np.sqrt(gamma * P / rho)

def shock_function(P_star, P, rho, gamma):
    # Velocity change across a shock
    A = 2 / ((gamma + 1) * rho)
    B = (gamma - 1) / (gamma + 1) * P
    return (P_star - P) * np.sqrt(A / (P_star + B))

def rarefaction_function(P_star, P, rho, gamma):
    c = sound_speed(gamma, P, rho)
    return (2 * c / (gamma - 1)) * ((P_star / P)**((gamma - 1) / (2 * gamma)) - 1)

def f_side(P_star, P, rho, gamma):
    if P_star > P:
        return shock_function(P_star, P, rho, gamma)
    else:
        return rarefaction_function(P_star, P, rho, gamma)


def objective(P_star, P_Li, P_Ri, rho_Li, rho_Ri, vLi, vRi, gamma):
    # Find velocity changes between each side and the 
    # contact discontinuity (CD)
    # In units of CD pressure P_star
    fL = f_side(P_star, P_Li, rho_Li, gamma)
    fR = f_side(P_star, P_Ri, rho_Ri, gamma)
    return fL + fR + (vRi - vLi)


def solve_riemann(vL, rho_L, P_L, vR, rho_R, P_R, gamma=5/3):

    """def objective(P_star):
        # Find velocity changes between each side and the 
        # contact discontinuity (CD)
        # In units of CD pressure P_star
        fL = f_side(P_star, P_L, rho_L, gamma)
        fR = f_side(P_star, P_R, rho_R, gamma)
        return fL + fR + (vR - vL)"""

    # Bracket for root finding
    P_min = 1e-8 * min(P_L, P_R)
    P_max = 1e4 * max(P_L, P_R)

    sol = root_scalar(objective, bracket=[P_min, P_max], args=(
        P_L, P_R, rho_L, rho_R, vL, vR, gamma), method='brentq')
    P_star = sol.root
    print('Contact discontinuity pressure=', P_star)

    # Post-contact velocity
    fL = f_side(P_star, P_L, rho_L, gamma)
    fR = f_side(P_star, P_R, rho_R, gamma)

    #print('result:', fL+fR + (vR-vL))
    u = 0.5 * (vL + vR) + 0.5 * (fR - fL)

    # Right-going wave speed
    if P_star > P_R:
        # Shock into right medium
        print('Right wave is shock')
        cR = sound_speed(gamma, P_R, rho_R)
        M_R = np.sqrt((P_star / P_R * (gamma + 1) + (gamma - 1)) / (2 * gamma))
        v_s_R = vR + M_R * cR
    else:
        # Rarefaction (head speed)
        print('Right wave is rarefaction')
        print('Shock jump predictions will be wrong. :(')
        cR = sound_speed(gamma, P_R, rho_R)
        v_s_R = vR + cR

    # Left-going wave speed
    if P_star > P_L:
        print('Left wave is shock')
        # Shock into left medium
        cL = sound_speed(gamma, P_L, rho_L)
        M_L = np.sqrt((P_star / P_L * (gamma + 1) + (gamma - 1)) / (2 * gamma))
        v_s_L = vL - M_L * cL
    else:
        print('Left wave is rarefaction')
        print('Shock jump predictions will be wrong. :(')
        # Rarefaction (head speed)
        cL = sound_speed(gamma, P_L, rho_L)
        v_s_L = vL - cL

    return {
        #"P_star": P_star,
        #"u": u,
        "vshock_R_lab_frame": v_s_R,
        "vshock_L_lab_frame": v_s_L
    }
#v_bg_shock

# Using the Tcl and chicl (since those are important settings), but it's not the velocity
# in the cloud, or anything like that.
def calc_shock_speed(rho_R, T_R, rho_L, T_L, Mach):
    """
    Parameters:
    -----------
    rho_R : float
        Background density ()
    T_R : float
        Background temperature (in units of 1e4 K for Benedikt's version)
    rho_L : float
        Wind density.
    rho_R : float
        Background density.

    """
    # Wind velocity
    vL = Mach * np.sqrt(5/3*T_L)  # wind velocity, Mach wrt wind temperature
    P_L = T_L*rho_L

    # Ambient medium (always stationary)
    vR = 0.0
    P_R = T_R * rho_R

    result = solve_riemann(vL, rho_L, P_L, vR, rho_R, P_R)

    return result['vshock_R_lab_frame'], result['vshock_L_lab_frame']


#sıfıra eşitleyen kodu yazalım
def system_to_solve(guess):
    #guess fsolveun her işlemde deneyeceği initial tahminleri
    rho_bg_guess, T_bg_guess = guess
    if rho_bg_guess <= 0 or T_bg_guess <= 0:
        return [1e6, 1e6]
    #şok hızını (velocity) hesaplayalım
    v_bg_shock, _ = calc_shock_speed(
            rho_R=rho_bg_guess, 
            T_R=T_bg_guess, 
            rho_L=rho_w_fixed, 
            T_L=T_w_fixed, 
            Mach=Mach_fixed)
    
    #Ms shock formulünü yazalım
    c_sound_bg = np.sqrt(gamma * T_bg_guess) # Arka plan ses hızı
    M_s = v_bg_shock / c_sound_bg

    #hazır koddan post shock oranlarını alalım
    ratios= post_shock_ratios(M_s, gamma)
    R_n= ratios['density']
    R_T= ratios['temperature']
    eq1= n_f_target - R_n * rho_w_fixed
    eq2= T_f_target - R_T * T_w_fixed
    return [eq1, eq2]

    #rastgele başlangıç değerleri verelim 
initial_guess = [2.0, 3.0]  # Örnek başlangıç değerleri
print("initial is being estimated")
solution, infodict, ier, mesg = fsolve(system_to_solve, initial_guess, full_output=True)
#solution: fsolve tarafından bulunan çözüm [n_initial, T_initial]
#infodict: fsolve tarafından sağlanan ek bilgiler
#ier: fsolve'un durumu hakkında bilgi (1: başarılı, 2: başarısız, vb.)
#mesg: fsolve tarafından sağlanan mesaj (başarı veya hata hakkında bilgi)
if ier == 1:
    print(f"n_initial = {solution[0]}")
    print(f"T_initial = {solution[1]}")
else:
    print("error:", mesg)








