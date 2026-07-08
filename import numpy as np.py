import numpy as np
from scipy.optimize import root_scalar
import argparse 
import sys


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



if __name__ == '__main__':
    if len(sys.argv) < 5 or sys.argv[1]=='-h':
        print('Nargs=', len(sys.argv))
        print('\nUsage: python predict_post_shock.py <rho_bg> <rho_w> <T_bg> <T_w> <Mach> \n')
    parser = argparse.ArgumentParser(description="Predict post shock conditions.")
    parser.add_argument("rho_bg", type=float, help="Background (right side) density")
    parser.add_argument("rho_w", type=float, help="Wind (left side) density")
    parser.add_argument("T_bg", type=float, help="Background (right side) temperature")
    parser.add_argument("T_w", type=float, help="Wind (left side) temperature")
    parser.add_argument("Mach", type=float, help="Wind Mach number (vwind/c_s,w)")
    args = parser.parse_args()
    print('\n\n\n')
    rho_bg = args.rho_bg
    T_bg = args.T_bg
    rho_w = args.rho_w
    T_w = args.T_w
    Mach = args.Mach

    v_bg_shock, v_wind_shock = calc_shock_speed(rho_R=rho_bg, T_R=T_bg, rho_L=rho_w, T_L=T_w, Mach=Mach)
    vwind = Mach*np.sqrt(5/3*T_w)

    # Shock speed wrt background medium, same as shock in lab frame
    M_bg_shock = v_bg_shock/np.sqrt(5/3 * T_bg)
    bg_shock = post_shock_ratios(M_bg_shock, 5/3)

    print('\n\n\n')
    print('Background post-shock')
    print('Shock into background:', M_bg_shock, '\n') 
    print('Post shock BG temperature:', bg_shock['temperature'] * T_bg, '\n')
    print('Post shock BG density:', bg_shock['density'] * rho_bg, '\n')
    print('Post shock BG pressure', bg_shock['pressure'] * rho_bg * T_bg, '\n')


    # Converting wind shock into wind frame
    v_left_shock_wind_frame = vwind - v_wind_shock

    M_wind_shock = v_left_shock_wind_frame/np.sqrt(5/3*T_w)

    wind_shock = post_shock_ratios(M_wind_shock, 5/3)
    
    print('\n\n\n')
    print('Wind post-shock')
    print('Shock into wind:', M_wind_shock, '\n')
    print('Post shock wind temperature:', wind_shock['temperature'] * T_w, '\n')
    print('Post shock wind density:', wind_shock['density'] * rho_w, '\n')
    print('Post shock BG pressure:', wind_shock['pressure'] * rho_w * T_w, '\n')