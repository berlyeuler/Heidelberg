import os
import sys
from glob import glob
import imageio
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
import yt
from scipy.optimize import root_scalar

# ============================================
# predict_post_shock.py'deki fonksiyonları kopyala
# ============================================

def post_shock_ratios(mach_number, gamma):
    """Calculate post-shock density, pressure, temperature ratios."""
    d2_d1 = (gamma + 1) * mach_number**2 / (2 + (gamma-1)*mach_number**2)
    p2_p1 = (2 * gamma * mach_number ** 2 - (gamma - 1)) / (gamma + 1)
    t2_t1 = ((2 * gamma * mach_number ** 2 - (gamma - 1)) * (2 + (gamma - 1) * mach_number ** 2) /
             ((gamma + 1) ** 2 * mach_number ** 2))
    return {'density': d2_d1, 'pressure': p2_p1, 'temperature': t2_t1}


def sound_speed(gamma, P, rho):
    return np.sqrt(gamma * P / rho)


def shock_function(P_star, P, rho, gamma):
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
    fL = f_side(P_star, P_Li, rho_Li, gamma)
    fR = f_side(P_star, P_Ri, rho_Ri, gamma)
    return fL + fR + (vRi - vLi)


def solve_riemann(vL, rho_L, P_L, vR, rho_R, P_R, gamma=5/3):
    P_min = 1e-8 * min(P_L, P_R)
    P_max = 1e4 * max(P_L, P_R)

    sol = root_scalar(objective, bracket=[P_min, P_max], args=(
        P_L, P_R, rho_L, rho_R, vL, vR, gamma), method='brentq')
    P_star = sol.root

    fL = f_side(P_star, P_L, rho_L, gamma)
    fR = f_side(P_star, P_R, rho_R, gamma)

    u = 0.5 * (vL + vR) + 0.5 * (fR - fL)

    if P_star > P_R:
        cR = sound_speed(gamma, P_R, rho_R)
        M_R = np.sqrt((P_star / P_R * (gamma + 1) + (gamma - 1)) / (2 * gamma))
        v_s_R = vR + M_R * cR
    else:
        cR = sound_speed(gamma, P_R, rho_R)
        v_s_R = vR + cR
        v_s_R = np.nan

    if P_star > P_L:
        cL = sound_speed(gamma, P_L, rho_L)
        M_L = np.sqrt((P_star / P_L * (gamma + 1) + (gamma - 1)) / (2 * gamma))
        v_s_L = vL - M_L * cL
    else:
        cL = sound_speed(gamma, P_L, rho_L)
        v_s_L = vL - cL
        v_s_L = np.nan

    return {
        "v_shocked_R": u,
        "vshock_R_lab_frame": v_s_R,
        "vshock_L_lab_frame": v_s_L
    }


def calc_shock_speed(rho_R, T_R, rho_L, T_L, Mach):
    """Calculate shock speeds using Riemann solver."""
    vL = Mach * np.sqrt(5/3*T_L)  # wind velocity
    P_L = T_L*rho_L
    
    vR = 0.0
    P_R = T_R * rho_R
    
    result = solve_riemann(vL, rho_L, P_L, vR, rho_R, P_R)
    
    return result['vshock_R_lab_frame'], result['vshock_L_lab_frame'], result['v_shocked_R']

# ============================================
# Ana kod buradan devam ediyor
# ============================================

output_frames_dir = "frames_tracked_vs_predicted"
os.makedirs(output_frames_dir, exist_ok=True)

# --- PARAMETRELER & TEORİK ŞOK HIZI ---
rho_bg, rho_wind, T_bg, T_wind, M_wind = [
    38.135,
    12.245,
    41.436,
    98.271,
    2.928,
]

# Şok hızını hesapla 
vshock_bg_lab, vshock_wind_lab, v_shocked_bg = calc_shock_speed(
    rho_bg, T_bg, rho_wind, T_wind, M_wind
)

print(f"Hesaplanan şok hızları:")
print(f"  vshock_bg_lab: {vshock_bg_lab}")
print(f"  vshock_wind_lab: {vshock_wind_lab}")
print(f"  v_shocked_bg: {v_shocked_bg}")

sim_dir = "/scratch/hpc-prf-radmix/hpcbeoe/sim_15_try2"
snaps = sorted(glob(f"{sim_dir}/id0/*.vtk")) + sorted(
    glob(f"{sim_dir}/*.vtk")
)
if not snaps:
    snaps = sorted(glob(f"{sim_dir}/id0/*.athdf")) + sorted(
        glob(f"{sim_dir}/*.athdf")
    )

var = "temperature"
T_min = 6.97436478913788e-09  # Kelvin cinsinden minimum sıcaklık
T_THRESHOLD = 50.0  # Normalize edilmiş eşik değeri

# Domain sınırları (otomatik al)
first_ds = yt.load(snaps[0])
X_MIN_DOMAIN = float(first_ds.domain_left_edge[0])
X_MAX_DOMAIN = float(first_ds.domain_right_edge[0])
print(f"Domain: X = [{X_MIN_DOMAIN}, {X_MAX_DOMAIN}]")


def make_comparison_frame(ds, variable, vshock_bg_lab, counter):
    time = float(ds.current_time)
    slc = yt.SlicePlot(ds, "z", ("gas", variable))

    slc_data = ds.slice("z", 0.0)
    x_arr = slc_data["gas", "x"].v
    y_arr = slc_data["gas", "y"].v

    # Ham sıcaklık verisi (Kelvin cinsinden)
    temp_raw = slc_data["gas", "temperature"].v
    
    # NORMALİZE ET: Kelvin'den kod birimine çevir
    temp_norm = temp_raw / T_min

    # y=0 etrafındaki geniş kesiti al (y: -2 ile +2 arası)
    center_mask = np.abs(y_arr) < 2.0
    x_c = x_arr[center_mask]
    temp_c = temp_norm[center_mask]

    # Sadece teorik şokun etrafındaki mantıklı bölgeyi tara
    x_predicted_curr = vshock_bg_lab * time
    valid_domain = (x_c <= (x_predicted_curr + 2.0)) & (x_c >= X_MIN_DOMAIN)

    x_valid = x_c[valid_domain]
    temp_valid = temp_c[valid_domain]

 # --- ŞOK TESPİTİ (Sıcaklık Sıçraması / Gradyan Kontrolü) ---
    # 1. Potansiyel sıcak bölgeleri tespit et
    candidate_indices = np.where(temp_valid > T_THRESHOLD)[0]

    x_tracked = np.nan

    if len(candidate_indices) > 0:
        # En sağdaki (en öndeki) adayı seç
        last_idx = candidate_indices[-1]

        # Cephenin hemen arkasındaki ve önündeki sıcaklığı karşılaştır (Sıçrama kontrolü)
        # last_idx - 5 (şok arkası) ve last_idx + 5 (şok önü)
        idx_back = max(0, last_idx - 5)
        idx_front = min(len(temp_valid) - 1, last_idx + 5)

        T_back = temp_valid[idx_back]
        T_front = temp_valid[idx_front]

        # Gerçek bir şok cephesi için arkadaki gazın önündekinden bariz sıcak olması gerekir
        # Örneğin: T_back / T_front > 1.8 (Bu oran soğuma olunca düşer)
        if T_front > 0 and (T_back / T_front) > 1.8:
            x_tracked = x_valid[last_idx]
        else:
            # Şok sıçraması yoksa/soğuduysa takibi kes
            x_tracked = np.nan

    # Debug bilgisi
    if not np.isnan(x_tracked):
        print(f"    Şok tespit edildi: {x_tracked:.2f}")
    else:
        print(f"    Şok sönümlendi / tespit edilemedi (x_tracked = NaN)")

    # --- ANNOTATIONS ---
    x_predicted = vshock_bg_lab * time
    slc.annotate_line(
        (x_predicted, 8.0, 0.0),
        (x_predicted, -8.0, 0.0),
        coord_system="data",
        color="cyan",
        plot_args={"linewidth": 2.5}
    )

    if not np.isnan(x_tracked):
        slc.annotate_line(
            (x_tracked, 8.0, 0.0),
            (x_tracked, -8.0, 0.0),
            coord_system="data",
            color="red",
            plot_args={"linestyle": "--", "linewidth": 2.5},
        )

    slc.annotate_line(
        (0.0, 8.0, 0.0), (0.0, -8.0, 0.0), coord_system="data", color="w"
    )

    slc.render()

    # LEJANT
    legend_lines = [
        Line2D([0], [0], color="cyan", linewidth=2,
               label=f"Predicted Shock ({x_predicted:.2f})"),
        Line2D([0], [0], color="red", linewidth=2, linestyle="--",
               label=(f"Tracked Shock ({x_tracked:.2f})"
                      if not np.isnan(x_tracked) else "Tracked: None")),
        Line2D([0], [0], color="w", linewidth=1.5,
               label="Init. Wind-BG Boundary"),
    ]

    ax = slc.plots[("gas", variable)].axes
    ax.legend(handles=legend_lines, loc="upper right", framealpha=0.85)

    if variable == "temperature":
        slc.set_zlim(("gas", "temperature"), T_min, 200 * T_min)

    p = slc.plots[("gas", variable)]
    frame_name = os.path.join(
        output_frames_dir, f"frame_{variable}_{counter:04d}.png"
    )
    p.figure.savefig(frame_name, bbox_inches="tight")

    print(
        f"Frame {counter:02d} | Time: {time:.3f} | "
        f"Tracked: {x_tracked if np.isnan(x_tracked) else f'{x_tracked:.2f}'} | "
        f"Predicted: {x_predicted:.2f}"
    )

    return x_tracked


# --- ANA DÖNGÜ ---
counter = 0

for snap in snaps:
    ds = yt.load(snap)
    x_tr = make_comparison_frame(ds, var, vshock_bg_lab, counter)

    # Hocanın istediği tek kural:
    # x_tracked (yani x_tr) NaN döndüğü an takibi kes ve videoyu bitir.
    if counter > 5 and np.isnan(x_tr):
        print(
            f"\n⛔ Şok cephesi sönümlendi (x_tracked = NaN)! Video {counter}. frame'de kesiliyor."
        )
        break

    counter += 1

# --- VİDEO OLUŞTURMA ---
frames = sorted(glob(f"{output_frames_dir}/*{var}*.png"))

if frames:
    movie_name = f"sim_15_try2_tracked_vs_predicted_{var}_cut.mp4"
    #b
    W, H = 1040, 496

    with imageio.get_writer(movie_name, format="FFMPEG", fps=8) as writer:
        for f in frames:
            img = Image.open(f)
            img_resized = img.resize((W, H))
            writer.append_data(np.array(img_resized))

    print(f"\n🎬 VİDEO TAMAMLANDI! Çıktı: {movie_name}")
    print(f"Toplam frame sayısı: {len(frames)}")
else:
    print("❌ Hiç frame oluşturulmadı!")