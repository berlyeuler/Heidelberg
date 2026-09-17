import os
import sys
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import yt

# --- EŞİK DEĞERLERİ (Hocanın önerisi: düşür) ---
T_TRESH = 50.0  # Eskiden 150.0'dı, çok yüksekti
D_TRESH = 200.0  # Eskiden 500.0'dı, çok yüksekti

# Ayrışma eşiği (32/512 = 0.0625)
SEPERATION_THRESHOLD = 32 / 512

sys.path.append("/pc2/users/h/hpccado/repos/athena-ccd-maxbg-bssdl-fork/")
from predict_post_shock import calc_shock_speed

output_frames_dir = "frames_tracked_vs_predicted"
os.makedirs(output_frames_dir, exist_ok=True)

# --- PARAMETRELER ---
rho_bg, rho_wind, T_bg, T_wind, M_wind = [38.135, 12.245, 41.436, 98.271, 2.928]
vshock_bg_lab, vshock_wind_lab, v_shocked_bg = calc_shock_speed(
    rho_bg, T_bg, rho_wind, T_wind, M_wind
)

# --- SİMÜLASYON VERİLERİ ---
sim_dir = "/scratch/hpc-prf-radmix/hpcbeoe/sim_15_try2"
snaps = sorted(glob(f"{sim_dir}/id0/*.vtk")) + sorted(glob(f"{sim_dir}/*.vtk"))
if not snaps:
    snaps = sorted(glob(f"{sim_dir}/id0/*.athdf")) + sorted(glob(f"{sim_dir}/*.athdf"))

print(f"📁 Bulunan snapshot sayısı: {len(snaps)}")

var = "temperature"
T_min = 6.97436478913788e-09

# --- FONKSİYON (SADECE SAĞ KENAR, SOL KENAR YOK) ---
def make_comparison_frame(ds, variable, vshock_bg_lab, counter):
    time = float(ds.current_time)
    
    slc = yt.SlicePlot(ds, "z", ("gas", variable))
    slc_data = ds.slice("z", 0.0)
    
    x_arr = slc_data["gas", "x"].v
    y_arr = slc_data["gas", "y"].v
    temp_norm = slc_data["gas", "temperature"].v / T_min
    density_arr = slc_data["gas", "density"].v 

    # y=0 civarından tek bir satır al
    center_mask = np.abs(y_arr) < 0.1
    x_line = x_arr[center_mask]
    temp_line = temp_norm[center_mask]
    density_line = density_arr[center_mask]

    # --- SADECE SAĞ KENARLARI BUL (LEFT YOK) ---
    # Sıcaklık için sağ kenar
    temp_above = temp_line > T_TRESH
    if np.any(temp_above):
        temp_indices = np.where(temp_above)[0]
        right_temp_pos = x_line[temp_indices[-1]]  # En sağdaki sıcak nokta
        print(f"  ✅ Temp found at x={right_temp_pos:.4f}")
    else:
        right_temp_pos = None
        print(f"  ❌ No temp point (max={temp_line.max():.2f} < {T_TRESH})")

    # Yoğunluk için sağ kenar
    dens_above = density_line > D_TRESH
    if np.any(dens_above):
        dens_indices = np.where(dens_above)[0]
        right_dens_pos = x_line[dens_indices[-1]]  # En sağdaki yoğun nokta
        print(f"  ✅ Density found at x={right_dens_pos:.4f}")
    else:
        right_dens_pos = None
        print(f"  ❌ No density point (max={density_line.max():.2f} < {D_TRESH})")

    # --- AYRIŞMA MESAFESİ VE ŞOK DURUMU ---
    if right_temp_pos is not None and right_dens_pos is not None:
        separation = abs(right_temp_pos - right_dens_pos)
        shock_active = separation <= SEPERATION_THRESHOLD
        
        if shock_active:
            x_tracked = (right_temp_pos + right_dens_pos) / 2
            print(f"  ✅ SHOCK ACTIVE! x={x_tracked:.4f}")
        else:
            x_tracked = right_dens_pos
            print(f"  ❌ SHOCK DISSIPATED (sep={separation:.4f} > {SEPERATION_THRESHOLD:.4f})")
    else:
        # Eksik veri varsa bu frame'i atla
        print(f"  ⚠️ SKIPPING frame {counter}: missing data")
        return time, np.nan, np.nan, vshock_bg_lab * time, False

    print(f"{'='*60}\n")
    
    # --- ÇİZGİLERİ ÇİZ ---
    x_predicted = vshock_bg_lab * time
    
    # Teorik tahmin (CYAN)
    slc.annotate_line(
        (x_predicted, 8.0, 0.0), (x_predicted, -8.0, 0.0),
        coord_system="data", color="cyan"
    )

    # Takip edilen şok (KIRMIZI) - SADECE AKTİFSE
    if shock_active:
        slc.annotate_line(
            (x_tracked, 8.0, 0.0), (x_tracked, -8.0, 0.0),
            coord_system="data", color="red",
            plot_args={"linestyle": "--", "linewidth": 2.5}
        )
        print(f"  🔴 Red line drawn at x={x_tracked:.3f}")

    # Başlangıç sınırı (BEYAZ)
    slc.annotate_line(
        (0.0, 8.0, 0.0), (0.0, -8.0, 0.0),
        coord_system="data", color="w"
    )

    slc.render()

    # --- LEJANT ---
    track_label = f"Shock Front ({x_tracked:.2f})" if shock_active else "Shock Dissipated"
    legend_lines = [
        Line2D([0], [0], color="cyan", linewidth=2, label=f"Predicted ({x_predicted:.2f})"),
        Line2D([0], [0], color="red" if shock_active else "gray", linewidth=2, 
               linestyle="--" if shock_active else "-", label=track_label),
        Line2D([0], [0], color="w", linewidth=1.5, label="Initial Boundary"),
    ]

    ax = slc.plots[("gas", variable)].axes
    ax.legend(handles=legend_lines, loc="upper right", framealpha=0.85)

    if variable == "density":
        slc.set_zlim(("gas", "density"), 10, 1000)
    elif variable == "temperature":
        slc.set_zlim(("gas", "temperature"), T_min, 200 * T_min)

    p = slc.plots[("gas", variable)]
    frame_name = os.path.join(output_frames_dir, f"frame_{variable}_{counter:04d}.png")
    p.figure.savefig(frame_name, bbox_inches="tight", dpi=100)
    p.figure.set_size_inches(10, 5)  # Sabit boyut
    
    # --- LOG ---
    shock_str = f"{x_tracked:.2f}" if shock_active else "DISSIPATED"
    print(f"Frame {counter:02d} | Time: {time:.3f} | "
          f"Shock: {shock_str} | Pred: {x_predicted:.2f} | Active: {'YES' if shock_active else 'NO'}")
    
    return time, separation, x_tracked, x_predicted, shock_active


# --- ANA DÖNGÜ ---
times = []
separations = []
shock_positions = []
predicted_positions = []
shock_status = []

counter = 0
for snap in snaps[:40]:
    ds = yt.load(snap)
    result = make_comparison_frame(ds, var, vshock_bg_lab, counter)
    
    time, separation, x_tracked, x_pred, active = result
    
    # Sadece geçerli frame'leri kaydet (None/nan olmayanlar)
    if not np.isnan(separation):
        times.append(time)
        separations.append(separation)
        shock_positions.append(x_tracked)
        predicted_positions.append(x_pred)
        shock_status.append(active)
    
    counter += 1

# --- PLOT 1: SEPARATION VS TIME ---
if times:
    plt.figure(figsize=(12, 6))
    plt.plot(times, separations, 'r-o', label='Separation', linewidth=2, markersize=6)
    plt.axhline(y=SEPERATION_THRESHOLD, color='k', linestyle='--', 
                label=f'Threshold ({SEPERATION_THRESHOLD:.4f})', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Separation (x position difference)', fontsize=12)
    plt.title('Shock Front Separation vs Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('separation_vs_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Plot saved as 'separation_vs_time.png'")

# --- PLOT 2: SHOCK POSITION VS TIME ---
if times:
    plt.figure(figsize=(12, 6))
    plt.plot(times, shock_positions, 'r-o', label='Tracked Shock', linewidth=2, markersize=6)
    plt.plot(times, predicted_positions, 'c-s', label='Predicted Shock', linewidth=2, markersize=6)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Shock Position (x)', fontsize=12)
    plt.title('Shock Position: Tracked vs Predicted', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('shock_position_vs_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Plot saved as 'shock_position_vs_time.png'")

# --- VİDEO OLUŞTUR ---
frames = sorted(glob(f"{output_frames_dir}/*{var}*.png"))

if frames:
    movie_name = f"sim_15_try2_tracked_vs_predicted_{var}.mp4"
    W, H = 1040, 496

    print(f"\n🎬 Video oluşturuluyor... ({len(frames)} frame)")
    with imageio.get_writer(movie_name, fps=8) as writer:
        for f in frames:
            img = imageio.imread(f)
            # Boyut kontrolü ve yeniden boyutlandırma
            if img.shape[1] != W or img.shape[0] != H:
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((W, H), Image.Resampling.LANCZOS)
                img = np.array(img_pil)
            writer.append_data(img)

    print(f"\n✅ TAMAM! Video: {movie_name}")
    print(f"📊 Grafikler: separation_vs_time.png ve shock_position_vs_time.png")