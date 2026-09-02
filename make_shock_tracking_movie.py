import os
import sys
from glob import glob
import imageio
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
import yt

T_TRESH = 150.0
D_TRESH = 500.0

# Ayrışma eşiği (Hocanın önerisi: 32/512)
GRID_SIZE = 512 
CELL_TOLERANCE = 32 
SEPERATION_THRESHOLD = CELL_TOLERANCE / GRID_SIZE

SKIP_INITIALSNAPSHOTS = 0

sys.path.append("/pc2/users/h/hpccado/repos/athena-ccd-maxbg-bssdl-fork/")
from predict_post_shock2 import calc_shock_speed

output_frames_dir = "frames_tracked_vs_predicted"
os.makedirs(output_frames_dir, exist_ok=True)

# --- PARAMETRELER & TEORİK ŞOK HIZI HESABI ---
rho_bg, rho_wind, T_bg, T_wind, M_wind = [
    38.135,
    12.245,
    41.436,
    98.271,
    2.928,
]
vshock_bg_lab, vshock_wind_lab, v_shocked_bg = calc_shock_speed(
    rho_bg, T_bg, rho_wind, T_wind, M_wind
)

# Simülasyon Veri Seti
sim_dir = "/scratch/hpc-prf-radmix/hpcbeoe/sim_15_try2"
snaps = sorted(glob(f"{sim_dir}/id0/*.vtk")) + sorted(
    glob(f"{sim_dir}/*.vtk")
)
if not snaps:
    snaps = sorted(glob(f"{sim_dir}/id0/*.athdf")) + sorted(
        glob(f"{sim_dir}/*.athdf")
    )

print(f"📁 Bulunan snapshot sayısı: {len(snaps)}")

var = "temperature"
T_min = 6.97436478913788e-09

# --- FONKSİYONU DIŞARI TAŞI (SADECE 1 KEZ TANIMLA) ---
def make_comparison_frame(ds, variable, vshock_bg_lab, counter):
    time = float(ds.current_time)
    
    print(f"\n{'='*60}")
    print(f"🔍 FRAME {counter} | TIME = {time:.3f} s")
    print(f"{'='*60}")
    
    slc = yt.SlicePlot(ds, "z", ("gas", variable))

    slc_data = ds.slice("z", 0.0)
    x_arr = slc_data["gas", "x"].v
    y_arr = slc_data["gas", "y"].v

    temp_norm = slc_data["gas", "temperature"].v / T_min
    density_arr = slc_data["gas", "density"].v 

    center_mask = np.abs(y_arr) < 0.1
    x_c = x_arr[center_mask]
    temp_c = temp_norm[center_mask]
    density_c = density_arr[center_mask]

    sort_idx = np.argsort(x_c)
    x_sorted = x_c[sort_idx]
    temp_sorted = temp_c[sort_idx]
    density_sorted = density_c[sort_idx]

    temp_threshold = temp_sorted > T_TRESH
    dens_threshold = density_sorted > D_TRESH

    print(f"📊 temp_threshold sum: {np.sum(temp_threshold)} nokta")
    print(f"📊 dens_threshold sum: {np.sum(dens_threshold)} nokta")

    # --- ŞOK CEPHESİNİ BUL ---
    if np.any(temp_threshold):
        temp_indices = np.where(temp_threshold)[0]
        last_temp_pos = x_sorted[temp_indices[-1]]
        
        print(f"📊 En sağdaki sıcak nokta: x={last_temp_pos:.3f}, T={temp_sorted[temp_indices[-1]]:.3f}")
        
        if temp_indices[-1] < len(x_sorted) - 1:
            next_idx = temp_indices[-1] + 1
            next_temp = temp_sorted[next_idx]
            next_x = x_sorted[next_idx]
            print(f"📊 Sonraki nokta: x={next_x:.3f}, T={next_temp:.3f}")
            print(f"📊 Sıcaklık sıçraması: {temp_sorted[temp_indices[-1]] - next_temp:.3f}")
    else:
        last_temp_pos = np.nan
        print("❌ Hiç sıcak nokta yok!")

    if np.any(dens_threshold):
        dens_indices = np.where(dens_threshold)[0]
        last_dens_pos = x_sorted[dens_indices[-1]]
        print(f"📊 En sağdaki yoğun nokta: x={last_dens_pos:.3f}")
    else:
        last_dens_pos = np.nan
        print("❌ Hiç yoğun nokta yok!")

    # --- ŞOK DURUMUNU BELİRLE ---
    x_tracked = np.nan
    shock_status = "Undefined"
    shock_active = False  # BAŞLANGIÇTA FALSE OLSUN

    if not np.isnan(last_temp_pos) and not np.isnan(last_dens_pos):
        separation = abs(last_temp_pos - last_dens_pos)
        print(f"📊 Ayrışma mesafesi: {separation:.4f} (Eşik: {SEPERATION_THRESHOLD:.4f})")
        
        if separation <= SEPERATION_THRESHOLD:
            x_tracked = (last_temp_pos + last_dens_pos) / 2
            shock_status = "Active"
            shock_active = True
            print(f"✅ ŞOK AKTİF!")
        else:
            shock_status = "Dissipated"
            shock_active = False
            print(f"❌ ŞOK DAĞILMIŞ! (Ayrışma eşikten büyük)")
    else:
        print("❌ Yetersiz veri!")

    print(f"{'='*60}\n")
    
    # --- ÇİZGİLERİ ÇİZ ---
    x_predicted = vshock_bg_lab * time
    
    # Teorik tahmin (CYAN)
    slc.annotate_line(
        (x_predicted, 8.0, 0.0),
        (x_predicted, -8.0, 0.0),
        coord_system="data",
        color="cyan",
    )

    # Takip edilen şok (KIRMIZI) - SADECE AKTİFSE
    if shock_active and not np.isnan(x_tracked):
        slc.annotate_line(
            (x_tracked, 8.0, 0.0),
            (x_tracked, -8.0, 0.0),
            coord_system="data",
            color="red",
            plot_args={"linestyle": "--", "linewidth": 2.5},
        )
        print(f"🔴 KIRMIZI ÇİZGİ ÇİZİLDİ: x={x_tracked:.3f}")
    else:
        print(f"⚪ KIRMIZI ÇİZGİ ÇİZİLMEDİ (şok aktif değil)")

    # Başlangıç Sınırı (BEYAZ)
    slc.annotate_line(
        (0.0, 8.0, 0.0), (0.0, -8.0, 0.0), coord_system="data", color="w"
    )

    slc.render()

    # --- LEJANT ---
    if shock_active:
        track_label = f"Shock Front ({x_tracked:.2f})"
    elif shock_status == "Dissipated":
        track_label = "Shock Dissipated"
    else:
        track_label = "Shock Undefined"

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
    p.figure.savefig(frame_name, bbox_inches="tight")
    
    # --- DOĞRU LOG MESAJI (counter sıfırlanmadan) ---
    if shock_active:
        shock_str = f"{x_tracked:.2f}"
    else:
        shock_str = "N/A"
    
    print(f"Frame {counter:02d} | Time: {time:.3f} | "
          f"Shock: {shock_str} | Pred: {x_predicted:.2f} | "
          f"Status: {shock_status} | KIRMIZI ÇİZGİ: {'EVET' if shock_active else 'HAYIR'}")


# --- ANA DÖNGÜ (counter burada doğru yönetiliyor) ---
counter = 0
for snap in snaps[:40]:
    ds = yt.load(snap)
    
    # Gereksiz veri okumalarını kaldırdım (fonksiyon zaten yapıyor)
    make_comparison_frame(ds, var, vshock_bg_lab, counter)
    counter += 1  # <--- BURADA ARTTIR

# --- VİDEO OLUŞTUR ---
frames = sorted(glob(f"{output_frames_dir}/*{var}*.png"))

if frames:
    movie_name = f"sim_15_try2_tracked_vs_predicted7_{var}.mp4"
    W, H = 1040, 496

    with imageio.get_writer(movie_name, format="FFMPEG", fps=8) as writer:
        for f in frames:
            img = Image.open(f)
            img_resized = img.resize((W, H))
            writer.append_data(np.array(img_resized))

    print(f"\n🎬 İŞLEM TAMAM! Video: {movie_name}")