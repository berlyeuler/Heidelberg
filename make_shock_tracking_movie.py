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
    
    
    slc = yt.SlicePlot(ds, "z", ("gas", variable))

    slc_data = ds.slice("z", 0.0)
    x_arr = slc_data["gas", "x"].v
    y_arr = slc_data["gas", "y"].v

    temp_norm = slc_data["gas", "temperature"].v / T_min
    density_arr = slc_data["gas", "density"].v 

    center_mask = np.abs(y_arr) < 0.1

    x_line = x_arr[center_mask]
    temp_line = temp_norm[center_mask]
    density_line = density_arr[center_mask]

    if not np.all(np.diff(x_line)>= 0):
         print("⚠️ x_line is not sorted! ")

    temp_threshold = temp_line > T_TRESH
    dens_threshold = density_line > D_TRESH

    # --- ŞOK CEPHESİNİ BUL ---
    if np.any(temp_threshold):
        temp_indices = np.where(temp_threshold)[0]

        left_temp_idx= temp_indices[0]
        right_temp_idx = temp_indices[-1]
        right_last_temp_pos = x_line[right_temp_idx]
        left_last_temp_pos = x_line[left_temp_idx]

        print(f"📊 Most right hot point: x={right_last_temp_pos:.4f}, T={temp_line[right_temp_idx]:.3f}")
        print(f"📊 Most left hot point: x={left_last_temp_pos:.4f}, T={temp_line[left_temp_idx]:.3f}")
        print(f"   Seperation : {right_last_temp_pos - left_last_temp_pos:.4f}")
        
        if right_temp_idx < len(x_line) - 1:
            next_idx = right_temp_idx + 1
            next_temp = temp_line[next_idx]
            next_x = x_line[next_idx]
            print(f"📊 next point: x={next_x:.4f}, T={next_temp:.3f}")
            print(f"📊 Temperature jump: {temp_line[right_temp_idx] - next_temp:.3f}")
    else:
        print("❌ Hiç sıcak nokta yok!")
        right_last_temp_pos = 0.0  
        left_last_temp_pos = 0.0

    if np.any(dens_threshold):
        dens_indices = np.where(dens_threshold)[0]

        left_dens_idx = dens_indices[0]
        right_dens_idx = dens_indices[-1]
        right_last_dens_pos = x_line[right_dens_idx]
        left_last_dens_pos = x_line[left_dens_idx]
        print(f"📊 most right denser point: x={right_last_dens_pos:.4f}")
        print(f"📊 most left denser point: x={left_last_dens_pos:.4f}")
        print(f"   Seperation : {right_last_dens_pos - left_last_dens_pos:.4f}")
    else:
        
        print("❌ Hiç yoğun nokta yok!")
        right_last_dens_pos = 0.0
        left_last_dens_pos = 0.0

        #ayrışma mesafelerini hesapla
        #sağ kenar icin 
        
    right_seperation = abs(right_last_temp_pos - right_last_dens_pos)
    print(f" Right edge seperation is {right_seperation:.4f} (Threshold is {SEPERATION_THRESHOLD:.4f})")

        #sol kenar icin 
        
    left_seperation = abs(left_last_temp_pos - left_last_dens_pos)
    print(f" Left edge seperation is {left_seperation:.4f} ")

    separation = right_seperation 
    shock_active = separation <= SEPERATION_THRESHOLD


    # --- ŞOK DURUMUNU BELİRLE ---
    
        
    if shock_active:
            x_tracked = (right_last_temp_pos + right_last_dens_pos) / 2
            print(f"✅ SHOCK ACTİVE! x_tracked = {x_tracked:.4f}")

    else:
        x_tracked = right_last_dens_pos
        print(f"❌ SHOCK İS SCATTERED! (SEPARATİON BİGGER THAN THRESHOLD)")
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
    if shock_active:
        slc.annotate_line(
            (x_tracked, 8.0, 0.0),
            (x_tracked, -8.0, 0.0),
            coord_system="data",
            color="red",
            plot_args={"linestyle": "--", "linewidth": 2.5},
        )
        print(f"🔴 KIRMIZI ÇİZGİ ÇİZİLDİ: x={x_tracked:.3f}")
    else:
        print(f"⚪ KIRMIZI ÇİZGİ ÇİZİLMEDİ (shock inactive)")

    # Başlangıç Sınırı (BEYAZ)
    slc.annotate_line(
        (0.0, 8.0, 0.0), (0.0, -8.0, 0.0), coord_system="data", color="w"
    )

    slc.render()

    # --- LEJANT ---
    if shock_active:
        track_label = f"Shock Front ({x_tracked:.2f})"

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
    
    # ---  LOG MESAJI (counter sıfırlanmadan) ---
   
    shock_str = f"{x_tracked:.2f}" if shock_active else "DISSIPATED"
    print(f"Frame {counter:02d} | Time: {time:.3f} | "
          f"Shock: {shock_str} | Pred: {x_predicted:.2f} |" f" Active: {'YES' if shock_active else 'NO'}")


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
