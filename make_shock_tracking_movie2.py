import os
import sys
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import yt

# --- EŞİK DEĞERLERİ ---
T_TRESH = 50.0  
D_TRESH = 50.0



# Ayrışma eşiği (32/512 = 0.0625)
SEPERATION_THRESHOLD = 32 / 512

sys.path.append("/pc2/users/h/hpccado/repos/athena-ccd-maxbg-bssdl-fork/")
from predict_post_shock import calc_shock_speed

var = "temperature"
T_min = 6.97436478913788e-09

# --- SİMÜLASYON TANIMLARI ---
simulations_config = {
    "sim_4": {
        "path": "/scratch/hpc-prf-radmix/cdoughty/reruns/sim_4",
        "params": {"rho_bg": 49.550, "rho_wind": 33.770, "T_bg": 5.420, "T_wind": 73.990, "M_wind": 1.380}

    },

    "sim_12": {
        "path": "/scratch/hpc-prf-radmix/cdoughty/reruns/sim_12",
        "params": {"rho_bg": 39.291, "rho_wind": 17.062, "T_bg": 7.941, "T_wind": 28.059, "M_wind": 4.671}
    },

    "sim_14": {
        "path": "/scratch/hpc-prf-radmix/cdoughty/reruns/sim_14",
        "params": {"rho_bg": 43.135, "rho_wind": 31.498, "T_bg": 14.068, "T_wind": 13.235, "M_wind": 3.961}
    },

}

    # try2 simülasyonunu da eklemek isterseniz alt satırın yorumunu kaldırabilirsiniz:
    # "sim_15_try2": {
    #     "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_15_try2",
    #     "params": {"rho_bg": 38.135, "rho_wind": 12.245, "T_bg": 41.436, "T_wind": 98.271, "M_wind": 2.928}
    # }


# --- FONKSİYON (FRAME OLUŞTURMA) ---
# Fonksiyon imzalara rho_bg ekleyin
def make_comparison_frame(ds, variable, vshock_bg_lab, counter, output_dir, rho_bg):
    time = float(ds.current_time)
    
    slc = yt.SlicePlot(ds, "z", ("gas", variable))
    slc_data = ds.slice("z", 0.0)
    
    x_arr = slc_data["gas", "x"].v
    y_arr = slc_data["gas", "y"].v
    temp_norm = slc_data["gas", "temperature"].v / T_min
    density_arr = slc_data["gas", "density"].v 

    # y bandındaki verileri filtrele
    y_mask = np.abs(y_arr) < 2.0
    x_filtered = x_arr[y_mask]
    temp_filtered = temp_norm[y_mask]
    dens_filtered = density_arr[y_mask]

    # x değerlerine göre sırala
    sort_idx = np.argsort(x_filtered)
    x_line = x_filtered[sort_idx]
    temp_line = temp_filtered[sort_idx]
    density_line = dens_filtered[sort_idx]

    # Eşiği rho_bg parametresi üzerinden tanımla
    D_TRESH = rho_bg * 1.5

  

    # --- SADECE SAĞ KENARLARI BUL ---
    # Sıcaklık için sağ kenar
    temp_above = temp_line > T_TRESH
    if np.any(temp_above):
        temp_indices = np.where(temp_above)[0]
        right_temp_pos = x_line[temp_indices[-1]]
    else:
        right_temp_pos = None

    # Yoğunluk için sağ kenar
    dens_above = density_line > D_TRESH
    if np.any(dens_above):
        dens_indices = np.where(dens_above)[0]
        right_dens_pos = x_line[dens_indices[-1]]
    else:
        right_dens_pos = None

   

    # --- AYRIŞMA MESAFESİ VE ŞOK DURUMU ---
    if right_temp_pos is not None and right_dens_pos is not None:
        separation = abs(right_temp_pos - right_dens_pos)
        shock_active = separation <= SEPERATION_THRESHOLD
        
        if shock_active:
            x_tracked = (right_temp_pos + right_dens_pos) / 2
        else:
            x_tracked = right_dens_pos
    else:
        return time, np.nan, np.nan, vshock_bg_lab * time, False

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
    p.figure.set_size_inches(10, 5)
    frame_name = os.path.join(output_dir, f"frame_{variable}_{counter:04d}.png")
    p.figure.savefig(frame_name, bbox_inches="tight", dpi=100)
    
    return time, separation, x_tracked, x_predicted, shock_active


# ==========================================
# --- TÜM SİMÜLASYONLAR İÇİN DÖNGÜ BAŞLIYOR ---
# ==========================================

for sim_name, config in simulations_config.items():
    print(f"\n==================================================")
    print(f"🚀 İŞLENİYOR: {sim_name}")
    print(f"==================================================")
    
    sim_dir = config["path"]
    p = config["params"]
    
    # 1. Şok Hızını Hesapla
    #breakpoint()
    vshock_bg_lab, vshock_wind_lab, v_shocked_bg = calc_shock_speed(
        p["rho_bg"], p["T_bg"], p["rho_wind"], p["T_wind"], p["M_wind"]
    )
    
    # 2. Snapshot'ları Bul
    snaps = sorted(glob(f"{sim_dir}/id0/*.vtk")) + sorted(glob(f"{sim_dir}/*.vtk"))
    if not snaps:
        snaps = sorted(glob(f"{sim_dir}/id0/*.athdf")) + sorted(glob(f"{sim_dir}/*.athdf"))

    print(f"📁 [{sim_name}] Bulunan snapshot sayısı: {len(snaps)}")
    if not snaps:
        print(f"⚠️ [{sim_name}] İçin hiç snapshot bulunamadı, atlanıyor!")
        continue

    # 3. Çıktı Dizinlerini Hazırla
    output_frames_dir = f"frames_{sim_name}"
    os.makedirs(output_frames_dir, exist_ok=True)

    times = []
    separations = []
    shock_positions = []
    predicted_positions = []
    shock_status = []

    # 4. Frame'leri İşle (İlk 40 snapshot)
    counter = 0
    for snap in snaps[:40]:
        try:
            ds = yt.load(snap)
            result = make_comparison_frame(ds, var, vshock_bg_lab, counter, output_frames_dir, p["rho_bg"])
            time, separation, x_tracked, x_pred, active = result
            
            if not np.isnan(separation):
                times.append(time)
                separations.append(separation)
                shock_positions.append(x_tracked)
                predicted_positions.append(x_pred)
                shock_status.append(active)
        except Exception as e:
            print(f"  ❌ Frame {counter} hatası ({snap}): {e}")
            
        counter += 1

    # 5. GRAFİK 1: SEPARATION VS TIME
    if times:
        plt.figure(figsize=(12, 6))
        plt.plot(times, separations, 'r-o', label='Separation', linewidth=2, markersize=6)
        plt.axhline(y=SEPERATION_THRESHOLD, color='k', linestyle='--', 
                    label=f'Threshold ({SEPERATION_THRESHOLD:.4f})', linewidth=2)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Separation (x position difference)', fontsize=12)
        plt.title(f'[{sim_name}] Shock Front Separation vs Time', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'separation_vs_time_{sim_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. GRAFİK 2: SHOCK POSITION VS TIME
    if times:
        plt.figure(figsize=(12, 6))
        plt.plot(times, shock_positions, 'r-o', label='Tracked Shock', linewidth=2, markersize=6)
        plt.plot(times, predicted_positions, 'c-s', label='Predicted Shock', linewidth=2, markersize=6)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Shock Position (x)', fontsize=12)
        plt.title(f'[{sim_name}] Shock Position: Tracked vs Predicted', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'shock_position_vs_time_{sim_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 7. VİDEO OLUŞTURMA
    frames = sorted(glob(f"{output_frames_dir}/*{var}*.png"))
    if frames:
        movie_name = f"{sim_name}_tracked_vs_predicted_{var}.mp4"
        W, H = 1040, 496

        print(f"🎬 Video oluşturuluyor ({sim_name})... ({len(frames)} frame)")
        with imageio.get_writer(movie_name, fps=8) as writer:
            for f in frames:
                img = imageio.imread(f)
                if img.shape[1] != W or img.shape[0] != H:
                    img_pil = Image.fromarray(img)
                    img_pil = img_pil.resize((W, H), Image.Resampling.LANCZOS)
                    img = np.array(img_pil)
                writer.append_data(img)

        print(f"✅ {sim_name} tamamlandı! Video: {movie_name}\n")

print("🎉 TÜM SİMÜLASYONLAR BAŞARIYLA İŞLENDİ!")