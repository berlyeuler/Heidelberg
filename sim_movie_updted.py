import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt
import os
# yt kütüphanesinin ev dizinine ayar/önbellek yazmasını tamamen kapatıyoruz
os.environ["YT_SUPPRESS_USER_CONFIG"] = "1"

import glob
 
import sys
import imageio.v2 as imageio

# --- TEK VE DİNAMİK CREATE_MOVIE FONKSİYONU ---
def create_movie(base_path, field, fps= 10):
    """
    Klasördeki tüm vtk dosyalarını döngüye sokarak seçilen alan için
    (density, pressure, temperature) kesit grafikleri çizer ve resimleri kaydeder.
    """
    # 1. Hocanın istediği gibi 'frames' klasörünü simülasyon yolunun altında tanımlıyoruz
    output_directory = os.path.join(base_path, "frames")
    
    # 2. Eğer o simülasyonun içinde 'frames' yoksa otomatik oluşturuyoruz
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"\n[STEP 1] {field} snapshots are loading from {base_path}... ")
    ts = yt.DatasetSeries(f"{base_path}/id0/cloud*.vtk")

    print(f"[STEP 2] Snapshots are being saved to '{output_directory}' klasörüne kaydediliyor...")
    
    for ds in ts:
        # vtk dosya adından snapshot numarasını güvenle çekmek için:
        # Örn: cloud.0100.vtk -> '0100'
        filename = os.path.basename(str(ds))
        snap_num = filename.split(".")[1]
# -------------------------------------------------------------
        # --- YENİ: DAHA ÖNCE OLUŞTURULAN RESİMLERİ ATLA ---
        # Eğer bu resim zaten frames klasöründe varsa, çizim yapmadan direkt bir sonrakine geçer.
        expected_frame_path = os.path.join(output_directory, f"frame_{field}_{snap_num}.png")
        if os.path.exists(expected_frame_path):
            print(f"Skipping the already excited frames")
            continue


        if field == "temperature":
            def _temp(field, data):
                return data[("athena", "pressure")] / data[("athena", "density")]
            
            ds.add_field(
                ("gas", "temperature"),
                function=_temp,
                sampling_type="cell",
                units="",
            )
            p = yt.SlicePlot(ds, "z", ("gas", "temperature"))
            p.set_cmap(("gas", "temperature"), "inferno")

        else:
            p = yt.SlicePlot(ds, "z", ("athena", field))
            if field == "density":
                p.set_cmap(("athena", "density"), "viridis")
            elif field == "pressure":
                p.set_cmap(("athena", "pressure"), "magma")

        # 3. Resmi doğrudan o simülasyona ait dinamik frames klasörünün içine kaydediyoruz
        # yt plotları plt.savefig yerine p.save ile kaydedildiğinde daha kararlıdır:
        p.save(os.path.join(output_directory, f"frame_{field}_{snap_num}.png"))

    print(f"All snapshots are saved in {output_directory}")


# --- DİNAMİK MAKE_MOVIE FONKSİYONU ---
def make_movie(base_path, field, suffix="custom", spesific_snaps=None, fps=10):
    print(f"'{field}'s video is preparing using frames from {base_path}")

    # Resimlerin okunduğu dinamik klasör yolu
    output_directory = os.path.join(base_path, "frames")
    
    # Doğru klasördeki resimleri aratıyoruz
    frames = sorted(glob.glob(os.path.join(output_directory, f"*_{field}*.png")))
    
    if not frames:
        print(f"Error: Couldn't find any snapshots in '{output_directory}'. Run 'create_movie' first.")
        return

    if spesific_snaps is not None:
        selected_frames = []
        for snap in spesific_snaps:
            snap_str = f"{snap:04d}"
            matching_frame = [f for f in frames if f"{snap_str}.png" in f]
            if matching_frame:
                selected_frames.append(matching_frame[0])
        frames = selected_frames
        

    # Videoyu kodun çalıştığı ana klasöre farklı isimlerle kaydeder
    movie_name = os.path.join(base_path, f"moviee_{field}_{suffix}.mp4")
    with imageio.get_writer(movie_name, format="FFMPEG", fps=fps) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
            
    print(f"Video successfully created: {movie_name}")


# --- ÇALIŞTIRMA ALANI (EXECUTION) ---

# 1. İşlem yapmak istediğin simülasyonların ana klasör yolları
sim_paths = [
    #'/scratch/hpc-prf-radmix/hpcbeoe/sim_6',
    #'/scratch/hpc-prf-radmix/berillium/sim_0'
    '/scratch/hpc-prf-radmix/hpcbeoe/sim_2'
    #'/scratch/hpc-prf-radmix/berillium/sim_5',
    #'/scratch/hpc-prf-radmix/berillium/sim_6'
    
]

# 2. Çizdirmek istediğin tüm alanlar
fields = ["density", "temperature"]

# 3. İç içe döngüyle hepsini tek seferde hallediyoruz!
for sim_path in sim_paths:
    print(f"\n==========================================")
    print(f"PROCESSING: {sim_path}")
    print(f"==========================================")
    
    for field in fields:
        # Önce eksik resimler varsa onları çizdirip frames klasörüne kaydeder
        create_movie(sim_path, field=field)

        
        # Sonra o resimlerden videoyu oluşturup doğrudan simülasyon klasörünün içine atar
        make_movie(sim_path, field=field, suffix="custom")