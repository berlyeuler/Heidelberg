import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt
import glob
import os 
import sys
import imageio.v2 as imageio

# --- TEK VE DİNAMİK CREATE_MOVIE FONKSİYONU ---
def create_movie(base_path, field="density", fps=10):
    """
    Klasördeki tüm vtk dosyalarını döngüye sokarak seçilen alan için
    (density, pressure, temperature) kesit grafikleri çizer ve resimleri kaydeder.
    """
    # 1. 'frames' klasörünü simülasyon yolunun altında tanımlıyoruz
    output_directory = os.path.join(base_path, "frames")
    
    # 2. Eğer o simülasyonun içinde 'frames' yoksa otomatik oluşturuyoruz
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    print(f"\n[STEP 1] {field} snapshots are loading from {base_path}... ")
    ts = yt.DatasetSeries(f"{base_path}/id0/cloud*.vtk")

    print(f"[STEP 2] Snapshots are being saved to '{output_directory}' klasörüne kaydediliyor...")
    
    for ds in ts:
        filename = os.path.basename(str(ds))
        snap_num = filename.split(".")[1]

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
        p.save(os.path.join(output_directory, f"frame_{field}_{snap_num}.png"))

    print(f"All snapshots are saved in {output_directory}")


# --- DİNAMİK MAKE_MOVIE FONKSİYONU ---
def make_movie(base_path, field="density", suffix="custom", spesific_snaps=None, fps=10):
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

    # --- DÜZELTME: Videoyu ezmemek için doğrudan simülasyon klasörünün içine yazar ---
    movie_name = os.path.join(base_path, f"movie3_{field}_{suffix}.mp4")
    
    with imageio.get_writer(movie_name, format="FFMPEG", fps=fps) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
            
    print(f"Video successfully created: {movie_name}")


# --- ÇALIŞTIRMA ALANI (EXECUTION) ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python movie101.py <simulation_dir_name>")
        sys.exit(1)

    # Slurm script'inden gelen klasör adını alıyoruz (Örn: sim_0, sim_1...)
    incoming_dir = sys.argv[1]
    
    # Ana dizin yolunu tanımlıyoruz ve gelen klasörle birleştiriyoruz
    base_scratch_path = "/scratch/hpc-prf-radmix/hpcbeoe"
    sim_path = os.path.join(base_scratch_path, incoming_dir)

    print(f"Simulation path: {sim_path}")

    # Önce resimleri oluşturuyoruz
    create_movie(sim_path, field="density")

    # Sonra videoyu basıyoruz
    make_movie(sim_path, field="density", suffix="custom")