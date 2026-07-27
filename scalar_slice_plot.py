import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yt
import imageio.v2 as imageio

def create_slice(base_path, fps=5):
    output_directory = os.path.join(base_path, "slice_frames")
    os.makedirs(output_directory, exist_ok=True)

    print(f"\n[1] Snapshots are loading: {base_path} ...")
    ts = yt.DatasetSeries(f"{base_path}/id0/cloud.*.vtk")
    print(f"[STEP 2] Snapshots are being saved to '{output_directory}'")

    frame_files = []

    for ds in ts[::15]:
        filename = os.path.basename(str(ds))
        snap_num = filename.split(".")[1]

        expected_frame_path = os.path.join(output_directory, f"frame_slice_{snap_num}.png")
        frame_files.append(expected_frame_path)

        if os.path.exists(expected_frame_path):
            print(f"Snap {snap_num} skipping, already exists.")
            continue

        # --- SLICE PLOT OLUŞTURMA ---
        # 1. Hangi scalar alanı varsa onu bulalım
        if ("athena", "specific_scalar[0]") in ds.field_list:
            target_field = ("athena", "specific_scalar[0]")
        elif ("athena", "scalar[0]") in ds.field_list:
            target_field = ("athena", "scalar[0]")
        else: 
            print(f"Scalar alanı bulunamadı! Mevcut alanlar: {ds.field_list}")
            return

        # 2. 'z' ekseni boyunca Slice alıyoruz (target_field tırnaksız değişken olarak girilmeli)
        plot = yt.SlicePlot(ds, "z", target_field)

        plot.set_log(target_field, False)

        # 3. Colorbar limitleri 0 - 1 (Hocanın isteği)
        plot.set_zlim(target_field, 0.0, 1.0)

        # 4. Renk paleti seçelim
        plot.set_cmap(target_field, "viridis")

        # 5. Görseli kaydet
        plot.save(expected_frame_path)
        print(f"Frame saved -> {expected_frame_path}")

    # --- VIDEO OLUŞTURMA KISMI ---
    print("\n[3] Video is preparing...")
    movie_path = os.path.join(base_path, "scalar_evolution.mp4")
    
    images = []
    first_shape = None
    for frame_file in sorted(frame_files):
        if os.path.exists(frame_file):
            img = imageio.imread(frame_file)

            if first_shape is None: 
                first_shape = img.shape
                
            if img.shape != first_shape:
                from PIL import Image
                img = np.array(Image.fromarray(img).resize((first_shape[1], first_shape[0])))

            images.append(img)

    if images:
        imageio.mimsave(movie_path, images, fps=fps)
        print(f"SUCCESS: Movie created -> {movie_path}")
    else:
        print("ERROR: There are no saved frames!")

# --- KULLANIM ---
if __name__ == "__main__":
    base_dir = "/scratch/hpc-prf-radmix/hpcbeoe/sim_3" 
    create_slice(base_dir, fps=3)


       
            

       


