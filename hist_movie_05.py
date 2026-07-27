import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

import glob
import os 
import imageio.v2 as imageio

def create_scalar_movie(base_path, scalar_field_name=("athena", "specific_scalar[0]"), fps=5):
    """
    Athena simülasyon çıktılarından Scalar Histogram videosu oluşturur.
    """
    output_directory = os.path.join(base_path, "scalar_histogram_frames")
    os.makedirs(output_directory, exist_ok=True)

    print(f"\n[1] Snapshots are loading: {base_path} ...")
    # vtk dosyalarını sıralı al
    ts = yt.DatasetSeries(f"{base_path}/id0/cloud.*.vtk")

    frame_files = []

    print(f"[2] Histogram frames are being creating ...")
    
    # 20'şerli snapshot adımlarıyla döngü
    for i, ds in enumerate(ts[::15]):
        filename = os.path.basename(str(ds))
        # cloud.0100.vtk -> "0100"
        snap_num = filename.split(".")[1]
        
        expected_frame_path = os.path.join(output_directory, f"frame_scalar_{snap_num}.png")
        frame_files.append(expected_frame_path)

        # Eğer görsel önceden üretilmişse tekrar çizme (zaman kazanmak için)
        if os.path.exists(expected_frame_path):
            print(f"Frame {snap_num} zaten var, atlanıyor...")
            continue

        # Veriyi çek
        ad = ds.all_data()
        
        # Try-Except ile alan adını ve numpy dönüşümünü garantiye alalım:
        try:
            scalar_data = ad[("athena", "specific_scalar[0]")].d
        except Exception:
            try:
                scalar_data = ad[("athena", "scalar[0]")].d
            except Exception:
                # Eğer ikisi de değilse mevcut tüm alanları basıp durdurur
                print(f"Scalar alanı bulunamadı! Mevcut alanlar: {ds.field_list}")
                return

        # Simülasyon zamanını (dt / time) yt'den çek
        current_time = float(ds.current_time)

        # FIGUR VE CIZIM
        plt.figure(figsize=(9, 6))

        # Scalar 0 ile 1 arasında olduğu için lineer histplot
        sns.histplot(
            scalar_data,
            bins=50,
            binrange=(0.0, 1.0),
            color="skyblue",
            element="step",
            fill=True,
            alpha=0.4,
            label=f"Snap: {snap_num} | Time: {current_time:.2f}"
        )

        # 0.5 Seperator çizgisi (Wind ve Background ayrımı için referans)
        plt.axvline(
            x=0.5, 
            color="red", 
            linestyle="--", 
            linewidth=1.5, 
            label="Mix Threshold (s = 0.5)"
        )

        # X ekseni sınırı (0 ile 1 arası)
        plt.xlim(0.0, 1.0)
        
        # Y ekseni hücre frekansı için log ölçek
        plt.yscale("log")

        # Süslemeler & Etiketler
        plt.xlabel("Scalar Value (0: BG, 1: Wind)", fontsize=12)
        plt.ylabel("Cell Count / Frequency (log scale)", fontsize=12)
        plt.title(f"Scalar Distribution - Snapshot {snap_num}", fontsize=14)
        plt.grid(True, which="both", linestyle=":", alpha=0.5)
        plt.legend(loc="upper center", fontsize=11)

        # Kaydet
        plt.savefig(expected_frame_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Kare kaydedildi: {expected_frame_path}")

    # [3] MOVIE / oluşturma
    print("\n[3] video is preapearing...")
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
        print("ERROR: There is no saved frames!")

# --- KULLANIM ---
if __name__ == "__main__":
    # Kendi dizin yolun:
    base_dir = "/scratch/hpc-prf-radmix/hpcbeoe/sim_3" 
    create_scalar_movie(base_dir, fps=3)



    


    
