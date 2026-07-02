import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

import glob
import os 
import imageio


if __name__ == "__main__":
    snap_numbers= ["0000", "0010", "0020", "0030", "0040", "0060", "0080", "0100"]  # Örnek snapshot numaraları, ihtiyaca göre değiştirilebilir
    #boş bi grafik taslağı açalım
    plt.figure(figsize= (10, 6))
    #tüm snapshotları döngüyle tek tek döndürelim
    for snap in snap_numbers:
        # Klasördeki id0 altındaki vtk dosyalarının yolunu dinamik yapıyoruz
        file_path= f"/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.{snap}.vtk"
        print(f"Yükleniyor: {file_path}")

        ds_current= yt.load(file_path) # 1. O anki snapshot dosyasını yüklüyoruz
        ad_current= ds_current.all_data() # 2. Tüm veriyi çekiyoruz
        field_data = ad_current[("athena", "density")]

        # seaborn ile logaritmik ve şık bir üst üste histogram çiziyoruz
        sns.histplot(field_data, 
                     log_scale=True, #log_scale=True → hem x hem de y eksenlerine logaritmik ölçeklendirme uygular.
                     label=f"Snapshot {snap}",
                     element= "step",
                     fill= False

                       )
    # Döngü bitti, şimdi ortak grafiği süslüyoruz
    plt.title("Density Histogram Evolution (0000 - 0100)")
    plt.xlabel("Density (log scale)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Grafiği kaydedeceğimiz yer
    output_plot = "/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/histogram_evolution.png"
    plt.savefig(output_plot, dpi=300)
    print(f"Saved: {output_plot}")


