import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt
import os


def predicted_histogram(ds, ad, field, bins=30):
    gamma = 5 / 3 
    
    # ==============================================================================
    # PATH'TEN OTOMATİK İSİM ÇEKME ALANI
    # ==============================================================================
    full_path = ds.parameter_filename
    path_parts = full_path.split(os.sep) 
    
    file_name = path_parts[-1]
    frame_num = file_name.split('.')[-2] 
    sim_name = path_parts[-3] 
    # ==============================================================================

    # --- Sabit Giriş Parametreleri ---
    rho_bg = 20.60
    T_bg = 19.78
    rho_wind = 21.04
    T_wind = 36.76
    init_pressure_bg = rho_bg * T_bg
    init_pressure_wind = rho_wind * T_wind

    # --- (PREDICT_POST_SHOCK.PY)dan GELEN KESİN ANALİTİK DEĞERLER ---
    pred_density_bg = 61.08
    pred_pressure_bg = 4278.29
    pred_temperature_bg = 70.03

    pred_density_wind = 51.04
    pred_temperature_wind = 83.80
    pred_pressure_wind = 4278.29

    # Veriyi numpy array formatına çeviriyoruz ki taşma yapmasın
    field_data = np.array(ad[("gas", field)], dtype=np.float64)

    if field == "density":
        init_bg = rho_bg
        init_wind = rho_wind
        pred_bg = pred_density_bg
        pred_wind = pred_density_wind
    elif field == "pressure":
        init_bg = init_pressure_bg
        init_wind = init_pressure_wind
        pred_bg = pred_pressure_bg
        pred_wind = pred_pressure_wind
    elif field == "temperature":
        field_data = field_data / 6.97436478913788e-09  #***** önemliii temperatureyi bu değere bölmeyi unutmaaa in every codee 
        init_bg = init_pressure_bg / rho_bg
        init_wind = init_pressure_wind / rho_wind
        pred_bg = pred_temperature_bg
        pred_wind = pred_temperature_wind
    else:
        print("Choose density, pressure or temperature")
        return
   
    label_bg = f"Predicted Background ({pred_bg:.2f})"
    label_wind = f"Predicted Wind ({pred_wind:.2f})"
    label_init_wind = f"Initial Wind ({init_wind:.2f})"
    label_init_bg = f"Initial Background ({init_bg:.2f})"
 
    density = np.array(ad[("athena", "density")], dtype=np.float64)

    # ==============================================================================
    #  (DİNAMİK BİNLER)
    # ==============================================================================
    # Verideki gerçek min ve max değerleri alıyoruz (kırpma yapmadan)
    actual_min = np.min(field_data)
    actual_max = np.max(field_data)
    
    # Eğer min değer 0 veya negatifse logspace hata verir, o yüzden güvenli bir alt sınır koyuyoruz
    if actual_min <= 0:
        actual_min = 1e-3 

    log_bins = np.logspace(np.log10(actual_min), np.log10(actual_max), 50)
    # ==============================================================================
    
    # --- HISTOGRAM GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))

    

    #  's' skalerine göre doğrudan dilimliyoruz
    field_data_wind = field_data[density >-1]
    field_data_bg = field_data[density <= -1]

    # 1. Rüzgar (Wind) Histogramı - Pembe
    if len(field_data_wind) > 0:
        weights_wind = np.ones_like(field_data_wind)/len(field_data_wind)
        plt.hist(
            field_data_wind,
            bins=log_bins,
            density=True,
            histtype="step",
            linewidth=2,
            edgecolor="deeppink",
            label="Wind",
            color="pink")
        
    # 2. Arka Plan (Background) Histogramı - Koyu Yeşil
    if len(field_data_bg) > 0:
        plt.hist(field_data_bg, 
                 bins=log_bins, 
                 density=True,
                 color="darkgreen", 
                 edgecolor="darkgreen",
                 label="Background", 
                 histtype="step", 
                 linewidth=2)

    # Eksen ayarları
    plt.xscale('log')
    plt.yscale('log')

    # Analitik Teorik Çizgiler
    plt.axvline(x=pred_bg, color="crimson", linestyle="--", linewidth=2.5, label=label_bg)
    plt.axvline(x=pred_wind, color="royalblue", linestyle="--", linewidth=2.5, label=label_wind)
    plt.axvline(x=init_wind, color="darkgreen", linestyle="-", linewidth=2.5, label=label_init_wind)
    plt.axvline(x=init_bg, color="orange", linestyle="-", linewidth=2.5, label=label_init_bg)

    # Başlık ve Etiketler
    plt.xlabel(f"{field.capitalize()} (log scale)")
    plt.ylabel("Probability Density")
    plt.title(f"{sim_name.upper()} (VTK {frame_num}) - {field.capitalize()} Histogram & Dual Rankine-Hugoniot Prediction")
    plt.legend(loc="upper right")
    
    # Otomatik Kaydetme
    output_dir = os.path.dirname(full_path)
    output_path = f"{output_dir}/{field}_{sim_name}_vtk{frame_num}_hist.png"
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully saved: {output_path}")


# ==============================================================================
# GRAFİKLERİ TETİKLEME ALANI
# ==============================================================================
if __name__ == "__main__":
    
    path_list = [
        '/scratch/hpc-prf-radmix/hpcbeoe/sim_9/id0/cloud.0050.vtk',
        #'/scratch/hpc-prf-radmix/hpcbeoe/sim_9/id0/cloud.0094.vtk'
    ]
    
    fields = ["density" , "pressure", "temperature"]
              # "density" , "pressure", "temperature"]
    
    for path in path_list:
        print(f"\nVeri yükleniyor: {path}")

        ds = yt.load(path)
        ad = ds.all_data()
        
        for field in fields:
            predicted_histogram(ds, ad, field)
            
    print("\n[Mükemmel] Tüm grafikler çizildi ve kaydedildi.")