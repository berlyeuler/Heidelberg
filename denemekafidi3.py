import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt
import os

def predicted_histogram(ds, field, bins=30):
    gamma = 5 / 3 
    
    # ==============================================================================
    # PATH'TEN OTOMATİK İSİM ÇEKME ALANI
    # ==============================================================================
    # ds.parameter_filename bize tam yolu verir (örn: /scratch/.../sim_3/id0/cloud.0015.vtk)
    full_path = ds.parameter_filename
    
    # Dosya adını (cloud.0015.vtk) ve bulunduğu klasörleri ayırıyoruz
    path_parts = full_path.split(os.sep) 
    
    # cloud.0015.vtk -> '0015' kısmını alır
    file_name = path_parts[-1]
    frame_num = file_name.split('.')[-2] 
    
    # id0'dan bir önceki klasör sim adıdır (sim_3)
    sim_name = path_parts[-3] 
    # ==============================================================================

    # --- Sabit Giriş Parametreleri ---
    rho_bg = 20.60
    T_bg = 19.78
    rho_wind = 21.04
    T_wind = 36.76
    init_pressure_bg = rho_bg * T_bg
    init_pressure_wind = rho_wind * T_wind

    # ---  (PREDICT_POST_SHOCK.PY)dan GELEN KESİN ANALİTİK DEĞERLER ---
    pred_density_bg = 61.08
    pred_pressure_bg = 4278.29
    pred_temperature_bg = 70.03

    pred_density_wind = 51.04
    pred_temperature_wind = 83.80
    pred_pressure_wind = 4278.29

    ad = ds.all_data()
    
    if field == "density":
        init_bg = rho_bg
        init_wind = rho_wind
        pred_bg = pred_density_bg
        pred_wind = pred_density_wind
        field_data = ad[("athena", "density")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"
        label_init_wind = f"Initial Wind ({init_wind:.2f})"
        label_init_bg = f"Initial Background ({init_bg:.2f})"

    elif field == "pressure":
        init_bg = init_pressure_bg
        init_wind = init_pressure_wind
        pred_bg = pred_pressure_bg
        pred_wind = pred_pressure_wind
        field_data = ad[("athena", "pressure")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"
        label_init_wind = f"Initial Wind ({init_wind:.2f})"
        label_init_bg = f"Initial Background ({init_bg:.2f})"
        
    elif field == "temperature":
        init_bg = init_pressure_bg / rho_bg
        init_wind = init_pressure_wind / rho_wind
        pred_bg = pred_temperature_bg
        pred_wind = pred_temperature_wind
        field_data = ad[("athena", "pressure")] / ad[("athena", "density")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"
        label_init_wind = f"Initial Wind ({init_wind:.2f})"
        label_init_bg = f"Initial Background ({init_bg:.2f})"
    else:
        print("Choose density, pressure or temperature")
        return
    
    # --- HISTOGRAM GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))

    # Etiket artık otomatik gelen sim_name ve frame_num'ı kullanıyor
    sns.histplot(field_data, bins=bins, color="pink", 
                 label=f"Simulation Data ({sim_name} - VTK {frame_num})", 
                 element="step", fill=False, log_scale=(True, True))

    plt.axvline(x=pred_bg, color="crimson", linestyle="--", linewidth=2.5, label=label_bg)
    plt.axvline(x=pred_wind, color="royalblue", linestyle="--", linewidth=2.5, label=label_wind)
    plt.axvline(x=init_wind, color="darkgreen", linestyle="-", linewidth=2.5, label=label_init_wind)
    plt.axvline(x=init_bg, color="orange", linestyle="-", linewidth=2.5, label=label_init_bg)

    # Başlık otomatik güncelleniyor
    plt.xlabel(f"{field.capitalize()} (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"{sim_name.upper()} (VTK {frame_num}) - {field.capitalize()} Histogram & Dual Rankine-Hugoniot Prediction")
    plt.legend()
    
    # Çıktı klasörünü de otomatik olarak verinin olduğu klasöre kaydediyoruz
    output_dir = os.path.dirname(full_path)
    output_path = f"{output_dir}/{field}_{sim_name}_vtk{frame_num}_hist.png"
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully saved: {output_path}")


# ==============================================================================
# GRAFİKLERİ TETİKLEME ALANI
# ==============================================================================
if __name__ == "__main__":
    
    # Sen sadece yüklemek istediğin yolları listeye ekle, gerisini kod halleder
    path_list = [
        '/scratch/hpc-prf-radmix/hpcbeoe/sim_3/id0/cloud.0015.vtk',
        '/scratch/hpc-prf-radmix/hpcbeoe/sim_3/id0/cloud.0036.vtk'
    ]
    
    fields = ["density", "pressure", "temperature"]
    
    for path in path_list:
        print(f"\nVeri yükleniyor: {path}")
        ds = yt.load(path)
        
        for field in fields:
            # Fonksiyona ekstra hiçbir şey yazmana gerek kalmadı!
            predicted_histogram(ds, field)
            
    print("\n[Mükemmel!] Tüm grafikler path'ten otomatik okunarak başarıyla çizildi.")