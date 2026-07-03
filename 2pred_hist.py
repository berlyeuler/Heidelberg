import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

ds = yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0100.vtk')

def predicted_histogram(ds, field, bins=30):
    gamma = 5 / 3 
    
    
    # 1. Background Bileşenleri
    M_bg = 8.9887
    rho_bg = 1.0
    T_bg = 10.0

    # 2. Wind Bileşenleri (Yeni eklenen kısım)
    M_wind = 4.0          # Wind Mach number = 4.0
    rho_wind = 1.0        # Initial wind density = 1.0
    T_wind = 100.0        # Initial wind temperature = 100.0

    ad = ds.all_data()
    
    # --- Rankine-Hugoniot Sıkıştırma Faktörleri ---
    # Background için:
    comp_bg = ((gamma + 1) * (M_bg**2)) / (2 + (gamma - 1) * (M_bg**2))
    press_ratio_bg = 1 + (2 * gamma / (gamma + 1)) * (M_bg**2 - 1)
    
    # Wind için:
    comp_wind = ((gamma + 1) * (M_wind**2)) / (2 + (gamma - 1) * (M_wind**2))
    press_ratio_wind = 1 + (2 * gamma / (gamma + 1)) * (M_wind**2 - 1)

    # Hangi alanı (field) çiziyorsak ona göre teorik değerleri hesaplayalım
    if field == "density":
        pred_bg = comp_bg * rho_bg
        pred_wind = comp_wind * rho_wind
        field_data = ad[("athena", "density")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"

    elif field == "pressure":
        pred_bg = (rho_bg * T_bg) * press_ratio_bg
        pred_wind = (rho_wind * T_wind) * press_ratio_wind
        field_data = ad[("athena", "pressure")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"

    elif field == "temperature":
        T_ratio_bg = press_ratio_bg / comp_bg
        pred_bg = T_bg * T_ratio_bg
        
        T_ratio_wind = press_ratio_wind / comp_wind
        pred_wind = T_wind * T_ratio_wind
        
        field_data = ad["athena", "pressure"] / ad["athena", "density"]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"
    
    else:
        print("choose density, pressure or temperature")
        return
    
    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))

    # Simülasyon verilerinin histogramı
    sns.histplot(field_data, bins=bins, color="pink", label="Simulation Data", 
                 element="step", fill=False, log_scale=(True, True))

    # 1. Çizgi: Background Tahmini (Crimson / Kırmızı)
    plt.axvline(x=pred_bg, color="crimson", linestyle="--", linewidth=2.5, label=label_bg)
    
    # 2. Çizgi: Wind Tahmini (Royalblue / Mavi) -> Hocanın tam olarak istediği çizgi!
    plt.axvline(x=pred_wind, color="royalblue", linestyle="--", linewidth=2.5, label=label_wind)
    
    # Grafiğin süslemeleri
    plt.xlabel(f"{field.capitalize()} (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"{field.capitalize()} Histogram & Dual Rankine-Hugoniot Prediction")
    plt.legend()
    
    output_path = f"/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/{field}_predicted_hist.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Succesfuly saved: {output_path}")

# Fonksiyonu çağırıyoruz
predicted_histogram(ds, "density")

