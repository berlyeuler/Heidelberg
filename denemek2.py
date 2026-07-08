import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

# Veri setini yüklüyoruz
ds = yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0100.vtk')

def predicted_histogram(ds, field, bins=30):
    gamma = 5 / 3 
    
    # --- Sabit Giriş Parametreleri ---
    rho_bg = 1.0
    T_bg = 10.0
    rho_wind = 1.0
    T_wind = 100.0
    init_pressure_bg = rho_bg * T_bg
    init_pressure_wind = rho_wind * T_wind

    # --- HOCANIN KODUNDAN (PREDICT_POST_SHOCK.PY) GELEN KESİN ANALİTİK DEĞERLER ---
    # 1. Background Şoklanmış Değerleri (M_bg = 8.9887 için)
    pred_density_bg = 3.86
    pred_pressure_bg = 1007.47     # Post-shock BG pressure
    pred_temperature_bg = 261.1    # Post-shock BG temperature (~261 K)

    # 2. Wind (Rüzgar) Şoklanmış Değerleri (Terminalde gördüğümüz kesin çıktılar!)
    pred_density_wind = 2.9342579      # Hocanın tam olarak görmek istediği "about 2.9" değeri!
    pred_temperature_wind = 343.34674  # Post-shock wind temperature
    pred_pressure_wind = 1007.46718    # Post-shock wind pressure

    ad = ds.all_data()
    
    # Hangi alanı (field) çiziyorsak ona göre teorik değerleri eşleştiriyoruz
    if field == "density":
        init_bg= rho_bg
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
        # yt ile sıcaklık alanını (P / rho) hesaplıyoruz
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

    # Simülasyon verilerinin histogramı
    sns.histplot(field_data, bins=bins, color="pink", label="Simulation Data", 
                 element="step", fill=False, log_scale=(True, True))

    # 1. Çizgi: Background Tahmini (Kırmızı kesikli çizgi)
    plt.axvline(x=pred_bg, color="crimson", linestyle="--", linewidth=2.5, label=label_bg)
    
    # 2. Çizgi: Wind Tahmini (Mavi kesikli çizgi) 
    plt.axvline(x=pred_wind, color="royalblue", linestyle="--", linewidth=2.5, label=label_wind)

    # 3. Çizgi: Başlangıç Wind Değeri (Yeşil kesikli çizgi)
    plt.axvline(x=init_wind, color="darkgreen", linestyle="**", linewidth=2.5, label=label_init_wind)

    # 4. Çizgi: Başlangıç Background Değeri (Turuncu kesikli çizgi)
    plt.axvline(x=init_bg, color="orange", linestyle="*", linewidth=2.5, label=label_init_bg)



    
    # Grafiğin süslemeleri
    plt.xlabel(f"{field.capitalize()} (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"{field.capitalize()} Histogram & Dual Rankine-Hugoniot Prediction")
    plt.legend()
    
    output_path = f"/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/{field}_predicted_hist_2.1line.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully saved: {output_path}")

# ==============================================================================
# GRAFİKLERİ TETİKLEME ALANI
# ==============================================================================
if __name__ == "__main__":
    # Üç adet histogramı da hocanın tam istediği analitik değerlerle sırayla üretir
    predicted_histogram(ds, "density")
    predicted_histogram(ds, "pressure")
    predicted_histogram(ds, "temperature")
    
    print("\n[Mükemmel!] İstediğin tüm histogramlar başarıyla güncellendi.")