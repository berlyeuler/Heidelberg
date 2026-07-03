import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

# Veri setini yüklüyoruz
ds = yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0100.vtk')

# ==============================================================================
# 1. BÖLÜM: DUAL RANKINE-HUGONIOT HISTOGRAM FONKSİYONU
# ==============================================================================
def predicted_histogram(ds, field, bins=30):
    gamma = 5 / 3 
    
    # --- Sabit Giriş Parametreleri ---
    rho_bg = 1.0
    T_bg = 10.0
    rho_wind = 1.0
    T_wind = 100.0

    # --- Hocanın predict_post_shock.py Kodundan Gelen Kesin Değerler ---
    # Background şok değerleri (M_bg = 8.9887 için analitik hesaplama)
    M_bg = 8.9887
    comp_bg = ((gamma + 1) * (M_bg**2)) / (2 + (gamma - 1) * (M_bg**2))
    press_ratio_bg = 1 + (2 * gamma / (gamma + 1)) * (M_bg**2 - 1)
    T_ratio_bg = press_ratio_bg / comp_bg

    pred_density_bg = comp_bg * rho_bg                     # ~3.86
    pred_pressure_bg = (rho_bg * T_bg) * press_ratio_bg    # ~1007.5
    pred_temperature_bg = T_bg * T_ratio_bg                # ~261.1

    # Wind şok değerleri (Terminal çıktısından birebir alınanlar)
    pred_density_wind = 2.9342579
    pred_temperature_wind = 343.3467489
    pred_pressure_wind = 1007.467187  # Post shock wind/BG pressure değeri

    ad = ds.all_data()
    
    # Hangi alanı (field) çiziyorsak ona göre teorik değerleri eşleştiriyoruz
    if field == "density":
        pred_bg = pred_density_bg
        pred_wind = pred_density_wind
        field_data = ad[("athena", "density")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"

    elif field == "pressure":
        pred_bg = pred_pressure_bg
        pred_wind = pred_pressure_wind
        field_data = ad[("athena", "pressure")]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"

    elif field == "temperature":
        pred_bg = pred_temperature_bg
        pred_wind = pred_temperature_wind
        field_data = ad["athena", "pressure"] / ad["athena", "density"]
        label_bg = f"Predicted Background ({pred_bg:.2f})"
        label_wind = f"Predicted Wind ({pred_wind:.2f})"
    
    else:
        print("Choose density, pressure or temperature")
        return
    
    # --- Histogram Çizimi ---
    plt.figure(figsize=(10, 6))

    # Simülasyon verilerinin histogramı
    sns.histplot(field_data, bins=bins, color="pink", label="Simulation Data", 
                 element="step", fill=False, log_scale=(True, True))

    # 1. Çizgi: Background Tahmini (Kırmızı kesikli)
    plt.axvline(x=pred_bg, color="crimson", linestyle="--", linewidth=2.5, label=label_bg)
    
    # 2. Çizgi: Wind Tahmini (Mavi kesikli)
    plt.axvline(x=pred_wind, color="royalblue", linestyle="--", linewidth=2.5, label=label_wind)
    
    plt.xlabel(f"{field.capitalize()} (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"{field.capitalize()} Histogram & Dual Rankine-Hugoniot Prediction")
    plt.legend()
    
    output_path = f"/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/{field}_predicted_hist.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully saved histogram: {output_path}")


# ==============================================================================
# 2. BÖLÜM: SUTHERLAND-DOPITA SOĞUMA HARİTASI (T_COOL CONTOUR)
# ==============================================================================
@np.vectorize
def sutherland_dopita_cooling_rate_function(temperature_dimless):
    normalization = 1e4  
    temperature = temperature_dimless * normalization * 8.61733e-8  # K -> keV
    
    if temperature < 1.0e-5:
        return 5.890872e-13 * (temperature/1.0e-5)**6.0
    elif temperature < 0.0017235:
        return 5.890872e-13 * (temperature / 1.0e-5) ** 6.0
    elif temperature < 0.02:
        return 15.438249 * (temperature/0.0017235)**0.6
    elif temperature < 0.13:
        return 66.831473 * (temperature/0.02)**-1.7
    elif temperature < 0.7:
        return 2.773501 * (temperature/0.13)**-0.5
    elif temperature < 5.0:
        return 1.195229 * (temperature/0.7)**0.22
    elif temperature < 100.0:
        return 1.842056 * (temperature/5.0)**0.4
    else:
        return 6.10541 * (temperature/100.0)**0.4
    
def calculate_cooling_time(density, temperature, gamma=5.0/3.0, small_lambda=1.0):
    lamb = sutherland_dopita_cooling_rate_function(temperature)
    cooling_time_array = temperature / ((gamma - 1.0) * density * small_lambda * lamb)
    return cooling_time_array

def plot_cooling_time_map():
    gamma = 5 / 3
    small_lambda = 1.0
    
    # Grafikte basacağımız kesin şok koordinatları
    pred_rho_bg = 3.86
    pred_T_bg = 261.1
    
    pred_rho_wind = 2.9342
    pred_T_wind = 343.3467

    # Eksen sınırları (Sıcaklık 343.35'i rahat kapsasın diye max_T=500 yapıldı)
    min_rho, max_rho = float(1.0), float(10.0)
    min_T, max_T = float(1.0), float(500.0)

    y_temperature = np.linspace(min_T, max_T, num=100)
    x_density = np.linspace(min_rho, max_rho, num=100)
    rho_mesh, T_mesh = np.meshgrid(x_density, y_temperature)
    t_cool_mesh = calculate_cooling_time(rho_mesh, T_mesh, gamma=gamma, small_lambda=small_lambda)

    plt.figure(figsize=(9, 7))
    contour_plot = plt.contourf(rho_mesh, T_mesh, np.log10(t_cool_mesh), levels=20, cmap='viridis')
    plt.colorbar(contour_plot, label='Log cooling Time ($t_{cool}$)')

    # --- Şoklanmış Durum Noktalarını Ekleme ---
    # 1. Shocked Background
    plt.scatter(pred_rho_bg, pred_T_bg, color='crimson', marker='*', s=200, 
                label=f'Shocked Background (T={pred_T_bg:.1f})')

    # 2. Shocked Wind
    plt.scatter(pred_rho_wind, pred_T_wind, color='cyan', marker='X', s=200, 
                label=f'Shocked Wind (T={pred_T_wind:.1f})')

    plt.xlabel('Density ($\\rho$)')
    plt.ylabel('Temperature ($T$)')
    plt.title('Cooling time ($t_{cool}$) and Shocked States')
    plt.legend()
    
    output_path = "/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cooling_time_map.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully saved cooling map: {output_path}")


# ==============================================================================
# KODU TETİKLEME / ÇALIŞTIRMA ALANI
# ==============================================================================
if __name__ == "__main__":
    # 1. Üç adet histogramı da sırayla üretip kaydeder
    predicted_histogram(ds, "density")
    predicted_histogram(ds, "pressure")
    predicted_histogram(ds, "temperature")
    
    # 2. Üzerinde yıldız ve çarpı olan renkli soğuma haritasını üretir
    plot_cooling_time_map()
    
    print("\n[Tebrikler!] Bütün grafikler güncel verilerle eksiksiz üretildi.")

