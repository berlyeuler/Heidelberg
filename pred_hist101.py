import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

ds = yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0100.vtk')

def predicted_histogram(ds, field, bins=30):
    M1 = 8.9887
    gamma = 5 / 3 
    rho1 = 1.0
    T = 10.0

    ad = ds.all_data()
    
    if field == "density":
        densty = ((gamma+1)*(M1**2)) / (2 + (gamma-1) * (M1**2))
        predicted_value = densty * rho1
        field_data = ad[("athena", "density")]
        label_name = f"Predicted Density ({predicted_value:.2f})"

    elif field == "pressure":
        press = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
        press_first = rho1 * T 
        predicted_value = press_first * press 
        field_data = ad[("athena", "pressure")]
        label_name = f"Predicted Pressure ({predicted_value:.2f})"

    elif field == "temperature":
        densty = ((gamma+1)*(M1**2)) / (2 + (gamma-1) * (M1**2))
        press = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
        T_ratio = press / densty
        predicted_value = T * T_ratio
        field_data = ad["athena", "pressure"] / ad["athena", "density"] # Sıcaklık için P/rho yaptık
        label_name = f"Predicted Temperature ({predicted_value:.2f})"
    
    else:
        print("choose density, pressure or temperature")
        return
    
    # Boş bir figür açalım
    plt.figure(figsize=(10, 6))

    # Gerçek simülasyon verilerinin grafiği (Seaborn ile log_scale daha rahat yönetilir)
    # Hem X hem Y eksenini logaritmik yapmak hocanın istediği gibi detayları açacaktır
    sns.histplot(field_data, bins=bins, color="pink", label="Simulation Data", 
                 element="step", fill=False, log_scale=(True, True))

    # Teorik tahmini gösteren kesikli çizgi
    plt.axvline(x=predicted_value,
                color="crimson",
                linestyle="--",
                linewidth=2.5,
                label=label_name
                )
    
    # Grafiğin süslemeleri
    plt.xlabel(f"{field.capitalize()} (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"{field.capitalize()} Histogram & Rankine-Hugoniot Prediction")
    plt.legend()
    
    # plt.show() YERİNE DOSYAYA KAYDEDİYORUZ:
    output_path = f"/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/{field}_predicted_hist.png"
    plt.savefig(output_path, dpi=300)
    plt.close() # Hafızayı temizlesin
    print(f"Grafik başarıyla kaydedildi: {output_path}")

# Fonksiyonu çağırıyoruz
predicted_histogram(ds, "density")





