import os
import matplotlib

matplotlib.use("Agg")  # Sessizce arka planda kaydet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

# 1. DOSYA YOLLARI
BASE_DIR = "/scratch/hpc-prf-radmix/hpcbeoe"
OUTPUT_DIR = os.path.join(BASE_DIR, "cold_gas_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. PARAMETRE ÜRETİMİ (Mach Sayıları için)
nvars = 5
nsamples = 30
sampler = qmc.LatinHypercube(d=nvars, seed=34)
Xinit = sampler.random(n=nsamples)

bounds_min = np.array([1.0, 1.0, 1.0, 1.0, 1.1])
bounds_max = np.array([50.0, 50.0, 20.0, 100.0, 5.0])
X_phys = bounds_min + Xinit * (bounds_max - bounds_min)

# 3. SUTHERLAND & DOPITA COOLING TIME HESABI
gamma = 5.0 / 3.0


@np.vectorize
def sutherland_dopita_cooling_rate_function(temperature_dimless):
    normalization = 1e4
    temperature = (
        temperature_dimless * normalization * 8.61733e-8
    )  # K to keV

    if temperature < 1.0e-5:
        return 5.890872e-13 * (temperature / 1.0e-5) ** 6.0
    elif temperature < 0.0017235:
        return 5.890872e-13 * (temperature / 1.0e-5) ** 6.0
    elif temperature < 0.02:
        return 15.438249 * (temperature / 0.0017235) ** 0.6
    elif temperature < 0.13:
        return 66.831473 * (temperature / 0.02) ** -1.7
    elif temperature < 0.7:
        return 2.773501 * (temperature / 0.13) ** -0.5
    elif temperature < 5.0:
        return 1.195229 * (temperature / 0.7) ** 0.22
    elif temperature < 100.0:
        return 1.842056 * (temperature / 5.0) ** 0.4
    else:
        return 6.10541 * (temperature / 100.0) ** 0.4


def calculate_cooling_time(
    density, temperature, gamma=5.0 / 3.0, small_lambda=1.0
):
    lamb = sutherland_dopita_cooling_rate_function(temperature)
    mu_e = 1.17
    cooling_time = (mu_e**2 * temperature) / (
        (gamma - 1.0) * density * small_lambda * lamb
    )
    return cooling_time


# 4. SİMÜLASYON VERİLERİNİ İŞLEME
TARGET_SIMS = [3, 4, 9, 10, 11, 12, 13, 14, 15]
t_cool_list = []
growth_rate_list = []
labels = []

for idx in TARGET_SIMS:
    sim_name = f"sim_{idx}"
    csv_path = os.path.join(BASE_DIR, sim_name, "cold_gas_ke_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, sim_name, "cold_gas_data.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        time_cols = [c for c in df.columns if "time" in c.lower()]
        mass_cols = [
            c for c in df.columns if "mass" in c.lower() or "cold" in c.lower()
        ]

        if time_cols and mass_cols:
            times = df[time_cols[0]].values
            masses = df[mass_cols[0]].values

            # 🟢 Hocanın önerisi: Kutudan çıkmadan önceki sabit büyüme rejimini al (time <= 1.7)
            mask = times <= 1.7
            times_filtered = times[mask]
            masses_filtered = masses[mask]

            if len(times_filtered) > 1:
                # Filtrelenmiş veri ile ortalama büyüme hızı
                dm_dt = np.mean(
                    np.diff(masses_filtered) / np.diff(times_filtered)
                )

                # Mach sayısından post-shock T ve rho tahmini
                # --- DÜZELTİLMİŞ KISIM ---

                # 1. Simülasyonun gerçek girdilerini LHS matrisinden dinamik çek:
                M1 = X_phys[idx, 4]        # Mach sayısı (4. indeks!)
                rho_wind = X_phys[idx, 1]  # Gerçek rüzgar yoğunluğu (1. indeks)
                T_wind = X_phys[idx, 3]    # Gerçek şok öncesi rüzgar sıcaklığı (3. indeks)

                # 2. Şok ilişkileri (Rankine-Hugoniot)
                densty = ((gamma + 1) * (M1**2)) / (2 + (gamma - 1) * (M1**2))
                press = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
                T_ratio = press / densty

                # 3. Gerçek parametrelerle çarp:
                pred_rho = densty * rho_wind
                pred_T = T_ratio * T_wind   # ❌ (* 100.0) kaldırıldı, yerine gerçek T_wind kondu!

                # 4. Doğru t_cool hesabı:
                t_c = calculate_cooling_time(pred_rho, pred_T, gamma=gamma, small_lambda=1.0)

                t_cool_list.append(t_c)
                growth_rate_list.append(dm_dt)
                labels.append(sim_name)

# 5. SCATTER PLOT ÇİZDİR
plt.figure(figsize=(8, 6))
plt.scatter(
    t_cool_list,
    np.array(growth_rate_list) / 1e5,
    color="crimson",
    s=80,
    zorder=3,
)

for i, label in enumerate(labels):
    plt.annotate(
        label,
        (t_cool_list[i], growth_rate_list[i] / 1e5),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.xscale("log")
plt.xlabel(r"Predicted Cooling Time ($t_{cool}$)", fontsize=12)
plt.ylabel(r"Mass Growth Rate ($dM/dt \times 10^{-5}$)", fontsize=12)
plt.title(
    r"Predicted Cooling Time vs Cold Gas Growth Rate ($t \leq 1.7$)",
    fontsize=14,
)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.savefig(
    os.path.join(OUTPUT_DIR, "tcool_vs_growth_rate.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("✅ Grafiğin hazır: tcool_vs_growth_rate.png")


