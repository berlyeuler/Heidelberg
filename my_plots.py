import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

# 1. PARAMETRELERİ AYNEN ÜRET (seed=34 sabit olduğu için indeksler sim_X ile birebir eşleşir)
nvars = 5
nsamples = 10
sampler = qmc.LatinHypercube(d=nvars, seed=34)
Xinit = sampler.random(n=nsamples)

lower = [1.0, 1.0, 1.0, 1.0, 1.1]
upper = [50.0, 50.0, 20.0, 100.0, 5.0]
X = qmc.scale(Xinit, lower, upper)

BASE_DIR = "/scratch/hpc-prf-radmix/hpcbeoe"
OUTPUT_DIR = os.path.join(BASE_DIR, "cold_gas_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sadece bakmak istediğimiz simülasyon indeksleri:
TARGET_SIMS = [3, 4, 9]

sim_results = []

for i in TARGET_SIMS:
    sim_name = f"sim_{i}"
    csv_path = os.path.join(BASE_DIR, sim_name, "cold_gas_data.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ {sim_name} için CSV bulunamadı: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    time_col = [c for c in df.columns if "time" in c.lower()][0]
    mass_col = [
        c for c in df.columns if "mass" in c.lower() or "cold" in c.lower()
    ][0]

    times = df[time_col].values
    cold_masses = df[mass_col].values

    if len(times) > 1:
        dm_dt = np.diff(cold_masses) / np.diff(times)
        time_mid = (times[:-1] + times[1:]) / 2.0

        # time >= 1.0 sonrası kararlı/doygunluk büyüme hızı ortalaması
        mask = time_mid >= 1.0
        if np.any(mask):
            avg_growth_rate = np.mean(dm_dt[mask]) / 1e5
        else:
            avg_growth_rate = np.mean(dm_dt) / 1e5

        rho_bg = X[i, 0]
        rho_w = X[i, 1]
        T_bg = X[i, 2]
        T_w = X[i, 3]
        Mach = X[i, 4]

        sim_results.append(
            {
                "sim": sim_name,
                "rho_bg": rho_bg,
                "rho_w": rho_w,
                "T_bg": T_bg,
                "T_w": T_w,
                "Mach": Mach,
                "rho_ratio": rho_bg / rho_w,
                "T_ratio": T_bg / T_w,
                "growth_rate": avg_growth_rate,
            }
        )

res_df = pd.DataFrame(sim_results)


def make_scatter(x_col, x_label, filename, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        res_df[x_col],
        res_df["growth_rate"],
        color="royalblue",
        s=140,
        edgecolors="black",
        zorder=3,
    )

    for _, row in res_df.iterrows():
        plt.annotate(
            row["sim"],
            (row[x_col], row["growth_rate"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
            weight="bold",
        )

    plt.title(title, fontsize=13)
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(r"Avg Growth Rate ($dM/dt \times 10^{-5}$)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Grafik hazır: {save_path}")


# Hocanın istediği 2 ana oran grafiği
make_scatter(
    "rho_ratio",
    r"Density Ratio ($\rho_{bg} / \rho_{wind}$)",
    "growth_vs_rho_ratio.png",
    r"Cold Gas Growth Rate vs Density Ratio ($\rho_{bg}/\rho_{wind}$)",
)
make_scatter(
    "T_ratio",
    r"Temperature Ratio ($T_{bg} / T_{wind}$)",
    "growth_vs_T_ratio.png",
    r"Cold Gas Growth Rate vs Temperature Ratio ($T_{bg}/T_{wind}$)",
)