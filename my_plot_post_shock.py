import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from predict_post_shock import calc_shock_speed, post_shock_ratios
from scipy.stats import qmc

# 1. PARAMETRELERİ ÜRET (seed=34)
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

TARGET_SIMS = [3, 4, 9]
sim_results = []

gamma = 5.0 / 3.0

for i in TARGET_SIMS:
    sim_name = f"sim_{i}"
    csv_path = os.path.join(BASE_DIR, sim_name, "cold_gas_data.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ {sim_name} için CSV bulunamadı!")
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

        mask = time_mid >= 1.0
        avg_growth_rate = (
            np.mean(dm_dt[mask]) / 1e5 if np.any(mask) else np.mean(dm_dt) / 1e5
        )

        rho_bg = X[i, 0]
        rho_w = X[i, 1]
        T_bg = X[i, 2]
        T_w = X[i, 3]
        Mach = X[i, 4]

        # 2. POST-SHOCK HESAPLAMASI (predict_post_shock.py mantığı)
        v_bg_shock, v_wind_shock = calc_shock_speed(
            rho_R=rho_bg, T_R=T_bg, rho_L=rho_w, T_L=T_w, Mach=Mach
        )
        vwind = Mach * np.sqrt(gamma * T_w)

        # Background post-shock
        M_bg_shock = v_bg_shock / np.sqrt(gamma * T_bg)
        bg_shock = post_shock_ratios(M_bg_shock, gamma)

        # Wind post-shock
        v_left_shock_wind_frame = vwind - v_wind_shock
        M_wind_shock = v_left_shock_wind_frame / np.sqrt(gamma * T_w)
        wind_shock = post_shock_ratios(M_wind_shock, gamma)

        # Post-shock fiziksel değerler
        rho_bg_post = bg_shock["density"] * rho_bg
        rho_w_post = wind_shock["density"] * rho_w

        T_bg_post = bg_shock["temperature"] * T_bg
        T_w_post = wind_shock["temperature"] * T_w

        # Post-shock Oranları
        p_rho_ratio = rho_bg_post / rho_w_post
        p_T_ratio = T_bg_post / T_w_post

        sim_results.append(
            {
                "sim": sim_name,
                "post_rho_ratio": p_rho_ratio,
                "post_T_ratio": p_T_ratio,
                "growth_rate": avg_growth_rate,
            }
        )

res_df = pd.DataFrame(sim_results)


def make_scatter(x_col, x_label, filename, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        res_df[x_col],
        res_df["growth_rate"],
        color="darkorange",
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
    print(f"✅ Grafik kaydedildi: {save_path}")


# Post-shock oran grafikleri
make_scatter(
    "post_rho_ratio",
    r"Post-shock Density Ratio ($\rho_{bg, shock} / \rho_{wind, shock}$)",
    "growth_vs_post_shock_rho_ratio.png",
    r"Cold Gas Growth Rate vs Post-Shock Density Ratio",
)

make_scatter(
    "post_T_ratio",
    r"Post-shock Temperature Ratio ($T_{bg, shock} / T_{wind, shock}$)",
    "growth_vs_post_shock_T_ratio.png",
    r"Cold Gas Growth Rate vs Post-Shock Temperature Ratio",
)