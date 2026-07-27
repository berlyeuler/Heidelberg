import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SIMULATIONS = {
    "sim_3": {
        "csv": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3/cold_gas_data.csv",
        "label": "Sim 3",
    },
    "sim_4": {
        "csv": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4/cold_gas_data.csv",
        "label": "Sim 4",
    },
    "sim_9": {
        "csv": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9/cold_gas_data.csv",
        "label": "Sim 9",
    },
}

OUTPUT_DIR = "/scratch/hpc-prf-radmix/hpcbeoe/cold_gas_plots"
NORM_FACTOR = 1e5  # Hocanın istediği normalizasyon sabiti


def run_growth_rate_from_csv():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for sim_key, sim_info in SIMULATIONS.items():
        csv_path = sim_info["csv"]

        if not os.path.exists(csv_path):
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

            # 1. Y değerlerini 1e5'e bölüyoruz
            dm_dt_normalized = dm_dt / NORM_FACTOR

            plt.plot(
                time_mid,
                dm_dt_normalized,
                label=f"{sim_info['label']} (dM/dt)",
                linewidth=2,
            )

    plt.title("Cold Gas Growth Rate Over Time (dM/dt)", fontsize=14)
    plt.xlabel("Time", fontsize=12)

    # 2. Y ekseni etiketini güncelliyoruz
    plt.ylabel(r"Growth Rate ($dM/dt \times 10^{-5}$)", fontsize=12)

    # 3. X eksenini time=2.5'te kesiyoruz
    plt.xlim(right=2.5)

    plt.legend(loc="best")

    save_path = os.path.join(
        OUTPUT_DIR, "cold_gas_growth_rate_comparison.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Harika! Düzeltilmiş grafik kaydedildi: {save_path}")


if __name__ == "__main__":
    run_growth_rate_from_csv()