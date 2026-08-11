import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

GOOD_SIMULATIONS = {
    "sim_6": {
        "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_6",
        "label": "Sim 6",
    },
    "sim_7": {
        "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_7",
        "label": "Sim 7",
    },
    "sim_9": {
        "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9",
        "label": "Sim 9",
    },
}

OUTPUT_DIR = "/scratch/hpc-prf-radmix/hpcbeoe/cold_gas_plots"


def plot_histograms():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. DENSITY HISTOGRAM ---
    plt.figure(figsize=(10, 6))
    for sim_key, sim_info in GOOD_SIMULATIONS.items():
        csv_path = os.path.join(sim_info["path"], "filtered_densities.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Logaritmik ölçekte histogram/KDE çizimi
            sns.histplot(
                df["density"],
                label=sim_info["label"],
                element="step",
                stat="density",
                common_norm=False,
                log_scale=True,
                alpha=0.4,
            )

    plt.xlabel("Density (Filtered: Density >= 600)", fontsize=12)
    plt.ylabel("Normalized Cell Count / Density", fontsize=12)
    plt.title("Density Distribution Across Best 3 Models", fontsize=14)
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "density_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✅ Density Histogram kaydedildi!")

    # --- 2. TEMPERATURE HISTOGRAM ---
    plt.figure(figsize=(10, 6))
    for sim_key, sim_info in GOOD_SIMULATIONS.items():
        csv_path = os.path.join(sim_info["path"], "filtered_temperatures.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sns.histplot(
                df["temperature"],
                label=sim_info["label"],
                element="step",
                stat="density",
                common_norm=False,
                log_scale=True,
                alpha=0.4,
            )

    plt.xlabel("Temperature (Filtered: T < 20,000 K)", fontsize=12)
    plt.ylabel("Normalized Cell Count / Density", fontsize=12)
    plt.title("Temperature Distribution Across Best 3 Models", fontsize=14)
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "temperature_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✅ Temperature Histogram kaydedildi!")


if __name__ == "__main__":
    plot_histograms()