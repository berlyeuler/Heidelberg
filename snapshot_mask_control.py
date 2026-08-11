import glob
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yt

yt.funcs.mylog.setLevel(50)

# Hocanın istediği 3 model
SIMULATIONS = {
    "sim_3": {
        "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3",
        "label": "Sim 3",
    },
    "sim_4": {
        "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4",
        "label": "Sim 4",
    },
    "sim_9": {
        "path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9",
        "label": "Sim 9",
    },
}

OUTPUT_DIR = "/scratch/hpc-prf-radmix/hpcbeoe/cold_gas_plots"


def plot_combined_mask_histograms():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    densities_dict = {}
    temperatures_dict = {}

    print("🚀 Reading single snapshot from each simulation...")

    for sim_key, sim_info in SIMULATIONS.items():
        pattern = os.path.join(sim_info["path"], "id0", "cloud*.vtk")
        vtk_files = sorted(glob.glob(pattern))

        if not vtk_files:
            print(f"⚠️ {sim_key} içinde VTK bulunamadı, atlanıyor.")
            continue

        # Sadece ilk snapshot'ı okuyoruz
        first_vtk = vtk_files[50]
        print(f"--> Processing {sim_key}: {os.path.basename(first_vtk)}")

        ds = yt.load(first_vtk)
        ad = ds.all_data()

        temp_yt = np.array(ad["temperature"], dtype=np.float64)
        real_temp = temp_yt / 6.97436478913788e-09
        dens_yt = np.array(ad[("athena", "density")], dtype=np.float64)

        # Maskeleri uygulayıp kaydediyoruz
        dens_mask = dens_yt >= 600.0
        temp_mask = real_temp < 2.0

        densities_dict[sim_info["label"]] = dens_yt[dens_mask]
        temperatures_dict[sim_info["label"]] = real_temp[temp_mask]

    # --- 1. COMBINED DENSITY HISTOGRAM ---
    plt.figure(figsize=(9, 6))
    for label, dens_data in densities_dict.items():
        sns.histplot(
            dens_data,
            label=label,
            element="step",
            stat="density",
            log_scale=True,
            alpha=0.3,
            fill=True,
        )

    plt.title(
        "Density Mask Check across Models (Density >= 600)", fontsize=13
    )
    plt.xlabel("Density", fontsize=11)
    plt.ylabel("Normalized Cell Count", fontsize=11)
    plt.legend(loc="best")
    dens_save_path = os.path.join(
        OUTPUT_DIR, "combined_density_histogram.png"
    )
    plt.savefig(dens_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2. COMBINED TEMPERATURE HISTOGRAM (DÜZELTİLMİŞ) ---
    plt.figure(figsize=(9, 6))
    for label, temp_data in temperatures_dict.items():
    # log_scale=True KALDIRILDI, bins=50 eklendi
     sns.histplot(
        temp_data,
        label=label,
        element="step",
        stat="density",
        bins=50,
        alpha=0.3,
        fill=True,
    )

    plt.title(
        "Temperature Mask Check across Models (T < 20,000 K)", fontsize=13
    )
    plt.xlabel("Temperature (Code Units)", fontsize=11)
    plt.ylabel("Normalized Cell Count", fontsize=11)
    plt.xlim(0.9, 2.1)  # Verinin olduğu aralığa odaklanıyoruz
    plt.legend(loc="best")
    temp_save_path = os.path.join(
        OUTPUT_DIR, "combined_temperature_histogram.png"
    )
    plt.savefig(temp_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ İki birleşik histogram da kaydedildi:")
    print(f" 1. {dens_save_path}")
    print(f" 2. {temp_save_path}")


if __name__ == "__main__":
    plot_combined_mask_histograms()