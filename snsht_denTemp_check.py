import glob
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yt

yt.funcs.mylog.setLevel(50)

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


def plot_mass_histogram():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    masses_dict = {}

    print("🚀 Extracting cell masses after masks...")

    for sim_key, sim_info in SIMULATIONS.items():
        pattern = os.path.join(sim_info["path"], "id0", "cloud*.vtk")
        vtk_files = sorted(glob.glob(pattern))

        if not vtk_files:
            continue

        first_vtk = vtk_files[50]
        ds = yt.load(first_vtk)
        ad = ds.all_data()

        # Verileri çekiyoruz
        temp_yt = np.array(ad["temperature"], dtype=np.float64)
        real_temp = temp_yt / 6.97436478913788e-09
        dens_yt = np.array(ad[("athena", "density")], dtype=np.float64)
        mass_yt = np.array(ad[("gas", "mass")], dtype=np.float64)

        # İki maskeyi BİRLİKTE uyguluyoruz (Hem cold hem dense olanlar)
        combined_mask = (real_temp < 2.0) & (dens_yt >= 600.0)

        masses_dict[sim_info["label"]] = mass_yt[combined_mask]

    # --- COMBINED MASS HISTOGRAM ---
    plt.figure(figsize=(9, 6))
    for label, mass_data in masses_dict.items():
        sns.histplot(
            mass_data,
            label=label,
            element="step",
            stat="density",
            log_scale=True,  # Kütle genelde çok geniş ölçektedir, log harika görünür
            alpha=0.3,
            fill=True,
        )

    plt.title(
        "Cell Mass Distribution (T < 20,000 K & Density >= 600)", fontsize=13
    )
    plt.xlabel("Cell Mass", fontsize=11)
    plt.ylabel("Normalized Cell Count", fontsize=11)
    plt.legend(loc="best")

    save_path = os.path.join(OUTPUT_DIR, "combined_mass_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Kütle histogramı kaydedildi: {save_path}")


if __name__ == "__main__":
    plot_mass_histogram()