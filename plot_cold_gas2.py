import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Server üzerinde çökmesini engeller
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

SIMULATIONS = {
    "sim_3": {"path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3", "label": "Sim 3"},
    "sim_4": {"path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4", "label": "Sim 4"},
    "sim_6": {"path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_6", "label": "Sim 6"},
    "sim_9": {"path": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9", "label": "Sim 9"},
}

OUTPUT_DIR = "/scratch/hpc-prf-radmix/hpcbeoe/cold_gas_plots"

def plot_from_saved_csv():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for sim_key, sim_info in SIMULATIONS.items():
        csv_path = os.path.join(sim_info["path"], "cold_gas_data.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️ {sim_key} için CSV bulunamadı, atlanıyor: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        plt.plot(df["time"], df["dense_cold_mass"], marker='o', markersize=3, linestyle='-', label=sim_info.get("label", sim_key))

    # Hata veren \ge sembolünü düzelttik (>= yaptık)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Dense Cold Gas Mass (T < 20,000 K & Density >= 600)", fontsize=12)
    plt.title("Cold Gas Mass Evolution Across Simulations", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="best", frameon=True)

    save_path = os.path.join(OUTPUT_DIR, "combined_dense_cold_gas.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafik 1 saniyede başarıyla kaydedildi: {save_path}")

if __name__ == "__main__":28
    plot_from_saved_csv()