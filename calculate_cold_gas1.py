import os
import glob
import numpy as np
import pandas as pd
import yt

# yt loglarını sessize al
yt.funcs.mylog.setLevel(50)

SIMULATIONS = {
    "sim_3": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3",
    "sim_4": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4",
    "sim_6": "/scratch/hpc-prf-radmix/hpcbeoe/sim_6",
    "sim_9": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9",
}

def calculate_and_save(sim_name, sim_path, density_threshold=600.0, temp_threshold_code=2.0):
    pattern = os.path.join(sim_path, "id0", "cloud*.vtk")
    vtk_files = sorted(glob.glob(pattern))

    if not vtk_files:
        print(f"⚠️ Uyarı: {sim_path} içinde VTK dosyası bulunamadı!")
        return

    times = []
    dense_cold_mass = []

    print(f"\n==========================================")
    print(f"🚀 Processing Simulation: {sim_name}")
    print(f"==========================================")

    for vtk_file in vtk_files:
        snap_name = os.path.basename(vtk_file)
        print(f"--> Reading snapshot: {snap_name}", flush=True)

        ds = yt.load(vtk_file)
        sim_time = float(ds.current_time)
        ad = ds.all_data()

        # Sıcaklık, yoğunluk ve kütle verileri
        temp_yt = np.array(ad["temperature"], dtype=np.float64)
        real_temp = temp_yt / 6.97436478913788e-09
        dens_yt = np.array(ad[("athena", "density")], dtype=np.float64)
        mass_yt = np.array(ad[("gas", "mass")], dtype=np.float64)

        mask = (real_temp < temp_threshold_code) & (dens_yt >= density_threshold)
        total_mass = float(mass_yt[mask].sum())

        times.append(sim_time)
        dense_cold_mass.append(total_mass)

        print(f"    Time: {sim_time:.2f} | Mass: {total_mass:.2e}", flush=True)

    # Hesaplama biter bitmez CSV'ye kaydediyoruz
    df = pd.DataFrame({"time": times, "dense_cold_mass": dense_cold_mass})
    save_path = os.path.join(sim_path, "cold_gas_data.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ Data saved successfully to: {save_path}")

if __name__ == "__main__":
    for sim_name, sim_path in SIMULATIONS.items():
        if os.path.exists(sim_path):
            calculate_and_save(sim_name, sim_path)
        else:
            print(f"Klasör bulunamadı: {sim_path}")