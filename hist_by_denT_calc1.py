import glob
import os
import numpy as np
import pandas as pd
import yt

yt.funcs.mylog.setLevel(50)

# Hocanın istediği 3 iyi model (Sim klasör yollarını kontrol et)
GOOD_SIMULATIONS = {
    "sim_3": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3",
    "sim_4": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4",
    "sim_9": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9",
}

DENSITY_CUT = 600.0
TEMP_CUT_CODE = 2.0  # Real temp < 20,000 K karşılığı


def extract_filtered_cell_data(sim_name, sim_path):
    pattern = os.path.join(sim_path, "id0", "cloud*.vtk")
    vtk_files = sorted(glob.glob(pattern))

    if not vtk_files:
        print(f"⚠️ {sim_name} içinde VTK dosyası bulunamadı!")
        return

    filtered_densities = []
    filtered_temperatures = []

    print(f"\n==========================================")
    print(f"🚀 Extracting Histogram Data: {sim_name}")
    print(f"==========================================")

    # Bütün snapshot'lardaki filtreye uyan hücrelerin değerlerini topluyoruz
    for vtk_file in vtk_files:
        snap_name = os.path.basename(vtk_file)
        print(f"--> Reading: {snap_name}", flush=True)

        ds = yt.load(vtk_file)
        ad = ds.all_data()

        temp_yt = np.array(ad["temperature"], dtype=np.float64)
        real_temp = temp_yt / 6.97436478913788e-09
        dens_yt = np.array(ad[("athena", "density")], dtype=np.float64)

        # 1. Density mask sonrası kalan yoğunluklar
        dens_mask = dens_yt >= DENSITY_CUT
        filtered_densities.extend(dens_yt[dens_mask])

        # 2. Temperature mask sonrası kalan sıcaklıklar
        temp_mask = real_temp < TEMP_CUT_CODE
        filtered_temperatures.extend(real_temp[temp_mask])

    # Verileri CSV olarak kaydedelim (Boyut çok büyük olmasın diye numpy compressed .npz de yapabiliriz ama CSV pratik)
    df_dens = pd.DataFrame({"density": filtered_densities})
    df_temp = pd.DataFrame({"temperature": filtered_temperatures})

    df_dens.to_csv(
        os.path.join(sim_path, "filtered_densities.csv"), index=False
    )
    df_temp.to_csv(
        os.path.join(sim_path, "filtered_temperatures.csv"), index=False
    )

    print(f"✅ {sim_name} histogram verileri kaydedildi!")


if __name__ == "__main__":
    for sim_name, sim_path in GOOD_SIMULATIONS.items():
        if os.path.exists(sim_path):
            extract_filtered_cell_data(sim_name, sim_path)