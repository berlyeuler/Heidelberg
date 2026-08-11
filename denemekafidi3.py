import glob
import os
import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

SIMULATIONS = {
    "sim_3": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3",
    "sim_4": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4",
    "sim_9": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9",
}

for sim_name, sim_path in SIMULATIONS.items():
    pattern = os.path.join(sim_path, "id0", "cloud*.vtk")
    vtk_files = sorted(glob.glob(pattern))

    if not vtk_files:
        continue

    first_vtk = vtk_files[50]
    ds = yt.load(first_vtk)
    ad = ds.all_data()

    temp_yt = np.array(ad["temperature"], dtype=np.float64)
    real_temp = temp_yt / 6.97436478913788e-09
    dens_yt = np.array(ad[("athena", "density")], dtype=np.float64)

    # İki maskenin birleşimi
    combined_mask = (real_temp < 2.0) & (dens_yt >= 600.0)

    # Hücre sayısını buluyoruz
    cell_count = np.sum(combined_mask)  # ya da len(dens_yt[combined_mask])
    print(
        f"{sim_name} -> Number of cells fitting mask: {cell_count:,}"
    )