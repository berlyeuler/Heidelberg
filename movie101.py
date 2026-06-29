import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

import glob
import os 
import imageio
import sys



# ds = yt.load('/scratch/hpc-prf-radmix/hpcbeoe/test_beril/id0/cloud.0200.vtk')
# ds.field_list
# ds.derived_field_list


def create_movie(base_path, field="density", fps=10):
    """Klasördeki tüm vtk dosyalarını döngüye sokarak seçilen alan için

    (density, pressure, temperature) kesit grafikleri çizer ve film üretir.
    """

    if not os.path.exists("frames"):
        os.makedirs("frames")

    print(
        f"/n[STEP 1] {field} snapshots are loading... "
    )

    # search_path=base_path.replace("cloud.0200.vtk", "cloud.*.vtk")
    ts= yt.DatasetSeries(f"{base_path}/cloud*.vtk")





    #sys.exit()



    print( "[STEP 2] Snapshots are being saved 'frames/' klasörüne kaydediliyor...")
    for ds in ts:
        snap_num= str(ds).split(".")[-1]
        #print(str(ds))
        #sys.exit()

        if field =="temperature":
            def _temp(field, data):
                return data[("athena", "pressure")] / data[("athena", "density" )]
            
            ds.add_field(
                ("gas", "temperature"),
                function= _temp,
                sampling_type="cell",
                units="",
          
            )
            p= yt.SlicePlot(ds,"z", ("gas", "temperature"))
            #p.set_zlim(("gas", "temperature"), 1.0, 300.0)
            p.set_cmap(("gas", "temperature"), "inferno")


        else:
            p= yt.SlicePlot(ds, "z", ("athena", field))
            if field== "density":
               # p.set_zlim(("athena", "density"), 0.1, 10.0)
                p.set_cmap(("athena", "density"), "viridis")

            elif field== "pressure":
               # p.set_zlim(("athena", "pressure"), 1.0, 1100.0)
                p.set_cmap(("athena", "pressure"), "magma")

        p.save(f"frames/frame_{field}_{snap_num}.png")

    print ("All snapshots are saved in 'frames/ field")


def make_movie(field="density", suffix="custom", spesific_snaps=None, fps=10):
    print(f"'{field}'s video is preparing")

    # Bak, buralara 4 boşluk (veya 1 TAB) girinti ekliyoruz:
    frames = sorted(glob.glob("frames/*_density*.png"))
    if not frames:
        frames = sorted(glob.glob("*_Slice_z_density.png"))

        if not frames:
            print("Errror, couldnt find a snapshot in 'frames/' file you should first run the 'create movie' ")
            return

    
    if spesific_snaps is not None:
        selected_frames= []
        for snap in spesific_snaps:
            snap_str = f"{snap:04d}"
            matching_frame= [f for f in frames if f"{snap_str}.png" in f]
            if matching_frame:
                selected_frames.append(matching_frame[0])

        frames = selected_frames
        print("wait")

    movie_name = f"movie_{field}_{suffix}.mp4"
    with imageio.get_writer(movie_name, format="FFMPEG", fps=fps) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    
create_movie('/scratch/hpc-prf-radmix/hpcbeoe/test_beril/id0')