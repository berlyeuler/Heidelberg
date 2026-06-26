import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

import glob
import os 
import imageio



ds = yt.load('snap200/id0/cloud.0200.vtk')
ds.field_list
ds.derived_field_list


def create_movie(base_path, field="density", fps=10):
    """Klasördeki tüm vtk dosyalarını döngüye sokarak seçilen alan için

    (density, pressure, temperature) kesit grafikleri çizer ve film üretir.
    """

    if not os.path.exists("frames"):
        os.makedirs("frames")

    print(
        f"/n[STEP 1] {field} snapshots are loading... "
    )

    search_path=base_path.replace("cloud.0200.vtk", "cloud.*.vtk")
    ts= yt.DatasetSeries(search_path)


    print( "[STEP 2] Snapshots are being saved 'frames/' klasörüne kaydediliyor...")
    for ds in ts:
        snap_num= str(ds).split(".")[-2]

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
            p.set_zlim(("gas", "temperature"), 1.0, 300.0)
            p.set_cmap(("gas", "temperature"), "inferno")


        else:
            p= yt.SlicePlot(ds, "z", ("athena", field))
            if field== "density":
                p.set_zlim(("athena", "density"), 0.1, 10.0)
                p.set_cmap(("athena", "density"), "viridis")

            elif field== "pressure":
                p.set_zlim(("athena", "pressure"), 1.0, 1100.0)
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




def slice_function(ds, axis, field):
    """
    Create a slice plot of the specified field along the given axis.

    Parameters:
    ds (yt dataset): The yt dataset to visualize.
    axis (str): The axis along which to slice ('x', 'y', or 'z').
    field (str): The field to visualize (e.g., 'density', 'temperature').

    Returns:
    None
    """

    
  
    # Create a slice plot
    slc = yt.SlicePlot(ds, axis, field)
    slc.save(f'slice_plot_{field}_{axis}.png')  # Save the plot as a PNG file
    # Set the color map and limits
    slc.set_cmap(field, 'viridis')
    slc.set_zlim(field, ds.all_data()[field].min(), ds.all_data()[field].max())
    

 

ad = ds.all_data()
density = ad['density']
erad = ad['Erad']
pressure = ad['pressure']

#a function to show 2 graphs in one plot
def plot_two_graphs(ds, axes, field1, field2):
    """
    Create a plot with two subplots for the specified fields.

    Parameters:
    ds (yt dataset): The yt dataset to visualize.
    axes (list): A list of axes along which to slice ('x', 'y', or 'z').
    field1 (str): The first field to visualize.
    field2 (str): The second field to visualize.

    Returns:
    None
    """
    slc1 = yt.SlicePlot(ds, axes[0], field1)
    slc2 = yt.SlicePlot(ds, axes[1], field2)

    slc1.set_cmap(field1, 'viridis')
    slc2.set_cmap(field2, 'viridis')

    tempf1  = f"temp_{field1}_{axes[0]}.png"
    tempf2  = f"temp_{field2}_{axes[1]}.png"

    slc1.save(tempf1)
    slc2.save(tempf2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    img1 = plt.imread('temp1.png')
    img2 = plt.imread('temp2.png')
    ax1.imshow(img1)
    ax1.axis('off')  
    ax2.imshow(img2)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()





#Now letz make a func for histogram grap

def hist_function(ds, field, bins=30):
    ad = ds.all_data()
    field_data = ad[field]
    plt.hist(field_data, bins=bins)
    plt.xlabel(field)
    plt.ylabel('Frequency')
    plt.title(f'{field} Histogram')
    plt.show()


#density_graph = slice_function(ds, 'z', 'density')
#plt.show()

#den_erad_graph = plot_two_graphs(ds, ['z', 'z'], 'density', 'Erad')
#plt.show()

#pressure_histogram = hist_function(ds, 'pressure', bins= 30)
#plt.show()

def pred_hist_func(ds, field, bins=30):
    M1 = 8.9887
    gamma = 5/3 
    rho1 = 1.0
    T= 10.0

    ad= ds.all_data()

    if field == "density":
        d= ((gamma + 1)* (M1**2)) /  (2 + (gamma - 1) * (M1**2))
        # Gerçek tahmini yoğunluk değeri = Oran * Başlangıç Yoğunluğu
        predicted_value= d* rho1
        field_data = ad[("athena", "density")]
        label_name = f"Predicted Density({predicted_value:.2f})"

    elif field == "pressure":
        p= 1 + (2 * gamma/ (gamma + 1)) * (M1**2)- 1

        p1= rho1* T
        predicted_value= p* p1
        field_data = ad[("athena", "pressure")]
        label_name = f"Predicted Pressure ({predicted_value:.2f})"

    elif field == "temperature":
        d = ((gamma + 1) * (M1**2)) / (2 + (gamma - 1) * (M1**2))
        p = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
        T_ratio = p / d
        predicted_value = T_ratio * T
        field_data = ad[("athena", "pressure")] / ad[("athena", "density")]

        label_name = f"Predicted Temperature ({predicted_value:.2f})"
    else:
        print("Choose 'density', 'pressure' or'temperature'")
        return
    plt.hist(
        field_data, bins=bins, alpha=0.7, color="pink", label="Prediction"
    )
    plt.axvline(
        x=predicted_value,
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        label=label_name,)
    plt.xlabel(field.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"{field.capitalize()} Histogram & Rankine-Hugoniot Prediction")
    plt.legend()

    plt.show()





import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_distributions(data_source, n_bins=25, label_x="Value"):
    """Verilen veri seti için CDF (Birikimli Dağılım) ve CCDF (Ters Birikimli Dağılım)

    grafiklerini yan yana iki panelde teorik çizgisiyle çizer.
    """
    # Veriyi numpy array formatına getiriyoruz (NaN veya boş değerleri temizlemek için)
    data = np.array(data_source)
    data = data[~np.isnan(data)]

    # Teori çizgisinin veriye tam oturması için ortalama (mu) ve standart sapmayı (sigma) veriden hesaplıyoruz
    mu = np.mean(data)
    sigma = np.std(data)

    # 1. Grafik Penceresini Oluşturma (Yan yana 2 panel, eksenleri ortak)
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    axs = fig.subplots(1, 2, sharex=True, sharey=True)

    # Matematiksel teori çizgisi için X ekseni aralığı
    x = np.linspace(data.min(), data.max(), 1000)

    # ---- SOL PANEL: Cumulative (CDF) ----
    # Simülasyon ECDF çizgisi
    axs[0].ecdf(data, label="CDF", color="tab:blue")

    # Adım adım birikimli histogram
    n, bins, patches = axs[0].hist(
        data,
        bins=n_bins,
        density=True,
        histtype="step",
        cumulative=True,
        label="Cumulative histogram",
        color="tab:orange",
    )

    # Teorik Gauss (Normal) Dağılım Hesabı (Sıvı dinamiği/İstatistik teorisi)
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * (1 / sigma * (x - mu)) ** 2
    )
    y = y.cumsum()
    y /= y[-1]  # En yüksek değeri 1'e normalize ediyoruz

    # Teorik çizgiyi sol panele çizdirme
    axs[0].plot(x, y, "k--", linewidth=1.5, label="Theory")
    axs[0].set_title("Cumulative Distributions (CDF)")

    # ---- SAĞ PANEL: Complementary (CCDF) ----
    # Simülasyon Ters ECDF çizgisi
    axs[1].ecdf(data, complementary=True, label="CCDF", color="tab:blue")

    # Ters birikimli histogram (cumulative=-1 ile tersten toplar)
    axs[1].hist(
        data,
        bins=bins,
        density=True,
        histtype="step",
        cumulative=-1,
        label="Reversed cumulative",
        color="tab:orange",
    )

    # Teorik ters çizgiyi sağ panele çizdirme (1 - y)
    axs[1].plot(x, 1 - y, "k--", linewidth=1.5, label="Theory")
    axs[1].set_title("Complementary Cumulative (CCDF)")

    # ---- GENEL SÜSLEMELER ----
    fig.suptitle("Statistical Cumulative Distributions Analysis", fontsize=14)

    for ax in axs:
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        ax.set_xlabel(label_x)
        ax.set_ylabel("Probability of Occurrence")
        ax.label_outer()  # İçteki ortak eksen yazılarını gizler, temiz gösterir

    plt.show()

# Örnek 1: Athena simülasyonundaki yoğunluk (density) verisiyle test etmek için:
#ad = ds.all_data()
#density_data = ad[("athena", "density")]

# Fonksiyonu çağırıyoruz:
#plot_cumulative_distributions(density_data, n_bins=30, label_x="Density Value")

#FOR TESTİNG EVERYTHİNG 
# --- for movie ---- 
# density:
#create_simulation_movie(file_path, field="density", fps=12)

    # pressure:
    # create_simulation_movie(file_path, field="pressure", fps=12)

    # Temperature:
    #create_simulation_movie(file_path, field="temperature", fps=12)

#   ---- for log hists  ---

# (density):
#ad = ds.all_data()
#density_data = ad[("athena", "density")]
#plot_cumulative_distributions(density_data, n_bins=30, label_x="Density Value")


if __name__ == "__main__":
    # Hocanın istediği 4 farklı vtk snapshot dosyasının listesi
    snap_numbers = ["0000", "0010", "0020", "0030"]
    
    # Tüm snapshot'ları AYNI grafik üzerine çizmek için bir döngü kuruyoruz
    for snap in snap_numbers:
        # Her döngüde dosya adını dinamik olarak değiştiriyoruz
        file_path = f"/scratch/hpc-prf-radmix/hpcbeoe/test_beril/cloud.{snap}.vtk"
        
        # Grafiklerin üst üste binmesi için senin ana çizim fonksiyonunu çağırıyoruz
        # NOT: Kodunun yukarılarında ismi 'make_movie' veya 'create_movie' ya da 'plot_histograms' olan, 
        # density histogramı çizen fonksiyonun adı neyse buraya TAM OLARAK ONU yaz:
        make_movie(file_path, field="density")
    
    # Döngü bittikten sonra hepsini tek bir görsel olarak kaydetmesi veya göstermesi için:
    # Eğer fonksiyonun kendi içinde plt.savefig varsa bu satıra gerek kalmayabilir.
    import matplotlib.pyplot as plt
    plt.savefig("/scratch/hpc-prf-radmix/hpcbeoe/test_beril/hepsi_bir_arada.png")
    print("Hocanın istediği gibi tüm histogramlar aynı grafiğe üst üste çizildi!")
