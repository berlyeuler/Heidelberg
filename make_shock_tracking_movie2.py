import os
import sys
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import yt

# --- EŞİK DEĞERLERİ ---
T_TRESH = 50.0  
D_TRESH = 50.0 #ama aşağıda tekrar tanımladık ve bu kod onu kullanıyor



# Ayrışma eşiği (32/512 = 0.0625)
SEPERATION_THRESHOLD = 32 / 512

sys.path.append("/pc2/users/h/hpccado/repos/athena-ccd-maxbg-bssdl-fork/")
from predict_post_shock import calc_shock_speed

var = "temperature"
T_min = 6.97436478913788e-09

# --- SİMÜLASYON TANIMLARI ---
simulations_config = {
    "sim_4": {
        "path": "/scratch/hpc-prf-radmix/cdoughty/reruns/sim_4",
        "params": {"rho_bg": 49.550, "rho_wind": 33.770, "T_bg": 5.420, "T_wind": 73.990, "M_wind": 1.380}

    },

    "sim_12": {
        "path": "/scratch/hpc-prf-radmix/cdoughty/reruns/sim_12",
        "params": {"rho_bg": 39.291, "rho_wind": 17.062, "T_bg": 7.941, "T_wind": 28.059, "M_wind": 4.671}
    },

    "sim_14": {
        "path": "/scratch/hpc-prf-radmix/cdoughty/reruns/sim_14",
        "params": {"rho_bg": 43.135, "rho_wind": 31.498, "T_bg": 14.068, "T_wind": 13.235, "M_wind": 3.961}
    },

}


# --- FONKSİYON (FRAME OLUŞTURMA) ---
# 
def make_comparison_frame(ds, variable, vshock_bg_lab, counter, output_dir, rho_bg): 
    #veri kümesi, çizilecek değişken, şok hızı, sayaç, çıktı dizini ve arka plan yoğunluğu parametrelerini alan fonksiyonu tanımlar.
    
    #-------- DEĞERLERİ ALIYORUZ ----------- 
    time = float(ds.current_time)
    #Veri kümesindeki mevcut simülasyon süresini alır ve sayısal (float) değere dönüştürür.
    
    slc = yt.SlicePlot(ds, "z", ("gas", variable))
    #yt kütüphanesini kullanarak z-ekseni boyunca belirtilen
    #gaz değişkeninin kesit Plot (dilim) nesnesini oluşturur.

    slc_data = ds.slice("z", 0.0)
    #z = 0.0 düzleminden ham simülasyon verilerini içeren bir
    # dilim (slice) nesnesi çıkarır.
    
    x_arr = slc_data["gas", "x"].v
    #Dilimdeki tüm hücrelerin x-koordinat verilerini 
    # birimlerinden arındırılmış saf NumPy dizisi (.v) olarak alır.
    y_arr = slc_data["gas", "y"].v

    temp_norm = slc_data["gas", "temperature"].v / T_min
    #Gaz sıcaklığı değerlerini alır ve T_min değerine 
    #bölerek normalize edilmiş sıcaklık dizisini hesaplar.

    density_arr = slc_data["gas", "density"].v 
    #Dilimdeki hücrelerin gaz yoğunluğu (density) 
    # değerlerini ham NumPy dizisi olarak alır.

    #-----------------------------------------------------------------------
    #--------- FİLTRELEMELER YAPIYORUZ--------------------------------------

    # y bandındaki verileri filtrele : Aşağıdaki adımın y-eksenindeki dar 
    # bir bant verisini filtreleyeceğini belirtir.
    y_mask = np.abs(y_arr) < 2.0
    #np.abs, NumPy kütüphanesinde bulunan ve bir sayının mutlak değerini 
    # hesaplayan bir fonksiyondur.
    #y-koordinatının mutlak değeri 2.0'dan küçük olan noktaları belirleyen 
    # mantıksal (True/False) bir maske oluşturur.

    x_filtered = x_arr[y_mask]
    #x-koordinatlarını sadece seçilen y bandına (y_mask) uyacak şekilde filtreler.

    temp_filtered = temp_norm[y_mask]
    #Normalize sıcaklık değerlerini sadece seçilen y bandına göre filtreler.

    dens_filtered = density_arr[y_mask]

    #------- x değerlerine göre sırala: Verilerin x-ekseni boyunca sıralanacağını belirtir.--------

    sort_idx = np.argsort(x_filtered)
    #Filtrelenmiş x-koordinatlarını küçükten büyüğe sıralayacak indeks dizisini hesaplar.

    x_line = x_filtered[sort_idx]
    #x-koordinatlarını sıralı indekslere göre dizerek profil çizgisini (line profile) oluşturur.

    temp_line = temp_filtered[sort_idx]
    density_line = dens_filtered[sort_idx]

    #------------------------------------------------------------------------------


    # Eşiği rho_bg parametresi üzerinden tanımla
    D_TRESH = rho_bg * 1.5 
    #Arka plan yoğunluğunun (rho_bg) 1.5 katını alarak 
    # tespit işlemleri için bir yoğunluk eşik değeri (D_TRESH) belirler.

  #*********************************************


    # --- SADECE SAĞ KENARLARI BUL ---
    #---- sıcaklık ve yoğunluk profillerinin x-ekseni üzerindeki 
    # en sağ sınırlarının (ön cephelerinin) bulunacağını belirtir.-------

    # Sıcaklık için sağ kenar
    temp_above = temp_line > T_TRESH
    #Sıcaklık değerlerinin belirlenen sıcaklık eşiğinden (T_TRESH) 
    # büyük olduğu noktaları içeren mantıksal (True/False) bir dizi oluşturur.

    if np.any(temp_above): 
        #Eşikten yüksek en az bir sıcaklık değeri olup olmadığını kontrol eder.

        temp_indices = np.where(temp_above)[0]
        #Sıcaklığın eşiği geçtiği hücrelerin indeks numaralarını bir dizi olarak alır.

        right_temp_pos = x_line[temp_indices[-1]]
        #Eşiği geçen son indeksi ([-1]) alarak, yüksek sıcaklık 
        # bölgesinin x-eksenindeki en sağ konumunu tespit eder.
        
    else:
        right_temp_pos = None

    # Yoğunluk için sağ kenar

    dens_above = density_line > D_TRESH
    #Yoğunluk değerlerinin önceden hesaplanan yoğunluk eşiğinden 
    # (D_TRESH) büyük olduğu noktaları belirleyen mantıksal bir dizi oluşturur.

    if np.any(dens_above):
        #Eşikten yüksek en az bir yoğunluk değeri olup olmadığını kontrol eder.

        dens_indices = np.where(dens_above)[0]

        right_dens_pos = x_line[dens_indices[-1]]
    else:
        right_dens_pos = None

   #-------------------------------------------------------------------------------------

    # ----------- AYRIŞMA MESAFESİ VE ŞOK DURUMU --------------
    #--------------- Şok cephesinin aktiflik durumunun ve konumunun belirleneceği bölümün başlangıcını gösterir.------------------
    if right_temp_pos is not None and right_dens_pos is not None:
        #Hem sıcaklık hem de yoğunluk için geçerli birer sağ kenar konumu bulunup bulunmadığını kontrol eder.

        separation = abs(right_temp_pos - right_dens_pos)
        #Sıcaklık ve yoğunluk cepheleri arasındaki x-ekseni mesafesini (ayrışmayı) mutlak değer olarak hesaplar.
        
        shock_active = separation <= SEPERATION_THRESHOLD
        #Ayrışma mesafesi belirlenen threshold (eşik) değerinden küçükse şokun aktif 
        # olduğunu ifade eden mantıksal (True/False) değeri atar.
        
        if shock_active:
            x_tracked = (right_temp_pos + right_dens_pos) / 2
            #Sıcaklık ve yoğunluk cephelerinin ortalamasını 
            # alarak takip edilecek şok konumunu belirler.

        else:
            x_tracked = right_dens_pos
    else:
        return time, np.nan, np.nan, vshock_bg_lab * time, False
        # Şok tespit edilemediği için fonksiyonu sonlandırır; zamanı, 
        # veri eksikliğini belirten np.nan değerlerini, teorik şok konumunu 
        # (vshock_bg_lab * time) ve şokun pasif olduğunu gösteren 
        # False bayrağını döndürür.

    #-------------------------------------------------------------------------------
    # ---------------------ÇİZGİLERİ ÇİZ--------------------------------------------
    x_predicted = vshock_bg_lab * time
    
    # Teorik tahmin (CYAN): Teorik şok konumu için camgöbeği 
    # (cyan) renkte çizgi ekleneceğini belirtir.
    
    slc.annotate_line(
        (x_predicted, 8.0, 0.0), (x_predicted, -8.0, 0.0),
        coord_system="data", color="cyan"
    )
    #yt görselinde teorik şok konumuna (x_predicted) 
    # $y = 8.0$ ile $y = -8.0$ arasında dikey,
    # düz bir camgöbeği çizgi çizer.

    # Takip edilen şok (KIRMIZI) - SADECE AKTİFSE
    if shock_active:
        #Kırmızı şok çizgisinin sadece şok aktif durumdayken çizilmesini sağlar.

        slc.annotate_line(
            (x_tracked, 8.0, 0.0), (x_tracked, -8.0, 0.0),
            coord_system="data", color="red",
            plot_args={"linestyle": "--", "linewidth": 2.5}
        )
        #Sıcaklık ve yoğunluk verisinden tespit edilen gerçek şok 
        # konumuna (x_tracked) $y \in 
        # [-8, 8]$ aralığında 2.5 kalınlığında dikey, kesikli 
        # (--) kırmızı bir çizgi ekler.

    # Başlangıç sınırı (BEYAZ): x = 0 başlangıç (referans) çizgisine dair açıklama.
    slc.annotate_line(
        (0.0, 8.0, 0.0), (0.0, -8.0, 0.0),
        coord_system="data", color="w"
    )

    slc.render()
    #Tüm eklenen çizgi ve işaretlemeleri yt görsel nesnesi üzerine işler ve görseli oluşturur.

    # --- LEJANT ---
    #--- Grafiğe eklenecek açıklama kutusunun (lejant/legend) hazırlanacağı bölümün başlangıcını gösterir.---

    track_label = f"Shock Front ({x_tracked:.2f})" if shock_active else "Shock Dissipated"
    #Şok aktifse iki basamak hassasiyetli konumla birlikte "Shock Front (konum verisi)" yazar; 
    #şok aktif değilse (sönümlenmişse) "Shock Dissipated..." metnini atar.

    legend_lines = [
        #Grafik lejantında kullanılacak özel çizgi stillerini tutan bir liste başlatır.

        Line2D([0], [0], color="cyan", linewidth=2, label=f"Predicted ({x_predicted:.2f})"),
        Line2D([0], [0], color="red" if shock_active else "gray", linewidth=2, 
               linestyle="--" if shock_active else "-", label=track_label),
        Line2D([0], [0], color="w", linewidth=1.5, label="Initial Boundary"),
    ]


    ax = slc.plots[("gas", variable)].axes
    #yt dilim görselinin Matplotlib eksen (axes) nesnesine erişerek 
    # doğrudan grafik üzerinde çizim yapabilmeyi sağlar.

    ax.legend(handles=legend_lines, loc="upper right", framealpha=0.85)
    #Oluşturulan çizgi stillerini (legend_lines) sağ üst köşeye ("upper right"), 
    # %85 opaklıkta bir arka plan kutusuyla lejant olarak ekler.

    if variable == "density":
        slc.set_zlim(("gas", "density"), 10, 1000)
        #Yoğunluk grafiğinin renk ölçeği (z-ekseni) 
        # alt ve üst sınırlarını 10 ile 1000 arasında sabitler.

    elif variable == "temperature":
        slc.set_zlim(("gas", "temperature"), T_min, 200 * T_min)
        #Sıcaklık grafiğinin renk ölçeği sınırlarını 
        # T_min ile 200 * T_min arasında sabitler.

    p = slc.plots[("gas", variable)]
    #İşlenen ilgili değişkene ait yt Plot nesnesini bir değişkene atar.

    p.figure.set_size_inches(10, 5)
    #Çıktı grafiğinin boyutunu 10 x 5 inç olarak ayarlar.

    frame_name = os.path.join(output_dir, f"frame_{variable}_{counter:04d}.png")
    #Çıktı klasörü, değişken adı ve 4 basamaklı sayaç numarasını 
    # (örneğin frame_density_0001.png) birleştirerek dosya kaydetme yolunu oluşturur.

    p.figure.savefig(frame_name, bbox_inches="tight", dpi=100)
    #Grafiği, kenar boşluklarını kırparak (bbox_inches="tight") 
    # ve 100 DPI çözünürlükte belirtilen dosya yoluna kaydeder.
    
    return time, separation, x_tracked, x_predicted, shock_active
    #Fonksiyonun sonucunda mevcut zamanı, iki cephe arasındaki mesafeyi, 
    # takip edilen konumu, teorik konumu ve şokun aktiflik durumunu döndürür.



# ==========================================
# --- TÜM SİMÜLASYONLAR İÇİN DÖNGÜ BAŞLIYOR ---
# ==========================================

for sim_name, config in simulations_config.items():
    #Tüm simülasyon konfigürasyonlarını içeren sözlük (simulations_config) 
    # üzerinde döngü 
    # başlatarak her simülasyonun adını (sim_name) ve ayarlarını 
    # (config) sırayla alır.

    print(f"\n==================================================")
    print(f"🚀 İŞLENİYOR: {sim_name}")
    print(f"==================================================")
    
    sim_dir = config["path"]
    #Mevcut simülasyona ait veri dosyalarının bulunduğu klasör yolunu (path) alır.

    p = config["params"]
    #Simülasyona ait fiziksel parametreler sözlüğünü 
    # (params) kısa bir değişken adı olan p'ye atar.
    
    # ---------------- 1. Şok Hızını Hesapla -------------------------------
    #breakpoint()


    vshock_bg_lab, vshock_wind_lab, v_shocked_bg = calc_shock_speed(
        p["rho_bg"], p["T_bg"], p["rho_wind"], p["T_wind"], p["M_wind"]
    )
    #Fiziksel parametreleri (rho_bg, T_bg, rho_wind, T_wind, M_wind) kullanarak 
    # calc_shock_speed fonksiyonu üzerinden laboratuvar çerçevesindeki farklı 
    # şok hızlarını hesaplar.
    
    #-------------------------------
    # 2. Snapshot'ları Bul: Simülasyon zaman adımı (snapshot) 
    # dosyalarının aranacağını belirtir.
    snaps = sorted(glob(f"{sim_dir}/id0/*.vtk")) + sorted(glob(f"{sim_dir}/*.vtk"))
    #Simülasyon klasöründeki .vtk uzantılı zaman adımı dosyalarını bulur,
    # alfabetik/sayısal olarak sıralar ve bir listede toplar.

    if not snaps:
        #Eğer .vtk uzantılı hiç dosya bulunamadıysa çalışacak koşulu başlatır. 
        # ama bizimkiler ztn vtk filelar o yuzden aslında bu 
        # kısma cok da gerek yok chat bilmedigi icin eklemis

        snaps = sorted(glob(f"{sim_dir}/id0/*.athdf")) + sorted(glob(f"{sim_dir}/*.athdf"))

    print(f"📁 [{sim_name}] Bulunan snapshot sayısı: {len(snaps)}")
    if not snaps:
        print(f"⚠️ [{sim_name}] İçin hiç snapshot bulunamadı, atlanıyor!")
        continue


    #------------------
    # 3. Çıktı Dizinlerini Hazırla: Çıktı klasörlerinin ve veri dizilerinin hazırlanacağını belirtir.
    output_frames_dir = f"frames_{sim_name}"
    #Kaydedilecek resim kareleri (frame) için 
    # simülasyon adına özel bir klasör ismi tanımlar.

    os.makedirs(output_frames_dir, exist_ok=True)
    #Tanımlanan klasörü oluşturur; klasör zaten varsa 
    # hata vermesini engeller (exist_ok=True).

    times = []  #Zaman değerlerini kaydetmek için boş bir liste oluşturur.
    separations = []
    shock_positions = []
    predicted_positions = []
    shock_status = []



    #----------------- 4. Frame'leri İşle (İlk 40 snapshot)-----------------
    # Zaman adımı görsellerinin işleneceği döngü bölümünün başlangıcını gösterir.
    counter = 0
    #İşlenen görsel/frame sayısını takip etmek için bir sayaç değişkeni tanımlayıp 0'a eşitler.

    for snap in snaps[:40]:
        #Bulunan snapshot dosyalarının sadece ilk 40 tanesi ([:40]) üzerinde dönen bir döngü başlatır.

        try:
            #Herhangi bir zaman adımında okuma/işleme hatası olursa 
            # kodun çökmesini önlemek için hata yakalama bloğu açar.

            ds = yt.load(snap)
            #yt kütüphanesini kullanarak o anki snapshot veri dosyasını hafızaya yükler.

            result = make_comparison_frame(ds, var, vshock_bg_lab, counter, output_frames_dir, p["rho_bg"])
            #Hazırlanan make_comparison_frame fonksiyonunu çağırarak görseli kaydeder 
            # ve analiz sonuçlarını result değişkenine döndürür.

            time, separation, x_tracked, x_pred, active = result
            #Fonksiyondan dönen result demetindeki (tuple) değerleri ilgili değişkenlere 
            # (time, separation, vb.) ayrı ayrı paketinden çıkararak dağıtır (unpacking).

            
            if not np.isnan(separation):
                #Ayrışma mesafesi (separation) geçersiz/boş (NaN) değilse, 
                # yani başarılı bir tespit yapıldıysa çalışacak koşulu başlatır.

                times.append(time)
                separations.append(separation)
                shock_positions.append(x_tracked)
                predicted_positions.append(x_pred)
                shock_status.append(active)
        except Exception as e: 
            #try bloğu içinde herhangi bir hata meydana gelirse hatayı yakalar 
            # ve hata mesajını e değişkenine atar.
            print(f"  ❌ Frame {counter} hatası ({snap}): {e}")
            #Hata oluşan frame numarasını, dosya adını ve hatanın detayını terminale yazar.
            
        counter += 1 #Bir sonraki dosya için sayacı 1 artırır.

    # 5. GRAFİK 1: SEPARATION VS TIME
    if times:
        plt.figure(figsize=(12, 6))
        plt.plot(times, separations, 'r-o', label='Separation', linewidth=2, markersize=6)
        plt.axhline(y=SEPERATION_THRESHOLD, color='k', linestyle='--', 
                    label=f'Threshold ({SEPERATION_THRESHOLD:.4f})', linewidth=2)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Separation (x position difference)', fontsize=12)
        plt.title(f'[{sim_name}] Shock Front Separation vs Time', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'separation_vs_time_{sim_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. GRAFİK 2: SHOCK POSITION VS TIME
    if times:
        plt.figure(figsize=(12, 6))
        plt.plot(times, shock_positions, 'r-o', label='Tracked Shock', linewidth=2, markersize=6)
        #Zamana karşılık ayrışma mesafesini kırmızı renkte ('r'), çizgi ve yuvarlak noktalarla ('-o') grafiğe çizer.
        plt.plot(times, predicted_positions, 'c-s', label='Predicted Shock', linewidth=2, markersize=6)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Shock Position (x)', fontsize=12)
        plt.title(f'[{sim_name}] Shock Position: Tracked vs Predicted', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'shock_position_vs_time_{sim_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 7. VİDEO OLUŞTURMA
    frames = sorted(glob(f"{output_frames_dir}/*{var}*.png"))
    #Çıktı klasöründeki ilgili değişkene (var) ait tüm .png görsellerini arar 
    # ve zaman sırasına göre dizilmeleri için sıralı bir liste haline getirir.

    if frames:
        #Klasörde işlenecek en az bir resim karesi (frame) bulunup bulunmadığını kontrol eder.

        movie_name = f"{sim_name}_tracked_vs_predicted_{var}.mp4"
        #Oluşturulacak .mp4 uzantılı video dosyasının adını simülasyon adı ve değişken bilgisiyle tanımlar.

        W, H = 1040, 496
        #Videonun genişlik ($W = 1040$ piksel) ve yükseklik ($H = 496$ piksel) standart boyutlarını belirler.


        print(f"🎬 Video oluşturuluyor ({sim_name})... ({len(frames)} frame)")
        #Kaç adet resim karesi kullanılarak video oluşturulduğunu terminale yazdırır.

        with imageio.get_writer(movie_name, fps=8) as writer:
            #Saniyede 8 kare (fps=8) oynatacak şekilde imageio kütüphanesi üzerinden 
            # video yazıcı nesnesini açar (güvenli dosya yönetimi için with bloğu kullanılır).

            for f in frames:
                #Sıralanmış resim dosyası yolları (frames) üzerinde dönen bir döngü başlatır.

                img = imageio.imread(f)
                #O anki resim dosyasını okunarak sayısal bir piksel dizisi (NumPy array) olarak hafızaya yükler.

                if img.shape[1] != W or img.shape[0] != H:
                    #Resmin genişliğinin (shape[1]) veya yüksekliğinin (shape[0]) hedeflenen
                    #  $1040 \times 496$ boyutlarından farklı olup olmadığını kontrol eder.

                    img_pil = Image.fromarray(img)
                    #Piksel dizisini Pillow (PIL) kütüphanesinin işleyebileceği bir görsel nesnesine dönüştürür.

                    img_pil = img_pil.resize((W, H), Image.Resampling.LANCZOS)
                    #Görseli yüksek kaliteli LANCZOS yeniden örnekleme algoritması 
                    # kullanarak tam 1040 x 496 boyutlarına ölçeklendirir.

                    img = np.array(img_pil)
                    #Yeniden boyutlandırılan PIL görselini tekrar imageio'nun yazabileceği 
                    # NumPy dizisi formatına çevirir.

                writer.append_data(img)
                #Boyutları doğrulanmış/düzeltilmiş resim karesini video dosyasına bir kare olarak ekler.

        print(f"✅ {sim_name} tamamlandı! Video: {movie_name}\n")

print("🎉 TÜM SİMÜLASYONLAR BAŞARIYLA İŞLENDİ!")
