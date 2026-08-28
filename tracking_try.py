from glob import glob
import numpy as np
import imageio.v2 as imageio
from PIL import Image

# Değişken ve simülasyon bilgileri
variable = "temperature"  # veya "density"
model_name = "sim_15_try4"

# 1. Resimlerin kaydedildiği GERÇEK dosya kalıbı:
frames = glob(f"frames/frame_{variable}_*.png")
frames.sort()

print(f"Toplam {len(frames)} adet frame bulundu.")

if len(frames) == 0:
    print("❌ HATA: Görseller bulunamadı! Lütfen 'frames/' klasörünü kontrol et.")
    exit()

# 2. Hocanın belirttiği sabit genişlik ve yükseklik
W, H = 1040, 496

output_video = f"{model_name}_{variable}.mp4"

# 3. MP4 videosunu yeniden boyutlandırarak kaydetme
with imageio.get_writer(output_video, format="FFMPEG", fps=10) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        # Her görseli (1040, 496) boyutuna getirip diziye çeviriyoruz
        image = np.array(Image.fromarray(image).resize((W, H)))
        writer.append_data(image)

print(f"🎬 Video başarıyla oluşturuldu: {output_video}")
