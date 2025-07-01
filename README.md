# 🧠 MRI Segmentasyon Uygulaması

Bu uygulama, UNet3D mimarisi kullanarak MRI görüntülerinde otomatik beyin segmentasyonu yapan bir web uygulamasıdır.

## 🚀 Özellikler

- **Çoklu Format Desteği**: `.nii`, `.nii.gz`, `.dcm`, `.zip` dosya formatlarını destekler
- **Otomatik Format Dönüşümü**: DICOM dosyalarını otomatik olarak NIfTI formatına çevirir
- **3D Segmentasyon**: 7 farklı beyin bölgesini segmentasyon yapar
- **Modern Web Arayüzü**: Kullanıcı dostu, responsive tasarım
- **Gerçek Zamanlı Görselleştirme**: Axial, Coronal, Sagittal kesitlerde sonuçları gösterir
- **Sonuç İndirme**: Segmentasyon sonuçlarını `.npy` formatında indirme

## 📋 Gereksinimler

- Python 3.8+
- CUDA destekli GPU (opsiyonel, CPU da kullanılabilir)
- En az 8GB RAM (büyük dosyalar için önerilir)

## 🛠️ Kurulum

1. **Projeyi klonlayın:**
```bash
git clone <repository-url>
cd cnn-segmentation
```

2. **Sanal ortam oluşturun (önerilir):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Model dosyasının varlığını kontrol edin:**
- `unet3d_model_v2.pth` dosyasının proje klasöründe olduğundan emin olun

## 🚀 Çalıştırma

```bash
python app.py
```

Uygulama `http://localhost:5000` adresinde çalışmaya başlayacaktır.

## 📁 Desteklenen Dosya Formatları

| Format | Açıklama |
|--------|----------|
| `.nii` | NIfTI formatı |
| `.nii.gz` | Sıkıştırılmış NIfTI formatı |
| `.dcm` | Tek DICOM dosyası |
| `.zip` | DICOM dosyalarını içeren ZIP arşivi |

## 🧠 Model Detayları

- **Mimari**: UNet3D
- **Input Shape**: `[1, 1, 128, 128, 128]` (batch, channel, depth, height, width)
- **Output**: 7 sınıflı segmentasyon maskesi
- **Sınıflar**: 0-6 arası beyin bölgesi etiketleri

## 📊 Kullanım

1. Web tarayıcınızda `http://localhost:5000` adresine gidin
2. "Dosya Seç" butonuna tıklayın veya dosyayı sürükleyip bırakın
3. Desteklenen formatlardaki MRI dosyanızı seçin
4. Segmentasyon işleminin tamamlanmasını bekleyin
5. Sonuçları görüntüleyin ve gerekirse indirin

## 🔧 Teknik Detaylar

### Dosya İşleme
- DICOM dosyaları otomatik olarak NIfTI formatına çevrilir
- Görüntüler `[128, 128, 128]` boyutuna yeniden boyutlandırılır
- Min-max normalizasyonu uygulanır

### Segmentasyon
- Model GPU varsa GPU'da, yoksa CPU'da çalışır
- 7 sınıflı segmentasyon yapılır
- Sonuçlar 3D numpy array olarak döndürülür

### Görselleştirme
- Axial, Coronal, Sagittal kesitler gösterilir
- Renkli segmentasyon maskesi
- Yüksek çözünürlüklü PNG formatında

## 📁 Proje Yapısı

```
cnn-segmentation/
├── app.py                 # Ana Flask uygulaması
├── unet3d_model_v2.pth    # Eğitilmiş model
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
├── templates/
│   └── index.html        # Web arayüzü
├── uploads/              # Geçici yükleme klasörü
└── results/              # Sonuç dosyaları
```

## ⚠️ Önemli Notlar

- Maksimum dosya boyutu: 500MB
- İşlem süresi dosya boyutuna ve donanıma bağlı olarak değişir
- Büyük dosyalar için yeterli RAM olduğundan emin olun
- GPU kullanımı işlem süresini önemli ölçüde azaltır

## 🐛 Sorun Giderme

### Yaygın Hatalar

1. **Model dosyası bulunamadı:**
   - `unet3d_model_v2.pth` dosyasının proje klasöründe olduğunu kontrol edin

2. **Bellek hatası:**
   - Daha küçük dosyalar deneyin
   - RAM'i artırın
   - GPU kullanın

3. **Format hatası:**
   - Desteklenen formatlardan birini kullandığınızdan emin olun
   - DICOM dosyalarının tam olduğunu kontrol edin

## 📞 Destek

Herhangi bir sorun yaşarsanız, lütfen issue açın veya iletişime geçin.

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. 