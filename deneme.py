import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Kullanılan cihaz: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, init_features=64):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = self._block(features * 4, features * 8)

        self.up3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.up1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.decoder1 = self._block(features * 2, features)

        self.output_layer = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output_layer(d1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

def visualize_predictions(original_image, prediction, slice_indices=None):
    """
    3 farklı düzlemde tahmin sonuçlarını görselleştir
    """
    if slice_indices is None:
        # Orta dilimleri al
        slice_indices = {
            'axial': original_image.shape[2] // 2,
            'sagittal': original_image.shape[0] // 2,
            'coronal': original_image.shape[1] // 2
        }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MR Görüntüsü ve Tahmin Sonuçları - 3 Düzlem', fontsize=16)
    
    # Renk haritası tanımla
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'white']
    cmap = plt.cm.colors.ListedColormap(colors[:len(np.unique(prediction))])
    
    # Axial düzlem
    axes[0, 0].imshow(original_image[:, :, slice_indices['axial']], cmap='gray')
    axes[0, 0].set_title(f'Axial - Orijinal (Slice {slice_indices["axial"]})')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(prediction[:, :, slice_indices['axial']], cmap=cmap, vmin=0, vmax=len(colors)-1)
    axes[1, 0].set_title(f'Axial - Tahmin (Slice {slice_indices["axial"]})')
    axes[1, 0].axis('off')
    
    # Sagittal düzlem
    axes[0, 1].imshow(original_image[slice_indices['sagittal'], :, :], cmap='gray')
    axes[0, 1].set_title(f'Sagittal - Orijinal (Slice {slice_indices["sagittal"]})')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(prediction[slice_indices['sagittal'], :, :], cmap=cmap, vmin=0, vmax=len(colors)-1)
    axes[1, 1].set_title(f'Sagittal - Tahmin (Slice {slice_indices["sagittal"]})')
    axes[1, 1].axis('off')
    
    # Coronal düzlem
    axes[0, 2].imshow(original_image[:, slice_indices['coronal'], :], cmap='gray')
    axes[0, 2].set_title(f'Coronal - Orijinal (Slice {slice_indices["coronal"]})')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(prediction[:, slice_indices['coronal'], :], cmap=cmap, vmin=0, vmax=len(colors)-1)
    axes[1, 2].set_title(f'Coronal - Tahmin (Slice {slice_indices["coronal"]})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Görselleştirme kaydedildi: prediction_visualization.png")

# Modeli yükle ve GPU'ya taşı
print("📦 Model yükleniyor...")
model = UNet3D(in_channels=1, out_channels=7, init_features=64)
model.load_state_dict(torch.load("unet3d_model_v2.pth", map_location=device))
model = model.to(device)  # Modeli GPU'ya taşı
model.eval()

# MR görüntüsünü yükle
print("🧠 MR görüntüsü yükleniyor...")
nii = nib.load("BAGIRSAKCI,_SEMIR,_10501184_t1_mprage_tra_p2_iso_21.nii.gz")
image = nii.get_fdata()
original_image = image.copy()  # Orijinal görüntüyü sakla
image = (image - image.min()) / (image.max() - image.min())
image_tensor = torch.tensor(image[np.newaxis, np.newaxis, ...], dtype=torch.float32).to(device)  # GPU'ya taşı

# Model ile tahmin
print("🤖 Tahmin yapılıyor...")
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # CPU'ya geri al

# Çıktıyı kaydet
output_nii = nib.Nifti1Image(prediction.astype(np.uint8), affine=nii.affine)
nib.save(output_nii, "prediction_output.nii.gz")

print("✅ Tahmin tamamlandı. Çıktı: prediction_output.nii.gz")
print(f"📊 Tahmin şekli: {prediction.shape}")
print(f"🎯 Sınıf değerleri: {np.unique(prediction)}")

# 3 düzlemde görselleştir
print("\n🎨 Görselleştirme oluşturuluyor...")
visualize_predictions(original_image, prediction) 