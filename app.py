import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import gzip
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import zipfile
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use('Agg')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# UNet3D Model Sınıfı
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, init_features=64):
        super(UNet3D, self).__init__()
        features = init_features

        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)

        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        # Output
        self.output_layer = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.up2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.output_layer(dec1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

# Model yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=1, out_channels=7)
model.load_state_dict(torch.load("unet3d_model_v2.pth", map_location=device))
model = model.to(device)
model.eval()

def convert_dicom_to_nii(dicom_folder, output_path):
    """DICOM klasörünü NIfTI dosyasına çevirir"""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        if not dicom_names:
            return False
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, output_path)
        return True
    except Exception as e:
        print(f"DICOM çevirme hatası: {e}")
        return False

def ensure_nii_gz_format(src_path, dst_path):
    """Dosyayı .nii.gz formatına çevirir"""
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return dst_path
    if src_path.endswith(".nii"):
        with open(src_path, 'rb') as f_in:
            with gzip.open(dst_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return dst_path
    elif src_path.endswith(".nii.gz"):
        shutil.copy(src_path, dst_path)
        return dst_path
    return None

def process_mri_file(file_path):
    """MRI dosyasını işler ve model için hazırlar"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Dosya uzantısını kontrol et
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.zip':
            # ZIP dosyasını aç
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # DICOM dosyalarını ara
            dicom_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
            
            if dicom_files:
                # DICOM klasörünü NIfTI'ye çevir
                dicom_folder = os.path.dirname(dicom_files[0])
                output_nii = os.path.join(temp_dir, "image.nii.gz")
                if convert_dicom_to_nii(dicom_folder, output_nii):
                    return output_nii
        else:
            # Doğrudan NIfTI dosyası
            output_nii = os.path.join(temp_dir, "image.nii.gz")
            if ensure_nii_gz_format(file_path, output_nii):
                return output_nii
    
    except Exception as e:
        print(f"Dosya işleme hatası: {e}")
    
    return None

def process_dicom_folder(files):
    """DICOM dosyalarını içeren klasörü işler"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # DICOM dosyalarını geçici klasöre kaydet
        dicom_files = []
        for file in files:
            if file.filename.lower().endswith('.dcm'):
                file_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(file_path)
                dicom_files.append(file_path)
        
        if dicom_files:
            # DICOM klasörünü NIfTI'ye çevir
            output_nii = os.path.join(temp_dir, "image.nii.gz")
            if convert_dicom_to_nii(temp_dir, output_nii):
                return output_nii
    
    except Exception as e:
        print(f"DICOM klasör işleme hatası: {e}")
    
    return None

def predict_segmentation(image_path):
    """MRI görüntüsünde segmentasyon tahmini yapar"""
    try:
        # Görüntüyü yükle
        nii = nib.load(image_path)
        image = nii.get_fdata()
        original_image = image.copy()  # Orijinal görüntüyü sakla
        
        # Normalize et
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Tensor'a çevir
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # GPU belleğini temizle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Görüntüyü daha küçük boyutlara ölçeklendir (GPU belleği için)
        target_size = (96, 96, 96)  # Daha küçük boyut
        image_tensor = F.interpolate(image_tensor, size=target_size, mode='trilinear', align_corners=False)
        
        # GPU'ya taşı
        image_tensor = image_tensor.to(device)
        
        # Tahmin yap
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Tahmin sonucunu orijinal boyuta geri ölçeklendir
            prediction_tensor = torch.tensor(prediction, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            prediction_resized = F.interpolate(prediction_tensor, size=original_image.shape, mode='nearest')
            prediction = prediction_resized.squeeze().numpy().astype(np.uint8)
        
        # GPU belleğini temizle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return original_image, prediction
    
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        # GPU belleğini temizle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None

def create_visualization(original_image, prediction, slice_indices=None):
    """3 farklı düzlemde tahmin sonuçlarını görselleştirir"""
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
    
    # Base64'e çevir
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str

def save_results_and_create_zip(original_image, prediction, nii_affine, results_folder):
    """
    3 düzlemde orijinal ve tahmin görsellerini PNG olarak, segmentasyon maskesini .nii.gz olarak kaydeder ve zipler.
    Zip dosyasının yolunu döndürür.
    """
    import tempfile
    import zipfile
    from PIL import Image
    import nibabel as nib
    import numpy as np
    import os

    # Geçici klasör
    temp_dir = tempfile.mkdtemp()
    slice_indices = {
        'axial': original_image.shape[2] // 2,
        'sagittal': original_image.shape[0] // 2,
        'coronal': original_image.shape[1] // 2
    }
    # PNG dosyalarını kaydet
    for plane, idx in slice_indices.items():
        # Orijinal
        orig_img = original_image.copy()
        if plane == 'axial':
            img = orig_img[:, :, idx]
        elif plane == 'sagittal':
            img = orig_img[idx, :, :]
        elif plane == 'coronal':
            img = orig_img[:, idx, :]
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        im = Image.fromarray(img)
        orig_path = os.path.join(temp_dir, f'original_{plane}.png')
        im.save(orig_path)
        # Tahmin
        pred = prediction.copy()
        if plane == 'axial':
            pred_img = pred[:, :, idx]
        elif plane == 'sagittal':
            pred_img = pred[idx, :, :]
        elif plane == 'coronal':
            pred_img = pred[:, idx, :]
        pred_img = (pred_img * (255 // (np.max(pred_img) if np.max(pred_img) > 0 else 1))).astype(np.uint8)
        pred_im = Image.fromarray(pred_img)
        pred_path = os.path.join(temp_dir, f'prediction_{plane}.png')
        pred_im.save(pred_path)
    # 3D segmentasyon maskesini .nii.gz olarak kaydet
    nii_path = os.path.join(temp_dir, 'segmentation_output.nii.gz')
    nii_img = nib.Nifti1Image(prediction.astype(np.uint8), affine=nii_affine)
    nib.save(nii_img, nii_path)
    # Zip dosyası oluştur
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'segmentation_results_{timestamp}.zip'
    zip_path = os.path.join(results_folder, zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for fname in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, fname)
            zipf.write(fpath, arcname=fname)
    # Geçici klasörü temizlemeye gerek yok, sistem otomatik temizler
    return zip_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    try:
        # Klasör yükleme kontrolü
        is_folder = request.form.get('is_folder', 'false').lower() == 'true'
        
        if is_folder or len(files) > 1:
            # Klasör yükleme - DICOM dosyaları
            processed_path = process_dicom_folder(files)
        else:
            # Tek dosya yükleme
            file = files[0]
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # MRI dosyasını işle
            processed_path = process_mri_file(file_path)
            
            # Geçici dosyayı temizle
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if not processed_path:
            return jsonify({'error': 'MRI dosyası işlenemedi'}), 400
        
        # Segmentasyon tahmini yap
        original_image, prediction = predict_segmentation(processed_path)
        if original_image is None or prediction is None:
            return jsonify({'error': 'Segmentasyon tahmini başarısız'}), 400
        
        # Görselleştirme oluştur
        visualization = create_visualization(original_image, prediction)
        
        # Sonucu kaydet
        result_filename = f'result_{len(os.listdir(app.config["RESULTS_FOLDER"])) + 1}.npy'
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        np.save(result_path, prediction)
        
        # --- ZIP dosyasını oluştur ---
        nii_affine = nib.load(processed_path).affine
        zip_filename = save_results_and_create_zip(original_image, prediction, nii_affine, app.config['RESULTS_FOLDER'])
        # ---
        
        return jsonify({
            'success': True,
            'visualization': visualization,
            'message': 'Segmentasyon tamamlandı!',
            'result_file': result_filename,
            'zip_file': zip_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'İşlem hatası: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    else:
        return jsonify({'error': 'Dosya bulunamadı'}), 404

@app.route('/download/zip/<zipname>')
def download_zip(zipname):
    zip_path = os.path.join(app.config['RESULTS_FOLDER'], zipname)
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    else:
        return jsonify({'error': 'Zip dosyası bulunamadı'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 