import zipfile
import os

# ZIP dosyasının yolu
zip_path = 'garbage-classification-v2.zip'
extract_dir = 'data/'

# ZIP dosyasını açma
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Dosya başarıyla açıldı ve 'data/' klasörüne yerleştirildi.")