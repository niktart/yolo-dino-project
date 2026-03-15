import os
import zipfile
import shutil
from pathlib import Path


def mount_drive():
    """Монтирует Google Drive (для Colab)"""
    from google.colab import drive
    drive.mount('/content/drive')
    print("? Google Drive mounted")


def extract_zip(zip_path, extract_to):
    """Распаковывает zip архив"""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"? Extracted {zip_path} to {extract_to}")


def extract_all_zips_from_dir(source_dir, target_dir):
    """
    Распаковывает все zip архивы из папки в отдельные подпапки
    
    Args:
        source_dir: папка с zip архивами
        target_dir: целевая папка (каждый архив распакуется в отдельную подпапку)
    """
    os.makedirs(target_dir, exist_ok=True)
    
    zip_files = [f for f in os.listdir(source_dir) if f.endswith(".zip")]
    print(f"?? Найдено архивов: {zip_files}")
    
    for zip_file in zip_files:
        zip_path = os.path.join(source_dir, zip_file)
        
        # Создаем имя папки из имени zip файла (без расширения)
        folder_name = os.path.splitext(zip_file)[0]
        # Очищаем имя от специальных символов
        folder_name = folder_name.replace(" ", "_").replace(".", "_").lower()
        
        extract_path = os.path.join(target_dir, folder_name)
        
        print(f"  ?? Распаковка {zip_file} -> {extract_path}")
        extract_zip(zip_path, extract_path)
    
    return zip_files


def copy_random_to_data(random_source, data_target):
    """Копирует random датасет в общую папку data"""
    random_dest = os.path.join(data_target, "random_dataset")
    
    if os.path.exists(random_dest):
        shutil.rmtree(random_dest)
    
    shutil.copytree(random_source, random_dest)
    print(f"? Random датасет скопирован в {random_dest}")
    return random_dest