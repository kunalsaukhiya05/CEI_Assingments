import os
import urllib.request
import zipfile
import ssl
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import EuroSAT
import collections

# Fix SSL context for downloading if needed
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = './data'
UCMERCED_URL = 'http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip'
EUROSAT_URL = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip' # EuroSAT rgb version

def download_and_extract(url, extract_to, filename):
    os.makedirs(extract_to, exist_ok=True)
    filepath = os.path.join(extract_to, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            return
            
    print(f"Extracting {filename}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        print(f"Bad zip file: {filepath}")

def setup_datasets():
    print("Setting up datasets...")
    # Extract UC Merced if not extracted
    if not os.path.exists(os.path.join(DATA_DIR, 'UCMerced_LandUse')):
        download_and_extract(UCMERCED_URL, DATA_DIR, 'UCMerced_LandUse.zip')
    
    # Extract EuroSAT if not extracted
    eurosat_dir = os.path.join(DATA_DIR, '2750')
    euro_zip = os.path.join(DATA_DIR, 'EuroSAT.zip')
    if not os.path.exists(eurosat_dir):
        if os.path.exists(euro_zip):
            print("Extracting EuroSAT.zip...")
            try:
                with zipfile.ZipFile(euro_zip, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
            except zipfile.BadZipFile:
                print("Bad zip file: EuroSAT.zip")
        else:
            download_and_extract(EUROSAT_URL, DATA_DIR, 'EuroSAT.zip')
    
    print("Datasets setup complete.")

def plot_class_distribution():
    print("Plotting EuroSAT class distribution...")
    eurosat_dir = os.path.join(DATA_DIR, '2750') # The extracted EuroSAT dir is typically named '2750'
    if not os.path.exists(eurosat_dir):
        print(f"Could not find EuroSAT dir: {eurosat_dir}")
        return
        
    classes = os.listdir(eurosat_dir)
    class_counts = {}
    for c in classes:
        c_path = os.path.join(eurosat_dir, c)
        if os.path.isdir(c_path):
            class_counts[c] = len(glob.glob(os.path.join(c_path, '*')))
            
    # Sort for better visualization
    class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('EuroSAT Class Distribution')
    plt.xlabel('Land-Use Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eurosat_class_distribution.png')
    plt.close()
    print("Saved eurosat_class_distribution.png")

def spatial_block_split(val_ratio=0.2, block_size=100):
    """
    Simulates spatial block splitting.
    Assuming images are sequentially numbered (e.g., AnnualCrop_1.jpg, AnnualCrop_2.jpg),
    sequential images might be spatially correlated. We group them into 'blocks' and split
    at the block level to prevent spatial leakage into the validation set.
    """
    print("Performing spatial block split for EuroSAT...")
    eurosat_dir = os.path.join(DATA_DIR, '2750')
    if not os.path.exists(eurosat_dir):
        print(f"Could not find EuroSAT dir: {eurosat_dir}")
        return

    train_files = []
    val_files = []
    
    classes = [d for d in os.listdir(eurosat_dir) if os.path.isdir(os.path.join(eurosat_dir, d))]
    for c in classes:
        c_path = os.path.join(eurosat_dir, c)
        files = sorted(glob.glob(os.path.join(c_path, '*.jpg')))
        
        # Group into blocks
        blocks = [files[i:i + block_size] for i in range(0, len(files), block_size)]
        
        # Shuffle blocks (simulating random selection of geographic regions)
        random.seed(42)
        random.shuffle(blocks)
        
        # Split blocks
        n_val_blocks = max(1, int(len(blocks) * val_ratio))
        val_blocks = blocks[:n_val_blocks]
        train_blocks = blocks[n_val_blocks:]
        
        for b in val_blocks:
            val_files.extend(b)
        for b in train_blocks:
            train_files.extend(b)
            
    print(f"Total training images: {len(train_files)}")
    print(f"Total validation images: {len(val_files)}")
    
    # Save splits to text files
    os.makedirs(os.path.join(DATA_DIR, 'splits'), exist_ok=True)
    with open(os.path.join(DATA_DIR, 'splits', 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    with open(os.path.join(DATA_DIR, 'splits', 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))
        
    print("Saved block splits to data/splits/train.txt and val.txt")

if __name__ == '__main__':
    setup_datasets()
    plot_class_distribution()
    spatial_block_split()
