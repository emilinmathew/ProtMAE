import os
import urllib.request
import tarfile
import json
import gzip

def download_proteinnet():
    """Download ProteinNet data for secondary structure prediction"""
    
    # Create data directory
    os.makedirs('proteinnet_data', exist_ok=True)
    
    # Download CASP12 (largest dataset) - Text-based format is easier to work with
    casp12_url = "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp12.tar.gz"
    casp12_path = "proteinnet_data/casp12.tar.gz"
    
    print("Downloading CASP12 dataset...")
    if not os.path.exists(casp12_path):
        urllib.request.urlretrieve(casp12_url, casp12_path)
        print("Downloaded CASP12 dataset")
    else:
        print("CASP12 already exists")
    
    # Extract the tar file
    print("Extracting CASP12...")
    with tarfile.open(casp12_path, 'r:gz') as tar:
        tar.extractall('proteinnet_data/')
    
    # Download secondary structure annotations
    ss_url = "https://www.dropbox.com/s/sne2ak1woy1lrqr/full_protein_dssp_annotations.json.gz?dl=1"
    ss_path = "proteinnet_data/dssp_annotations.json.gz"
    
    print("Downloading secondary structure annotations...")
    if not os.path.exists(ss_path):
        urllib.request.urlretrieve(ss_url, ss_path)
        print("Downloaded secondary structure annotations")
    
    # Extract secondary structure data
    print("Extracting secondary structure annotations...")
    with gzip.open(ss_path, 'rt') as f:
        ss_data = json.load(f)
    
    # Save as regular JSON for easier access
    with open('proteinnet_data/dssp_annotations.json', 'w') as f:
        json.dump(ss_data, f)
    
    print("Setup complete! Data is in 'proteinnet_data/' directory")
    print("Files:")
    for root, dirs, files in os.walk('proteinnet_data'):
        for file in files:
            print(f"  {os.path.join(root, file)}")

if __name__ == "__main__":
    download_proteinnet()