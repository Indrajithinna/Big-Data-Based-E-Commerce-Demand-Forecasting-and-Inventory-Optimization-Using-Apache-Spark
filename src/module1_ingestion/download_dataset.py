# =============================================================================
# Module 1: Data Collection and Ingestion
# Script: download_dataset.py
# =============================================================================
# PURPOSE:
#   Downloads the UCI Online Retail II dataset from the UCI ML Repository.
#   This is a real-world e-commerce transactional dataset with ~1M records
#   from a UK-based online retailer (2009-2011).
#
# BIS701 RELEVANCE:
#   - Demonstrates data ingestion, the first step in any Big Data pipeline
#   - Handles real-world data in Excel format, requiring format conversion
#   - Prepares data for distributed processing with Spark
#
# INPUT:  UCI dataset URL (from config)
# OUTPUT: data/raw/online_retail.csv (converted from Excel to CSV)
# =============================================================================

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config, ensure_directories


def download_file(url, destination, chunk_size=8192):
    """
    Download a file from URL with progress bar.
    
    Args:
        url: Source URL to download from.
        destination: Local file path to save to.
        chunk_size: Download chunk size in bytes.
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download complete: {os.path.getsize(destination) / (1024*1024):.1f} MB")


def extract_zip(zip_path, extract_to):
    """
    Extract a ZIP file to the specified directory.
    
    Args:
        zip_path: Path to the ZIP file.
        extract_to: Directory to extract contents into.
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
        extracted = z.namelist()
    print(f"Extracted {len(extracted)} files: {extracted}")
    return extracted


def convert_excel_to_csv(excel_path, csv_path):
    """
    Convert the UCI Excel dataset to CSV for easier Spark ingestion.
    
    The UCI Online Retail II dataset comes as an .xlsx file with two sheets
    (Year 2009-2010 and Year 2010-2011). We combine both sheets into one CSV.
    
    NOTE: We use pandas here ONLY for the one-time Excel→CSV conversion.
    All subsequent processing uses PySpark.
    
    Args:
        excel_path: Path to the .xlsx file.
        csv_path: Path to save the combined CSV.
    """
    import pandas as pd  # Only used for Excel reading (one-time conversion)
    
    print(f"Converting Excel to CSV: {excel_path}")
    
    # Read all sheets from the Excel file
    xlsx = pd.ExcelFile(excel_path)
    sheets = xlsx.sheet_names
    print(f"Found sheets: {sheets}")
    
    dfs = []
    for sheet in sheets:
        print(f"  Reading sheet: {sheet}")
        df = pd.read_excel(excel_path, sheet_name=sheet)
        print(f"    Rows: {len(df):,}, Columns: {list(df.columns)}")
        dfs.append(df)
    
    # Combine all sheets
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined):,} rows, {len(combined.columns)} columns")
    print(f"Columns: {list(combined.columns)}")
    print(f"Date range: {combined['InvoiceDate'].min()} to {combined['InvoiceDate'].max()}")
    
    # Save as CSV
    combined.to_csv(csv_path, index=False)
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\nSaved to: {csv_path} ({file_size_mb:.1f} MB)")
    
    return combined.shape


def download_and_prepare_dataset(config=None):
    """
    Main function: download UCI dataset, extract, and convert to CSV.
    
    Args:
        config: Configuration dictionary. Loaded from file if None.
    
    Returns:
        str: Path to the prepared CSV file.
    """
    if config is None:
        config = load_config()
    
    ensure_directories(config)
    
    raw_dir = config["paths"]["raw_data_dir"]
    csv_path = config["paths"]["raw_dataset_file"]
    source_url = config["dataset"]["source_url"]
    
    # Check if CSV already exists
    if os.path.exists(csv_path):
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"Dataset already exists: {csv_path} ({file_size_mb:.1f} MB)")
        print("To re-download, delete the file and run again.")
        return csv_path
    
    # Step 1: Download ZIP
    zip_path = os.path.join(raw_dir, "online_retail_ii.zip")
    if not os.path.exists(zip_path):
        try:
            download_file(source_url, zip_path)
        except requests.exceptions.RequestException as e:
            print(f"\nError downloading dataset: {e}")
            print("\nAlternative: Download manually from:")
            print("  https://archive.ics.uci.edu/dataset/502/online+retail+ii")
            print(f"  Save the .xlsx file to: {raw_dir}")
            return None
    
    # Step 2: Extract ZIP
    extracted = extract_zip(zip_path, raw_dir)
    
    # Step 3: Find the Excel file
    excel_files = [f for f in extracted if f.endswith(".xlsx")]
    if not excel_files:
        # Search raw dir for xlsx files
        excel_files = [f for f in os.listdir(raw_dir) if f.endswith(".xlsx")]
    
    if not excel_files:
        print("Error: No .xlsx file found after extraction.")
        return None
    
    excel_path = os.path.join(raw_dir, excel_files[0])
    print(f"\nFound Excel file: {excel_path}")
    
    # Step 4: Convert to CSV
    rows, cols = convert_excel_to_csv(excel_path, csv_path)
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  File: {csv_path}")
    print(f"  Rows: {rows:,}")
    print(f"  Columns: {cols}")
    print(f"{'='*60}")
    
    return csv_path


if __name__ == "__main__":
    print("=" * 60)
    print("Module 1: Dataset Download & Preparation")
    print("UCI Online Retail II Dataset")
    print("=" * 60)
    result = download_and_prepare_dataset()
    if result:
        print(f"\nReady for next step: python -m src.module1_ingestion.data_generator")
    else:
        print("\nDataset download failed. See instructions above.")
