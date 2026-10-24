# =============================================================================
# Module 1: Data Collection and Ingestion
# Script: data_generator.py
# =============================================================================
# PURPOSE:
#   Enriches the UCI Online Retail dataset with simulated fields that are
#   required for the inventory optimization module but are NOT present in
#   the original dataset.
#
#   SIMULATED FIELDS (clearly marked — NOT real business data):
#     - Product_Category: Assigned from a predefined list based on StockCode
#     - Discount: Random realistic discount (0-30%)
#     - Inventory_Level: Simulated based on product demand patterns
#     - Supplier: Randomly assigned supplier IDs
#     - Delivery_Date: Order_Date + random lead time (3-14 days)
#     - Revenue: Derived as Quantity × Unit_Price × (1 - Discount)
#
# BIS701 RELEVANCE:
#   - Demonstrates data enrichment and augmentation in Big Data pipelines
#   - Simulates realistic business data for end-to-end pipeline testing
#   - Shows ETL (Extract, Transform, Load) concepts
#
# INPUT:  data/raw/online_retail.csv (from download_dataset.py)
# OUTPUT: data/raw/ecommerce_sales_enriched.csv (enriched dataset)
# =============================================================================

import os
import sys
import hashlib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config, ensure_directories


# --- Predefined category mapping ---
# These categories are SIMULATED. They are assigned based on the first
# character(s) of StockCode to create a realistic distribution.
PRODUCT_CATEGORIES = [
    "Home Decor",
    "Kitchen & Dining",
    "Garden & Outdoor",
    "Gifts & Novelties",
    "Stationery & Craft",
    "Bags & Accessories",
    "Lighting",
    "Toys & Games",
    "Seasonal & Holiday",
    "Textiles & Fabrics",
    "Electronics & Gadgets",
    "Health & Beauty",
]

# Supplier names (SIMULATED)
SUPPLIER_NAMES = [
    f"Supplier_{chr(65 + i)}" for i in range(20)  # Supplier_A through Supplier_T
]


def assign_category(stock_code, seed_str, categories):
    """
    Assign a product category based on a deterministic hash of the StockCode.
    This ensures the same product always gets the same category.
    
    Args:
        stock_code: Product stock code string.
        seed_str: String to use for hashing.
        categories: List of category names.
    
    Returns:
        str: Assigned category name.
    """
    # Use hash for deterministic but seemingly random assignment
    hash_val = int(hashlib.md5(str(seed_str).encode()).hexdigest(), 16)
    return categories[hash_val % len(categories)]


def generate_enriched_dataset(config=None):
    """
    Read the raw UCI dataset CSV and add simulated fields for the project.
    
    Uses pandas for this one-time enrichment step. The enriched CSV will
    then be loaded into Spark for all subsequent Big Data processing.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        str: Path to the enriched CSV file.
    """
    import pandas as pd  # Used only for one-time enrichment
    
    if config is None:
        config = load_config()
    
    ensure_directories(config)
    
    raw_csv = config["paths"]["raw_dataset_file"]
    enriched_csv = config["paths"]["enriched_dataset_file"]
    sim_config = config["dataset"]["simulation"]
    col_mapping = config["dataset"]["column_mapping"]
    
    # Check if enriched file exists
    if os.path.exists(enriched_csv):
        file_size_mb = os.path.getsize(enriched_csv) / (1024 * 1024)
        print(f"Enriched dataset already exists: {enriched_csv} ({file_size_mb:.1f} MB)")
        print("Delete it to regenerate.")
        return enriched_csv
    
    # Check if raw CSV exists
    if not os.path.exists(raw_csv):
        print(f"Error: Raw dataset not found at {raw_csv}")
        print("Run download_dataset.py first.")
        return None
    
    print(f"Loading raw dataset: {raw_csv}")
    df = pd.read_csv(raw_csv, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Set random seed for reproducibility
    np.random.seed(sim_config["seed"])
    
    # -------------------------------------------------------------------------
    # Step 1: Rename columns to project schema
    # -------------------------------------------------------------------------
    print("\n[Step 1] Renaming columns to project schema...")
    df = df.rename(columns=col_mapping)
    print(f"  Columns after renaming: {list(df.columns)}")
    
    # -------------------------------------------------------------------------
    # Step 2: Add SIMULATED Product_Category
    # -------------------------------------------------------------------------
    print("[Step 2] Assigning SIMULATED product categories...")
    categories = PRODUCT_CATEGORIES[:sim_config["num_categories"]]
    df["Product_Category"] = df["Product_ID"].apply(
        lambda x: assign_category(x, x, categories)
    )
    cat_dist = df["Product_Category"].value_counts()
    print(f"  Category distribution (top 5):")
    for cat, count in cat_dist.head().items():
        print(f"    {cat}: {count:,}")
    
    # -------------------------------------------------------------------------
    # Step 3: Add SIMULATED Discount
    # -------------------------------------------------------------------------
    print("[Step 3] Generating SIMULATED discounts...")
    min_disc, max_disc = sim_config["discount_range"]
    # Most products have no discount; ~30% have some discount
    has_discount = np.random.random(len(df)) < 0.30
    discount_values = np.round(
        np.random.uniform(min_disc + 0.05, max_disc, len(df)), 2
    )
    df["Discount"] = np.where(has_discount, discount_values, 0.0)
    print(f"  Products with discount: {has_discount.sum():,} ({has_discount.mean()*100:.1f}%)")
    print(f"  Avg discount (when applied): {df.loc[df['Discount']>0, 'Discount'].mean():.2%}")
    
    # -------------------------------------------------------------------------
    # Step 4: Calculate DERIVED Revenue
    # -------------------------------------------------------------------------
    print("[Step 4] Calculating DERIVED revenue...")
    df["Revenue"] = np.round(df["Quantity"] * df["Unit_Price"] * (1 - df["Discount"]), 2)
    print(f"  Total revenue: ${df['Revenue'].sum():,.2f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Add SIMULATED Supplier
    # -------------------------------------------------------------------------
    print("[Step 5] Assigning SIMULATED suppliers...")
    suppliers = SUPPLIER_NAMES[:sim_config["num_suppliers"]]
    df["Supplier"] = df["Product_ID"].apply(
        lambda x: assign_category(x, f"supplier_{x}", suppliers)
    )
    print(f"  Unique suppliers: {df['Supplier'].nunique()}")
    
    # -------------------------------------------------------------------------
    # Step 6: Add SIMULATED Delivery_Date
    # -------------------------------------------------------------------------
    print("[Step 6] Generating SIMULATED delivery dates...")
    min_lead, max_lead = sim_config["lead_time_range"]
    lead_times = np.random.randint(min_lead, max_lead + 1, len(df))
    df["Delivery_Date"] = df["Order_Date"] + pd.to_timedelta(lead_times, unit="D")
    print(f"  Avg lead time: {lead_times.mean():.1f} days")
    
    # -------------------------------------------------------------------------
    # Step 7: Add SIMULATED Inventory_Level (per product)
    # -------------------------------------------------------------------------
    print("[Step 7] Generating SIMULATED inventory levels...")
    # Calculate average monthly demand per product to set realistic inventory
    product_demand = df.groupby("Product_ID")["Quantity"].agg(["sum", "count"]).reset_index()
    product_demand.columns = ["Product_ID", "total_qty", "num_orders"]
    # Inventory = avg monthly demand × multiplier + random noise
    date_range_days = (df["Order_Date"].max() - df["Order_Date"].min()).days
    months = max(date_range_days / 30, 1)
    product_demand["avg_monthly_demand"] = product_demand["total_qty"] / months
    product_demand["Inventory_Level"] = np.maximum(
        (product_demand["avg_monthly_demand"] * sim_config["inventory_multiplier"]
         + np.random.normal(0, 10, len(product_demand))).astype(int),
        0  # No negative inventory
    )
    # Merge back
    inv_map = dict(zip(product_demand["Product_ID"], product_demand["Inventory_Level"]))
    df["Inventory_Level"] = df["Product_ID"].map(inv_map)
    print(f"  Products with inventory: {len(inv_map):,}")
    print(f"  Avg inventory level: {product_demand['Inventory_Level'].mean():.0f}")
    
    # -------------------------------------------------------------------------
    # Step 8: Reorder columns and save
    # -------------------------------------------------------------------------
    final_columns = [
        "Order_ID", "Customer_ID", "Product_ID", "Product_Name",
        "Product_Category", "Order_Date", "Quantity", "Unit_Price",
        "Discount", "Revenue", "Region", "Inventory_Level",
        "Supplier", "Delivery_Date"
    ]
    df = df[final_columns]
    
    print(f"\n[Step 8] Saving enriched dataset...")
    df.to_csv(enriched_csv, index=False)
    file_size_mb = os.path.getsize(enriched_csv) / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"Enriched dataset created successfully!")
    print(f"  File: {enriched_csv}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nField Classification:")
    print(f"  ORIGINAL:  Order_ID, Customer_ID, Product_ID, Product_Name,")
    print(f"             Order_Date, Quantity, Unit_Price, Region")
    print(f"  DERIVED:   Revenue (Quantity × Unit_Price × (1 - Discount))")
    print(f"  SIMULATED: Product_Category, Discount, Inventory_Level,")
    print(f"             Supplier, Delivery_Date")
    print(f"{'='*60}")
    
    return enriched_csv


if __name__ == "__main__":
    print("=" * 60)
    print("Module 1: Data Enrichment / Simulation")
    print("Adding simulated fields to UCI Online Retail dataset")
    print("=" * 60)
    result = generate_enriched_dataset()
    if result:
        print(f"\nReady for Module 3: python -m src.module3_preprocessing.spark_preprocessing")
