# =============================================================================
# Module 5: Demand Forecasting
# Script: feature_engineering.py
# =============================================================================
# PURPOSE:
#   Prepares features for machine learning demand forecasting models.
#   All feature engineering is done using PySpark DataFrame operations
#   and window functions — NOT pandas.
#
# FEATURES CREATED:
#   Temporal: month, day_of_week, day_of_month, quarter, week_of_year
#   Lag:      lag_1, lag_7, lag_14, lag_30 (demand N days ago)
#   Rolling:  rolling_mean_7, rolling_mean_14, rolling_mean_30
#   Price:    avg_unit_price, avg_discount
#   Encoded:  product_category_index, region_index (via StringIndexer)
#
# BIS701 RELEVANCE:
#   - PySpark window functions for time-series feature creation
#   - Spark MLlib StringIndexer for categorical encoding
#   - Distributed feature engineering on large datasets
#   - ML pipeline preparation
#
# INPUT:  Cleaned Spark DataFrame
# OUTPUT: Feature-engineered DataFrame ready for ML training
# =============================================================================

import os
import sys

from pyspark.sql import Window
from pyspark.sql.functions import (
    col, sum as spark_sum, avg, count, countDistinct,
    lag, round as spark_round, lit, when, coalesce,
    year, month, dayofmonth, dayofweek, quarter, weekofyear
)
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config


def create_daily_demand_dataset(df):
    """
    Aggregate transactional data to daily product-level demand.
    
    Raw data has multiple transactions per product per day. For forecasting,
    we need one row per product per day with total demand.
    
    Args:
        df: Cleaned Spark DataFrame (transactional level).
    
    Returns:
        DataFrame: Daily demand per product.
    """
    print("\n[Feature Engineering] Creating daily demand dataset...")
    
    daily_demand = (
        df.groupBy("Product_ID", "Product_Name", "Product_Category",
                    "Region", "Order_Date_Only")
        .agg(
            spark_sum("Quantity").alias("Daily_Demand"),
            spark_round(spark_sum("Revenue"), 2).alias("Daily_Revenue"),
            spark_round(avg("Unit_Price"), 2).alias("Avg_Unit_Price"),
            spark_round(avg("Discount"), 4).alias("Avg_Discount"),
            count("*").alias("Num_Transactions")
        )
        .withColumnRenamed("Order_Date_Only", "Date")
    )
    
    # Add temporal features
    daily_demand = (
        daily_demand
        .withColumn("Year", year("Date"))
        .withColumn("Month", month("Date"))
        .withColumn("Day_of_Month", dayofmonth("Date"))
        .withColumn("Day_of_Week", dayofweek("Date"))
        .withColumn("Quarter", quarter("Date"))
        .withColumn("Week_of_Year", weekofyear("Date"))
    )
    
    row_count = daily_demand.count()
    products = daily_demand.select("Product_ID").distinct().count()
    print(f"  Daily demand records: {row_count:,}")
    print(f"  Unique products: {products:,}")
    
    return daily_demand


def add_lag_features(df, lag_days):
    """
    Add lag features — demand from N days ago for the same product.
    
    Lag features capture recent demand history and are critical for
    time-series forecasting. They answer: "What was the demand for
    this product N days ago?"
    
    Uses PySpark Window functions for efficient distributed computation.
    
    Args:
        df: Daily demand DataFrame.
        lag_days: List of lag periods (e.g., [1, 7, 14, 30]).
    
    Returns:
        DataFrame: DataFrame with lag feature columns.
    """
    print(f"\n[Feature Engineering] Adding lag features: {lag_days}...")
    
    # Window: partition by product, order by date
    window = Window.partitionBy("Product_ID").orderBy("Date")
    
    for n in lag_days:
        lag_col_name = f"Lag_{n}_Demand"
        df = df.withColumn(lag_col_name, lag("Daily_Demand", n).over(window))
        # Fill nulls with 0 (no data for early dates)
        df = df.withColumn(lag_col_name, coalesce(col(lag_col_name), lit(0)))
    
    print(f"  Added lag columns: {[f'Lag_{n}_Demand' for n in lag_days]}")
    
    return df


def add_rolling_features(df, windows):
    """
    Add rolling average features — mean demand over a sliding window.
    
    Rolling averages smooth out daily fluctuations and capture trends.
    They answer: "What is the average demand over the past N days?"
    
    Uses PySpark Window with rangeBetween for true calendar-day windows.
    
    Args:
        df: Daily demand DataFrame with lag features.
        windows: List of rolling window sizes in days.
    
    Returns:
        DataFrame: DataFrame with rolling average columns.
    """
    print(f"\n[Feature Engineering] Adding rolling averages: {windows}...")
    
    for w in windows:
        col_name = f"Rolling_Mean_{w}"
        # Window: past w rows for same product (approximation of w days)
        window = (
            Window.partitionBy("Product_ID")
            .orderBy("Date")
            .rowsBetween(-w, -1)  # Past w rows, excluding current
        )
        df = df.withColumn(col_name, spark_round(avg("Daily_Demand").over(window), 2))
        df = df.withColumn(col_name, coalesce(col(col_name), lit(0)))
    
    print(f"  Added rolling columns: {[f'Rolling_Mean_{w}' for w in windows]}")
    
    return df


def encode_categorical_features(df):
    """
    Encode categorical string columns as numeric indices using StringIndexer.
    
    Spark MLlib models require numeric inputs. StringIndexer maps each
    unique string to a numeric index (most frequent → 0, etc.).
    
    This is a Spark ML pipeline component — demonstrates MLlib's
    Transformer/Estimator API.
    
    Args:
        df: DataFrame with categorical columns.
    
    Returns:
        tuple: (encoded_DataFrame, fitted_pipeline)
    """
    print("\n[Feature Engineering] Encoding categorical features...")
    
    # Define indexers for categorical columns
    indexers = [
        StringIndexer(
            inputCol="Product_Category",
            outputCol="Category_Index",
            handleInvalid="keep"
        ),
        StringIndexer(
            inputCol="Region",
            outputCol="Region_Index",
            handleInvalid="keep"
        ),
        StringIndexer(
            inputCol="Product_ID",
            outputCol="Product_Index",
            handleInvalid="keep"
        ),
    ]
    
    # Build and fit pipeline
    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    df_encoded = model.transform(df)
    # Convert high-cardinality indices (Product_ID, Region) to continuous numeric
    # to avoid DecisionTree maxBins limitation (>32 categories) in Spark MLlib
    df_encoded = df_encoded.withColumn("Product_Index", col("Product_Index") + 0.0)
    df_encoded = df_encoded.withColumn("Region_Index", col("Region_Index") + 0.0)
    
    print(f"  Encoded: Product_Category → Category_Index (categorical)")
    print(f"  Encoded: Region → Region_Index (continuous)")
    print(f"  Encoded: Product_ID → Product_Index (continuous)")
    
    return df_encoded, model


def assemble_feature_vector(df, feature_columns):
    """
    Assemble individual feature columns into a single 'features' vector.
    
    Spark MLlib models expect a single vector column containing all features.
    VectorAssembler combines multiple columns into one Vector column.
    
    Args:
        df: DataFrame with individual feature columns.
        feature_columns: List of column names to include in the feature vector.
    
    Returns:
        DataFrame: DataFrame with 'features' vector column.
    """
    print(f"\n[Feature Engineering] Assembling feature vector ({len(feature_columns)} features)...")
    
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="skip"  # Skip rows with null features
    )
    
    df_assembled = assembler.transform(df)
    
    print(f"  Feature vector column 'features' created")
    print(f"  Features: {feature_columns}")
    
    return df_assembled


def prepare_ml_dataset(df, config=None):
    """
    Execute the full feature engineering pipeline.
    
    Pipeline:
    1. Aggregate to daily product-level demand
    2. Add lag features (1, 7, 14, 30 days)
    3. Add rolling average features (7, 14, 30 days)
    4. Encode categorical features
    5. Assemble feature vector
    6. Split into train/test sets (chronological)
    
    Args:
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        tuple: (train_df, test_df, feature_columns, encoder_model)
    """
    if config is None:
        config = load_config()
    
    forecast_cfg = config["forecasting"]
    
    print("=" * 60)
    print("Module 5: Feature Engineering for Demand Forecasting")
    print("=" * 60)
    
    # Step 1: Create daily demand dataset
    daily_df = create_daily_demand_dataset(df)
    
    # Step 2: Add lag features
    daily_df = add_lag_features(daily_df, forecast_cfg["lag_features"])
    
    # Step 3: Add rolling averages
    daily_df = add_rolling_features(daily_df, forecast_cfg["rolling_windows"])
    
    # Step 4: Encode categorical features
    daily_df, encoder_model = encode_categorical_features(daily_df)
    
    # Step 5: Define feature columns for ML
    feature_columns = [
        # Temporal features
        "Month", "Day_of_Month", "Day_of_Week", "Quarter", "Week_of_Year",
        # Lag features
        "Lag_1_Demand", "Lag_7_Demand", "Lag_14_Demand", "Lag_30_Demand",
        # Rolling averages
        "Rolling_Mean_7", "Rolling_Mean_14", "Rolling_Mean_30",
        # Price features
        "Avg_Unit_Price", "Avg_Discount",
        # Encoded categorical
        "Category_Index", "Region_Index", "Product_Index",
    ]
    
    # Step 6: Assemble feature vector
    daily_df = assemble_feature_vector(daily_df, feature_columns)
    
    # Step 7: Define label column
    label_col = "Daily_Demand"
    
    # Step 8: Filter out rows with null labels
    ml_df = daily_df.filter(col(label_col).isNotNull())
    
    # Step 9: Chronological train/test split
    # Sort by date, use first train_ratio% as training
    train_ratio = forecast_cfg["train_ratio"]
    
    # Find the split date
    dates = ml_df.select("Date").distinct().orderBy("Date").collect()
    split_idx = int(len(dates) * train_ratio)
    split_date = dates[split_idx]["Date"]
    
    train_df = ml_df.filter(col("Date") < lit(split_date))
    test_df = ml_df.filter(col("Date") >= lit(split_date))
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"\n[Split] Chronological train/test split at {split_date}")
    print(f"  Train: {train_count:,} rows ({train_ratio*100:.0f}%)")
    print(f"  Test:  {test_count:,} rows ({(1-train_ratio)*100:.0f}%)")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Label: {label_col}")
    
    return train_df, test_df, feature_columns, encoder_model


if __name__ == "__main__":
    from src.module3_preprocessing.spark_preprocessing import run_preprocessing
    
    spark, cleaned_df = run_preprocessing()
    train_df, test_df, features, encoder = prepare_ml_dataset(cleaned_df)
    
    print("\nTrain schema:")
    train_df.select("Date", "Product_ID", "Daily_Demand", "features").show(5, truncate=False)
    
    spark.stop()
