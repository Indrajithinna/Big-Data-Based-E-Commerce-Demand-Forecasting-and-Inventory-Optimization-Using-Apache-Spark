# =============================================================================
# Module 3: Data Preprocessing
# Script: spark_preprocessing.py
# =============================================================================
# PURPOSE:
#   Implements the data cleaning and transformation pipeline using PySpark.
#   This is the CORE Big Data processing module — all operations use Spark
#   DataFrames and Spark SQL, NOT pandas.
#
# OPERATIONS DEMONSTRATED:
#   - SparkSession creation and configuration
#   - Schema definition with StructType
#   - CSV data ingestion into Spark DataFrame
#   - Missing value handling (dropna, fillna)
#   - Duplicate removal (dropDuplicates)
#   - Data type casting
#   - Outlier filtering
#   - Date/time feature extraction (year, month, day, day_of_week, quarter)
#   - Derived column creation (withColumn)
#   - Data quality validation
#   - Spark SQL queries (registerTempView + spark.sql)
#   - Writing cleaned data to Parquet format
#
# BIS701 RELEVANCE:
#   - Apache Spark → Distributed data processing engine
#   - PySpark → Python API for Spark
#   - Spark DataFrames → Distributed, column-oriented data structures
#   - Spark SQL → SQL-based analytics on distributed data
#   - Parquet → Columnar storage format optimized for Big Data
#
# INPUT:  data/raw/ecommerce_sales_enriched.csv
# OUTPUT: data/processed/cleaned_sales/ (Parquet format)
# =============================================================================

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, DoubleType, TimestampType, DateType
)
from pyspark.sql.functions import (
    col, when, lit, trim, lower, upper,
    year, month, dayofmonth, dayofweek, quarter, weekofyear,
    date_format, datediff, to_date, to_timestamp,
    count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    stddev, round as spark_round, abs as spark_abs,
    isnan, isnull, coalesce
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config, ensure_directories


def create_spark_session(config):
    """
    Create and configure a SparkSession for data processing.
    
    SparkSession is the entry point to Spark functionality. It combines
    the old SQLContext and HiveContext into a single unified interface.
    
    Args:
        config: Configuration dictionary with Spark settings.
    
    Returns:
        SparkSession: Configured Spark session.
    """
    spark_config = config["spark"]
    
    # Ensure worker python matches current executable (critical on Windows)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    # Ensure Hadoop native library path on Windows
    if sys.platform == "win32":
        hadoop_bin = r"C:\hadoop\bin"
        if os.path.exists(hadoop_bin):
            os.environ["HADOOP_HOME"] = os.path.dirname(hadoop_bin)
            if hadoop_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = hadoop_bin + ";" + os.environ.get("PATH", "")

    builder = (
        SparkSession.builder
        .appName(spark_config["app_name"])
        .master(spark_config["master"])
        .config("spark.driver.memory", spark_config["driver_memory"])
        .config("spark.executor.memory", spark_config["executor_memory"])
        .config("spark.sql.shuffle.partitions", spark_config["shuffle_partitions"])
        # Optimize for local mode
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    )
    
    if sys.platform == "win32" and os.path.exists(r"C:\hadoop\bin"):
        builder = builder.config("spark.driver.extraJavaOptions", "-Djava.library.path=C:/hadoop/bin")
        builder = builder.config("spark.executor.extraJavaOptions", "-Djava.library.path=C:/hadoop/bin")
    
    spark = builder.getOrCreate()
    
    if sys.platform == "win32" and os.path.exists(r"C:\hadoop\bin\hadoop.dll"):
        try:
            spark._jvm.java.lang.System.load(r"C:\hadoop\bin\hadoop.dll")
        except Exception:
            pass
    
    # Set log level to reduce console noise
    spark.sparkContext.setLogLevel(spark_config.get("log_level", "WARN"))
    
    print(f"[Spark] Session created: {spark_config['app_name']}")
    print(f"[Spark] Master: {spark_config['master']}")
    print(f"[Spark] Version: {spark.version}")
    
    return spark


def define_schema():
    """
    Define the explicit schema for the enriched e-commerce dataset.
    
    Defining schema explicitly (rather than relying on schema inference)
    is a Big Data best practice because:
    1. Avoids an extra pass over the data for type inference
    2. Ensures data types are correct from the start
    3. Catches data quality issues early
    4. Improves Spark performance
    
    Returns:
        StructType: Spark schema definition.
    """
    schema = StructType([
        StructField("Order_ID", StringType(), nullable=True),
        StructField("Customer_ID", StringType(), nullable=True),
        StructField("Product_ID", StringType(), nullable=True),
        StructField("Product_Name", StringType(), nullable=True),
        StructField("Product_Category", StringType(), nullable=True),
        StructField("Order_Date", TimestampType(), nullable=True),
        StructField("Quantity", IntegerType(), nullable=True),
        StructField("Unit_Price", DoubleType(), nullable=True),
        StructField("Discount", DoubleType(), nullable=True),
        StructField("Revenue", DoubleType(), nullable=True),
        StructField("Region", StringType(), nullable=True),
        StructField("Inventory_Level", IntegerType(), nullable=True),
        StructField("Supplier", StringType(), nullable=True),
        StructField("Delivery_Date", TimestampType(), nullable=True),
    ])
    
    return schema


def load_data(spark, config):
    """
    Load the enriched CSV dataset into a Spark DataFrame.
    
    This demonstrates data ingestion — reading external data sources
    into the Spark distributed processing engine.
    
    Args:
        spark: SparkSession instance.
        config: Configuration dictionary.
    
    Returns:
        DataFrame: Raw Spark DataFrame.
    """
    input_path = config["paths"]["enriched_dataset_file"]
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Enriched dataset not found at: {input_path}\n"
            "Run Module 1 first: python -m src.module1_ingestion.data_generator"
        )
    
    schema = define_schema()
    
    print(f"\n[Ingestion] Loading data from: {input_path}")
    
    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .schema(schema)
        .load(input_path)
    )
    
    total_rows = df.count()
    print(f"[Ingestion] Loaded {total_rows:,} rows")
    print(f"[Ingestion] Schema:")
    df.printSchema()
    
    return df


def handle_missing_values(df, config):
    """
    Handle missing/null values in the dataset.
    
    Strategy:
    - Drop rows where critical fields are null (Order_ID, Product_ID, Order_Date)
    - Fill numeric nulls with defaults (Quantity=0, Discount=0)
    - Fill text nulls with 'Unknown'
    
    Args:
        df: Input Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    print("\n[Preprocessing] Handling missing values...")
    
    # Count nulls per column before cleaning
    print("  Null counts before cleaning:")
    null_counts = {}
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull() | isnan(col_name) if df.schema[col_name].dataType in [DoubleType(), FloatType()] else col(col_name).isNull()).count()
        null_counts[col_name] = null_count
        if null_count > 0:
            print(f"    {col_name}: {null_count:,}")
    
    before_count = df.count()
    
    # Drop rows with null critical fields
    critical_fields = ["Order_ID", "Product_ID", "Order_Date", "Quantity", "Unit_Price"]
    df = df.dropna(subset=critical_fields)
    
    after_critical = df.count()
    print(f"  Dropped {before_count - after_critical:,} rows with null critical fields")
    
    # Fill remaining nulls
    df = (
        df
        .fillna({"Customer_ID": "Unknown"})
        .fillna({"Product_Name": "Unknown Product"})
        .fillna({"Product_Category": "Uncategorized"})
        .fillna({"Region": "Unknown"})
        .fillna({"Discount": 0.0})
        .fillna({"Revenue": 0.0})
        .fillna({"Inventory_Level": 0})
        .fillna({"Supplier": "Unknown_Supplier"})
    )
    
    print(f"  Remaining rows: {df.count():,}")
    
    return df


def remove_duplicates(df):
    """
    Remove duplicate records from the dataset.
    
    Duplicates are identified by the combination of:
    Order_ID + Product_ID (same product in same order).
    
    Args:
        df: Input Spark DataFrame.
    
    Returns:
        DataFrame: Deduplicated DataFrame.
    """
    print("\n[Preprocessing] Removing duplicates...")
    
    before_count = df.count()
    df = df.dropDuplicates(["Order_ID", "Product_ID"])
    after_count = df.count()
    
    removed = before_count - after_count
    print(f"  Removed {removed:,} duplicate rows ({removed/max(before_count,1)*100:.1f}%)")
    print(f"  Remaining rows: {after_count:,}")
    
    return df


def filter_outliers(df, config):
    """
    Remove outliers based on business rules from config.
    
    Outlier handling rules:
    1. Remove negative quantities (returns/cancellations) if configured
    2. Remove zero/negative prices
    3. Cap extreme quantities and prices
    4. Filter date range
    
    Args:
        df: Input Spark DataFrame.
        config: Configuration dictionary with preprocessing settings.
    
    Returns:
        DataFrame: Filtered DataFrame.
    """
    print("\n[Preprocessing] Filtering outliers...")
    
    preprocess_cfg = config["preprocessing"]
    before_count = df.count()
    
    # 1. Remove negative/zero quantities (these are returns or cancellations)
    if preprocess_cfg["remove_negative_quantity"]:
        df = df.filter(col("Quantity") > 0)
        after = df.count()
        print(f"  After removing Quantity <= 0: {after:,} rows (removed {before_count - after:,})")
        before_count = after
    
    # 2. Remove zero/negative prices
    if preprocess_cfg["remove_zero_price"]:
        df = df.filter(col("Unit_Price") > 0)
        after = df.count()
        print(f"  After removing Unit_Price <= 0: {after:,} rows (removed {before_count - after:,})")
        before_count = after
    
    # 3. Cap extreme quantities
    max_qty = preprocess_cfg["max_quantity_per_txn"]
    df = df.filter(col("Quantity") <= max_qty)
    after = df.count()
    print(f"  After capping Quantity <= {max_qty}: {after:,} rows (removed {before_count - after:,})")
    before_count = after
    
    # 4. Cap extreme prices
    max_price = preprocess_cfg["max_unit_price"]
    df = df.filter(col("Unit_Price") <= max_price)
    after = df.count()
    print(f"  After capping Unit_Price <= {max_price}: {after:,} rows (removed {before_count - after:,})")
    before_count = after
    
    # 5. Filter date range
    min_date = preprocess_cfg["min_date"]
    max_date = preprocess_cfg["max_date"]
    df = df.filter(
        (col("Order_Date") >= lit(min_date)) & 
        (col("Order_Date") <= lit(max_date))
    )
    after = df.count()
    print(f"  After date filter [{min_date} to {max_date}]: {after:,} rows")
    
    return df


def extract_date_features(df):
    """
    Extract temporal features from Order_Date for analytics and forecasting.
    
    These features are essential for:
    - Time-series aggregation (daily/weekly/monthly sales)
    - Seasonal pattern detection
    - ML model features (day_of_week, month, quarter)
    
    Args:
        df: Input Spark DataFrame with Order_Date column.
    
    Returns:
        DataFrame: DataFrame with additional date feature columns.
    """
    print("\n[Preprocessing] Extracting date/time features...")
    
    df = (
        df
        # Convert to date type for aggregation
        .withColumn("Order_Date_Only", to_date(col("Order_Date")))
        # Extract components
        .withColumn("Year", year(col("Order_Date")))
        .withColumn("Month", month(col("Order_Date")))
        .withColumn("Day", dayofmonth(col("Order_Date")))
        .withColumn("Day_of_Week", dayofweek(col("Order_Date")))
        .withColumn("Quarter", quarter(col("Order_Date")))
        .withColumn("Week_of_Year", weekofyear(col("Order_Date")))
        # Year-Month string for grouping
        .withColumn("Year_Month", date_format(col("Order_Date"), "yyyy-MM"))
        # Calculate delivery lead time (if Delivery_Date exists)
        .withColumn(
            "Lead_Time_Days",
            when(col("Delivery_Date").isNotNull(),
                 datediff(col("Delivery_Date"), col("Order_Date")))
            .otherwise(lit(7))  # Default lead time
        )
    )
    
    # Recalculate Revenue to ensure consistency
    df = df.withColumn(
        "Revenue",
        spark_round(col("Quantity") * col("Unit_Price") * (1 - col("Discount")), 2)
    )
    
    print(f"  Added features: Year, Month, Day, Day_of_Week, Quarter, Week_of_Year,")
    print(f"                  Year_Month, Order_Date_Only, Lead_Time_Days")
    
    return df


def validate_data_quality(df, spark):
    """
    Run data quality checks using Spark SQL to validate the cleaned dataset.
    
    This demonstrates Spark SQL — registering a DataFrame as a temp view
    and running SQL queries on it, similar to Hive queries in the
    Hadoop ecosystem.
    
    Args:
        df: Cleaned Spark DataFrame.
        spark: SparkSession instance.
    """
    print("\n[Quality Check] Validating cleaned dataset with Spark SQL...")
    
    # Register DataFrame as a temporary SQL view
    df.createOrReplaceTempView("sales")
    
    # --- Quality Check 1: Row count ---
    result = spark.sql("SELECT COUNT(*) as total_rows FROM sales")
    total_rows = result.collect()[0]["total_rows"]
    print(f"\n  Total rows: {total_rows:,}")
    
    # --- Quality Check 2: Date range ---
    result = spark.sql("""
        SELECT 
            MIN(Order_Date) as min_date,
            MAX(Order_Date) as max_date,
            COUNT(DISTINCT Order_Date_Only) as distinct_days
        FROM sales
    """)
    row = result.collect()[0]
    print(f"  Date range: {row['min_date']} to {row['max_date']}")
    print(f"  Distinct days: {row['distinct_days']:,}")
    
    # --- Quality Check 3: Unique counts ---
    result = spark.sql("""
        SELECT
            COUNT(DISTINCT Order_ID) as unique_orders,
            COUNT(DISTINCT Product_ID) as unique_products,
            COUNT(DISTINCT Customer_ID) as unique_customers,
            COUNT(DISTINCT Region) as unique_regions,
            COUNT(DISTINCT Product_Category) as unique_categories,
            COUNT(DISTINCT Supplier) as unique_suppliers
        FROM sales
    """)
    row = result.collect()[0]
    print(f"\n  Unique Orders: {row['unique_orders']:,}")
    print(f"  Unique Products: {row['unique_products']:,}")
    print(f"  Unique Customers: {row['unique_customers']:,}")
    print(f"  Unique Regions: {row['unique_regions']:,}")
    print(f"  Unique Categories: {row['unique_categories']:,}")
    print(f"  Unique Suppliers: {row['unique_suppliers']:,}")
    
    # --- Quality Check 4: Numerical ranges ---
    result = spark.sql("""
        SELECT
            MIN(Quantity) as min_qty, MAX(Quantity) as max_qty, AVG(Quantity) as avg_qty,
            MIN(Unit_Price) as min_price, MAX(Unit_Price) as max_price, AVG(Unit_Price) as avg_price,
            MIN(Revenue) as min_rev, MAX(Revenue) as max_rev, SUM(Revenue) as total_rev,
            MIN(Discount) as min_disc, MAX(Discount) as max_disc, AVG(Discount) as avg_disc
        FROM sales
    """)
    row = result.collect()[0]
    print(f"\n  Quantity:   min={row['min_qty']}, max={row['max_qty']}, avg={row['avg_qty']:.1f}")
    print(f"  Unit Price: min={row['min_price']:.2f}, max={row['max_price']:.2f}, avg={row['avg_price']:.2f}")
    print(f"  Revenue:    min={row['min_rev']:.2f}, max={row['max_rev']:.2f}, total={row['total_rev']:,.2f}")
    print(f"  Discount:   min={row['min_disc']:.2f}, max={row['max_disc']:.2f}, avg={row['avg_disc']:.4f}")
    
    # --- Quality Check 5: Null check on cleaned data ---
    result = spark.sql("""
        SELECT
            SUM(CASE WHEN Order_ID IS NULL THEN 1 ELSE 0 END) as null_order_id,
            SUM(CASE WHEN Product_ID IS NULL THEN 1 ELSE 0 END) as null_product_id,
            SUM(CASE WHEN Quantity IS NULL THEN 1 ELSE 0 END) as null_quantity,
            SUM(CASE WHEN Unit_Price IS NULL THEN 1 ELSE 0 END) as null_price
        FROM sales
    """)
    row = result.collect()[0]
    print(f"\n  Null Order_ID: {row['null_order_id']}")
    print(f"  Null Product_ID: {row['null_product_id']}")
    print(f"  Null Quantity: {row['null_quantity']}")
    print(f"  Null Unit_Price: {row['null_price']}")
    
    # --- Quality Check 6: Top regions by revenue (Spark SQL demo) ---
    print("\n  Top 5 Regions by Revenue (Spark SQL):")
    spark.sql("""
        SELECT 
            Region,
            COUNT(*) as num_transactions,
            ROUND(SUM(Revenue), 2) as total_revenue,
            ROUND(AVG(Revenue), 2) as avg_revenue
        FROM sales
        GROUP BY Region
        ORDER BY total_revenue DESC
        LIMIT 5
    """).show(truncate=False)


def save_cleaned_data(df, config, spark):
    """
    Save the cleaned DataFrame in Parquet format.
    
    Parquet is chosen because:
    1. Columnar format — efficient for analytical queries
    2. Built-in compression — reduces storage
    3. Schema preservation — types are embedded in the file
    4. Spark-native — optimal read/write performance
    5. Standard Big Data format — used in Hadoop/Spark ecosystems
    
    Also exports a CSV version for the dashboard.
    
    Args:
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
        spark: SparkSession instance.
    """
    output_parquet = config["paths"]["cleaned_dataset_file"]
    output_csv_dir = os.path.join(config["paths"]["processed_data_dir"], "cleaned_sales_csv")
    
    print(f"\n[Save] Writing cleaned data...")
    
    # Save as Parquet (Big Data format)
    df.write.mode("overwrite").parquet(output_parquet)
    print(f"  Parquet: {output_parquet}")
    
    # Also save as single CSV for dashboard/visualization
    # coalesce(1) merges partitions into a single file
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_csv_dir)
    print(f"  CSV: {output_csv_dir}")
    
    # Rename the part file to a clean name
    csv_dir = output_csv_dir
    if os.path.exists(csv_dir):
        for f in os.listdir(csv_dir):
            if f.startswith("part-") and f.endswith(".csv"):
                src = os.path.join(csv_dir, f)
                dst = os.path.join(config["paths"]["processed_data_dir"], "cleaned_sales.csv")
                import shutil
                shutil.copy2(src, dst)
                print(f"  Single CSV: {dst}")
                break
    
    row_count = df.count()
    print(f"  Total cleaned rows saved: {row_count:,}")


def run_preprocessing(config=None):
    """
    Execute the full PySpark preprocessing pipeline.
    
    Pipeline steps:
    1. Create SparkSession
    2. Load raw CSV data with explicit schema
    3. Handle missing values
    4. Remove duplicates
    5. Filter outliers
    6. Extract date/time features
    7. Validate data quality with Spark SQL
    8. Save cleaned data as Parquet
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        tuple: (SparkSession, cleaned_DataFrame)
    """
    if config is None:
        config = load_config()
    
    ensure_directories(config)
    
    print("=" * 60)
    print("Module 3: Data Preprocessing Pipeline (PySpark)")
    print("=" * 60)
    
    # Step 1: Create Spark session
    spark = create_spark_session(config)
    
    # Step 2: Load data
    df = load_data(spark, config)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df, config)
    
    # Step 4: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 5: Filter outliers
    df = filter_outliers(df, config)
    
    # Step 6: Extract date features
    df = extract_date_features(df)
    
    # Step 7: Cache the DataFrame for performance
    # Caching keeps the DataFrame in memory across multiple operations
    df = df.cache()
    
    # Step 8: Validate quality
    validate_data_quality(df, spark)
    
    # Step 9: Save cleaned data
    save_cleaned_data(df, config, spark)
    
    print(f"\n{'='*60}")
    print("Preprocessing pipeline complete!")
    print(f"{'='*60}")
    
    return spark, df


if __name__ == "__main__":
    spark, cleaned_df = run_preprocessing()
    
    # Show sample data
    print("\nSample cleaned data (5 rows):")
    cleaned_df.show(5, truncate=False)
    
    print("\nReady for Module 4: python -m src.module4_analytics.sales_analytics")
    
    spark.stop()
