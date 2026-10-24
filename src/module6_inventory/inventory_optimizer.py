# =============================================================================
# Module 6: Inventory Optimization
# Script: inventory_optimizer.py
# =============================================================================
# PURPOSE:
#   Computes inventory optimization recommendations based on predicted
#   demand from Module 5. Implements classic inventory management formulas
#   adapted for the Big Data context using PySpark.
#
# CALCULATIONS:
#   - Safety Stock = Z × σ(daily_demand) × √(Lead_Time)
#   - Reorder Point = Avg_Daily_Demand × Lead_Time + Safety_Stock
#   - Reorder Quantity = Predicted_Demand_Next_Period − Current_Inventory + Safety_Stock
#   - Stockout Risk: Current_Inventory < Reorder_Point
#   - Overstock Risk: Current_Inventory > 2 × Predicted_Next_Period_Demand
#
# ASSUMPTIONS (clearly stated — NOT real business data):
#   - Lead Time: Configurable, default 7 days
#   - Service Level: Configurable, default 95% (Z = 1.65)
#   - Review Period: Configurable, default 30 days
#   - Inventory levels are SIMULATED (see Module 1)
#
# BIS701 RELEVANCE:
#   - Demonstrates business logic implementation on PySpark DataFrames
#   - Shows how Big Data analytics translates to actionable business decisions
#   - Integrates ML predictions with operations research concepts
#
# INPUT:  Demand predictions, cleaned sales data
# OUTPUT: Inventory recommendations CSV in data/results/inventory/
# =============================================================================

import os
import sys
import math

from pyspark.sql.functions import (
    col, lit, sqrt as spark_sqrt, when, round as spark_round,
    sum as spark_sum, avg, stddev, count, countDistinct,
    first, max as spark_max, min as spark_min
)
from pyspark.sql import Window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config


def calculate_demand_statistics(df, config=None):
    """
    Calculate demand statistics per product needed for inventory formulas.
    
    Statistics computed:
    - Average daily demand
    - Standard deviation of daily demand
    - Total historical demand
    - Number of active selling days
    
    All done in PySpark (not pandas).
    
    Args:
        df: Cleaned sales DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame: Product-level demand statistics.
    """
    if config is None:
        config = load_config()
    
    print("\n[Inventory] Calculating demand statistics per product...")
    
    # First aggregate to daily demand per product
    daily_demand = (
        df.groupBy("Product_ID", "Product_Name", "Product_Category", "Order_Date_Only")
        .agg(
            spark_sum("Quantity").alias("Daily_Demand")
        )
    )
    
    # Then compute statistics across days for each product
    demand_stats = (
        daily_demand.groupBy("Product_ID", "Product_Name", "Product_Category")
        .agg(
            spark_round(avg("Daily_Demand"), 4).alias("Avg_Daily_Demand"),
            spark_round(stddev("Daily_Demand"), 4).alias("Std_Daily_Demand"),
            spark_sum("Daily_Demand").alias("Total_Demand"),
            count("*").alias("Active_Days"),
            spark_max("Daily_Demand").alias("Max_Daily_Demand"),
            spark_min("Daily_Demand").alias("Min_Daily_Demand"),
        )
        # Fill null std dev (products with only 1 day of sales)
        .fillna({"Std_Daily_Demand": 0.0})
    )
    
    product_count = demand_stats.count()
    print(f"  Computed statistics for {product_count:,} products")
    
    return demand_stats


def get_current_inventory(df):
    """
    Get current inventory level for each product.
    
    NOTE: Inventory levels are SIMULATED (see Module 1 data_generator.py).
    In a real system, this would query the warehouse management system.
    
    Args:
        df: Cleaned sales DataFrame with Inventory_Level column.
    
    Returns:
        DataFrame: Product-level inventory.
    """
    print("\n[Inventory] Getting current inventory levels (SIMULATED)...")
    
    # Get the most recent inventory level per product
    inventory = (
        df.groupBy("Product_ID")
        .agg(
            first("Inventory_Level").alias("Current_Inventory"),
            first("Supplier").alias("Supplier")
        )
    )
    
    return inventory


def get_predicted_demand(config):
    """
    Load predicted demand from Module 5 forecasting output.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        dict: Product_ID → Predicted total demand for next period.
               Returns None if prediction file not found.
    """
    import pandas as pd
    
    pred_path = os.path.join(config["paths"]["forecast_output_dir"], "demand_predictions.csv")
    
    if not os.path.exists(pred_path):
        print(f"  Warning: Predictions file not found at {pred_path}")
        print("  Using historical average demand as fallback.")
        return None
    
    pred_df = pd.read_csv(pred_path)
    
    # Aggregate predicted demand per product
    product_predictions = (
        pred_df.groupby("Product_ID")["Predicted_Demand"]
        .sum()
        .to_dict()
    )
    
    print(f"  Loaded predictions for {len(product_predictions):,} products")
    return product_predictions


def compute_inventory_optimization(spark, df, config=None):
    """
    Main inventory optimization function.
    
    Implements the following formulas:
    
    1. SAFETY STOCK:
       Safety_Stock = Z_score × Std_Daily_Demand × √(Lead_Time)
       
       Where:
       - Z_score: Based on desired service level (95% → 1.65)
       - Std_Daily_Demand: Standard deviation of daily demand
       - Lead_Time: Time from order placement to delivery (days)
       
       Purpose: Buffer inventory to handle demand uncertainty during lead time.
    
    2. REORDER POINT:
       Reorder_Point = Avg_Daily_Demand × Lead_Time + Safety_Stock
       
       Purpose: The inventory level at which a new order should be placed.
       When current inventory drops below this level, reorder.
    
    3. REORDER QUANTITY:
       Reorder_Qty = Predicted_Demand_Next_Period − Current_Inventory + Safety_Stock
       (if > 0, else 0)
       
       Purpose: How many units to order to meet predicted demand.
    
    4. STOCKOUT RISK:
       Risk = TRUE if Current_Inventory < Reorder_Point
       
       Purpose: Identifies products at immediate risk of running out.
    
    5. OVERSTOCK RISK:
       Risk = TRUE if Current_Inventory > Overstock_Multiplier × Predicted_Next_Period_Demand
       
       Purpose: Identifies products with excess inventory (capital tied up).
    
    Args:
        spark: SparkSession instance.
        df: Cleaned sales DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame: Inventory optimization recommendations.
    """
    if config is None:
        config = load_config()
    
    inv_config = config["inventory"]
    lead_time = inv_config["lead_time_days"]
    z_score = inv_config["z_score"]
    service_level = inv_config["service_level"]
    review_period = inv_config["review_period_days"]
    overstock_mult = inv_config["overstock_multiplier"]
    
    print("=" * 60)
    print("Module 6: Inventory Optimization")
    print("=" * 60)
    print(f"\n  Parameters (from config — NOT real business data):")
    print(f"    Lead Time:         {lead_time} days")
    print(f"    Service Level:     {service_level*100:.0f}%")
    print(f"    Z-score:           {z_score}")
    print(f"    Review Period:     {review_period} days")
    print(f"    Overstock Mult:    {overstock_mult}×")
    
    # Step 1: Calculate demand statistics
    demand_stats = calculate_demand_statistics(df, config)
    
    # Step 2: Get current inventory
    inventory = get_current_inventory(df)
    
    # Step 3: Load predicted demand (from forecasting module)
    predicted_demand_dict = get_predicted_demand(config)
    
    # Step 4: Join demand stats with inventory
    optimization = (
        demand_stats
        .join(inventory, on="Product_ID", how="left")
        .fillna({"Current_Inventory": 0, "Supplier": "Unknown"})
    )
    
    # Step 5: Calculate Safety Stock
    # Safety_Stock = Z × σ × √(Lead_Time)
    optimization = optimization.withColumn(
        "Safety_Stock",
        spark_round(
            lit(z_score) * col("Std_Daily_Demand") * lit(math.sqrt(lead_time)),
            0
        ).cast("int")
    )
    
    # Step 6: Calculate Reorder Point
    # Reorder_Point = Avg_Daily_Demand × Lead_Time + Safety_Stock
    optimization = optimization.withColumn(
        "Reorder_Point",
        spark_round(
            col("Avg_Daily_Demand") * lit(lead_time) + col("Safety_Stock"),
            0
        ).cast("int")
    )
    
    # Step 7: Calculate Predicted Next Period Demand
    # Use ML predictions if available, otherwise use historical average × review period
    optimization = optimization.withColumn(
        "Predicted_Next_Period_Demand",
        spark_round(col("Avg_Daily_Demand") * lit(review_period), 0).cast("int")
    )
    
    # If we have ML predictions, update with those values
    # If we have ML predictions, update with those values
    if predicted_demand_dict:
        # Create a Spark DataFrame for joining (100% Spark-native, no Python worker UDF needed)
        pred_rows = [(str(k), int(round(float(v)))) for k, v in predicted_demand_dict.items()]
        pred_spark_df = spark.createDataFrame(pred_rows, ["Product_ID", "ML_Predicted_Demand"])
        
        optimization = optimization.join(pred_spark_df, on="Product_ID", how="left")
        optimization = optimization.withColumn(
            "Predicted_Next_Period_Demand",
            when(col("ML_Predicted_Demand").isNotNull(), col("ML_Predicted_Demand"))
            .otherwise(col("Predicted_Next_Period_Demand"))
        ).drop("ML_Predicted_Demand")
    
    # Step 8: Calculate Reorder Quantity
    # Reorder_Qty = max(0, Predicted_Demand - Current_Inventory + Safety_Stock)
    optimization = optimization.withColumn(
        "Reorder_Quantity",
        when(
            col("Predicted_Next_Period_Demand") - col("Current_Inventory") + col("Safety_Stock") > 0,
            spark_round(
                col("Predicted_Next_Period_Demand") - col("Current_Inventory") + col("Safety_Stock"),
                0
            ).cast("int")
        ).otherwise(lit(0))
    )
    
    # Step 9: Stockout Risk
    optimization = optimization.withColumn(
        "Stockout_Risk",
        when(col("Current_Inventory") < col("Reorder_Point"), lit("HIGH"))
        .when(col("Current_Inventory") < col("Reorder_Point") * 1.5, lit("MEDIUM"))
        .otherwise(lit("LOW"))
    )
    
    # Step 10: Overstock Risk
    optimization = optimization.withColumn(
        "Overstock_Risk",
        when(
            col("Current_Inventory") > col("Predicted_Next_Period_Demand") * lit(overstock_mult),
            lit("HIGH")
        )
        .when(
            col("Current_Inventory") > col("Predicted_Next_Period_Demand") * lit(overstock_mult * 0.75),
            lit("MEDIUM")
        )
        .otherwise(lit("LOW"))
    )
    
    # Step 11: Overall recommendation
    optimization = optimization.withColumn(
        "Recommendation",
        when(col("Stockout_Risk") == "HIGH", lit("URGENT REORDER"))
        .when(col("Stockout_Risk") == "MEDIUM", lit("REORDER SOON"))
        .when(col("Overstock_Risk") == "HIGH", lit("REDUCE STOCK / PROMOTE"))
        .when(col("Overstock_Risk") == "MEDIUM", lit("MONITOR - SLOW MOVING"))
        .otherwise(lit("OPTIMAL"))
    )
    
    # Select final columns
    result = optimization.select(
        "Product_ID", "Product_Name", "Product_Category", "Supplier",
        "Avg_Daily_Demand", "Std_Daily_Demand", "Total_Demand", "Active_Days",
        "Current_Inventory", "Safety_Stock", "Reorder_Point",
        "Predicted_Next_Period_Demand", "Reorder_Quantity",
        "Stockout_Risk", "Overstock_Risk", "Recommendation"
    ).orderBy("Stockout_Risk", "Reorder_Quantity", ascending=[True, False])
    
    return result


def save_inventory_results(result_df, config):
    """
    Save inventory optimization results to CSV and print summary.
    
    Args:
        result_df: Inventory optimization DataFrame.
        config: Configuration dictionary.
    """
    output_dir = config["paths"]["inventory_output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    path = os.path.join(output_dir, "inventory_recommendations.csv")
    result_pandas = result_df.toPandas()
    result_pandas.to_csv(path, index=False)
    print(f"\n[Save] Inventory recommendations saved to: {path}")
    
    # Print summary statistics
    total = len(result_pandas)
    stockout_high = len(result_pandas[result_pandas["Stockout_Risk"] == "HIGH"])
    stockout_med = len(result_pandas[result_pandas["Stockout_Risk"] == "MEDIUM"])
    overstock_high = len(result_pandas[result_pandas["Overstock_Risk"] == "HIGH"])
    overstock_med = len(result_pandas[result_pandas["Overstock_Risk"] == "MEDIUM"])
    urgent = len(result_pandas[result_pandas["Recommendation"] == "URGENT REORDER"])
    optimal = len(result_pandas[result_pandas["Recommendation"] == "OPTIMAL"])
    
    print(f"\n{'='*50}")
    print("  INVENTORY OPTIMIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total Products Analyzed:     {total:>8,}")
    print(f"  Stockout Risk - HIGH:        {stockout_high:>8,}")
    print(f"  Stockout Risk - MEDIUM:      {stockout_med:>8,}")
    print(f"  Overstock Risk - HIGH:       {overstock_high:>8,}")
    print(f"  Overstock Risk - MEDIUM:     {overstock_med:>8,}")
    print(f"  URGENT REORDER:              {urgent:>8,}")
    print(f"  OPTIMAL:                     {optimal:>8,}")
    print(f"{'='*50}")
    
    # Save summary as JSON
    summary = {
        "total_products": total,
        "stockout_risk_high": stockout_high,
        "stockout_risk_medium": stockout_med,
        "overstock_risk_high": overstock_high,
        "overstock_risk_medium": overstock_med,
        "urgent_reorder": urgent,
        "optimal": optimal,
        "parameters": {
            "lead_time_days": config["inventory"]["lead_time_days"],
            "service_level": config["inventory"]["service_level"],
            "z_score": config["inventory"]["z_score"],
            "review_period_days": config["inventory"]["review_period_days"],
        },
        "assumptions": [
            "Inventory levels are SIMULATED (not from a real warehouse system)",
            "Lead time is a configurable constant (real lead times vary by supplier)",
            "Demand is assumed to follow a normal distribution for safety stock calculation",
            "Service level Z-score assumes normally distributed demand during lead time",
        ]
    }
    
    import json
    summary_path = os.path.join(output_dir, "inventory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return result_pandas


def run_inventory_optimization(spark, df, config=None):
    """
    Execute the full inventory optimization pipeline.
    
    Args:
        spark: SparkSession instance.
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame: Inventory recommendations (Pandas).
    """
    if config is None:
        config = load_config()
    
    # Compute optimization
    result_df = compute_inventory_optimization(spark, df, config)
    
    # Save results
    result_pandas = save_inventory_results(result_df, config)
    
    return result_pandas


if __name__ == "__main__":
    from src.module3_preprocessing.spark_preprocessing import run_preprocessing
    
    spark, cleaned_df = run_preprocessing()
    results = run_inventory_optimization(spark, cleaned_df)
    
    print("\nSample recommendations:")
    print(results[["Product_ID", "Current_Inventory", "Reorder_Point",
                    "Reorder_Quantity", "Stockout_Risk", "Recommendation"]].head(10))
    
    print("\nReady for Module 7: streamlit run src/module7_dashboard/app.py")
    spark.stop()
