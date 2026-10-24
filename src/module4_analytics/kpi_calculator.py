# =============================================================================
# Module 4: Exploratory Big Data Analytics
# Script: kpi_calculator.py
# =============================================================================
# PURPOSE:
#   Calculates key performance indicators (KPIs) from the cleaned sales data
#   using PySpark. These KPIs provide an executive summary of the business.
#
# KPIs CALCULATED:
#   - Total Revenue: Sum of all revenue (Revenue = Qty × Price × (1 - Discount))
#   - Total Orders: Count of distinct Order_IDs
#   - Total Units Sold: Sum of all Quantity values
#   - Average Order Value (AOV): Total Revenue / Total Orders
#   - Average Daily Demand: Total Units Sold / Number of Active Days
#   - Product Growth Rate: (Current Month Revenue / Previous Month Revenue) - 1
#   - Inventory Turnover: Total Units Sold / Average Inventory Level
#   - Stockout Rate: Products at risk of stockout / Total Products
#
# BIS701 RELEVANCE:
#   - Demonstrates aggregate computations on distributed data
#   - Shows how Big Data analytics produces business intelligence KPIs
#   - Spark SQL for complex metric calculations
# =============================================================================

import os
import sys
import json

from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    stddev, round as spark_round, countDistinct, datediff, lit, when, lag
)
from pyspark.sql import Window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config


def calculate_kpis(spark, df, config=None):
    """
    Calculate all business KPIs from the cleaned sales data.
    
    FORMULAS:
        Total Revenue = SUM(Revenue)
        Total Orders = COUNT(DISTINCT Order_ID)
        Total Units Sold = SUM(Quantity)
        Average Order Value = Total Revenue / Total Orders
        Average Daily Demand = Total Units Sold / Number of Days
        Inventory Turnover = Total Units Sold / Average Inventory Level
    
    Args:
        spark: SparkSession instance.
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        dict: Dictionary of KPI name → value.
    """
    if config is None:
        config = load_config()
    
    print("\n[KPIs] Calculating Key Performance Indicators...")
    
    df.createOrReplaceTempView("sales")
    
    # --- Core KPIs ---
    core_kpis = spark.sql("""
        SELECT
            ROUND(SUM(Revenue), 2) as Total_Revenue,
            COUNT(DISTINCT Order_ID) as Total_Orders,
            SUM(Quantity) as Total_Units_Sold,
            COUNT(DISTINCT Product_ID) as Total_Products,
            COUNT(DISTINCT Customer_ID) as Total_Customers,
            COUNT(DISTINCT Region) as Total_Regions,
            ROUND(SUM(Revenue) / COUNT(DISTINCT Order_ID), 2) as Avg_Order_Value,
            MIN(Order_Date_Only) as First_Date,
            MAX(Order_Date_Only) as Last_Date,
            DATEDIFF(MAX(Order_Date_Only), MIN(Order_Date_Only)) as Date_Range_Days
        FROM sales
    """).collect()[0]
    
    total_revenue = float(core_kpis["Total_Revenue"])
    total_orders = int(core_kpis["Total_Orders"])
    total_units = int(core_kpis["Total_Units_Sold"])
    total_products = int(core_kpis["Total_Products"])
    total_customers = int(core_kpis["Total_Customers"])
    total_regions = int(core_kpis["Total_Regions"])
    aov = float(core_kpis["Avg_Order_Value"])
    date_range_days = int(core_kpis["Date_Range_Days"]) or 1
    
    # Average Daily Demand = Total Units / Days
    avg_daily_demand = round(total_units / date_range_days, 2)
    
    # --- Inventory Turnover ---
    # Formula: Inventory Turnover = Total Units Sold / Average Inventory Level
    # A higher value means products are selling faster relative to stock
    avg_inventory = spark.sql("""
        SELECT ROUND(AVG(Inventory_Level), 2) as avg_inv
        FROM (
            SELECT Product_ID, FIRST(Inventory_Level) as Inventory_Level
            FROM sales
            GROUP BY Product_ID
        )
    """).collect()[0]["avg_inv"]
    
    avg_inventory = float(avg_inventory) if avg_inventory else 1.0
    inventory_turnover = round(total_units / max(avg_inventory, 1), 2)
    
    # --- Product Growth Rate ---
    # Compare last complete month's revenue to previous month
    monthly_revenue = spark.sql("""
        SELECT 
            Year_Month,
            ROUND(SUM(Revenue), 2) as Monthly_Revenue
        FROM sales
        GROUP BY Year_Month
        ORDER BY Year_Month DESC
        LIMIT 3
    """).collect()
    
    if len(monthly_revenue) >= 2:
        current_month_rev = float(monthly_revenue[0]["Monthly_Revenue"])
        previous_month_rev = float(monthly_revenue[1]["Monthly_Revenue"])
        growth_rate = round(
            ((current_month_rev - previous_month_rev) / max(previous_month_rev, 1)) * 100, 2
        )
    else:
        growth_rate = 0.0
    
    # --- Stockout Risk ---
    # Products where current inventory < average daily demand × lead time
    lead_time = config["inventory"]["lead_time_days"]
    
    stockout_info = spark.sql(f"""
        SELECT
            COUNT(*) as total_products,
            SUM(CASE WHEN Inventory_Level < avg_daily_qty * {lead_time} THEN 1 ELSE 0 END) as at_risk
        FROM (
            SELECT 
                Product_ID,
                FIRST(Inventory_Level) as Inventory_Level,
                SUM(Quantity) / {date_range_days} as avg_daily_qty
            FROM sales
            GROUP BY Product_ID
        )
    """).collect()[0]
    
    products_at_risk = int(stockout_info["at_risk"])
    stockout_rate = round((products_at_risk / max(total_products, 1)) * 100, 2)
    
    # --- Compile KPIs ---
    kpis = {
        "Total_Revenue": total_revenue,
        "Total_Orders": total_orders,
        "Total_Units_Sold": total_units,
        "Total_Products": total_products,
        "Total_Customers": total_customers,
        "Total_Regions": total_regions,
        "Average_Order_Value": aov,
        "Average_Daily_Demand": avg_daily_demand,
        "Revenue_Growth_Rate_Pct": growth_rate,
        "Inventory_Turnover": inventory_turnover,
        "Average_Inventory_Level": avg_inventory,
        "Products_At_Stockout_Risk": products_at_risk,
        "Stockout_Rate_Pct": stockout_rate,
        "Date_Range_Days": date_range_days,
        "First_Date": str(core_kpis["First_Date"]),
        "Last_Date": str(core_kpis["Last_Date"]),
    }
    
    # --- Print KPIs ---
    print("\n" + "=" * 50)
    print("  KEY PERFORMANCE INDICATORS")
    print("=" * 50)
    print(f"  Total Revenue:           ${total_revenue:>14,.2f}")
    print(f"  Total Orders:            {total_orders:>14,}")
    print(f"  Total Units Sold:        {total_units:>14,}")
    print(f"  Total Products:          {total_products:>14,}")
    print(f"  Total Customers:         {total_customers:>14,}")
    print(f"  Avg Order Value:         ${aov:>14,.2f}")
    print(f"  Avg Daily Demand:        {avg_daily_demand:>14,.2f} units")
    print(f"  Revenue Growth:          {growth_rate:>14,.2f}%")
    print(f"  Inventory Turnover:      {inventory_turnover:>14,.2f}")
    print(f"  Products at Risk:        {products_at_risk:>14,}")
    print(f"  Stockout Rate:           {stockout_rate:>14,.2f}%")
    print(f"  Date Range:              {date_range_days:>14,} days")
    print("=" * 50)
    
    # Save KPIs as JSON
    output_dir = config["paths"]["analytics_output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    kpi_path = os.path.join(output_dir, "kpis.json")
    with open(kpi_path, "w") as f:
        json.dump(kpis, f, indent=2, default=str)
    print(f"\n  KPIs saved to: {kpi_path}")
    
    return kpis


def calculate_product_growth_rates(spark, df, config=None):
    """
    Calculate month-over-month growth rate for each product.
    
    Formula: Growth Rate = ((Current Month Revenue - Previous Month Revenue)
                            / Previous Month Revenue) × 100
    
    Uses PySpark window functions (lag) for the previous month comparison.
    
    Args:
        spark: SparkSession instance.
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        DataFrame: Product growth rates.
    """
    if config is None:
        config = load_config()
    
    print("\n[KPIs] Calculating product growth rates...")
    
    # Monthly revenue per product
    monthly_product = (
        df.groupBy("Product_ID", "Product_Name", "Year_Month")
        .agg(
            spark_round(spark_sum("Revenue"), 2).alias("Monthly_Revenue"),
            spark_sum("Quantity").alias("Monthly_Quantity")
        )
    )
    
    # Window for lag (previous month's revenue for same product)
    window = Window.partitionBy("Product_ID").orderBy("Year_Month")
    
    growth = (
        monthly_product
        .withColumn("Prev_Month_Revenue", lag("Monthly_Revenue", 1).over(window))
        .withColumn(
            "Growth_Rate_Pct",
            when(col("Prev_Month_Revenue") > 0,
                 spark_round(
                     ((col("Monthly_Revenue") - col("Prev_Month_Revenue"))
                      / col("Prev_Month_Revenue")) * 100, 2
                 ))
            .otherwise(lit(None))
        )
        .orderBy("Product_ID", "Year_Month")
    )
    
    # Save
    output_dir = config["paths"]["analytics_output_dir"]
    path = os.path.join(output_dir, "product_growth_rates.csv")
    growth.toPandas().to_csv(path, index=False)
    print(f"  Saved to: {path}")
    
    return growth


if __name__ == "__main__":
    from src.module3_preprocessing.spark_preprocessing import run_preprocessing
    
    spark, cleaned_df = run_preprocessing()
    kpis = calculate_kpis(spark, cleaned_df)
    growth = calculate_product_growth_rates(spark, cleaned_df)
    
    print("\nReady for Module 5: python -m src.module5_forecasting.feature_engineering")
    spark.stop()
