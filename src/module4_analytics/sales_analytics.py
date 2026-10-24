# =============================================================================
# Module 4: Exploratory Big Data Analytics
# Script: sales_analytics.py
# =============================================================================
# PURPOSE:
#   Performs comprehensive sales analytics using PySpark DataFrame operations
#   and Spark SQL. Computes daily/weekly/monthly sales, product-wise and
#   category-wise breakdowns, regional analysis, and temporal trends.
#
# ANALYTICS PRODUCED:
#   - Daily / Weekly / Monthly sales aggregation
#   - Product-wise sales (top sellers & slow movers)
#   - Category-wise sales distribution
#   - Region-wise sales analysis
#   - Revenue and quantity trends
#   - Seasonal patterns
#   - Customer purchasing patterns
#
# BIS701 RELEVANCE:
#   - PySpark DataFrame operations (groupBy, agg, join, window functions)
#   - Spark SQL queries (registerTempView, spark.sql)
#   - Distributed aggregation and analytics
#   - Big Data analytical processing patterns
#
# INPUT:  Cleaned sales DataFrame (from Module 3)
# OUTPUT: Analytics CSV files in data/results/analytics/
# =============================================================================

import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    stddev, round as spark_round, desc, asc, lit, lag,
    year, month, dayofmonth, dayofweek, quarter, weekofyear,
    date_format, to_date, datediff,
    countDistinct, when, row_number, rank, dense_rank,
    first, last, collect_list, concat_ws
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config, ensure_directories


def daily_sales_analysis(df, output_dir):
    """
    Aggregate sales data at daily granularity.
    
    Groups all transactions by date and computes:
    - Total orders, units sold, revenue per day
    - Average order value per day
    
    Args:
        df: Cleaned Spark DataFrame.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Daily sales aggregation.
    """
    print("\n[Analytics] Daily Sales Analysis...")
    
    daily = (
        df.groupBy("Order_Date_Only")
        .agg(
            countDistinct("Order_ID").alias("Num_Orders"),
            spark_sum("Quantity").alias("Total_Quantity"),
            spark_round(spark_sum("Revenue"), 2).alias("Total_Revenue"),
            spark_round(avg("Revenue"), 2).alias("Avg_Order_Value"),
            countDistinct("Customer_ID").alias("Unique_Customers"),
            countDistinct("Product_ID").alias("Unique_Products")
        )
        .orderBy("Order_Date_Only")
    )
    
    row_count = daily.count()
    print(f"  Generated {row_count:,} daily records")
    
    # Save to CSV
    path = os.path.join(output_dir, "daily_sales.csv")
    daily.toPandas().to_csv(path, index=False)
    print(f"  Saved to: {path}")
    
    return daily


def weekly_sales_analysis(df, output_dir):
    """
    Aggregate sales data at weekly granularity.
    
    Args:
        df: Cleaned Spark DataFrame.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Weekly sales aggregation.
    """
    print("\n[Analytics] Weekly Sales Analysis...")
    
    weekly = (
        df.groupBy("Year", "Week_of_Year")
        .agg(
            countDistinct("Order_ID").alias("Num_Orders"),
            spark_sum("Quantity").alias("Total_Quantity"),
            spark_round(spark_sum("Revenue"), 2).alias("Total_Revenue"),
            spark_round(avg("Revenue"), 2).alias("Avg_Order_Value"),
            countDistinct("Customer_ID").alias("Unique_Customers")
        )
        .orderBy("Year", "Week_of_Year")
    )
    
    row_count = weekly.count()
    print(f"  Generated {row_count:,} weekly records")
    
    path = os.path.join(output_dir, "weekly_sales.csv")
    weekly.toPandas().to_csv(path, index=False)
    print(f"  Saved to: {path}")
    
    return weekly


def monthly_sales_analysis(df, output_dir):
    """
    Aggregate sales data at monthly granularity.
    
    Args:
        df: Cleaned Spark DataFrame.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Monthly sales aggregation.
    """
    print("\n[Analytics] Monthly Sales Analysis...")
    
    monthly = (
        df.groupBy("Year_Month", "Year", "Month")
        .agg(
            countDistinct("Order_ID").alias("Num_Orders"),
            spark_sum("Quantity").alias("Total_Quantity"),
            spark_round(spark_sum("Revenue"), 2).alias("Total_Revenue"),
            spark_round(avg("Revenue"), 2).alias("Avg_Order_Value"),
            countDistinct("Customer_ID").alias("Unique_Customers"),
            countDistinct("Product_ID").alias("Unique_Products")
        )
        .orderBy("Year_Month")
    )
    
    row_count = monthly.count()
    print(f"  Generated {row_count:,} monthly records")
    
    path = os.path.join(output_dir, "monthly_sales.csv")
    monthly.toPandas().to_csv(path, index=False)
    print(f"  Saved to: {path}")
    
    return monthly


def product_wise_analysis(df, output_dir, top_n=20):
    """
    Analyze sales by product — find top sellers and slow movers.
    
    Uses PySpark window functions for ranking, which is a key
    Big Data concept for distributed analytical operations.
    
    Args:
        df: Cleaned Spark DataFrame.
        output_dir: Directory to save results.
        top_n: Number of top products to report.
    
    Returns:
        tuple: (product_sales_df, top_products_df, slow_products_df)
    """
    print(f"\n[Analytics] Product-wise Sales Analysis (top {top_n})...")
    
    product_sales = (
        df.groupBy("Product_ID", "Product_Name", "Product_Category")
        .agg(
            countDistinct("Order_ID").alias("Num_Orders"),
            spark_sum("Quantity").alias("Total_Quantity"),
            spark_round(spark_sum("Revenue"), 2).alias("Total_Revenue"),
            spark_round(avg("Unit_Price"), 2).alias("Avg_Unit_Price"),
            spark_round(avg("Discount"), 4).alias("Avg_Discount"),
            countDistinct("Customer_ID").alias("Unique_Customers"),
            countDistinct("Region").alias("Num_Regions")
        )
    )
    
    # Rank products by revenue using window function
    window = Window.orderBy(desc("Total_Revenue"))
    product_sales = product_sales.withColumn("Revenue_Rank", dense_rank().over(window))
    
    # Top sellers
    top_products = product_sales.filter(col("Revenue_Rank") <= top_n).orderBy("Revenue_Rank")
    
    # Slow movers (bottom by quantity)
    window_qty = Window.orderBy(asc("Total_Quantity"))
    product_sales_ranked = product_sales.withColumn("Qty_Rank", dense_rank().over(window_qty))
    slow_products = product_sales_ranked.filter(col("Qty_Rank") <= top_n).orderBy("Qty_Rank")
    
    total_products = product_sales.count()
    print(f"  Total unique products: {total_products:,}")
    
    # Save all results
    path = os.path.join(output_dir, "product_sales.csv")
    product_sales.toPandas().to_csv(path, index=False)
    print(f"  All products saved to: {path}")
    
    path = os.path.join(output_dir, "top_products.csv")
    top_products.toPandas().to_csv(path, index=False)
    print(f"  Top {top_n} products saved to: {path}")
    
    path = os.path.join(output_dir, "slow_products.csv")
    slow_products.toPandas().to_csv(path, index=False)
    print(f"  Slow-moving products saved to: {path}")
    
    return product_sales, top_products, slow_products


def category_wise_analysis(df, output_dir):
    """
    Analyze sales distribution across product categories.
    
    Args:
        df: Cleaned Spark DataFrame.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Category-level aggregation.
    """
    print("\n[Analytics] Category-wise Sales Analysis...")
    
    category_sales = (
        df.groupBy("Product_Category")
        .agg(
            countDistinct("Order_ID").alias("Num_Orders"),
            countDistinct("Product_ID").alias("Num_Products"),
            spark_sum("Quantity").alias("Total_Quantity"),
            spark_round(spark_sum("Revenue"), 2).alias("Total_Revenue"),
            spark_round(avg("Revenue"), 2).alias("Avg_Revenue"),
            spark_round(avg("Discount"), 4).alias("Avg_Discount"),
            countDistinct("Customer_ID").alias("Unique_Customers")
        )
        .orderBy(desc("Total_Revenue"))
    )
    
    print(f"  Categories found: {category_sales.count()}")
    
    path = os.path.join(output_dir, "category_sales.csv")
    category_sales.toPandas().to_csv(path, index=False)
    print(f"  Saved to: {path}")
    
    return category_sales


def region_wise_analysis(df, output_dir):
    """
    Analyze sales by region/country.
    
    Args:
        df: Cleaned Spark DataFrame.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Region-level aggregation.
    """
    print("\n[Analytics] Region-wise Sales Analysis...")
    
    region_sales = (
        df.groupBy("Region")
        .agg(
            countDistinct("Order_ID").alias("Num_Orders"),
            countDistinct("Product_ID").alias("Num_Products"),
            spark_sum("Quantity").alias("Total_Quantity"),
            spark_round(spark_sum("Revenue"), 2).alias("Total_Revenue"),
            spark_round(avg("Revenue"), 2).alias("Avg_Revenue"),
            countDistinct("Customer_ID").alias("Unique_Customers")
        )
        .orderBy(desc("Total_Revenue"))
    )
    
    print(f"  Regions found: {region_sales.count()}")
    
    path = os.path.join(output_dir, "region_sales.csv")
    region_sales.toPandas().to_csv(path, index=False)
    print(f"  Saved to: {path}")
    
    return region_sales


def seasonal_analysis(df, spark, output_dir):
    """
    Detect seasonal patterns using Spark SQL.
    
    Analyzes sales by month-of-year across all years to identify
    recurring seasonal peaks and troughs.
    
    Demonstrates Spark SQL for complex analytical queries.
    
    Args:
        df: Cleaned Spark DataFrame.
        spark: SparkSession instance.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Seasonal pattern data.
    """
    print("\n[Analytics] Seasonal Pattern Analysis (Spark SQL)...")
    
    # Register as temp view for SQL queries
    df.createOrReplaceTempView("sales")
    
    # Monthly seasonality — average sales per month-of-year across all years
    seasonal = spark.sql("""
        SELECT
            Month,
            COUNT(DISTINCT Year) as Years_Covered,
            ROUND(AVG(monthly_qty), 0) as Avg_Monthly_Quantity,
            ROUND(AVG(monthly_rev), 2) as Avg_Monthly_Revenue,
            ROUND(AVG(monthly_orders), 0) as Avg_Monthly_Orders
        FROM (
            SELECT
                Year,
                Month,
                SUM(Quantity) as monthly_qty,
                SUM(Revenue) as monthly_rev,
                COUNT(DISTINCT Order_ID) as monthly_orders
            FROM sales
            GROUP BY Year, Month
        ) monthly_agg
        GROUP BY Month
        ORDER BY Month
    """)
    
    print("  Monthly Seasonal Averages:")
    seasonal.show(12, truncate=False)
    
    # Day-of-week patterns
    dow_pattern = spark.sql("""
        SELECT
            Day_of_Week,
            CASE Day_of_Week
                WHEN 1 THEN 'Sunday'
                WHEN 2 THEN 'Monday'
                WHEN 3 THEN 'Tuesday'
                WHEN 4 THEN 'Wednesday'
                WHEN 5 THEN 'Thursday'
                WHEN 6 THEN 'Friday'
                WHEN 7 THEN 'Saturday'
            END as Day_Name,
            COUNT(*) as Num_Transactions,
            ROUND(SUM(Revenue), 2) as Total_Revenue,
            ROUND(AVG(Quantity), 2) as Avg_Quantity
        FROM sales
        GROUP BY Day_of_Week
        ORDER BY Day_of_Week
    """)
    
    print("  Day-of-Week Patterns:")
    dow_pattern.show(truncate=False)
    
    # Save results
    path = os.path.join(output_dir, "seasonal_monthly.csv")
    seasonal.toPandas().to_csv(path, index=False)
    
    path = os.path.join(output_dir, "seasonal_dow.csv")
    dow_pattern.toPandas().to_csv(path, index=False)
    
    print(f"  Saved seasonal patterns to {output_dir}")
    
    return seasonal


def customer_analysis(df, spark, output_dir):
    """
    Analyze customer purchasing patterns.
    
    Demonstrates complex Spark SQL queries including subqueries,
    window functions, and CASE statements.
    
    Args:
        df: Cleaned Spark DataFrame.
        spark: SparkSession instance.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame: Customer analysis results.
    """
    print("\n[Analytics] Customer Purchasing Analysis (Spark SQL)...")
    
    df.createOrReplaceTempView("sales")
    
    # Customer summary
    customer_summary = spark.sql("""
        SELECT
            Customer_ID,
            COUNT(DISTINCT Order_ID) as Total_Orders,
            ROUND(SUM(Revenue), 2) as Total_Spent,
            ROUND(AVG(Revenue), 2) as Avg_Order_Value,
            SUM(Quantity) as Total_Items,
            COUNT(DISTINCT Product_ID) as Unique_Products,
            MIN(Order_Date) as First_Purchase,
            MAX(Order_Date) as Last_Purchase,
            DATEDIFF(MAX(Order_Date), MIN(Order_Date)) as Customer_Lifetime_Days
        FROM sales
        WHERE Customer_ID != 'Unknown'
        GROUP BY Customer_ID
        ORDER BY Total_Spent DESC
    """)
    
    total_customers = customer_summary.count()
    print(f"  Total customers: {total_customers:,}")
    
    # Customer segments by spending (using Spark SQL CASE)
    customer_segments = spark.sql("""
        SELECT
            CASE
                WHEN total_spent > 5000 THEN 'High Value'
                WHEN total_spent > 1000 THEN 'Medium Value'
                WHEN total_spent > 100 THEN 'Low Value'
                ELSE 'Minimal'
            END as Segment,
            COUNT(*) as Num_Customers,
            ROUND(AVG(total_spent), 2) as Avg_Spending,
            ROUND(SUM(total_spent), 2) as Total_Revenue
        FROM (
            SELECT
                Customer_ID,
                SUM(Revenue) as total_spent
            FROM sales
            WHERE Customer_ID != 'Unknown'
            GROUP BY Customer_ID
        ) cust
        GROUP BY
            CASE
                WHEN total_spent > 5000 THEN 'High Value'
                WHEN total_spent > 1000 THEN 'Medium Value'
                WHEN total_spent > 100 THEN 'Low Value'
                ELSE 'Minimal'
            END
        ORDER BY Avg_Spending DESC
    """)
    
    print("\n  Customer Segments:")
    customer_segments.show(truncate=False)
    
    # Save
    path = os.path.join(output_dir, "customer_summary.csv")
    # Save only top 1000 customers for dashboard (full dataset stays in Spark)
    customer_summary.limit(1000).toPandas().to_csv(path, index=False)
    
    path = os.path.join(output_dir, "customer_segments.csv")
    customer_segments.toPandas().to_csv(path, index=False)
    
    print(f"  Saved customer analytics to {output_dir}")
    
    return customer_summary


def run_analytics(spark, df, config=None):
    """
    Execute the full analytics pipeline.
    
    Args:
        spark: SparkSession instance.
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        dict: Dictionary of analytics DataFrames.
    """
    if config is None:
        config = load_config()
    
    output_dir = config["paths"]["analytics_output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    top_n = config["analytics"]["top_n"]
    
    print("=" * 60)
    print("Module 4: Exploratory Big Data Analytics")
    print("=" * 60)
    
    results = {}
    
    # Run all analytics
    results["daily"] = daily_sales_analysis(df, output_dir)
    results["weekly"] = weekly_sales_analysis(df, output_dir)
    results["monthly"] = monthly_sales_analysis(df, output_dir)
    
    product_all, product_top, product_slow = product_wise_analysis(df, output_dir, top_n)
    results["product_sales"] = product_all
    results["top_products"] = product_top
    results["slow_products"] = product_slow
    
    results["category_sales"] = category_wise_analysis(df, output_dir)
    results["region_sales"] = region_wise_analysis(df, output_dir)
    results["seasonal"] = seasonal_analysis(df, spark, output_dir)
    results["customers"] = customer_analysis(df, spark, output_dir)
    
    print(f"\n{'='*60}")
    print("Analytics pipeline complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    from src.module3_preprocessing.spark_preprocessing import run_preprocessing
    
    spark, cleaned_df = run_preprocessing()
    results = run_analytics(spark, cleaned_df)
    
    print("\nReady for Module 4 KPIs: python -m src.module4_analytics.kpi_calculator")
    spark.stop()
