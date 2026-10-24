# =============================================================================
# Main Pipeline Orchestrator
# Script: run_pipeline.py
# =============================================================================
# PURPOSE:
#   Runs the complete Big Data pipeline end-to-end:
#     Module 3 → Module 4 → Module 5 → Module 6 → Module 8
#
#   (Modules 1-2 are run separately for data preparation.)
#
# HOW TO RUN:
#   python run_pipeline.py
#   python run_pipeline.py --skip-forecast    # Skip ML (faster for testing)
#   python run_pipeline.py --module 3         # Run only Module 3
#
# =============================================================================

import os
import sys
import time
import argparse
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    hadoop_bin = r"C:\hadoop\bin"
    if os.path.exists(hadoop_bin):
        os.environ["HADOOP_HOME"] = os.path.dirname(hadoop_bin)
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + ";" + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config_loader import load_config, ensure_directories


def run_full_pipeline(args):
    """
    Execute the full data processing and analytics pipeline.
    
    Pipeline order:
    1. Module 3: Data Preprocessing (PySpark)
    2. Module 4: Exploratory Analytics (PySpark + Spark SQL)
    3. Module 5: Demand Forecasting (Spark MLlib)
    4. Module 6: Inventory Optimization (PySpark)
    5. Module 8: System Evaluation
    """
    start_time = time.time()
    
    print("=" * 70)
    print("  BIG DATA E-COMMERCE DEMAND FORECASTING & INVENTORY OPTIMIZATION")
    print("  Full Pipeline Execution")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    config = load_config()
    ensure_directories(config)
    
    # Check if enriched dataset exists
    if not os.path.exists(config["paths"]["enriched_dataset_file"]):
        print("\n❌ Enriched dataset not found!")
        print("   Run these commands first:")
        print("   python -m src.module1_ingestion.download_dataset")
        print("   python -m src.module1_ingestion.data_generator")
        return
    
    target_module = args.module if hasattr(args, "module") else None
    skip_forecast = args.skip_forecast if hasattr(args, "skip_forecast") else False
    
    spark = None
    cleaned_df = None
    
    try:
        # =====================================================================
        # Module 3: Data Preprocessing
        # =====================================================================
        if target_module is None or target_module == 3:
            print("\n" + "=" * 60)
            print("▶ RUNNING MODULE 3: Data Preprocessing")
            print("=" * 60)
            
            from src.module3_preprocessing.spark_preprocessing import run_preprocessing
            spark, cleaned_df = run_preprocessing(config)
            
            print("✅ Module 3 complete.\n")
        
        # =====================================================================
        # Module 4: Exploratory Analytics
        # =====================================================================
        if target_module is None or target_module == 4:
            print("\n" + "=" * 60)
            print("▶ RUNNING MODULE 4: Exploratory Big Data Analytics")
            print("=" * 60)
            
            # Load data if not already loaded
            if spark is None or cleaned_df is None:
                from src.module3_preprocessing.spark_preprocessing import (
                    create_spark_session, load_data, handle_missing_values,
                    remove_duplicates, filter_outliers, extract_date_features
                )
                spark = create_spark_session(config)
                cleaned_df = spark.read.parquet(config["paths"]["cleaned_dataset_file"])
            
            from src.module4_analytics.sales_analytics import run_analytics
            from src.module4_analytics.kpi_calculator import calculate_kpis, calculate_product_growth_rates
            
            analytics_results = run_analytics(spark, cleaned_df, config)
            kpis = calculate_kpis(spark, cleaned_df, config)
            growth = calculate_product_growth_rates(spark, cleaned_df, config)
            
            print("✅ Module 4 complete.\n")
        
        # =====================================================================
        # Module 5: Demand Forecasting
        # =====================================================================
        if (target_module is None or target_module == 5) and not skip_forecast:
            print("\n" + "=" * 60)
            print("▶ RUNNING MODULE 5: Demand Forecasting (Spark MLlib)")
            print("=" * 60)
            
            if spark is None:
                from src.module3_preprocessing.spark_preprocessing import create_spark_session
                spark = create_spark_session(config)
            if cleaned_df is None:
                cleaned_df = spark.read.parquet(config["paths"]["cleaned_dataset_file"])
            
            from src.module5_forecasting.advanced_model import run_forecasting
            best_name, best_preds, all_metrics, best_model = run_forecasting(
                spark, cleaned_df, config
            )
            
            print(f"✅ Module 5 complete. Best model: {best_name}\n")
        elif skip_forecast:
            print("\n⏭ Skipping Module 5 (--skip-forecast flag)")
        
        # =====================================================================
        # Module 6: Inventory Optimization
        # =====================================================================
        if target_module is None or target_module == 6:
            print("\n" + "=" * 60)
            print("▶ RUNNING MODULE 6: Inventory Optimization")
            print("=" * 60)
            
            if spark is None:
                from src.module3_preprocessing.spark_preprocessing import create_spark_session
                spark = create_spark_session(config)
            if cleaned_df is None:
                cleaned_df = spark.read.parquet(config["paths"]["cleaned_dataset_file"])
            
            from src.module6_inventory.inventory_optimizer import run_inventory_optimization
            inventory_results = run_inventory_optimization(spark, cleaned_df, config)
            
            print("✅ Module 6 complete.\n")
        
        # =====================================================================
        # Module 8: System Evaluation
        # =====================================================================
        if target_module is None or target_module == 8:
            print("\n" + "=" * 60)
            print("▶ RUNNING MODULE 8: System Evaluation")
            print("=" * 60)
            
            from src.module8_evaluation.system_evaluation import generate_system_evaluation
            evaluation = generate_system_evaluation(config)
            
            print("✅ Module 8 complete.\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        if spark:
            spark.stop()
            print("[Spark] Session stopped.")
    
    # Summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print(f"  Total time: {minutes}m {seconds}s")
    print(f"  Results saved to: {config['paths']['results_dir']}")
    print(f"\n  To view the dashboard:")
    print(f"    streamlit run src/module7_dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the E-Commerce Demand Forecasting pipeline"
    )
    parser.add_argument(
        "--module", type=int, choices=[3, 4, 5, 6, 8],
        help="Run only a specific module (3-8)"
    )
    parser.add_argument(
        "--skip-forecast", action="store_true",
        help="Skip the ML forecasting module (faster for testing)"
    )
    
    args = parser.parse_args()
    run_full_pipeline(args)
