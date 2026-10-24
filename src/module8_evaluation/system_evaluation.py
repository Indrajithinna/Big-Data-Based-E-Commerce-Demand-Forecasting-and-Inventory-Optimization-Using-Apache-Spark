# =============================================================================
# Module 8: System Evaluation
# Script: system_evaluation.py
# =============================================================================
# PURPOSE:
#   Evaluates the overall system performance, generates a summary report,
#   and maps the project implementation to BIS701 syllabus concepts.
#
# BIS701 RELEVANCE:
#   - Maps each module to Big Data Analytics concepts
#   - Summarizes technology usage and justification
#   - Documents model performance for the project report
# =============================================================================

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config


def generate_system_evaluation(config=None):
    """
    Generate a comprehensive system evaluation report.
    
    Loads results from all modules and creates a structured evaluation.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        dict: Evaluation report.
    """
    if config is None:
        config = load_config()
    
    print("=" * 60)
    print("Module 8: System Evaluation")
    print("=" * 60)
    
    analytics_dir = config["paths"]["analytics_output_dir"]
    forecast_dir = config["paths"]["forecast_output_dir"]
    inventory_dir = config["paths"]["inventory_output_dir"]
    
    report = {
        "title": "System Evaluation Report",
        "project": "Big Data-Based E-Commerce Demand Forecasting & Inventory Optimization",
        "generated_at": datetime.now().isoformat(),
        "modules": {},
        "bis701_mapping": {},
        "technology_justification": {},
    }
    
    # --- Module Evaluation ---
    
    # Module 1: Data Ingestion
    report["modules"]["module_1_ingestion"] = {
        "name": "Data Collection & Ingestion",
        "status": "Complete" if os.path.exists(config["paths"]["enriched_dataset_file"]) else "Not Run",
        "description": "Downloaded UCI Online Retail II dataset, converted from Excel to CSV, "
                       "enriched with simulated fields (Product_Category, Discount, Inventory_Level, "
                       "Supplier, Delivery_Date).",
        "bis701_concepts": ["Data Ingestion", "ETL Pipeline", "Data Enrichment"],
    }
    
    # Module 3: Preprocessing
    report["modules"]["module_3_preprocessing"] = {
        "name": "Data Preprocessing",
        "status": "Complete" if os.path.exists(os.path.join(config["paths"]["processed_data_dir"], "cleaned_sales.csv")) else "Not Run",
        "description": "PySpark pipeline: schema definition, missing value handling, "
                       "deduplication, outlier removal, date feature extraction, data quality validation.",
        "bis701_concepts": ["Apache Spark", "PySpark DataFrames", "Spark SQL",
                            "Distributed Data Processing", "Data Quality"],
        "spark_operations": [
            "SparkSession creation", "CSV ingestion with explicit schema",
            "dropna()", "dropDuplicates()", "filter()", "withColumn()",
            "createOrReplaceTempView()", "spark.sql()", "write.parquet()",
        ]
    }
    
    # Module 4: Analytics
    kpi_path = os.path.join(analytics_dir, "kpis.json")
    kpis = {}
    if os.path.exists(kpi_path):
        with open(kpi_path) as f:
            kpis = json.load(f)
    
    report["modules"]["module_4_analytics"] = {
        "name": "Exploratory Big Data Analytics",
        "status": "Complete" if kpis else "Not Run",
        "kpis": kpis,
        "analytics_produced": [
            "Daily/Weekly/Monthly sales aggregation",
            "Product-wise analysis (top sellers, slow movers)",
            "Category-wise sales distribution",
            "Region-wise sales analysis",
            "Seasonal patterns (monthly, day-of-week)",
            "Customer purchasing patterns & segments",
            "KPI calculation (Revenue, AOV, Turnover, Stockout Rate)",
        ],
        "bis701_concepts": ["Spark SQL", "Distributed Aggregation",
                            "Window Functions", "Business Intelligence"],
    }
    
    # Module 5: Forecasting
    model_comp_path = os.path.join(forecast_dir, "model_comparison.json")
    model_comp = {}
    if os.path.exists(model_comp_path):
        with open(model_comp_path) as f:
            model_comp = json.load(f)
    
    report["modules"]["module_5_forecasting"] = {
        "name": "Demand Forecasting",
        "status": "Complete" if model_comp else "Not Run",
        "models": model_comp.get("models", {}),
        "best_model": model_comp.get("best_model", "Not determined"),
        "selection_criterion": model_comp.get("selection_criterion", ""),
        "bis701_concepts": ["Spark MLlib", "Machine Learning on Big Data",
                            "Feature Engineering", "Model Evaluation"],
        "ml_techniques": [
            "Moving Average (PySpark window functions)",
            "Linear Regression (Spark MLlib)",
            "Random Forest Regression (Spark MLlib)",
            "Gradient-Boosted Trees (Spark MLlib)",
        ],
    }
    
    # Module 6: Inventory
    inv_summary_path = os.path.join(inventory_dir, "inventory_summary.json")
    inv_summary = {}
    if os.path.exists(inv_summary_path):
        with open(inv_summary_path) as f:
            inv_summary = json.load(f)
    
    report["modules"]["module_6_inventory"] = {
        "name": "Inventory Optimization",
        "status": "Complete" if inv_summary else "Not Run",
        "summary": inv_summary,
        "formulas": {
            "Safety Stock": "Z × σ(daily_demand) × √(Lead_Time)",
            "Reorder Point": "Avg_Daily_Demand × Lead_Time + Safety_Stock",
            "Reorder Quantity": "max(0, Predicted_Demand − Current_Inventory + Safety_Stock)",
            "Stockout Risk": "Current_Inventory < Reorder_Point",
            "Overstock Risk": "Current_Inventory > 2 × Predicted_Demand",
        },
        "bis701_concepts": ["Business Logic on Big Data", "Decision Support Systems"],
    }
    
    # --- BIS701 Syllabus Mapping ---
    report["bis701_mapping"] = {
        "Hadoop / HDFS": {
            "concept": "Distributed file storage for large-scale data",
            "implementation": "HDFS operations module (Module 2). In local mode, mirrors HDFS "
                              "directory structure. Data organized as raw → processed → results.",
            "value": "Demonstrates how Big Data is stored across a distributed cluster.",
        },
        "MapReduce": {
            "concept": "Distributed processing paradigm (map + reduce)",
            "implementation": "Spark's internal execution engine uses DAG-based computation "
                              "that generalizes MapReduce. GroupBy + Agg operations in PySpark "
                              "are the modern equivalent of MapReduce patterns.",
            "value": "Spark provides a more efficient superset of MapReduce.",
        },
        "Apache Spark": {
            "concept": "In-memory distributed computing engine",
            "implementation": "Main processing engine for Modules 3-6. All data cleaning, "
                              "analytics, feature engineering, and ML use PySpark.",
            "value": "10-100× faster than MapReduce for iterative workloads like ML.",
        },
        "PySpark": {
            "concept": "Python API for Apache Spark",
            "implementation": "Used throughout the project for DataFrame operations, "
                              "Spark SQL queries, and MLlib model training.",
            "value": "Enables Python developers to leverage Spark's distributed computing.",
        },
        "Spark SQL": {
            "concept": "SQL interface for Spark DataFrames",
            "implementation": "Data quality validation (Module 3), seasonal analysis (Module 4), "
                              "KPI calculation (Module 4). Uses registerTempView + spark.sql().",
            "value": "Familiar SQL syntax for complex analytical queries on Big Data.",
        },
        "Hive": {
            "concept": "Data warehouse on Hadoop, SQL-based analytics",
            "implementation": "Spark SQL provides Hive-compatible SQL analytics. "
                              "The project uses Spark's built-in Hive support through "
                              "createOrReplaceTempView and SQL queries.",
            "value": "Spark SQL is the modern equivalent of Hive for ad-hoc SQL analytics.",
        },
        "MongoDB": {
            "concept": "NoSQL document-oriented database",
            "implementation": "Module 2: Optional storage for results, forecasts, and "
                              "inventory recommendations. Schema-flexible document storage.",
            "value": "Demonstrates NoSQL storage for semi-structured analytics results.",
        },
        "Machine Learning": {
            "concept": "Predictive modeling on large datasets",
            "implementation": "Module 5: Four models trained using Spark MLlib — "
                              "Moving Average, Linear Regression, Random Forest, GBT.",
            "value": "Distributed ML training on Big Data using Spark's native ML library.",
        },
        "Visualization / BI": {
            "concept": "Business Intelligence dashboards",
            "implementation": "Module 7: Streamlit dashboard with Plotly charts. "
                              "Executive, Sales, Forecast, and Inventory pages.",
            "value": "Translates Big Data analytics into actionable business insights.",
        },
    }
    
    # --- Technology Justification ---
    report["technology_justification"] = {
        "Why Spark over pandas": (
            "Spark processes data in a distributed manner across multiple cores/nodes. "
            "For the ~1M record UCI dataset, this demonstrates scalability. In production "
            "with billions of records, Spark would be essential while pandas would fail."
        ),
        "Why Spark MLlib over scikit-learn": (
            "Spark MLlib trains models on distributed DataFrames without collecting "
            "data to a single machine. For large datasets, this avoids memory limits. "
            "The project uses MLlib's RandomForestRegressor and GBTRegressor."
        ),
        "Why Parquet over CSV": (
            "Parquet is a columnar storage format with built-in compression and schema. "
            "It's 2-10× faster for analytical queries and uses 75% less disk space."
        ),
        "Why MongoDB (optional)": (
            "MongoDB stores semi-structured analytics results (KPIs, forecasts, "
            "recommendations) that don't fit a fixed relational schema. Demonstrates "
            "NoSQL concepts from the BIS701 syllabus."
        ),
        "Why Streamlit over Power BI": (
            "Streamlit integrates directly with the Python analytics pipeline. "
            "No separate license required. Results are displayed in real-time "
            "from the same data files produced by Spark."
        ),
    }
    
    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("  SYSTEM EVALUATION SUMMARY")
    print("=" * 60)
    
    for mod_key, mod_info in report["modules"].items():
        status = mod_info["status"]
        name = mod_info["name"]
        emoji = "✅" if status == "Complete" else "❌"
        print(f"  {emoji} {name}: {status}")
    
    print(f"\n  BIS701 Concepts Covered: {len(report['bis701_mapping'])}")
    
    if model_comp:
        print(f"\n  Best Forecasting Model: {model_comp.get('best_model', 'N/A')}")
        best_metrics = model_comp.get("models", {}).get(model_comp.get("best_model", ""), {})
        if best_metrics:
            print(f"    MAE:  {best_metrics.get('MAE', 'N/A')}")
            print(f"    RMSE: {best_metrics.get('RMSE', 'N/A')}")
            print(f"    R²:   {best_metrics.get('R2', 'N/A')}")
    
    print("=" * 60)
    
    # Save report
    output_dir = config["paths"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "system_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    report = generate_system_evaluation()
