# Big Data-Based E-Commerce Demand Forecasting and Inventory Optimization Using Apache Spark

> **BE VTU — BIS701 Big Data Analytics Project**

## Overview

This project builds a Big Data analytics system that analyzes large-scale e-commerce sales data, forecasts future product demand using Spark MLlib, and provides inventory optimization recommendations. It demonstrates practical use of Apache Spark, PySpark, Spark SQL, MongoDB, and machine learning on Big Data.

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| Apache Spark / PySpark | Distributed data processing & analytics engine |
| Spark SQL | SQL-based analytics on Big Data |
| Spark MLlib | Machine learning for demand forecasting |
| MongoDB | NoSQL storage for results (optional) |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive visualizations |
| Python 3.x | Application logic |

## Project Modules

1. **Data Collection & Ingestion** — Download and prepare e-commerce dataset
2. **Big Data Storage** — Store data in HDFS/local filesystem + MongoDB
3. **Data Preprocessing** — Clean and transform data using PySpark
4. **Exploratory Big Data Analytics** — Sales analytics, KPIs using Spark SQL
5. **Demand Forecasting** — ML models (Moving Avg, Linear Reg, Random Forest, GBT)
6. **Inventory Optimization** — Safety stock, reorder points, stockout/overstock risk
7. **Visualization & Dashboard** — Streamlit-based interactive dashboard
8. **System Evaluation** — Model metrics, performance analysis

## Prerequisites

- Python 3.9+
- Java 8 or 11 (required for Spark)
- Apache Spark 3.4+ (bundled with PySpark)
- MongoDB (optional — set `mongodb.enabled: true` in config)

## Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd "Big Data-Based E-Commerce Demand Forecasting and Inventory Optimization Using Apache Spark"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and prepare dataset
python -m src.module1_ingestion.download_dataset
python -m src.module1_ingestion.data_generator

# 5. Run the full pipeline
python run_pipeline.py

# 6. Launch dashboard
streamlit run src/module7_dashboard/app.py
```

## Configuration

All parameters are in `config/config.yaml`. Key settings:
- Spark memory allocation
- ML model hyperparameters
- Inventory optimization parameters (lead time, service level)
- MongoDB connection (disabled by default)

## Dataset

Uses the **UCI Online Retail II** dataset with simulated enrichment fields.
See `docs/project_report.md` for full field classification (original vs. simulated).

## License

Academic use — BE VTU BIS701 Project.
