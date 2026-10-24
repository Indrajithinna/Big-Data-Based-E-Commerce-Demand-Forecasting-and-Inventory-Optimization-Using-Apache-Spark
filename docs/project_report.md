# Big Data-Based E-Commerce Demand Forecasting and Inventory Optimization Using Apache Spark

## BE VTU — BIS701 Big Data Analytics Project Report

---

## Abstract

This project presents a Big Data analytics system for e-commerce demand forecasting and inventory optimization using Apache Spark. The system ingests large-scale e-commerce transactional data (~1 million records), performs distributed data processing and cleaning using PySpark, conducts exploratory Big Data analytics with Spark SQL, trains machine learning demand forecasting models using Spark MLlib (Moving Average, Linear Regression, Random Forest, and Gradient-Boosted Trees), and generates inventory optimization recommendations including safety stock, reorder points, and stockout/overstock risk assessment. The results are visualized through an interactive Streamlit-based web dashboard. The project demonstrates practical application of Big Data technologies covered in the BIS701 syllabus including Apache Hadoop/HDFS, Apache Spark, PySpark, Spark SQL, MongoDB, and machine learning on distributed datasets.

**Keywords:** Big Data Analytics, Apache Spark, PySpark, Demand Forecasting, Inventory Optimization, E-Commerce, Machine Learning, Spark MLlib

---

## 1. Introduction

E-commerce businesses generate massive volumes of transactional data daily. Analyzing this data to forecast product demand and optimize inventory levels is critical for profitability. Traditional approaches using tools like Excel or pandas become impractical as data volumes grow into millions or billions of records. Big Data technologies such as Apache Spark enable distributed processing of these large datasets across multiple machines, making complex analytics feasible at scale.

This project builds a complete Big Data analytics pipeline that:
1. Ingests and stores large-scale e-commerce sales data
2. Cleans and transforms data using PySpark distributed processing
3. Computes business analytics and KPIs using Spark SQL
4. Forecasts future product demand using Spark MLlib machine learning
5. Generates inventory optimization recommendations
6. Visualizes results through an interactive web dashboard

### 1.1 Problem Statement

E-commerce retailers face the challenge of maintaining optimal inventory levels. Too much inventory ties up capital and incurs storage costs (overstocking), while too little inventory results in lost sales and customer dissatisfaction (stockouts). Accurate demand forecasting is essential but difficult due to:

- Large volumes of transactional data (millions of records)
- Multiple products, categories, and regions
- Seasonal and temporal demand patterns
- Noisy and incomplete data

Traditional single-machine analytics tools cannot efficiently process datasets at this scale. A Big Data approach using distributed computing is necessary.

### 1.2 Existing System

Conventional demand forecasting systems typically:
- Use spreadsheet-based analysis (Excel) limited to small datasets
- Employ single-machine Python scripts (pandas + scikit-learn)
- Cannot scale beyond the memory of a single computer
- Lack integration between analytics, forecasting, and inventory optimization
- Do not use distributed processing for large-scale data

**Limitations of Existing Systems:**
- Cannot handle datasets with millions or billions of records
- Processing time increases linearly (or worse) with data size
- No distributed computing — single point of failure
- Limited to batch processing, not designed for Big Data pipelines

### 1.3 Proposed System

The proposed system uses Apache Spark as the main distributed processing engine to:
- Process ~1 million e-commerce transactions using PySpark
- Perform SQL-based analytics using Spark SQL (similar to Hive)
- Train machine learning models using Spark MLlib
- Store results in MongoDB (NoSQL) and CSV/Parquet files
- Visualize insights through a Streamlit web dashboard

**Advantages of Proposed System:**
- Distributed processing — scales to any data size
- In-memory computation — 10-100× faster than MapReduce
- Integrated ML pipeline — training on distributed DataFrames
- End-to-end pipeline — ingestion through visualization
- Modular architecture — each component can be upgraded independently

### 1.4 Objectives

1. Build a scalable data processing pipeline using Apache Spark/PySpark
2. Demonstrate distributed SQL analytics using Spark SQL
3. Train and compare multiple demand forecasting models using Spark MLlib
4. Implement inventory optimization formulas for reorder planning
5. Create an interactive dashboard for business intelligence
6. Map the implementation to BIS701 Big Data Analytics concepts

### 1.5 Scope

**In Scope:**
- E-commerce transactional data analysis
- Product-level and category-level demand forecasting
- Inventory optimization (safety stock, reorder point, reorder quantity)
- Stockout and overstock risk assessment
- Interactive web-based dashboard

**Out of Scope:**
- Real-time streaming analytics
- Deep learning models (e.g., LSTM, Transformer)
- Multi-warehouse optimization
- Supply chain logistics optimization
- Customer churn prediction

---

## 2. Literature Survey

> **Note:** The following references are to general concepts and publicly available resources. No fabricated citations are included.

1. **Apache Spark Documentation** — The official Spark documentation describes its architecture, RDD operations, DataFrame API, Spark SQL, and MLlib. Spark's in-memory processing makes it significantly faster than Hadoop MapReduce for iterative algorithms like machine learning.

2. **UCI Machine Learning Repository** — The UCI repository hosts the Online Retail II dataset, a real-world e-commerce transactional dataset from a UK-based retailer with approximately 1 million transactions from 2009-2011.

3. **Inventory Management Theory** — Classic inventory management uses formulas like Economic Order Quantity (EOQ), Safety Stock, and Reorder Point to determine when and how much to order. The Safety Stock formula (Z × σ × √L) accounts for demand uncertainty during lead time.

4. **Time Series Forecasting** — Demand forecasting methods range from simple approaches (moving averages) to machine learning (Random Forest, Gradient-Boosted Trees) and deep learning (LSTM). Tree-based ensemble methods often perform well on tabular data with temporal features.

5. **Big Data Analytics in E-Commerce** — E-commerce companies like Amazon and Alibaba use Big Data technologies for demand forecasting, dynamic pricing, and inventory optimization. These systems process petabytes of data using distributed computing frameworks.

---

## 3. Methodology

### 3.1 System Architecture

```
E-Commerce Sales Data (CSV/Excel)
        ↓
Module 1: Data Ingestion (Python)
        ↓
Module 2: Big Data Storage (HDFS / Local FS + MongoDB)
        ↓
Module 3: Data Preprocessing (PySpark)
        ↓
Module 4: Exploratory Analytics (PySpark + Spark SQL)
        ↓
        ↓→ Feature Engineering (PySpark)
        ↓
Module 5: Demand Forecasting (Spark MLlib)
        ↓
Module 6: Inventory Optimization (PySpark)
        ↓
Module 7: Dashboard (Streamlit + Plotly)
        ↓
Module 8: System Evaluation
```

### 3.2 Data Flow Diagram

**Level 0 (Context):**
```
[E-Commerce Data] → [Demand Forecasting System] → [Inventory Recommendations]
                                                  → [Dashboard/Reports]
```

**Level 1:**
```
[Raw CSV] → [Ingestion] → [Storage] → [PySpark Preprocessing]
   → [Spark SQL Analytics] → [MLlib Forecasting] → [Inventory Optimizer]
   → [MongoDB/CSV] → [Streamlit Dashboard]
```

### 3.3 Use Case Diagram

| Actor | Use Case |
|-------|----------|
| Analyst | Upload sales data |
| Analyst | View sales analytics |
| Analyst | Generate demand forecast |
| Analyst | View inventory recommendations |
| Analyst | Download results |
| Analyst | Filter by date/product/category/region |
| System | Process data with PySpark |
| System | Train ML models |
| System | Calculate reorder points |
| System | Store results in MongoDB |

---

## 4. Module Description

### Module 1: Data Collection and Ingestion
- **Purpose:** Download and prepare the e-commerce dataset
- **Technology:** Python (requests, pandas for Excel conversion only)
- **Input:** UCI Online Retail II dataset (Excel format)
- **Output:** Enriched CSV file with original + simulated fields
- **BIS701:** Data ingestion, ETL pipeline concepts

### Module 2: Big Data Storage
- **Purpose:** Store and organize data for distributed processing
- **Technology:** HDFS (local FS for demo), MongoDB (optional)
- **Input:** Raw and processed data files
- **Output:** Organized storage hierarchy
- **BIS701:** HDFS, MongoDB, distributed vs. NoSQL storage

### Module 3: Data Preprocessing
- **Purpose:** Clean and transform data using distributed processing
- **Technology:** PySpark DataFrames, Spark SQL
- **Operations:** Schema definition, null handling, deduplication, outlier removal, date extraction, data quality validation
- **Input:** Raw enriched CSV
- **Output:** Cleaned Parquet dataset
- **BIS701:** Apache Spark, PySpark, distributed data processing

### Module 4: Exploratory Big Data Analytics
- **Purpose:** Compute business analytics and KPIs
- **Technology:** PySpark (groupBy, agg, window functions), Spark SQL
- **Analytics:** Daily/weekly/monthly sales, product/category/region analysis, seasonal patterns, customer segments
- **KPIs:** Total Revenue, AOV, Daily Demand, Inventory Turnover, Stockout Rate
- **BIS701:** Spark SQL (Hive-like analytics), distributed aggregation

### Module 5: Demand Forecasting
- **Purpose:** Predict future product demand
- **Technology:** Spark MLlib (LinearRegression, RandomForestRegressor, GBTRegressor)
- **Models:** Moving Average, Linear Regression, Random Forest, Gradient-Boosted Trees
- **Features:** Temporal (month, day_of_week), Lag (1/7/14/30 day), Rolling averages, Price, Encoded categoricals
- **Metrics:** MAE, RMSE, MAPE, R²
- **BIS701:** Machine learning on Big Data, Spark MLlib

### Module 6: Inventory Optimization
- **Purpose:** Generate reorder recommendations
- **Technology:** PySpark
- **Formulas:** Safety Stock, Reorder Point, Reorder Quantity, Stockout/Overstock Risk
- **BIS701:** Business logic on distributed data, decision support

### Module 7: Visualization and Dashboard
- **Purpose:** Interactive business intelligence dashboard
- **Technology:** Streamlit, Plotly
- **Pages:** Executive Dashboard, Sales Analytics, Demand Forecast, Inventory Analytics
- **BIS701:** Business Intelligence, data visualization

### Module 8: System Evaluation
- **Purpose:** Evaluate system performance and BIS701 alignment
- **Output:** Evaluation report, model comparison, technology justification

---

## 5. Algorithms

### 5.1 Moving Average
```
Prediction(t) = (1/N) × Σ(Demand(t-i)) for i = 1 to N
```
Simple average of the last N days of demand. N = 7 (configurable).

### 5.2 Linear Regression (Spark MLlib)
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```
Learns optimal weights (β) to minimize squared error. Uses elastic net regularization.

### 5.3 Random Forest Regression (Spark MLlib)
```
Prediction = (1/T) × Σ(Tree_t(x)) for t = 1 to T
```
Ensemble of T decision trees, each trained on a random subset. T = 100 trees, max depth = 10.

### 5.4 Gradient-Boosted Trees (Spark MLlib)
```
F_m(x) = F_{m-1}(x) + η × h_m(x)
```
Sequential boosting where each tree h_m corrects the residuals of F_{m-1}. Learning rate η = 0.1.

### 5.5 Safety Stock Formula
```
Safety_Stock = Z × σ_demand × √(Lead_Time)
```
Where Z = 1.65 (95% service level), σ = std dev of daily demand, Lead_Time in days.

### 5.6 Reorder Point Formula
```
Reorder_Point = Avg_Daily_Demand × Lead_Time + Safety_Stock
```
The inventory level at which a new order should be placed.

---

## 6. Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Programming language |
| Apache Spark | 3.4+ | Distributed processing engine |
| PySpark | 3.4+ | Python API for Spark |
| Spark SQL | (built-in) | SQL analytics on DataFrames |
| Spark MLlib | (built-in) | Machine learning library |
| MongoDB | 6.0+ | NoSQL document database (optional) |
| Streamlit | 1.28+ | Web dashboard framework |
| Plotly | 5.17+ | Interactive visualization |
| Pandas | 2.0+ | Used only for dashboard/small result sets |
| Java | 8/11 | Required by Spark runtime |

---

## 7. Implementation

The project is implemented as a modular Python package with the following structure:

```
project/
├── config/config.yaml          # Central configuration
├── src/
│   ├── config_loader.py        # Configuration utility
│   ├── module1_ingestion/      # Data download & enrichment
│   ├── module2_storage/        # HDFS & MongoDB operations
│   ├── module3_preprocessing/  # PySpark cleaning pipeline
│   ├── module4_analytics/      # Sales analytics & KPIs
│   ├── module5_forecasting/    # ML demand forecasting
│   ├── module6_inventory/      # Inventory optimization
│   ├── module7_dashboard/      # Streamlit web dashboard
│   └── module8_evaluation/     # System evaluation
├── run_pipeline.py             # Main pipeline orchestrator
└── requirements.txt            # Python dependencies
```

All configurable parameters are centralized in `config/config.yaml` to avoid hard-coded values.

---

## 8. Results

> **Note:** Results are generated when the pipeline is executed on the dataset. The values below are placeholders — actual values will be populated after running the pipeline.

### 8.1 Dataset Summary
- **Source:** UCI Online Retail II Dataset
- **Total Records:** ~1,067,371 (before cleaning)
- **Cleaned Records:** To be determined after execution
- **Date Range:** December 2009 — December 2011

### 8.2 Model Performance
Model performance will be evaluated using MAE, RMSE, MAPE, and R² on a chronological test set (last 20% of data by date).

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Moving Average | TBD | TBD | TBD | TBD |
| Linear Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| Gradient-Boosted Trees | TBD | TBD | TBD | TBD |

> Results will be populated when the pipeline is executed. The best model is selected empirically based on lowest RMSE.

### 8.3 Inventory Optimization Summary
- Total products analyzed: TBD
- Products at high stockout risk: TBD
- Products overstocked: TBD
- Urgent reorders needed: TBD

---

## 9. Performance Evaluation

### 9.1 Processing Performance
- Dataset size: ~1M records
- Spark processing mode: local[*] (all available cores)
- Expected processing time: 2-5 minutes (varies by hardware)

### 9.2 Model Evaluation Metrics

**MAE (Mean Absolute Error):** Average absolute difference between actual and predicted demand. Interpretable in original units (e.g., "off by 5 units on average").

**RMSE (Root Mean Squared Error):** Square root of average squared errors. Penalizes large errors more than MAE. Used as the primary model selection criterion.

**MAPE (Mean Absolute Percentage Error):** Percentage error relative to actual values. Scale-independent, useful for comparing across products. Not computed when actual demand is zero.

**R² (Coefficient of Determination):** Proportion of variance explained by the model. Values closer to 1.0 indicate better fit.

---

## 10. BIS701 Syllabus Alignment

| BIS701 Concept | Project Implementation |
|---------------|----------------------|
| **Hadoop / HDFS** | Module 2: HDFS operations for distributed file storage |
| **MapReduce** | Spark's DAG execution generalizes MapReduce; groupBy+agg patterns |
| **Apache Spark** | Modules 3-6: Main processing engine for all analytics and ML |
| **PySpark** | Python API used throughout for DataFrame operations |
| **Spark SQL** | Module 3 validation, Module 4 analytics (registerTempView + sql()) |
| **Hive** | Spark SQL provides Hive-compatible SQL interface |
| **MongoDB** | Module 2: NoSQL storage for results (optional) |
| **Machine Learning** | Module 5: 4 models via Spark MLlib |
| **Visualization** | Module 7: Streamlit + Plotly dashboard |
| **NoSQL** | MongoDB document storage demonstrates schema-flexible databases |
| **Big Data Pipeline** | End-to-end: Ingestion → Processing → Analytics → ML → Visualization |

---

## 11. Limitations

1. **Dataset size:** The UCI dataset (~1M records) is moderate. True Big Data benefits emerge at billions of records. However, the pipeline is designed to scale.
2. **Simulated fields:** Product_Category, Discount, Inventory_Level, Supplier, and Delivery_Date are simulated, not from a real warehouse system.
3. **Local mode:** Spark runs in local mode for demo. A multi-node cluster would demonstrate full distributed processing.
4. **Static inventory:** Inventory levels do not update dynamically as orders are placed.
5. **No streaming:** The system uses batch processing only. Real-time demand forecasting would require Spark Structured Streaming.
6. **Limited time series:** The dataset covers only ~2 years. Longer history would improve seasonal pattern detection.

---

## 12. Future Scope

1. **Real-time streaming:** Implement Spark Structured Streaming for live demand forecasting
2. **Deep learning models:** Add LSTM or Transformer-based time series models
3. **Multi-node cluster:** Deploy on AWS EMR, Azure HDInsight, or Google Cloud Dataproc
4. **Real-time inventory tracking:** Connect to warehouse management systems via APIs
5. **Dynamic pricing:** Integrate demand forecasting with pricing optimization
6. **Customer segmentation:** Add RFM analysis and customer lifetime value prediction
7. **A/B testing:** Test different forecasting models in production with online evaluation

---

## 13. Conclusion

This project demonstrates the practical application of Big Data technologies for e-commerce demand forecasting and inventory optimization. By using Apache Spark as the main processing engine, the system efficiently handles large-scale transactional data through distributed DataFrames and Spark SQL. The Spark MLlib-based forecasting pipeline compares four models (Moving Average, Linear Regression, Random Forest, and Gradient-Boosted Trees) and selects the best performer empirically. Inventory optimization recommendations are generated using established operations research formulas for safety stock, reorder points, and risk assessment. The Streamlit dashboard provides an interactive interface for business users to explore analytics and make data-driven inventory decisions.

The project covers key BIS701 syllabus concepts including Hadoop/HDFS, Apache Spark, PySpark, Spark SQL, MongoDB, machine learning on Big Data, and business intelligence visualization, providing a complete end-to-end demonstration of Big Data analytics in a real-world business context.

---

## 14. References

1. Apache Spark Documentation. https://spark.apache.org/docs/latest/
2. PySpark API Reference. https://spark.apache.org/docs/latest/api/python/
3. UCI Machine Learning Repository — Online Retail II Dataset. https://archive.ics.uci.edu/dataset/502/online+retail+ii
4. MongoDB Documentation. https://www.mongodb.com/docs/
5. Streamlit Documentation. https://docs.streamlit.io/
6. Plotly Python Documentation. https://plotly.com/python/
7. Apache Hadoop Documentation. https://hadoop.apache.org/docs/
8. Spark MLlib Guide. https://spark.apache.org/docs/latest/ml-guide.html

> **Note:** All references are to official documentation and publicly accessible resources. No fabricated research papers or citations are included.
