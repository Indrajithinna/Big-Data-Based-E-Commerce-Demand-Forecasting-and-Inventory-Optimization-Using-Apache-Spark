# =============================================================================
# Module 5: Demand Forecasting
# Script: baseline_model.py
# =============================================================================
# PURPOSE:
#   Implements baseline demand forecasting models:
#   1. Moving Average — simple average of recent demand (no ML)
#   2. Linear Regression — basic ML baseline using Spark MLlib
#
#   These baselines establish a performance floor. More complex models
#   (Module 5: advanced_model.py) should ideally beat these baselines.
#
# BIS701 RELEVANCE:
#   - Spark MLlib LinearRegression → distributed ML training
#   - Moving Average via PySpark window functions
#   - Model evaluation with MAE, RMSE, MAPE
#   - Demonstrates ML pipeline pattern: Estimator → Model → Predictions
#
# INPUT:  Feature-engineered train/test DataFrames
# OUTPUT: Predictions DataFrame, model metrics
# =============================================================================

import os
import sys

from pyspark.sql import Window
from pyspark.sql.functions import (
    col, avg, abs as spark_abs, sqrt as spark_sqrt,
    sum as spark_sum, count, when, lit,
    round as spark_round, coalesce
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config


def evaluate_model(predictions_df, label_col="Daily_Demand", prediction_col="prediction"):
    """
    Evaluate forecasting model using MAE, RMSE, and MAPE.
    
    METRICS EXPLAINED:
    - MAE (Mean Absolute Error): Average of |actual - predicted|.
      Interpretable in original units (e.g., "off by 5 units on average").
      
    - RMSE (Root Mean Squared Error): sqrt(mean((actual - predicted)²)).
      Penalizes large errors more than MAE. Same units as the target.
      
    - MAPE (Mean Absolute Percentage Error): mean(|actual - predicted| / actual) × 100.
      Scale-independent percentage error. Useful for comparing across products.
      CAVEAT: Undefined when actual = 0; we exclude zero-demand days.
    
    Args:
        predictions_df: DataFrame with actual and predicted columns.
        label_col: Name of actual value column.
        prediction_col: Name of predicted value column.
    
    Returns:
        dict: Metric name → value.
    """
    # MAE
    mae_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="mae"
    )
    mae = mae_evaluator.evaluate(predictions_df)
    
    # RMSE
    rmse_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="rmse"
    )
    rmse = rmse_evaluator.evaluate(predictions_df)
    
    # MAPE (manual calculation — exclude zero demand)
    non_zero = predictions_df.filter(col(label_col) > 0)
    if non_zero.count() > 0:
        mape_df = non_zero.withColumn(
            "ape",
            spark_abs(col(label_col) - col(prediction_col)) / col(label_col) * 100
        )
        mape = mape_df.agg(avg("ape")).collect()[0][0]
        mape = round(mape, 2)
    else:
        mape = None  # Cannot compute MAPE with no non-zero actuals
    
    # R² (coefficient of determination)
    r2_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="r2"
    )
    r2 = r2_evaluator.evaluate(predictions_df)
    
    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": mape,
        "R2": round(r2, 4),
    }
    
    return metrics


def moving_average_forecast(train_df, test_df, config=None):
    """
    Baseline 1: Moving Average forecast.
    
    Predicts future demand as the average of the last N days of demand
    for each product. This is the simplest possible forecast.
    
    Uses PySpark window functions for the moving average calculation.
    
    Formula: Predicted_Demand = AVG(demand over last window_size days)
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        config: Configuration dictionary.
    
    Returns:
        tuple: (predictions_df, metrics_dict)
    """
    if config is None:
        config = load_config()
    
    window_size = config["forecasting"]["models"]["moving_average"]["window_size"]
    
    print(f"\n[Baseline] Moving Average (window={window_size} days)...")
    
    # Calculate the moving average from training data (last window_size entries per product)
    # For each product, get the average demand from the training set
    product_avg = (
        train_df.groupBy("Product_ID")
        .agg(
            spark_round(avg("Daily_Demand"), 2).alias("prediction")
        )
    )
    
    # Join predictions to test set
    predictions = (
        test_df
        .join(product_avg, on="Product_ID", how="left")
        .withColumn("prediction", coalesce(col("prediction"), lit(0)))
    )
    
    # Evaluate
    metrics = evaluate_model(predictions)
    
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  R²:   {metrics['R2']}")
    
    return predictions, metrics


def linear_regression_forecast(train_df, test_df, config=None):
    """
    Baseline 2: Linear Regression using Spark MLlib.
    
    A simple linear model that learns a weighted combination of features
    to predict demand. Serves as the ML baseline.
    
    Uses Spark MLlib's LinearRegression estimator:
    - Input: 'features' vector column (from VectorAssembler)
    - Output: 'prediction' column
    
    Args:
        train_df: Training DataFrame with 'features' and 'Daily_Demand' columns.
        test_df: Test DataFrame.
        config: Configuration dictionary.
    
    Returns:
        tuple: (predictions_df, metrics_dict, trained_model)
    """
    if config is None:
        config = load_config()
    
    lr_config = config["forecasting"]["models"]["linear_regression"]
    
    print(f"\n[Baseline] Linear Regression (Spark MLlib)...")
    print(f"  maxIter={lr_config['max_iter']}, regParam={lr_config['reg_param']}")
    
    # Create and train model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="Daily_Demand",
        predictionCol="prediction",
        maxIter=lr_config["max_iter"],
        regParam=lr_config["reg_param"],
        elasticNetParam=lr_config["elastic_net"],
    )
    
    lr_model = lr.fit(train_df)
    
    # Print model summary
    print(f"  Coefficients: {lr_model.coefficients[:5]}... (showing first 5)")
    print(f"  Intercept: {lr_model.intercept:.4f}")
    
    # Training summary
    training_summary = lr_model.summary
    print(f"  Training RMSE: {training_summary.rootMeanSquaredError:.4f}")
    print(f"  Training R²:   {training_summary.r2:.4f}")
    
    # Predict on test set
    predictions = lr_model.transform(test_df)
    
    # Ensure non-negative predictions (demand can't be negative)
    predictions = predictions.withColumn(
        "prediction",
        when(col("prediction") < 0, lit(0)).otherwise(col("prediction"))
    )
    
    # Evaluate
    metrics = evaluate_model(predictions)
    
    print(f"\n  Test Metrics:")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  R²:   {metrics['R2']}")
    
    return predictions, metrics, lr_model


if __name__ == "__main__":
    from src.module3_preprocessing.spark_preprocessing import run_preprocessing
    from src.module5_forecasting.feature_engineering import prepare_ml_dataset
    
    spark, cleaned_df = run_preprocessing()
    train_df, test_df, features, encoder = prepare_ml_dataset(cleaned_df)
    
    # Run baselines
    ma_preds, ma_metrics = moving_average_forecast(train_df, test_df)
    lr_preds, lr_metrics, lr_model = linear_regression_forecast(train_df, test_df)
    
    print("\n" + "=" * 50)
    print("BASELINE COMPARISON")
    print("=" * 50)
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10}")
    print("-" * 60)
    print(f"{'Moving Average':<20} {ma_metrics['MAE']:>10} {ma_metrics['RMSE']:>10} {str(ma_metrics['MAPE']):>10} {ma_metrics['R2']:>10}")
    print(f"{'Linear Regression':<20} {lr_metrics['MAE']:>10} {lr_metrics['RMSE']:>10} {str(lr_metrics['MAPE']):>10} {lr_metrics['R2']:>10}")
    
    spark.stop()
