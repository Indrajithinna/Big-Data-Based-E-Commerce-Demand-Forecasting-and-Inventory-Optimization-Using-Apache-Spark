# =============================================================================
# Module 5: Demand Forecasting
# Script: advanced_model.py
# =============================================================================
# PURPOSE:
#   Implements advanced demand forecasting models using Spark MLlib:
#   1. Random Forest Regression — ensemble of decision trees
#   2. Gradient-Boosted Trees (GBT) — sequential boosting of weak learners
#
#   These models should outperform the baselines from baseline_model.py.
#   Performance is compared empirically — no claims are made without evidence.
#
# BIS701 RELEVANCE:
#   - Spark MLlib ensemble methods → distributed ML training
#   - RandomForestRegressor and GBTRegressor → Spark's tree-based models
#   - Feature importance analysis → model interpretability
#   - Model persistence (save/load) → deployment-ready models
#
# INPUT:  Feature-engineered train/test DataFrames
# OUTPUT: Best model, predictions, comparison metrics
# =============================================================================

import os
import sys
import json

from pyspark.sql.functions import col, when, lit, round as spark_round
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config
from src.module5_forecasting.baseline_model import evaluate_model


def random_forest_forecast(train_df, test_df, config=None):
    """
    Random Forest Regression using Spark MLlib.
    
    Random Forest builds multiple decision trees on random subsets of
    data and features, then averages their predictions. This reduces
    overfitting compared to a single decision tree.
    
    Key parameters:
    - numTrees: Number of trees in the forest (more = better but slower)
    - maxDepth: Maximum depth of each tree (controls complexity)
    
    Args:
        train_df: Training DataFrame with 'features' column.
        test_df: Test DataFrame.
        config: Configuration dictionary.
    
    Returns:
        tuple: (predictions_df, metrics_dict, trained_model, feature_importances)
    """
    if config is None:
        config = load_config()
    
    rf_config = config["forecasting"]["models"]["random_forest"]
    
    print(f"\n[Advanced] Random Forest Regression (Spark MLlib)...")
    print(f"  numTrees={rf_config['num_trees']}, maxDepth={rf_config['max_depth']}")
    
    # Create Random Forest model
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="Daily_Demand",
        predictionCol="prediction",
        numTrees=rf_config["num_trees"],
        maxDepth=rf_config["max_depth"],
        seed=rf_config["seed"],
        subsamplingRate=0.8,
    )
    
    # Train the model
    print("  Training Random Forest... (this may take a few minutes)")
    rf_model = rf.fit(train_df)
    
    # Feature importances
    importances = rf_model.featureImportances.toArray()
    print(f"  Top feature importances:")
    # Get feature names from the assembler
    feature_names = [
        "Month", "Day_of_Month", "Day_of_Week", "Quarter", "Week_of_Year",
        "Lag_1_Demand", "Lag_7_Demand", "Lag_14_Demand", "Lag_30_Demand",
        "Rolling_Mean_7", "Rolling_Mean_14", "Rolling_Mean_30",
        "Avg_Unit_Price", "Avg_Discount",
        "Category_Index", "Region_Index", "Product_Index",
    ]
    importance_pairs = sorted(
        zip(feature_names[:len(importances)], importances),
        key=lambda x: x[1],
        reverse=True
    )
    for name, imp in importance_pairs[:7]:
        print(f"    {name}: {imp:.4f}")
    
    # Predict on test set
    predictions = rf_model.transform(test_df)
    predictions = predictions.withColumn(
        "prediction",
        when(col("prediction") < 0, lit(0)).otherwise(spark_round(col("prediction"), 2))
    )
    
    # Evaluate
    metrics = evaluate_model(predictions)
    
    print(f"\n  Test Metrics:")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  R²:   {metrics['R2']}")
    
    return predictions, metrics, rf_model, importance_pairs


def gbt_forecast(train_df, test_df, config=None):
    """
    Gradient-Boosted Trees (GBT) Regression using Spark MLlib.
    
    GBT builds trees sequentially, where each new tree corrects the
    errors of the previous ensemble. Generally produces better results
    than Random Forest but is more prone to overfitting.
    
    Key parameters:
    - maxIter: Number of boosting iterations (trees)
    - maxDepth: Maximum depth of each tree
    - stepSize: Learning rate (smaller = slower learning, less overfitting)
    
    Args:
        train_df: Training DataFrame with 'features' column.
        test_df: Test DataFrame.
        config: Configuration dictionary.
    
    Returns:
        tuple: (predictions_df, metrics_dict, trained_model, feature_importances)
    """
    if config is None:
        config = load_config()
    
    gbt_config = config["forecasting"]["models"]["gradient_boosted_trees"]
    
    print(f"\n[Advanced] Gradient-Boosted Trees (Spark MLlib)...")
    print(f"  maxIter={gbt_config['max_iter']}, maxDepth={gbt_config['max_depth']}, "
          f"stepSize={gbt_config['step_size']}")
    
    # Create GBT model
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="Daily_Demand",
        predictionCol="prediction",
        maxIter=gbt_config["max_iter"],
        maxDepth=gbt_config["max_depth"],
        stepSize=gbt_config["step_size"],
        seed=gbt_config["seed"],
        subsamplingRate=0.8,
    )
    
    # Train the model
    print("  Training GBT... (this may take several minutes)")
    gbt_model = gbt.fit(train_df)
    
    # Feature importances
    importances = gbt_model.featureImportances.toArray()
    feature_names = [
        "Month", "Day_of_Month", "Day_of_Week", "Quarter", "Week_of_Year",
        "Lag_1_Demand", "Lag_7_Demand", "Lag_14_Demand", "Lag_30_Demand",
        "Rolling_Mean_7", "Rolling_Mean_14", "Rolling_Mean_30",
        "Avg_Unit_Price", "Avg_Discount",
        "Category_Index", "Region_Index", "Product_Index",
    ]
    importance_pairs = sorted(
        zip(feature_names[:len(importances)], importances),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"  Top feature importances:")
    for name, imp in importance_pairs[:7]:
        print(f"    {name}: {imp:.4f}")
    
    # Predict on test set
    predictions = gbt_model.transform(test_df)
    predictions = predictions.withColumn(
        "prediction",
        when(col("prediction") < 0, lit(0)).otherwise(spark_round(col("prediction"), 2))
    )
    
    # Evaluate
    metrics = evaluate_model(predictions)
    
    print(f"\n  Test Metrics:")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  R²:   {metrics['R2']}")
    
    return predictions, metrics, gbt_model, importance_pairs


def compare_all_models(all_metrics, output_dir):
    """
    Compare all models side-by-side and select the best one.
    
    Selection criterion: lowest RMSE on the test set.
    (RMSE is used because it penalizes large errors, which are
    particularly costly in inventory planning.)
    
    Args:
        all_metrics: Dict of {model_name: metrics_dict}.
        output_dir: Directory to save comparison results.
    
    Returns:
        str: Name of the best model.
    """
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (on test set)")
    print("=" * 70)
    print(f"  {'Model':<25} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'R²':>10}")
    print("  " + "-" * 65)
    
    best_model = None
    best_rmse = float("inf")
    
    for name, metrics in all_metrics.items():
        mape_str = f"{metrics['MAPE']}" if metrics['MAPE'] is not None else "N/A"
        print(f"  {name:<25} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f} "
              f"{mape_str:>10} {metrics['R2']:>10.4f}")
        
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model = name
    
    print("  " + "-" * 65)
    print(f"  Best Model (by RMSE): {best_model}")
    print("=" * 70)
    
    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "models": all_metrics,
            "best_model": best_model,
            "selection_criterion": "lowest RMSE on test set"
        }, f, indent=2, default=str)
    
    print(f"  Comparison saved to: {comparison_path}")
    
    return best_model


def save_best_model(model, model_name, config):
    """
    Save the best model to disk for later use.
    
    Spark MLlib models can be saved and loaded for deployment,
    demonstrating the full ML lifecycle.
    
    Args:
        model: Trained Spark ML model.
        model_name: Name of the model.
        config: Configuration dictionary.
    """
    models_dir = config["paths"].get("models_dir",
                                      os.path.join(os.path.dirname(config["paths"]["results_dir"]), "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_name.replace(" ", "_").lower())
    model.write().overwrite().save(model_path)
    print(f"\n[Model] Saved {model_name} to: {model_path}")


def save_predictions(predictions_df, output_dir):
    """
    Save forecast predictions to CSV for the dashboard.
    
    Args:
        predictions_df: DataFrame with predictions.
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select relevant columns for dashboard
    result = (
        predictions_df
        .select(
            "Date", "Product_ID", "Product_Name", "Product_Category",
            "Region", "Daily_Demand", "prediction"
        )
        .withColumnRenamed("Daily_Demand", "Actual_Demand")
        .withColumnRenamed("prediction", "Predicted_Demand")
    )
    
    path = os.path.join(output_dir, "demand_predictions.csv")
    result.toPandas().to_csv(path, index=False)
    print(f"[Save] Predictions saved to: {path}")
    
    return path


def run_forecasting(spark, df, config=None):
    """
    Execute the full demand forecasting pipeline.
    
    Pipeline:
    1. Feature engineering
    2. Train baseline models (Moving Average, Linear Regression)
    3. Train advanced models (Random Forest, GBT)
    4. Compare all models
    5. Save best model and predictions
    
    Args:
        spark: SparkSession instance.
        df: Cleaned Spark DataFrame.
        config: Configuration dictionary.
    
    Returns:
        tuple: (best_model_name, best_predictions, all_metrics, best_model)
    """
    if config is None:
        config = load_config()
    
    from src.module5_forecasting.feature_engineering import prepare_ml_dataset
    from src.module5_forecasting.baseline_model import (
        moving_average_forecast, linear_regression_forecast
    )
    
    output_dir = config["paths"]["forecast_output_dir"]
    
    print("\n" + "=" * 60)
    print("Module 5: Demand Forecasting Pipeline")
    print("=" * 60)
    
    # Step 1: Feature engineering
    train_df, test_df, feature_cols, encoder = prepare_ml_dataset(df, config)
    
    # Cache for performance (multiple model trainings)
    train_df = train_df.cache()
    test_df = test_df.cache()
    
    # Step 2: Train all models
    all_metrics = {}
    all_predictions = {}
    all_models = {}
    
    # Baseline 1: Moving Average
    ma_preds, ma_metrics = moving_average_forecast(train_df, test_df, config)
    all_metrics["Moving Average"] = ma_metrics
    all_predictions["Moving Average"] = ma_preds
    
    # Baseline 2: Linear Regression
    lr_preds, lr_metrics, lr_model = linear_regression_forecast(train_df, test_df, config)
    all_metrics["Linear Regression"] = lr_metrics
    all_predictions["Linear Regression"] = lr_preds
    all_models["Linear Regression"] = lr_model
    
    # Advanced 1: Random Forest
    rf_preds, rf_metrics, rf_model, rf_importance = random_forest_forecast(
        train_df, test_df, config
    )
    all_metrics["Random Forest"] = rf_metrics
    all_predictions["Random Forest"] = rf_preds
    all_models["Random Forest"] = rf_model
    
    # Advanced 2: GBT
    gbt_preds, gbt_metrics, gbt_model, gbt_importance = gbt_forecast(
        train_df, test_df, config
    )
    all_metrics["Gradient-Boosted Trees"] = gbt_metrics
    all_predictions["Gradient-Boosted Trees"] = gbt_preds
    all_models["Gradient-Boosted Trees"] = gbt_model
    
    # Step 3: Compare models
    best_model_name = compare_all_models(all_metrics, output_dir)
    
    # Step 4: Save best model and its predictions
    best_predictions = all_predictions[best_model_name]
    if best_model_name in all_models:
        save_best_model(all_models[best_model_name], best_model_name, config)
    
    # Save predictions
    save_predictions(best_predictions, output_dir)
    
    # Save feature importances from tree model
    tree_name = best_model_name if best_model_name in ["Random Forest", "Gradient-Boosted Trees"] else "Random Forest"
    imp = rf_importance if tree_name == "Random Forest" else gbt_importance
    imp_path = os.path.join(output_dir, "feature_importances.json")
    with open(imp_path, "w") as f:
        json.dump([{"feature": n, "importance": round(v, 6)} for n, v in imp], f, indent=2)
    print(f"  Feature importances saved to: {imp_path}")
    
    # Unpersist cached DataFrames
    train_df.unpersist()
    test_df.unpersist()
    
    print(f"\n{'='*60}")
    print(f"Forecasting pipeline complete! Best model: {best_model_name}")
    print(f"{'='*60}")
    
    return best_model_name, best_predictions, all_metrics, all_models.get(best_model_name)


if __name__ == "__main__":
    from src.module3_preprocessing.spark_preprocessing import run_preprocessing
    
    spark, cleaned_df = run_preprocessing()
    best_name, best_preds, metrics, model = run_forecasting(spark, cleaned_df)
    
    print(f"\nBest model: {best_name}")
    print("\nReady for Module 6: python -m src.module6_inventory.inventory_optimizer")
    spark.stop()
