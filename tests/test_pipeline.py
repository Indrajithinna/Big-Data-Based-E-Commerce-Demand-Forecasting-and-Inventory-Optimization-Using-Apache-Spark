"""
Test Suite for Big Data-Based E-Commerce Demand Forecasting and Inventory Optimization.
Validates all pipeline stages, output artifacts, metrics, and data integrity.
"""

import os
import sys
import json
import unittest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.module7_dashboard.app import load_all_data


class TestECommercePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.paths = cls.config["paths"]

    def test_01_configuration(self):
        """Verify configuration file loads and contains all required sections."""
        self.assertIn("paths", self.config)
        self.assertIn("spark", self.config)
        self.assertIn("preprocessing", self.config)
        self.assertIn("analytics", self.config)
        self.assertIn("forecasting", self.config)
        self.assertIn("inventory", self.config)
        self.assertIn("dashboard", self.config)

    def test_02_preprocessed_data_exists(self):
        """Verify that preprocessed Parquet and CSV files exist and are populated."""
        parquet_path = self.paths["cleaned_dataset_file"]
        csv_path = os.path.join(self.paths["processed_data_dir"], "cleaned_sales.csv")
        self.assertTrue(os.path.exists(parquet_path), "Cleaned parquet dir missing")
        self.assertTrue(os.path.exists(csv_path), "Cleaned sales CSV missing")
        self.assertGreater(os.path.getsize(csv_path), 50_000_000, "Cleaned CSV too small")

    def test_03_analytics_results(self):
        """Verify that Module 4 produced all required analytics outputs."""
        analytics_dir = self.paths["analytics_output_dir"]
        expected_files = [
            "daily_sales.csv", "weekly_sales.csv", "monthly_sales.csv",
            "product_sales.csv", "top_products.csv", "slow_products.csv",
            "category_sales.csv", "region_sales.csv", "seasonal_monthly.csv",
            "seasonal_dow.csv", "customer_summary.csv", "customer_segments.csv",
            "product_growth_rates.csv", "kpis.json"
        ]
        for f in expected_files:
            file_path = os.path.join(analytics_dir, f)
            self.assertTrue(os.path.exists(file_path), f"Missing analytics file: {f}")
            self.assertGreater(os.path.getsize(file_path), 0, f"Empty analytics file: {f}")

        # Check KPI values
        kpi_file = os.path.join(analytics_dir, "kpis.json")
        with open(kpi_file, "r") as f:
            kpis = json.load(f)
        self.assertGreater(kpis.get("Total_Revenue", 0), 10_000_000)
        self.assertGreater(kpis.get("Total_Orders", 0), 30_000)
        self.assertGreater(kpis.get("Total_Units_Sold", 0), 5_000_000)

    def test_04_demand_forecasting_results(self):
        """Verify that Module 5 trained models and produced predictions."""
        forecast_dir = self.paths["forecast_output_dir"]
        comp_file = os.path.join(forecast_dir, "model_comparison.json")
        pred_file = os.path.join(forecast_dir, "demand_predictions.csv")
        imp_file = os.path.join(forecast_dir, "feature_importances.json")

        self.assertTrue(os.path.exists(comp_file), "model_comparison.json missing")
        self.assertTrue(os.path.exists(pred_file), "demand_predictions.csv missing")
        self.assertTrue(os.path.exists(imp_file), "feature_importances.json missing")

        with open(comp_file, "r") as f:
            comp = json.load(f)
        self.assertIn("models", comp)
        self.assertIn("best_model", comp)
        self.assertIn("Linear Regression", comp["models"])
        self.assertIn("Random Forest", comp["models"])
        self.assertIn("Gradient-Boosted Trees", comp["models"])

        # Check saved model directory
        models_dir = self.paths["models_dir"]
        self.assertTrue(os.path.exists(models_dir), "models/ dir missing")
        self.assertGreater(len(os.listdir(models_dir)), 0, "No models saved in models/")

    def test_05_inventory_optimization_results(self):
        """Verify that Module 6 produced inventory recommendations and summary."""
        inventory_dir = self.paths["inventory_output_dir"]
        rec_file = os.path.join(inventory_dir, "inventory_recommendations.csv")
        sum_file = os.path.join(inventory_dir, "inventory_summary.json")

        self.assertTrue(os.path.exists(rec_file), "inventory_recommendations.csv missing")
        self.assertTrue(os.path.exists(sum_file), "inventory_summary.json missing")

        with open(sum_file, "r") as f:
            summary = json.load(f)
        self.assertGreater(summary.get("total_products", 0), 1000)
        self.assertIn("stockout_risk_high", summary)
        self.assertIn("overstock_risk_high", summary)

    def test_06_system_evaluation_report(self):
        """Verify that Module 8 generated the system evaluation report."""
        report_file = os.path.join(self.paths["results_dir"], "system_evaluation.json")
        self.assertTrue(os.path.exists(report_file), "system_evaluation.json missing")

        with open(report_file, "r") as f:
            report = json.load(f)
        self.assertIn("modules", report)
        self.assertIn("bis701_mapping", report)

    def test_07_dashboard_data_loading(self):
        """Verify that Module 7 dashboard loads all 19 data keys cleanly."""
        data = load_all_data(self.config)
        self.assertIn("kpis", data)
        self.assertIn("daily_sales", data)
        self.assertIn("predictions", data)
        self.assertIn("inventory", data)
        self.assertIn("filter_categories", data)
        self.assertGreater(len(data["filter_categories"]), 0)


if __name__ == "__main__":
    unittest.main()
