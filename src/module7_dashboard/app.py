# =============================================================================
# Module 7: Visualization and Dashboard
# Script: app.py — Streamlit Main Application
# =============================================================================
# PURPOSE:
#   Interactive web dashboard for the E-Commerce Demand Forecasting &
#   Inventory Optimization system. Built with Streamlit + Plotly.
#
# PAGES:
#   1. Executive Dashboard — KPIs, alerts, overview
#   2. Sales Analytics — Revenue trends, category/product/region charts
#   3. Demand Forecast — Historical vs predicted, error analysis
#   4. Inventory Analytics — Reorder points, stockout/overstock alerts
#
# BIS701 RELEVANCE:
#   - Business Intelligence visualization
#   - Dashboard design for Big Data analytics results
#   - Web-based interface for data-driven decision making
#
# HOW TO RUN:
#   streamlit run src/module7_dashboard/app.py
# =============================================================================

import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config


# =============================================================================
# Configuration and Data Loading
# =============================================================================

@st.cache_data
def load_all_data(config):
    """Load all result CSV/JSON files for the dashboard."""
    data = {}
    analytics_dir = config["paths"]["analytics_output_dir"]
    forecast_dir = config["paths"]["forecast_output_dir"]
    inventory_dir = config["paths"]["inventory_output_dir"]
    processed_dir = config["paths"]["processed_data_dir"]
    
    # Helper to safely load CSV
    def safe_load_csv(path, name):
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[name] = df
            return df
        else:
            st.warning(f"Data file not found: {path}")
            return pd.DataFrame()
    
    # Helper to safely load JSON
    def safe_load_json(path, name):
        if os.path.exists(path):
            with open(path, "r") as f:
                data[name] = json.load(f)
            return data[name]
        return {}
    
    # Load analytics data
    safe_load_csv(os.path.join(analytics_dir, "daily_sales.csv"), "daily_sales")
    safe_load_csv(os.path.join(analytics_dir, "weekly_sales.csv"), "weekly_sales")
    safe_load_csv(os.path.join(analytics_dir, "monthly_sales.csv"), "monthly_sales")
    safe_load_csv(os.path.join(analytics_dir, "product_sales.csv"), "product_sales")
    safe_load_csv(os.path.join(analytics_dir, "top_products.csv"), "top_products")
    safe_load_csv(os.path.join(analytics_dir, "slow_products.csv"), "slow_products")
    safe_load_csv(os.path.join(analytics_dir, "category_sales.csv"), "category_sales")
    safe_load_csv(os.path.join(analytics_dir, "region_sales.csv"), "region_sales")
    safe_load_csv(os.path.join(analytics_dir, "seasonal_monthly.csv"), "seasonal_monthly")
    safe_load_csv(os.path.join(analytics_dir, "seasonal_dow.csv"), "seasonal_dow")
    safe_load_csv(os.path.join(analytics_dir, "customer_segments.csv"), "customer_segments")
    
    # Load KPIs
    safe_load_json(os.path.join(analytics_dir, "kpis.json"), "kpis")
    
    # Load forecast data
    safe_load_csv(os.path.join(forecast_dir, "demand_predictions.csv"), "predictions")
    safe_load_json(os.path.join(forecast_dir, "model_comparison.json"), "model_comparison")
    safe_load_json(os.path.join(forecast_dir, "feature_importances.json"), "feature_importances")
    
    # Load inventory data
    safe_load_csv(os.path.join(inventory_dir, "inventory_recommendations.csv"), "inventory")
    safe_load_json(os.path.join(inventory_dir, "inventory_summary.json"), "inventory_summary")
    
    # Load cleaned dataset for filters
    cleaned_csv = os.path.join(processed_dir, "cleaned_sales.csv")
    if os.path.exists(cleaned_csv):
        # Load only unique filter values (not full dataset)
        cleaned = pd.read_csv(cleaned_csv, usecols=["Product_Category", "Region", "Product_ID", "Product_Name"],
                              nrows=500000)
        data["filter_categories"] = sorted(cleaned["Product_Category"].dropna().unique().tolist())
        data["filter_regions"] = sorted(cleaned["Region"].dropna().unique().tolist())
        data["filter_products"] = sorted(cleaned["Product_ID"].dropna().unique().tolist()[:100])
    else:
        data["filter_categories"] = []
        data["filter_regions"] = []
        data["filter_products"] = []
    
    return data


def format_number(num, prefix="", suffix=""):
    """Format large numbers with commas and optional prefix/suffix."""
    if num is None:
        return "N/A"
    if abs(num) >= 1_000_000:
        return f"{prefix}{num/1_000_000:,.1f}M{suffix}"
    elif abs(num) >= 1_000:
        return f"{prefix}{num/1_000:,.1f}K{suffix}"
    return f"{prefix}{num:,.0f}{suffix}"


# =============================================================================
# Page: Executive Dashboard
# =============================================================================

def render_executive_dashboard(data):
    """Render the Executive Dashboard page with KPIs and alerts."""
    
    st.header("📊 Executive Dashboard")
    
    kpis = data.get("kpis", {})
    inv_summary = data.get("inventory_summary", {})
    
    if not kpis:
        st.warning("KPI data not available. Run the pipeline first.")
        return
    
    # --- KPI Cards (Row 1) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Revenue", format_number(kpis.get("Total_Revenue"), prefix="$"))
    with col2:
        st.metric("📦 Total Orders", format_number(kpis.get("Total_Orders")))
    with col3:
        st.metric("📋 Total Units Sold", format_number(kpis.get("Total_Units_Sold")))
    with col4:
        st.metric("🏪 Total Products", format_number(kpis.get("Total_Products")))
    
    # --- KPI Cards (Row 2) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💳 Avg Order Value", format_number(kpis.get("Average_Order_Value"), prefix="$"))
    with col2:
        growth = kpis.get("Revenue_Growth_Rate_Pct", 0)
        st.metric("📈 Revenue Growth", f"{growth:.1f}%",
                  delta=f"{growth:.1f}%")
    with col3:
        st.metric("⚠️ Stockout Risk",
                  f"{inv_summary.get('stockout_risk_high', 'N/A')} products")
    with col4:
        st.metric("📦 Overstock Alert",
                  f"{inv_summary.get('overstock_risk_high', 'N/A')} products")
    
    st.divider()
    
    # --- Revenue Trend ---
    monthly = data.get("monthly_sales", pd.DataFrame())
    if not monthly.empty:
        st.subheader("📈 Revenue Trend (Monthly)")
        fig = px.line(
            monthly, x="Year_Month", y="Total_Revenue",
            title="Monthly Revenue Trend",
            labels={"Year_Month": "Month", "Total_Revenue": "Revenue ($)"},
            template="plotly_dark"
        )
        fig.update_traces(line_color="#00d4ff", line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Inventory Alert Table ---
    inventory = data.get("inventory", pd.DataFrame())
    if not inventory.empty:
        st.subheader("🚨 Urgent Reorder Alerts")
        urgent = inventory[inventory["Recommendation"] == "URGENT REORDER"]
        if not urgent.empty:
            st.dataframe(
                urgent[["Product_ID", "Product_Name", "Current_Inventory",
                        "Reorder_Point", "Reorder_Quantity", "Stockout_Risk"]].head(10),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("✅ No urgent reorders needed.")


# =============================================================================
# Page: Sales Analytics
# =============================================================================

def render_sales_analytics(data, filters):
    """Render the Sales Analytics page."""
    
    st.header("📊 Sales Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Revenue Trends", "🏷️ Category Analysis",
        "🛍️ Product Analysis", "🌍 Regional Analysis"
    ])
    
    with tab1:
        render_revenue_trends(data)
    
    with tab2:
        render_category_analysis(data)
    
    with tab3:
        render_product_analysis(data)
    
    with tab4:
        render_regional_analysis(data)


def render_revenue_trends(data):
    """Revenue trend charts."""
    daily = data.get("daily_sales", pd.DataFrame())
    monthly = data.get("monthly_sales", pd.DataFrame())
    
    if not monthly.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                monthly, x="Year_Month", y="Total_Revenue",
                title="Monthly Revenue",
                labels={"Year_Month": "Month", "Total_Revenue": "Revenue ($)"},
                template="plotly_dark", color_discrete_sequence=["#00d4ff"]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                monthly, x="Year_Month", y="Total_Quantity",
                title="Monthly Units Sold",
                labels={"Year_Month": "Month", "Total_Quantity": "Units"},
                template="plotly_dark"
            )
            fig.update_traces(line_color="#ff6b6b", line_width=2)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    seasonal = data.get("seasonal_monthly", pd.DataFrame())
    dow = data.get("seasonal_dow", pd.DataFrame())
    
    if not seasonal.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                seasonal, x="Month", y="Avg_Monthly_Revenue",
                title="Average Monthly Revenue (Seasonality)",
                template="plotly_dark", color_discrete_sequence=["#ffd93d"]
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        if not dow.empty:
            with col2:
                fig = px.bar(
                    dow, x="Day_Name", y="Total_Revenue",
                    title="Revenue by Day of Week",
                    template="plotly_dark", color_discrete_sequence=["#6bcb77"]
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)


def render_category_analysis(data):
    """Category-level analytics."""
    category = data.get("category_sales", pd.DataFrame())
    
    if not category.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                category, values="Total_Revenue", names="Product_Category",
                title="Revenue by Category",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                category, x="Product_Category", y="Total_Quantity",
                title="Units Sold by Category",
                template="plotly_dark",
                color="Product_Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400, showlegend=False,
                            xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Category Details")
        st.dataframe(category, use_container_width=True, hide_index=True)


def render_product_analysis(data):
    """Product-level analytics."""
    top = data.get("top_products", pd.DataFrame())
    slow = data.get("slow_products", pd.DataFrame())
    
    if not top.empty:
        st.subheader("🏆 Top Selling Products")
        fig = px.bar(
            top.head(15), x="Total_Revenue", y="Product_Name",
            orientation="h", title="Top 15 Products by Revenue",
            template="plotly_dark", color_discrete_sequence=["#00d4ff"]
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    
    if not slow.empty:
        st.subheader("🐢 Slow-Moving Products")
        st.dataframe(
            slow[["Product_ID", "Product_Name", "Total_Quantity", "Total_Revenue",
                  "Num_Orders"]].head(15),
            use_container_width=True, hide_index=True
        )


def render_regional_analysis(data):
    """Regional analytics."""
    region = data.get("region_sales", pd.DataFrame())
    
    if not region.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                region.head(15), x="Region", y="Total_Revenue",
                title="Revenue by Region (Top 15)",
                template="plotly_dark", color_discrete_sequence=["#ff6b6b"]
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                region.head(10), values="Num_Orders", names="Region",
                title="Orders by Region (Top 10)",
                template="plotly_dark"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Regional Details")
        st.dataframe(region, use_container_width=True, hide_index=True)


# =============================================================================
# Page: Demand Forecast
# =============================================================================

def render_demand_forecast(data, filters):
    """Render the Demand Forecast page."""
    
    st.header("🔮 Demand Forecast")
    
    predictions = data.get("predictions", pd.DataFrame())
    comparison = data.get("model_comparison", {})
    importances = data.get("feature_importances", [])
    
    # --- Model Comparison ---
    if comparison:
        st.subheader("📊 Model Comparison")
        models = comparison.get("models", {})
        best = comparison.get("best_model", "")
        
        if models:
            model_df = pd.DataFrame([
                {"Model": name, **metrics}
                for name, metrics in models.items()
            ])
            
            st.info(f"**Best Model:** {best} (selected by lowest RMSE on test set)")
            st.dataframe(model_df, use_container_width=True, hide_index=True)
            
            # Visualize comparison
            fig = go.Figure()
            for metric in ["MAE", "RMSE"]:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=model_df["Model"],
                    y=model_df[metric],
                ))
            fig.update_layout(
                title="Model Performance Comparison",
                barmode="group", template="plotly_dark", height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Feature Importances ---
    if importances:
        st.subheader("🎯 Feature Importances")
        imp_df = pd.DataFrame(importances)
        fig = px.bar(
            imp_df.head(15), x="importance", y="feature",
            orientation="h", title="Top Features",
            template="plotly_dark", color_discrete_sequence=["#ffd93d"]
        )
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Actual vs Predicted ---
    if not predictions.empty:
        st.subheader("📈 Actual vs Predicted Demand")
        
        # Filter by product if selected
        if "Date" in predictions.columns:
            predictions["Date"] = pd.to_datetime(predictions["Date"])
        
        # Aggregate by date
        if "Date" in predictions.columns:
            daily_pred = predictions.groupby("Date").agg({
                "Actual_Demand": "sum",
                "Predicted_Demand": "sum"
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_pred["Date"], y=daily_pred["Actual_Demand"],
                name="Actual", line=dict(color="#00d4ff", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=daily_pred["Date"], y=daily_pred["Predicted_Demand"],
                name="Predicted", line=dict(color="#ff6b6b", width=2, dash="dash")
            ))
            fig.update_layout(
                title="Aggregate Daily Demand: Actual vs Predicted",
                template="plotly_dark", height=450,
                xaxis_title="Date", yaxis_title="Total Demand (units)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast error distribution
        if "Actual_Demand" in predictions.columns and "Predicted_Demand" in predictions.columns:
            predictions["Error"] = predictions["Actual_Demand"] - predictions["Predicted_Demand"]
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    predictions, x="Error", nbins=50,
                    title="Forecast Error Distribution",
                    template="plotly_dark", color_discrete_sequence=["#6bcb77"]
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top products by forecast accuracy
                product_errors = predictions.groupby("Product_ID").agg({
                    "Actual_Demand": "sum",
                    "Predicted_Demand": "sum"
                }).reset_index()
                product_errors["MAPE"] = (
                    abs(product_errors["Actual_Demand"] - product_errors["Predicted_Demand"])
                    / product_errors["Actual_Demand"].clip(lower=1) * 100
                ).round(2)
                product_errors = product_errors.sort_values("MAPE").head(10)
                
                fig = px.bar(
                    product_errors, x="Product_ID", y="MAPE",
                    title="Top 10 Most Accurate Product Forecasts (MAPE %)",
                    template="plotly_dark", color_discrete_sequence=["#ffd93d"]
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page: Inventory Analytics
# =============================================================================

def render_inventory_analytics(data, filters):
    """Render the Inventory Analytics page."""
    
    st.header("📦 Inventory Analytics")
    
    inventory = data.get("inventory", pd.DataFrame())
    inv_summary = data.get("inventory_summary", {})
    
    if inventory.empty:
        st.warning("Inventory data not available. Run the pipeline first.")
        return
    
    # --- Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 Stockout Risk (High)",
                  inv_summary.get("stockout_risk_high", 0))
    with col2:
        st.metric("🟡 Stockout Risk (Medium)",
                  inv_summary.get("stockout_risk_medium", 0))
    with col3:
        st.metric("📦 Overstock (High)",
                  inv_summary.get("overstock_risk_high", 0))
    with col4:
        st.metric("✅ Optimal",
                  inv_summary.get("optimal", 0))
    
    st.divider()
    
    # --- Risk Distribution ---
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = inventory["Stockout_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.pie(
            risk_counts, values="Count", names="Risk Level",
            title="Stockout Risk Distribution",
            color="Risk Level",
            color_discrete_map={"HIGH": "#ff4444", "MEDIUM": "#ffbb33", "LOW": "#00C851"},
            template="plotly_dark"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        rec_counts = inventory["Recommendation"].value_counts().reset_index()
        rec_counts.columns = ["Recommendation", "Count"]
        fig = px.bar(
            rec_counts, x="Recommendation", y="Count",
            title="Recommendations Breakdown",
            template="plotly_dark",
            color="Recommendation",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Inventory vs Reorder Point ---
    st.subheader("📊 Inventory Level vs Reorder Point")
    
    # Show top 20 products at highest stockout risk
    at_risk = inventory[inventory["Stockout_Risk"] == "HIGH"].head(20)
    if not at_risk.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current Inventory",
            x=at_risk["Product_ID"],
            y=at_risk["Current_Inventory"],
            marker_color="#ff6b6b"
        ))
        fig.add_trace(go.Bar(
            name="Reorder Point",
            x=at_risk["Product_ID"],
            y=at_risk["Reorder_Point"],
            marker_color="#ffd93d"
        ))
        fig.add_trace(go.Bar(
            name="Safety Stock",
            x=at_risk["Product_ID"],
            y=at_risk["Safety_Stock"],
            marker_color="#6bcb77"
        ))
        fig.update_layout(
            title="High-Risk Products: Inventory vs Reorder Point",
            barmode="group", template="plotly_dark", height=450,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Detailed Table ---
    st.subheader("📋 Inventory Recommendations")
    
    # Add filter for recommendation type
    rec_filter = st.multiselect(
        "Filter by Recommendation",
        inventory["Recommendation"].unique().tolist(),
        default=["URGENT REORDER", "REORDER SOON"]
    )
    
    filtered = inventory[inventory["Recommendation"].isin(rec_filter)] if rec_filter else inventory
    
    st.dataframe(
        filtered[["Product_ID", "Product_Name", "Product_Category", "Supplier",
                  "Current_Inventory", "Safety_Stock", "Reorder_Point",
                  "Predicted_Next_Period_Demand", "Reorder_Quantity",
                  "Stockout_Risk", "Overstock_Risk", "Recommendation"]],
        use_container_width=True, hide_index=True
    )
    
    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Recommendations CSV",
        csv, "inventory_recommendations.csv", "text/csv"
    )
    
    # --- Assumptions ---
    st.divider()
    with st.expander("⚙️ Parameters & Assumptions"):
        params = inv_summary.get("parameters", {})
        st.write(f"- **Lead Time:** {params.get('lead_time_days', 7)} days")
        st.write(f"- **Service Level:** {params.get('service_level', 0.95)*100:.0f}%")
        st.write(f"- **Z-score:** {params.get('z_score', 1.65)}")
        st.write(f"- **Review Period:** {params.get('review_period_days', 30)} days")
        
        st.warning("⚠️ Inventory levels are **SIMULATED** for this academic project. "
                   "In a real system, inventory data would come from a warehouse management system.")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="E-Commerce Demand Forecasting & Inventory Optimization",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme styling
    st.markdown("""
    <style>
        .stMetric {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #0f3460;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 5px;
        }
        h1 {
            color: #00d4ff !important;
        }
        h2 {
            color: #e0e0e0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("Configuration file not found. Ensure config/config.yaml exists.")
        return
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("🔧 Controls")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📌 Navigation",
            ["🏠 Executive Dashboard", "📊 Sales Analytics",
             "🔮 Demand Forecast", "📦 Inventory Analytics"],
            index=0
        )
        
        st.markdown("---")
        
        # Filters
        st.subheader("🔍 Filters")
        
        # Load data
        data = load_all_data(config)
        
        selected_categories = st.multiselect(
            "Product Category",
            data.get("filter_categories", []),
            default=[]
        )
        
        selected_regions = st.multiselect(
            "Region",
            data.get("filter_regions", []),
            default=[]
        )
        
        st.markdown("---")
        st.caption("BE VTU — BIS701 Big Data Analytics Project")
        st.caption("Apache Spark | PySpark | MongoDB | Streamlit")
    
    filters = {
        "categories": selected_categories,
        "regions": selected_regions,
    }
    
    # --- Main Title ---
    st.title("📊 E-Commerce Demand Forecasting & Inventory Optimization")
    st.caption("Big Data Analytics Dashboard | Powered by Apache Spark & PySpark")
    
    # --- Render Selected Page ---
    if page == "🏠 Executive Dashboard":
        render_executive_dashboard(data)
    elif page == "📊 Sales Analytics":
        render_sales_analytics(data, filters)
    elif page == "🔮 Demand Forecast":
        render_demand_forecast(data, filters)
    elif page == "📦 Inventory Analytics":
        render_inventory_analytics(data, filters)


if __name__ == "__main__":
    main()
