import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set up page
st.set_page_config(page_title="📦 Delivery Visibility Dashboard", layout="wide")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("extended_sample_orders_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Status"] = df["Status"].str.strip().str.title()  # Normalize casing
    df["Product"] = df["Product"].str.strip().str.title()
    return df

data = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Orders")
products = st.sidebar.multiselect("Select Product(s)", options=data["Product"].unique(), default=data["Product"].unique())
statuses = st.sidebar.multiselect("Select Status", options=data["Status"].unique(), default=data["Status"].unique())
date_range = st.sidebar.date_input("Select Date Range", [data["Date"].min(), data["Date"].max()])

# Apply filters
data["OnlyDate"] = data["Date"].dt.date  # Available for all filtered views
filtered_data = data[
    (data["Product"].isin(products)) &
    (data["Status"].isin(statuses)) &
    (data["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# Title
st.title("📦 Delivery Visibility Dashboard")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", len(filtered_data))
col2.metric("Delivered", (filtered_data["Status"] == "Delivered").sum())
col3.metric("Delayed", (filtered_data["Status"] == "Delayed").sum())
col4.metric("Pending", (filtered_data["Status"] == "Pending").sum())

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecasting", "🚨 Anomalies", "📋 Raw Data"])

with tab1:
    # Bar Chart: Order Count by Product
    st.subheader("📊 Order Count by Product")
    product_counts = filtered_data["Product"].value_counts().reset_index()
    product_counts.columns = ["Product", "Order Count"]
    st.plotly_chart(px.bar(product_counts, x="Product", y="Order Count", color="Product", title="Order Count by Product"))

    # Line Chart: Delivery Status Over Time
    st.subheader("📅 Delivery Status Over Time")
    status_over_time = filtered_data.groupby(["OnlyDate", "Status"]).size().reset_index(name="Count")
    status_over_time["OnlyDate"] = pd.to_datetime(status_over_time["OnlyDate"])
    fig_line = px.line(status_over_time, x="OnlyDate", y="Count", color="Status", markers=True,
                       title="Delivery Status Trend Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

    # Pie Chart: Distribution of Delivery Status
    st.subheader("📌 Delivery Status Distribution")
    status_counts = filtered_data["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    st.plotly_chart(px.pie(status_counts, values="Count", names="Status", title="Delivery Status Distribution", hole=0.4))

with tab2:
    # Forecasting Section (3-day Moving Average for Delivered Orders)
    st.subheader("🔮 Forecasting - 3-Day Moving Average")
    forecast_data = filtered_data[filtered_data["Status"] == "Delivered"]
    delivered_daily = forecast_data.groupby("OnlyDate").size().reset_index(name="Delivered_Count")
    delivered_daily["3_day_MA"] = delivered_daily["Delivered_Count"].rolling(window=3).mean()
    fig_forecast = px.line(delivered_daily, x="OnlyDate", y=["Delivered_Count", "3_day_MA"],
                           labels={"value": "Count", "variable": "Legend"},
                           title="Delivered Orders with 3-Day Moving Average")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ML Classifier: Predict Delay (Smarter Version)
    st.subheader("🤖 Predictive ML Classifier (Smarter Random Forest Model)")
    ml_data = filtered_data.copy()
    if len(ml_data["Status"].unique()) < 2:
        st.warning("Not enough class variety to train a predictive model.")
    else:
        ml_data["DayOfWeek"] = ml_data["Date"].dt.dayofweek
        ml_data["IsWeekend"] = ml_data["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
        product_freq = ml_data["Product"].value_counts().to_dict()
        ml_data["ProductFreq"] = ml_data["Product"].map(product_freq)

        le_product = LabelEncoder()
        ml_data["ProductEncoded"] = le_product.fit_transform(ml_data["Product"])

        ml_data["Target"] = ml_data["Status"].apply(lambda x: 1 if x == "Delayed" else 0)

        features = ["ProductEncoded", "DayOfWeek", "IsWeekend", "ProductFreq"]
        X = ml_data[features]
        y = ml_data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, zero_division=0)

        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
        st.text("Confusion Matrix:\n" + str(cm))
        st.text("Classification Report:\n" + cr)

with tab3:
    # Anomaly Detection Section (for Delayed Orders)
    st.subheader("🚨 Anomaly Detection - Delayed Orders")
    delay_data = filtered_data[filtered_data["Status"] == "Delayed"]
    delay_counts = delay_data.groupby("OnlyDate").size().reset_index(name="Delayed_Count")

    if not delay_counts.empty:
        mean = delay_counts["Delayed_Count"].mean()
        std = delay_counts["Delayed_Count"].std()
        delay_counts["z_score"] = (delay_counts["Delayed_Count"] - mean) / std
        delay_counts["Anomaly"] = delay_counts["z_score"].abs() > 2

        fig_anomaly = px.scatter(delay_counts, x="OnlyDate", y="Delayed_Count",
                                 color="Anomaly",
                                 color_discrete_map={True: 'red', False: 'green'},
                                 title="Anomaly Detection in Delayed Orders")
        st.plotly_chart(fig_anomaly, use_container_width=True)
    else:
        st.info("No delayed orders found in the selected filter to analyze anomalies.")

with tab4:
    st.subheader("📋 Filtered Order Data")
    st.dataframe(filtered_data.sort_values(by="Date").reset_index(drop=True))
