import io
import os

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")
REQUIRED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

st.set_page_config(page_title="SuperKart Forecast", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Quarterly Sales Forecast")
st.caption("Forecast product-store revenue using the deployed ensemble model.")

single_tab, batch_tab, about_tab = st.tabs(["Single prediction", "Batch prediction", "Model notes"])

with single_tab:
    st.subheader("Product and store details")
    with st.form("single_prediction_form"):
        product_col, store_col = st.columns(2)
        with product_col:
            product_weight = st.number_input("Product weight", min_value=0.01, value=12.66, step=0.10)
            sugar_content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
            allocated_area = st.number_input(
                "Allocated display-area ratio", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f"
            )
            product_mrp = st.number_input("Product MRP", min_value=0.01, value=117.08, step=1.0)
            product_family = st.selectbox("Product ID family", ["FD", "DR", "NC"])
        with store_col:
            store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
            city_tier = st.selectbox("Store city tier", ["Tier 1", "Tier 2", "Tier 3"])
            store_type = st.selectbox(
                "Store type",
                ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
            )
            store_age = st.number_input("Store age (years)", min_value=0, value=16, step=1)
            product_category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])
        submitted = st.form_submit_button("Forecast sales", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_tier,
            "Store_Type": store_type,
            "Product_Id_char": product_family,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_category,
        }
        try:
            with st.spinner("Generating forecast..."):
                response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
                response.raise_for_status()
            prediction = response.json()["prediction"]
            st.success("Forecast completed")
            st.metric("Predicted product-store sales revenue", f"{prediction:,.2f}")
        except requests.RequestException as error:
            detail = error.response.json().get("error") if error.response is not None else str(error)
            st.error(f"Prediction request failed: {detail}")

with batch_tab:
    st.subheader("Upload a CSV for batch inference")
    st.write("Required columns:", ", ".join(REQUIRED_COLUMNS))
    uploaded_file = st.file_uploader("Batch CSV", type=["csv"])
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.dataframe(batch_data.head(10), use_container_width=True)
        missing = [column for column in REQUIRED_COLUMNS if column not in batch_data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif st.button("Run batch forecast", type="primary"):
            csv_bytes = batch_data.to_csv(index=False).encode("utf-8")
            files = {"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")}
            try:
                with st.spinner("Forecasting batch..."):
                    response = requests.post(
                        f"{BACKEND_URL}/v1/predictbatch",
                        files=files,
                        timeout=120,
                    )
                    response.raise_for_status()
                result = response.json()
                output = batch_data.copy()
                output["Predicted_Product_Store_Sales_Total"] = result["predictions"]
                st.success(f"Generated {result['count']:,} forecasts")
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="SuperKart_Batch_Predictions.csv",
                    mime="text/csv",
                )
            except requests.RequestException as error:
                detail = error.response.json().get("error") if error.response is not None else str(error)
                st.error(f"Batch request failed: {detail}")

with about_tab:
    st.markdown(
        "**Model:** tuned Random Forest with an embedded preprocessing pipeline.  \n"
        "**Use:** planning support for product-store revenue and inventory decisions.  \n"
        "**Caution:** predictions are estimates, not guarantees. New store formats, unusual "
        "values, or changing market conditions require review and model monitoring."
    )
