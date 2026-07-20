import pandas as pd
import streamlit as st

from src.constants.paths import RAW_DATA_PATH
from src.pipeline.prediction_pipeline import PredictionPipeline


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📈",
    layout="wide"
)


@st.cache_resource
def get_pipeline():
    return PredictionPipeline()


@st.cache_data
def load_reference_data():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def get_feature_schema():
    pipeline = get_pipeline()
    reference_df = load_reference_data()
    expected_columns = list(getattr(pipeline.preprocessor, "feature_names_in_", []))
    schema = []

    for column in expected_columns:
        if column not in reference_df.columns:
            continue

        series = reference_df[column].dropna()
        dtype_kind = reference_df[column].dtype.kind
        default_value = series.iloc[0] if not series.empty else ""
        unique_values = series.astype(str).unique().tolist() if not series.empty else []

        schema.append(
            {
                "name": column,
                "dtype_kind": dtype_kind,
                "default": default_value,
                "options": unique_values,
                "numeric": dtype_kind in {"i", "u", "f"},
            }
        )

    return schema


st.title("Customer Churn Prediction")
st.write(
    "Enter the customer's information below. The application uses the trained model artifacts to predict whether the customer is likely to churn."
)

pipeline = get_pipeline()
schema = get_feature_schema()
reference_df = load_reference_data()

with st.form("churn_prediction_form"):
    st.subheader("Customer Details")

    record = {}
    left_col, right_col = st.columns(2)

    for index, field in enumerate(schema):
        container = left_col if index % 2 == 0 else right_col
        name = field["name"]
        default_value = field["default"]

        with container:
            if field["numeric"]:
                record[name] = st.number_input(
                    name,
                    value=float(default_value) if pd.notna(default_value) else 0.0,
                    step=1.0 if field["dtype_kind"] in {"i", "u"} else 0.1,
                    format="%.4f",
                )
            else:
                options = [str(option) for option in field["options"] if str(option).strip()]
                if len(options) <= 20 and options:
                    default_index = 0
                    if pd.notna(default_value):
                        default_str = str(default_value)
                        if default_str in options:
                            default_index = options.index(default_str)
                    record[name] = st.selectbox(name, options, index=default_index)
                else:
                    record[name] = st.text_input(name, value=str(default_value) if pd.notna(default_value) else "")

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    try:
        input_df = pd.DataFrame([record])
        prediction = pipeline.predict(input_df)

        # Convert prediction to a user-friendly message
        if prediction[0] in [1, "Yes", "Churn", True]:
            st.error("⚠️ Prediction: Customer is likely to churn.")
        else:
            st.success("✅ Prediction: Customer is likely to stay.")

        st.subheader("Input Summary")
        st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.caption("Model artifacts are loaded from the existing artifacts directory. No retraining is performed during prediction.")