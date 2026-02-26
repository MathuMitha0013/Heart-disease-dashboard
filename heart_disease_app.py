import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Heart Disease Analysis Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

@st.cache_data
def preprocess_data(df):
    data = df.copy()

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 40, 50, 60, 100],
        labels=["Young", "Middle", "Senior", "Elder"]
    )

    categorical_cols = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal"
    ]

    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data

df = load_data()
data = preprocess_data(df)

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Section",
    [
        "Home",
        "Data Overview",
        "Distributions",
        "Relationships",
        "Target Analysis",
        "Summary"
    ]
)

if page == "Home":

    st.title("Heart Disease Analysis Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patients", len(data))

    with col2:
        st.metric("Heart Disease Cases", int(data["target"].sum()))

    with col3:
        st.metric("Average Age", round(data["age"].mean(), 1))

    with col4:
        st.metric("Average Cholesterol", round(data["chol"].mean(), 1))

    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)


elif page == "Data Overview":

    st.title("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"Rows: {data.shape[0]}")

    with col2:
        st.info(f"Columns: {data.shape[1]}")

    with col3:
        memory = data.memory_usage().sum() / 1024
        st.info(f"Memory Usage: {memory:.2f} KB")

    st.subheader("Data Types and Missing Values")

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes,
        "Non Null Count": data.count(),
        "Missing Values": data.isnull().sum()
    })

    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)


elif page == "Distributions":

    st.title("Feature Distribution")

    feature = st.selectbox("Select Feature", data.columns)

    if pd.api.types.is_numeric_dtype(data[feature]):

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                data,
                x=feature,
                nbins=30,
                title=f"Distribution of {feature}"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                data,
                y=feature,
                title=f"Box Plot of {feature}"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        vc = data[feature].value_counts().reset_index()
        vc.columns = ["Category", "Count"]

        fig = px.pie(
            vc,
            values="Count",
            names="Category",
            title=f"Distribution of {feature}"
        )

        st.plotly_chart(fig, use_container_width=True)


elif page == "Relationships":

    st.title("Relationship Analysis")

    corr_matrix = data.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter Plot")

    numeric_cols = data.select_dtypes(include=np.number).columns

    x_feature = st.selectbox("Select X Axis", numeric_cols)
    y_feature = st.selectbox("Select Y Axis", numeric_cols)

    fig = px.scatter(
        data,
        x=x_feature,
        y=y_feature,
        color="target",
        title=f"{x_feature} vs {y_feature}"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Target Analysis":

    st.title("Target Analysis")

    target_counts = data["target"].value_counts()

    fig = px.pie(
        values=target_counts.values,
        names=["No Disease", "Disease"],
        title="Heart Disease Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature by Target")

    feature = st.selectbox(
        "Select Feature",
        data.columns.drop("target")
    )

    if pd.api.types.is_numeric_dtype(data[feature]):

        fig = px.box(
            data,
            x="target",
            y=feature,
            title=f"{feature} by Target"
        )

    else:
        cross_tab = pd.crosstab(data[feature], data["target"])
        fig = px.bar(
            cross_tab,
            title=f"{feature} vs Target"
        )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Summary":

    st.title("Summary")

    st.write("Age shows a significant relationship with heart disease prevalence.")
    st.write("Maximum heart rate has negative correlation with disease risk.")
    st.write("Chest pain type is an important diagnostic factor.")
    st.write("Exercise induced angina is associated with higher risk.")
    st.write("Cholesterol and resting blood pressure contribute to overall risk assessment.")

st.markdown("Heart Disease Analysis Dashboard built using Streamlit and Plotly")