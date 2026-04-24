import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Car Price Dashboard", layout="wide")

st.markdown("""
<div style="background-color:#1f77b4;padding:15px;border-radius:10px">
<h2 style="color:white;text-align:center;">🚗 Car Price Prediction Dashboard</h2>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DETAILS.csv")
    
    # Feature Engineering
    df['brand'] = df['name'].apply(lambda x: x.split()[0])
    df['car_age'] = 2024 - df['year']
    
    # Drop unnecessary columns
    df = df.drop(['name', 'year'], axis=1)
    
    return df

df = load_data()

# -------------------------------
# EDA SECTION
# -------------------------------
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

# Price Distribution
with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(df["selling_price"], bins=50, kde=True, ax=ax1)
    ax1.set_title("Selling Price Distribution")
    st.pyplot(fig1)

# Fuel Type Count
with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x="fuel", data=df, ax=ax2)
    ax2.set_title("Fuel Type Distribution")
    st.pyplot(fig2)

# -------------------------------
# SECOND ROW
# -------------------------------
col3, col4 = st.columns(2)

# Transmission
with col3:
    fig3, ax3 = plt.subplots()
    sns.countplot(x="transmission", data=df, ax=ax3)
    ax3.set_title("Transmission Type")
    st.pyplot(fig3)

# Owner
with col4:
    fig4, ax4 = plt.subplots()
    sns.countplot(x="owner", data=df, ax=ax4)
    ax4.set_title("Owner Type")
    st.pyplot(fig4)

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

fig5, ax5 = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax5)
st.pyplot(fig5)

# -------------------------------
# MODEL TRAINING (CACHED)
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    
    categorical_columns = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X, y)
    
    return pipeline

model = train_model(df)

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.header("Car Price Prediction")

st.sidebar.header("Enter Car Details")

user_input = {}

for column in df.drop('selling_price', axis=1).columns:
    if df[column].dtype == 'object':
        user_input[column] = st.sidebar.selectbox(column, df[column].unique())
    else:
        user_input[column] = st.sidebar.number_input(column, value=0)

input_df = pd.DataFrame(user_input, index=[0])

st.subheader("Input Data")
st.dataframe(input_df)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Price: ₹ {int(prediction[0]):,}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("Feature Importance")

rf_model = model.named_steps['model']
preprocessor = model.named_steps['preprocessor']

feature_names = preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig6, ax6 = plt.subplots(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_df.head(10), ax=ax6)
ax6.set_title("Top 10 Important Features")
st.pyplot(fig6)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed for Deployment 🚀")
