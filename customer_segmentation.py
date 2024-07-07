import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYdE4E0e68yxmVBlfNiYrarpyN9Vm9Edzmng&s");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# Load model, scaler, and encoders
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('le_education.pkl', 'rb') as le_education_file:
    le_education = pickle.load(le_education_file)

with open('le_marital_status.pkl', 'rb') as le_marital_status_file:
    le_marital_status = pickle.load(le_marital_status_file)

# Define mapping for dropdown options
education_mapping = {0: 'Undergraduate', 1: 'Graduate', 2: 'Postgraduate'}
marital_status_mapping = {0: 'Single', 1: 'Married'}

# Reverse mapping for label encoding
reverse_education_mapping = {v: k for k, v in education_mapping.items()}
reverse_marital_status_mapping = {v: k for k, v in marital_status_mapping.items()}

# Cluster information (replace with actual descriptions and recommendations)
cluster_info = {
    1: {
        "description": "Cluster 1: Customers have higher income and education levels, are generally married with moderate children, and prefer premium products like wines and gold.",
        "recommendation": "Focus on premium offerings, implement exclusive loyalty programs, and enhance customer service to retain their high spending and engagement levels."
    },
    2: {
        "description": "Cluster 2: Customer group comprises married customers with moderate income and education, having more children and preferring affordable products like fruits and sweets.",
        "recommendation": "Utilize value-oriented pricing, tailor family-oriented marketing, and leverage promotional strategies to appeal to their budget-conscious nature."
    },
    3: {
        "description": "Cluster 3: Comprises of younger customers with lower income and education levels, fewer children, and a preference for basic necessities and low-cost products.",
        "recommendation": "Emphasize affordability, improve digital engagement with exclusive online promotions, and streamline purchasing processes for convenience."
    }
}

# Streamlit UI
st.title("Customer Clustering Prediction")

# Form to collect user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        Education = st.selectbox("Education:", options=list(education_mapping.values()))
        Marital_Status = st.selectbox("Marital Status:", options=list(marital_status_mapping.values()))
        Income = st.number_input("Income:", min_value=0)
        MntWines = st.number_input("MntWines:", min_value=0)
        MntFruits = st.number_input("MntFruits:", min_value=0)
        MntMeatProducts = st.number_input("MntMeatProducts:", min_value=0)
        MntFishProducts = st.number_input("MntFishProducts:", min_value=0)
        MntSweetProducts = st.number_input("MntSweetProducts:", min_value=0)
        MntGoldProds = st.number_input("MntGoldProds:", min_value=0)

    with col2:
        NumDealsPurchases = st.number_input("NumDealsPurchases:", min_value=0)
        NumWebPurchases = st.number_input("NumWebPurchases:", min_value=0)
        NumCatalogPurchases = st.number_input("NumCatalogPurchases:", min_value=0)
        NumStorePurchases = st.number_input("NumStorePurchases:", min_value=0)
        NumWebVisitsMonth = st.number_input("NumWebVisitsMonth:", min_value=0)
        AcceptedCmp3 = st.number_input("AcceptedCmp3:", min_value=0, max_value=1)
        AcceptedCmp4 = st.number_input("AcceptedCmp4:", min_value=0, max_value=1)
        AcceptedCmp5 = st.number_input("AcceptedCmp5:", min_value=0, max_value=1)
        AcceptedCmp1 = st.number_input("AcceptedCmp1:", min_value=0, max_value=1)
        AcceptedCmp2 = st.number_input("AcceptedCmp2:", min_value=0, max_value=1)
        Age = st.number_input("Age:", min_value=0)
        TotalChildren = st.number_input("TotalChildren:", min_value=0)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Prepare the input data
    input_data = [
        reverse_education_mapping[Education],
        reverse_marital_status_mapping[Marital_Status],
        Income, MntWines, MntFruits, MntMeatProducts, MntFishProducts,
        MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases,
        NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth,
        AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2,
        Age, TotalChildren
    ]

    # Scaling the numerical features
    scaled_data = scaler.transform([input_data])

    # Making prediction
    prediction = model.predict(scaled_data)
    predicted_cluster = prediction[0] + 1  # Adjusting cluster numbers to start from 1

    # Display the result
    st.write(f"The predicted customer cluster is: **{predicted_cluster}**")
    st.write(f"Description: {cluster_info[predicted_cluster]['description']}")
    st.write(f"Strategic Recommendation: {cluster_info[predicted_cluster]['recommendation']}")

    # Visualize input data for the predicted cluster
    st.write("## Visualizations for the Predicted Cluster")

    if predicted_cluster == 1:
        # Visualization 1: Spending distribution
        labels = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        spending = [MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds]

        fig1, ax1 = plt.subplots()
        ax1.pie(spending, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        ax1.set_title("Spending Distribution for Cluster 1")
        st.pyplot(fig1)

        # Visualization 2: Income vs Spending
        fig2, ax2 = plt.subplots()
        spending_data = [MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds]
        sns.barplot(x=['Total Spending'], y=spending_data, ax=ax2)
        ax2.set_title("Income vs Total Spending for Cluster 1")
        st.pyplot(fig2)

        # Visualization 3: Age Distribution
        fig3, ax3 = plt.subplots()
        sns.histplot([Age], bins=10, kde=True, ax=ax3)
        ax3.set_title("Age Distribution for Cluster 1")
        st.pyplot(fig3)

    elif predicted_cluster == 2:
        # Visualization 1: Spending distribution
        labels = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        spending = [MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds]

        fig1, ax1 = plt.subplots()
        ax1.pie(spending, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        ax1.set_title("Spending Distribution for Cluster 2")
        st.pyplot(fig1)

        # Visualization 2: Purchase behavior
        purchase_labels = ['Deals Purchases', 'Web Purchases', 'Catalog Purchases', 'Store Purchases']
        purchase_counts = [NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases]

        fig2, ax2 = plt.subplots()
        sns.barplot(x=purchase_labels, y=purchase_counts, ax=ax2)
        ax2.set_ylabel('Number of Purchases')
        ax2.set_title("Purchase Behavior for Cluster 2")
        st.pyplot(fig2)

        # Visualization 3: Age Distribution
        fig3, ax3 = plt.subplots()
        sns.histplot([Age], bins=10, kde=True, ax=ax3)
        ax3.set_title("Age Distribution for Cluster 2")
        st.pyplot(fig3)

    elif predicted_cluster == 3:
        # Visualization 1: Spending distribution
        labels = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        spending = [MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds]

        fig1, ax1 = plt.subplots()
        ax1.pie(spending, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        ax1.set_title("Spending Distribution for Cluster 3")
        st.pyplot(fig1)
