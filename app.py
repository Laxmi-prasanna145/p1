import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as iogo
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Segmenter AI", layout="wide")

# --- MOCK LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    st.title("🔐 Business Analyst Portal")
    with st.form("login_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if user == "admin" and pw == "password123": # Simple mock check
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid credentials")

# --- MAIN APP ---
def main_app():
    st.sidebar.title("Navigation")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.title("📊 Customer Segmentation Dashboard")
    st.info("Upload your CSV dataset to begin automatic clustering.")

    uploaded_file = st.file_uploader("Upload Customer Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        # --- PREPROCESSING ---
        # 1. Select numeric features automatically
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'CustomerID' in numeric_cols: numeric_cols.remove('CustomerID')
        
        # 2. Handle Categorical (Gender)
        processed_df = df.copy()
        if 'Gender' in df.columns:
            le = LabelEncoder()
            processed_df['Gender'] = le.fit_transform(df['Gender'])
            if 'Gender' not in numeric_cols: numeric_cols.append('Gender')

        # 3. Scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(processed_df[numeric_cols])

        # --- CLUSTERING ---
        st.sidebar.header("Settings")
        k = st.sidebar.slider("Number of Clusters", 2, 10, 5)
        
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        df['Cluster'] = df['Cluster'].astype(str)

        # --- VISUALIZATIONS ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Distribution (PCA)")
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled_data)
            fig_pca = px.scatter(components, x=0, y=1, color=df['Cluster'],
                                 title="2D Cluster View", labels={'0': 'PCA 1', '1': 'PCA 2'})
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            st.subheader("Spending vs Income")
            if 'AnnualIncome' in df.columns and 'SpendingScore' in df.columns:
                fig_scatter = px.scatter(df, x="AnnualIncome", y="SpendingScore", color="Cluster", 
                                         hover_data=['Age'], title="Income vs Spending")
                st.plotly_chart(fig_scatter, use_container_width=True)

        # --- ANALYSIS INFO ---
        st.subheader("Segment Analysis")
        summary = df.groupby('Cluster')[numeric_cols].mean()
        st.dataframe(summary.style.highlight_max(axis=0))

        st.success(f"Analysis Complete! Found {k} distinct customer groups.")
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

# Logic to switch pages
if not st.session_state['logged_in']:
    login_page()
else:
    main_app()
