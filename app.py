import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmenter Pro", layout="wide")

# --- LOGIN LOGIC ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐 Business Analyst Login")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pw == "admin123":
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Invalid Credentials")
    st.stop() # Prevents the rest of the app from running until login

# --- MAIN DASHBOARD ---
st.title("📊 Customer Segmentation AI")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    
    # 2. Automated Preprocessing
    # We drop ID-like columns and handle strings
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['CustomerID'], errors='ignore')
    
    # If Gender exists, encode it
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df_numeric['Gender'] = le.fit_transform(df['Gender'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # 3. Sidebar Controls
    st.sidebar.header("Model Settings")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)

    # 4. Apply K-Means
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    df['Cluster'] = df['Cluster'].apply(lambda x: f"Segment {x}")

    # 5. Visualizations
    tab1, tab2, tab3 = st.tabs(["Clustering Map", "Segment Profiles", "Data View"])

    with tab1:
        st.subheader("Interactive Cluster Map")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = df['Cluster']
        
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Cluster', 
                         title="Customer Groups in 2D Space", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Comparison of Segment Averages")
        # Calculate means for each cluster
        cluster_means = df.groupby('Cluster')[df_numeric.columns].mean().reset_index()
        
        # Radar Chart (Spider Chart)
        categories = df_numeric.columns.tolist()
        fig_radar = go.Figure()

        for i in range(len(cluster_means)):
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_means.iloc[i, 1:].values,
                theta=categories,
                fill='toself',
                name=cluster_means.iloc[i, 0]
            ))
        
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.subheader("Segmented Dataset")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Report", data=csv, file_name="segmented_customers.csv")

else:
    st.warning("Please upload a CSV file to see the analysis.")
