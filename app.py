import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

st.title("K-Means Clustering Curah Hujan (2020-2024)")

uploaded_file = st.file_uploader(
    "Upload file Excel (curah-hujan-2020-2024.xlsx)", 
    type=["xlsx"]
)

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    st.write("Data Asli", df.head())

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
    df = df.replace(8888, np.nan).dropna(subset=['RR', 'Tanggal'])

    n_clusters = st.slider("Jumlah Klaster", 2, 6, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['RR']])

    st.write("Data dengan Cluster", df.head())

    # Visualisasi dengan matplotlib (scatter plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        ax.scatter(cluster_data['Tanggal'], cluster_data['RR'], label=f'Cluster {i}')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Curah Hujan (RR)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualisasi interaktif dengan Plotly Express
    fig2 = px.scatter(df, x='Tanggal', y='RR', color='Cluster',
                      title='Visualisasi Interaktif Clustering Curah Hujan',
                      labels={'RR': 'Curah Hujan (RR)', 'Tanggal': 'Tanggal'},
                      color_continuous_scale=px.colors.qualitative.Safe)
    fig2.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig2, use_container_width=True)

    # Rata-rata curah hujan per cluster
    st.write("Rata-rata Curah Hujan per Cluster:")
    st.dataframe(df.groupby('Cluster')['RR'].mean().reset_index())
else:
    st.info("curah-hujan-2020-2024.xlsx")
