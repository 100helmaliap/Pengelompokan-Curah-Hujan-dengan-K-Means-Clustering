import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("K-Means Clustering Curah Hujan (2020-2024)")

uploaded_file = st.file_uploader(
    "Upload file Excel (curah-hujan-2020-2024.xlsx)", 
    type=["xlsx"]
)

if uploaded_file:
    # Baca data dari sheet pertama
    df = pd.read_excel(uploaded_file, sheet_name=0)
    st.write("Data Asli", df.head())

    # Pastikan kolom Tanggal bertipe datetime
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

    # Bersihkan data: ganti 8888 dengan NaN dan drop NaN
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
    df = df.replace(8888, np.nan).dropna(subset=['RR', 'Tanggal'])

    # Pilih jumlah klaster
    n_clusters = st.slider("Jumlah Klaster", 2, 6, 3)

    # K-Means clustering (hanya pada kolom RR)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['RR']])

    st.write("Data dengan Cluster", df.head())

    # Visualisasi hasil clustering
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        ax.scatter(cluster_data['Tanggal'], cluster_data['RR'], label=f'Cluster {i}')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Curah Hujan (RR)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Rata-rata tiap cluster
    st.write("Rata-rata Curah Hujan per Cluster:")
    st.dataframe(df.groupby('Cluster')['RR'].mean().reset_index())
else:
    st.info("curah-hujan-2020-2024.xlsx")
