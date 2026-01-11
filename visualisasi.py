import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

#Card Metrics dan Button Filter
df = pd.read_csv('StudentPerformance.csv')
siswa_count = df.shape[0]
jumlah_k = df[df["Extracurricular Activities"] == "Yes"].shape[0]
rata_rata = df["Performance Index"].mean()
def chart():
    col1,col2,col3= st.columns([4,4,4])
    with col1:
        st.metric(label = 'Total Siswa', value = siswa_count)
    with col2:
        st.metric(label = 'Siswa Mengikuti Ekstrakulikuler', value = jumlah_k)
    with col3:
         st.metric(label = 'Rata - Rata Nilai', value = rata_rata)
    if 'selected_ekskul' not in st.session_state:
        st.session_state.selected_ekskul = None
    st.dataframe(df.head(5))

    
    category_df = df['Extracurricular Activities'].value_counts(dropna=False).reset_index()
    category_df.columns = ['Extracurricular Activities','count']
    
    fig = px.pie(category_df, names='Extracurricular Activities', values='count', title='Distribus Siswa Ekstrakulikuler')
    st.plotly_chart(fig, use_container_width=True)

    st.write("Dapat dilihat diatas bahwa jumlah siswa yang mengikuti kegiatan di luar kelas lebih banyak dari pada siswa yang tidak mengikuti kegiatan di luar kelas ")

    # Distribusi Performa
    st.subheader("Distribusi Performance Index")

    fig_dist = px.histogram(
        df,
        x="Performance Index",
        nbins=20,
        title="Distribusi Performance Index"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.write("""Distribusi **Performance Index** menunjukkan pola yang relatif mendekati distribusi normal, dengan konsentrasi nilai terbesar 
                berada pada rentang menengah hingga tinggi (sekitar 40–75). Hal ini mengindikasikan bahwa sebagian besar individu memiliki 
                tingkat performa yang cukup baik dan stabil.""")

    st.subheader("Boxplot Kolom Numerik")

    # Ambil kolom numerik saja
    num_cols = df.select_dtypes(include=np.number).columns

    # Ubah ke long format
    df_long = df[num_cols].melt(
        var_name="Kolom",
        value_name="Nilai"
    )

    # Plot boxplot
    fig_all_box = px.box(
        df_long,
        x="Kolom",
        y="Nilai",
    )
    st.plotly_chart(fig_all_box, use_container_width=True)
    st.write("""Boxplot menunjukkan bahwa **Previous Scores** dan **Performance Index** memiliki sebaran nilai paling luas, 
                menandakan variasi data yang tinggi antar individu.""")

    st.subheader("Performance Index vs Extracurricular Activities")

    fig_ekskul = px.box(
        df,
        x="Extracurricular Activities",
        y="Performance Index",
        title="Perbandingan Performance Index Berdasarkan Ekstrakurikuler"
    )

    st.plotly_chart(fig_ekskul, use_container_width=True)
    st.write("""Boxplot menunjukkan bahwa median **Performance Index** siswa yang mengikuti kegiatan ekstrakurikuler (Yes) dan 
                yang tidak mengikuti (No) berada pada tingkat yang relatif sama. Hal ini mengindikasikan bahwa partisipasi dalam 
                kegiatan ekstrakurikuler tidak menunjukkan perbedaan yang signifikan terhadap tingkat performa akademik. Variasi 
                nilai pada kedua kelompok juga hampir serupa, sehingga pengaruh ekstrakurikuler terhadap Performance Index cenderung 
                tidak dominan dibandingkan faktor akademik lainnya.""")


