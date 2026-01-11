import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def ml_model2():
    df = pd.read_csv('StudentPerformance.csv')
    # Create classification target
    df = df.drop_duplicates()
    df['Performance Class'] = np.where(df['Performance Index'] >= df['Performance Index'].median(), 1, 0)
    
    #1. Membagi kolom numerik dan kategorik
    numbers = df.select_dtypes(include = ['number']).columns

     #2. Deteksi dan penanganan outlier dengan IQR Method
    st.write('### 1. Deteksi Outlier ') 
    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    st.write(f"Jumlah data sebelum pembersihan: **{df.shape[0]} baris**")
    df = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]
    st.write(f"Jumlah data setelah pembersihan outlier: **{df.shape[0]} baris**")
    st.write("""Disini dapat disimpulkan bahwa dataset yang digunakan tidak memiliki outlier sama sekali""")

    #3. Membaca Dataset
    df_select = df.copy()
    st.write('**Dataset yang digunakan**')
    st.dataframe(df.head())

    st.write('### 2. Normalisasi menggunakan MinMax Scaler')
    for col in numbers:
        df_select[col] = MinMaxScaler().fit_transform(df_select[col].values.reshape(len(df_select), 1))
    st.write("""Normalisasi menggunakan MinMaxScaler merupakan tahapan praproses data yang bertujuan untuk 
                mengubah skala setiap fitur ke dalam rentang tertentu, umumnya 0 hingga 1. Metode ini bekerja 
                dengan menyesuaikan nilai minimum dan maksimum dari setiap fitur, sehingga seluruh variabel memiliki 
                skala yang seragam. Normalisasi MinMaxScaler membantu mencegah dominasi fitur dengan nilai besar terhadap 
                fitur lainnya serta meningkatkan kinerja model, khususnya pada algoritma yang sensitif terhadap perbedaan 
                skala data.""")


    st.write('### 3. Korelasi Linear antar Kolom Numerik')
    corr = df_select[numbers].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True, key = 'Correlation')
    st.write('''Dilihat dari Korelasi akan dilakukan drop kolom, kolom yang akan di drop adalah kolom 
                performance index dan performance class karena kedua kolom tersebut akan menjadi target 
                pada model ini''')

    df = df[numbers]

    st.write("### 4. Train–Test Split ")
    #Pemisahan Data Untuk clasifikasi
    X_c = df.drop(['Performance Index','Performance Class'], axis=1)
    y_c = df['Performance Class']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    st.write("Pada model ini dilakukan split data dengan 80% data latih dan 20% data uji")
    st.write(f"X_train: {len(X_train_c)}", f", X_test: {len(X_test_c)}")

    st.write("### 5. Handling Imbalance Class")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Mengapa Imbalance Class Penting?**
        Menangani imbalanced data sangat penting karena ketidakseimbangan jumlah data antar kelas dapat membuat 
        model bias terhadap kelas mayoritas, sehingga akurasi terlihat tinggi tetapi kemampuan model dalam mengenali 
        kelas minoritas—yang sering kali justru paling penting—menjadi rendah. Tanpa penanganan yang tepat, 
        hasil evaluasi model bisa menyesatkan dan keputusan yang diambil berdasarkan model tersebut menjadi kurang 
        akurat dan tidak dapat diandalkan.
        """)
    with col2:
        st.write('**Sebelum Balancing Class**')
        col1, col2 = st.columns(2)
        with col1:
            total_0 = (y_train_c == 0).sum()
            st.metric(label="Label 0", value = total_0)
        with col2:
            total_1 = (y_train_c == 1).sum()
            st.metric(label="Label 1", value = total_1)

    st.write('### 6. Pemodelan Klasifikasi')
    #Inisiasi model Logistic Regression
    model_logreg = LogisticRegression()

    #Train model
    model_logreg.fit(X_train_c, y_train_c)

    #Prediksi menggunakan data test
    y_pred_logreg = model_logreg.predict(X_test_c)

    train_accuracy_logreg = model_logreg.score(X_train_c, y_train_c)
    st.write("Akurasi Training =", round(train_accuracy_logreg * 100, 2), "%")
    col1, col2 = st.columns([6,4])
    with col1:
        st.write('**Parameter Model Logistic Regression**')
        feature_names = X_train_c.columns
        beta_0 = model_logreg.intercept_           # Intercept
        beta = model_logreg.coef_[0]                 # Koefisien fitur

        st.write("**β0 (Intercept)**") 
        st.write(beta_0)
        st.write("""Nilai **intercept (β₀)** sebesar −33,69 menunjukkan bahwa ketika seluruh variabel 
                    independen bernilai nol, model memprediksi Performance Index sebesar −33,69. Nilai 
                    ini berperan sebagai titik awal garis regresi dan lebih bersifat sebagai parameter 
                    matematis untuk menyesuaikan model.""")

        st.write("**β1, β2, ..., βn (Koefisien per Feature)**")
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient (β)": beta}).sort_values(by="Coefficient (β)", ascending=False)    
        st.dataframe(coef_df)

    st.write("""Hasil ini menunjukkan bahwa model memiliki akurasi training sebesar 97,4%, yang berarti model 
                mampu mengklasifikasikan data latih dengan sangat baik. Nilai akurasi yang tinggi mengindikasikan 
                bahwa pola pada data training berhasil dipelajari.""")

    #Import metrik evaluasi
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score

    # Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_logreg)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low", "Actual High"],
        columns=["Predicted Low", "Predicted High"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix - Logistic Regression"
    )

    st.plotly_chart(fig_cm, use_container_width=True)
    # Metrik Evaluasi
    accuracy = accuracy_score(y_test_c, y_pred_logreg)
    precision = precision_score(y_test_c, y_pred_logreg)
    recall = recall_score(y_test_c, y_pred_logreg)
    f1 = f1_score(y_test_c, y_pred_logreg)

    st.write("**Evaluation Metrics**")
    st.write(f"- Accuracy  : {accuracy:.4f}")
    st.write(f"- Precision : {precision:.4f}")
    st.write(f"- Recall    : {recall:.4f}")
    st.write(f"- F1-Score  : {f1:.4f}")

    st.write("""Model Logistic Regression mampu mengklasifikasikan data dengan sangat baik. Sebanyak 911 data kelas Low 
                dan 1009 data kelas High berhasil diprediksi dengan benar. Kesalahan prediksi relatif kecil, yaitu 35 data 
                Low yang salah diprediksi sebagai High dan 20 data High yang salah diprediksi sebagai Low.""")
    st.write("""Nilai evaluasi yang **tinggi—akurasi 97,22%**, **precision 96,65%**, **recall 98,06%**, dan **F1-score 97,35%**—menunjukkan 
                bahwa model tidak hanya akurat secara keseluruhan, tetapi juga seimbang dalam meminimalkan kesalahan prediksi 
                dan mampu mengenali kedua kelas dengan baik. Hal ini mengindikasikan bahwa model memiliki performa yang stabil 
                dan andal untuk tugas klasifikasi ini.""")
    
    joblib.dump(model_logreg, "logistic_regression_student_performance.pkl")
    joblib.dump(list(numbers),"numeric_columns.pkl")


