import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from sklearn.linear_model import Ridge

def ml_model():
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

    st.write('### 2. Korelasi Linear antar Kolom Numerik')
    corr = df_select[numbers].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write('''Dilihat dari Korelasi akan dilakukan drop kolom, kolom yang akan di drop adalah kolom 
                performance index dan performance class karena kedua kolom tersebut akan menjadi target 
                pada model ini''')
    
    df = df[numbers]
    #8. Train Test Split
    st.write("### 3. Train–Test Split ")
    X = df.drop(["Performance Index", 'Performance Class'], axis=1)
    y = df["Performance Index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Pada model ini dilakukan split data dengan 80% data latih dan 20% data uji")
    st.write(f"X_train: {len(X_train)}", f", X_test: {len(X_test)}")

    from sklearn.preprocessing import StandardScaler
    st.write('### 4. Standarisasi')
    st.write('''Pada penelitian ini, standarisasi dilakukan menggunakan metode StandardScaler dari library scikit-learn. 
                Proses fit dilakukan hanya pada data latih (X_train) untuk mencegah terjadinya data leakage, 
                kemudian parameter hasil standarisasi tersebut digunakan untuk mentransformasikan data uji (X_test). 
                Tahapan ini penting karena dapat meningkatkan performa model, terutama pada algoritma yang sensitif terhadap 
                perbedaan skala data.''')
    #Normalisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write('### 5. Pemodelan Regresi Linear')
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    st.write("Akurasi Training =", round(train_accuracy * 100, 2), "%")
    st.write('''Hasil pemodelan **Regresi Linear** menunjukkan akurasi training sebesar **98,85%**, 
                yang menandakan bahwa model memiliki kemampuan yang sangat baik.''')
    y_pred_linear = model.predict(X_test)
    st.write("**Linear Regression**")
    st.write("MSE:", mean_squared_error(y_test, y_pred_linear))
    st.write("R² Score:", r2_score(y_test, y_pred_linear))

    col1, col2 = st.columns([6,4])
    with col1:
        st.write('**Parameter Model Logistic Regression**')
        feature_names = X_train.columns
        beta_0 = model.intercept_           # Intercept
        beta = model.coef_                  # Koefisien fitur

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

        st.write("**Hours Studied (β = 2,85)**")
        st.write("""Setiap peningkatan 1 satuan jam belajar akan meningkatkan nilai prediksi 
                    sebesar 2,85, dengan asumsi variabel lain konstan.""")
        st.write("**Previous Scores (β = 1,02)**")
        st.write("""Setiap kenaikan 1 satuan nilai sebelumnya meningkatkan nilai prediksi sebesar 1,02.""")
        st.write("**Sleep Hours (β = 0,47)**")
        st.write("Setiap tambahan 1 jam tidur meningkatkan nilai prediksi sebesar 0,47.")
        st.write("**Sample Question Papers Practiced (β = 0,19)**")
        st.write("Setiap tambahan 1 latihan soal meningkatkan nilai prediksi sebesar 0,19.")


    st.write("### 6. Ridge dan Lasso Regression - Hyperparameter Tuning")
    # Inisialisasi model dan parameter
    ridge = Ridge()
    lasso = Lasso(max_iter=10000)
    alphas = np.logspace(-3, 3, 20)  # 0.001 sampai 1000
    param_grid = {'alpha': alphas}

    # GridSearchCV
    with st.spinner("Training Ridge model dengan GridSearchCV..."):
        ridge_grid = GridSearchCV(
            ridge,
            param_grid,
            cv=10,
            scoring='neg_mean_squared_error'
        )
        ridge_grid.fit(X_train_scaled, y_train)

    with st.spinner("Training Lasso model dengan GridSearchCV..."):
        lasso_grid = GridSearchCV(
            lasso,
            param_grid,
            cv=10,
            scoring='neg_mean_squared_error'
        )
        lasso_grid.fit(X_train_scaled, y_train)

    # Output ke Streamlit
    st.write("### 📌 Hasil GridSearch Ridge")
    st.write("**Best Alpha:**", ridge_grid.best_params_['alpha'])
    st.write("**Best Score (Neg MSE):**", ridge_grid.best_score_)

    st.write("### 📌 Hasil GridSearch Lasso")
    st.write("**Best Alpha:**", lasso_grid.best_params_['alpha'])
    st.write("**Best Score (Neg MSE):**", lasso_grid.best_score_)

    st.write("### 7. Training Model dengan Alpha Terbaik")

    # ===================== SETUP ALPHA TERBAIK =====================
    best_alpha_ridge = ridge_grid.best_params_['alpha']
    best_alpha_lasso = lasso_grid.best_params_['alpha']

    # ===================== TRAIN MODEL FINAL =====================
    ridge_best = Ridge(alpha=best_alpha_ridge)
    lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)

    ridge_best.fit(X_train_scaled, y_train)
    lasso_best.fit(X_train_scaled, y_train)

    st.success("Model Ridge & Lasso berhasil dilatih dengan alpha terbaik 🎯")

    y_pred_linear = model.predict(X_test)
    y_pred_ridge = ridge_best.predict(X_test_scaled)
    y_pred_lasso = lasso_best.predict(X_test_scaled)

    def regression_metrics(y_true, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "R2": r2_score(y_true, y_pred)
        }
 
    st.write("### 8. Perbandingan Performa Model")
    linear_metrics = regression_metrics(y_test, y_pred_linear)
    ridge_metrics = regression_metrics(y_test, y_pred_ridge)
    lasso_metrics = regression_metrics(y_test, y_pred_lasso)
    comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "RMSE": [
            linear_metrics["RMSE"],
            ridge_metrics["RMSE"],
            lasso_metrics["RMSE"]
        ],
        "MAE": [
            linear_metrics["MAE"],
            ridge_metrics["MAE"],
            lasso_metrics["MAE"]
        ],
        "MAPE (%)": [
            linear_metrics["MAPE (%)"],
            ridge_metrics["MAPE (%)"],
            lasso_metrics["MAPE (%)"]
        ],
            "R² Score": [
            linear_metrics["R2"],
            ridge_metrics["R2"],
            lasso_metrics["R2"]
        ]
    })

    st.dataframe(comparison_df)
    joblib.dump(model, "regression_student_performance.pkl")

