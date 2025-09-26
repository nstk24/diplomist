import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загружаем данные
file_path = "Raman_krov_SSZ-zdorovye.xlsx"
df_health = pd.read_excel(file_path, sheet_name="health")
df_disease = pd.read_excel(file_path, sheet_name="heart disease")

wavenumber = df_health["wavenumber"].values
X_health = df_health.drop(columns=["wavenumber"]).T.values
X_disease = df_disease.drop(columns=["wavenumber"]).T.values

# --- Сводка по данным ---
summary_df = pd.DataFrame({
    "n_healthy":[X_health.shape[0]],
    "n_disease":[X_disease.shape[0]],
    "n_points_per_spectrum":[X_health.shape[1]],
    "wavenumber_min":[float(np.min(wavenumber))],
    "wavenumber_max":[float(np.max(wavenumber))],
    "wavenumber_monotonic_increasing":[bool(np.all(np.diff(wavenumber) > 0))],
    "health_nan_count":[int(np.isnan(X_health).sum())],
    "disease_nan_count":[int(np.isnan(X_disease).sum())],
})
print("Сводка по данным:")
print(summary_df, "\n")

# --- График 1: средние спектры ---
plt.figure()
plt.plot(wavenumber, X_health.mean(axis=0), label="Healthy — mean")
plt.plot(wavenumber, X_disease.mean(axis=0), label="Heart disease — mean")
plt.xlabel("Wavenumber"); plt.ylabel("Intensity")
plt.title("Mean Raman spectra by class"); plt.legend(); plt.tight_layout(); plt.show()

# --- График 2: стандартные отклонения ---
plt.figure()
plt.plot(wavenumber, X_health.std(axis=0), label="Healthy — std")
plt.plot(wavenumber, X_disease.std(axis=0), label="Heart disease — std")
plt.xlabel("Wavenumber"); plt.ylabel("Std. deviation")
plt.title("Class-wise variability"); plt.legend(); plt.tight_layout(); plt.show()

# --- PCA с двумя нормировками ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.vstack([X_health, X_disease])
y = np.array([0]*X_health.shape[0] + [1]*X_disease.shape[0])

# 1. StandardScaler per feature
Xn = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(Xn)
print("PCA (StandardScaler per feature) explained variance ratios:", pca.explained_variance_ratio_)

plt.figure()
plt.scatter(Z[y==0,0], Z[y==0,1], label="Healthy", alpha=0.8)
plt.scatter(Z[y==1,0], Z[y==1,1], label="Heart disease", alpha=0.8)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA scatter (StandardScaler per feature)")
plt.legend(); plt.tight_layout(); plt.show()

# 2. SNV per spectrum
def snv_per_spectrum(X, eps=1e-8):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + eps
    return (X - mu)/sd

X_snv = snv_per_spectrum(X)
pca_snv = PCA(n_components=2, random_state=42)
Z_snv = pca_snv.fit_transform(X_snv)
print("PCA (SNV per spectrum) explained variance ratios:", pca_snv.explained_variance_ratio_)

plt.figure()
plt.scatter(Z_snv[y==0,0], Z_snv[y==0,1], label="Healthy (SNV)", alpha=0.8)
plt.scatter(Z_snv[y==1,0], Z_snv[y==1,1], label="Heart disease (SNV)", alpha=0.8)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA scatter (SNV per spectrum)")
plt.legend(); plt.tight_layout(); plt.show()

print("Готово: сводка, 4 графика и объяснённые дисперсии выведены.")
