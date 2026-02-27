import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Cargar Datos
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Dividir para Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Definición de Modelos
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial (Degree 2)": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("linear", LinearRegression())
    ]),
    "Polynomial (Degree 3)": Pipeline([
        ("poly", PolynomialFeatures(degree=3)),
        ("linear", LinearRegression())
    ])
}

results = {}
best_r2 = -float("inf")
best_model = None
best_model_name = ""

# 4. Entrenamiento y Evaluación
plt.figure(figsize=(18, 6))

for i, (name, model) in enumerate(models.items()):
    # Fit
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {"R2": r2, "RMSE": rmse}
    
    print(f"\n--- {name} ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Check best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

    # Plot Valor Real vs Valor Predicho
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.title(f"{name}\nReal vs Predicho")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predicho")

plt.tight_layout()
plt.savefig("comparacion_modelos.png")
print("\n[INFO] Gráfica de comparación guardada como 'comparacion_modelos.png'")

# 5. Análisis de Residuos del Mejor Modelo
y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title(f"Distribución de Residuos - {best_model_name}")
plt.xlabel("Error")
plt.savefig("residuos_mejor_modelo.png")
print(f"[INFO] Análisis de residuos guardado como 'residuos_mejor_modelo.png'")

# 6. Guardar el mejor modelo
joblib.dump(best_model, "modelo.pkl")
print(f"\n[SUCCESS] El mejor modelo ({best_model_name}) ha sido guardado en 'modelo.pkl'")
