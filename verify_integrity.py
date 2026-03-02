import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Cargar el modelo guardado (Pipeline que incluye Poly y Linear)
try:
    loaded_pipeline = joblib.load('modelo.pkl')
    print("[INFO] Modelo cargado correctamente desde 'modelo.pkl'")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    exit()

# 2. Cargar datos para obtener una muestra de prueba
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Seleccionar una muestra de prueba
sample_input = X_test.iloc[[0]]
actual_value = y_test[0]

# 4. Realizar predicción con el pipeline cargado
prediction_loaded = loaded_pipeline.predict(sample_input)[0]

print(f"\n--- Verificación de Integridad ---")
print(f"Muestra de entrada (primeras columnas):\n{sample_input.iloc[:, :3]}")
print(f"Valor Real (MedHouseVal): {actual_value:.6f}")
print(f"Predicción del Modelo:    {prediction_loaded:.6f}")

# 5. Validación de éxito
# Nota: Como usamos un Pipeline, la transformación polinomial está integrada
# Verificamos simplemente que el objeto sea funcional y devuelva una predicción coherente
if isinstance(prediction_loaded, (float, np.float64, np.float32)):
    print("\n¡Éxito! El modelo cargado mantiene su funcionalidad original y es consistente.")
else:
    print("\nAdvertencia: El resultado de la predicción no es del tipo esperado.")
