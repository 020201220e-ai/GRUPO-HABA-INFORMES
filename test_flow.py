import pandas as pd
import joblib

# 1. Crear un diccionario con datos de prueba de ejemplo (simulando JSON)
sample_json_input = {
    'MedInc': 3.8793,
    'HouseAge': 29.0,
    'AveRooms': 5.429,
    'AveBedrms': 1.097,
    'Population': 1425.0,
    'AveOccup': 3.07,
    'Latitude': 34.05,
    'Longitude': -118.25
}

print('--- Simulación de Flujo de API ---')
print(f'Entrada Recibida: {sample_json_input}')

# 2. Convertir el diccionario en un DataFrame (simulando la conversión en la API)
df_simulated = pd.DataFrame([sample_json_input])

# 3. Cargar el Pipeline de Producción
# En nuestra implementación, el Pipeline contiene tanto el PolynomialFeatures como la Regresión
try:
    production_pipeline = joblib.load('modelo.pkl')
    print("[INFO] Pipeline de producción cargado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    exit()

# 4. Realizar la predicción final
# El pipeline aplica automáticamente PolynomialFeatures(degree=2) antes de predecir
final_prediction = production_pipeline.predict(df_simulated)[0]

# 5. Imprimir el resultado final
print(f'Predicción Final (MedHouseVal): {final_prediction:.4f}')
print(f'Equivalente en USD (aprox): ${final_prediction * 100:.2f}k')
