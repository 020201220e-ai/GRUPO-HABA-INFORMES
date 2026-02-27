import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Predicción California Housing", layout="wide")

st.title("🏠 Predicción de Precio de Vivienda en California")
st.markdown("""
Esta aplicación utiliza un modelo de **Regresión Polinomial (Grado 2)** para predecir el valor medio de las viviendas 
en distritos de California basándose en datos del censo.
""")

st.sidebar.header("Parámetros del Distrito")
st.sidebar.info("Ajuste los valores para realizar una predicción.")

# Configuración de entrada de datos
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Ingreso Medio (en $10,000s)", min_value=0.0, value=3.5, step=0.1)
    HouseAge = st.number_input("Edad Media de la Casa", min_value=1.0, value=28.0, step=1.0)
    AveRooms = st.number_input("Promedio de Habitaciones", min_value=1.0, value=5.0, step=0.1)
    AveBedrms = st.number_input("Promedio de Dormitorios", min_value=0.1, value=1.0, step=0.1)

with col2:
    Population = st.number_input("Población del Distrito", min_value=1.0, value=1400.0, step=10.0)
    AveOccup = st.number_input("Ocupación Promedio (personas/vivienda)", min_value=0.5, value=3.0, step=0.1)
    Latitude = st.number_input("Latitud", value=34.0, format="%.2f")
    Longitude = st.number_input("Longitud", value=-118.0, format="%.2f")

# Botón de predicción
if st.button("🚀 Calcular Valor Estimado"):
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }
    
    try:
        # Nota: URL local para desarrollo. En producción cambiar a URL de Hosting.
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"### El precio estimado de la vivienda es: ${prediction * 100:.2f}k")
            st.metric("Valor del Target (MedHouseVal)", f"{prediction:.4f}")
        else:
            st.error(f"Error en la API: {response.text}")
    except Exception as e:
        st.error(f"No se pudo conectar con la API. Asegúrese de que el servidor FastAPI esté corriendo en http://127.0.0.1:8000. Error: {e}")

st.markdown("---")
st.caption("Taller 1.1 - Modelos de Regresión Lineal Multiple y Polinomial")
