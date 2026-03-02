import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Predictor de Vivienda California Pro", layout="wide", initial_sidebar_state="expanded")

# Estilo Moderno para Streamlit
st.markdown("""
<style>
    .main { background-color: #0f172a; color: white; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #6366f1; color: white; border: none; }
    .stButton>button:hover { background-color: #4f46e5; border: none; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Predictor de Vivienda California Pro")
st.markdown("""
Estimación de valor de propiedad de nivel profesional utilizando **Regresión Polinomial Avanzada**.
""")

st.sidebar.header("⚙️ Configuración")
api_mode = st.sidebar.radio("Entorno de API de Destino", ["Localhost (8000)", "Producción (Render)"])
api_url = "http://127.0.0.1:8000/predict" if "Localhost" in api_mode else "https://TU_API_AQUI.onrender.com/predict"

st.sidebar.markdown("---")
st.sidebar.header("📍 Parámetros del Distrito")

# Columnas de entrada
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Ingreso Medio ($10k)", min_value=0.0, value=3.5, step=0.1, help="Ingreso medio en el grupo de bloques")
    HouseAge = st.number_input("Edad Media de la Vivienda", min_value=1.0, value=28.0, step=1.0)
    AveRooms = st.number_input("Promedio de Habitaciones por Hogar", min_value=1.0, value=5.0, step=0.1)
    AveBedrms = st.number_input("Promedio de Dormitorios por Hogar", min_value=0.1, value=1.0, step=0.1)

with col2:
    Population = st.number_input("Población del Distrito", min_value=1.0, value=1400.0, step=10.0)
    AveOccup = st.number_input("Ocupación Promedio", min_value=0.5, value=3.0, step=0.1)
    Latitude = st.number_input("Latitud", value=34.0, format="%.4f")
    Longitude = st.number_input("Longitud", value=-118.0, format="%.4f")

st.markdown("---")

# Lógica de predicción
if st.button("🚀 Calcular Valor de Mercado"):
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
    
    with st.spinner("Analizando patrones del mercado..."):
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                val_usd = prediction * 100000
                
                st.balloons()
                st.success(f"### Valor de Mercado Estimado: **${val_usd:,.0f} USD**")
                
                c1, c2 = st.columns(2)
                c1.metric("Salida Cruda del Modelo", f"{prediction:.4f}")
                c2.metric("Estado de Confianza", "Alto" if 0.5 < prediction < 5.0 else "Nominal")
                
            else:
                st.error(f"⚠️ Error de API ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ No se pudo conectar a {api_url}. Por favor, asegúrate de que la API esté funcionando.")
        except Exception as e:
            st.error(f"❌ Ocurrió un error inesperado: {e}")

st.sidebar.caption("v2.0.0 | Edición Premium")
