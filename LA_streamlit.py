import streamlit as st
import pandas as pd
import altair as alt
import requests
import json
import os
import time

# URL DE TU API (Backend) - Asegúrate de que coincida con uvicorn
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAIA - Frontend", layout="wide")

# -------------------- CLIENTE HTTP PARA API --------------------
def api_login(username, password):
    """Pregunta a la API si las credenciales son buenas."""
    try:
        response = requests.post(f"{API_URL}/check-login", auth=(username, password))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con la API. Asegúrate de ejecutar 'python -m uvicorn api:app'")
        return False

def api_register(username, password):
    """Envía los datos a la API para crear usuario."""
    try:
        payload = {"username": username, "password": password}
        response = requests.post(f"{API_URL}/register", json=payload)
        return response.status_code == 200
    except:
        return False

def api_predict(data_dict, username, password):
    """Envía los datos del crimen a la API y recibe la predicción."""
    try:
        response = requests.post(
            f"{API_URL}/predict", 
            json=data_dict, 
            auth=(username, password)
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error API: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

def api_chat(message, username, password):
    """Envía mensajes al Chatbot de la API."""
    try:
        payload = {"message": message}
        response = requests.post(f"{API_URL}/chat", json=payload, auth=(username, password))
        if response.status_code == 200:
            return response.json()['response']
        return "Error en el Chatbot API"
    except:
        return "Error de conexión con el Chatbot"

# -------------------- SISTEMA DE LOGIN (FRONTEND) --------------------
if "username" not in st.session_state: st.session_state["username"] = None
if "password" not in st.session_state: st.session_state["password"] = None

def login_screen():
    st.markdown("## 🔒 RAIA Client - Acceso Remoto")
    tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse (Vía API)"])

    with tab1:
        u = st.text_input("Usuario", key="l_user")
        p = st.text_input("Contraseña", type="password", key="l_pass")
        if st.button("Conectar"):
            if api_login(u, p):
                st.session_state["username"] = u
                st.session_state["password"] = p
                st.success("✅ Conectado a la API")
                st.rerun()
            else:
                st.error("Credenciales inválidas o API apagada")

    with tab2:
        nu = st.text_input("Nuevo Usuario", key="r_user")
        np = st.text_input("Nueva Contraseña", type="password", key="r_pass")
        if st.button("Enviar Registro"):
            if api_register(nu, np):
                st.success("Usuario creado en el servidor. Ahora puedes entrar.")
            else:
                st.error("Error al registrar (quizás ya existe).")

if not st.session_state["username"]:
    login_screen()
    st.stop()

# ==================== APLICACIÓN PRINCIPAL ====================

# --- BARRA LATERAL (LOGOUT) ---
st.sidebar.title(f"👤 {st.session_state['username']}")
if st.sidebar.button("🔴 Cerrar Sesión"):
    st.session_state["username"] = None
    st.session_state["password"] = None
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Menú", ["Dashboard (Local)", "Chatbot IA", "Predicción (Nube/API)"])

# Carga de CSV SOLO para visualización (Mapas/Dashboard)
@st.cache_data
def load_viz_data():
    try:
        df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
        df = df.dropna(subset=["LAT", "LON", "AREA NAME"])
        if "DATE OCC" in df.columns: df["YEAR"] = pd.to_datetime(df["DATE OCC"], errors='coerce').dt.year
        return df, sorted(df["AREA NAME"].unique())
    except:
        return pd.DataFrame(), []

df, areas = load_viz_data()

# -------------------- PÁGINA 1: DASHBOARD --------------------
if page == "Dashboard (Local)":
    st.title("📊 Dashboard de Datos")
    st.subheader("Visualización de Datos Locales")
    if not df.empty:
        st.map(df.sample(500)[["LAT", "LON"]].rename(columns={"LAT":"lat", "LON":"lon"}))
    else:
        st.warning("No se encontró el CSV local para el mapa.")

# -------------------- PÁGINA 2: CHATBOT --------------------
elif page == "Chatbot IA":
    st.title("💬 Chatbot Inteligente")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta a la IA (Ej: Tendencia robos)..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Llamada a la API
        with st.spinner("Pensando..."):
            response = api_chat(prompt, st.session_state["username"], st.session_state["password"])
        
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# -------------------- PÁGINA 3: PREDICCIÓN --------------------
elif page == "Predicción (Nube/API)":
    st.title("🧠 Consultar Inteligencia Artificial Remota")
    
    with st.form("api_form"):
        col1, col2 = st.columns(2)
        area = col1.selectbox("Zona", areas if areas else ["Central"])
        
        # Coordenadas
        lat, lon = (34.05, -118.24)
        if not df.empty:
            sample = df[df["AREA NAME"] == area].iloc[0]
            lat, lon = sample["LAT"], sample["LON"]
            
        date = col1.date_input("Fecha")
        hour = col1.slider("Hora", 0, 23, 12)
        age = col2.slider("Edad Víctima", 18, 90, 30)
        
        if st.form_submit_button("Enviar Petición a API"):
            payload = {
                "area": area,
                "lat": float(lat), "lon": float(lon),
                "date_year": date.year, "date_month": date.month,
                "day_of_week": date.weekday(),
                "hour": hour, "victim_age": age
            }
            
            with st.spinner("Esperando respuesta del servidor..."):
                result = api_predict(payload, st.session_state["username"], st.session_state["password"])
            
            if result:
                st.success("Respuesta recibida:")
                c1, c2 = st.columns(2)
                c1.metric("Predicción", result['prediction'], f"{result['confidence']:.1%}")
                
                # Gráfico
                if 'top_3' in result:
                    top3_df = pd.DataFrame(result['top_3'])
                    chart = alt.Chart(top3_df).mark_bar().encode(
                        x='probabilitat', y=alt.Y('crim', sort='-x')
                    )
                    c2.altair_chart(chart, use_container_width=True)