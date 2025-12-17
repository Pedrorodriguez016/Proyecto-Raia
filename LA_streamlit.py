import streamlit as st
import pandas as pd
import altair as alt
import requests
import json
import os
import time
import datetime
import math
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim

# URL DE TU API (Backend)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAIA - Frontend", layout="wide")

# -------------------- DICCIONARIO DE TRADUCCIÓN --------------------
CRIME_TRANSLATIONS = {
    # Crímenes violentos
    "BATTERY - SIMPLE ASSAULT": "AGRESIÓN - ASALTO SIMPLE",
    "ASSAULT WITH DEADLY WEAPON": "ASALTO CON ARMA MORTAL",
    "AGGRAVATED ASSAULT": "AGRESIÓN AGRAVADA",
    "INTIMATE PARTNER - SIMPLE ASSAULT": "VIOLENCIA DE PAREJA - ASALTO SIMPLE",
    "INTIMATE PARTNER - AGGRAVATED ASSAULT": "VIOLENCIA DE PAREJA - AGRESIÓN AGRAVADA",
    "CRIMINAL THREATS": "AMENAZAS CRIMINALES",
    
    # Robos
    "ROBBERY": "ROBO CON VIOLENCIA",
    "THEFT": "HURTO",
    "THEFT PLAIN - PETTY": "HURTO MENOR",
    "THEFT FROM MOTOR VEHICLE": "HURTO DE VEHÍCULO MOTORIZADO",
    "THEFT OF IDENTITY": "ROBO DE IDENTIDAD",
    "SHOPLIFTING": "HURTO EN TIENDA",
    "BURGLARY": "ROBO CON ALLANAMIENTO",
    "BURGLARY FROM VEHICLE": "ROBO DE VEHÍCULO",
    "PICKPOCKET": "CARTERISTA",
    "PURSE SNATCHING": "ROBO DE BOLSO",
    
    # Vehículos
    "VEHICLE - STOLEN": "VEHÍCULO ROBADO",
    "VANDALISM": "VANDALISMO",
    
    # Otros
    "FRAUD": "FRAUDE",
    "TRESPASSING": "ALLANAMIENTO DE MORADA",
    "BRANDISH WEAPON": "EXHIBIR ARMA",
    "WEAPON": "ARMA",
    "DISCHARGE FIREARM": "DISPARO DE ARMA DE FUEGO",
    "DRUNK ROLL": "ROBO A EBRIO",
    "BIKE - STOLEN": "BICICLETA ROBADA",
    "DOCUMENT FORGERY": "FALSIFICACIÓN DE DOCUMENTOS",
    "EMBEZZLEMENT": "MALVERSACIÓN",
    "EXTORTION": "EXTORSIÓN",
    "KIDNAPPING": "SECUESTRO",
    "RAPE": "VIOLACIÓN",
    "SEXUAL": "DELITO SEXUAL",
    "HOMICIDE": "HOMICIDIO",
    "ARSON": "INCENDIO PROVOCADO",
}

def translate_crime(crime_name):
    """Traduce nombre de crimen de inglés a español"""
    if not crime_name:
        return "Desconocido"
    
    crime_upper = crime_name.upper()
    
    # Buscar traducción exacta
    if crime_upper in CRIME_TRANSLATIONS:
        return CRIME_TRANSLATIONS[crime_upper]
    
    # Buscar traducción parcial (por palabras clave)
    for eng, esp in CRIME_TRANSLATIONS.items():
        if eng in crime_upper:
            return esp
    
    # Si no hay traducción, retornar original
    return crime_name

def format_duration(min_val):
    if min_val >= 60:
        h = min_val // 60
        m = min_val % 60
        return f"{h}h {m}min"
    return f"{min_val} min"

# -------------------- GESTIÓN DE CACHE / HISTORIAL --------------------
HISTORY_FILE = "prediction_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_to_history(input_data, result, area_name, username):
    history = load_history()
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": username,
        "input": input_data,
        "result": result,
        "area": area_name
    }
    # Guardamos al inicio de la lista
    history.insert(0, entry)
    # Escribimos en JSON
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        st.error(f"Error guardando historial: {e}")

# -------------------- CLIENTE HTTP PARA API --------------------
def api_login(username, password):
    try:
        response = requests.post(f"{API_URL}/check-login", auth=(username, password), timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        st.error("No se puede conectar con la API. Ejecuta 'python -m uvicorn api:app'")
        return False

def api_register(username, password):
    try:
        payload = {"username": username, "password": password}
        response = requests.post(f"{API_URL}/register", json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

def api_predict(data_dict, username, password):
    try:
        response = requests.post(f"{API_URL}/predict", json=data_dict, auth=(username, password), timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.warning("El modelo aún no está cargado en el servidor.")
            return None
        else:
            st.error(f"Error API: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

def api_chat(message, username, password):
    try:
        payload = {"message": message}
        response = requests.post(f"{API_URL}/chat", json=payload, auth=(username, password), timeout=10)
        if response.status_code == 200:
            return response.json()['response']
        return "Error en el Chatbot API"
    except:
        return "Error de conexión con el Chatbot"

# -------------------- SISTEMA DE LOGIN --------------------
if "username" not in st.session_state: st.session_state["username"] = None
if "password" not in st.session_state: st.session_state["password"] = None

def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("RAIA SECURE ACCESS")
        st.markdown("Autenticación requerida para acceder al sistema.")
        
        tab1, tab2 = st.tabs(["Ingresar", "Registro"])
        
        with tab1:
            u = st.text_input("Usuario", key="l_user")
            p = st.text_input("Contraseña", type="password", key="l_pass")
            if st.button("Iniciar Sesión"):
                with st.spinner("Autenticando..."):
                    if api_login(u, p):
                        st.session_state["username"] = u
                        st.session_state["password"] = p
                        st.success("Acceso concedido")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Credenciales inválidas o servidor caído.")
        
        with tab2:
            nu = st.text_input("Nuevo Usuario", key="r_user")
            np = st.text_input("Nueva Contraseña", type="password", key="r_pass")
            if st.button("Crear Cuenta"):
                if api_register(nu, np):
                    st.success("Usuario creado.")
                else:
                    st.error("Error al crear usuario.")

if not st.session_state["username"]:
    login_screen()
    st.stop()

# ==================== APP PRINCIPAL ====================
st.sidebar.markdown(f"### {st.session_state['username']}")
if st.sidebar.button("Cerrar Sesión"):
    st.session_state["username"] = None
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Navegación", 
    ["Dashboard", "Predicción de Crimen", "Historial / Cache", "Asistente IA", "Navegador Seguro (OSM)"]
)

# Carga de datos local para visualización
@st.cache_data(show_spinner="Cargando dataset optimizado...")
def load_viz_data():
    try:
        csv_path = os.path.join("output", "crimes_clean.csv")
        if not os.path.exists(csv_path):
            # Fallback al original si no existe el limpio
            csv_path = "Crime_Data_from_2020_to_Present.csv"
            
        cols = ["LAT", "LON", "AREA NAME", "DATE OCC", "Crm Cd Desc", "Vict Age", "Vict Sex"]
        # Intentamos leer solo columnas necesarias, pero crimes_clean podría tener nombres distintos
        # Si es el dataset limpio, asumimos que ya está bien formad o verificamos columnas.
        # Para seguridad, leemos normal primero y filtramos, o usamos usecols si estamos seguros.
        # Al ser 'clean', es probable que ya no necesite tantos drops/conversions pero mantenemos la robustez.
        df = pd.read_csv(csv_path) 
        
        # Estandarización de nombres si difieren en el clean (asumiendo estructura similar)
        # Si el clean ya tiene YEAR, DATE OCC parseado, mejor.
        
        if "LAT" not in df.columns and "lat" in df.columns: df.rename(columns={"lat": "LAT", "lon": "LON"}, inplace=True)
        if "AREA NAME" not in df.columns and "area_name" in df.columns: df.rename(columns={"area_name": "AREA NAME"}, inplace=True)
        
        # Si ha funcionado, devolvemos
        if "LAT" in df.columns and "LON" in df.columns:
            df = df.dropna(subset=["LAT", "LON"])
        
        if "DATE OCC" in df.columns: 
            df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors='coerce')
            df["YEAR"] = df["DATE OCC"].dt.year
        elif "date_occ" in df.columns:
             df["DATE OCC"] = pd.to_datetime(df["date_occ"], errors='coerce')
             df["YEAR"] = df["DATE OCC"].dt.year
             
        # Si ha funcionado, devolvemos
        return df, sorted(df["AREA NAME"].dropna().unique()) if "AREA NAME" in df.columns else []
    except Exception as e:
        print(f"Error loading data: {e}")
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), []

df_local = pd.DataFrame()
areas = []

# Debug visual
placeholder = st.empty()
placeholder.info("Inicializando aplicación...")

try:
    df_local, areas = load_viz_data()
    placeholder.success("Datos cargados correctamente.")
    time.sleep(1)
    placeholder.empty()
except Exception as e:
    placeholder.error(f"Error fatal cargando datos: {e}")


# --- PAGE: DASHBOARD ---
if page == "Dashboard":
    st.title("Estadísticas de Seguridad")
    st.markdown("Visión general de incidentes en Los Ángeles.")
    
    if not df_local.empty:
        # --- SELECCIÓN DE VISTA ---
        view_options = [
            "Evolución Temporal", 
            "Distribución Horaria", 
            "Comparativa Zonas", 
            "Top Crímenes", 
            "Análisis Víctimas", 
            "Mapa de Calor Interactivo"
        ]
        
        selected_view = st.selectbox("Selecciona Visualización", view_options)
        
        # --- FILTROS (SOLO PARA MAPA) ---
        df_filtered = df_local
        
        if selected_view == "Mapa de Calor Interactivo":
            st.sidebar.markdown("### Filtros del Mapa")
            
            # Filtro de años
            if 'YEAR' in df_local.columns:
                years_available = sorted(df_local['YEAR'].dropna().unique())
                selected_years = st.sidebar.multiselect("Años", years_available, default=[])
                if selected_years:
                    df_filtered = df_local[df_local['YEAR'].isin(selected_years)]
            
            # Filtro de zonas
            selected_areas = st.sidebar.multiselect("Zonas", areas, default=[])
            if selected_areas:
                df_filtered = df_filtered[df_filtered['AREA NAME'].isin(selected_areas)]
                
            # Filtro por tipo de crimen
            if 'Crm Cd Desc' in df_local.columns:
                top_crimes = df_local['Crm Cd Desc'].value_counts().head(10).index.tolist()
                crime_options = {translate_crime(c): c for c in top_crimes}
                selected_crimes_es = st.sidebar.multiselect("Tipos de Crimen (Top 10)", list(crime_options.keys()), default=[])
                selected_crimes = [crime_options[c] for c in selected_crimes_es]
                if selected_crimes:
                    df_filtered = df_filtered[df_filtered['Crm Cd Desc'].isin(selected_crimes)]
        
        # --- MÉTRICAS DE LA SELECCIÓN ACTUAL ---
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Registros", f"{len(df_filtered):,}")
        with c2: st.metric("Zonas Activas", len(df_filtered['AREA NAME'].unique()))
        with c3: 
            if 'YEAR' in df_filtered.columns: st.metric("Años Analizados", len(df_filtered['YEAR'].unique()))
            else: st.metric("Dataset", "Completo")
        with c4:
            if 'Crm Cd Desc' in df_filtered.columns: st.metric("Tipos de Crimen", len(df_filtered['Crm Cd Desc'].unique()))
        
        st.markdown("---")
        
        # --- VISUALIZACIÓN ---
        img_dir = os.path.join("output", "figures")
        
        if selected_view == "Evolución Temporal":
            img_path = os.path.join(img_dir, "02_evolucion_mensual.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Evolución mensual de incidentes (Histórico Global)", use_container_width=True)
            else:
                st.info("Imagen no encontrada.")
                
        elif selected_view == "Distribución Horaria":
            img_path = os.path.join(img_dir, "05_distribution_hour.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Distribución de incidentes por hora (Histórico Global)", use_container_width=True)
            else:
                st.info("Imagen no encontrada.")
                
        elif selected_view == "Comparativa Zonas":
            img_path = os.path.join(img_dir, "04_top_areas.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Incidentes por área (Histórico Global)", use_container_width=True)
            else:
                st.info("Imagen no encontrada.")
                
        elif selected_view == "Top Crímenes":
            img_path = os.path.join(img_dir, "01_top_crimes.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Tipos de crímenes más frecuentes (Histórico Global)", use_container_width=True)
            else:
                st.info("Imagen no encontrada.")
                
        elif selected_view == "Análisis Víctimas":
            c_vic1, c_vic2 = st.columns(2)
            with c_vic1:
                img_age = os.path.join(img_dir, "06_victim_age.png")
                if os.path.exists(img_age):
                    st.image(img_age, caption="Distribución por Edad", use_container_width=True)
            with c_vic2:
                img_sex = os.path.join(img_dir, "07_victim_sex.png")
                if os.path.exists(img_sex):
                    st.image(img_sex, caption="Distribución por Género", use_container_width=True)
            
            img_heat = os.path.join(img_dir, "09_heatmap_zona_sexo.png")
            if os.path.exists(img_heat):
                st.markdown("---")
                st.image(img_heat, caption="Relación Zona vs Género", use_container_width=True)

        elif selected_view == "Mapa de Calor Interactivo":
            st.markdown("### Mapa de Calor Interactivo")
            
            def create_dashboard_map(df_data, data_limit):
                m = folium.Map(location=[34.05, -118.24], zoom_start=10, tiles='OpenStreetMap')
                sample_data = df_data.sample(min(data_limit, len(df_data)), random_state=42) if len(df_data) > data_limit else df_data
                heat_data = [[row['LAT'], row['LON']] for index, row in sample_data.iterrows()]
                if heat_data:
                    HeatMap(heat_data, radius=13, blur=18, min_opacity=0.3).add_to(m)
                return m

            m_dash = create_dashboard_map(df_filtered, 2000)
            st_folium(m_dash, width=1200, height=600, key="dashboard_map", returned_objects=[])
            st.caption("Mapa de Calor basado en OpenStreetMap: Las zonas rojas indican mayor concentración de incidentes.")

    else:
        st.warning("No hay datos locales disponibles para el mapa.")

# --- PAGE: PREDICCIÓN ---
elif page == "Predicción de Crimen":
    st.title("Predicción de crimen")
    st.markdown("Introduce los parámetros para estimar la probabilidad de crimen.")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            area = st.selectbox("Zona", areas if areas else ["Central"])
            date = st.date_input("Fecha")
            hour = st.slider("Hora del día", 0, 23, 12, format="%d:00")
        with c2:
            age = st.slider("Edad potencial víctima", 18, 90, 30)
            sex = st.selectbox("Género", ["M", "F", "X"], format_func=lambda x: "Hombre" if x=="M" else "Mujer" if x=="F" else "Otro")
            
            # Auto-localización interna (no visible para el usuario)
            lat, lon = (34.05, -118.24)  # Coordenadas por defecto de LA
            if not df_local.empty:
                sample = df_local[df_local["AREA NAME"] == area].iloc[0]
                lat, lon = sample["LAT"], sample["LON"]

    if st.button("Generar Predicción", use_container_width=True):
        payload = {
            "area": area,
            "lat": float(lat), "lon": float(lon),
            "date_year": date.year, "date_month": date.month,
            "day_of_week": date.weekday(),
            "hour": hour, "victim_age": age,
            "victim_sex": sex
        }
        
        with st.spinner("Analizando patrones..."):
            result = api_predict(payload, st.session_state["username"], st.session_state["password"])
        
        if result:
            st.success("Análisis Completado")
            
            # Guardamos en Cache/Historial vinculando al usuario
            save_to_history(payload, result, area, st.session_state["username"])
            
            # --- VISUALIZACIÓN MEJORADA CON SEVERIDAD ---
            st.markdown("---")
            # Layout de resultados
            col_res, col_chart = st.columns([1, 2])
            
            with col_res:
                st.markdown("### Predicción Principal")
                
                # Métrica con color según severidad
                prediction_text = translate_crime(result['prediction'])
                confidence = result['confidence']
                
                st.metric(
                    label="Tipo de Crimen Predicho", 
                    value=prediction_text[:50] + "..." if len(prediction_text) > 50 else prediction_text
                )      
            with col_chart:
                if 'top_3' in result:
                    st.markdown("### Top 3 Predicciones")
                    
                    # Traducir los crímenes del top 3
                    top3_translated = [{
                        'crim': translate_crime(item['crim']),
                        'probabilitat': item['probabilitat']
                    } for item in result['top_3']]
                    
                    top3_df = pd.DataFrame(top3_translated)
                    
                    # Gráfico de barras horizontales
                    chart = alt.Chart(top3_df).mark_bar(color='steelblue').encode(
                        x=alt.X('probabilitat:Q', title='Probabilidad', axis=alt.Axis(format='%')),
                        y=alt.Y('crim:N', sort='-x', title='Tipo de Crimen'),
                        tooltip=[
                            alt.Tooltip('crim:N', title='Crimen'),
                            alt.Tooltip('probabilitat:Q', title='Probabilidad', format='.1%')
                        ]
                    ).properties(height=250)
                    
                    st.altair_chart(chart, use_container_width=True)
            # Información contextual adicional
# --- PAGE: HISTORIAL ---
elif page == "Historial / Cache":
    st.title("Historial de Predicciones")
    st.markdown("Registro de todas las consultas realizadas con funciones de filtrado y exportación.")
    
    history = load_history()
    
    # Filtrar solo el historial del usuario actual
    user_history = [h for h in history if h.get("user") == st.session_state["username"]]
    
    if user_history:

        
        filtered_history = user_history.copy()
        
        # Botones de exportación

        
        # Mostrar historial
        st.markdown("---")
        st.markdown("### Detalle de Consultas")
        
        for idx, item in enumerate(filtered_history):
            with st.expander(f"{item.get('timestamp', 'N/A')} - {item.get('area', 'Unknown')} ({idx+1}/{len(filtered_history)})"):
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("**Parámetros de Entrada:**")
                    input_data = item.get('input', {})
                    st.json(input_data)
                
                with col_d2:
                    st.markdown("**Resultado:**")
                    result = item.get('result', {})
                    
                    pred = translate_crime(result.get('prediction', 'N/A'))
                    st.markdown(f"**Predicción:** {pred}")
                    
                    if 'top_3' in result:
                        st.markdown("**Top 3 Predicciones:**")
                        for i, crime in enumerate(result['top_3'], 1):
                            st.markdown(f"{i}. {translate_crime(crime['crim'])} - {crime['probabilitat']:.1%}")
    
    else:
        st.info("No tienes predicciones guardadas en tu historial.")
        st.markdown("Realiza predicciones en la sección **Predicción de Crimen** para ver tu historial aquí.")

    # --- SECCIÓN: HISTORIAL DE RUTAS (NUEVO) ---
    st.markdown("---")
    st.markdown("### Historial de Rutas Seguras")
    
    if st.session_state.get("route_history"):
        st.markdown(f"**Rutas calculadas en esta sesión: {len(st.session_state['route_history'])}**")
        
        for idx, hist_route in enumerate(st.session_state["route_history"]):
            with st.expander(f"Ruta {idx+1}: {hist_route.get('origin_name')} -> {hist_route.get('dest_name')}"):
                st.write(f"**Fecha:** {hist_route.get('timestamp')}")
                st.write(f"**Modo:** {hist_route.get('transport_mode')}")
                st.write(f"**Mejor Ruta:** #{hist_route.get('best_idx')+1}")
                st.write(f"**Zonas Peligrosas Detectadas:** {len(hist_route.get('dangerous_zones', []))}")
                
        if st.button("Limpiar Historial de Rutas"):
            st.session_state["route_history"] = []
            st.success("Historial de rutas limpiado")
            st.rerun()
    else:
        st.info("No hay rutas calculadas en esta sesión.")

# --- PAGE: CHATBOT ---
elif page == "Asistente IA":
    st.title("Asistente Virtual Inteligente")
    st.markdown("Pregunta sobre estadísticas, zonas peligrosas o tendencias. Usa comandos para funciones especiales.")
    
    # Inicializar estado del chatbot
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Mensaje de bienvenida
        welcome_msg = (
            "¡Hola! Soy tu asistente de análisis criminal conectado a la API del sistema RAIA.\n\n"
            "Puedo ayudarte con:\n"
            "- Análisis temporal de criminalidad\n"
            "- Comparación entre zonas\n"
            "- Consultas sobre intervalos horarios\n"
            "- Tendencias y estadísticas\n\n"
            "Escribe tu pregunta en lenguaje natural, por ejemplo:\n"
            "- '¿Cuál es la tendencia de robos este año?'\n"
            "- 'Compara Central versus Hollywood'\n"
            "- 'Crímenes entre las 20 y las 23'\n\n"
            "También puedes usar comandos específicos si prefieres."
        )
        st.session_state["messages"].append({"role": "assistant", "content": welcome_msg})
    
    # Mostrar historial de mensajes
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Barra lateral con información contextual
    st.sidebar.markdown("### Contexto del Chatbot")
    st.sidebar.info(f"Mensajes en conversación: {len(st.session_state['messages'])}")
    
    if st.sidebar.button("Limpiar Conversación"):
        st.session_state["messages"] = []
        st.rerun()
    
    # Sugerencias rápidas
    st.sidebar.markdown("### Sugerencias Rápidas")
    suggestions = [
        "¿Cuál es la zona más peligrosa?",
        "Muestra tendencias de este año",
        "Compara Central vs Hollywood",
        "Crímenes entre las 20 y las 23"
    ]
    
    for suggestion in suggestions:
        if st.sidebar.button(suggestion, key=f"sug_{suggestion}"):
            # Añadir sugerencia como mensaje del usuario
            st.session_state["messages"].append({"role": "user", "content": suggestion})
            st.rerun()
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta o comando..."):
        # Añadir mensaje del usuario
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Procesar respuesta - TODO pasa por la API
        with st.chat_message("assistant"):
            with st.spinner("Consultando con la API..."):
                # Siempre llamar a la API (maneja comandos y consultas naturales)
                response = api_chat(prompt, st.session_state["username"], st.session_state["password"])
                
                # Si hay error de conexión, mostrar mensaje claro
                if "Error" in response:
                    response = "Error de conexión con la API. Asegúrate de que el servidor backend esté ejecutándose en `http://127.0.0.1:8000`"
                
                st.markdown(response)
        
        # Guardar respuesta
        st.session_state["messages"].append({"role": "assistant", "content": response})

# --- PAGE: RUTA SEGURA (OSM) ---
elif page == "Navegador Seguro (OSM)":
    st.title("Navegador de Rutas Seguras Avanzado")
    st.markdown("Planifica tu desplazamiento evitando zonas conflictivas con análisis predictivo de IA.")

    # --- IMPORTS LOCALES PARA MAPAS (Ya incluidos globalmente) ---
    
    # Inicializar caché de geocodificación en session_state
    if "geocode_cache" not in st.session_state:
        st.session_state["geocode_cache"] = {}
    
    # Inicializar historial de rutas
    if "route_history" not in st.session_state:
        st.session_state["route_history"] = []

    # --- FUNCIONES DE RUTA CON CACHÉ ---
    def get_lat_lon_cached(address):
        """Convierte dirección en coordenadas usando la API centralizada"""
        # Normalizar la dirección para caché
        address_key = address.lower().strip()
        
        # Verificar caché local (para velocidad inmediata en sesión)
        if address_key in st.session_state["geocode_cache"]:

            return st.session_state["geocode_cache"][address_key]
        
        # Llamar a la API
        try:
            response = requests.get(
                f"{API_URL}/geocode", 
                params={"address": address},
                auth=(st.session_state["username"], st.session_state["password"]),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["found"]:
                    lat, lon = data["lat"], data["lon"]
                    coords = (lat, lon)
                    st.session_state["geocode_cache"][address_key] = coords

                    return coords
                else:
                    st.warning(data.get("error", "No se encontró la dirección."))
                    return None
            else:
                st.error(f"Error API Geocoding: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error de conexión con servicio de mapas: {e}")
            return None

    def get_nearest_area(lat, lon):
        """Determina el área (barrio) más cercana basada en centroides históricos para contexto del modelo"""
        if df_local.empty: return "Central"
        
        # Calcular y cachear centroides si no existen
        if "area_centroids" not in st.session_state:
            try:
                # Agrupar por nombre de área y sacar promedio de lat/lon
                centroids = df_local.groupby("AREA NAME")[["LAT", "LON"]].mean().reset_index()
                st.session_state["area_centroids"] = centroids
            except:
                return "Central"
        
        centroids = st.session_state["area_centroids"]
        if centroids.empty: return "Central"
        
        # Encontrar el más cercano (distancia Euclidiana simple es suficiente)
        # (lat-lat)^2 + (lon-lon)^2
        # Vectorizado para velocidad
        distances = (centroids["LAT"] - lat)**2 + (centroids["LON"] - lon)**2
        closest_idx = distances.idxmin()
        return centroids.iloc[closest_idx]["AREA NAME"]

    def identify_dangerous_zones(start_coords, end_coords, travel_date, travel_hour, username, password, user_age, user_sex):
        """Identifica zonas peligrosas en el área entre origen y destino usando predicciones ML"""
        
        dangerous_zones = []
        
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords
        
        # Validar que ambas coordenadas estén en Los Ángeles
        LA_LAT_MIN, LA_LAT_MAX = 33.7, 34.35
        LA_LON_MIN, LA_LON_MAX = -118.7, -118.0
        
        if not ((LA_LAT_MIN <= lat1 <= LA_LAT_MAX) and (LA_LON_MIN <= lon1 <= LA_LON_MAX)):
            st.error(f"El origen ({lat1:.4f}, {lon1:.4f}) está fuera de Los Ángeles")
            return []
        
        if not ((LA_LAT_MIN <= lat2 <= LA_LAT_MAX) and (LA_LON_MIN <= lon2 <= LA_LON_MAX)):
            st.error(f"El destino ({lat2:.4f}, {lon2:.4f}) está fuera de Los Ángeles")
            return []
        
        # Calcular distancia de la ruta para adaptar densidad del grid
        route_distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        
        # Grid MEJORADO: Aumentamos densidad para mejor detección
        if route_distance > 0.15:  # Ruta larga (>17km aprox)
            grid_size = 10  # 100 puntos - más detalle
        elif route_distance > 0.08:  # Ruta media (>9km)
            grid_size = 8   # 64 puntos
        else:  # Ruta corta
            grid_size = 6   # 36 puntos
        
        # Área de búsqueda REDUCIDA (20% de margen) CON LÍMITES DE LA
        min_lat = max(LA_LAT_MIN, min(lat1, lat2) - abs(lat2 - lat1) * 0.20)
        max_lat = min(LA_LAT_MAX, max(lat1, lat2) + abs(lat2 - lat1) * 0.20)
        min_lon = max(LA_LON_MIN, min(lon1, lon2) - abs(lon2 - lon1) * 0.20)
        max_lon = min(LA_LON_MAX, max(lon1, lon2) + abs(lon2 - lon1) * 0.20)
        
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        total_points = grid_size * grid_size
        st.info(f"Escaneando {total_points} puntos estratégicos en Los Ángeles...")
        progress_bar = st.progress(0, text="Analizando área...")
        
        analyzed_points = 0
        dangerous_count = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                
                payload = {
                    "area": get_nearest_area(lat, lon),
                    "lat": lat,
                    "lon": lon,
                    "date_year": travel_date.year,
                    "date_month": travel_date.month,
                    "day_of_week": travel_date.weekday(),
                    "hour": travel_hour,
                    "victim_age": user_age,
                    "victim_sex": user_sex
                }
                
                try:
                    result = api_predict(payload, username, password)
                    if result:
                        severity = result.get('severity', 'PELIGROSO')
                        severity_conf = result.get('severity_confidence', 0)
                        
                        # Umbral sincronizado con el análisis de ruta (0.55)
                        if severity == "PELIGROSO" and severity_conf > 0.55:
                            dangerous_zones.append({
                                "lat": lat,
                                "lon": lon,
                                "severity_conf": severity_conf,
                                "crime_type": translate_crime(result.get('prediction', 'Desconocido'))[:40]
                            })
                            dangerous_count += 1
                except:
                    pass
                
                analyzed_points += 1
                progress_bar.progress(analyzed_points / total_points, 
                                    text=f"{analyzed_points}/{total_points} • {dangerous_count} zonas")
        
        progress_bar.empty()
        
        if len(dangerous_zones) > 0:
            pass 
        else:
            pass
        
        return dangerous_zones

    def calculate_threat_score(waypoint_lat, waypoint_lon, dangerous_zones):
        """Calcula score de amenaza de un punto basándose en cercanía y severidad de zonas peligrosas"""
        
        if not dangerous_zones:
            return 0.0
        
        total_threat = 0.0
        
        for zone in dangerous_zones:
            distance = math.sqrt((waypoint_lat - zone['lat'])**2 + (waypoint_lon - zone['lon'])**2)
            
            # Normalizar severidad (0.65 a 1.0 -> 0-1)
            severity_normalized = (zone['severity_conf'] - 0.65) / 0.35
            
            # Amenaza: Severidad ponderada por inversa de distancia
            if distance < 0.0001: 
                distance = 0.0001
            
            threat = (severity_normalized * 10) / (distance ** 1.5)
            total_threat += threat
        
        return total_threat
    
    def get_osrm_route_intelligent(start_coords, end_coords, dangerous_zones, transport_mode="Auto"):
        """Genera rutas optimizadas evitando zonas de riesgo detectadas (búsqueda vía API)."""
        
        all_routes = []
        auth = (st.session_state["username"], st.session_state["password"])
        
        # 1. Ruta Directa (Ruta Base)
        try:
            payload = {
                "waypoints": [start_coords, end_coords],
                "mode": transport_mode
            }
            r = requests.post(f"{API_URL}/route", json=payload, auth=auth, timeout=10)
            
            if r.status_code == 200:
                data = r.json()
                if "routes" in data:
                    all_routes.extend(data["routes"])
        except Exception as e:
            print(f"Error ruta directa: {e}")
            pass
        
        # 2. Generación de rutas alternativas inteligentemente si hay riesgos
        if dangerous_zones and len(all_routes) < 6:
            lat1, lon1 = start_coords
            lat2, lon2 = end_coords
            
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
            
            if dist > 0:
                dx = (lat2 - lat1) / dist
                dy = (lon2 - lon1) / dist
                
                perp_dx = -dy
                perp_dy = dx
                
                candidate_waypoints = []
                
                # Calcular waypoints estratégicos para evasión
                for side in [-1, 1]:
                    for deviation in [0.10, 0.15, 0.20, 0.25, 0.30]:
                        offset = dist * deviation
                        wp_lat = mid_lat + (perp_dx * offset * side)
                        wp_lon = mid_lon + (perp_dy * offset * side)
                        
                        threat = calculate_threat_score(wp_lat, wp_lon, dangerous_zones)
                        
                        candidate_waypoints.append({
                            'lat': wp_lat,
                            'lon': wp_lon,
                            'threat': threat
                        })
                
                candidate_waypoints.sort(key=lambda x: x['threat'])
                best_waypoints = candidate_waypoints[:4]
                
                # Generar rutas para los mejores waypoints USANDO LA API
                for wp in best_waypoints:
                    try:
                        # [Start, Waypoint, End]
                        payload = {
                            "waypoints": [[lat1, lon1], [wp['lat'], wp['lon']], [lat2, lon2]],
                            "mode": transport_mode
                        }
                        r = requests.post(f"{API_URL}/route", json=payload, auth=auth, timeout=10)
                        
                        if r.status_code == 200:
                            data = r.json()
                            if "routes" in data and len(data["routes"]) > 0:
                                route = data["routes"][0]
                                if not is_duplicate_route(route, all_routes):
                                    all_routes.append(route)
                                    
                                    if len(all_routes) >= 6:
                                        break
                    except:
                        pass
        
        return all_routes if all_routes else []
    
    def is_duplicate_route(route, existing_routes):
        """Verifica si una ruta ya existe en la lista"""
        route_dist = route.get("distance", 0)
        for existing in existing_routes:
            existing_dist = existing.get("distance", 0)
            if abs(route_dist - existing_dist) < 100:  # 100m tolerancia
                return True
        return False

    def analyze_route_risk_historical(route_geojson, crime_df):
        """Analiza riesgo basado en datos históricos (método antiguo)"""
        risk_points = []
        if not route_geojson or crime_df.empty: return []
        path = route_geojson['coordinates']
        recent_df = crime_df[crime_df['YEAR'] >= crime_df['YEAR'].max() - 1]
        
        for i, point in enumerate(path):
            if i % 10 != 0: continue
            lon, lat = point
            nearby = recent_df[
                (recent_df['LAT'].between(lat - 0.002, lat + 0.002)) & 
                (recent_df['LON'].between(lon - 0.002, lon + 0.002))
            ]
            if len(nearby) > 5:
                risk_points.append({
                    "lat": lat, "lon": lon, 
                    "count": len(nearby), 
                    "desc": translate_crime(nearby['Crm Cd Desc'].mode()[0]) if not nearby.empty else "Varios",
                    "crime_type": translate_crime(nearby['Crm Cd Desc'].mode()[0]) if not nearby.empty else "Varios",
                    "is_dangerous": True,
                    "intensity": 1  # Asumimos riesgo moderado
                })
        return risk_points
    
    def analyze_route_risk_ml(route_geojson, travel_date, travel_hour, username, password, user_age, user_sex):
        """Analiza riesgo usando predicciones ML en tiempo real, personalizado según perfil del usuario"""
        risk_points = []
        if not route_geojson: return []
        
        path = route_geojson['coordinates']
        
        # OPTIMIZADO: Muestreamos cada 20 puntos (antes era cada 10) para velocidad
        sampled_points = [point for i, point in enumerate(path) if i % 20 == 0]
        
        # Limitar a máximo 15 puntos por ruta para velocidad
        if len(sampled_points) > 15:
            step = len(sampled_points) // 15
            sampled_points = sampled_points[::step][:15]
        
        for point in sampled_points:
            lon, lat = point
            
            payload = {
                "area": get_nearest_area(lat, lon),
                "lat": lat,
                "lon": lon,
                "date_year": travel_date.year,
                "date_month": travel_date.month,
                "day_of_week": travel_date.weekday(),
                "hour": travel_hour,
                "victim_age": user_age,
                "victim_sex": user_sex
            }
            
            try:
                result = api_predict(payload, username, password)
                if result:
                    confidence = result.get('confidence', 0)
                    prediction_type = translate_crime(result.get('prediction', 'Desconocido'))
                    severity = result.get('severity', 'PELIGROSO')
                    severity_conf = result.get('severity_confidence', 0)
                    
                    is_dangerous = (severity == "PELIGROSO" and severity_conf > 0.55)
                    
                    # Solo agregar puntos peligrosos para no saturar el mapa
                    if is_dangerous and confidence > 0.20:
                        if severity_conf > 0.75:
                            intensity = 2
                        elif severity_conf > 0.60:
                            intensity = 1
                        else:
                            intensity = 0
                        
                        risk_points.append({
                            "lat": lat,
                            "lon": lon,
                            "count": int(confidence * 100),
                            "desc": f"[RIESGO] {prediction_type} ({confidence:.0%})",
                            "is_dangerous": True,
                            "intensity": intensity,
                            "severity_conf": severity_conf,
                            "crime_type": prediction_type[:40]
                        })
            except:
                continue
        
        return risk_points



    # --- INTERFAZ ---
    st.info("Tip: Escribe lugares de Los Ángeles como 'Union Station', 'Hollywood Boulevard', 'Venice Beach', 'LAX Airport', 'Staples Center', etc.")
    
    c1, c2 = st.columns(2)
    with c1:
        origin = st.text_input("Origen", placeholder="Ej: Union Station")
    with c2:
        dest = st.text_input("Destino", placeholder="Ej: Hollywood Boulevard")
    
    # Selector de fecha y hora del viaje
    st.markdown("### ¿Cuándo planeas viajar?")
    col_date, col_hour = st.columns([2, 1])
    with col_date:
        travel_date = st.date_input("Fecha del viaje", value=datetime.date.today())
    with col_hour:
        travel_hour = st.slider("Hora", 0, 23, datetime.datetime.now().hour, format="%d:00")
    
    # NUEVO: Perfil del viajero y modo de transporte
    st.markdown("### Configuración del Viaje")
    col_age, col_sex, col_mode, col_transport = st.columns(4)
    with col_age:
        user_age = st.number_input("Edad", min_value=10, max_value=100, value=30, step=1)
    with col_sex:
        user_sex = st.selectbox("Sexo", options=["M", "F", "X"], 
                               format_func=lambda x: {"M": "Masculino", "F": "Femenino", "X": "Otro"}[x])
    with col_mode:
        st.markdown("")  # Espacio en blanco para mantener layout
    with col_transport:
        transport_mode = st.selectbox("Transporte", 
                                      options=["Auto", "Caminando"],
                                      help="Modo de transporte (aproximado)",
                                      key="transport_mode_selector")
    
    # Inicializar estado para la ruta si no existe
    if "route_data" not in st.session_state:
        st.session_state["route_data"] = None

    if st.button("Calcular Ruta Segura"):
        if not origin or not dest:
            st.warning("Introduce ambas direcciones.")
        else:
            with st.spinner(f"Analizando ruta para '{transport_mode}' con IA..."):
                start = get_lat_lon_cached(origin)
                end = get_lat_lon_cached(dest)
                
                if start and end:
                    dangerous_zones = identify_dangerous_zones(
                        start, end, travel_date, travel_hour,
                        st.session_state["username"], st.session_state["password"],
                        user_age, user_sex
                    )
                    

                    routes_list = get_osrm_route_intelligent(start, end, dangerous_zones, transport_mode)
                    
                    if routes_list:
                        analyzed_routes = []
                        progress_bar = st.progress(0, text="Analizando seguridad...")
                        
                        for idx, route_obj in enumerate(routes_list):
                            geo = route_obj["geometry"]
                            
                            # Método de análisis siempre ML
                            curr_risks = analyze_route_risk_ml(geo, travel_date, travel_hour, 
                                                              st.session_state["username"], 
                                                              st.session_state["password"],
                                                              user_age, user_sex)
                            
                            curr_count = len(curr_risks)
                            
                            duration = route_obj.get("duration", float('inf'))
                            distance = route_obj.get("distance", float('inf'))
                            
                            analyzed_routes.append({
                                "id": idx,
                                "geo": geo,
                                "risks": curr_risks,
                                "count": curr_count,
                                "duration": duration,
                                "distance": distance
                            })
                            
                            progress_bar.progress((idx + 1) / len(routes_list))
                        
                        progress_bar.empty()
                        
                        analyzed_routes_sorted = sorted(
                            analyzed_routes, 
                            key=lambda r: (r["count"], r["duration"])
                        )
                        
                        best_route = analyzed_routes_sorted[0]
                        best_idx = best_route["id"]
                        
                        if best_route["count"] == 0:
                            st.success(f"Ruta #{best_idx+1} recomendada (Sin riesgos detectados)")
                        else:
                            st.warning(f"Ruta #{best_idx+1} recomendada con precaución ({best_route['count']} zonas de riesgo)")


                        # Guardamos TODAS las rutas en sesión junto con datos del viaje
                        route_data_obj = {
                            "start": start,
                            "end": end,
                            "routes": analyzed_routes,
                            "best_idx": best_idx,
                            "dangerous_zones": dangerous_zones,
                            "origin_name": origin,
                            "dest_name": dest,
                            "travel_date": travel_date.strftime("%Y-%m-%d"),
                            "travel_hour": travel_hour,
                            "transport_mode": transport_mode,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state["route_data"] = route_data_obj
                        
                        # Guardar en historial de rutas
                        st.session_state["route_history"].insert(0, route_data_obj)
                        # Limitar historial a últimas 10 rutas
                        st.session_state["route_history"] = st.session_state["route_history"][:10]
                    else:
                        st.error("No se pudo calcular la ruta con OSRM.")
                else:
                    st.error("No se encontraron las direcciones. Intenta ser más específico.")

    # Si hay datos de ruta guardados, pintamos el mapa
    if st.session_state["route_data"]:
        data = st.session_state["route_data"]
        
        # Verificar coherencia de datos (¿Ha cambiado el usuario algo sin recalcular?)
        current_params_match = (
            data.get("transport_mode") == transport_mode and
            data.get("origin_name") == origin and
            data.get("dest_name") == dest
        )
        
        if not current_params_match:
            st.warning("**Cambios detectados:** Los parámetros de búsqueda (transporte, origen o destino) han cambiado. Haz clic en **'Calcular Ruta Segura'** para actualizar el mapa.")

        
        def create_route_map(route_info):
            start, end = route_info["start"], route_info["end"]
            routes = route_info["routes"]
            best_idx = route_info["best_idx"]
            dangerous_zones = route_info.get("dangerous_zones", [])
            transport_mode = route_info.get("transport_mode", "Auto")
            
            # Crear mapa
            m = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=13)
            
            # 0. Configuración de capas
            
            # 1. Pintar TODAS las rutas con gradiente de color según peligro
            # Calculamos el rango de riesgos para normalizar colores
            risk_counts = [r["count"] for r in routes]
            min_risk = min(risk_counts) if risk_counts else 0
            max_risk = max(risk_counts) if risk_counts else 0
            risk_range = max_risk - min_risk if max_risk > min_risk else 1
            
            for r in routes:
                is_best = (r["id"] == best_idx)
                
                # Sistema de color por nivel de peligro (gradiente)
                if is_best:
                    # La mejor siempre verde brillante
                    color = '#00cc66'
                    weight = 7
                    opacity = 1.0
                else:
                    # Calculamos nivel de peligro normalizado (0.0 = seguro, 1.0 = muy peligroso)
                    danger_level = (r["count"] - min_risk) / risk_range if risk_range > 0 else 0
                    
                    # Gradiente de color: Verde → Amarillo → Naranja → Rojo
                    if danger_level < 0.33:
                        color = '#FFEB3B'  # Amarillo (peligro bajo)
                    elif danger_level < 0.66:
                        color = '#FF9800'  # Naranja (peligro medio)
                    else:
                        color = '#F44336'  # Rojo (peligro alto)
                    
                    weight = 5
                    opacity = 0.75
                
                # CÁLCULO DE TIEMPO Y DISTANCIA PARA VISUALIZACIÓN
                distance_km = r.get("distance", 0) / 1000
                
                # Calcular tiempo (OSRM)
                duration_min = int(r.get("duration", 0) / 60)
                time_str = format_duration(duration_min)
                
                # Tooltip y Popup mejorados
                tooltip_txt = f"Ruta {r['id']+1}: {distance_km:.1f} km • {time_str} • {r['count']} puntos de riesgo" + (" [RECOMENDADA]" if is_best else "")
                
                popup_html = f"""
                <div style='font-family: Arial; font-size: 14px; width: 160px;'>
                    <b>Ruta {r['id']+1}</b> {'(Recomendada)' if is_best else ''}<br>
                    <hr style='margin: 5px 0;'>
                    <b>Tiempo:</b> {time_str}<br>
                    <b>Distancia:</b> {distance_km:.1f} km<br>
                    <b>Puntos de Riesgo:</b> {r['count']}
                </div>
                """
                
                folium.GeoJson(
                    r["geo"], 
                    name=f"Ruta {r['id']+1}",
                    style_function=lambda x, c=color, w=weight, o=opacity: {'color': c, 'weight': w, 'opacity': o},
                    tooltip=tooltip_txt,
                    popup=folium.Popup(popup_html, max_width=200)
                ).add_to(m)
                
                # Pintamos marcadores de riesgo en TODAS las rutas con diferente opacidad
                if r["risks"]:
                    for risk in r["risks"]:
                        # Configuración según nivel de peligrosidad
                        intensity = risk.get('intensity', 0)
                        is_dangerous = risk.get('is_dangerous', False)
                        
                        if is_dangerous:
                            # Gradiente de peligrosidad: rojo intenso → naranja
                            if intensity == 2:  # Muy peligroso
                                color_mk = '#8B0000'  # Rojo oscuro
                                fill_mk = '#FF0000'  # Rojo brillante
                                radius = 9
                                icon = '!'
                            elif intensity == 1:  # Peligroso moderado
                                color_mk = '#CC0000'  # Rojo medio
                                fill_mk = '#FF3333'
                                radius = 7
                                icon = '!'
                            else:  # Peligroso leve (intensity == 0)
                                color_mk = '#FF6600'  # Naranja
                                fill_mk = '#FF8C00'
                                radius = 6
                                icon = '!'
                        else:
                            # Zonas seguras en amarillo suave
                            color_mk = '#FFD700'  # Dorado
                            fill_mk = '#FFEB3B'
                            radius = 5
                            icon = 'i'
                        
                        # Opacidad según si es la ruta recomendada
                        op_mk = 0.9 if is_best else 0.4
                        fill_op_mk = 0.7 if is_best else 0.3
                        
                        # Popup enriquecido con más información
                        popup_html = f"""
                        <div style='font-family: Arial; min-width: 200px;'>
                            <b style='font-size: 14px;'>{icon} Alerta de Seguridad</b><br>
                            <hr style='margin: 5px 0;'>
                            <b>Crimen:</b> {risk.get('crime_type', 'Desconocido')}<br>
                            <b>Confianza:</b> {risk.get('severity_conf', 0)*100:.1f}%<br>
                            <b>Ruta:</b> #{r['id']+1} {'[RECOMENDADA]' if is_best else ''}
                        </div>
                        """
                        
                        folium.CircleMarker(
                            location=[risk['lat'], risk['lon']],
                            radius=radius,
                            color=color_mk,
                            fill=True,
                            fill_color=fill_mk,
                            fill_opacity=fill_op_mk,
                            opacity=op_mk,
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=f"{icon} {risk.get('crime_type', 'Ver detalles')}"
                        ).add_to(m)
            
            # 2. Marcadores Inicio/Fin
            folium.Marker(start, popup="Origen", icon=folium.Icon(color='green', icon='play')).add_to(m)
            folium.Marker(end, popup="Destino", icon=folium.Icon(color='black', icon='stop')).add_to(m)
            
            # 3. Capa de Calor (Contexto)
            heat_data = [[row['LAT'], row['LON']] for index, row in df_local.sample(min(1000, len(df_local))).iterrows()]
            HeatMap(heat_data, radius=10, blur=15, gradient={0.4: 'yellow', 0.65: 'orange', 1: 'red'}).add_to(m)


            
            # 4. Agregar leyenda interactiva
            legend_html = f'''
            <div id="legend_container" style="position: fixed; 
                        bottom: 50px; right: 50px; width: 230px; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid #ccc; border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif; overflow: hidden;">
                
                <div onclick="var content = document.getElementById('legend_content'); content.style.display = (content.style.display == 'none' ? 'block' : 'none');" 
                     style="padding: 8px 10px; background-color: #f8f9fa; cursor: pointer; font-weight: bold; border-bottom: 1px solid #eee; text-align: center; color: #333;">
                     Leyenda de Riesgo ↕
                </div>
                
                <div id="legend_content" style="padding: 10px; display: block; background-color: rgba(255,255,255,0.95);">
                    <p style='margin: 5px 0;'><span style='color: #FF0000; font-size: 18px;'>⬤</span> Muy Peligroso (>75%)</p>
                    <p style='margin: 5px 0;'><span style='color: #FF3333; font-size: 16px;'>⬤</span> Peligroso (60-75%)</p>
                    <p style='margin: 5px 0;'><span style='color: #FF8C00; font-size: 14px;'>⬤</span> Riesgo Leve (55-60%)</p>
                    <p style='margin: 5px 0;'><span style='color: #FFEB3B; font-size: 12px;'>⬤</span> Zona Segura</p>
                    <hr style='margin: 8px 0; border-top: 1px solid #eee;'>
                    <p style='margin: 5px 0; font-size: 12px;'><span style='color: #00cc66; font-weight: bold;'>━━</span> Ruta Recomendada</p>
                    <p style='margin: 5px 0; font-size: 12px;'><span style='color: #FFEB3B;'>━━</span> Alternativas</p>
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m

        m_route = create_route_map(data)

        # Información de ruta seleccionada
        st.markdown(f"### Detalles de la Ruta")
        
        best_r = next(r for r in data["routes"] if r["id"] == data["best_idx"])
        distance_km = best_r.get("distance", 0) / 1000
        # Duración OSRM (precisa para todos los modos)
        duration_min = int(best_r.get("duration", 0) / 60)
        risk_count = best_r.get("count", 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rutas Analizadas", len(data["routes"]))
        with col2:
            st.metric("Ruta Recomendada", f"#{data['best_idx']+1}")
        with col3:
            st.metric("Distancia", f"{distance_km:.1f} km")
        with col4:
            st.metric("Tiempo Estimado", format_duration(duration_min))
        
        # Evaluación de seguridad - PUNTOS IDENTIFICADOS
        
        if risk_count == 0:
            st.success("Ruta Limpia: La ruta seleccionada evita todos los puntos conflictivos detectados.")
        elif risk_count <= 2:
            st.success(f"Ruta Muy Segura: Solo atraviesa {risk_count} punto{'s' if risk_count > 1 else ''} de riesgo identificado. Excelente opción.")
        elif risk_count <= 5:
            st.info(f"Ruta Segura: Atraviesa {risk_count} puntos de riesgo. Opción razonable con precauciones normales.")
        elif risk_count <= 10:
            st.warning(f"Precaución Moderada: La ruta atraviesa {risk_count} puntos calientes. Mantén alerta, especialmente en horario nocturno.")
        else:
            st.error(f"Alto Riesgo: Ruta con {risk_count} puntos peligrosos detectados. Considera viajar en otro horario o elegir otra ruta.")
        
        # Detalles de cada ruta (expandible)
        with st.expander("Ver detalles de todas las rutas"):
            for r in sorted(data["routes"], key=lambda x: x["count"]):
                is_selected = (r["id"] == data["best_idx"])
                
                duration_min = int(r.get("duration", 0) / 60)
                distance_km = r.get("distance", 0) / 1000
                
                # Barra de progreso visual de peligrosidad
                max_risk = max(route["count"] for route in data["routes"])
                danger_pct = (r["count"] / max_risk * 100) if max_risk > 0 else 0
                
                col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
                with col_a:
                    status = "RECOMENDADA" if is_selected else ""
                    st.write(f"Ruta {r['id']+1} {status}")
                with col_b:
                    st.write(f"{r['count']} zonas riesgo")
                with col_c:
                    st.write(format_duration(duration_min))
                with col_d:
                    st.write(f"{distance_km:.1f} km")
                
                # Barra de peligrosidad
                st.progress(danger_pct / 100, text=f"Peligrosidad: {danger_pct:.0f}%")
                st.markdown("---")
        # Mostrar el mapa de la ruta
        st_folium(m_route, width=900, height=500, key="safe_route_map", returned_objects=[])