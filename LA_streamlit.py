import streamlit as st
import pandas as pd
import altair as alt
import requests
import json
import os
import time
import datetime

# URL DE TU API (Backend)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAIA - Frontend", layout="wide", page_icon="🚓")

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
        response = requests.post(f"{API_URL}/check-login", auth=(username, password))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con la API. Ejecuta 'python -m uvicorn api:app'")
        return False

def api_register(username, password):
    try:
        payload = {"username": username, "password": password}
        response = requests.post(f"{API_URL}/register", json=payload)
        return response.status_code == 200
    except:
        return False

def api_predict(data_dict, username, password):
    try:
        response = requests.post(f"{API_URL}/predict", json=data_dict, auth=(username, password))
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.warning("⚠️ El modelo aún no está cargado en el servidor.")
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
        response = requests.post(f"{API_URL}/chat", json=payload, auth=(username, password))
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
        st.title("🔒 RAIA SECURE ACCESS")
        st.markdown("Autenticación requerida para acceder al sistema.")
        
        tab1, tab2 = st.tabs(["Ingresar", "Registro"])
        
        with tab1:
            u = st.text_input("Usuario", key="l_user")
            p = st.text_input("Contraseña", type="password", key="l_pass")
            if st.button("🚀 Iniciar Sesión"):
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
            if st.button("📝 Crear Cuenta"):
                if api_register(nu, np):
                    st.success("Usuario creado.")
                else:
                    st.error("Error al crear usuario.")

if not st.session_state["username"]:
    login_screen()
    st.stop()

# ==================== APP PRINCIPAL ====================
st.sidebar.markdown(f"### 👤 {st.session_state['username']}")
if st.sidebar.button("Cerrar Sesión"):
    st.session_state["username"] = None
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Navegación", 
    ["📊 Dashboard", "🧠 Predicción de Crimen", "📜 Historial / Cache", "💬 Asistente IA", "🗺️ Navegador Seguro (OSM)"]
)

# Carga de datos local para visualización
@st.cache_data
def load_viz_data():
    try:
        df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
        df = df.dropna(subset=["LAT", "LON", "AREA NAME"])
        if "DATE OCC" in df.columns: df["YEAR"] = pd.to_datetime(df["DATE OCC"], errors='coerce').dt.year
        return df, sorted(df["AREA NAME"].unique())
    except:
        return pd.DataFrame(), []

df_local, areas = load_viz_data()

# --- FUNCIONES PARA CHATBOT (DEFINIDAS ANTES DE USO) ---
def process_command(command: str, df: pd.DataFrame, areas: list) -> str:
    """Procesa comandos especiales del chatbot."""
    cmd = command.lower().strip()
    
    if cmd == "/help":
        return (
            "## 📚 Comandos Disponibles\n\n"
            "- `/help` - Muestra esta ayuda\n"
            "- `/stats` - Estadísticas generales del dataset\n"
            "- `/zones` - Lista de zonas monitoreadas\n"
            "- `/trends` - Tendencias temporales recientes\n"
            "- `/top` - Top 10 crímenes más frecuentes\n"
            "- `/clear` - Limpia el historial de conversación\n\n"
            "💬 También puedes hacer preguntas como:\n"
            "- '¿Cuál es la zona más peligrosa?'\n"
            "- '¿A qué hora hay más crímenes?'\n"
            "- 'Compara Central vs Hollywood'"
        )
    
    elif cmd == "/stats":
        if df.empty:
            return "⚠️ No hay datos disponibles."
        
        total = len(df)
        zones = len(areas)
        years = len(df['YEAR'].unique()) if 'YEAR' in df.columns else "N/A"
        crimes = len(df['Crm Cd Desc'].unique()) if 'Crm Cd Desc' in df.columns else "N/A"
        
        return (
            f"## 📊 Estadísticas Generales\n\n"
            f"- **Total de registros:** {total:,}\n"
            f"- **Zonas monitoreadas:** {zones}\n"
            f"- **Años de datos:** {years}\n"
            f"- **Tipos de crímenes diferentes:** {crimes}\n\n"
            f"🔍 Usa `/zones` para ver las zonas o `/top` para ver los crímenes más frecuentes."
        )
    
    elif cmd == "/zones":
        if not areas:
            return "⚠️ No hay zonas disponibles."
        
        # Calcular incidentes por zona
        if not df.empty and 'AREA NAME' in df.columns:
            zone_counts = df['AREA NAME'].value_counts().to_dict()
            zones_info = "\n".join([
                f"{i+1}. **{zone}** - {zone_counts.get(zone, 0):,} incidentes" 
                for i, zone in enumerate(sorted(areas)[:15])
            ])
            return f"## 🏙️ Zonas Monitoreadas (Top 15)\n\n{zones_info}\n\n💡 Tip: Usa filtros en el Dashboard para análisis detallado."
        else:
            zones_list = "\n".join([f"{i+1}. {zone}" for i, zone in enumerate(areas[:15])])
            return f"## 🏙️ Zonas Disponibles\n\n{zones_list}"
    
    elif cmd == "/trends":
        if df.empty or 'YEAR' not in df.columns:
            return "⚠️ No hay datos temporales disponibles."
        
        yearly_counts = df.groupby('YEAR').size().sort_index()
        if len(yearly_counts) >= 2:
            last_year = yearly_counts.index[-1]
            last_count = yearly_counts.iloc[-1]
            prev_count = yearly_counts.iloc[-2]
            change = ((last_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
            
            trend_emoji = "📈" if change > 0 else "📉"
            
            return (
                f"## {trend_emoji} Tendencias Recientes\n\n"
                f"- **Año actual ({last_year}):** {last_count:,} incidentes\n"
                f"- **Año anterior:** {prev_count:,} incidentes\n"
                f"- **Cambio:** {change:+.1f}%\n\n"
                f"{'🔺 La criminalidad ha aumentado.' if change > 0 else '✅ La criminalidad ha disminuido.'}\n\n"
                f"📊 Revisa el Dashboard para análisis detallado."
            )
        else:
            return "📊 Datos insuficientes para mostrar tendencias."
    
    elif cmd == "/top":
        if df.empty or 'Crm Cd Desc' not in df.columns:
            return "⚠️ No hay datos de crímenes disponibles."
        
        top_crimes = df['Crm Cd Desc'].value_counts().head(10)
        crimes_list = "\n".join([
            f"{i+1}. **{crime}** - {count:,} casos ({count/len(df)*100:.1f}%)" 
            for i, (crime, count) in enumerate(top_crimes.items())
        ])
        
        return f"## 🔴 Top 10 Crímenes Más Frecuentes\n\n{crimes_list}\n\n🔍 Usa el Dashboard para ver gráficos interactivos."
    
    elif cmd == "/clear":
        return "🗑️ Usa el botón 'Limpiar Conversación' en la barra lateral para reiniciar el chat."
    
    else:
        return f"⚠️ Comando desconocido: `{command}`\n\n💡 Usa `/help` para ver todos los comandos disponibles."


def process_local_query(query: str, df: pd.DataFrame, areas: list) -> str:
    """Procesa consultas en lenguaje natural localmente."""
    query_lower = query.lower()
    
    # Detectar tipo de consulta
    if any(word in query_lower for word in ['zona', 'area', 'peligrosa', 'peligroso', 'peor']):
        if df.empty or 'AREA NAME' not in df.columns:
            return "⚠️ No hay datos de zonas disponibles."
        
        zone_counts = df['AREA NAME'].value_counts()
        worst_zone = zone_counts.index[0]
        worst_count = zone_counts.iloc[0]
        
        return (
            f"🚨 La zona más peligrosa es **{worst_zone}** con {worst_count:,} incidentes registrados.\n\n"
            f"📊 Esto representa el {worst_count/len(df)*100:.1f}% del total de crímenes."
        )
    
    elif any(word in query_lower for word in ['hora', 'tiempo', 'cuando', 'cuándo']):
        if df.empty or 'TIME OCC' not in df.columns:
            return "⚠️ No hay datos horarios disponibles."
        
        df['HOUR'] = (df['TIME OCC'] // 100) % 24
        hourly = df.groupby('HOUR').size().sort_values(ascending=False)
        peak_hour = hourly.index[0]
        peak_count = hourly.iloc[0]
        
        return (
            f"🕐 La hora con más incidentes es las **{peak_hour}:00** con {peak_count:,} casos.\n\n"
            f"⚠️ Se recomienda extremar precauciones durante este horario."
        )
    
    else:
        return (
            "🤔 No pude entender tu pregunta. Intenta:\n\n"
            "- Usar comandos como `/stats`, `/zones`, `/top`\n"
            "- Hacer preguntas más específicas sobre zonas, horarios o tipos de crimen\n"
            "- Usar el botón de sugerencias en la barra lateral"
        )

# --- PAGE: DASHBOARD ---
if page == "📊 Dashboard":
    st.title("Estadísticas de Seguridad")
    st.markdown("Visión general de incidentes en Los Ángeles.")
    
    if not df_local.empty:
        # --- FILTROS DINÁMICOS ---
        st.sidebar.markdown("### 🔍 Filtros del Dashboard")
        
        # Filtro de años
        if 'YEAR' in df_local.columns:
            years_available = sorted(df_local['YEAR'].dropna().unique())
            selected_years = st.sidebar.multiselect(
                "Años", 
                years_available, 
                default=years_available[-3:] if len(years_available) >= 3 else years_available
            )
            if selected_years:
                df_filtered = df_local[df_local['YEAR'].isin(selected_years)]
            else:
                df_filtered = df_local
        else:
            df_filtered = df_local
        
        # Filtro de zonas
        selected_areas = st.sidebar.multiselect(
            "Zonas",
            areas,
            default=areas[:5] if len(areas) > 5 else areas
        )
        if selected_areas:
            df_filtered = df_filtered[df_filtered['AREA NAME'].isin(selected_areas)]
        
        # Filtro por tipo de crimen (si existe)
        if 'Crm Cd Desc' in df_filtered.columns:
            top_crimes = df_filtered['Crm Cd Desc'].value_counts().head(10).index.tolist()
            selected_crimes = st.sidebar.multiselect(
                "Tipos de Crimen",
                top_crimes,
                default=[]
            )
            if selected_crimes:
                df_filtered = df_filtered[df_filtered['Crm Cd Desc'].isin(selected_crimes)]
        
        # --- MÉTRICAS PRINCIPALES ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Registros", f"{len(df_filtered):,}")
        with c2:
            st.metric("Zonas Activas", len(df_filtered['AREA NAME'].unique()))
        with c3:
            if 'YEAR' in df_filtered.columns:
                st.metric("Años Analizados", len(df_filtered['YEAR'].unique()))
            else:
                st.metric("Dataset", "Completo")
        with c4:
            if 'Crm Cd Desc' in df_filtered.columns:
                st.metric("Tipos de Crimen", len(df_filtered['Crm Cd Desc'].unique()))
        
        # --- GRÁFICOS ANALÍTICOS ---
        st.markdown("---")
        st.markdown("### 📈 Análisis Temporal y Geográfico")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Evolución Temporal", "🕐 Distribución Horaria", "🏙️ Comparativa Zonas", "📊 Top Crímenes"])
        
        with tab1:
            # Evolución temporal de crímenes
            if 'YEAR' in df_filtered.columns and 'DATE OCC' in df_filtered.columns:
                df_temp = df_filtered.copy()
                df_temp['YEAR_MONTH'] = pd.to_datetime(df_temp['DATE OCC']).dt.to_period('M').astype(str)
                temporal_counts = df_temp.groupby('YEAR_MONTH').size().reset_index(name='count')
                
                chart_temporal = alt.Chart(temporal_counts).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('YEAR_MONTH:N', title='Mes', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('count:Q', title='Número de Incidentes'),
                    tooltip=['YEAR_MONTH', 'count']
                ).properties(height=400, title='Evolución Mensual de Incidentes')
                
                st.altair_chart(chart_temporal, use_container_width=True)
                
                # Estadísticas de tendencia
                if len(temporal_counts) >= 2:
                    last_count = temporal_counts.iloc[-1]['count']
                    prev_count = temporal_counts.iloc[-2]['count']
                    change_pct = ((last_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                    
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.metric("Último Mes", f"{int(last_count):,}", f"{change_pct:+.1f}%")
                    with col_t2:
                        avg_monthly = temporal_counts['count'].mean()
                        st.metric("Promedio Mensual", f"{int(avg_monthly):,}")
            else:
                st.info("No hay datos temporales disponibles para análisis")
        
        with tab2:
            # Distribución por hora del día
            if 'TIME OCC' in df_filtered.columns:
                df_hour = df_filtered.copy()
                df_hour['HOUR'] = (df_hour['TIME OCC'] // 100) % 24
                hourly_counts = df_hour.groupby('HOUR').size().reset_index(name='count')
                
                chart_hourly = alt.Chart(hourly_counts).mark_bar(color='steelblue').encode(
                    x=alt.X('HOUR:O', title='Hora del Día', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('count:Q', title='Número de Incidentes'),
                    tooltip=['HOUR', 'count']
                ).properties(height=400, title='Distribución de Incidentes por Hora')
                
                st.altair_chart(chart_hourly, use_container_width=True)
                
                # Identificar hora más peligrosa
                most_dangerous_hour = hourly_counts.loc[hourly_counts['count'].idxmax()]
                safest_hour = hourly_counts.loc[hourly_counts['count'].idxmin()]
                
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.error(f"⚠️ Hora más peligrosa: **{int(most_dangerous_hour['HOUR'])}:00** ({int(most_dangerous_hour['count']):,} incidentes)")
                with col_h2:
                    st.success(f"✅ Hora más segura: **{int(safest_hour['HOUR'])}:00** ({int(safest_hour['count']):,} incidentes)")
            else:
                st.info("No hay datos horarios disponibles")
        
        with tab3:
            # Comparativa entre zonas
            if 'AREA NAME' in df_filtered.columns:
                zone_counts = df_filtered['AREA NAME'].value_counts().head(10).reset_index()
                zone_counts.columns = ['Zone', 'Count']
                
                chart_zones = alt.Chart(zone_counts).mark_bar(color='coral').encode(
                    x=alt.X('Count:Q', title='Número de Incidentes'),
                    y=alt.Y('Zone:N', sort='-x', title='Zona'),
                    tooltip=['Zone', 'Count']
                ).properties(height=400, title='Top 10 Zonas con Más Incidentes')
                
                st.altair_chart(chart_zones, use_container_width=True)
                
                # Exportar datos de zonas
                col_e1, col_e2 = st.columns([3, 1])
                with col_e2:
                    csv_zones = zone_counts.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Exportar CSV",
                        data=csv_zones,
                        file_name=f"zonas_peligrosas_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No hay datos de zonas disponibles")
        
        with tab4:
            # Top tipos de crímenes
            if 'Crm Cd Desc' in df_filtered.columns:
                crime_counts = df_filtered['Crm Cd Desc'].value_counts().head(15).reset_index()
                crime_counts.columns = ['Crime_Type', 'Count']
                
                # Traducir nombres de crímenes para el gráfico
                crime_counts['Crime_Type_ES'] = crime_counts['Crime_Type'].apply(translate_crime)
                
                chart_crimes = alt.Chart(crime_counts).mark_bar(color='darkred').encode(
                    x=alt.X('Count:Q', title='Frecuencia'),
                    y=alt.Y('Crime_Type_ES:N', sort='-x', title='Tipo de Crimen'),
                    tooltip=[
                        alt.Tooltip('Crime_Type_ES:N', title='Tipo de Crimen'),
                        alt.Tooltip('Count:Q', title='Frecuencia')
                    ]
                ).properties(height=500, title='Top 15 Tipos de Crímenes Más Frecuentes')
                
                st.altair_chart(chart_crimes, use_container_width=True)
                
                # Estadísticas adicionales
                st.markdown("#### 🔍 Detalles")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    crime_name_es = translate_crime(crime_counts.iloc[0]['Crime_Type'])
                    st.metric("Crimen Más Común", crime_name_es[:30] + "..." if len(crime_name_es) > 30 else crime_name_es)
                with col_c2:
                    st.metric("Frecuencia", f"{int(crime_counts.iloc[0]['Count']):,}")
                with col_c3:
                    pct_top = (crime_counts.iloc[0]['Count'] / len(df_filtered) * 100)
                    st.metric("% del Total", f"{pct_top:.1f}%")
            else:
                st.info("No hay datos de tipos de crimen disponibles")
        
        # --- MAPA INTERACTIVO (Folium/OSM) ---
        st.markdown("---")
        st.markdown("### 🗺️ Mapa de Calor Interactivo")
        try:
            import folium
            from streamlit_folium import st_folium
            from folium.plugins import HeatMap
            
            # Crear mapa con datos filtrados (no cacheable debido a filtros dinámicos)
            def create_dashboard_map(df_data, data_limit):
                """Crea el mapa base con los datos filtrados."""
                m = folium.Map(location=[34.05, -118.24], zoom_start=10, tiles='OpenStreetMap')
                # Datos de calor
                sample_data = df_data.sample(min(data_limit, len(df_data)), random_state=42) if len(df_data) > data_limit else df_data
                heat_data = [[row['LAT'], row['LON']] for index, row in sample_data.iterrows()]
                if heat_data:
                    HeatMap(heat_data, radius=13, blur=18, min_opacity=0.3).add_to(m)
                return m

            m_dash = create_dashboard_map(df_filtered, 2000)
            
            # returned_objects=[] evita que el mapa recargue la página al interactuar
            st_folium(m_dash, width=1200, height=600, key="dashboard_map", returned_objects=[])
            st.caption("Mapa de Calor basado en OpenStreetMap: Las zonas rojas indican mayor concentración de incidentes.")
            
        except ImportError:
            st.warning("Instalando soporte de mapas avanzados...")
            st.map(df_local.sample(500)[["LAT", "LON"]].rename(columns={"LAT":"lat", "LON":"lon"}))

    else:
        st.warning("No hay datos locales disponibles para el mapa.")

# --- PAGE: PREDICCIÓN ---
elif page == "🧠 Predicción de Crimen":
    st.title("Motor de Inteligencia Artificial")
    st.markdown("Introduce los parámetros para estimar la probabilidad de crimen.")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            area = st.selectbox("📍 Zona", areas if areas else ["Central"])
            date = st.date_input("📅 Fecha")
            hour = st.slider("⏰ Hora del día", 0, 23, 12, format="%d:00")
        with c2:
            age = st.slider("👤 Edad potencial víctima", 18, 90, 30)
            sex = st.selectbox("Género", ["M", "F", "X"], format_func=lambda x: "Hombre" if x=="M" else "Mujer" if x=="F" else "Otro")
            
            # Auto-loc
            lat, lon = (34.05, -118.24)
            if not df_local.empty:
                sample = df_local[df_local["AREA NAME"] == area].iloc[0]
                lat, lon = sample["LAT"], sample["LON"]
            st.info(f"Coords auto: {lat:.4f}, {lon:.4f}")

    if st.button("🔮 Generar Predicción", use_container_width=True):
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
            st.success("✅ Análisis Completado")
            
            # Guardamos en Cache/Historial vinculando al usuario
            save_to_history(payload, result, area, st.session_state["username"])
            
            # --- VISUALIZACIÓN MEJORADA CON SEVERIDAD ---
            st.markdown("---")
            
            # Mostrar severidad con colores
            severity = result.get('severity', 'DESCONOCIDO')
            severity_conf = result.get('severity_confidence', 0)
            
            if severity == 'PELIGROSO':
                st.error(f"🔴 **NIVEL DE PELIGRO: {severity}** (Confianza: {severity_conf:.1%})")
                st.markdown(
                    "⚠️ **Este tipo de crimen implica riesgo personal directo.** "
                    "Se recomienda extremar precauciones y evitar la zona en el horario especificado."
                )
            else:
                st.success(f"🟢 **NIVEL DE PELIGRO: {severity}** (Confianza: {severity_conf:.1%})")
                st.markdown(
                    "✅ Este tipo de crimen generalmente no implica contacto personal directo, "
                    "aunque se recomienda mantener precauciones habituales."
                )
            
            # Layout de resultados
            col_res, col_chart = st.columns([1, 2])
            
            with col_res:
                st.markdown("### 🎯 Predicción Principal")
                
                # Métrica con color según severidad
                prediction_text = translate_crime(result['prediction'])
                confidence = result['confidence']
                
                st.metric(
                    label="Tipo de Crimen Predicho", 
                    value=prediction_text[:50] + "..." if len(prediction_text) > 50 else prediction_text
                )
                st.metric("Confianza del Modelo", f"{confidence:.1%}")
                
                # Indicador visual de confianza
                st.progress(confidence)
                
                if confidence > 0.7:
                    st.info("💪 Alta confianza en la predicción")
                elif confidence > 0.5:
                    st.warning("⚠️ Confianza moderada")
                else:
                    st.error("⚠️ Baja confianza - interpretar con cautela")
                
            with col_chart:
                if 'top_3' in result:
                    st.markdown("### 📊 Top 3 Predicciones")
                    
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
                    
                    # Tabla detallada
                    with st.expander("📋 Ver tabla detallada"):
                        top3_df['Probabilidad'] = top3_df['probabilitat'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(
                            top3_df[['crim', 'Probabilidad']].rename(columns={'crim': 'Tipo de Crimen'}),
                            use_container_width=True
                        )
            
            # Información contextual adicional
            st.markdown("---")
            st.markdown("### 📍 Contexto de la Predicción")
            
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                st.metric("📅 Fecha", date.strftime("%d/%m/%Y"))
            with col_c2:
                st.metric("🕐 Hora", f"{hour}:00")
            with col_c3:
                st.metric("🌍 Zona", area)
            with col_c4:
                st.metric("👤 Perfil", f"{sex}, {age} años")
            
            # Recomendaciones personalizadas
            st.markdown("### 💡 Recomendaciones")
            
            recommendations = []
            
            if severity == 'PELIGROSO':
                recommendations.append("🚨 Evitar el área en el horario especificado si es posible")
                recommendations.append("👥 Viajar en grupo si debe transitar por la zona")
                recommendations.append("📱 Mantener el teléfono con batería y contactos de emergencia")
            
            if 6 <= hour <= 10:
                recommendations.append("🌅 Horario matutino - generalmente más seguro")
            elif 18 <= hour <= 23:
                recommendations.append("🌃 Horario nocturno - aumentar precauciones")
            elif 0 <= hour <= 5:
                recommendations.append("🌙 Madrugada - extremar precauciones, zona muy vulnerable")
            
            if age < 25:
                recommendations.append("👦 Perfil joven - manténgase alerta en zonas concurridas")
            elif age > 60:
                recommendations.append("👴 Perfil senior - considere transporte seguro")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Botón para nueva predicción
            if st.button("🔄 Realizar Nueva Predicción"):
                st.rerun()

# --- PAGE: HISTORIAL ---
elif page == "📜 Historial / Cache":
    st.title("📜 Historial de Predicciones")
    st.markdown("Registro de todas las consultas realizadas con funciones de filtrado y exportación.")
    
    history = load_history()
    
    # Filtrar solo el historial del usuario actual
    user_history = [h for h in history if h.get("user") == st.session_state["username"]]
    
    if user_history:
        # Estadísticas del historial
        st.markdown("### 📊 Resumen de tu Actividad")
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        
        with col_h1:
            st.metric("Total Consultas", len(user_history))
        
        with col_h2:
            unique_areas = len(set(h.get('area', 'Unknown') for h in user_history))
            st.metric("Zonas Consultadas", unique_areas)
        
        with col_h3:
            if user_history:
                avg_confidence = sum(h.get('result', {}).get('confidence', 0) for h in user_history) / len(user_history)
                st.metric("Confianza Promedio", f"{avg_confidence:.1%}")
            else:
                st.metric("Confianza Promedio", "N/A")
        
        with col_h4:
            dangerous_count = sum(1 for h in user_history if h.get('result', {}).get('severity') == 'PELIGROSO')
            st.metric("Zonas Peligrosas", dangerous_count)
        
        # Opciones de filtrado
        st.markdown("---")
        st.markdown("### 🔍 Filtrar Historial")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            # Filtro por zona
            all_hist_areas = sorted(set(h.get('area', 'Unknown') for h in user_history))
            selected_hist_area = st.selectbox("Filtrar por Zona", ["Todas"] + all_hist_areas)
        
        with col_f2:
            # Filtro por severidad
            severity_filter = st.selectbox("Filtrar por Severidad", ["Todos", "PELIGROSO", "SEGURO"])
        
        with col_f3:
            # Límite de resultados
            limit_results = st.slider("Mostrar últimos N resultados", 5, len(user_history), min(20, len(user_history)))
        
        # Aplicar filtros
        filtered_history = user_history.copy()
        
        if selected_hist_area != "Todas":
            filtered_history = [h for h in filtered_history if h.get('area') == selected_hist_area]
        
        if severity_filter != "Todos":
            filtered_history = [h for h in filtered_history if h.get('result', {}).get('severity') == severity_filter]
        
        filtered_history = filtered_history[:limit_results]
        
        # Botones de exportación
        st.markdown("---")
        col_e1, col_e2, col_e3 = st.columns([2, 1, 1])
        
        with col_e1:
            st.markdown(f"**Mostrando {len(filtered_history)} de {len(user_history)} consultas**")
        
        with col_e2:
            # Exportar a CSV
            if filtered_history:
                export_data = []
                for item in filtered_history:
                    export_data.append({
                        'Timestamp': item.get('timestamp', 'N/A'),
                        'Usuario': item.get('user', 'N/A'),
                        'Zona': item.get('area', 'N/A'),
                        'Predicción': translate_crime(item.get('result', {}).get('prediction', 'N/A')),
                        'Confianza': item.get('result', {}).get('confidence', 0),
                        'Severidad': item.get('result', {}).get('severity', 'N/A'),
                        'Edad': item.get('input', {}).get('victim_age', 'N/A'),
                        'Sexo': item.get('input', {}).get('victim_sex', 'N/A'),
                        'Hora': item.get('input', {}).get('hour', 'N/A')
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Exportar CSV",
                    data=csv,
                    file_name=f"historial_{st.session_state['username']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col_e3:
            # Exportar a JSON
            json_export = json.dumps(filtered_history, indent=2, ensure_ascii=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar JSON",
                data=json_export,
                file_name=f"historial_{st.session_state['username']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Mostrar historial
        st.markdown("---")
        st.markdown("### 📋 Detalle de Consultas")
        
        for idx, item in enumerate(filtered_history):
            severity = item.get('result', {}).get('severity', 'DESCONOCIDO')
            severity_emoji = "🔴" if severity == "PELIGROSO" else "🟢" if severity == "SEGURO" else "⚪"
            
            with st.expander(f"{severity_emoji} {item.get('timestamp', 'N/A')} - {item.get('area', 'Unknown')} ({idx+1}/{len(filtered_history)})"):
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("**📥 Parámetros de Entrada:**")
                    input_data = item.get('input', {})
                    st.json(input_data)
                
                with col_d2:
                    st.markdown("**📤 Resultado:**")
                    result = item.get('result', {})
                    
                    pred = translate_crime(result.get('prediction', 'N/A'))
                    conf = result.get('confidence', 0)
                    sev = result.get('severity', 'N/A')
                    sev_conf = result.get('severity_confidence', 0)
                    
                    st.markdown(f"**Predicción:** {pred}")
                    st.markdown(f"**Confianza:** {conf:.1%}")
                    st.markdown(f"**Severidad:** {sev} ({sev_conf:.1%})")
                    
                    if 'top_3' in result:
                        st.markdown("**Top 3 Predicciones:**")
                        for i, crime in enumerate(result['top_3'], 1):
                            st.markdown(f"{i}. {translate_crime(crime['crim'])} - {crime['probabilitat']:.1%}")
    
    else:
        st.info("🔍 No tienes predicciones guardadas en tu historial.")
        st.markdown("Realiza predicciones en la sección **🧠 Predicción de Crimen** para ver tu historial aquí.")

# --- PAGE: CHATBOT ---
elif page == "💬 Asistente IA":
    st.title("🤖 Asistente Virtual Inteligente")
    st.markdown("Pregunta sobre estadísticas, zonas peligrosas o tendencias. Usa comandos para funciones especiales.")
    
    # Inicializar estado del chatbot
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Mensaje de bienvenida
        welcome_msg = (
            "¡Hola! Soy tu asistente de análisis criminal conectado a la API del sistema RAIA.\n\n"
            "🔹 **Puedo ayudarte con:**\n"
            "- Análisis temporal de criminalidad\n"
            "- Comparación entre zonas\n"
            "- Consultas sobre intervalos horarios\n"
            "- Tendencias y estadísticas\n\n"
            "💬 Escribe tu pregunta en lenguaje natural, por ejemplo:\n"
            "- '¿Cuál es la tendencia de robos este año?'\n"
            "- 'Compara Central versus Hollywood'\n"
            "- 'Crímenes entre las 20 y las 23'\n\n"
            "🔧 También puedes usar comandos específicos si prefieres."
        )
        st.session_state["messages"].append({"role": "assistant", "content": welcome_msg})
    
    # Mostrar historial de mensajes
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Barra lateral con información contextual
    st.sidebar.markdown("### 📊 Contexto del Chatbot")
    st.sidebar.info(f"Mensajes en conversación: {len(st.session_state['messages'])}")
    
    if st.sidebar.button("🗑️ Limpiar Conversación"):
        st.session_state["messages"] = []
        st.rerun()
    
    # Sugerencias rápidas
    st.sidebar.markdown("### 💡 Sugerencias Rápidas")
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
    if prompt := st.chat_input("✏️ Escribe tu pregunta o comando..."):
        # Añadir mensaje del usuario
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Procesar respuesta - TODO pasa por la API
        with st.chat_message("assistant"):
            with st.spinner("🧠 Consultando con la API..."):
                # Siempre llamar a la API (maneja comandos y consultas naturales)
                response = api_chat(prompt, st.session_state["username"], st.session_state["password"])
                
                # Si hay error de conexión, mostrar mensaje claro
                if "Error" in response:
                    response = "❌ Error de conexión con la API. Asegúrate de que el servidor backend esté ejecutándose en `http://127.0.0.1:8000`"
                
                st.markdown(response)
        
        # Guardar respuesta
        st.session_state["messages"].append({"role": "assistant", "content": response})

# --- PAGE: RUTA SEGURA (OSM) ---
elif page == "🗺️ Navegador Seguro (OSM)":
    st.title("🗺️ Navegador de Rutas Seguras Avanzado")
    st.markdown("Planifica tu desplazamiento evitando zonas conflictivas con análisis predictivo de IA.")

    # --- IMPORTS LOCALES PARA MAPAS ---
    try:
        import folium
        from streamlit_folium import st_folium
        from geopy.geocoders import Nominatim
        from folium.plugins import HeatMap
    except ImportError:
        st.error("⚠️ Faltan librerías. Por favor instala: `pip install folium streamlit-folium geopy`")
        st.stop()
    
    # Inicializar caché de geocodificación en session_state
    if "geocode_cache" not in st.session_state:
        st.session_state["geocode_cache"] = {}
    
    # Inicializar historial de rutas
    if "route_history" not in st.session_state:
        st.session_state["route_history"] = []

    # --- FUNCIONES DE RUTA CON CACHÉ ---
    def get_lat_lon_cached(address):
        """Convierte dirección en coordenadas con caché para evitar llamadas repetidas"""
        # Normalizar la dirección para caché
        address_key = address.lower().strip()
        
        # Verificar caché
        if address_key in st.session_state["geocode_cache"]:
            st.info(f"✅ Usando ubicación en caché: {address}")
            return st.session_state["geocode_cache"][address_key]
        
        # Geocoding nuevo - SIEMPRE en Los Angeles automáticamente
        geolocator = Nominatim(user_agent="raia_navigator_project_v3")
        try:
            # Agregar automáticamente Los Angeles si no está en la búsqueda
            search_query = f"{address}, Los Angeles, California, USA"
            loc = geolocator.geocode(search_query, timeout=10)
            
            # Validar que esté en el área metropolitana de Los Ángeles
            # Límites: Lat 33.7-34.35, Lon -118.7 a -118.0
            if loc:
                lat, lon = loc.latitude, loc.longitude
                
                # Verificar límites estrictos de LA
                if (33.7 <= lat <= 34.35) and (-118.7 <= lon <= -118.0):
                    coords = (lat, lon)
                    st.session_state["geocode_cache"][address_key] = coords
                    st.success(f"📍 Ubicación encontrada: {loc.address}")
                    return coords
                else:
                    st.warning(f"⚠️ '{address}' está fuera de Los Ángeles. Ubicación: ({lat:.4f}, {lon:.4f})")
                    return None
            
            # Si no encuentra nada
            st.error(f"❌ No se encontró '{address}' en Los Ángeles. Intenta con un nombre más específico.")
            return None
            
        except Exception as e:
            st.error(f"Error de geocodificación: {e}")
            return None

    def identify_dangerous_zones(start_coords, end_coords, travel_date, travel_hour, username, password, user_age, user_sex):
        """Identifica zonas peligrosas en el área entre origen y destino usando predicciones ML"""
        import math
        
        dangerous_zones = []
        
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords
        
        # Validar que ambas coordenadas estén en Los Ángeles
        LA_LAT_MIN, LA_LAT_MAX = 33.7, 34.35
        LA_LON_MIN, LA_LON_MAX = -118.7, -118.0
        
        if not ((LA_LAT_MIN <= lat1 <= LA_LAT_MAX) and (LA_LON_MIN <= lon1 <= LA_LON_MAX)):
            st.error(f"❌ El origen ({lat1:.4f}, {lon1:.4f}) está fuera de Los Ángeles")
            return []
        
        if not ((LA_LAT_MIN <= lat2 <= LA_LAT_MAX) and (LA_LON_MIN <= lon2 <= LA_LON_MAX)):
            st.error(f"❌ El destino ({lat2:.4f}, {lon2:.4f}) está fuera de Los Ángeles")
            return []
        
        # Calcular distancia de la ruta para adaptar densidad del grid
        route_distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        
        # Grid OPTIMIZADO: Reducido drásticamente para velocidad
        if route_distance > 0.15:  # Ruta larga (>17km aprox)
            grid_size = 6  # 36 puntos - rápido
        elif route_distance > 0.08:  # Ruta media (>9km)
            grid_size = 5  # 25 puntos - muy rápido
        else:  # Ruta corta
            grid_size = 4  # 16 puntos - ultra rápido
        
        # Área de búsqueda REDUCIDA (20% de margen) CON LÍMITES DE LA
        min_lat = max(LA_LAT_MIN, min(lat1, lat2) - abs(lat2 - lat1) * 0.20)
        max_lat = min(LA_LAT_MAX, max(lat1, lat2) + abs(lat2 - lat1) * 0.20)
        min_lon = max(LA_LON_MIN, min(lon1, lon2) - abs(lon2 - lon1) * 0.20)
        max_lon = min(LA_LON_MAX, max(lon1, lon2) + abs(lon2 - lon1) * 0.20)
        
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        total_points = grid_size * grid_size
        st.info(f"🔍 Escaneando {total_points} puntos estratégicos en Los Ángeles...")
        progress_bar = st.progress(0, text="Analizando área...")
        
        analyzed_points = 0
        dangerous_count = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                
                payload = {
                    "area": "Central",
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
                        
                        # Umbral más permisivo para capturar más zonas
                        if severity == "PELIGROSO" and severity_conf > 0.60:
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
                                    text=f"🔍 {analyzed_points}/{total_points} • {dangerous_count} zonas")
        
        progress_bar.empty()
        
        if len(dangerous_zones) > 0:
            st.success(f"✅ Detectadas {len(dangerous_zones)} zonas peligrosas - Generando rutas...")
        else:
            st.success(f"✅ Área segura - Generando rutas óptimas...")
        
        return dangerous_zones

    def calculate_threat_score(waypoint_lat, waypoint_lon, dangerous_zones):
        """Calcula score de amenaza de un punto basándose en cercanía y severidad de zonas peligrosas"""
        import math
        
        if not dangerous_zones:
            return 0.0
        
        total_threat = 0.0
        
        for zone in dangerous_zones:
            # Distancia euclidiana al punto
            distance = math.sqrt((waypoint_lat - zone['lat'])**2 + (waypoint_lon - zone['lon'])**2)
            
            # Severidad ponderada (0.65 a 1.0 → normalizado a 0-1)
            severity_normalized = (zone['severity_conf'] - 0.65) / 0.35
            
            # Amenaza inversamente proporcional a distancia, ponderada por severidad
            # Zonas muy cercanas (< 0.01 ≈ 1km) y muy severas son muy peligrosas
            if distance < 0.0001:  # Evitar división por cero
                distance = 0.0001
            
            # Fórmula: Amenaza = Severidad / Distancia^2 (decae rápidamente con distancia)
            threat = (severity_normalized * 10) / (distance ** 1.5)
            total_threat += threat
        
        return total_threat
    
    def get_osrm_route_intelligent(start_coords, end_coords, dangerous_zones):
        """Genera rutas EVITANDO zonas peligrosas con algoritmo de optimización por severidad"""
        import math
        
        all_routes = []
        
        # PASO 1: Ruta directa (baseline)
        url_direct = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson&alternatives=true"
        try:
            r = requests.get(url_direct, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if "routes" in data:
                    all_routes.extend(data["routes"])
        except:
            pass
        
        # PASO 2: Algoritmo inteligente con ponderación por severidad
        if dangerous_zones and len(all_routes) < 6:
            lat1, lon1 = start_coords
            lat2, lon2 = end_coords
            
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
            
            if dist > 0:
                dx = (lat2 - lat1) / dist
                dy = (lon2 - lon1) / dist
                
                # Vector perpendicular
                perp_dx = -dy
                perp_dy = dx
                
                # NUEVA LÓGICA: Calcular score de amenaza para cada lado
                # En lugar de contar zonas, ponderamos por severidad y distancia
                
                # Probar varios waypoints y elegir los de menor amenaza
                candidate_waypoints = []
                
                # Generar candidatos en ambos lados con diferentes desvíos
                for side in [-1, 1]:
                    for deviation in [0.10, 0.15, 0.20, 0.25, 0.30]:
                        offset = dist * deviation
                        wp_lat = mid_lat + (perp_dx * offset * side)
                        wp_lon = mid_lon + (perp_dy * offset * side)
                        
                        # Calcular score de amenaza para este waypoint
                        threat = calculate_threat_score(wp_lat, wp_lon, dangerous_zones)
                        
                        candidate_waypoints.append({
                            'lat': wp_lat,
                            'lon': wp_lon,
                            'threat': threat,
                            'deviation': deviation,
                            'side': side
                        })
                
                # Ordenar por menor amenaza
                candidate_waypoints.sort(key=lambda x: x['threat'])
                
                # Mostrar el proceso de optimización
                if len(candidate_waypoints) > 0:
                    safest = candidate_waypoints[0]['threat']
                    worst = candidate_waypoints[-1]['threat']
                    if worst > 0:
                        improvement = ((worst - safest) / worst * 100)
                        st.info(f"🎯 Optimización: Waypoint más seguro tiene {improvement:.0f}% menos amenaza que el peor")
                
                # Tomar los 4 mejores waypoints (más seguros)
                best_waypoints = candidate_waypoints[:4]
                
                # Generar rutas para los mejores waypoints
                for wp in best_waypoints:
                    url_safe = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{wp['lon']},{wp['lat']};{lon2},{lat2}?overview=full&geometries=geojson"
                    
                    try:
                        r = requests.get(url_safe, timeout=5)
                        if r.status_code == 200:
                            data = r.json()
                            if "routes" in data and len(data["routes"]) > 0:
                                route = data["routes"][0]
                                if not is_duplicate_route(route, all_routes):
                                    all_routes.append(route)
                                    
                                    # Limitar a máximo 6 rutas
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
                    "count": len(nearby), "desc": nearby['Crm Cd Desc'].mode()[0] if not nearby.empty else "Varios"
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
                "area": "Central",
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
                            "desc": f"🔴 {prediction_type} ({confidence:.0%})",
                            "is_dangerous": True,
                            "intensity": intensity,
                            "severity_conf": severity_conf,
                            "crime_type": prediction_type[:40]
                        })
            except:
                continue
        
        return risk_points



    # --- INTERFAZ ---
    st.info("💡 **Tip**: Escribe lugares de Los Ángeles como 'Union Station', 'Hollywood Boulevard', 'Venice Beach', 'LAX Airport', 'Staples Center', etc.")
    
    c1, c2 = st.columns(2)
    with c1:
        origin = st.text_input("Origen", placeholder="Ej: Union Station")
    with c2:
        dest = st.text_input("Destino", placeholder="Ej: Hollywood Boulevard")
    
    # Selector de fecha y hora del viaje
    st.markdown("### ⏰ ¿Cuándo planeas viajar?")
    col_date, col_hour = st.columns([2, 1])
    with col_date:
        travel_date = st.date_input("Fecha del viaje", value=datetime.date.today())
    with col_hour:
        travel_hour = st.slider("Hora", 0, 23, datetime.datetime.now().hour, format="%d:00")
    
    # NUEVO: Perfil del viajero y modo de transporte
    st.markdown("### 👤 Configuración del Viaje")
    col_age, col_sex, col_mode, col_transport = st.columns(4)
    with col_age:
        user_age = st.number_input("Edad", min_value=10, max_value=100, value=30, step=1)
    with col_sex:
        user_sex = st.selectbox("Sexo", options=["M", "F", "X"], 
                               format_func=lambda x: {"M": "👨 Masculino", "F": "👩 Femenino", "X": "⚧ Otro"}[x])
    with col_mode:
        use_ml = st.checkbox("🧠 Usar IA", value=True, 
                            help="Predicciones ML personalizadas")
    with col_transport:
        transport_mode = st.selectbox("🚗 Transporte", 
                                      options=["Auto", "Caminando", "Bicicleta"],
                                      help="Modo de transporte (aproximado)")
    
    # Mostrar caché de geocodificación
    if st.session_state.get("geocode_cache"):
        with st.expander("📍 Ubicaciones en Caché"):
            cache_df = pd.DataFrame([
                {"Dirección": addr, "Lat": coords[0], "Lon": coords[1]}
                for addr, coords in st.session_state["geocode_cache"].items()
            ])
            st.dataframe(cache_df, use_container_width=True)
            if st.button("🗑️ Limpiar Caché de Geocodificación"):
                st.session_state["geocode_cache"] = {}
                st.success("Caché limpiado")


    # Inicializar estado para la ruta si no existe
    if "route_data" not in st.session_state:
        st.session_state["route_data"] = None

    if st.button("🗺️ Calcular Ruta Segura"):
        if not origin or not dest:
            st.warning("Introduce ambas direcciones.")
        else:
            with st.spinner("🧠 Analizando área con IA (optimizado para velocidad)..."):
                start = get_lat_lon_cached(origin)
                end = get_lat_lon_cached(dest)
                
                if start and end:
                    # PASO 1: Identificar zonas peligrosas con IA (NUEVO ENFOQUE)
                    dangerous_zones = identify_dangerous_zones(
                        start, end, travel_date, travel_hour,
                        st.session_state["username"], st.session_state["password"],
                        user_age, user_sex
                    )
                    
                    # PASO 2: Generar rutas inteligentes que eviten esas zonas
                    st.info(f"🗺️ Generando rutas optimizadas...")
                    routes_list = get_osrm_route_intelligent(start, end, dangerous_zones)
                    
                    if routes_list:
                        # --- ANÁLISIS FINO DE CADA RUTA ---
                        analyzed_routes = []
                        
                        # Análisis rápido de rutas ya optimizadas
                        progress_text = f"📊 Analizando {len(routes_list)} rutas..."
                        progress_bar = st.progress(0, text=progress_text)
                        
                        for idx, route_obj in enumerate(routes_list):
                            geo = route_obj["geometry"]
                            
                            # Elegimos método de análisis según preferencia del usuario
                            if use_ml:
                                curr_risks = analyze_route_risk_ml(geo, travel_date, travel_hour, 
                                                                  st.session_state["username"], 
                                                                  st.session_state["password"],
                                                                  user_age, user_sex)
                            else:
                                curr_risks = analyze_route_risk_historical(geo, df_local)
                            
                            curr_count = len(curr_risks)
                            
                            # Extraemos datos de OSRM (duración en segundos, distancia en metros)
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
                            
                            # Actualizar barra de progreso
                            progress_bar.progress((idx + 1) / len(routes_list), 
                                                text=f"Ruta {idx+1}/{len(routes_list)}: {curr_count} riesgos detectados")
                        
                        progress_bar.empty()  # Limpiamos la barra
                        
                        # Ordenamos: Primero por menos riesgos, luego por menor duración
                        analyzed_routes_sorted = sorted(
                            analyzed_routes, 
                            key=lambda r: (r["count"], r["duration"])
                        )
                        
                        # La primera del ranking ordenado es la mejor
                        best_route = analyzed_routes_sorted[0]
                        best_idx = best_route["id"]
                        min_risk_count = best_route["count"]
                        
                        # Estadísticas comparativas
                        worst_risk_count = analyzed_routes_sorted[-1]["count"]
                        avg_risk_count = sum(r["count"] for r in analyzed_routes) / len(analyzed_routes)
                        
                        # Mensaje inteligente según resultados
                        if min_risk_count == 0:
                            st.success(f"🎉 ¡Excelente! Ruta #{best_idx+1} completamente limpia. Se analizaron {len(routes_list)} alternativas.")
                        elif min_risk_count < avg_risk_count:
                            improvement = ((avg_risk_count - min_risk_count) / avg_risk_count * 100)
                            st.success(f"✅ Ruta #{best_idx+1} recomendada: {min_risk_count} zonas de riesgo ({improvement:.0f}% mejor que el promedio)")
                        else:
                            st.warning(f"⚠️ Ruta #{best_idx+1} es la menos mala: {min_risk_count} zonas de riesgo (de {len(routes_list)} opciones analizadas)")


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
        
        def create_route_map(route_info):
            start, end = route_info["start"], route_info["end"]
            routes = route_info["routes"]
            best_idx = route_info["best_idx"]
            dangerous_zones = route_info.get("dangerous_zones", [])
            
            # Crear mapa
            m = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=13)
            
            # 0. Marcar zonas peligrosas detectadas en el escaneo inicial
            for zone in dangerous_zones:
                folium.Marker(
                    location=[zone['lat'], zone['lon']],
                    icon=folium.Icon(color='red', icon='times', prefix='fa'),
                    popup=f"⚠️ ZONA PELIGROSA<br>{zone['crime_type']}<br>Confianza: {zone['severity_conf']*100:.0f}%",
                    tooltip="Zona peligrosa detectada"
                ).add_to(m)
            
            # 1. Pintar TODAS las rutas con gradiente de color según peligro
            # Calculamos el rango de riesgos para normalizar colores
            risk_counts = [r["count"] for r in routes]
            min_risk = min(risk_counts)
            max_risk = max(risk_counts)
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
                
                tooltip_txt = f"Ruta {r['id']+1}: {r['count']} zonas de riesgo" + (" ✅ RECOMENDADA" if is_best else "")
                
                folium.GeoJson(
                    r["geo"], 
                    name=f"Ruta {r['id']+1}",
                    style_function=lambda x, c=color, w=weight, o=opacity: {'color': c, 'weight': w, 'opacity': o},
                    tooltip=tooltip_txt
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
                                color = '#8B0000'  # Rojo oscuro
                                fill_color = '#FF0000'  # Rojo brillante
                                radius = 9
                                icon = '⚠️'
                            elif intensity == 1:  # Peligroso moderado
                                color = '#CC0000'  # Rojo medio
                                fill_color = '#FF3333'
                                radius = 7
                                icon = '⚠️'
                            else:  # Peligroso leve (intensity == 0)
                                color = '#FF6600'  # Naranja
                                fill_color = '#FF8C00'
                                radius = 6
                                icon = '⚠️'
                        else:
                            # Zonas seguras en amarillo suave
                            color = '#FFD700'  # Dorado
                            fill_color = '#FFEB3B'
                            radius = 5
                            icon = 'ℹ️'
                        
                        # Opacidad según si es la ruta recomendada
                        opacity = 0.9 if is_best else 0.4
                        fill_opacity = 0.7 if is_best else 0.3
                        
                        # Popup enriquecido con más información
                        popup_html = f"""
                        <div style='font-family: Arial; min-width: 200px;'>
                            <b style='font-size: 14px;'>{icon} {risk['desc'].split()[0]} {risk['desc'].split()[1]}</b><br>
                            <hr style='margin: 5px 0;'>
                            <b>Crimen:</b> {risk.get('crime_type', 'Desconocido')}<br>
                            <b>Confianza:</b> {risk.get('severity_conf', 0)*100:.1f}%<br>
                            <b>Ruta:</b> #{r['id']+1} {'✅ Recomendada' if is_best else ''}
                        </div>
                        """
                        
                        folium.CircleMarker(
                            location=[risk['lat'], risk['lon']],
                            radius=radius,
                            color=color,
                            fill=True,
                            fill_color=fill_color,
                            fill_opacity=fill_opacity,
                            opacity=opacity,
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
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 240px; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid grey; border-radius: 10px; padding: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <h4 style='margin: 0 0 10px 0; text-align: center;'>📊 Leyenda de Riesgo</h4>
                <p style='margin: 5px 0; font-size: 12px;'><span style='color: red;'>📍 ✖</span> Zonas Peligrosas ({len(dangerous_zones)} detectadas)</p>
                <hr style='margin: 8px 0;'>
                <p style='margin: 5px 0;'><span style='color: #FF0000; font-size: 18px;'>⬤</span> Muy Peligroso (>75%)</p>
                <p style='margin: 5px 0;'><span style='color: #FF3333; font-size: 16px;'>⬤</span> Peligroso (60-75%)</p>
                <p style='margin: 5px 0;'><span style='color: #FF8C00; font-size: 14px;'>⬤</span> Riesgo Leve (55-60%)</p>
                <p style='margin: 5px 0;'><span style='color: #FFEB3B; font-size: 12px;'>⬤</span> Zona Segura</p>
                <hr style='margin: 10px 0;'>
                <p style='margin: 5px 0; font-size: 12px;'><span style='color: #00cc66; font-weight: bold;'>━━</span> Ruta Recomendada</p>
                <p style='margin: 5px 0; font-size: 12px;'><span style='color: #FFEB3B;'>━━</span> Alternativas</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m

        m_route = create_route_map(data)

        # Información de rutas analizadas con CÁLCULO DE COSTES
        st.markdown(f"### 🛣️ Comparativa de Rutas y Costes")
        
        best_r = next(r for r in data["routes"] if r["id"] == data["best_idx"])
        distance_km = best_r.get("distance", 0) / 1000
        duration_min = int(best_r.get("duration", 0) / 60)
        risk_count = best_r["count"]
        
        # Calcular costes estimados según modo de transporte
        transport_mode = data.get("transport_mode", "Auto")
        
        if transport_mode == "Auto":
            # Promedio: $0.40/km (combustible + desgaste)
            cost_usd = distance_km * 0.40
            co2_kg = distance_km * 0.12  # 120g CO2/km promedio
            cost_symbol = "💵"
        elif transport_mode == "Bicicleta":
            cost_usd = 0
            co2_kg = 0
            cost_symbol = "🚴"
        else:  # Caminando
            cost_usd = 0
            co2_kg = 0
            cost_symbol = "🚶"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Rutas Analizadas", len(data["routes"]))
        with col2:
            st.metric("Ruta Recomendada", f"#{data['best_idx']+1}")
        with col3:
            st.metric("Distancia", f"{distance_km:.1f} km")
        with col4:
            st.metric("Tiempo Estimado", f"{duration_min} min")
        with col5:
            if cost_usd > 0:
                st.metric(f"{cost_symbol} Coste Aprox.", f"${cost_usd:.2f}")
            else:
                st.metric(f"{cost_symbol} Coste", "Gratis")
        
        # Impacto ambiental (si aplica)
        if co2_kg > 0:
            st.info(f"🌱 Huella de carbono estimada: {co2_kg:.2f} kg CO₂")
        
        # Evaluación de seguridad
        total_dangerous_zones = len(data.get("dangerous_zones", []))
        if risk_count == 0:
            st.success("✅ **Ruta Limpia**: La ruta seleccionada evita todas las zonas conflictivas detectadas.")
        elif risk_count < total_dangerous_zones * 0.3:
            st.info(f"ℹ️ **Ruta Segura**: Solo atraviesa {risk_count} de {total_dangerous_zones} zonas detectadas como peligrosas.")
        elif risk_count < total_dangerous_zones * 0.6:
            st.warning(f"⚠️ **Precaución**: La ruta atraviesa {risk_count} zonas calientes. Mantén alerta.")
        else:
            st.error(f"🚨 **Alto Riesgo**: Ruta con {risk_count} zonas peligrosas. Considera viajar en otro horario o usar transporte público.")
        
        # Detalles de cada ruta (expandible)
        with st.expander("📋 Ver detalles de todas las rutas"):
            for r in sorted(data["routes"], key=lambda x: x["count"]):
                is_selected = (r["id"] == data["best_idx"])
                emoji = "✅" if is_selected else "⚪"
                
                # Calcular métricas
                duration_min = int(r.get("duration", 0) / 60)
                distance_km = r.get("distance", 0) / 1000
                
                # Barra de progreso visual de peligrosidad
                max_risk = max(route["count"] for route in data["routes"])
                danger_pct = (r["count"] / max_risk * 100) if max_risk > 0 else 0
                
                col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
                with col_a:
                    status = "🏆 RECOMENDADA" if is_selected else ""
                    st.write(f"{emoji} **Ruta {r['id']+1}** {status}")
                with col_b:
                    st.write(f"🚨 {r['count']} zonas riesgo")
                with col_c:
                    st.write(f"⏱️ {duration_min} min")
                with col_d:
                    st.write(f"📏 {distance_km:.1f} km")
                
                # Barra de peligrosidad
                st.progress(danger_pct / 100, text=f"Peligrosidad: {danger_pct:.0f}%")
                st.markdown("---")
        
        # Opciones de compartir y exportar
        st.markdown("---")
        st.markdown("### 📤 Compartir y Exportar Ruta")
        
        col_share1, col_share2, col_share3 = st.columns(3)
        
        with col_share1:
            # Exportar resumen de ruta a JSON
            route_summary = {
                "origen": data.get("origin_name", "N/A"),
                "destino": data.get("dest_name", "N/A"),
                "fecha": data.get("travel_date", "N/A"),
                "hora": data.get("travel_hour", "N/A"),
                "ruta_recomendada": data["best_idx"] + 1,
                "distancia_km": round(distance_km, 2),
                "duracion_min": duration_min,
                "zonas_riesgo": risk_count,
                "coste_estimado_usd": round(cost_usd, 2) if cost_usd > 0 else 0,
                "modo_transporte": transport_mode
            }
            
            json_route = json.dumps(route_summary, indent=2, ensure_ascii=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Ruta (JSON)",
                data=json_route,
                file_name=f"ruta_segura_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_share2:
            # Crear texto para compartir
            share_text = f"""🗺️ RAIA - Ruta Segura Calculada

📍 {data.get('origin_name', 'Origen')} ➡️ {data.get('dest_name', 'Destino')}
📅 {data.get('travel_date', 'N/A')} a las {data.get('travel_hour', 'N/A')}:00

✅ Ruta #{data['best_idx']+1} (Recomendada)
📏 Distancia: {distance_km:.1f} km
⏱️ Tiempo: {duration_min} min
🚨 Zonas de riesgo: {risk_count}
{f'💵 Coste estimado: ${cost_usd:.2f}' if cost_usd > 0 else '💚 Coste: Gratis'}

Generado por RAIA - Sistema de Rutas Seguras con IA
"""
            
            st.download_button(
                label="📄 Compartir Resumen (TXT)",
                data=share_text.encode('utf-8'),
                file_name=f"ruta_raia_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col_share3:
            # Generar link para Google Maps (como referencia)
            gmaps_url = f"https://www.google.com/maps/dir/{data['start'][0]},{data['start'][1]}/{data['end'][0]},{data['end'][1]}"
            st.markdown(f"[🔗 Abrir en Google Maps]({gmaps_url})")
            st.caption("⚠️ Google Maps no incluye análisis de seguridad")

        st.markdown("---")
        
        # Historial de rutas calculadas
        if st.session_state.get("route_history") and len(st.session_state["route_history"]) > 1:
            with st.expander(f"📜 Historial de Rutas ({len(st.session_state['route_history'])} calculadas)"):
                st.markdown("Rutas calculadas recientemente en esta sesión:")
                
                for idx, hist_route in enumerate(st.session_state["route_history"]):
                    st.markdown(f"**{idx+1}. {hist_route.get('origin_name')} → {hist_route.get('dest_name')}**")
                    st.caption(f"🕐 {hist_route.get('timestamp')} | 🚗 {hist_route.get('transport_mode')}")
                    st.markdown("---")
                
                if st.button("🗑️ Limpiar Historial de Rutas"):
                    st.session_state["route_history"] = []
                    st.success("Historial limpiado")
                    st.rerun()
        
        # Mostrar el mapa de la ruta
        st_folium(m_route, width=900, height=500, key="safe_route_map", returned_objects=[])