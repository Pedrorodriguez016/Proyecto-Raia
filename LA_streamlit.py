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

# --- PAGE: DASHBOARD ---
if page == "📊 Dashboard":
    st.title("Estadísticas de Seguridad")
    st.markdown("Visión general de incidentes en Los Ángeles.")
    
    if not df_local.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Registros", f"{len(df_local):,}")
        with c2:
            st.metric("Zonas Monitoreadas", len(areas))
        
        # --- MAPA INTERACTIVO (Folium/OSM) ---
        try:
            import folium
            from streamlit_folium import st_folium
            from folium.plugins import HeatMap
            
            @st.cache_resource
            def create_dashboard_map(data_limit):
                """Crea el mapa base una sola vez para evitar parpadeos."""
                m = folium.Map(location=[34.05, -118.24], zoom_start=10)
                # Datos de calor (usamos una muestra fija para cachear)
                sample_data = df_local.sample(min(data_limit, len(df_local)), random_state=42)
                heat_data = [[row['LAT'], row['LON']] for index, row in sample_data.iterrows()]
                HeatMap(heat_data, radius=13, blur=18).add_to(m)
                return m

            m_dash = create_dashboard_map(2000)
            
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
            st.success("Análisis Completado")
            
            # Guardamos en Cache/Historial vinculando al usuario
            save_to_history(payload, result, area, st.session_state["username"])
            
            col_res, col_chart = st.columns([1, 2])
            with col_res:
                st.markdown("### Resultado Principal")
                st.metric(label="Predicción", value=result['prediction'])
                st.metric("Confianza del Modelo", f"{result['confidence']:.1%}")
                
            with col_chart:
                if 'top_3' in result:
                    top3_df = pd.DataFrame(result['top_3'])
                    chart = alt.Chart(top3_df).mark_bar().encode(
                        x=alt.X('probabilitat', title='Probabilidad'),
                        y=alt.Y('crim', sort='-x', title='Tipo de Crimen'),
                        tooltip=['crim', 'probabilitat']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

# --- PAGE: HISTORIAL ---
elif page == "📜 Historial / Cache":
    st.title("Historial de Predicciones")
    st.markdown("Registro de todas las consultas realizadas (Guardado en `prediction_history.json`).")
    
    history = load_history()
    
    # Filtrar solo el historial del usuario actual
    user_history = [h for h in history if h.get("user") == st.session_state["username"]]
    
    if user_history:
        for idx, item in enumerate(user_history):
            with st.expander(f"📅 {item.get('timestamp', 'N/A')} - {item.get('area', 'Unknown')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Input:**")
                    st.json(item['input'])
                with c2:
                    st.markdown("**Resultado:**")
                    pred = item['result'].get('prediction', 'N/A')
                    conf = item['result'].get('confidence', 0)
                    st.write(f"🏆 {pred} ({conf:.1%})")
                    if 'top_3' in item['result']:
                        st.dataframe(item['result']['top_3'])
    else:
        st.info("No tienes predicciones guardadas en tu historial.")

# --- PAGE: CHATBOT ---
elif page == "💬 Asistente IA":
    st.title("Asistente Virtual")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Pensando..."):
            response = api_chat(prompt, st.session_state["username"], st.session_state["password"])
        
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# --- PAGE: RUTA SEGURA (OSM) ---
elif page == "🗺️ Navegador Seguro (OSM)":
    st.title("🗺️ Navegador de Rutas Seguras")
    st.markdown("Planifica tu desplazamiento evitando zonas conflictivas. Motor: **OpenStreetMap**.")

    # --- IMPORTS LOCALES PARA MAPAS ---
    try:
        import folium
        from streamlit_folium import st_folium
        from geopy.geocoders import Nominatim
        from folium.plugins import HeatMap
    except ImportError:
        st.error("⚠️ Faltan librerías. Por favor instala: `pip install folium streamlit-folium geopy`")
        st.stop()

    # --- FUNCIONES DE RUTA ---
    @st.cache_data
    def get_lat_lon(address):
        """Convierte dirección en coordenadas (Geocoding) con reintentos"""
        # Es importante un User-Agent único según política de OSM
        geolocator = Nominatim(user_agent="raia_navigator_project_v2")
        try:
            # 1. Intento principal: Específico en LA
            loc = geolocator.geocode(f"{address}, Los Angeles, CA", timeout=10)
            if loc: return (loc.latitude, loc.longitude)
            
            # 2. Intento secundario: Búsqueda más libre (por si el usuario ya puso ciudad)
            loc = geolocator.geocode(address, timeout=10)
            # Verificamos que esté restringido geográficamente (aprox) para no irnos a Europa
            if loc and (33.0 < loc.latitude < 35.0) and (-120.0 < loc.longitude < -117.0):
                return (loc.latitude, loc.longitude)
                
            return None
        except Exception as e:
            print(f"Error Geocoding: {e}")
            return None

    def identify_dangerous_zones(start_coords, end_coords, travel_date, travel_hour, username, password, user_age, user_sex):
        """Identifica zonas peligrosas en el área entre origen y destino usando predicciones ML"""
        import math
        
        dangerous_zones = []
        
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords
        
        # Calcular distancia de la ruta para adaptar densidad del grid
        route_distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        
        # Grid adaptativo MEJORADO: cobertura exhaustiva
        if route_distance > 0.15:  # Ruta larga (>17km aprox)
            grid_size = 18  # 324 puntos - cobertura ultra-densa
        elif route_distance > 0.08:  # Ruta media (>9km)
            grid_size = 15  # 225 puntos - cobertura densa
        else:  # Ruta corta
            grid_size = 12  # 144 puntos - cobertura moderada
        
        # Área de búsqueda EXPANDIDA (35% de margen para rutas muy alternativas)
        min_lat = min(lat1, lat2) - abs(lat2 - lat1) * 0.35
        max_lat = max(lat1, lat2) + abs(lat2 - lat1) * 0.35
        min_lon = min(lon1, lon2) - abs(lon2 - lon1) * 0.35
        max_lon = max(lon1, lon2) + abs(lon2 - lon1) * 0.35
        
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        total_points = grid_size * grid_size
        st.info(f"🔍 Fase 1: Escaneando {total_points} puntos con IA ({user_sex}, {user_age} años, {travel_hour}:00)")
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
                        
                        # Umbral más estricto para el escaneo inicial
                        if severity == "PELIGROSO" and severity_conf > 0.65:
                            dangerous_zones.append({
                                "lat": lat,
                                "lon": lon,
                                "severity_conf": severity_conf,
                                "crime_type": result.get('prediction', 'Desconocido')[:30]
                            })
                            dangerous_count += 1
                except:
                    pass
                
                analyzed_points += 1
                progress_bar.progress(analyzed_points / total_points, 
                                    text=f"🔍 {analyzed_points}/{total_points} • {dangerous_count} zonas peligrosas")
        
        progress_bar.empty()
        
        if len(dangerous_zones) > 0:
            st.warning(f"⚠️ Detectadas {len(dangerous_zones)} zonas peligrosas - Generando rutas para evitarlas...")
        else:
            st.success(f"✅ Área relativamente segura - Generando rutas óptimas...")
        
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
        successful_predictions = 0
        failed_predictions = 0
        
        # Muestreamos puntos de la ruta (cada 10 para mejor precisión)
        sampled_points = [point for i, point in enumerate(path) if i % 10 == 0]
        
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
                "victim_age": user_age,  # ✅ Personalizado
                "victim_sex": user_sex   # ✅ Personalizado
            }
            
            try:
                result = api_predict(payload, username, password)
                if result:
                    successful_predictions += 1
                    confidence = result.get('confidence', 0)
                    prediction_type = result.get('prediction', 'Desconocido')
                    
                    # Usar severidad predicha por el modelo ML
                    severity = result.get('severity', 'PELIGROSO')
                    severity_conf = result.get('severity_confidence', 0)
                    
                    # Solo considerar PELIGROSO si el modelo está razonablemente seguro (>55%)
                    is_dangerous = (severity == "PELIGROSO" and severity_conf > 0.55)
                    
                    # Usar umbral dinámico según peligrosidad predicha
                    if is_dangerous:
                        threshold = 0.20  # Más sensible para crímenes peligrosos
                    else:
                        threshold = 0.35  # Menos sensible para crímenes seguros
                    
                    if confidence > threshold:
                        risk_level = "🔴 PELIGROSO" if is_dangerous else "🟡 SEGURO"
                        
                        # Calcular nivel de intensidad (0=bajo, 1=medio, 2=alto)
                        if is_dangerous:
                            if severity_conf > 0.75:
                                intensity = 2  # Muy peligroso
                            elif severity_conf > 0.60:
                                intensity = 1  # Peligroso moderado
                            else:
                                intensity = 0  # Peligroso leve
                        else:
                            intensity = -1  # Seguro
                        
                        risk_points.append({
                            "lat": lat,
                            "lon": lon,
                            "count": int(confidence * 100),
                            "desc": f"{risk_level} {prediction_type} ({confidence:.0%})",
                            "is_dangerous": is_dangerous,
                            "intensity": intensity,
                            "severity_conf": severity_conf,
                            "crime_type": prediction_type[:40]  # Truncamos para popups
                        })
                            
                else:
                    failed_predictions += 1
            except Exception as e:
                failed_predictions += 1
                continue
        
        # Resumen silencioso (solo para logs internos)
        dangerous_count = sum(1 for r in risk_points if r.get("is_dangerous", False))
        safe_count = len(risk_points) - dangerous_count
        
        return risk_points



    # --- INTERFAZ ---
    c1, c2 = st.columns(2)
    with c1:
        origin = st.text_input("Origen", placeholder="Ej: Union Station")
    with c2:
        dest = st.text_input("Destino", placeholder="Ej: Staples Center")
    
    # Selector de fecha y hora del viaje
    st.markdown("### ⏰ ¿Cuándo planeas viajar?")
    col_date, col_hour = st.columns([2, 1])
    with col_date:
        travel_date = st.date_input("Fecha del viaje", value=datetime.date.today())
    with col_hour:
        travel_hour = st.slider("Hora", 0, 23, datetime.datetime.now().hour, format="%d:00")
    
    # NUEVO: Perfil del viajero
    st.markdown("### 👤 Perfil del viajero")
    col_age, col_sex, col_mode = st.columns([1, 1, 2])
    with col_age:
        user_age = st.number_input("Edad", min_value=10, max_value=100, value=30, step=1)
    with col_sex:
        user_sex = st.selectbox("Sexo", options=["M", "F", "X"], 
                               format_func=lambda x: {"M": "👨 Masculino", "F": "👩 Femenino", "X": "⚧ Otro"}[x])
    with col_mode:
        use_ml = st.checkbox("🧠 Usar predicciones ML (recomendado)", value=True, 
                            help="Si está activado, usa el modelo de IA para predecir riesgos futuros personalizados según tu perfil.")


    # Inicializar estado para la ruta si no existe
    if "route_data" not in st.session_state:
        st.session_state["route_data"] = None

    if st.button("🗺️ Calcular Ruta Segura"):
        if not origin or not dest:
            st.warning("Introduce ambas direcciones.")
        else:
            with st.spinner("🧠 Fase 1: Escaneando área con IA para localizar zonas peligrosas..."):
                start = get_lat_lon(origin)
                end = get_lat_lon(dest)
                
                if start and end:
                    # PASO 1: Identificar zonas peligrosas con IA (NUEVO ENFOQUE)
                    dangerous_zones = identify_dangerous_zones(
                        start, end, travel_date, travel_hour,
                        st.session_state["username"], st.session_state["password"],
                        user_age, user_sex
                    )
                    
                    # PASO 2: Generar rutas inteligentes que eviten esas zonas
                    st.info(f"🗺️ Fase 2: Generando rutas inteligentes que eviten las {len(dangerous_zones)} zonas peligrosas...")
                    routes_list = get_osrm_route_intelligent(start, end, dangerous_zones)
                    
                    if routes_list:
                        # --- ANÁLISIS FINO DE CADA RUTA ---
                        analyzed_routes = []
                        
                        # Ahora hacemos un análisis detallado de las rutas ya optimizadas
                        progress_text = f"📊 Fase 3: Análisis detallado de {len(routes_list)} rutas optimizadas..."
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


                        # Guardamos TODAS las rutas en sesión
                        st.session_state["route_data"] = {
                            "start": start,
                            "end": end,
                            "routes": analyzed_routes,
                            "best_idx": best_idx,
                            "dangerous_zones": dangerous_zones  # Guardamos las zonas detectadas
                        }
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

        # Información de rutas analizadas
        st.markdown(f"### 🛣️ Comparativa de Rutas")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rutas Analizadas", len(data["routes"]))
        with col2:
            st.metric("Ruta Recomendada", f"#{data['best_idx']+1}")
        with col3:
            best_r = next(r for r in data["routes"] if r["id"] == data["best_idx"])
            risk_count = best_r["count"]
            st.metric("Zonas de Riesgo", risk_count)
        with col4:
            # Tiempo estimado
            duration_min = int(best_r.get("duration", 0) / 60)
            st.metric("Tiempo Estimado", f"{duration_min} min")
        
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
                if danger_pct > 70:
                    color_bar = "🟥" * int(danger_pct/10)
                elif danger_pct > 40:
                    color_bar = "🟧" * int(danger_pct/10)
                else:
                    color_bar = "🟩" * int(danger_pct/10)
                st.caption(f"Peligrosidad: {color_bar} {danger_pct:.0f}%")

        st_folium(m_route, width=900, height=500, key="safe_route_map", returned_objects=[])