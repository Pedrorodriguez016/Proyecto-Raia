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

    def get_osrm_route(start_coords, end_coords):
        """Genera rutas alternativas coherentes usando OSRM + waypoints inteligentes"""
        import math
        
        all_routes = []
        
        # PASO 1: Intentar obtener alternativas reales de OSRM
        url_alternatives = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson&alternatives=true"
        try:
            r = requests.get(url_alternatives, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if "routes" in data:
                    all_routes.extend(data["routes"])
        except:
            pass
        
        # PASO 2: Si tenemos menos de 3 rutas, generamos alternativas inteligentes
        if len(all_routes) < 3:
            # Calculamos vector directorio y perpendiculares
            lat1, lon1 = start_coords
            lat2, lon2 = end_coords
            
            # Punto medio
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            
            # Distancia entre puntos (aproximada)
            dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
            
            # Vector normalizado de origen a destino
            if dist > 0:
                dx = (lat2 - lat1) / dist
                dy = (lon2 - lon1) / dist
                
                # Vector perpendicular (rotación 90°)
                perp_dx = -dy
                perp_dy = dx
                
                # Generamos waypoints a diferentes distancias perpendiculares
                # Pequeño desvío (10%), mediano (20%), grande (25% de la distancia)
                deviations = [0.10, 0.20, 0.25]
                
                for deviation in deviations:
                    offset = dist * deviation
                    
                    # Probamos ambos lados de la línea (izquierda y derecha)
                    for side in [-1, 1]:
                        waypoint_lat = mid_lat + (perp_dx * offset * side)
                        waypoint_lon = mid_lon + (perp_dy * offset * side)
                        
                        # Pedimos ruta pasando por este waypoint
                        url_via = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{waypoint_lon},{waypoint_lat};{lon2},{lat2}?overview=full&geometries=geojson"
                        
                        try:
                            r = requests.get(url_via, timeout=5)
                            if r.status_code == 200:
                                data = r.json()
                                if "routes" in data and len(data["routes"]) > 0:
                                    route = data["routes"][0]
                                    # Evitamos duplicados comparando distancias aproximadas
                                    is_duplicate = False
                                    route_dist = route.get("distance", 0)
                                    for existing in all_routes:
                                        existing_dist = existing.get("distance", 0)
                                        if abs(route_dist - existing_dist) < 100: # 100m de tolerancia
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        all_routes.append(route)
                                        
                                    # Limitamos a máximo 6 rutas para no saturar
                                    if len(all_routes) >= 6:
                                        return all_routes
                        except:
                            continue
        
        return all_routes if all_routes else []

    def analyze_route_risk(route_geojson, crime_df):
        """Analiza si la ruta pasa por zonas calientes."""
        risk_points = []
        if not route_geojson or crime_df.empty: return []

        # Extraemos puntos de la ruta (muestreo simple)
        path = route_geojson['coordinates'] # [[lon, lat], ...]
        # Solo usamos crímenes recientes (último año disponible) para ser justos
        recent_df = crime_df[crime_df['YEAR'] >= crime_df['YEAR'].max() - 1]
        
        # Algoritmo simplificado: Checkeo de proximidad
        # En una app real usaríamos KDTree, aquí filtro por caja para velocidad
        for i, point in enumerate(path):
            if i % 10 != 0: continue # Muestreamos cada 10 puntos para velocidad
            lon, lat = point
            
            # Filtro caja rápida (0.002 grados ~ 200 metros)
            nearby = recent_df[
                (recent_df['LAT'].between(lat - 0.002, lat + 0.002)) & 
                (recent_df['LON'].between(lon - 0.002, lon + 0.002))
            ]
            
            if len(nearby) > 5: # UMBRAL DE PELIGRO
                risk_points.append({
                    "lat": lat, "lon": lon, 
                    "count": len(nearby), "desc": nearby['Crm Cd Desc'].mode()[0] if not nearby.empty else "Varios"
                })
        return risk_points

    # --- INTERFAZ ---
    c1, c2 = st.columns(2)
    with c1:
        origin = st.text_input("Origen", placeholder="Ej: Union Station")
    with c2:
        dest = st.text_input("Destino", placeholder="Ej: Staples Center")

    # Inicializar estado para la ruta si no existe
    if "route_data" not in st.session_state:
        st.session_state["route_data"] = None

    if st.button("🗺️ Calcular Ruta Segura"):
        if not origin or not dest:
            st.warning("Introduce ambas direcciones.")
        else:
            with st.spinner("Analizando alternativas para encontrar la ruta más segura..."):
                start = get_lat_lon(origin)
                end = get_lat_lon(dest)
                
                if start and end:
                    routes_list = get_osrm_route(start, end)
                    
                    if routes_list:
                        # --- LÓGICA DE SEGURIDAD MEJORADA ---
                        analyzed_routes = []
                        
                        # Iteramos todas las rutas alternativas
                        for idx, route_obj in enumerate(routes_list):
                            geo = route_obj["geometry"]
                            curr_risks = analyze_route_risk(geo, df_local)
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
                        
                        # Ordenamos: Primero por menos riesgos, luego por menor duración
                        analyzed_routes_sorted = sorted(
                            analyzed_routes, 
                            key=lambda r: (r["count"], r["duration"])
                        )
                        
                        # La primera del ranking ordenado es la mejor
                        best_route = analyzed_routes_sorted[0]
                        best_idx = best_route["id"]
                        min_risk_count = best_route["count"]
                        
                        st.success(f"Se analizaron {len(routes_list)} rutas. Ruta #{best_idx+1} recomendada ({min_risk_count} zonas de riesgo).")

                        # Guardamos TODAS las rutas en sesión
                        st.session_state["route_data"] = {
                            "start": start,
                            "end": end,
                            "routes": analyzed_routes,
                            "best_idx": best_idx
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
            
            # Crear mapa
            m = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=13)
            
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
                
                # Solo pintamos los marcadores de riesgo en la ruta seleccionada para no saturar
                if is_best and r["risks"]:
                    for risk in r["risks"]:
                         folium.CircleMarker(
                            location=[risk['lat'], risk['lon']],
                            radius=6, color='red', fill=True, fill_color='#cc0000', fill_opacity=0.8,
                            popup=f"⚠️ Riesgo: {risk['desc']}"
                        ).add_to(m)
            
            # 2. Marcadores Inicio/Fin
            folium.Marker(start, popup="Origen", icon=folium.Icon(color='green', icon='play')).add_to(m)
            folium.Marker(end, popup="Destino", icon=folium.Icon(color='black', icon='stop')).add_to(m)
            
            # 3. Capa de Calor (Contexto)
            heat_data = [[row['LAT'], row['LON']] for index, row in df_local.sample(min(1000, len(df_local))).iterrows()]
            HeatMap(heat_data, radius=10, blur=15, gradient={0.4: 'yellow', 0.65: 'orange', 1: 'red'}).add_to(m)
            
            return m

        m_route = create_route_map(data)

        # Información de rutas analizadas
        st.markdown(f"### 🛣️ Comparativa de Rutas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rutas Analizadas", len(data["routes"]))
        with col2:
            st.metric("Ruta Recomendada", f"#{data['best_idx']+1}")
        with col3:
            best_r = next(r for r in data["routes"] if r["id"] == data["best_idx"])
            risk_count = best_r["count"]
            st.metric("Zonas de Riesgo", risk_count, delta=None if risk_count == 0 else "⚠️")
        
        # Resumen final
        if risk_count > 0:
            st.warning(f"⚠️ Atención: La ruta recomendada atraviesa {risk_count} zonas calientes. Las alternativas eran peores.")
        else:
            st.success("✅ Ruta Limpia: La ruta seleccionada evita todas las zonas conflictivas detectadas.")
        
        # Detalles de cada ruta (expandible)
        with st.expander("📋 Ver detalles de todas las rutas"):
            for r in data["routes"]:
                is_selected = (r["id"] == data["best_idx"])
                emoji = "✅" if is_selected else "⚪"
                st.write(f"{emoji} **Ruta {r['id']+1}**: {r['count']} zonas de riesgo" + (" **(RECOMENDADA)**" if is_selected else ""))

        st_folium(m_route, width=900, height=500, key="safe_route_map", returned_objects=[])