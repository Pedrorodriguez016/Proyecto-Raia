from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import os
import re
import random

# -------------------- CONFIG --------------------
app = FastAPI(title="RAIA API - Full Service", version="3.0")

# Variables globales para datos
artifacts = {}
global_df = None
unique_areas = []

@app.on_event("startup")
def startup_event():
    init_db()  # Función local ahora
    load_ml_models()
    load_data_for_chatbot()

def load_ml_models():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts['model'] = tf.keras.models.load_model(os.path.join(base_dir, "crime_model.keras"))
        artifacts['scaler'] = joblib.load(os.path.join(base_dir, "scaler.joblib"))
        artifacts['le'] = joblib.load(os.path.join(base_dir, "label_encoder.joblib"))
        artifacts['le_severity'] = joblib.load(os.path.join(base_dir, "severity_encoder.joblib"))
        artifacts['feature_cols'] = joblib.load(os.path.join(base_dir, "feature_cols.joblib"))
        print("API: Modelos de IA cargados (multi-salida con severidad).")
    except Exception as e:
        print(f"API Error Modelos: {e}")

def load_data_for_chatbot():
    """Carga el CSV para que el chatbot pueda consultar estadísticas reales."""
    global global_df, unique_areas
    try:
        csv_path = os.path.join("output", "crimes_clean.csv")
        if not os.path.exists(csv_path):
            csv_path = "Crime_Data_from_2020_to_Present.csv"
            
        if os.path.exists(csv_path):
            # Cargamos, intentando inferir columnas si es el clean o el raw
            df = pd.read_csv(csv_path)
            
            # Preproceso ligero
            df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
            df["YEAR"] = df["DATE OCC"].dt.year
            df["HOUR OCC"] = (df["TIME OCC"] // 100) % 24
            
            global_df = df
            unique_areas = sorted(df["AREA NAME"].dropna().unique().tolist())
            print(f"API: Datos para Chatbot cargados ({len(df)} registros).")
        else:
            print("API: No se encontró el CSV. El chatbot tendrá funciones limitadas.")
    except Exception as e:
        print(f"API Error Datos: {e}")

# -------------------- LÓGICA DEL CHATBOT (Backend) --------------------
CRIME_MAP = {
    'robo': 'ROBBERY', 'robos': 'ROBBERY', 'atraco': 'ROBBERY', 'atracos': 'ROBBERY',
    'homicidio': 'HOMICIDE', 'homicidios': 'HOMICIDE', 'asesinato': 'HOMICIDE',
    'agresión': 'AGGRAVATED ASSAULT', 'agresiones': 'AGGRAVATED ASSAULT', 'asalto': 'ASSAULT',
    'violación': 'RAPE', 'violaciones': 'RAPE',
    'hurto': 'THEFT', 'hurtos': 'THEFT', 'robo menor': 'THEFT'
}
SINONIMS_TEMPORAL = r'(tendencia|tendencias|evolución|cambio|a lo largo del tiempo|ha subido|ha bajado|datos anuales|año|años)'
SINONIMS_COMPARACIO = r'(diferencia|comparación|comparar|versus|vs|contra|cuál es mejor|cuál es peor|dos zonas|compara)'
SINONIMS_DEMOGRAFIC = r'(menores|menos de|más de|mayores|entre|edad|género|sexo|mujeres|hombres|víctima)'
SINONIMS_INTERVAL_HORARI = r'(entre\s+las?\s+(\d{1,2})\s+y\s+las?\s+(\d{1,2}))'

def process_chat_logic(prompt: str):
    if global_df is None:
        return "El sistema de datos no está disponible en el servidor."
        
    prompt_lower = prompt.lower()
    df_full = global_df
    
    # 1. Detectar Intenciones
    detected_crime = None
    for k, v in CRIME_MAP.items():
        if k in prompt_lower:
            detected_crime = v
            break
            
    # 2. Análisis de Intervalos (Ej: "entre las 20 y las 23")
    match_interval = re.search(SINONIMS_INTERVAL_HORARI, prompt_lower)
    if match_interval:
        try:
            start, end = int(match_interval.group(2)), int(match_interval.group(3))
            if start <= end:
                count = len(df_full[(df_full['HOUR OCC'] >= start) & (df_full['HOUR OCC'] <= end)])
            else:
                count = len(df_full[(df_full['HOUR OCC'] >= start) | (df_full['HOUR OCC'] <= end)])
            return f"He analizado el período entre las {start}:00 y las {end}:00. Se registraron **{count:,} incidentes**."
        except: pass

    # 3. Comparación (Ej: "Compara Central versus Hollywood")
    if re.search(SINONIMS_COMPARACIO, prompt_lower):
        areas_found = [a for a in unique_areas if a.lower() in prompt_lower]
        if len(areas_found) >= 2:
            a1, a2 = areas_found[:2]
            c1 = len(df_full[df_full['AREA NAME'] == a1])
            c2 = len(df_full[df_full['AREA NAME'] == a2])
            diff = abs(c1 - c2)
            winner = a1 if c1 > c2 else a2
            return f"Comparativa: **{a1}** ({c1:,}) vs **{a2}** ({c2:,}). La zona de **{winner}** tiene {diff:,} crímenes más."

    # 4. Tendencias
    if re.search(SINONIMS_TEMPORAL, prompt_lower):
        target = detected_crime or "todos los crímenes"
        df_trend = df_full[df_full['Crm Cd Desc'] == detected_crime] if detected_crime else df_full
        counts = df_trend['YEAR'].value_counts().sort_index()
        if len(counts) > 1:
            last, prev = counts.iloc[-1], counts.iloc[-2]
            change = ((last - prev) / prev) * 100
            trend = "subido" if change > 0 else "bajado"
            return f"La tendencia de '{target}' ha **{trend}** un {abs(change):.1f}% en el último año registrado."

    # Respuesta genérica
    return "Soy el cerebro de la API. Pregúntame sobre 'tendencias', 'comparaciones entre zonas' o 'intervalos horarios'."

# -------------------- GESTIÓN DE USUARIOS (INTEGRADA) --------------------
import json
import hashlib

DB_FILE = "users_db.json"

def _load_users():
    if not os.path.exists(DB_FILE): return {}
    try:
        with open(DB_FILE, "r") as f: return json.load(f)
    except json.JSONDecodeError: return {}

def _save_users(users_dict):
    with open(DB_FILE, "w") as f: json.dump(users_dict, f, indent=4)

def init_db():
    if not os.path.exists(DB_FILE):
        # Usuario admin por defecto: admin / 1234
        users = {"admin": hashlib.sha256(b"1234").hexdigest()}
        _save_users(users)

def check_credentials_logic(username, password):
    users = _load_users()
    if username in users:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        if users[username] == hashed_pw:
            return True
    return False

def add_user_logic(username, password):
    users = _load_users()
    if username in users: return False
    users[username] = hashlib.sha256(password.encode()).hexdigest()
    _save_users(users)
    return True

# -------------------- SEGURIDAD (FASTAPI) --------------------
security = HTTPBasic()
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if not check_credentials_logic(credentials.username, credentials.password):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    return credentials.username

# -------------------- ENDPOINTS --------------------
class UserData(BaseModel):
    username: str
    password: str

@app.post("/check-login")
def check_login(username: str = Depends(authenticate)):
    return {"status": "success", "user": username}

@app.post("/register")
def register(user_data: UserData):
    if add_user_logic(user_data.username, user_data.password):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Usuario ya existe")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest, username: str = Depends(authenticate)):
    """Endpoint para el chatbot inteligente."""
    response_text = process_chat_logic(request.message)
    return {"response": response_text}

class CrimeInput(BaseModel):
    area: str
    lat: float
    lon: float
    date_year: int
    date_month: int
    day_of_week: int
    hour: int
    victim_age: int
    victim_sex: str  # 'F', 'M', 'X'


@app.post("/predict")
def predict_crime(data: CrimeInput, username: str = Depends(authenticate)):
    if 'model' not in artifacts: raise HTTPException(503, "Modelo no cargado")
    
    try:
        # Mapping Sex manual como en el training
        sex_map = {'F': 0, 'M': 1, 'X': 2}
        sex_val = sex_map.get(data.victim_sex, 2)

        input_df = pd.DataFrame([{
            'YEAR': data.date_year, 'LAT': data.lat, 'LON': data.lon,
            'TIME OCC': data.hour * 100, 'Vict Age': data.victim_age,
            'Vict Sex': sex_val,
            'MONTH_NUM': data.date_month, 'Premis Cd': 0.0, 
            'Weapon Used Cd': 0.0, 'DAY_OF_WEEK': data.day_of_week
        }])
        
        # Procesar
        input_df = input_df[artifacts['feature_cols']]
        X_scaled = artifacts['scaler'].transform(input_df)
        
        # MODELO MULTI-SALIDA: Devuelve [crime_probs, severity_probs]
        predictions = artifacts['model'].predict(X_scaled, verbose=0)
        
        # Extraer predicciones de ambas salidas
        crime_probs = predictions[0][0]  # Primera salida: tipo de crimen
        severity_probs = predictions[1][0]  # Segunda salida: severidad
        
        # Resultados del tipo de crimen
        top_idx = np.argmax(crime_probs)
        prediction = artifacts['le'].inverse_transform([top_idx])[0]
        confidence = float(np.max(crime_probs))
        
        top_3_idx = np.argsort(crime_probs)[-3:][::-1]
        top_3 = [{
            "crim": artifacts['le'].inverse_transform([i])[0], 
            "probabilitat": float(crime_probs[i])
        } for i in top_3_idx]
        
        # Resultados de severidad
        severity_idx = np.argmax(severity_probs)
        severity = artifacts['le_severity'].inverse_transform([severity_idx])[0]
        severity_confidence = float(np.max(severity_probs))

        return {
            "prediction": prediction, 
            "confidence": confidence, 
            "top_3": top_3,
            "severity": severity,
            "severity_confidence": severity_confidence
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# -------------------- NUEVOS ENDPOINTS: GEOCODING Y RUTAS (BFF Pattern) --------------------
from geopy.geocoders import Nominatim
import requests

@app.get("/geocode")
def geocode_address(address: str, username: str = Depends(authenticate)):
    """Geocodifica una dirección restringida a Los Ángeles."""
    geolocator = Nominatim(user_agent="raia_api_service_v1")
    try:
        # Agregar contexto de LA si falta
        search_query = f"{address}, Los Angeles, California, USA" if "Los Angeles" not in address else address
        loc = geolocator.geocode(search_query, timeout=10)
        
        if loc:
            lat, lon = loc.latitude, loc.longitude
            # Validar límites de LA (aprox)
            if (33.7 <= lat <= 34.35) and (-118.7 <= lon <= -118.0):
                return {"found": True, "lat": lat, "lon": lon, "address": loc.address}
            else:
                return {"found": False, "error": "Ubicación fuera del área metropolitana de Los Ángeles"}
        return {"found": False, "error": "Dirección no encontrada"}
    except Exception as e:
        raise HTTPException(500, f"Error geocoding: {str(e)}")

class RouteRequest(BaseModel):
    waypoints: list[list[float]] # Lista de [lat, lon]
    mode: str = "driving" # driving, foot, bike

@app.post("/route")
def get_osrm_route(data: RouteRequest, username: str = Depends(authenticate)):
    """Obtiene rutas desde OSRM (proxied via API). Soporta waypoints intermedios."""
    
    # Determinar endpoint según modo
    if data.mode == "Caminando":
        base_url = "https://routing.openstreetmap.de/routed-foot/route/v1/foot"

    else: # Auto
        base_url = "http://router.project-osrm.org/route/v1/driving"
        
    # Construir string de coordenadas: lon,lat;lon,lat;...
    try:
        coords_str = ";".join([f"{p[1]},{p[0]}" for p in data.waypoints])
    except:
        raise HTTPException(400, "Formato de waypoints inválido. Use [[lat, lon], ...]")

    url = f"{base_url}/{coords_str}?overview=full&geometries=geojson&alternatives=true"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 400:
             # OSRM a veces falla si los puntos están muy lejos o son inválidos
            raise HTTPException(400, "No se pudo calcular ruta entre estos puntos.")
        else:
            raise HTTPException(r.status_code, "Error en servicio OSRM externo")
    except Exception as e:
        raise HTTPException(500, f"Error de conexión con OSRM: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
