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
import auth_manager

# -------------------- CONFIG --------------------
app = FastAPI(title="RAIA API - Full Service", version="3.0")

# Variables globales para datos
artifacts = {}
global_df = None
unique_areas = []

@app.on_event("startup")
def startup_event():
    auth_manager.init_db()
    load_ml_models()
    load_data_for_chatbot()

def load_ml_models():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts['model'] = tf.keras.models.load_model(os.path.join(base_dir, "crime_model.keras"))
        artifacts['scaler'] = joblib.load(os.path.join(base_dir, "scaler.joblib"))
        artifacts['le'] = joblib.load(os.path.join(base_dir, "label_encoder.joblib"))
        artifacts['feature_cols'] = joblib.load(os.path.join(base_dir, "feature_cols.joblib"))
        print("✅ API: Modelos de IA cargados.")
    except Exception as e:
        print(f"❌ API Error Modelos: {e}")

def load_data_for_chatbot():
    """Carga el CSV para que el chatbot pueda consultar estadísticas reales."""
    global global_df, unique_areas
    try:
        csv_path = "Crime_Data_from_2020_to_Present.csv"
        if os.path.exists(csv_path):
            # Cargamos solo columnas necesarias para ahorrar memoria
            cols = ["DATE OCC", "TIME OCC", "AREA NAME", "Crm Cd Desc", "Vict Sex", "Vict Age"]
            df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
            
            # Preproceso ligero
            df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
            df["YEAR"] = df["DATE OCC"].dt.year
            df["HOUR OCC"] = (df["TIME OCC"] // 100) % 24
            
            global_df = df
            unique_areas = sorted(df["AREA NAME"].dropna().unique().tolist())
            print(f"✅ API: Datos para Chatbot cargados ({len(df)} registros).")
        else:
            print("⚠️ API: No se encontró el CSV. El chatbot tendrá funciones limitadas.")
    except Exception as e:
        print(f"❌ API Error Datos: {e}")

# -------------------- LÓGICA DEL CHATBOT (Backend) --------------------
CRIME_MAP = {
    'robatori': 'ROBBERY', 'robatoris': 'ROBBERY', 'homicidi': 'HOMICIDE',
    'agressió': 'AGGRAVATED ASSAULT', 'violació': 'RAPE', 'furt': 'THEFT'
}
SINONIMS_TEMPORAL = r'(tendències|evolución|canvi|al llarg del temps|ha pujat|ha baixat|dades anuals|evolució)'
SINONIMS_COMPARACIO = r'(diferència|comparació|versus|contra|quin és millor|quin és pitjor|dues zones|compara)'
SINONIMS_DEMOGRAFIC = r'(menors|menys de|més de|majors|entre|anys|edat|gènere|sexe|dones|homes|víctima)'
SINONIMS_INTERVAL_HORARI = r'(entre\s+les\s+(\d{1,2})\s+i\s+les\s+(\d{1,2}))'

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
            
    # 2. Análisis de Intervalos (Ej: "entre les 20 i les 23")
    match_interval = re.search(SINONIMS_INTERVAL_HORARI, prompt_lower)
    if match_interval:
        try:
            start, end = int(match_interval.group(2)), int(match_interval.group(3))
            if start <= end:
                count = len(df_full[(df_full['HOUR OCC'] >= start) & (df_full['HOUR OCC'] <= end)])
            else:
                count = len(df_full[(df_full['HOUR OCC'] >= start) | (df_full['HOUR OCC'] <= end)])
            return f"He analitzat el període entre les {start}:00 i les {end}:00. Es van registrar **{count:,} incidents**."
        except: pass

    # 3. Comparación (Ej: "Comparació Central versus Hollywood")
    if re.search(SINONIMS_COMPARACIO, prompt_lower):
        areas_found = [a for a in unique_areas if a.lower() in prompt_lower]
        if len(areas_found) >= 2:
            a1, a2 = areas_found[:2]
            c1 = len(df_full[df_full['AREA NAME'] == a1])
            c2 = len(df_full[df_full['AREA NAME'] == a2])
            diff = abs(c1 - c2)
            winner = a1 if c1 > c2 else a2
            return f"Comparativa: **{a1}** ({c1:,}) vs **{a2}** ({c2:,}). La zona de **{winner}** té {diff:,} crims més."

    # 4. Tendencias
    if re.search(SINONIMS_TEMPORAL, prompt_lower):
        target = detected_crime or "tots els crims"
        df_trend = df_full[df_full['Crm Cd Desc'] == detected_crime] if detected_crime else df_full
        counts = df_trend['YEAR'].value_counts().sort_index()
        if len(counts) > 1:
            last, prev = counts.iloc[-1], counts.iloc[-2]
            change = ((last - prev) / prev) * 100
            trend = "pujat" if change > 0 else "baixat"
            return f"La tendència de '{target}' ha **{trend}** un {abs(change):.1f}% l'últim any registrat."

    # Respuesta genérica
    return "Sóc el cervell de l'API. Pregunta'm sobre 'tendències', 'comparacions entre zones' o 'intervals horaris'."

# -------------------- SEGURIDAD --------------------
security = HTTPBasic()
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if not auth_manager.check_credentials(credentials.username, credentials.password):
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
    if auth_manager.add_user(user_data.username, user_data.password):
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

@app.post("/predict")
def predict_crime(data: CrimeInput, username: str = Depends(authenticate)):
    if 'model' not in artifacts: raise HTTPException(503, "Modelo no cargado")
    
    try:
        input_df = pd.DataFrame([{
            'YEAR': data.date_year, 'LAT': data.lat, 'LON': data.lon,
            'TIME OCC': data.hour * 100, 'Vict Age': data.victim_age,
            'MONTH_NUM': data.date_month, 'Premis Cd': 0.0, 
            'Weapon Used Cd': 0.0, 'DAY_OF_WEEK': data.day_of_week
        }])
        
        # Procesar
        input_df = input_df[artifacts['feature_cols']]
        X_scaled = artifacts['scaler'].transform(input_df)
        pred_probs = artifacts['model'].predict(X_scaled, verbose=0)
        
        # Resultados
        top_idx = np.argmax(pred_probs)
        prediction = artifacts['le'].inverse_transform([top_idx])[0]
        confidence = float(np.max(pred_probs))
        
        top_3_idx = np.argsort(pred_probs[0])[-3:][::-1]
        top_3 = [{"crim": artifacts['le'].inverse_transform([i])[0], "probabilitat": float(pred_probs[0][i])} for i in top_3_idx]

        return {"prediction": prediction, "confidence": confidence, "top_3": top_3}
    except Exception as e:
        raise HTTPException(500, str(e))