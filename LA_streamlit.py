import streamlit as st
import pandas as pd
import altair as alt
import time
import json
import os
import re 
import random 
import numpy as np 

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Crímenes LA - Proyecto ML", layout="wide")

st.title("🚨 Crímenes en Los Ángeles (2020 - Presente)")

# -------------------- CONSTANTS I MAPATGES --------------------
PREDICTION_FILE = "prediction_history.json"

# Diccionari de mapatge de sinònims per als crims més comuns
CRIME_MAP = {
    'robatori': 'ROBBERY',
    'robatoris': 'ROBBERY',
    'homicidi': 'HOMICIDE',
    'agressió': 'AGGRAVATED ASSAULT',
    'agressions': 'AGGRAVATED ASSAULT',
    'violació': 'RAPE',
    'furt': 'THEFT',
    'furts': 'THEFT',
}

# Conjunts de sinònims per a la detecció en el chatbot
SINONIMS_TEMPORAL = r'(tendències|evolución|canvi|al llarg del temps|ha pujat|ha baixat|dades anuals|evolució)'
SINONIMS_COMPARACIO = r'(diferència|comparació|versus|contra|quin és millor|quin és pitjor|dues zones|compara)'
SINONIMS_DEMOGRAFIC = r'(menors|menys de|més de|majors|entre|anys|edat|gènere|sexe|dones|homes|víctima)'
SINONIMS_HORARI_PUNTUAL = r'(hora|quan és pitjor|nit|matí|tarda|moment més perillós)'
SINONIMS_INTERVAL_HORARI = r'(entre\s+les\s+(\d{1,2})\s+i\s+les\s+(\d{1,2}))' 

# -------------------- INICIALITZACIÓ DE LA PERSISTÈNCIA --------------------
def initialize_prediction_history():
    """Assegura que el fitxer JSON d'historial existeix."""
    if not os.path.exists(PREDICTION_FILE):
        with open(PREDICTION_FILE, 'w') as f:
            json.dump([], f)
initialize_prediction_history()

# -------------------- INICIALITZACIÓ DE L'ESTAT DE LA SESSIÓ --------------------
if 'modo_sel' not in st.session_state: st.session_state['modo_sel'] = "Año"
if 'year_sel' not in st.session_state: st.session_state['year_sel'] = None
if 'area_sel' not in st.session_state: st.session_state['area_sel'] = None
if 'crime_sel' not in st.session_state: st.session_state['crime_sel'] = None
if 'page' not in st.session_state: st.session_state['page'] = "Dashboard"

if "messages" not in st.session_state:
    initial_message = """
    **🤖 Assistent Avançat**
    Sóc un model d'IA dissenyat per analitzar el conjunt de dades de crims de LA.
    
    Prova amb sinònims o frases complexes! Per exemple:
    * `Quina ha sigut la evolució dels robatoris a l'últim any?`
    * `Compara la perillositat de Central i Newton.`
    * `Quants crims hi va haver entre les 22 i les 04 hores?`
    """
    st.session_state["messages"] = [
        {"role": "assistant", "content": initial_message}
    ]


# -------------------- CACHE DE DADES --------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
    except FileNotFoundError:
        st.error("❌ Error: Fitxer de dades 'Crime_Data_from_2020_to_Present.csv' no trobat.")
        st.stop()
        
    if "DATE OCC" in df.columns:
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
        df["YEAR"] = df["DATE OCC"].dt.year
    if "TIME OCC" in df.columns:
        df["HOUR OCC"] = (df["TIME OCC"] // 100) % 24 
    
    df = df.dropna(subset=["Crm Cd Desc", "LAT", "LON", "AREA NAME", "HOUR OCC", "Vict Age", "Vict Sex", "Crm Cd"])
    
    areas = sorted(df["AREA NAME"].dropna().unique().tolist())
    sexes = sorted([s for s in df["Vict Sex"].dropna().unique().tolist() if s in ['F', 'M']])
    crime_codes_df = df[["Crm Cd", "Crm Cd Desc"]].drop_duplicates().sort_values(by="Crm Cd Desc")
    crime_options = [f"{desc} ({code})" for code, desc in crime_codes_df.values]
    
    return df, areas, sexes, crime_options, crime_codes_df

try:
    df, unique_areas, unique_sexes, crime_options, crime_codes_df = load_data()
    st.success("✅ Dades carregades i preprocessades correctament.")
except Exception as e:
    st.error(f"❌ Error al carregar les dades: {e}")
    st.stop()


# -------------------- FUNCIONS DE PERSISTÈNCIA --------------------

def save_prediction(new_entry):
    """Afegeix una nova entrada a l'historial JSON."""
    try:
        with open(PREDICTION_FILE, 'r') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    history.append(new_entry)
    
    with open(PREDICTION_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    st.success("💾 Predicció guardada a l'historial!")

def load_prediction_history():
    """Carrega tot l'historial de prediccions des del JSON."""
    try:
        with open(PREDICTION_FILE, 'r') as f:
            history = json.load(f)
            return pd.DataFrame(history)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()


# -------------------- FUNCIÓ CLAU: SIMULACIÓ D'IA AVANÇADA (AMPLIADA) --------------------

def get_time_period(hour):
    """Classifica l'hora en un període del dia."""
    if 5 <= hour < 12: return "el matí"
    elif 12 <= hour < 18: return "la tarda"
    elif 18 <= hour < 22: return "la nit (primera part)"
    else: return "la matinada (alta nit)"

def process_message(prompt, df_full):
    """
    Simula una resposta més flexible amb sinònims i aleatorietat.
    """
    prompt_lower = prompt.lower()
    
    # 1. Detecció de conceptes
    detected_crime = None
    for keyword, official_name in CRIME_MAP.items():
        if keyword in prompt_lower:
            detected_crime = official_name
            break
            
    detected_area = next((area for area in unique_areas if area.lower() in prompt_lower), None)
    match_age = re.search(r'(menors|majors|entre)\s+(\d+)\s+(anys|edat)', prompt_lower)
    match_interval = re.search(SINONIMS_INTERVAL_HORARI, prompt_lower) 

    
    # 2. ANÀLISI HORÀRIA D'INTERVALS
    if match_interval:
        try:
            start_hour = int(match_interval.group(2))
            end_hour = int(match_interval.group(3))
            
            if start_hour <= end_hour:
                df_filtered = df_full[(df_full['HOUR OCC'] >= start_hour) & (df_full['HOUR OCC'] <= end_hour)]
            else: 
                df_filtered = df_full[(df_full['HOUR OCC'] >= start_hour) | (df_full['HOUR OCC'] <= end_hour)]
            
            count = len(df_filtered)
            
            return random.choice([
                f"He analitzat el període entre les {start_hour}:00 i les {end_hour}:00 hores. Es van registrar un total de **{count:,} incidents** en aquest interval.",
                f"La distribució horària mostra que el període entre {start_hour}:00 i {end_hour}:00 hores és responsable de **{count:,} crims**. És un interval significatiu de temps.",
            ])
            
        except Exception:
            return "No he pogut processar correctament l'interval horari. Assegura't d'escriure 'entre les HH i les HH'."

    # 3. ANÀLISI HORÀRIA PUNCTUAL
    elif re.search(SINONIMS_HORARI_PUNTUAL, prompt_lower):
        most_frequent_hour = df_full['HOUR OCC'].mode().iloc[0]
        peak_period = get_time_period(most_frequent_hour)
        busiest_area = df_full['AREA NAME'].mode().iloc[0]

        return random.choice([
            f"L'anàlisi horària mostra que el moment més freqüent per als incidents és durant **{peak_period}** (aproximadament a les {most_frequent_hour}:00 h). Això pot indicar un patró de delinqüència durant les hores de més trànsit.",
            f"Basat en les dades, la major concentració d'activitat criminal es produeix a **{peak_period}**. A l'àrea de {busiest_area}, aquest patró és especialment notable.",
        ])

    # 4. ANÀLISI TEMPORAL
    elif re.search(SINONIMS_TEMPORAL, prompt_lower):
        target = detected_crime or "tots els crims"
        
        if detected_crime:
            df_trend = df_full[df_full['Crm Cd Desc'] == detected_crime].copy()
            if df_trend.empty:
                 return random.choice([
                    f"No hi ha prou dades per analitzar la tendència del crim '{target}'.",
                    f"Ho sento, el registre de '{target}' és massa escàs per calcular una evolució fiable.",
                ])
            
            yearly_counts = df_trend['YEAR'].value_counts().sort_index()
            
            if len(yearly_counts) >= 2:
                last_year = yearly_counts.index[-1]
                second_last_year = yearly_counts.index[-2]
                last_count = yearly_counts.iloc[-1]
                second_last_count = yearly_counts.iloc[-2]

                diff = last_count - second_last_count
                percentage_change = (diff / second_last_count) * 100 if second_last_count != 0 else (100 if diff > 0 else 0)
                
                trend_direction = "augmentat" if diff > 0 else "disminuït"
                
                return random.choice([
                    f"He analitzat la tendència: el crim '{target}' ha **{trend_direction}** un {percentage_change:.2f}% entre {second_last_year} i {last_year}.",
                    f"L'evolució de '{target}' al darrer any mostra un canvi de {second_last_count:,} a {last_count:,} incidents, que és una {trend_direction} de {percentage_change:.2f}%."
                ])
            else:
                return f"Només tenim dades per a un o dos anys ({', '.join(map(str, yearly_counts.index))}) per a {target}. No es pot calcular una tendència completa."


    # 5. COMPARACIÓ ENTRE ENTITATS
    elif re.search(SINONIMS_COMPARACIO, prompt_lower):
        areas_mentioned = [a for a in unique_areas if a.lower() in prompt_lower]
        
        if len(areas_mentioned) >= 2:
            area1, area2 = areas_mentioned[:2]
            count1 = len(df_full[df_full['AREA NAME'] == area1])
            count2 = len(df_full[df_full['AREA NAME'] == area2])
            
            if count1 == 0 and count2 == 0:
                return f"No es van trobar incidents registrats ni a {area1} ni a {area2}."

            if count1 > count2:
                diff_area = area1
                diff_count = count1 - count2
            else:
                diff_area = area2
                diff_count = count2 - count1
                
            return random.choice([
                f"Comparació directa: L'àrea de **{diff_area}** (con {max(count1, count2):,} incidents) té una diferència notable de **{diff_count:,} crims més** respecte a l'altra zona ({min(count1, count2):,}).",
                f"Després d'analitzar {area1} i {area2}, el risc és significativament més alt a **{diff_area}**, amb **{diff_count:,} incidents addicionals**."
            ])
        else:
            return "Per fer una comparació, necessito identificar almenys dues entitats (dos anys, dues zones o dos tipus de crims). Intenta preguntar 'Quina és la diferència entre Hollywood i Central?'"

    # 6. ANÀLISI DEMOGRÀFICA
    elif re.search(SINONIMS_DEMOGRAFIC, prompt_lower):
        demographic = "dones" if "dones" in prompt_lower or "femení" in prompt_lower else "homes"
        
        return random.choice([
            f"L'anàlisi de patrons demogràfics mostra que el crim més comú en la mostra és '{df_full['Crm Cd Desc'].mode().iloc[0]}'. Pots utilitzar el **Dashboard** per filtrar '{demographic}' i l'edat ({'Edat: ' + str(match_age.group(2)) if match_age else 'Totes les edats'}) per a una anàlisi detallada.",
            f"Les dades indiquen que els incidents relacionats amb '{demographic}' sovint són del tipus '{df_full['Crm Cd Desc'].mode().iloc[0]}'. Visita la pestanya 'Dashboard' per a un filtratge precís."
        ])
    
    # 7. PREGUNTA NO CLASSIFICADA
    else:
        return random.choice([
            "Aquesta pregunta és complexa per al meu analitzador. Intenta reformular la teva pregunta enfocant-la en 'tendències', 'comparacions' o 'dades demogràfiques'.",
            "No tinc una resposta precalculada per a aquesta consulta. T'aconsello anar al **Dashboard** i utilitzar els filtres (Zona o Tipus de crim) per trobar la resposta visualment.",
            "Ho sento, no he pogut classificar la teva intenció. Recorda que sóc bo amb tendències, comparacions i anàlisi d'intervals horaris (Ex: 'entre les 20 i les 05').",
        ])

# -------------------- SIMULACIÓ PREDICCIÓ SPATIO-TEMPORAL (CNN/LSTM) --------------------

def generate_hourly_risk(area, df_full):
    """
    Simula la predicció de risc horari per a una àrea, imitant la lògica temporal (LSTM).
    """
    df_area = df_full[df_full['AREA NAME'] == area].copy()
    
    # 1. Base de Risc Horària (Basada en dades històriques)
    hourly_counts = df_area.groupby('HOUR OCC').size()
    hourly_risk = (hourly_counts / hourly_counts.max()) * 100 
    
    # 2. Simular Suavització/Predicció de la CNN/LSTM
    if len(hourly_risk) > 4:
        # Suavitzar els punts amb una finestra (simulant la capa LSTM/CNN)
        simulated_prediction = hourly_risk.rolling(window=3, center=True, min_periods=1).mean().fillna(hourly_risk)
        # Afegir una petita aleatorietat per simular incertesa
        simulated_prediction += np.random.uniform(-5, 5, len(simulated_prediction))
    else:
        simulated_prediction = hourly_risk
        
    simulated_prediction = simulated_prediction.clip(0, 100)
    
    # Crear DataFrame de sortida (24 hores)
    risk_df = pd.DataFrame({'HOUR OCC': range(24)})
    risk_df = risk_df.merge(simulated_prediction.rename('Risk_Score'), on='HOUR OCC', how='left').fillna(0)
    
    # Simular que les hores d'alta nit tenen un lleuger repunt (efecte temporal/no lineal)
    for h in [0, 1, 2, 3, 4]:
        risk_df.loc[risk_df['HOUR OCC'] == h, 'Risk_Score'] *= 1.1
    
    risk_df['Risk_Score'] = risk_df['Risk_Score'].clip(0, 100).astype(int)
    
    return risk_df

def get_area_center(area, df_full):
    """Obté el centre d'una àrea per al mapa."""
    df_area = df_full[df_full['AREA NAME'] == area]
    if not df_area.empty:
        return df_area['LAT'].mean(), df_area['LON'].mean()
    return 34.0522, -118.2437 # Coordenades per defecte de LA

def get_area_crime_distribution(area, hour, df_full, top_n=5, risk_factor=1.0):
    """
    Calcula la distribució percentual de crims per tipus per a l'àrea i hora,
    i l'ajusta lleugerament segons el factor de risc personalitzat.
    """
    # Filtra els crims de l'àrea i l'hora (simulant el focus)
    df_filtered = df_full[
        (df_full['AREA NAME'] == area) & 
        (df_full['HOUR OCC'].isin([(hour - 1) % 24, hour, (hour + 1) % 24]))
    ].copy()
    
    if df_filtered.empty:
        return {}

    # Distribució base (percentatges)
    distribution = df_filtered['Crm Cd Desc'].value_counts(normalize=True).head(top_n)
    
    # Ajustar la distribució segons el risc: si el risc és alt, simular que els crims violents pugen
    if risk_factor > 1.1: 
        adjustment = (risk_factor - 1.1) / 5 # Petita pujada de pes
        
        # Simular que Robbery i Assault (violents) guanyen pes
        if 'ROBBERY' in distribution.index:
            distribution['ROBBERY'] += adjustment
        if 'AGGRAVATED ASSAULT' in distribution.index:
            distribution['AGGRAVATED ASSAULT'] += adjustment * 0.5
            
        # Re-normalitzar perquè la suma torni a ser 1
        distribution = distribution.apply(lambda x: max(0, x))
        distribution = distribution / distribution.sum()

    # Format de sortida
    return {
        desc: f"{prob*100:.1f}%"
        for desc, prob in distribution.items()
    }


# -------------------- SIDEBAR I NAVEGACIÓ --------------------
st.sidebar.header("🗺️ Navegació i Modes")
st.session_state['page'] = st.sidebar.radio(
    "Selecciona la vista:",
    ("Dashboard", "Chatbot Interactiu", "Predicció ML", "Historial Prediccions", "Mapa de Risc Dinàmic (NN)"), 
    key='page_radio_key'
)

# -------------------- FUNCIÓ DE FILTRATGE DEL DASHBOARD --------------------
def filtrar_dades(df):
    modo = st.session_state['modo_sel']
    df_filtered = df.copy()

    st.session_state['modo_sel'] = st.sidebar.radio(
        "Filtrar per:",
        ("Año", "Zona", "Tipo de crimen"),
        key='modo_radio_dashboard_key' 
    )

    if modo == "Año":
        years = sorted(df["YEAR"].dropna().unique().astype(int))
        if st.session_state['year_sel'] is None and years:
            st.session_state['year_sel'] = years[-1] 
        st.session_state['year_sel'] = st.sidebar.selectbox(
            "Selecciona un año:", years,
            index=years.index(st.session_state['year_sel']) if st.session_state['year_sel'] in years else 0,
            key='year_select_key'
        )
        if st.session_state['year_sel']:
            df_filtered = df_filtered[df_filtered["YEAR"] == st.session_state['year_sel']]
        st.subheader(f"📊 Dashboard Filtrado por Año: {st.session_state['year_sel']}")

    elif modo == "Zona":
        areas = sorted(df["AREA NAME"].dropna().unique())
        if st.session_state['area_sel'] is None and areas:
            st.session_state['area_sel'] = areas[0]
        st.session_state['area_sel'] = st.sidebar.selectbox(
            "Selecciona una zona:", areas,
            index=areas.index(st.session_state['area_sel']) if st.session_state['area_sel'] in areas else 0,
            key='area_select_key'
        )
        if st.session_state['area_sel']:
            df_filtered = df_filtered[df_filtered["AREA NAME"] == st.session_state['area_sel']]
        st.subheader(f"📊 Dashboard Filtrado por Zona: {st.session_state['area_sel']}")

    elif modo == "Tipo de crimen":
        crime_types = sorted(df["Crm Cd Desc"].dropna().unique())
        if st.session_state['crime_sel'] is None and crime_types:
            st.session_state['crime_sel'] = crime_types[0]
        st.session_state['crime_sel'] = st.sidebar.selectbox(
            "Selecciona un tipo de crimen:", crime_types,
            index=crime_types.index(st.session_state['crime_sel']) if st.session_state['crime_sel'] in crime_types else 0,
            key='crime_select_key'
        )
        if st.session_state['crime_sel']:
            df_filtered = df_filtered[df_filtered["Crm Cd Desc"] == st.session_state['crime_sel']]
        st.subheader(f"📊 Dashboard Filtrado por Tipo: {st.session_state['crime_sel']}")

    return df_filtered


# ==================== PÀGINA 1: DASHBOARD ====================

if st.session_state['page'] == "Dashboard":
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Paràmetres del Filtre")

    df_filtered = filtrar_dades(df)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Total de Crímenes Filtrados", value=f"{len(df_filtered):,}")
        st.subheader("Top 10 Tipos de Crímenes")
        top_crimes = df_filtered["Crm Cd Desc"].value_counts().reset_index()
        top_crimes.columns = ["Tipo de Crimen", "Count"]
        top_crimes = top_crimes.head(10)
        
        if not top_crimes.empty:
            chart = alt.Chart(top_crimes).mark_bar().encode(
                x=alt.X("Count", title="Recuento"),
                y=alt.Y("Tipo de Crimen", title="", sort="-x"),
                tooltip=["Tipo de Crimen", "Count"]
            ).properties(title="Recuento por Tipo de Crimen").interactive() 
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No hi ha dades suficients per mostrar el Top 10.")

    with col2:
        st.subheader("🗺️ Mapa de Crímenes (Punts)")
        df_map = df_filtered.rename(columns={"LAT": "lat", "LON": "lon"})
        st.map(df_map[["lat", "lon"]].head(5000))

    st.markdown("---")
    st.subheader("📜 Datos Detallados (Primeres 500 Filas)")
    st.dataframe(df_filtered.head(500), use_container_width=True, height=300)


# ==================== PÀGINA 2: CHATBOT INTERACTIU ====================

elif st.session_state['page'] == "Chatbot Interactiu":
    st.subheader("💬 Assistent de Consultes Avançat (Simulació d'IA)")
    st.warning("Aquest model intenta respondre a preguntes complexes sobre **tendències, comparacions i dades demogràfiques** buscant patrons al text.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Fes una pregunta oberta sobre la distribució de crims..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("El meu LLM intern està analitzant la teva pregunta..."):
            response = process_message(prompt, df)
            
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# ==================== PÀGINA 3: PREDICCIÓ ML ====================

elif st.session_state['page'] == "Predicció ML":
    st.header("🧠 Simulació de Predicció de Machine Learning")
    st.info("Introdueix les característiques d'un incident potencial per predir si serà classificat com a Crim Violent (simulació).")

    with st.form("ml_prediction_form"):
        st.subheader("Paràmetres de l'Incident")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            age = st.slider("Edat de la Víctima (Vict Age):", min_value=0, max_value=99, value=30, step=1)
        with col_f2:
            sex = st.selectbox("Gènere de la Víctima (Vict Sex):", options=unique_sexes, index=0)

        col_f3, col_f4 = st.columns(2)
        with col_f3:
            hour = st.number_input("Hora del Dia de l'Incident (0-23):", min_value=0, max_value=23, value=18, step=1)
        with col_f4:
            area = st.selectbox("Àrea/Zona (AREA NAME):", options=unique_areas, index=unique_areas.index('CENTRAL') if 'CENTRAL' in unique_areas else 0)
            
        selected_crime_option = st.selectbox(
            "Codi de Crim (Crm Cd):", 
            options=crime_options, 
            index=0, 
            help="Selecciona el tipus de crim per la descripció. El número entre parèntesis és el Codi de Crim utilitzat per a la predicció."
        )

        submitted = st.form_submit_button("Realitzar Predicció (Simulada)")
    
    if submitted:
        crime_code_str = selected_crime_option.split('(')[-1].replace(')', '')
        crime_code = int(crime_code_str)
        crime_description = selected_crime_option.split('(')[0].strip()

        with st.spinner('Executant el model de Machine Learning...'):
            time.sleep(2) 

        # LÒGICA DE PREDICCIÓ AMB 6 NIVELLS DE PROBABILITAT
        high_prob_codes = [110, 121, 230, 440]
        medium_risk_codes = [480, 510, 330] 
        
        if crime_code == 110: 
            probability = 0.95 
        elif crime_code in high_prob_codes and (hour >= 21 or area == 'CENTRAL'): 
            probability = 0.80
        elif crime_code in high_prob_codes and age < 25: 
            probability = 0.65
        elif crime_code in medium_risk_codes and hour >= 20: 
            probability = 0.50
        elif crime_code in medium_risk_codes or age < 30: 
            probability = 0.35
        else:
            probability = 0.15
            
        prediction_result = "Alta Probabilitat de Crim Violent" if probability > 0.6 else "Baixa Probabilitat de Crim Violent"

        st.subheader("Resultat de la Predicció")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(
                label="Predicció Classificació",
                value=prediction_result,
                delta=f"{round(probability*100)}% de Probabilitat",
                delta_color="inverse" if probability < 0.6 else "normal"
            )
        
        new_entry = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "area": area,
            "hour": hour,
            "age": age,
            "sex": sex,
            "crime_description": crime_description,
            "crime_code": crime_code,
            "result_class": prediction_result,
            "probability": round(probability, 2)
        }
        save_prediction(new_entry) 
        
        with col_res2:
            st.write(f"Característiques utilitzades:")
            st.write(f"* **Àrea:** {area}, **Hora:** {hour}:00 h")
            st.write(f"* **Edat Víc.:** {age} anys, **Crim:** {crime_code} - {crime_description}")
            

# ==================== PÀGINA 4: HISTORIAL DE PREDICCIONS ====================

elif st.session_state['page'] == "Historial Prediccions":
    st.header("🗄️ Historial de Prediccions Guardades")
    st.info("Aquestes dades són persistents. Es carreguen des del fitxer `prediction_history.json` cada vegada que s'obri l'aplicació.")
    
    history_df = load_prediction_history()
    
    if not history_df.empty:
        history_df = history_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        history_df.index = history_df.index + 1
        
        cols_to_show = ['timestamp', 'area', 'hour', 'age', 'sex', 'crime_description', 'result_class', 'probability']
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.metric("Total de Prediccions Guardades", len(history_df))
        with col_h2:
            violent_count = (history_df['result_class'] == 'Alta Probabilitat de Crim Violent').sum()
            st.metric("Prediccions d'Alta Probabilitat", f"{violent_count} ({round(violent_count/len(history_df)*100)}%)")

        st.markdown("---")
        st.dataframe(history_df[cols_to_show], use_container_width=True, height=500)
        
    else:
        st.warning("Encara no s'ha guardat cap predicció a l'historial.")
        st.write("Vés a la secció **'Predicció ML'** per començar a guardar dades.")


# ==================== PÀGINA 5: MAPA DE RISC DINÀMIC (NN) ====================

elif st.session_state['page'] == "Mapa de Risc Dinàmic (NN)":
    st.header("📈 Anàlisi Spatio-Temporal del Risc (Simulació CNN/LSTM)")
    st.info("Aquesta simulació imita una Xarxa Neuronal Convolucional (CNN) que prediu la intensitat de risc (calor) en una zona al llarg de 24 hores, ajustada per factors demogràfics.")
    
    # --- Columna 1: Àrea, Edat ---
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_area = st.selectbox(
            "Barri a Analitzar:", 
            options=unique_areas, 
            index=unique_areas.index('CENTRAL') if 'CENTRAL' in unique_areas else 0,
            key='area_risk_map_sel'
        )
        selected_age = st.slider("Edat de l'Usuari (perfil de risc):", min_value=0, max_value=99, value=30, step=1, key='age_map_nn')
    
    # --- Columna 2: Hora i Sexe ---
    with col_sel2:
        selected_hour = st.number_input("Hora de Predicció (0-23):", 
                                          min_value=0, max_value=23, value=20, step=1, key='hour_risk_map_sel')
        selected_sex = st.selectbox("Gènere de l'Usuari:", options=unique_sexes, index=0, key='sex_map_nn')

    
    st.markdown("---")
    
    # --- Risc Horari de l'Àrea Seleccionada (Base Temporal) ---
    with st.spinner(f"Executant el model temporal (LSTM) per a {selected_area}..."):
        time.sleep(0.5)
        risk_df_hourly = generate_hourly_risk(selected_area, df)
        
    st.subheader(f"1. Risc Predicció Horària per a {selected_area}")
    st.caption("Evolució del risc (0-100) al llarg del dia. Els valors han estat suavitzats per simular la lògica de predicció LSTM.")

    chart_hourly = alt.Chart(risk_df_hourly).mark_area(line={'color':'darkblue'}, color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='#66BB6A', offset=0.3), alt.GradientStop(color='#EF5350', offset=0.8)],
        x1=1, x2=1, y1=1, y2=0
    )).encode(
        x=alt.X('HOUR OCC', title='Hora del Dia', axis=alt.Axis(format='d')),
        y=alt.Y('Risk_Score', title='Puntuació de Risc Predita (0-100)'),
        tooltip=[alt.Tooltip('HOUR OCC', title='Hora'), alt.Tooltip('Risk_Score', title='Risc')]
    ).properties(
        height=300
    ).interactive()
    
    st.altair_chart(chart_hourly, use_container_width=True)
    
    # --- Mapa Dinàmic de Risc Simulat (Ajust Spatio-Temporal i Demogràfic) ---
    st.subheader(f"2. Mapa de Risc Personalitzat per a {selected_area} a les {selected_hour}:00 h")
    st.caption("Visualització ajustada a l'hora i perfil de risc ('Edat' i 'Gènere') seleccionats.")
    
    # 1. Preparar dades de crims: Filtrar només per l'àrea seleccionada i l'hora seleccionada +/- 1 hora
    
    # Filtre temporal per simular el focus de la predicció
    df_map_risk = df[
        (df['AREA NAME'] == selected_area) & 
        (df['HOUR OCC'].isin([(selected_hour - 1) % 24, selected_hour, (selected_hour + 1) % 24]))
    ].rename(columns={"LAT": "lat", "LON": "lon"})
    
    if df_map_risk.empty:
        st.warning(f"No hi ha suficients dades històriques de crims per a l'àrea '{selected_area}' a les {selected_hour}:00 h per crear la visualització.")
    else:
        # --- APLICACIÓ DEL FILTRE DE PERSONALITZACIÓ I CÀLCUL DE RISC FINAL ---
        
        # 2. Obtenir Risc Base (Sortida de l'LSTM)
        area_risk_score_base = risk_df_hourly[risk_df_hourly['HOUR OCC'] == selected_hour]['Risk_Score'].iloc[0]
        
        # 3. Calcular Factor de Risc Demogràfic (Simulació de la capa final de la NN)
        demographic_factor = 1.0
        if selected_age < 25:
            demographic_factor *= 1.10 # Joves augmenten lleugerament el risc simulat
        if selected_sex == 'F':
            demographic_factor *= 1.05 # Gènere Femení augmenta lleugerament el risc simulat
            
        # 4. Càlcul del Risc Final Personalitzat
        area_risk_score_final = min(100, area_risk_score_base * demographic_factor * 1.05) # Multiplicador per fer el canvi visible
        
        # 5. Aplicar Risc Simulat al Mapa (Simulació CNN)
        
        # El risc final s'utilitza per ponderar els punts del mapa
        df_map_risk['Simulated_Risk'] = (
            (area_risk_score_final * 0.5) + # Risc base de l'àrea i hora
            (df_map_risk['HOUR OCC'] == selected_hour).astype(int) * 30 + # Els crims exactament a l'hora tenen més pes
            np.random.rand(len(df_map_risk)) * 30 # Soroll per simular la variància local
        )
        
        # Normalitzar el risc per obtenir el gradient de color (Risc alt = Vermell)
        max_risk = df_map_risk['Simulated_Risk'].max()
        df_map_risk['color'] = df_map_risk['Simulated_Risk'].apply(
            lambda x: [255, max(0, int(255 - (x/max_risk)*255)), 0, 180] 
        ) 

        # Centrar el mapa al voltant de l'àrea seleccionada
        center_lat, center_lon = get_area_center(selected_area, df)
        
        # 6. Mostrar el Mapa
        st.map(
            df_map_risk[["lat", "lon", "color"]].head(5000), 
            color='color',
            size=5,
            zoom=13 
        )
        st.caption(f"Puntuació de Risc Base (LSTM/CNN): **{int(area_risk_score_base)} / 100**. Puntuació de Risc Personalitzada: **{int(area_risk_score_final)} / 100** (Ajustada per Edat: {selected_age}, Gènere: {selected_sex}).")
        
        st.markdown("---")

        # 7. CÀLCUL I MOSTRA DE DISTRIBUCIÓ DE CRIMES
        st.subheader("3. Distribució de Tipus de Crímens Predita")
        
        # S'utilitza el factor de risc demogràfic (demographic_factor) per ajustar el pes de la distribució
        crime_probabilities = get_area_crime_distribution(selected_area, selected_hour, df, risk_factor=demographic_factor)
        
        if crime_probabilities:
            # Preparar dades per a tots dos gràfics
            chart_data = pd.DataFrame(crime_probabilities.items(), columns=['Tipo de Crimen', 'Porcentaje'])
            chart_data['Porcentaje'] = chart_data['Porcentaje'].str.replace('%', '').astype(float)
            
            # Ajustar la proporció de columnes: Més espai per al gràfic de barres
            col_dist1, col_dist2 = st.columns([2, 1]) 
            
            # --- COLUMNA 1: GRÀFIC DE BARRES (CORREGIT) ---
            with col_dist1:
                st.write("**Percentatge de Probabilitat Ajustada per Tipus de Crimen:**")
                
                # 1. Definicions base
                base = alt.Chart(chart_data).encode(
                    y=alt.Y("Tipo de Crimen", title="", sort="-x"), 
                    tooltip=["Tipo de Crimen", alt.Tooltip("Porcentaje", format=".1f%")]
                ).properties(height=250).interactive()
                
                # 2. Capa de Barres
                bars = base.mark_bar().encode(
                    x=alt.X("Porcentaje", title="Probabilitat (%)"),
                    color=alt.Color("Porcentaje", scale=alt.Scale(range=['#66BB6A', '#EF5350'])) # Gradient de colors
                )
                
                # 3. Capa de Text (Etiquetes de Percentatge sobre les barres)
                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3,  # Desplaçament cap a la dreta del final de la barra
                ).encode(
                    x=alt.X("Porcentaje", title="Probabilitat (%)"),
                    text=alt.Text('Porcentaje', format='.1f%'), # Mostra el valor formatat
                    order=alt.Order('Porcentaje', sort='descending'),
                    color=alt.value('black') 
                )
                
                # 4. Combinar les capes de barres i text
                bar_chart_with_labels = (bars + text).resolve_scale(
                    color='independent'
                )
                
                st.altair_chart(bar_chart_with_labels, use_container_width=True)

            # --- COLUMNA 2: GRÀFIC DE PASTÍS ---
            with col_dist2:
                st.write("**Distribució de Risc (Top 5)**")
                # Gràfic de pastís amb format de percentatge corregit
                pie_chart = alt.Chart(chart_data).mark_arc(outerRadius=100).encode(
                    theta=alt.Theta(field="Porcentaje", type="quantitative"),
                    color=alt.Color(field="Tipo de Crimen", type="nominal"),
                    tooltip=["Tipo de Crimen", alt.Tooltip("Porcentaje", format=".1f%")]
                )
                st.altair_chart(pie_chart, use_container_width=True)

        else:
            st.warning("No hi ha dades suficients per calcular la distribució de crims per a aquesta hora i barri.")