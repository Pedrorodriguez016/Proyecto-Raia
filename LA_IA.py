"""Script de gestión de datos de crímenes en LA - BALANCED EDITION.
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# --- TENSORFLOW ---
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_PATH = Path("Crime_Data_from_2020_to_Present.csv")

def collect_data(path: Path) -> pd.DataFrame:
    print(f"[Collect] Leyendo archivo: {path}")
    try:
        df = pd.read_csv(path)
        print(f"[Collect] Registros cargados: {len(df):,}")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en {path}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    print("[Clean] Limpiando datos y normalizando coordenadas...")
    
    columnas_a_eliminar = [
        "Date Rptd", "AREA", "Part 1-2", "Crm Cd", "Mocodes", 
        "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4", 
        "Cross Street", "Status", "Status Desc"
    ]
    df = df.drop(columns=[c for c in columnas_a_eliminar if c in df.columns])

    # Fechas
    if "DATE OCC" in df.columns:
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
        df = df.dropna(subset=["DATE OCC"])
        df["YEAR"] = df["DATE OCC"].dt.year
        df["MONTH"] = df["DATE OCC"].dt.to_period("M").astype(str)
        df["MONTH_NUM"] = df["DATE OCC"].dt.month
        df["DAY_OF_WEEK"] = df["DATE OCC"].dt.dayofweek

    # Limpieza IA
    cols_clave_ia = ["Premis Cd", "Weapon Used Cd"]
    for col in cols_clave_ia:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Coordenadas
    if "LAT" in df.columns and "LON" in df.columns:
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        
        def normalize(v, limit):
            if pd.isna(v): return None
            v = float(v)
            while abs(v) > limit and abs(v) > 0:
                v /= 10.0
            return v

        df["LAT"] = df["LAT"].apply(lambda x: normalize(x, 90))
        df["LON"] = df["LON"].apply(lambda x: normalize(x, 180))
        df = df[(df["LAT"] != 0) & (df["LON"] != 0)]
        df = df.dropna(subset=["LAT", "LON"])
        
    print(f"[Clean] Registros válidos finales: {len(df):,}")
    return df


# ----------------------------------------------------------
# CLASIFICACIÓN DE SEVERIDAD
# ----------------------------------------------------------
def classify_crime_severity(crime_desc):
    """Clasifica un crimen en PELIGROSO o SEGURO según riesgo personal"""
    if pd.isna(crime_desc):
        return "PELIGROSO"  # Por defecto, ser conservador
    
    crime_upper = str(crime_desc).upper()
    
    # PELIGROSO: Crímenes violentos o con riesgo personal directo
    DANGEROUS_KEYWORDS = [
        # Violencia grave
        'HOMICIDE', 'MURDER', 'MANSLAUGHTER',
        'RAPE', 'SODOMY', 'SEXUAL', 'LEWD',
        'KIDNAPPING', 'ABDUCTION',
        
        # Robos con violencia
        'ROBBERY',  # Robo con violencia/amenaza
        
        # Asaltos y agresiones
        'ASSAULT', 'BATTERY', 'AGGRAVATED',
        'INTIMATE PARTNER',  # Violencia doméstica
        
        # Armas
        'WEAPON', 'BRANDISH', 'SHOTS FIRED', 'DISCHARGE FIREARM',
        'BOMB', 'EXPLOSIVE',
        
        # Amenazas directas
        'CRIMINAL THREATS', 'THREATENING',
        
        # Allanamientos (persona puede estar presente)
        'BURGLARY', 'BREAKING',
        
        # Robos personales
        'PURSE SNATCHING', 'PICKPOCKET', 'THEFT, PERSON',
        
        # Secuestro/contacto forzado
        'CHILD STEALING', 'STALKING'
    ]
    
    # 🟢 SEGURO: Crímenes contra propiedad sin contacto personal
    # (Todo lo demás: robos de vehículos, fraudes, vandalismos, hurtos, etc.)
    # Si NO contiene ninguna palabra peligrosa, es SEGURO
    
    if any(keyword in crime_upper for keyword in DANGEROUS_KEYWORDS):
        return "PELIGROSO"
    else:
        return "SEGURO"


# ----------------------------------------------------------
# 🔥 NUEVA VERSIÓN ROBUSTA DE prepare_data() - MULTI-SALIDA
# ----------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    print("[Prepare] Seleccionando variables para IA...")
    target_col = "Crm Cd Desc"
    
    # 1. Definir columnas
    feature_cols = [
        "YEAR", "LAT", "LON", "TIME OCC", "Vict Age", "Vict Sex", "MONTH_NUM", 
        "Premis Cd", "Weapon Used Cd", "DAY_OF_WEEK"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Creamos work_df ANTES de intentar acceder a él
    work_df = df[feature_cols + [target_col]].dropna()
    
    # Sex Encoding manual (F=0, M=1, X=2, Otro=2)
    if "Vict Sex" in work_df.columns:
        work_df["Vict Sex"] = work_df["Vict Sex"].map({'F': 0, 'M': 1, 'X': 2}).fillna(2).astype(int)
    
    # 2. FILTRAR SOLO EL TOP 10
    top_classes = work_df[target_col].value_counts().head(10).index
    work_df = work_df[work_df[target_col].isin(top_classes)]
    
    # NUEVO: Agregar columna de SEVERIDAD (2 NIVELES)
    print("[Prepare] Clasificando crímenes por severidad (PELIGROSO/SEGURO)...")
    work_df['SEVERITY'] = work_df[target_col].apply(classify_crime_severity)
    
    severity_counts = work_df['SEVERITY'].value_counts()
    print(f"  PELIGROSO: {severity_counts.get('PELIGROSO', 0):,} casos")
    print(f"  SEGURO:    {severity_counts.get('SEGURO', 0):,} casos")
    
    # Mostrar balance
    total = len(work_df)
    pct_dangerous = (severity_counts.get('PELIGROSO', 0) / total * 100) if total > 0 else 0
    print(f"  Balance: {pct_dangerous:.1f}% peligrosos, {100-pct_dangerous:.1f}% seguros")
    
    # --- BALANCEO CON CLASS WEIGHTS ---
    print("[Prepare] Usando todo el dataset (Top 10) y calculando pesos de clase...")

    print(f"[Prepare] Clases a predecir: {list(top_classes)}")
    
    # Encoders para ambas salidas
    le_crime = LabelEncoder()
    le_severity = LabelEncoder()
    
    y_crime = le_crime.fit_transform(work_df[target_col])
    y_severity = le_severity.fit_transform(work_df['SEVERITY'])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(work_df[feature_cols])
    
    return X, y_crime, y_severity, le_crime, le_severity, scaler, feature_cols

def split_data(X, y_crime, y_severity):
    if X is None:
        return None
    print("[Split] 80/20 Train/Test...")
    # Split debe hacerse igual para ambas salidas
    X_train, X_test, y_crime_train, y_crime_test, y_severity_train, y_severity_test = train_test_split(
        X, y_crime, y_severity, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_crime  # Estratificar por tipo de crimen
    )
    return X_train, X_test, y_crime_train, y_crime_test, y_severity_train, y_severity_test


def train_neural_model(X_train, X_test, y_crime_train, y_crime_test, y_severity_train, y_severity_test, num_crime_classes, num_severity_classes):
    print("\n--- ENTRENANDO RED NEURONAL MULTI-SALIDA ---")
    
    # Calcular pesos de clase para balancear el entreno (solo para crímenes)
    from sklearn.utils import class_weight
    import numpy as np
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_crime_train),
        y=y_crime_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"[Training] Pesos de clase calculados para crímenes")

    # 🔥 ARQUITECTURA MULTI-SALIDA
    # Input compartido
    inputs = layers.Input(shape=(X_train.shape[1],))
    
    # Capas compartidas (feature extraction)
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    # SALIDA 1: Tipo de Crimen (principal)
    crime_output = layers.Dense(64, activation='relu', name='crime_branch')(shared)
    crime_output = layers.Dropout(0.2)(crime_output)
    crime_output = layers.Dense(num_crime_classes, activation='softmax', name='crime_type')(crime_output)
    
    # SALIDA 2: Severidad (secundaria)
    severity_output = layers.Dense(32, activation='relu', name='severity_branch')(shared)
    severity_output = layers.Dropout(0.2)(severity_output)
    severity_output = layers.Dense(num_severity_classes, activation='softmax', name='severity_level')(severity_output)
    
    # Crear modelo con múltiples salidas
    model = models.Model(inputs=inputs, outputs=[crime_output, severity_output])
    
    # Compilar con pérdidas y métricas para cada salida
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'crime_type': 'sparse_categorical_crossentropy',
            'severity_level': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'crime_type': 1.0,  # Peso principal
            'severity_level': 0.5  # Peso secundario
        },
        metrics={
            'crime_type': ['accuracy'],
            'severity_level': ['accuracy']
        }
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_crime_type_loss',  # Monitorear la pérdida principal
        patience=10,
        restore_best_weights=True,
        mode='min',  # Minimizar la pérdida
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_crime_type_loss', 
        factor=0.5, 
        patience=5, 
        mode='min',  # Minimizar la pérdida
        verbose=1
    )

    history = model.fit(
        X_train, 
        {'crime_type': y_crime_train, 'severity_level': y_severity_train},
        epochs=50, 
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        # NOTA: class_weight no es compatible con modelos multi-salida
        # El balanceo se maneja con loss_weights en compile()
        verbose=1
    )

    # Evaluar ambas salidas
    results = model.evaluate(
        X_test, 
        {'crime_type': y_crime_test, 'severity_level': y_severity_test}, 
        verbose=0
    )
    
    print(f"\n[Results] Crime Type Accuracy: {results[3]:.2%}")  # crime_type_accuracy
    print(f"[Results] Severity Accuracy: {results[4]:.2%}")    # severity_level_accuracy

    return model


# ----------------------------------------------------------
# 🔥 GUARDADO DEL MODELO + ARTEFACTOS (MULTI-SALIDA)
# ----------------------------------------------------------
def save_artifacts(model, scaler, label_encoder_crime, label_encoder_severity, feature_cols):
    print("\n[Save] Guardando modelo y artefactos para Streamlit...")

    model.save("crime_model.keras")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(label_encoder_crime, "label_encoder.joblib")
    joblib.dump(label_encoder_severity, "severity_encoder.joblib")  # NUEVO
    joblib.dump(feature_cols, "feature_cols.joblib")

    print("Modelo multi-salida guardado correctamente:")
    print(" - crime_model.keras")
    print(" - scaler.joblib")
    print(" - label_encoder.joblib (tipos de crimen)")
    print(" - severity_encoder.joblib (niveles de severidad)")
    print(" - feature_cols.joblib")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()

    df = collect_data(DATA_PATH)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42)

    df = clean_data(df)

    # PREPARAR DATOS (ahora retorna más valores)
    X, y_crime, y_severity, le_crime, le_severity, scaler, feature_cols = prepare_data(df)

    # SPLIT (ahora maneja ambas salidas)
    splits = split_data(X, y_crime, y_severity)
    if splits:
        X_train, X_test, y_crime_train, y_crime_test, y_severity_train, y_severity_test = splits
        
        model = train_neural_model(
            X_train, X_test, 
            y_crime_train, y_crime_test,
            y_severity_train, y_severity_test,
            num_crime_classes=len(le_crime.classes_),
            num_severity_classes=len(le_severity.classes_)
        )

        # GUARDAR MODELO + ARTEFACTOS (con ambos encoders)
        save_artifacts(model, scaler, le_crime, le_severity, feature_cols)


if __name__ == "__main__":
    main()
