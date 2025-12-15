"""Script de gestión de datos de crímenes en LA - BALANCED EDITION.

1. Explore  -> Gráficas ORIGINALES intactas.
2. Model    -> Red Neuronal TensorFlow (Top 10 Crímenes).
3. Save     -> Guarda modelo + scaler + label encoder + feature_cols para Streamlit.
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
# 🔥 NUEVA VERSIÓN ROBUSTA DE prepare_data()
# ----------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    print("[Prepare] Seleccionando variables para IA...")
    target_col = "Crm Cd Desc"
    
    # 1. Definir columnas
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
    
    # --- 🔥 NUEVO: BALANCEO CON CLASS WEIGHTS (Mejor que Undersampling agresivo) ---
    print("[Prepare] Usando todo el dataset (Top 10) y calculando pesos de clase...")
    
    # ----------------------------------------------------

    print(f"[Prepare] Clases a predecir: {list(top_classes)}")
    
    le_target = LabelEncoder()
    y = le_target.fit_transform(work_df[target_col])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(work_df[feature_cols])
    
    return X, y, le_target, scaler, feature_cols

def split_data(X, y):
    if X is None:
        return None
    print("[Split] 80/20 Train/Test...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_neural_model(X_train, X_test, y_train, y_test, num_classes):
    print("\n--- ENTRENANDO RED NEURONAL ---")
    
    # Calcular pesos de clase para balancear el entreno
    from sklearn.utils import class_weight
    import numpy as np
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"[Training] Pesos de clase calculados: {class_weights_dict}")

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Results] Accuracy Test: {acc:.2%}")

    return model


# ----------------------------------------------------------
# 🔥 GUARDADO DEL MODELO + ARTEFACTOS
# ----------------------------------------------------------
def save_artifacts(model, scaler, label_encoder, feature_cols):
    print("\n[Save] Guardando modelo y artefactos para Streamlit...")

    model.save("crime_model.keras")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(label_encoder, "label_encoder.joblib")
    joblib.dump(feature_cols, "feature_cols.joblib")

    print("✅ Modelo guardado correctamente:")
    print(" - crime_model.keras")
    print(" - scaler.joblib")
    print(" - label_encoder.joblib")
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

    # PREPARAR DATOS
    X, y, le, scaler, feature_cols = prepare_data(df)

    # SPLIT
    splits = split_data(X, y)
    if splits:
        model = train_neural_model(*splits, num_classes=len(le.classes_))

        # GUARDAR MODELO + ARTEFACTOS
        save_artifacts(model, scaler, le, feature_cols)


if __name__ == "__main__":
    main()
