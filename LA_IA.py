"""Script de gestión de datos de crímenes en LA - BALANCED EDITION.

1. Explore  -> Gráficas ORIGINALES intactas.
2. Model    -> Red Neuronal TensorFlow (Top 10 Crímenes).
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

    # Limpieza de nulos para variables clave de la IA
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

def explore_data(df: pd.DataFrame, output_dir: Path | None = None, show_graphs: bool = True) -> None:
    if df.empty: return
    print("[Explore] Generando batería completa de gráficos (ORIGINALES)...")
    
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top 15 Tipos de Crimen
    if "Crm Cd Desc" in df.columns:
        top_crimes = df["Crm Cd Desc"].value_counts().head(15)
        plt.figure(figsize=(11, 6))
        sns.barplot(x=top_crimes.values, y=top_crimes.index, orient="h")
        plt.title("Top 15 Tipos de Crimen")
        plt.tight_layout()
        _save_or_show(output_dir, "01_top_crimes.png", show_graphs)

    # 2. Evolución mensual
    if "MONTH" in df.columns:
        monthly = df.groupby("MONTH").size().sort_index()
        plt.figure(figsize=(12, 5))
        monthly.plot(color="#1f77b4")
        plt.title("Evolución mensual de crímenes")
        plt.ylabel("Casos")
        plt.xticks(rotation=90)
        plt.tight_layout()
        _save_or_show(output_dir, "02_evolucion_mensual.png", show_graphs)

    # 3. HEATMAP AÑO VS MES
    if {"YEAR", "MONTH"}.issubset(df.columns):
        pivot = df.pivot_table(index="YEAR", columns="MONTH", values="DR_NO", aggfunc="count")
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot.fillna(0), cmap="Blues")
        plt.title("Heatmap: Intensidad Año vs Mes")
        plt.tight_layout()
        _save_or_show(output_dir, "03_heatmap_year_month.png", show_graphs)

    # 4. Top Áreas
    if "AREA NAME" in df.columns:
        area_counts = df["AREA NAME"].value_counts().head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=area_counts.values, y=area_counts.index, orient="h")
        plt.title("Top Áreas con más crímenes")
        plt.tight_layout()
        _save_or_show(output_dir, "04_top_areas.png", show_graphs)

    # 5. Distribución por Hora
    if "TIME OCC" in df.columns:
        horas = (df["TIME OCC"] // 100).clip(lower=0, upper=23)
        plt.figure(figsize=(10, 4))
        sns.countplot(x=horas, color="#ff7f0e")
        plt.title("Distribución de crímenes por hora")
        plt.tight_layout()
        _save_or_show(output_dir, "05_distribution_hour.png", show_graphs)

    # 6. Distribución Edad Víctima
    if "Vict Age" in df.columns:
        edades = df["Vict Age"][df["Vict Age"] > 0]
        plt.figure(figsize=(10, 4))
        sns.histplot(edades, bins=40, kde=True, color="#2ca02c")
        plt.title("Distribución de Edad de Víctimas")
        plt.tight_layout()
        _save_or_show(output_dir, "06_victim_age.png", show_graphs)

    # 7. Sexo Víctima
    if "Vict Sex" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df["Vict Sex"].fillna("X"), palette="Set2")
        plt.title("Distribución por Sexo")
        plt.tight_layout()
        _save_or_show(output_dir, "07_victim_sex.png", show_graphs)

    # 8. Armas usadas
    if "Weapon Desc" in df.columns:
        weapons = df["Weapon Desc"].dropna().value_counts().head(15)
        plt.figure(figsize=(11, 6))
        sns.barplot(x=weapons.values, y=weapons.index, orient="h", color="#9467bd")
        plt.title("Top 15 Armas Usadas")
        plt.tight_layout()
        _save_or_show(output_dir, "08_top_weapons.png", show_graphs)

    # 9. HEATMAP: ZONA vs SEXO
    if "AREA NAME" in df.columns and "Vict Sex" in df.columns:
        df_sex = df[df["Vict Sex"].isin(["M", "F"])].copy()
        top_areas_list = df_sex["AREA NAME"].value_counts().head(15).index
        df_filtered = df_sex[df_sex["AREA NAME"].isin(top_areas_list)]
        
        ct = pd.crosstab(df_filtered["AREA NAME"], df_filtered["Vict Sex"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Reds", linewidths=.5)
        plt.title("MAPA DE CALOR: Zonas Peligrosas vs Sexo")
        plt.tight_layout()
        _save_or_show(output_dir, "09_heatmap_zona_sexo.png", show_graphs)

    # 10. Desbalance de Clases
    if "Crm Cd Desc" in df.columns:
        top = df["Crm Cd Desc"].value_counts().head(30)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=top.values, y=top.index, orient="h", color="#8c564b")
        plt.xscale("log")
        plt.title("Top 30 Crímenes (Escala Logarítmica)")
        plt.tight_layout()
        _save_or_show(output_dir, "10_class_imbalance.png", show_graphs)

def prepare_data(df: pd.DataFrame):
    print("[Prepare] Seleccionando variables para IA (Estrategia Top 10)...")
    target_col = "Crm Cd Desc"
    
    feature_cols = [
        "YEAR", "LAT", "LON", "TIME OCC", "Vict Age", "MONTH_NUM", 
        "Premis Cd", "Weapon Used Cd", "DAY_OF_WEEK"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    work_df = df[feature_cols + [target_col]].dropna()
    
    # --- ESTRATEGIA TOP 10 ---
    # Aquí es donde cambiamos el número para buscar el equilibrio
    top_classes = work_df[target_col].value_counts().head(10).index
    work_df = work_df[work_df[target_col].isin(top_classes)]
    
    print(f"[Prepare] Clases a predecir: {list(top_classes)}")
    
    # 1. Label Encoding
    le_target = LabelEncoder()
    y = le_target.fit_transform(work_df[target_col])
    
    # 2. Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(work_df[feature_cols])
    
    return X, y, le_target, scaler

def split_data(X, y):
    if X is None: return None
    print("[Split] Train/Test (80/20)...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_neural_model(X_train, X_test, y_train, y_test, num_classes):
    print("\n--- INICIO ETAPA: RED NEURONAL (TensorFlow) ---")
    
    model = models.Sequential([
        # Capa 1
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Capa 2
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Capa 3
        layers.Dense(64, activation='relu'),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("[Train] Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n[Evaluate] Evaluando modelo...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Results] PRECISIÓN FINAL (Accuracy) en Test: {acc:.2%}")
    
    return model

def _save_or_show(output_dir, filename, show):
    if output_dir:
        plt.savefig(output_dir / filename, dpi=100)
        print(f"[Graph] Guardado: {filename}")
    if show: plt.show()
    else: plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    
    df = collect_data(DATA_PATH)
    if args.sample: df = df.sample(min(args.sample, len(df)), random_state=42)
    
    df = clean_data(df)
    
    # Generamos gráficas a menos que se indique lo contrario
    explore_data(df, Path("output/figures"), show_graphs=not args.no_graphs)
    
    X, y, le, scaler = prepare_data(df)
    if X is not None:
        splits = split_data(X, y)
        if splits:
            train_neural_model(*splits, num_classes=len(le.classes_))

if __name__ == "__main__":
    main()