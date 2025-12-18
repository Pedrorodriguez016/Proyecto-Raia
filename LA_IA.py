from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_PATH = Path(__file__).parent / "Crime_Data_from_2020_to_Present.csv"

def collect_data(path: Path) -> pd.DataFrame:
    print(f"[Collect] Leyendo archivo: {path}")
    if not path.is_file():
        print(f"ERROR: No se encontró el archivo en {path}. Por favor, asegúrese de que el archivo CSV exista en la ruta especificada.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        print(f"[Collect] Registros cargados: {len(df):,}")
        return df
    except Exception as e:
        print(f"ERROR al leer el archivo: {e}")
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

# Helper to save or display figures
def _save_or_show(output: Path | None, filename: str, show: bool) -> None:
    # Save figure (if output path provided) and optionally display it
    if output:
        output.mkdir(parents=True, exist_ok=True)
        plt.savefig(output / filename, bbox_inches="tight")
    if show:
        plt.show()
    # Close the figure to free memory
    plt.close()

def explore_data(df: pd.DataFrame, output: Path | None = None, show_graphs: bool = True) -> None:
    # Generate exploratory visualizations and save/display them
    if df.empty:
        return
    print("[Explore] Generando batería completa de gráficos...")

    # 1. Top 15 Tipos de Crimen
    if "Crm Cd Desc" in df.columns:
        top_crimes = df["Crm Cd Desc"].value_counts().head(15)
        plt.figure(figsize=(11, 6))
        sns.barplot(x=top_crimes.values, y=top_crimes.index, orient="h")
        plt.title("Top 15 Tipos de Crimen")
        plt.tight_layout()
        _save_or_show(output, "01_top_crimes.png", show_graphs)

    # 2. Evolución mensual (Línea temporal)
    if "MONTH" in df.columns:
        monthly = df.groupby("MONTH").size().sort_index()
        plt.figure(figsize=(12, 5))
        monthly.plot(color="#1f77b4")
        plt.title("Evolución mensual de crímenes")
        plt.ylabel("Casos")
        plt.xticks(rotation=90)
        plt.tight_layout()
        _save_or_show(output, "02_evolucion_mensual.png", show_graphs)

    # 3. HEATMAP AÑO VS MES
    if {"YEAR", "MONTH"}.issubset(df.columns):
        pivot = df.pivot_table(index="YEAR", columns="MONTH", values="DR_NO", aggfunc="count")
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot.fillna(0), cmap="Blues")
        plt.title("Heatmap: Intensidad Año vs Mes")
        plt.tight_layout()
        _save_or_show(output, "03_heatmap_year_month.png", show_graphs)

    # 4. Top Áreas (Comisarías)
    if "AREA NAME" in df.columns:
        area_counts = df["AREA NAME"].value_counts().head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=area_counts.values, y=area_counts.index, orient="h")
        plt.title("Top Áreas con más crímenes")
        plt.tight_layout()
        _save_or_show(output, "04_top_areas.png", show_graphs)

    # 5. Distribución por Hora
    if "TIME OCC" in df.columns:
        horas = (df["TIME OCC"] // 100).clip(lower=0, upper=23)
        plt.figure(figsize=(10, 4))
        sns.countplot(x=horas, color="#ff7f0e")
        plt.title("Distribución de crímenes por hora")
        plt.tight_layout()
        _save_or_show(output, "05_distribution_hour.png", show_graphs)

    # 6. Distribución Edad Víctima
    if "Vict Age" in df.columns:
        edades = df["Vict Age"][df["Vict Age"] > 0]
        plt.figure(figsize=(10, 4))
        sns.histplot(edades, bins=40, kde=True, color="#2ca02c")
        plt.title("Distribución de Edad de Víctimas")
        plt.tight_layout()
        _save_or_show(output, "06_victim_age.png", show_graphs)

    # 7. Sexo Víctima
    if "Vict Sex" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df["Vict Sex"].fillna("X"), hue=df["Vict Sex"].fillna("X"), palette="Set2", legend=False)
        plt.title("Distribución por Sexo")
        plt.tight_layout()
        _save_or_show(output, "07_victim_sex.png", show_graphs)

    # 8. Armas usadas
    if "Weapon Desc" in df.columns:
        weapons = df["Weapon Desc"].dropna().value_counts().head(15)
        plt.figure(figsize=(11, 6))
        sns.barplot(x=weapons.values, y=weapons.index, orient="h", color="#9467bd")
        plt.title("Top 15 Armas Usadas")
        plt.tight_layout()
        _save_or_show(output, "08_top_weapons.png", show_graphs)

    # 9. HEATMAP: ZONA vs SEXO
    if "AREA NAME" in df.columns and "Vict Sex" in df.columns:
        df_sex = df[df["Vict Sex"].isin(["M", "F"])].copy()
        top_areas_list = df_sex["AREA NAME"].value_counts().head(15).index
        df_filtered = df_sex[df_sex["AREA NAME"].isin(top_areas_list)]
        ct = pd.crosstab(df_filtered["AREA NAME"], df_filtered["Vict Sex"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Reds", linewidths=.5)
        plt.title("MAPA DE CALOR: Zonas Peligrosas vs Sexo (Quién y Dónde)")
        plt.tight_layout()
        _save_or_show(output, "09_heatmap_zona_sexo.png", show_graphs)

    # 10. Desbalance de Clases
    if "Crm Cd Desc" in df.columns:
        top = df["Crm Cd Desc"].value_counts().head(30)
        plt.figure(figsize=(12, 7))
        sns.barplot(x=top.values, y=top.index, orient="h", color="#8c564b")
        plt.xscale("log")
        plt.title("Top 30 Crímenes (Escala Logarítmica)")
        plt.tight_layout()
        _save_or_show(output, "10_class_imbalance.png", show_graphs)


def classify_crime_severity(crime_desc):
    """Clasifica un crimen en PELIGROSO o SEGURO según riesgo personal"""
    if pd.isna(crime_desc):
        return "PELIGROSO"  # Por defecto, ser conservador
    
    crime_upper = str(crime_desc).upper()
    
    # PELIGROSO: Crímenes violentos o con riesgo personal directo
    DANGEROUS_KEYWORDS = [
        'HOMICIDE', 'MURDER', 'MANSLAUGHTER',
        'RAPE', 'SODOMY', 'SEXUAL', 'LEWD',
        'KIDNAPPING', 'ABDUCTION','ROBBERY',
        'ASSAULT', 'BATTERY', 'AGGRAVATED',
        'INTIMATE PARTNER','WEAPON', 'BRANDISH',
        'SHOTS FIRED', 'DISCHARGE FIREARM','BOMB',
        'EXPLOSIVE','CRIMINAL THREATS', 'THREATENING',
        'BURGLARY', 'BREAKING','PURSE SNATCHING',
        'PICKPOCKET', 'THEFT, PERSON','CHILD STEALING',
        'STALKING'
    ]
    
    if any(keyword in crime_upper for keyword in DANGEROUS_KEYWORDS):
        return "PELIGROSO"
    else:
        return "SEGURO"


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
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_crime_train),
        y=y_crime_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"[Training] Pesos de clase calculados para crímenes")

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--sample", type=int, help="Number of rows to sample from the dataset (optional)")
    args = p.parse_args()

    df = collect_data(DATA_PATH)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42)

    df = clean_data(df)

    # ------------------------------------------------------------
    # Exploratory visualisations (optional, controlled by --no-graphs)
    # ------------------------------------------------------------
    if not args.no_graphs:
        # Save plots in a folder called "output/figures" next to the script
        from pathlib import Path
        output_path = Path(__file__).parent / "output" / "figures"
        explore_data(df, output=output_path, show_graphs=True)

    X, y_crime, y_severity, le_crime, le_severity, scaler, feature_cols = prepare_data(df)

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
        
        save_artifacts(model, scaler, le_crime, le_severity, feature_cols)


if __name__ == "__main__":
    main()
