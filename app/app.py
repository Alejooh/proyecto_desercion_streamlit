import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import base64

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(page_title="Deserción estudiantil", layout="centered")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_desercion_logit.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "data_ml.csv"

NUM_FEATURES = [
    "nivel",
    "materias_cursadas",
    "promedio_periodo",
    "asistencia_prom",
    "tasa_reprobacion",
    "max_no_vez",
    "repitencia_prom",
]
CAT_FEATURES = ["carrera"]
TARGET = "DESERCION_NEXT"

# =========================
# FUNCIONES AUXILIARES
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def plot_hist(series: pd.Series, title: str, bins: int = 20):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=140)
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(series.name, fontsize=9)
    ax.set_ylabel("Frecuencia", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def model_uses_categorical(model) -> bool:
    """True si el pipeline tiene un ColumnTransformer con un transformer llamado 'cat'."""
    try:
        prep = model.named_steps["prep"]
        return hasattr(prep, "named_transformers_") and ("cat" in prep.named_transformers_)
    except Exception:
        return False

def get_logit_importances(model, uses_cat: bool):
    """Extrae importancias por |coef| para Regresión Logística. Compatible con y sin OHE."""
    try:
        if "clf" not in model.named_steps:
            return None

        coefs = model.named_steps["clf"].coef_[0]

        if uses_cat:
            prep = model.named_steps["prep"]
            ohe = prep.named_transformers_.get("cat", None)
            if ohe is None:
                return None
            cat_names = ohe.get_feature_names_out(CAT_FEATURES)
            feature_names = NUM_FEATURES + list(cat_names)
        else:
            feature_names = NUM_FEATURES

        if len(feature_names) != len(coefs):
            return None

        imp = pd.DataFrame(
            {
                "feature": feature_names,
                "coef": coefs,
                "abs_coef": np.abs(coefs),
            }
        ).sort_values("abs_coef", ascending=False)

        imp["efecto"] = np.where(imp["coef"] > 0, "Aumenta riesgo", "Disminuye riesgo")
        return imp
    except Exception:
        return None

def plot_confusion(cm: np.ndarray):
    """Matriz de confusión grande y legible."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=150)
    ax.imshow(cm)
    ax.set_title("Matriz de confusión", fontsize=12)
    ax.set_xlabel("Predicción", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=10)
    ax.set_yticklabels(["0", "1"], fontsize=10)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_importance(top: pd.DataFrame):
    """Gráfico grande y legible de importancia (|coef|)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.barh(top["feature"], top["abs_coef"])
    ax.set_title("Importancia de variables (|coef|)", fontsize=12)
    ax.set_xlabel("|coef|", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =========================
# UI - ENCABEZADO
# =========================
st.title("Predicción de deserción estudiantil")
st.caption("Modelo: regresión logística. Objetivo: DESERCION_NEXT (no vuelve en el siguiente período)")

c1, c2 = st.columns(2)
with c1:
    st.write("Modelo:", str(MODEL_PATH))
    st.write("Existe modelo:", MODEL_PATH.exists())
with c2:
    st.write("Datos procesados:", str(DATA_PATH))
    st.write("Existe data_ml.csv:", DATA_PATH.exists())

if not MODEL_PATH.exists():
    st.error("No se encontró el modelo. Debe existir en: models/modelo_desercion_logit.pkl")
    st.stop()

if not DATA_PATH.exists():
    st.error("No se encontró el archivo de datos. Debe existir en: data/processed/data_ml.csv")
    st.stop()

model = load_model()
data = load_data()

USES_CAT = model_uses_categorical(model)
FEATURES = NUM_FEATURES + CAT_FEATURES if USES_CAT else NUM_FEATURES

st.caption(f"El pipeline {'USA' if USES_CAT else 'NO usa'} variables categóricas. Features usadas: {FEATURES}")

tabs = st.tabs(["EDA", "Métricas", "Predicción", "Importancia variables"])

# =========================
# TAB 1: EDA
# =========================
with tabs[0]:
    st.subheader("Análisis exploratorio (EDA)")

    st.write("Shape:", data.shape)
    st.write("Estudiantes únicos:", data["ESTUDIANTE"].nunique() if "ESTUDIANTE" in data.columns else "N/A")
    st.write("Períodos únicos:", data["PERIODO"].nunique() if "PERIODO" in data.columns else "N/A")

    if "PERIODO" in data.columns:
        st.write("Filas por período:")
        st.write(data["PERIODO"].value_counts())

    st.dataframe(data.head(10), use_container_width=True)

    if TARGET in data.columns:
        st.markdown("Distribución del target")
        vc = data[TARGET].value_counts(dropna=False)
        st.write(vc.rename_axis(TARGET).to_frame("count"))
        st.write(data[TARGET].value_counts(normalize=True).round(3).rename("proporción"))

    st.markdown("Distribuciones de variables numéricas")
    cols = st.columns(2)
    for i, col in enumerate(NUM_FEATURES):
        with cols[i % 2]:
            if col in data.columns:
                plot_hist(data[col], title=f"Distribución: {col}", bins=20)

    st.markdown("Estadísticas descriptivas (numéricas)")
    st.dataframe(data[NUM_FEATURES].describe().T, use_container_width=True)

# =========================
# TAB 2: MÉTRICAS
# =========================
with tabs[1]:
    st.subheader("Métricas de evaluación del modelo (TEST = último período)")

    required = FEATURES + [TARGET, "PERIODO"]
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        st.error(f"Faltan columnas en data_ml.csv: {missing_cols}")
        st.stop()

    # Último periodo por PERIODO_KEY (si existe)
    if "PERIODO_KEY" in data.columns:
        last_period = data.sort_values("PERIODO_KEY")["PERIODO"].iloc[-1]
    else:
        last_period = sorted(data["PERIODO"].unique().tolist())[-1]

    st.write("Periodo usado como TEST:", last_period)

    test_df = data[data["PERIODO"] == last_period].copy()
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    st.write("Distribución real en TEST:")
    st.write(y_test.value_counts().rename_axis(TARGET).to_frame("count"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision (clase 1)", f"{prec:.3f}")
    m3.metric("Recall (clase 1)", f"{rec:.3f}")
    m4.metric("F1 (clase 1)", f"{f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    st.markdown("Matriz de confusión (tabla)")
    st.dataframe(
        pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]),
        use_container_width=True,
    )

    st.markdown("Matriz de confusión (gráfico)")
    plot_confusion(cm)

    st.markdown("Reporte de clasificación")
    st.text(classification_report(y_test, y_pred, digits=3))

# =========================
# TAB 3: PREDICCIÓN MANUAL
# =========================
with tabs[2]:
    st.subheader("Predicción para un estudiante (ingreso manual)")
    st.markdown("Ingresa las variables agregadas del estudiante para el período actual.")

    threshold = st.slider(
        "Umbral de clasificación (clase 1 = deserción)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        nivel = st.number_input("Nivel", min_value=1, max_value=30, value=1, step=1)
        materias_cursadas = st.number_input("Materias cursadas", min_value=1, max_value=15, value=5, step=1)
        max_no_vez = st.number_input("Máx NO. VEZ", min_value=1, max_value=10, value=1, step=1)

    with col2:
        promedio_periodo = st.number_input(
            "Promedio del período (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.01
        )
        asistencia_prom = st.number_input(
            "Asistencia promedio (0-100)", min_value=0.0, max_value=100.0, value=85.0, step=1.0
        )

    with col3:
        tasa_reprobacion = st.number_input(
            "Tasa de reprobación (0-1)", min_value=0.0, max_value=1.0, value=0.20, step=0.01
        )
        repitencia_prom = st.number_input(
            "Repitencia promedio (NO. VEZ)", min_value=1.0, max_value=10.0, value=1.2, step=0.1
        )

    row = {
        "nivel": int(nivel),
        "materias_cursadas": int(materias_cursadas),
        "promedio_periodo": float(promedio_periodo),
        "asistencia_prom": float(asistencia_prom),
        "tasa_reprobacion": float(tasa_reprobacion),
        "max_no_vez": int(max_no_vez),
        "repitencia_prom": float(repitencia_prom),
    }

    if USES_CAT:
        carreras = sorted(data["carrera"].dropna().unique().tolist()) if "carrera" in data.columns else []
        carrera_default = carreras[0] if carreras else "CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL"
        carrera = st.selectbox(
            "Carrera",
            options=carreras if carreras else [carrera_default],
            index=0,
        )
        row["carrera"] = str(carrera)

    input_df = pd.DataFrame([row])

    st.write("Input usado para predicción:")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predecir riesgo"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0, 1]
            pred = int(proba >= threshold)
            st.write("Probabilidad de deserción (siguiente período):", f"{proba:.3f}")
            st.write("Umbral usado:", f"{threshold:.2f}")
        else:
            pred = int(model.predict(input_df)[0])
            proba = None

        if pred == 1:
            st.error("Riesgo ALTO de deserción (DESERCION_NEXT = 1)")
        else:
            st.success("Riesgo BAJO de deserción (DESERCION_NEXT = 0)")

# =========================
# TAB 4: IMPORTANCIA
# =========================
with tabs[3]:
    st.subheader("Variables más importantes (Regresión Logística)")

    imp = get_logit_importances(model, USES_CAT)
    if imp is None:
        st.warning("No se pudo extraer importancia de variables desde el pipeline.")
    else:
        st.dataframe(imp, use_container_width=True)

        top = imp.head(min(10, len(imp))).sort_values("abs_coef")

        st.markdown("Gráfico de importancia")
        plot_importance(top)

        st.markdown(
            "Interpretación: coeficiente positivo incrementa la probabilidad de DESERCION_NEXT=1; "
            "coeficiente negativo reduce el riesgo."
        )