"""
Dashboard Streamlit del clasificador de reclamos municipales.

Interfaz interactiva para que un funcionario (o un vecino, en una demo)
redacte un reclamo y vea en que categoria municipal lo clasificaria el
modelo. El dashboard **no carga el modelo directamente**: siempre habla con
la API (`src/api.py`) via HTTP, tal como esta pensada la arquitectura del
proyecto (ver README.md, seccion "Arquitectura de inferencia"). Esto permite
escalar la API y el dashboard como servicios independientes (dos contenedores
en `docker-compose.yml`) y evita duplicar la logica de inferencia.

Uso:
    streamlit run src/dashboard.py

Variable de entorno:
    API_URL  URL base de la API (default: http://localhost:8000). Nunca se
             hardcodea una URL de produccion en el codigo, tal como exige
             CLAUDE.md.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 10  # segundos, para no dejar la interfaz colgada si la API no responde

st.set_page_config(
    page_title="Clasificador de Reclamos Municipales",
    page_icon="🏛️",
    layout="centered",
)


@st.cache_resource
def get_http_session() -> requests.Session:
    """Reutiliza una unica sesion HTTP para todas las llamadas a la API en
    vez de abrir una conexion nueva en cada rerun de Streamlit (cada
    interaccion del usuario vuelve a ejecutar este script completo)."""
    return requests.Session()


def check_health() -> dict:
    """Consulta `GET /health`. Nunca deja que una excepcion de red llegue a
    la interfaz: siempre devuelve un dict entendible."""
    session = get_http_session()
    try:
        response = session.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return {"ok": bool(data.get("model_loaded")), "detail": data.get("detail")}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "detail": f"No se pudo conectar con la API en '{API_URL}'. ¿Esta corriendo el servicio?"}
    except requests.exceptions.Timeout:
        return {"ok": False, "detail": "La API demoro demasiado en responder."}
    except requests.exceptions.RequestException:
        return {"ok": False, "detail": "Ocurrio un problema inesperado al consultar el estado de la API."}


@st.cache_data(ttl=300)
def get_metrics() -> dict | None:
    """Consulta `GET /metrics`. Se cachea 5 minutos: las metricas solo
    cambian si se vuelve a correr `src/train.py`, no en cada interaccion."""
    session = get_http_session()
    try:
        response = session.get(f"{API_URL}/metrics", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def classify_complaint(texto: str) -> tuple[dict | None, str | None]:
    """Llama a `POST /predict`. Devuelve (resultado, mensaje_de_error).

    Traduce cualquier problema (API caida, timeout, texto invalido, modelo
    no disponible) a un mensaje en espanol entendible — nunca se le muestra
    al usuario un traceback ni un error crudo de `requests`.
    """
    session = get_http_session()
    try:
        response = session.post(
            f"{API_URL}/predict", json={"texto": texto}, timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.ConnectionError:
        return None, f"No se pudo conectar con la API en '{API_URL}'. ¿Esta corriendo el servicio?"
    except requests.exceptions.Timeout:
        return None, "La API demoro demasiado en responder. Intenta nuevamente."
    except requests.exceptions.RequestException as exc:
        return None, f"Ocurrio un problema de red inesperado: {exc}"

    if response.status_code == 200:
        return response.json(), None
    if response.status_code in (422, 503):
        detalle = response.json().get("detail", "La API rechazo la solicitud.")
        return None, detalle
    return None, f"La API respondio con un error inesperado (codigo {response.status_code})."


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title("🏛️ Clasificador de Reclamos Municipales")
st.caption(
    "Clasifica automaticamente un reclamo ciudadano en una de 6 categorias "
    "municipales, usando el modelo entrenado con `src/train.py` y servido "
    "por la API de FastAPI."
)

if "historial" not in st.session_state:
    st.session_state.historial = []

with st.sidebar:
    st.subheader("Estado de la API")
    st.caption(f"URL configurada: `{API_URL}`")

    health = check_health()
    if health["ok"]:
        st.success("Modelo disponible ✅")
    else:
        st.error("Modelo no disponible ⚠️")
        if health["detail"]:
            st.caption(health["detail"])

    if st.button("🔄 Reintentar conexion", width="stretch"):
        st.rerun()

    st.divider()
    st.subheader("Rendimiento del modelo")
    metrics = get_metrics()
    if metrics:
        accuracy = metrics["test_metrics"]["accuracy"]
        f1_macro = metrics["test_metrics"]["classification_report"]["macro avg"]["f1-score"]
        st.metric("Accuracy (test)", f"{accuracy:.1%}")
        st.metric("F1 macro (test)", f"{f1_macro:.1%}")
        st.caption(
            "Estos numeros son altos porque el dataset de entrenamiento es "
            "sintetico (ver README.md, seccion 'Dataset', para el detalle)."
        )
    else:
        st.caption("Metricas no disponibles todavia (¿se corrio `src/train.py`?).")

st.subheader("Nuevo reclamo")
texto = st.text_area(
    "Redacta el reclamo tal como lo escribiria un vecino",
    placeholder="Ej: hay un hoyo enorme en mi calle y ya se pincho una rueda...",
    height=120,
)

clasificar = st.button("Clasificar reclamo", type="primary")

if clasificar:
    if not texto or not texto.strip():
        st.warning("Escribe el texto del reclamo antes de clasificar.")
    else:
        with st.spinner("Clasificando reclamo..."):
            resultado, error = classify_complaint(texto.strip())

        if error:
            st.error(error)
        else:
            categoria = resultado["categoria"]
            confianza = resultado["confianza"]
            probabilidades = resultado["probabilidades"]

            st.success(f"**Categoría asignada:** {categoria}")
            st.metric("Confianza del modelo", f"{confianza:.1%}")

            st.caption("Probabilidad estimada por categoría:")
            df_proba = (
                pd.Series(probabilidades, name="Probabilidad (%)")
                .mul(100)
                .sort_values(ascending=False)
                .to_frame()
            )
            st.bar_chart(df_proba)

            st.session_state.historial.insert(
                0,
                {
                    "hora": datetime.now().strftime("%H:%M:%S"),
                    "texto": texto.strip(),
                    "categoria": categoria,
                    "confianza": f"{confianza:.1%}",
                },
            )
            st.session_state.historial = st.session_state.historial[:10]

if st.session_state.historial:
    st.divider()
    st.subheader("Historial de esta sesión")
    st.dataframe(
        pd.DataFrame(st.session_state.historial),
        width="stretch",
        hide_index=True,
    )
