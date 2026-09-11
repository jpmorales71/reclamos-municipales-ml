"""
API REST del clasificador de reclamos municipales.

Expone el pipeline entrenado (`models/model.pkl`, generado por `src/train.py`)
a traves de tres endpoints:

- `POST /predict` : clasifica un reclamo en una de las 6 categorias.
- `GET  /health`  : estado del servicio y si el modelo esta cargado.
- `GET  /metrics` : metricas de evaluacion de la ultima ejecucion de entrenamiento.

La documentacion interactiva (Swagger) queda disponible automaticamente en
`/docs`, y el esquema OpenAPI en `/openapi.json` (criterio 4.1 de la
rubrica). Los errores (modelo no disponible, texto invalido, fallas
inesperadas) siempre responden con un mensaje claro en JSON, nunca con un
traceback crudo.

Uso:
    python src/api.py
    # o, desde la raiz del proyecto:
    uvicorn src.api:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Aseguramos que 'src/' este en sys.path para que 'import text_utils' resuelva
# como modulo de nivel superior, igual que en train.py: el pipeline
# serializado en model.pkl referencia ese modulo por nombre al deserializarse
# (ver docstring de text_utils.py para el detalle de por que importa el
# nombre exacto del modulo).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import normalize_corpus  # noqa: F401,E402  (requerido para el unpickling del pipeline)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reclamos-api")

# ---------------------------------------------------------------------------
# Configuracion (rutas resueltas relativas a este archivo, sobreescribibles
# por variable de entorno para Docker/despliegue)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model.pkl")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(BASE_DIR / "models" / "metrics.json")))

CATEGORIAS_ESPERADAS = [
    "Aseo y Ornato",
    "Vialidad y Pavimentación",
    "Alumbrado Público",
    "Áreas Verdes",
    "Seguridad Ciudadana",
    "Permisos y Patentes Comerciales",
]

# Estado del modelo cargado, compartido entre requests. Se llena en el
# lifespan de arranque (ver mas abajo) y se consulta en /health y /predict.
model_state: dict[str, Any] = {"model": None, "error": None}


def load_model() -> None:
    """Carga el pipeline entrenado desde disco.

    Si falla (el archivo no existe porque aun no se corrio `train.py`, esta
    corrupto, etc.) no se detiene el arranque de la API: se deja constancia
    del error para que `/health` y `/predict` respondan con un mensaje claro
    en vez de que el proceso completo no levante.
    """
    try:
        model = joblib.load(MODEL_PATH)
        categorias_modelo = set(getattr(model, "classes_", []))
        if categorias_modelo and categorias_modelo != set(CATEGORIAS_ESPERADAS):
            logger.warning(
                "Las categorias del modelo cargado no coinciden con las "
                "esperadas. Modelo: %s | Esperadas: %s",
                sorted(categorias_modelo), sorted(CATEGORIAS_ESPERADAS),
            )
        model_state["model"] = model
        model_state["error"] = None
        logger.info("Modelo cargado correctamente desde '%s'.", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - se registra y se expone via /health
        model_state["model"] = None
        model_state["error"] = str(exc)
        logger.error("No se pudo cargar el modelo desde '%s': %s", MODEL_PATH, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al iniciar la API (no en cada request)."""
    load_model()
    yield


app = FastAPI(
    title="API - Clasificador de Reclamos Municipales",
    description=(
        "Clasifica reclamos ciudadanos en una de 6 categorias municipales "
        "usando un pipeline TF-IDF + Naive Bayes / Logistic Regression "
        "entrenado con `src/train.py`. Proyecto academico IPCHILE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# En este proyecto academico el dashboard puede correr en otro origen
# (otro puerto/dominio), por eso se habilita CORS sin restriccion. En un
# despliegue real conviene acotar `allow_origins` al dominio del dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Esquemas Pydantic: documentan automaticamente el request/response en /docs
# y validan la entrada antes de que llegue a la logica de negocio.
# ---------------------------------------------------------------------------
class ReclamoRequest(BaseModel):
    """Cuerpo esperado por `POST /predict`."""

    texto: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Texto del reclamo ciudadano, redactado libremente.",
        examples=["hay un hoyo enorme en mi calle y ya se pincho una rueda"],
    )

    @field_validator("texto")
    @classmethod
    def texto_no_vacio(cls, value: str) -> str:
        """Rechaza textos que son solo espacios en blanco."""
        if not value.strip():
            raise ValueError("El texto del reclamo no puede estar vacio.")
        return value


class PrediccionResponse(BaseModel):
    """Respuesta de `POST /predict`."""

    categoria: str = Field(..., description="Categoria municipal predicha.")
    confianza: float = Field(
        ..., ge=0, le=1, description="Probabilidad estimada de la categoria predicha."
    )
    probabilidades: dict[str, float] = Field(
        ..., description="Probabilidad estimada para cada una de las 6 categorias."
    )


class HealthResponse(BaseModel):
    """Respuesta de `GET /health`."""

    status: str = Field(..., description="'ok' si el modelo esta cargado, 'degradado' si no.")
    model_loaded: bool
    model_path: str
    detail: str | None = Field(None, description="Motivo del error, solo si status es 'degradado'.")


# ---------------------------------------------------------------------------
# Manejo de errores: siempre JSON con un mensaje entendible, nunca un
# traceback crudo expuesto al usuario final.
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Traduce los errores de validacion de Pydantic a un mensaje entendible."""
    errores = [err["msg"] for err in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "Los datos enviados no son validos.", "errores": errores},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ultima red de seguridad: registra el error completo en el log del
    servidor y responde algo generico y seguro al cliente."""
    logger.exception("Error no controlado en %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Ocurrio un error interno al procesar la solicitud. Intenta nuevamente."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Monitoreo"])
async def root() -> dict[str, str]:
    """Mensaje de bienvenida con referencia a la documentacion interactiva."""
    return {
        "mensaje": "API del clasificador de reclamos municipales.",
        "documentacion": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoreo"])
async def health() -> HealthResponse:
    """Reporta si la API esta activa y si el modelo se cargo correctamente.

    Util para probes de Docker/Render y para que el dashboard avise al
    usuario si el servicio no esta listo, en vez de fallar en silencio.
    """
    is_ok = model_state["model"] is not None
    return HealthResponse(
        status="ok" if is_ok else "degradado",
        model_loaded=is_ok,
        model_path=str(MODEL_PATH),
        detail=None if is_ok else model_state["error"],
    )


@app.get("/metrics", tags=["Monitoreo"])
async def metrics() -> dict[str, Any]:
    """Devuelve las metricas de evaluacion de la ultima ejecucion de `src/train.py`.

    Incluye precision/recall/f1 por categoria, los hiperparametros elegidos
    por GridSearchCV y la nota explicita de por que RMSE/MAE no aplican
    (ver `models/metrics.json`).
    """
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Aun no existen metricas guardadas. Ejecuta 'python src/train.py' "
                "para entrenar el modelo y generar 'models/metrics.json'."
            ),
        )
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_inference(model, texto: str) -> tuple[str, float, dict[str, float]]:
    """Corre la inferencia sincrona (normalizacion + TF-IDF + clasificador).

    Se calcula `predict_proba` una sola vez y la categoria predicha se
    deriva como el argmax de esas probabilidades, en vez de llamar tambien
    a `model.predict()` (que repetiria desde cero la normalizacion y la
    vectorizacion TF-IDF del mismo texto). Si el modelo no soporta
    probabilidades, se recurre a `predict()` como respaldo.
    """
    textos = [texto]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(textos)[0]
        idx_max = int(proba.argmax())
        categoria_predicha = str(model.classes_[idx_max])
        probabilidades = {
            str(categoria): round(float(prob), 4) for categoria, prob in zip(model.classes_, proba)
        }
        confianza = probabilidades[categoria_predicha]
    else:
        categoria_predicha = str(model.predict(textos)[0])
        probabilidades = {}
        confianza = 1.0
    return categoria_predicha, confianza, probabilidades


@app.post("/predict", response_model=PrediccionResponse, tags=["Clasificacion"])
async def predict(reclamo: ReclamoRequest) -> PrediccionResponse:
    """Clasifica un reclamo ciudadano en una de las 6 categorias municipales.

    La inferencia es CPU-bound (TF-IDF + clasificador de scikit-learn, todo
    codigo sincrono): se ejecuta en un thread pool (`run_in_threadpool`) en
    vez de directamente en la corrutina, para no bloquear el event loop de
    asyncio mientras se resuelve. Esto reduce la latencia bajo carga
    concurrente (ver la seccion "Prueba de carga" en README.md, donde se
    midio el impacto real de este cambio).
    """
    model = model_state["model"]
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo no esta disponible todavia. Ejecuta "
                "'python src/train.py' para generar 'models/model.pkl' y "
                "vuelve a intentar."
            ),
        )

    categoria_predicha, confianza, probabilidades = await run_in_threadpool(
        run_inference, model, reclamo.texto
    )

    return PrediccionResponse(
        categoria=categoria_predicha,
        confianza=confianza,
        probabilidades=probabilidades,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
