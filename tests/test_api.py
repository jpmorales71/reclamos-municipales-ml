"""
Tests funcionales de la API (`src/api.py`) con pytest.

Cubren casos normales (prediccion valida, health, metrics) y casos de error
(texto vacio, texto muy corto, campo faltante, modelo no disponible), para
verificar que la API responde siempre con un mensaje claro en vez de un
traceback crudo (requisito de la rubrica, seccion "Pruebas requeridas").

Requiere que exista `models/model.pkl` y `models/metrics.json` (generados
por `python src/train.py`) para los casos que dependen del modelo cargado.

Uso:
    pytest tests/test_api.py -v
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Agregamos 'src/' al path para poder importar la app sin instalar el
# proyecto como paquete, siguiendo la misma convencion que usan
# train.py/api.py para resolver 'text_utils'.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import api as api_module  # noqa: E402  (import despues de ajustar sys.path)

CATEGORIAS_VALIDAS = {
    "Aseo y Ornato",
    "Vialidad y Pavimentación",
    "Alumbrado Público",
    "Áreas Verdes",
    "Seguridad Ciudadana",
    "Permisos y Patentes Comerciales",
}


@pytest.fixture()
def client():
    """Cliente de test que dispara el evento de arranque (carga del modelo)."""
    with TestClient(api_module.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Casos normales
# ---------------------------------------------------------------------------
def test_root_devuelve_mensaje_de_bienvenida(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "documentacion" in body


def test_health_reporta_modelo_cargado(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metrics_devuelve_reporte_de_evaluacion(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "test_metrics" in body
    assert "classification_report" in body["test_metrics"]
    # RMSE/MAE deben quedar documentados como no aplicables, no omitidos.
    assert "metricas_no_aplicables" in body
    assert "RMSE" in body["metricas_no_aplicables"]
    assert "MAE" in body["metricas_no_aplicables"]


def test_predict_clasifica_un_reclamo_valido(client):
    payload = {"texto": "hay un hoyo enorme en mi calle y ya se pincho una rueda"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["categoria"] in CATEGORIAS_VALIDAS
    assert 0.0 <= body["confianza"] <= 1.0
    assert set(body["probabilidades"].keys()) == CATEGORIAS_VALIDAS
    # las probabilidades de todas las categorias deben sumar ~1
    assert sum(body["probabilidades"].values()) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "texto,categoria_esperada",
    [
        ("el poste de la esquina no tiene luz hace una semana", "Alumbrado Público"),
        ("el pasto de la plaza esta muy largo y hay basura", "Aseo y Ornato"),
    ],
)
def test_predict_reconoce_vocabulario_caracteristico(client, texto, categoria_esperada):
    """Verifica sobre texto que no pertenece al dataset de entrenamiento."""
    response = client.post("/predict", json={"texto": texto})
    assert response.status_code == 200
    assert response.json()["categoria"] == categoria_esperada


# ---------------------------------------------------------------------------
# Casos de error: deben responder con codigo HTTP correcto y un mensaje
# entendible, nunca con un traceback crudo.
# ---------------------------------------------------------------------------
def test_predict_rechaza_texto_vacio(client):
    response = client.post("/predict", json={"texto": "     "})
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert "errores" in body


def test_predict_rechaza_texto_demasiado_corto(client):
    response = client.post("/predict", json={"texto": "hi"})
    assert response.status_code == 422


def test_predict_rechaza_campo_faltante(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_rechaza_tipo_de_dato_invalido(client):
    response = client.post("/predict", json={"texto": 12345})
    assert response.status_code == 422


def test_predict_responde_503_si_el_modelo_no_esta_cargado(client, monkeypatch):
    """Simula que el modelo no pudo cargarse (ej. aun no se corrio train.py)."""
    monkeypatch.setitem(api_module.model_state, "model", None)
    monkeypatch.setitem(api_module.model_state, "error", "archivo de prueba no encontrado")

    response = client.post("/predict", json={"texto": "un reclamo cualquiera de prueba"})

    assert response.status_code == 503
    assert "train.py" in response.json()["detail"]


def test_health_reporta_degradado_si_el_modelo_no_esta_cargado(client, monkeypatch):
    monkeypatch.setitem(api_module.model_state, "model", None)
    monkeypatch.setitem(api_module.model_state, "error", "archivo de prueba no encontrado")

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degradado"
    assert body["model_loaded"] is False
    assert body["detail"] == "archivo de prueba no encontrado"
