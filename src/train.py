"""
Entrenamiento del clasificador de reclamos municipales.

Este script:
    1. Carga el dataset sintetico (`data/reclamos.csv`).
    2. Normaliza el texto (minusculas, sin tildes, sin puntuacion) con la
       misma logica documentada y validada en `notebooks/exploracion.ipynb`.
    3. Separa los datos en train/test de forma estratificada.
    4. Busca el mejor modelo con GridSearchCV, comparando dos algoritmos
       (Multinomial Naive Bayes y Logistic Regression) sobre una
       vectorizacion TF-IDF.
    5. Evalua el modelo ganador con classification_report (precision,
       recall, f1-score por categoria) y matriz de confusion.
    6. Persiste el pipeline completo (`models/model.pkl`) y las metricas
       (`models/metrics.json`), junto a la imagen de la matriz de confusion.

Se usa scikit-learn (TF-IDF + Naive Bayes / Logistic Regression) y no
TensorFlow/Keras: con ~700 registros de texto corto y 6 categorias
discretas, un modelo de deep learning no tiene suficientes datos para
superar a un modelo lineal clasico, y ademas scikit-learn ofrece menor
latencia de inferencia para la API (ver README.md, seccion "Stack
tecnologico", para la justificacion completa).

Uso:
    python src/train.py
    python src/train.py --test-size 0.25 --cv-folds 10
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Aseguramos que 'src/' este en sys.path para que `import text_utils` funcione
# sin importar desde que directorio se invoque este script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import normalize_corpus  # noqa: E402  (import tras ajustar sys.path)

# ---------------------------------------------------------------------------
# Rutas del proyecto (relativas a este archivo, para que el script corra
# de forma independiente sin importar desde donde se invoque)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "reclamos.csv"
DEFAULT_MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42

# Nota: `normalize_text`/`normalize_corpus` viven en `text_utils.py` (no aca)
# a proposito: el Pipeline serializado en `models/model.pkl` guarda una
# referencia a `normalize_corpus` por su modulo de origen, y si viviera
# dentro de `train.py` ejecutado como script quedaria atada a `__main__`,
# rompiendo la carga del modelo desde `api.py`. Ver docstring de
# `text_utils.py` para el detalle.


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_dataset(data_path: Path) -> tuple[pd.Series, pd.Series]:
    """Carga el CSV de reclamos y devuelve (texto, categoria).

    Lanza un error explicito y legible si el archivo no existe o si faltan
    las columnas esperadas, en vez de dejar que falle mas adelante con un
    traceback dificil de interpretar.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en '{data_path}'. "
            "Verifica que exista data/reclamos.csv antes de entrenar."
        )

    df = pd.read_csv(data_path)
    columnas_esperadas = {"texto", "categoria"}
    if not columnas_esperadas.issubset(df.columns):
        raise ValueError(
            f"El dataset debe tener las columnas {columnas_esperadas}, "
            f"pero tiene {set(df.columns)}."
        )

    df = df.dropna(subset=["texto", "categoria"]).drop_duplicates(subset=["texto"])
    return df["texto"], df["categoria"]


# ---------------------------------------------------------------------------
# Construccion del pipeline y de la grilla de hiperparametros
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Crea el pipeline completo: normalizacion -> TF-IDF -> clasificador.

    Al incluir la normalizacion dentro del pipeline, el objeto serializado
    en `models/model.pkl` puede recibir texto crudo directamente (tal como
    llega desde la API), sin que el codigo de inferencia deba reimplementar
    la limpieza de texto.
    """
    return Pipeline(
        steps=[
            ("normalize", FunctionTransformer(normalize_corpus)),
            ("tfidf", TfidfVectorizer()),
            # El clasificador se sobreescribe en cada combinacion probada
            # por GridSearchCV (ver build_param_grid).
            ("clf", MultinomialNB()),
        ]
    )


def build_param_grid() -> list[dict]:
    """Define la grilla de busqueda de hiperparametros.

    Se comparan dos algoritmos candidatos (ambos justificados en el stack
    tecnologico del proyecto): Multinomial Naive Bayes, optimizando `alpha`
    (suavizado de Laplace), y Logistic Regression, optimizando `C`
    (inverso de la regularizacion). Ademas se explora el rango de n-gramas
    y la frecuencia minima de documento del vectorizador TF-IDF.
    """
    common_tfidf_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
    }
    return [
        {
            **common_tfidf_grid,
            "clf": [MultinomialNB()],
            "clf__alpha": [0.1, 0.5, 1.0, 2.0],
        },
        {
            **common_tfidf_grid,
            "clf": [LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)],
            "clf__C": [0.1, 1.0, 10.0],
        },
    ]


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_test, y_pred, labels: list[str], output_path: Path) -> None:
    """Genera y guarda la matriz de confusion como imagen PNG."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=True,
        ax=ax,
    )
    ax.set_title("Matriz de confusion - Clasificador de reclamos municipales")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_metrics_report(
    grid_search: GridSearchCV,
    dataset_info: dict,
    y_test,
    y_pred,
    labels: list[str],
) -> dict:
    """Arma el diccionario de metricas que se guarda en `models/metrics.json`."""
    best_params_serializable = {
        key: (repr(value) if not isinstance(value, (int, float, str, tuple, list)) else value)
        for key, value in grid_search.best_params_.items()
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "model_selection": {
            "search_strategy": (
                "GridSearchCV con validacion cruzada estratificada, "
                "optimizando f1_macro. Se comparan MultinomialNB (alpha) "
                "y LogisticRegression (C) sobre TF-IDF."
            ),
            "best_estimator": grid_search.best_params_["clf"].__class__.__name__,
            "best_params": best_params_serializable,
            "best_cv_f1_macro": grid_search.best_score_,
        },
        "test_metrics": {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
        },
        "metricas_no_aplicables": {
            "RMSE": (
                "No aplica: el Root Mean Squared Error mide el error entre "
                "valores numericos continuos (problemas de regresion). "
                "Este proyecto es de clasificacion multiclase sobre "
                "categorias discretas sin orden natural, por lo que RMSE "
                "no tiene una interpretacion valida aqui."
            ),
            "MAE": (
                "No aplica por la misma razon que RMSE: el Mean Absolute "
                "Error tambien esta definido para variables numericas "
                "continuas. Las metricas correctas para este problema son "
                "precision, recall, F1-score (por categoria y macro/weighted) "
                "y la matriz de confusion, incluidas en 'test_metrics'."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Orquestacion principal
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el clasificador de reclamos municipales.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Ruta al CSV de reclamos.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Carpeta de salida para el modelo y las metricas.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion del dataset reservada para test.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cantidad de folds para GridSearchCV.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Semilla para reproducibilidad.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENTRENAMIENTO - Clasificador de Reclamos Municipales")
    print("=" * 70)

    print(f"\n[1/5] Cargando dataset desde '{args.data_path}'...")
    X, y = load_dataset(args.data_path)
    labels = sorted(y.unique())
    print(f"      {len(X)} registros validos, {len(labels)} categorias: {labels}")

    print(f"\n[2/5] Separando train/test (test_size={args.test_size}, estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    print(f"      Train: {len(X_train)} registros | Test: {len(X_test)} registros")

    print(f"\n[3/5] Buscando el mejor modelo con GridSearchCV ({args.cv_folds}-fold CV)...")
    pipeline = build_pipeline()
    param_grid = build_param_grid()
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=args.cv_folds,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    print(f"      Mejor modelo: {grid_search.best_params_['clf'].__class__.__name__}")
    print(f"      Mejores hiperparametros: { {k: v for k, v in grid_search.best_params_.items() if k != 'clf'} }")
    print(f"      f1_macro (CV): {grid_search.best_score_:.4f}")

    print("\n[4/5] Evaluando en el conjunto de test...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"      Accuracy en test: {test_accuracy:.4f}")
    print("\n      Reporte de clasificacion (precision / recall / f1 por categoria):")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    confusion_matrix_path = args.models_dir / "confusion_matrix.png"
    save_confusion_matrix(y_test, y_pred, labels, confusion_matrix_path)
    print(f"      Matriz de confusion guardada en '{confusion_matrix_path}'")

    print("\n      Nota metodologica: RMSE y MAE no se calculan porque son")
    print("      metricas de regresion y este es un problema de clasificacion")
    print("      multiclase (ver 'metricas_no_aplicables' en metrics.json).")

    print("\n[5/5] Guardando modelo y metricas...")
    model_path = args.models_dir / "model.pkl"
    joblib.dump(best_model, model_path)
    print(f"      Modelo guardado en '{model_path}'")

    dataset_info = {
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "categories": labels,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    metrics = build_metrics_report(grid_search, dataset_info, y_test, y_pred, labels)
    metrics_path = args.models_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"      Metricas guardadas en '{metrics_path}'")

    print("\n" + "=" * 70)
    print("Entrenamiento finalizado con exito.")
    print("=" * 70)


if __name__ == "__main__":
    main()
