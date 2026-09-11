"""
Utilidades de normalizacion de texto compartidas entre `train.py` y `api.py`.

Se aislan en su propio modulo (en vez de definirlas dentro de `train.py`) por
una razon tecnica concreta: el pipeline de scikit-learn serializado en
`models/model.pkl` incluye un `FunctionTransformer` que referencia
`normalize_corpus` por su modulo de origen. Si esa funcion viviera dentro de
`train.py` y `train.py` se ejecutara como script (`python src/train.py`), se
guardaria con `__module__ == "__main__"`, y al cargar el modelo desde
`api.py` (un `__main__` distinto) la carga fallaria con
`AttributeError: module '__main__' has no attribute 'normalize_corpus'`.
Al vivir en un modulo con nombre propio, la referencia se resuelve igual sin
importar quien entrene o quien cargue el modelo.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normaliza un reclamo: minusculas, sin tildes y sin signos de puntuacion.

    Se aplica tanto en el entrenamiento como en la inferencia, para evitar
    que el preprocesamiento se desincronice entre ambos.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_corpus(texts) -> list[str]:
    """Aplica `normalize_text` a una coleccion de textos.

    Se define como funcion de modulo (no lambda) para que sea "picklable"
    dentro del Pipeline de scikit-learn y pueda cargarse despues en la API.
    """
    return [normalize_text(str(text)) for text in texts]
