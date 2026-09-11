# Imagen base: Python 3.11 (version minima definida en CLAUDE.md), variante
# "slim" para mantener la imagen liviana. scikit-learn/pandas/matplotlib
# traen wheels precompilados para esta variante, no hace falta compilar
# nada desde el codigo fuente.
FROM python:3.11-slim

WORKDIR /app

# Evita archivos .pyc y hace que los logs de Python/Streamlit salgan sin
# buffer (aparecen de inmediato en `docker compose logs`). Las dos variables
# de Streamlit evitan el prompt interactivo de telemetria y el intento de
# abrir un navegador dentro del contenedor.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Las dependencias se instalan antes de copiar el codigo fuente para
# aprovechar la cache de capas de Docker: mientras no cambie
# requirements.txt, un cambio en src/ no obliga a reinstalar todo de nuevo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo fuente y dataset (unico input real que necesita el entrenamiento).
# `models/`, `notebooks/` y `tests/` quedan fuera de la imagen a proposito
# (ver .dockerignore): la imagen genera su propio `models/` en el build de
# abajo, en vez de depender de artefactos locales que podrian no existir en
# un clon nuevo del repositorio (el .pkl no se versiona en git).
COPY data/ ./data/
COPY src/ ./src/

# Entrena el modelo durante el build de la imagen, para que quede lista para
# servir sin pasos manuales adicionales al levantarla (genera
# models/model.pkl, models/metrics.json y models/confusion_matrix.png).
RUN python src/train.py

# La API escucha en 8000; el dashboard (que usa esta misma imagen con otro
# `command:` desde docker-compose.yml) escucha en 8501.
EXPOSE 8000 8501

# Comando por defecto: levantar la API. docker-compose.yml sobreescribe este
# comando para el servicio "dashboard".
CMD ["python", "src/api.py"]
