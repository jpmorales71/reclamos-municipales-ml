"""
Prueba de carga simple para `POST /predict`.

Mide el tiempo de respuesta bajo distintos niveles de concurrencia, tal como
exige la rúbrica (criterio 4.3: "prueba de carga simple, tiempo de respuesta
promedio bajo N requests concurrentes, documentado en el README"). No es
parte de la suite de `pytest` (no valida correctitud funcional -- de eso se
encarga `tests/test_api.py`): es un script de benchmarking que se corre
manualmente contra una instancia de la API ya levantada.

Uso:
    # en una terminal:
    python src/api.py

    # en otra terminal:
    python tests/load_test.py
    python tests/load_test.py --api-url http://localhost:8000 --n-requests 200 --concurrency-levels 1 10 50
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

# Variedad de textos de ejemplo para no medir el efecto de cache de un unico
# texto repetido (TfidfVectorizer no cachea, pero es buena practica variar
# la entrada igual que en un uso real).
SAMPLE_TEXTS = [
    "hay un hoyo enorme en mi calle y ya se pincho una rueda",
    "el poste de la esquina no tiene luz hace una semana",
    "el pasto de la plaza esta muy largo y hay basura",
    "estan vendiendo alcohol sin patente en la esquina",
    "hay robos todas las noches en el sitio eriazo",
    "el camion de la basura no ha pasado en dos semanas",
]


def send_request(session: requests.Session, api_url: str, texto: str) -> tuple[float, int]:
    """Envia un POST /predict y devuelve (tiempo_en_segundos, status_code).

    Nunca lanza: si la request falla (timeout, conexion rechazada, etc.) se
    devuelve status_code=-1 para que se contabilice como error.

    Recibe una `requests.Session` ya creada (en vez de usar `requests.post`,
    que crea una sesion nueva en cada llamada) porque en Windows cada
    `Session` nueva dispara una deteccion de proxy del sistema via WinHTTP
    que puede tardar del orden de 1-2 segundos -- con `requests.post` esto se
    pagaba en *cada una* de las N requests, inflando la latencia medida por
    ~100x y ocultando por completo el tiempo real de inferencia del modelo.
    Reutilizar una Session (cuyo pool de conexiones de urllib3 es seguro para
    uso concurrente entre threads) evita ese costo y además reutiliza
    conexiones TCP, que es lo que se quiere medir en una prueba de carga.
    """
    start = time.perf_counter()
    try:
        response = session.post(f"{api_url}/predict", json={"texto": texto}, timeout=30)
        return time.perf_counter() - start, response.status_code
    except requests.exceptions.RequestException:
        return time.perf_counter() - start, -1


def run_load_test(session: requests.Session, api_url: str, n_requests: int, concurrency: int) -> dict:
    """Dispara `n_requests` contra /predict con el nivel de concurrencia dado."""
    textos = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(n_requests)]
    tiempos_ms: list[float] = []
    errores = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Warm-up: cada worker thread nuevo que crea ThreadPoolExecutor paga,
        # en la primera conexion HTTP que hace, un costo fijo de entre 1 y 2
        # segundos ajeno a la API (en Windows, la resolucion de proxy del
        # sistema via WinHTTP). Sin este warm-up, ese costo se mide como si
        # fuera latencia de /predict y contamina p95/max con "concurrency"
        # outliers de ~2s que no tienen nada que ver con el modelo. Se
        # ejercita cada thread una vez, fuera del cronometro, antes de medir.
        list(executor.map(lambda _: send_request(session, api_url, textos[0]), range(concurrency)))

        inicio_total = time.perf_counter()
        futures = [executor.submit(send_request, session, api_url, texto) for texto in textos]
        for future in as_completed(futures):
            elapsed, status_code = future.result()
            tiempos_ms.append(elapsed * 1000)
            if status_code != 200:
                errores += 1
    duracion_total = time.perf_counter() - inicio_total

    tiempos_ms.sort()
    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "duracion_total_s": duracion_total,
        "throughput_req_s": n_requests / duracion_total if duracion_total > 0 else 0.0,
        "errores": errores,
        "latencia_promedio_ms": statistics.mean(tiempos_ms),
        "latencia_mediana_ms": statistics.median(tiempos_ms),
        "latencia_p95_ms": tiempos_ms[max(int(len(tiempos_ms) * 0.95) - 1, 0)],
        "latencia_min_ms": min(tiempos_ms),
        "latencia_max_ms": max(tiempos_ms),
    }


def print_resultado(resultado: dict) -> None:
    print(
        f"Concurrencia={resultado['concurrency']:>3} | "
        f"N={resultado['n_requests']:>4} | "
        f"Errores={resultado['errores']:>3} | "
        f"Throughput={resultado['throughput_req_s']:6.1f} req/s | "
        f"Latencia prom={resultado['latencia_promedio_ms']:7.1f} ms | "
        f"mediana={resultado['latencia_mediana_ms']:7.1f} ms | "
        f"p95={resultado['latencia_p95_ms']:7.1f} ms | "
        f"min={resultado['latencia_min_ms']:6.1f} ms | "
        f"max={resultado['latencia_max_ms']:7.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de carga simple para POST /predict.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL base de la API.")
    parser.add_argument("--n-requests", type=int, default=100, help="Cantidad de requests por nivel de concurrencia.")
    parser.add_argument(
        "--concurrency-levels", type=int, nargs="+", default=[1, 10, 50],
        help="Niveles de concurrencia a probar (uno tras otro).",
    )
    args = parser.parse_args()

    # Una sola Session para todo el script (health check + las N requests de
    # cada nivel de concurrencia): ver la nota en send_request() sobre por
    # que "requests.post"/"requests.get" sueltos arruinan la medicion.
    session = requests.Session()

    try:
        health = session.get(f"{args.api_url}/health", timeout=5)
        health.raise_for_status()
        if not health.json().get("model_loaded"):
            print("Advertencia: el modelo no esta cargado en la API; los resultados podrian no ser representativos.")
    except requests.exceptions.RequestException as exc:
        print(f"No se pudo conectar con la API en '{args.api_url}': {exc}")
        print("Levanta la API primero con 'python src/api.py'.")
        return

    print(f"Prueba de carga contra {args.api_url}/predict")
    print("=" * 110)
    for concurrency in args.concurrency_levels:
        resultado = run_load_test(session, args.api_url, args.n_requests, concurrency)
        print_resultado(resultado)
    print("=" * 110)


if __name__ == "__main__":
    main()
