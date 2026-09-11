# Proyecto: Clasificador de Reclamos Municipales con Machine Learning

## Contexto del proyecto

Este es un proyecto académico para la actividad "Diseño, Implementación y Evaluación de una
Aplicación Web con Machine Learning" de IPCHILE. Debe ser evaluado como **DESTACADO** según
la rúbrica oficial (ver sección "Rúbrica de evaluación" más abajo).

## Problema empresarial

En la Municipalidad, los reclamos ciudadanos se clasifican manualmente por categoría antes de
ser derivados al departamento correspondiente. Este proceso es lento e inconsistente. El
objetivo es automatizar esa clasificación mediante un modelo de Machine Learning, reduciendo
tiempos de derivación y mejorando la consistencia del proceso.

## Categorías de clasificación

El modelo clasifica cada reclamo en una de estas 6 categorías:

1. Aseo y Ornato
2. Vialidad y Pavimentación
3. Alumbrado Público
4. Áreas Verdes
5. Seguridad Ciudadana
6. Permisos y Patentes Comerciales

## Dataset

- Dataset sintético (no se usan datos reales de la Municipalidad, por Ley 21.719 de Protección
  de Datos Personales).
- ~700 registros, texto en español, redactado como lo escribiría un vecino real (informal,
  con errores típicos, sin estructura formal).
- Columnas: `texto`, `categoria`.
- Ubicación: `data/reclamos.csv`.

## Stack tecnológico (decisiones justificadas, no cambiar sin discutirlo)

- **Lenguaje**: Python 3.11+
- **ML**: Scikit-learn (TF-IDF + Naive Bayes o Logistic Regression). NO usar TensorFlow/Keras:
  el volumen de datos y la naturaleza del problema (texto corto, categorías discretas) no
  justifica deep learning; scikit-learn generaliza mejor con este tamaño de dataset y tiene
  menor latencia de inferencia.
- **API**: FastAPI (no Flask) — se aprovecha la documentación automática vía Swagger/OpenAPI
  en `/docs`, requisito directo del criterio 4.1 de la rúbrica.
- **Frontend**: Streamlit — interfaz interactiva sin necesidad de HTML/CSS/JS manual.
- **Contenerización**: Docker + docker-compose (dos servicios: api y dashboard).
- **Despliegue**: API en Render (free tier), dashboard en Streamlit Community Cloud.

## Estructura de carpetas objetivo

```
reclamos-municipales-ml/
├── CLAUDE.md
├── README.md
├── .gitignore
├── .dockerignore
├── data/
│   └── reclamos.csv
├── notebooks/
│   └── exploracion.ipynb
├── src/
│   ├── text_utils.py
│   ├── train.py
│   ├── api.py
│   └── dashboard.py
├── models/
│   ├── model.pkl
│   ├── metrics.json
│   └── confusion_matrix.png
├── tests/
│   ├── test_api.py
│   └── load_test.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Rúbrica de evaluación (cumplir SIEMPRE en nivel Destacado, 25 pts c/u)

| Criterio | Texto "Destacado" | Cómo se cumple en este proyecto |
|---|---|---|
| 3.1 Diseño de solución ML | Diseño completo con justificación clara y documentación detallada | EDA documentado + justificación del algoritmo en README + diagrama de flujo |
| 3.2 App interactiva (Flask/Django/Streamlit) | App optimizada, intuitiva y escalable | Streamlit con `st.cache_resource`, manejo de errores, Docker |
| 4.1 Integración vía API | API bien documentada y funcional | FastAPI con Swagger automático, modelos Pydantic, endpoints `/predict`, `/health`, `/metrics` |
| 4.3 App funcional y optimizada en contexto real | Estable, eficiente y bien documentada | `train_test_split`, métricas completas, tests con pytest, prueba de carga documentada |

## Métricas de evaluación exigidas

- Precision, Recall, F1-Score por categoría (classification_report completo).
- Matriz de confusión (guardada como imagen en `models/`).
- **RMSE y MAE NO aplican**: son métricas de regresión y este es un problema de clasificación
  multiclase. Esto debe quedar explícito y justificado en el informe, no omitido en silencio.
- Optimización de hiperparámetros con `GridSearchCV` (alpha en Naive Bayes o C en Logistic
  Regression).

## Pruebas requeridas

- Tests funcionales de la API con `pytest` (casos normales y de error).
- Prueba de carga simple: tiempo de respuesta promedio bajo N requests concurrentes,
  documentado en el README.
- Checklist de usabilidad: feedback visual de estado de carga, mensajes de error claros al
  usuario (no tracebacks crudos).

## Convenciones de trabajo

- Comentarios y docstrings en español.
- Nombres de variables y funciones en inglés (convención estándar de Python).
- Cada script debe correr de forma independiente y documentar sus dependencias en
  `requirements.txt`.
- No hardcodear URLs (usar variables de entorno para la URL de la API en el dashboard).
- Antes de cualquier cambio grande (ej. reestructurar `train.py` o migrar el stack de la API),
  usar Plan Mode y esperar aprobación explícita antes de ejecutar.

## Qué evitar

- No usar TensorFlow/Keras salvo que se justifique explícitamente un cambio de enfoque.
- No dejar RMSE/MAE sin mencionar ni sin justificar su ausencia.
- No dejar la URL de la API como `localhost` en el código que se sube a producción.
- No omitir el manejo de errores en la API ni en el dashboard.
