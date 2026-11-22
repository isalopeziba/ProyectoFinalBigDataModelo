# Servicio de IA – Detección de transacciones sospechosas

API en FastAPI que recibe transacciones del backend y devuelve un **puntaje de riesgo** entre 0 y 1.
La API no decide si una transacción es sospechosa: eso lo hace el backend comparando el puntaje con el umbral.


## Endpoints

| Método | Ruta | Para qué |
|---|---|---|
| GET | `/health` | Saber si el servicio está vivo y el modelo cargado |
| GET | `/model-info` | Versión, umbral sugerido, categorías válidas y métricas |
| POST | `/predict` | Puntaje de una transacción |
| POST | `/predict/batch` | Puntajes de hasta 10.000 transacciones (importación de CSV) |




