# Servicio de IA – Detección de transacciones sospechosas (modelo v1.1.0)

Recibe una transacción del backend y devuelve un **puntaje de riesgo** entre 0 y 1.
La API no decide si es sospechosa: eso lo hace el backend con el umbral.

## Para probarla sin el backend

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Para probarla sin el backend: http://localhost:8000/docs



## Qué envía el backend: `POST /predict`

```json
{
  "transaccion_id": "TX-0083726",
  "monto": 245800,
  "fecha_hora": "2026-11-21T10:30:00",
  "categoria": "compras en línea",
  "monto_promedio_30d": 62400,
  "monto_promedio_categoria": 15200,
  "primera_vez_categoria": true,
  "segundos_desde_ultima": 540
}
```

| Campo | Regla |
|---|---|
| monto | En COP, mayor que 0 |
| fecha_hora | Hora de Colombia, `AAAA-MM-DDTHH:MM:SS` |
| categoria | Una de las 14 de la tabla de abajo |
| monto_promedio_30d | Promedio del cliente en los 30 días anteriores. Sin historial: el monto actual |
| monto_promedio_categoria | Promedio anterior del cliente en esa categoría. Sin historial: el monto actual |
| primera_vez_categoria | `true` si nunca había comprado en esa categoría |
| segundos_desde_ultima | Segundos desde su transacción anterior. Sin historial: 2592000 |

Las 4 últimas se calculan **solo con transacciones anteriores** del mismo cliente.

## Qué responde

```json
{ "transaccion_id": "TX-0083726", "puntaje_riesgo": 0.0024, "version_modelo": "1.1.0" }
```

## Decisión (en el backend)

- **Sospechosa** si `puntaje_riesgo >= 0.9241` o si se activa una regla crítica → abrir caso.
- **Sin alerta** en otro caso.

El umbral también está en `GET /model-info` (`umbral_sugerido`).

## Errores

| Código | Cuándo | Qué hacer |
|---|---|---|
| 422 | Dato inválido (monto ≤ 0, categoría que no existe, campo faltante) | Revisar el dato |
| 500 o sin respuesta en 3 s | Falla del servicio | `estado_analisis = "error"` y reintentar |

Para varias transacciones: `POST /predict/batch` con `{ "transacciones": [ ... ] }` (máx. 10.000).

## Categorías válidas

| En línea | Punto de venta | Otro |
|---|---|---|
| compras en línea | compras en supermercado | alimentación y restaurantes |
| compras diversas en línea | compras en establecimientos | cuidado personal |
| compras de supermercado en línea | compras diversas en establecimientos | entretenimiento |
| | | gasolina y transporte |
| | | hogar |
| | | niños y mascotas |
| | | salud y bienestar |
| | | viajes |



## Endpoints

| Método | Ruta | Para qué |
|---|---|---|
| GET | `/health` | Saber si el servicio está vivo y el modelo cargado |
| GET | `/model-info` | Versión, umbral sugerido, categorías válidas y métricas |
| POST | `/predict` | Puntaje de una transacción |
| POST | `/predict/batch` | Puntajes de hasta 10.000 transacciones (importación de CSV) |




