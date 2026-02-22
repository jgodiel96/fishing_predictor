# Encuesta para Pescadores - Recoleccion de Datos

**Objetivo:** Recopilar datos reales de pesca desde orilla para entrenar modelo ML supervisado.

**Tiempo estimado:** 60-90 segundos por encuesta

---

## Estrategia de Abordaje

### Frase de Apertura
> "Hola! Estamos desarrollando una app gratuita para predecir las mejores zonas y horarios de pesca desde orilla. ¿Nos ayudas con 1 minuto? Te compartimos las predicciones cuando este lista."

### Tips para el Evento
1. **NO parecer oficial** - ropa casual, actitud relajada
2. **Ofrecer algo a cambio** - acceso a la app, predicciones por WhatsApp
3. **Respetar si no quieren dar zona exacta** - las zonas generales tambien sirven
4. **"Nada" es dato valido** - saber donde NO hay pesca es igual de util
5. **No insistir** - si se ponen incomodos, agradecer y pasar al siguiente

---

## Cuestionario Rapido (Version Papel)

### 1. ZONA DE PESCA (marcar una)
- [ ] Punta Coles / Reserva
- [ ] Vila Vila / Pozo Redondo
- [ ] Ilo Puerto / Fundicion
- [ ] Boca del Rio / Tacna
- [ ] Punta Mesa / Santa Rosa
- [ ] Punta Picata / Los Palos
- [ ] Otra: ________________

### 2. CUANDO PESCASTE (ultima salida)
**Fecha:**
- [ ] Hoy
- [ ] Ayer
- [ ] Esta semana
- [ ] Este mes
- [ ] Fecha exacta: ___/___/2026

**Hora de inicio:**
- [ ] Madrugada (4-6am)
- [ ] Manana (6-10am)
- [ ] Mediodia (10am-2pm)
- [ ] Tarde (2-6pm)
- [ ] Noche (6pm+)

**Duracion aprox:**
- [ ] 1-2 horas
- [ ] 3-4 horas
- [ ] 5+ horas

### 3. QUE CAPTURASTE
**Especies (marcar todas):**
- [ ] Cabrilla
- [ ] Corvina
- [ ] Robalo
- [ ] Pejerrey
- [ ] Lenguado
- [ ] Tramboyo
- [ ] Chita
- [ ] Otra: ________________
- [ ] NADA (dia malo)

**Cantidad total aprox:**
- [ ] 0 (nada)
- [ ] 1-2
- [ ] 3-5
- [ ] 6-10
- [ ] Mas de 10

**Tamano promedio:**
- [ ] Chico (< 25cm)
- [ ] Mediano (25-40cm)
- [ ] Grande (> 40cm)

### 4. METODO DE PESCA
- [ ] Spinning (senuelos artificiales)
- [ ] Rockfishing (jigs pequenos)
- [ ] Carnada natural
- [ ] Mosca
- [ ] Otro: ________________

**Senuelo/carnada que funciono:**
________________

### 5. CONDICIONES (si recuerda)
**Estado del mar:**
- [ ] Calmo
- [ ] Regular
- [ ] Picado/Bravo

**Temperatura del agua (sensacion):**
- [ ] Fria
- [ ] Normal
- [ ] Tibia

**Viento:**
- [ ] Poco/Nada
- [ ] Moderado
- [ ] Fuerte

**Claridad del agua:**
- [ ] Clara
- [ ] Turbia
- [ ] Muy turbia

---

## Preguntas Opcionales (si hay confianza)

### 6. TIPO DE FONDO/ESTRUCTURA
- [ ] Roca grande
- [ ] Pozas
- [ ] Arena
- [ ] Mixto roca/arena
- [ ] Muelle/escollera

### 7. FRECUENCIA DE PESCA
- [ ] Varias veces por semana
- [ ] Semanal
- [ ] Quincenal
- [ ] Mensual
- [ ] Ocasional

### 8. EXPERIENCIA
- [ ] Principiante (< 1 ano)
- [ ] Intermedio (1-5 anos)
- [ ] Experimentado (5+ anos)

---

## Contacto (Opcional)

> "Te avisamos cuando la app este lista. Es gratis!"

**WhatsApp:** ______________________

**Email:** ______________________

---

## Para el Encuestador

**Fecha del evento:** ___/___/2026

**Ubicacion del evento:** ________________

**Numero de encuesta:** ____

**Notas adicionales:**
_________________________________________________
_________________________________________________

---

## Formato de Datos para el Modelo

Cada encuesta genera un registro con:

| Campo | Tipo | Ejemplo |
|-------|------|---------|
| zona | categoria | "punta_coles" |
| fecha | date | "2026-02-15" |
| hora_inicio | categoria | "manana" |
| duracion_horas | float | 3.5 |
| especie_1 | categoria | "cabrilla" |
| especie_2 | categoria | null |
| cantidad | int | 4 |
| tamano | categoria | "mediano" |
| metodo | categoria | "spinning" |
| mar | categoria | "calmo" |
| temp_agua | categoria | "normal" |
| viento | categoria | "poco" |
| claridad | categoria | "clara" |
| exito | bool | true (cantidad > 0) |

### Coordenadas por Zona (para el modelo)

| Zona | Lat | Lon |
|------|-----|-----|
| punta_coles | -17.702 | -71.385 |
| vila_vila | -17.630 | -71.340 |
| ilo_puerto | -17.640 | -71.340 |
| pozo_redondo | -17.680 | -71.370 |
| fundicion | -17.660 | -71.350 |
| boca_rio | -18.050 | -70.850 |
| punta_mesa | -17.850 | -71.200 |
| santa_rosa | -17.750 | -71.300 |
| punta_picata | -18.200 | -70.900 |

---

## Modelo de Dos Capas

```
Capa 1: Modelo Heuristico (actual)
├── Input: Condiciones oceanograficas (SST, olas, corrientes, etc.)
├── Output: Score base 0-100
└── Metodo: Reglas basadas en conocimiento oceanografico

Capa 2: Modelo Supervisado (nuevo)
├── Input: Score Capa 1 + Features adicionales
├── Target: Exito real de pesca (datos de encuestas)
├── Output: Score ajustado 0-100
└── Metodo: Gradient Boosting entrenado con datos reales
```

El modelo de Capa 2 aprende a CORREGIR las predicciones de Capa 1 basandose en datos reales.

---

*Documento creado: 2026-02-15*
