# ATLAS / LAB — Tecnología del movimiento

**[Abrir el pitch y la demostración pública](https://atlas-anatomia-del-gol.lahpm.chatgpt.site)**

Código y resultados del estudio de Quiñones: segmentación SAM 3.1, pose SAM 3D Body, revisión manual, rig MHR, visualización Three.js, simulación MuJoCo y render Blender Cycles.

## Ejecutar la web

```sh
python3 -m http.server 8000 --directory dist
```

Abrir http://localhost:8000. La demostración funciona con los resultados incluidos; no necesita claves ni una GPU remota.

## Contenido

| Carpeta | Tecnología |
| --- | --- |
| `dist/` | Web completa, visor 3D, editor de articulaciones, videos y datos |
| `pipeline/processing/` | Inferencia de máscaras y pose, exportación MHR, revisión anotada |
| `pipeline/blender/` | Rig original, cinemática inversa, transferencia de movimiento y escena |
| `pipeline/physics/` | Simulación de contacto, auditoría, prueba CUDA y películas Blender |
| `scripts/` | Verificación de resultados incluidos |

Ver **[pipeline/README.md](pipeline/README.md)** para reproducir la física y configurar los modelos externos. Los scripts de inferencia fueron adaptados para rutas portables; no se volvió a ejecutar la inferencia completa durante la preparación de este repositorio. Los resultados publicados corresponden al estudio original documentado abajo.

## Próxima etapa: tokens de movimiento y predicción

El pitch propone un dataset 3D de videos sincronizados: vistas perpendiculares con zoom del jugador y panorama de la cancha. El estado articular se representa en un espacio de configuración con dimensión **D = suma de los grados de libertad articulares**; un movimiento es una secuencia de transformaciones en ese espacio. Se propone tokenizar combinaciones de ángulos y sus cambios para entrenar una red neuronal.

El objetivo será predecir rendimiento multidimensional, ranking contextual, objetivos de entrenamiento y contribución esperada a la probabilidad de ganar, e informar la valuación con datos contractuales y comparables. La reconstrucción multivista de partidos permitirá explorar alternativas en simulación. **Ese modelo predictivo y la reconstrucción de partidos completos son una hoja de ruta; este repositorio no contiene una red entrenada de ranking o valuación.**

## Derechos y procedencia

Consultar [THIRD_PARTY.md](THIRD_PARTY.md). La publicación del repositorio no convierte el video de AtlasFC ni los pesos de terceros en material con una nueva licencia abierta. Los modelos de inferencia deben obtenerse por separado bajo sus condiciones. No se incluyen credenciales, acceso a estaciones privadas ni pesos de inferencia.

---

# ATLAS / LAB — Anatomía del gol

Estudio interactivo de Julián Quiñones en el segundo gol de Atlas ante Pachuca, final de ida del Clausura 2022, 26 de mayo de 2022. Diseño blanco, tipografía de máquina de escribir y acentos rojos.

La aplicación estática está en `dist/`. No requiere servidor de inferencia ni servicios de pago para reproducir el estudio. Incluye el fragmento de video, máscaras, coordenadas, correcciones, un humanoide animado y películas del proceso. Three.js 0.180.0 está incluido localmente con su licencia.

## Recorrido

1. Video oficial y control común por fotograma.
2. Máscara de Quiñones y segmentación independiente del balón con SAM 3.1.
3. Esqueleto 2D y evolución del ángulo de la rodilla.
4. Comparación inicial/revisada, edición de puntos y descarga de ajustes.
5. Humanoide MHR, cámara orbital, esqueleto y vectores articulares.
6. Simulación física de un remate: colisiones, tiro a gol, cámara orbital, zoom y toggles de ángulos, velocidad y aceleración. Película del mismo experimento en Blender Cycles.

## Fuente y reloj

- Fuente: [AtlasFC](https://www.youtube.com/watch?v=y4dl3bDOxYI).
- Frecuencia: 30000/1001 fotogramas por segundo.
- Fragmento: 342 fotogramas, 11.4114 segundos.
- Primer fotograma decodificado del fragmento: **2422** en la fuente original. El corte solicitado fue 80.8 s; el primer fotograma real corresponde a 80.8140667 s.
- Conversión exacta: `clipTime = (sourceFrame - 2422) * 1001 / 30000`.
- Contacto aproximado: fotograma fuente 2662; fotograma 240 del fragmento; 8.008 s.
- Animación principal: fuente 2500–2710; 0–7.007 s en el GLB; 2.6026–9.6096 s en el fragmento.

La película final usa el tiempo físico de la simulación y ralentiza el contacto para permitir la órbita. `dist/assets/physics/film-timing.json` conserva ese tiempo por fotograma. El video observado sigue siendo una repetición en cámara lenta: su reloj no determina velocidades físicas. La pieza de transferencia al rig es una visualización de esa repetición, separada de la simulación final.

## Evidencia y reconstrucción

SAM 3.1 ejecutó inferencia independiente en los 342 fotogramas, con resolución interna de 384 px. Se aceptaron 217 máscaras del jugador y se estimaron 217 poses con SAM 3D Body. Los otros 125 registros permanecen ausentes. El balón tiene 35 máscaras independientes: todos los fotogramas 2637–2670 y una observación parcialmente ocluida en 2679. Interpolar la guía de recorte no convierte esa guía en una medición del balón.

Se revisaron directamente 38 posiciones visibles en 16 fotogramas consecutivos del remate. Las correcciones afectan la muñeca derecha y los tobillos; las articulaciones ocultas conservan su estimación. El ajuste tridimensional usa cinemática inversa con profundidad estimada y longitudes óseas conservadas. Seis fotogramas de transición suavizan la entrada y salida de esos ajustes; no son nuevas observaciones.

El humanoide usa 18,439 vértices, 127 huesos y los pesos originales MHR. La animación principal contiene 202 poses observadas y nueve huecos breves interpolados. No se extrapolan los huecos largos. La pelvis se centra para estudiar el gesto: **no existe una trayectoria medida del jugador sobre la cancha**. El humanoide no es un escaneo anatómico de Quiñones.

El ángulo 3D de la web se calcula directamente sobre los huesos animados. Los vectores representan desplazamiento durante 0.1001 segundos de la repetición y se amplían visualmente cuatro veces. No representan velocidades físicas. La inclinación del desplazamiento del tobillo se calcula respecto al plano horizontal del modelo, con la pelvis centrada. En el fotograma 2662, la rodilla proyectada mide 167.9°, la rodilla del rig corregido 113.2° y la dirección relativa del tobillo 27.8°; son cantidades geométricas distintas, no mediciones intercambiables. `vector-analysis.csv` contiene los 342 fotogramas, huecos explícitos, ángulos y desplazamientos del rig. La dirección inicial del balón (17.9°) es una proyección entre dos centros visibles en la imagen, afectada por el movimiento de cámara; no es un ángulo balístico 3D.

## Física del remate

MuJoCo 3.13.0 integra un humanoide articulado y un balón dinámico a 0.10 ms. La bota derecha se mantiene plantada mediante una restricción; las articulaciones usan servos de posición con fuerza limitada. El balón parte sin velocidad horizontal y no tiene actuadores. La abertura de la portería es 7.32 × 2.44 m, a 12 m; el fondo es un plano rígido de contención.

Es un experimento mecánico de remate zurdo, con anatomía y tiempos supuestos, no una medición de fuerza o velocidad de Quiñones. Las superficies visibles coinciden con los volúmenes del motor. El video, el rig MHR y la simulación tienen relojes y procedencias separados.

La web permite girar, acercar, pausar, explorar el contacto y activar ángulos del lado izquierdo, velocidades de articulaciones/ball y aceleraciones. Reproduce 1,371 estados de la simulación; no ejecuta un motor físico remoto. Las aceleraciones son derivadas temporales de las velocidades exportadas. Los vectores tienen escala declarada y longitud visual limitada, sin limitar sus cifras.

Comprobaciones independientes en `dist/assets/physics/audit.json`: cero movimiento horizontal del balón al retirar el contacto humano; gravedad ajustada −9.810003 m/s² en vuelo; error geométrico máximo de 0.5 μm; distancia pie–balón positiva en 2,501 tiempos intermedios; cruce de gol. El ensayo registra cero interpenetración entre pares de colisión. La compatibilidad del modelo completo se verificó con un paso CUDA en una NVIDIA RTX PRO 6000 Blackwell mediante MuJoCo Warp; el tiro individual se calculó en CPU por su baja latencia. La película usa Cycles / Metal en la Mac Studio.

El bloque Performance Lab explica su uso para entrenamientos personalizados y ranking de performance con pruebas comparables. No inventa un ranking con un único remate.

## Edición

Los ajustes del editor afectan la anotación 2D de cada fotograma y sus lecturas. Se conservan durante la sesión y pueden descargarse con índices de fuente, tiempos y coordenadas. El editor no vuelve a ejecutar SAM ni recalcula el render o el GLB.

## Alcance y derechos

Es un caso de un jugador y una jugada. No es un análisis de todos los jugadores, del partido completo ni una valoración económica. Las cifras físicas se atribuyen al experimento simulado; no se presentan como fuerza o velocidad real de Quiñones, ni como riesgo de lesión o valor de mercado.

El fragmento de 720p se volvió a exportar a CRF17 desde la misma fuente, conservando sus 342 fotogramas, sin inventar resolución adicional.

El fragmento audiovisual se atribuye a su fuente; no se declara una licencia abierta sobre él. No se redistribuye el partido ni el video original completo. La licencia del modelo MHR está en `dist/assets/MHR-LICENSE.txt`; la de Three.js, en `dist/vendor/LICENSE-three.txt`. Las fuentes, intervenciones y límites están disponibles también dentro de la web.

## Verificación

`python3 scripts/verify_assets.py` verifica índices, tiempos, máscaras, coordenadas finitas, estructura del GLB, activos y películas. Requiere Pillow y `ffprobe` para las verificaciones de imágenes y video. No realiza navegación ni captura del sitio.

Para una vista local, servir `dist/` mediante un servidor HTTP estático. El despliegue público se mantiene por separado de este repositorio.
