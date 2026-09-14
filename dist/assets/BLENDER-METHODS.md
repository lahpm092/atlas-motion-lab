# Atlas / reconstrucción de movimiento

La cancha mide 105 × 68 metros; las porterías miden 7.32 × 2.44 metros. Esta geometría sirve de escenario visual. No representa una calibración de la cámara de televisión.

El humanoide conserva la superficie anatómica original de Momentum Human Rig v1.0.1: 18,439 vértices base, 127 huesos, pesos de piel y matrices inversas de enlace. Se distribuye con su licencia Apache-2.0. El uniforme rojo y negro es una superficie técnica del modelo; el cuerpo y la cara son genéricos y no constituyen un escaneo ni una reproducción de la identidad física de Julián Quiñones.

Las poses se importan desde SAM 3D Body aplicado a fotogramas reales de la repetición. El ajuste mantiene el centro horizontal de la pelvis fijo y alinea los contactos con el suelo de manera inferida. No se ha medido la traslación absoluta del futbolista por la cancha. La profundidad, las partes ocultas y los contactos siguen siendo estimaciones de una sola cámara.

El GLB contiene el intervalo continuo de los fotogramas fuente 2500–2710: 202 poses observadas y nueve fotogramas inferidos entre observaciones cercanas. Su duración es 7.007 segundos a 30000/1001 fotogramas por segundo. Empieza a los 2.6026 segundos del archivo de vídeo recortado, cuyo primer fotograma decodificado es el 2422. Las oclusiones largas quedan fuera de la animación. Los archivos `humanoid-before.glb` y `humanoid.glb` comparten exactamente esa línea temporal.

La revisión manual usa 38 objetivos anatómicos marcados sobre 16 fotogramas consecutivos del remate, desde el 2655 hasta el 2670. El ajuste de muñeca derecha y ambos tobillos conserva la profundidad estimada de cada extremo y la longitud de sus dos segmentos óseos. Las anotaciones ocluidas permanecen ausentes; seis fotogramas exteriores facilitan una transición inferida. La pequeña diferencia de reproyección mide el ajuste matemático a esos objetivos aproximados y no demuestra exactitud subpíxel de las articulaciones reales.

Los vectores se calculan a partir de las posiciones del esqueleto importado. La longitud gráfica de las flechas se escala y limita para favorecer su lectura; no es una escala directa de velocidad física. La animación del GLB usa segundos de la repetición televisiva. Como la fuente reproduce la acción en cámara lenta, una derivada por segundo de reproducción no equivale a la velocidad física del jugador. El vídeo cinematográfico añade una ralentización explícita en el remate para permitir la órbita de la cámara.

Las observaciones manuales del balón permanecen en coordenadas de imagen. Su representación tridimensional, cuando está visible, usa un plano de pantalla y una profundidad inferida; no es una trayectoria balística calibrada. Los intervalos con centro oculto no se presentan como observaciones.

La superficie se renderiza con Cycles y ray tracing. Las líneas y flechas se proyectan desde las posiciones tridimensionales correctas y se componen sobre la superficie para permanecer legibles. La auditoría de transferencia compara las articulaciones de Blender con las articulaciones MHR importadas; verifica la implementación, no la precisión del movimiento real.

La película usa 240 fotogramas a 24 fps, resolución 1920 × 1080 y 24 muestras adaptativas de Cycles. Se repartió entre una Mac Studio M3 Ultra con MetalRT y una NVIDIA RTX PRO 6000 con OptiX, con eliminación de ruido en GPU. La cámara orbita 250 grados y extiende tres segundos la ventana del remate. El archivo Blender es autónomo: no requiere bibliotecas, imágenes, tipografías ni rutas externas. El arco rojo fino representa únicamente el recorrido estimado del tobillo izquierdo durante el remate, de 88.50 a 89.22 segundos de la fuente.

Archivos de referencia: `rig-semantics.json`, `blender-manifest.json`, `rig-motion.json`, `retarget-audit.json`, `render-validation.json` y `MHR-LICENSE.txt`.

Rig original: https://github.com/facebookresearch/MHR
Pose monocular: https://github.com/facebookresearch/sam-3d-body
