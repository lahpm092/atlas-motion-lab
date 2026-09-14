# Atlas — simulación de contacto

MuJoCo 3.13.0, unidades SI. El cuerpo articulado tiene articulaciones dinámicas y servos con límites de fuerza. La bota de apoyo se mantiene plantada mediante una restricción. El balón no tiene actuadores ni velocidad inicial horizontal. Las superficies visibles coinciden con los volúmenes de colisión. La portería tiene abertura de 7.32 × 2.44 m y contención rígida en el fondo.

Este experimento representa un remate de izquierda. La anatomía, las masas y la duración son supuestos del modelo; no son mediciones de Quiñones. El video observado, la pose MHR y esta simulación tienen procedencia y relojes distintos. La vista web reproduce los estados calculados; no vuelve a simular al arrastrar la cámara.

Para reproducir desde un entorno Python 3.12:

    pip install -r requirements.txt
    python simulate.py --ball-x .28 --ankle -.4 --swing .16 --export
    python audit.py

Las salidas completas se crean en output/. La copia ligera simulation.json de la web omite qpos y qvel, que el programa vuelve a producir.

Para comprobar compatibilidad CUDA, después de generar output/model.xml:

    python gpu_compatibility.py

MuJoCo Warp ejecutó un paso del mismo modelo en una NVIDIA RTX PRO 6000 Blackwell. El cálculo del tiro individual se ejecutó en CPU (alrededor de dos segundos); el motor está diseñado para baja latencia. Warp sirve para cálculos paralelos en GPU. La prueba de compatibilidad no es una simulación completa en GPU.

Comprobaciones independientes: contacto habilitado/deshabilitado, gravedad en vuelo libre, cruce de la portería, coincidencia geométrica y distancia pie–balón durante la interpolación. audit.json contiene los resultados. El paso de integración es de 0.10 ms; la región del contacto se exporta con esa misma resolución temporal. Los vectores de aceleración son diferencias temporales de las velocidades del motor.

Fuentes del motor: https://github.com/google-deepmind/mujoco y https://github.com/google-deepmind/mujoco_warp (Apache 2.0). El render se genera en Blender 5.1.2 / Cycles / Metal. La película ralentiza el contacto; film-timing.json conserva el tiempo físico de cada fotograma.
