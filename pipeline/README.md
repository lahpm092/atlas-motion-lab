# Reproducir la tecnología

## 1. Física: experimento independiente

Desde la raíz del repositorio, con Python 3.12:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r pipeline/physics/requirements.txt
python pipeline/physics/simulate.py --ball-x .28 --ankle -.4 --swing .16 --export
python pipeline/physics/audit.py
```

Las salidas completas se guardan en `pipeline/physics/output/`. El balón se mueve por contacto con el cuerpo; no se prescribe su trayectoria. La bota derecha queda plantada. No es una estimación de las fuerzas reales del jugador. Los resultados de referencia están en `dist/assets/physics/`.

Opcional, en NVIDIA/CUDA con MuJoCo Warp instalado:

```sh
python pipeline/physics/gpu_compatibility.py
```

Esta prueba ejecuta un paso CUDA del modelo. La simulación original del tiro se calculó en CPU; no se afirma equivalencia numérica completa entre backends.

## 2. Película de la física

Blender 5.1.2 con su ejecutable disponible como `blender`:

```sh
blender -b --python pipeline/physics/build_blender.py
blender -b --python pipeline/physics/render_film.py
pip install imageio-ffmpeg==0.6.0
python pipeline/physics/encode_film.py
```

Los scripts originales seleccionan Metal en Apple Silicon. En NVIDIA, seleccionar CUDA/OptiX en los scripts o usar la escena lista para abrir: `dist/assets/physics/Atlas_Physics.blend`. No requieren acceso a las máquinas usadas para el estudio. La película ralentiza el contacto; el archivo de tiempos conserva el reloj físico.

## 3. Segmentación y pose: Apple Silicon

Los resultados ya están incluidos. Para repetir inferencia, preparar un entorno separado con NumPy, OpenCV, Pillow y `mlx-vlm==0.4.3`. Descargar el modelo MLX SAM 3.1 y configurar `FIGHTLAB_SAM31` con su directorio. El estudio usó una conversión local de 4 bits de `mlx-community/sam3.1-bf16`; sus pesos no se incluyen.

Obtener la fuente oficial indicada en la web bajo las condiciones aplicables y colocar el video original completo en `pipeline/source/atlas-quinones-pachuca-2022-source.mp4`. El clip de `dist/` no sustituye al original para estos índices absolutos.

```sh
python pipeline/processing/sam31_atlas_v3.py --stride 1 --resolution 384
python pipeline/processing/segment_ball_dense.py
```

La selección de identidad usa heurísticas del uniforme y encuadre de esta jugada; no es un tracker general de todos los jugadores. Las guías del balón son recortes, no máscaras interpoladas.

Para la pose, instalar el proyecto oficial https://github.com/facebookresearch/sam-3d-body con sus dependencias en un entorno compatible con PyTorch/MPS. Configurar `SAM3D_SOURCE`, `SAM3D_CHECKPOINT` y `MHR_MODEL` a la fuente y pesos descargados. Valores predeterminados bajo `pipeline/models/`.

```sh
python pipeline/processing/pose3d_atlas_local.py
python pipeline/processing/export_motion_local.py
```

Estos scripts usan MPS y mmap para reducir memoria. Los JSON intermedios no son automáticamente el dataset final de la web: la revisión, alineación temporal y ausencias se conservan en `dist/assets/analysis.json`. No se automatiza ni se oculta la revisión manual.

## 4. Corrección y transferencia MHR

Se incluyen el rig derivado de MHR, los estados inferidos originales y las 38 correcciones revisadas. Con NumPy:

```sh
python pipeline/blender/apply_manual_ik.py --input pipeline/processing/motion_mhr.npz --targets pipeline/processing/manual-strike-sequence.json --output pipeline/processing/motion-corrected.npz --mapping pipeline/processing/model_mappings.npz
blender -b --python pipeline/blender/build_scene.py
blender -b --python pipeline/blender/apply_motion.py -- --root pipeline/blender --motion pipeline/processing/motion-corrected.npz
```

La IK conserva longitudes y profundidad estimada. `normalize_glb_time.py` documenta la corrección del reloj del exportador. La escena final de referencia `dist/assets/Atlas_Quinones_Motion.blend` y los GLB ya contienen los resultados revisados. El MHR se centra en la pelvis; no reconstruye un recorrido calibrado por la cancha.

## 5. Verificación del estudio incluido

Con Pillow y FFmpeg/ffprobe disponibles:

```sh
python scripts/verify_assets.py
```

Los datos de video, pose 3D inferida y física simulada tienen relojes y procedencias distintos. No convertir una repetición en cámara lenta en velocidades reales del atleta.
