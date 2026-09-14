"""Bounded compatibility probe; the single-shot production uses low-latency CPU."""
from pathlib import Path
import json,time,hashlib
import numpy as np
print('IMPORT_MUJOCO',flush=True)
import mujoco
print('IMPORT_WARP',flush=True)
import mujoco_warp as mjw,warp as wp
ROOT=Path(__file__).resolve().parent
print('INITIALIZE_DEVICE',flush=True)
wp.init()
with wp.ScopedDevice('cuda:0'):
    cpu=mujoco.MjModel.from_xml_path(str(ROOT/'output/model.xml'))
    cd=mujoco.MjData(cpu)
    model=mjw.put_model(cpu)
    data=mjw.put_data(cpu,cd,nconmax=128,njmax=1024)
    before=data.qpos.numpy().copy();start=time.perf_counter()
    mjw.step(model,data);wp.synchronize()
    after=data.qpos.numpy()
    report={'engine':'MuJoCo Warp','version':mjw.__version__ if hasattr(mjw,'__version__') else mujoco.__version__,'device':str(wp.get_device()),'gpu':wp.get_device().name,'steps':1,'secondsIncludingJIT':time.perf_counter()-start,'finite':bool(np.isfinite(after).all()),'maxPositionChange':float(np.max(np.abs(after-before))),'purpose':'Compatibility probe of the exact articulated model, including contacts and planted-foot weld. Production single-shot data uses MuJoCo CPU for latency.'}
    report['modelSHA256']=hashlib.sha256((ROOT/'output/model.xml').read_bytes()).hexdigest()
    (ROOT/'output/gpu-compatibility.json').write_text(json.dumps(report,indent=2));print('GPU_COMPATIBILITY',json.dumps(report),flush=True)
