from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'blender/Atlas_Physics.blend'))
s=bpy.context.scene
p=bpy.context.preferences.addons['cycles'].preferences;p.compute_device_type='METAL';p.get_devices()
for d in p.devices:d.use=d.type=='METAL'
s.cycles.device='GPU';s.render.use_persistent_data=True
for fi in range(1,241):
    s.frame_set(fi);s.render.filepath=str(ROOT/'blender/frames'/f'frame_{fi:04d}.png');bpy.ops.render.render(write_still=True)
    if fi%24==0:print(f'FINAL_PHYSICS_FILM {fi}/240',flush=True)
