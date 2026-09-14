from pathlib import Path
import bpy,json,math,bisect,sys
from mathutils import Vector,Matrix
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'output/simulation.json').read_text());fs=data['frames'];times=[f['t'] for f in fs]
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'blender/Atlas_Physics.blend'))
s=bpy.context.scene;s.render.resolution_x=1920;s.render.resolution_y=1080;s.render.resolution_percentage=100;s.cycles.samples=32;s.render.fps=24
s.render.use_persistent_data=True;s.frame_start=1;s.frame_end=240
try:
    p=bpy.context.preferences.addons['cycles'].preferences;p.compute_device_type='METAL';p.get_devices()
    for d in p.devices:d.use=d.type=='METAL'
    s.cycles.device='GPU'
except Exception:pass
contact=data['metadata']['firstBootContact'];end_contact=data['metadata']['lastBootContact']
objects=[bpy.data.objects.get(g['name']) for g in data['geoms']]
for o in list(bpy.data.objects):
    if o.name.startswith(('Velocity ','Vector tip ')):bpy.data.objects.remove(o,do_unlink=True)
vector_material=bpy.data.materials.new('Velocity vectors');vector_material.diffuse_color=(.45,.006,.002,1);vector_material.use_nodes=True;vector_material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.45,.006,.002,1)
vectors=[]
for name in ['left_hip','left_knee','left_ankle','left_wrist','left_elbow','right_hip','right_knee','right_ankle','right_wrist','right_elbow','ball_center']:
    bpy.ops.mesh.primitive_cylinder_add(vertices=10,radius=.0025,depth=1);shaft=bpy.context.object;shaft.name='Velocity '+name;shaft.rotation_mode='QUATERNION';shaft.data.materials.append(vector_material)
    bpy.ops.mesh.primitive_cone_add(vertices=16,radius1=.014,radius2=0,depth=.045);tip=bpy.context.object;tip.name='Vector tip '+name;tip.rotation_mode='QUATERNION';tip.data.materials.append(vector_material)
    vectors.append((data['sites'].index(name),shaft,tip))
def smooth(t):return max(0,min(1,t))**2*(3-2*max(0,min(1,t)))
def sample(t):
    ix=min(len(fs)-2,max(0,bisect.bisect_right(times,t)-1));a,b=fs[ix],fs[ix+1];u=max(0,min(1,(t-a['t'])/(b['t']-a['t'])));return a,b,u
physical_times=[]
for fi in range(240):
    if fi<60:t=(contact-.025)*fi/60
    elif fi<150:t=contact-.025+(end_contact+.055-(contact-.025))*(fi-60)/90
    else:t=end_contact+.055+(2.9-(end_contact+.055))*(fi-150)/89
    physical_times.append(t)
    a,b,u=sample(t)
    for si,shaft,tip in vectors:
        pos=Vector(a['j'][si]).lerp(Vector(b['j'][si]),u);v=Vector(a['v'][si]).lerp(Vector(b['v'][si]),u);mag=v.length;length=min(.9,mag*.06);shown=45<=fi<=160 and length>.015
        direction=v.normalized() if mag>1e-7 else Vector((0,0,1));q=direction.to_track_quat('Z','Y')
        shaft.location=pos+direction*length/2;shaft.rotation_quaternion=q;shaft.scale=(1,1,max(length,.00001)) if shown else (.00001,.00001,.00001)
        tip.location=pos+direction*length;tip.rotation_quaternion=q;tip.scale=(1,1,1) if shown else (.00001,.00001,.00001)
        for o in[shaft,tip]:
            o.keyframe_insert('location',frame=fi+1);o.keyframe_insert('rotation_quaternion',frame=fi+1);o.keyframe_insert('scale',frame=fi+1)
    for i,o in enumerate(objects):
        if not o or data['geoms'][i]['type']==0:continue
        o.location=Vector(a['p'][i]).lerp(Vector(b['p'][i]),u)
        ar=a['r'][i];br=b['r'][i];qa=Matrix([ar[:3],ar[3:6],ar[6:]]).to_quaternion();qb=Matrix([br[:3],br[3:6],br[6:]]).to_quaternion();o.rotation_quaternion=qa.slerp(qb,u)
        o.keyframe_insert('location',frame=fi+1);o.keyframe_insert('rotation_quaternion',frame=fi+1)
    # A close orbital study, then a rising camera that reveals the entire shot.
    theta=-.98+min(fi/150,1)*math.pi*1.15
    target=Vector((.10,0,.85));pos=target+Vector((math.cos(theta)*4.65,math.sin(theta)*4.65,1.30))
    flight=smooth((fi-150)/65)
    target=target.lerp(Vector((6.8,0,.75)),flight)
    pos=pos.lerp(Vector((6.8,-20,6.8)),flight)
    cam=s.camera;cam.location=pos;cam.rotation_euler=(target-pos).to_track_quat('-Z','Y').to_euler();cam.data.lens=48-12*flight;cam.data.keyframe_insert('lens',frame=fi+1)
    cam.keyframe_insert('location',frame=fi+1);cam.keyframe_insert('rotation_euler',frame=fi+1)
s['physical_model']='MuJoCo dynamic articulated body, planted support, unforced ball, exact visible collision surfaces.'
s['simulation_time_per_frame']=json.dumps(physical_times);s['source']='Independent physical reconstruction inspired by the observed left-footed shot. Not measured athlete force or timing.'
bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'blender/Atlas_Physics.blend'))
(ROOT/'blender/film-timing.json').write_text(json.dumps({'fps':24,'frames':240,'physicalTimes':physical_times,'engine':'Cycles','samples':32,'device':'Mac Studio / Metal','resolution':[1920,1080]},indent=2))
from bpy_extras.object_utils import world_to_camera_view
framing=[]
for frame in [210,240]:
    s.frame_set(frame);bpy.context.view_layer.update();points=[]
    for i,o in enumerate(objects):
        if not o or data['geoms'][i]['type']==0:continue
        points.extend(world_to_camera_view(s,s.camera,o.matrix_world@Vector(corner)) for corner in o.bound_box)
    bounds=[min(v.x for v in points),max(v.x for v in points),min(v.y for v in points),max(v.y for v in points)]
    framing.append({'frame':frame,'bounds':bounds})
    assert min(bounds[0],bounds[2])>.01 and max(bounds[1],bounds[3])<.99, f'Final camera crops geometry: {bounds}'
(ROOT/'blender/camera-audit.json').write_text(json.dumps(framing,indent=2))
frames=ROOT/'blender/frames';frames.mkdir(exist_ok=True);s.render.image_settings.file_format='PNG';s.render.filepath=str(frames/'frame_')
indices=[1,80,150,210,240] if '--probes' in sys.argv else [] if '--prepare' in sys.argv else range(1,241)
if '--probes' in sys.argv:frames=ROOT/'blender/probes';frames.mkdir(exist_ok=True)
for fi in indices:
    s.frame_set(fi);s.render.filepath=str(frames/f'frame_{fi:04d}.png');bpy.ops.render.render(write_still=True)
    if fi%12==0:print(f'PHYSICS_FILM {fi}/240',flush=True)
