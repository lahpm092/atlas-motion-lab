"""Render the same MuJoCo collision surfaces and exported solver transforms."""
from pathlib import Path
import bpy,json,math,sys
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'output/simulation.json').read_text())
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=32;s.cycles.use_denoising=True
try:
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='METAL'
    s.cycles.device='GPU'
except Exception:pass
s.render.resolution_x=1600;s.render.resolution_y=1000;s.render.resolution_percentage=100
s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.55;s.view_settings.view_transform='AgX';s.view_settings.look='AgX - Medium High Contrast'
def material(name,color,rough=.3,metal=.0):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    return m
white=material('Ceramic anatomical surface',(.35,.40,.37),.3,.15);ink=material('Graphite boots',(.014,.019,.016),.4,.05);red=material('Atlas ball',(.36,.009,.004),.4);floor=material('Studio pitch',(.82,.82,.80),.8);line=material('Pitch markings',(.24,.28,.23),.7)
def capsule(radius,half):
    verts=[];faces=[];segments=48;lat=20
    angles=[math.pi*i/lat for i in range(lat//2+1)]+[math.pi*i/lat for i in range(lat//2,lat+1)]
    for i,theta in enumerate(angles):
        z=radius*math.cos(theta)+(half if i<=lat/2 else -half);r=radius*math.sin(theta)
        for j in range(segments):verts.append((r*math.cos(2*math.pi*j/segments),r*math.sin(2*math.pi*j/segments),z))
        if i:
            for j in range(segments):a=(i-1)*segments+j;b=(i-1)*segments+(j+1)%segments;c=i*segments+(j+1)%segments;d=i*segments+j;faces.append((a,b,c,d))
    mesh=bpy.data.meshes.new('capsule');mesh.from_pydata(verts,[],faces);mesh.update();o=bpy.data.objects.new('capsule',mesh);bpy.context.collection.objects.link(o);return o
objects=[]
for g in data['geoms']:
    size=g['size'];typ=g['type']
    if typ==0:bpy.ops.mesh.primitive_plane_add(size=200);o=bpy.context.object;o.data.materials.append(floor)
    elif typ in[2,4]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=32,radius=size[0] if typ==2 else 1);o=bpy.context.object
        if typ==4:o.scale=size;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    elif typ==3:o=capsule(size[0],size[1])
    elif typ==6:bpy.ops.mesh.primitive_cube_add(size=2);o=bpy.context.object;o.scale=size;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    else:objects.append(None);continue
    o.name=g['name'];o['physics_geom']=g['name'];o.rotation_mode='QUATERNION'
    if typ!=0:o.data.materials.append(red if g['name']=='ball' else ink if g['name'].endswith('_boot') else white)
    for p in o.data.polygons:p.use_smooth=typ not in[0,6]
    if g['name']=='net_back':o.hide_render=True
    objects.append(o)
def rod(a,b,r,mat):
    delta=Vector(b)-Vector(a);o=capsule(r,delta.length/2);o.location=(Vector(a)+Vector(b))/2;o.rotation_mode='QUATERNION';o.rotation_quaternion=delta.to_track_quat('Z','Y');o.data.materials.append(mat);return o
for depth,width in [(16.5,20.16),(5.5,9.16)]:
    rod((12-depth,-width,.003),(12,-width,.003),.008,line);rod((12-depth,width,.003),(12,width,.003),.008,line);rod((12-depth,-width,.003),(12-depth,width,.003),.008,line)
rod((12,-34,.003),(12,34,.003),.015,line)
for i in range(41):y=-3.6+i*.18;rod((13.98,y,.03),(13.98,y,2.4),.0025,line)
for i in range(14):h=.06+i*.18;rod((13.98,-3.66,h),(13.98,3.66,h),.0025,line)
for y in[-3.66,3.66]:rod((12,y,2.44),(14,y,2.44),.028,white);rod((14,y,0),(14,y,2.44),.025,white)
def light(name,pos,energy,size):
    d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=pos;o.rotation_euler=(Vector((1,0,.8))-o.location).to_track_quat('-Z','Y').to_euler()
light('Softbox',(-1,-4,7),950,5);light('Rim',(2,4,5),650,4);light('Goal fill',(9,-2,6),900,6)
bpy.ops.object.camera_add(location=(3,-4.5,2.1));cam=bpy.context.object;cam.data.lens=48;s.camera=cam;cam.rotation_euler=(Vector((.2,0,.9))-cam.location).to_track_quat('-Z','Y').to_euler()
f=min(data['frames'],key=lambda f:abs(f['t']-data['metadata']['firstBootContact']))
for i,o in enumerate(objects):
    if o is None or data['geoms'][i]['type']==0:continue
    o.location=f['p'][i];r=f['r'][i];o.rotation_quaternion=Matrix([r[:3],r[3:6],r[6:]]).to_quaternion()
out=ROOT/'blender';out.mkdir(exist_ok=True)
s.render.image_settings.file_format='PNG';s.render.filepath=str(out/'contact.png');bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(out/'Atlas_Physics.blend'));bpy.ops.render.render(write_still=True)
