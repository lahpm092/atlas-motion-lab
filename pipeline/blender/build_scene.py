import bpy,numpy as np,math,json,argparse,sys
from pathlib import Path
from mathutils import Vector,Matrix,Quaternion,kdtree
ROOT=Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
A=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
D=dict(np.load(ROOT/'assets'/'rig.npz'));rest=D['rest_state'];parents=D['skeleton.joint_parents']
def mat(name,c,rough=.5,metal=0):
 m=bpy.data.materials.new(name);m.diffuse_color=(*c,1);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*c,1);p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal;return m
white=mat('Porcelain / neutral digital athlete',(.72,.735,.74),.31,.05)
ivory=mat('Laboratory floor',(.82,.835,.82),.65)
red=mat('Atlas / cadmium red',(.53,.008,.019),.43)
black=mat('Carbon / technical cloth',(.012,.013,.014),.44)
ink=mat('Measurement / graphite',(.045,.047,.047),.62)
acid=mat('Ball tracking / signal green',(.64,.92,.019),.28)
vector=mat('Joint vectors / red',(.66,.006,.02),.26)
chrome=mat('Goal frame / ceramic aluminium',(.65,.68,.68),.25,.35)
netmat=mat('Net / graphite',(.16,.18,.17),.7)
# Subtle fabric microsurface, physically lit.
for m in [red,black]:
 nt=m.node_tree;n=nt.nodes.new('ShaderNodeTexNoise');n.inputs['Scale'].default_value=320;n.inputs['Detail'].default_value=2;b=nt.nodes.new('ShaderNodeBump');b.inputs['Strength'].default_value=.16;b.inputs['Distance'].default_value=.0003;nt.links.new(n.outputs['Fac'],b.inputs['Height']);nt.links.new(b.outputs['Normal'],nt.nodes.get('Principled BSDF').inputs['Normal'])
def cube(name,loc,dim,material,bevel=0):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.name=name;o.dimensions=dim;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(material)
 if bevel:md=o.modifiers.new('Manufactured edge','BEVEL');md.width=bevel;md.segments=3;o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
 return o
def line(name,pts,r=.018,material=ink,closed=False):
 cu=bpy.data.curves.new(name,'CURVE');cu.dimensions='3D';cu.resolution_u=1;cu.bevel_depth=r;cu.resolution_u=2;cu.bevel_resolution=2;s=cu.splines.new('POLY');s.points.add(len(pts)-1)
 for q,p in zip(s.points,pts):q.co=(*p,1)
 s.use_cyclic_u=closed;o=bpy.data.objects.new(name,cu);bpy.context.collection.objects.link(o);cu.materials.append(material);return o
def label(name,text,loc,size,material=ink,rot=(0,0,0),align='LEFT'):
 c=bpy.data.curves.new(name,'FONT');c.body=text;c.size=size;c.align_x=align;c.space_character=1.14;c.extrude=.0004
 for p in ['/System/Library/Fonts/Supplemental/Courier New.ttf','/System/Library/Fonts/Courier.ttc']:
  if Path(p).exists():c.font=bpy.data.fonts.load(p);break
 o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=rot;c.materials.append(material);return o
# Regulation pitch, metres; deliberately clinical surface.
cube('105 x 68 m / regulation pitch',(0,0,-.12),(105,68,.24),ivory,.04)
cube('Infinite white room',(0,0,-.28),(700,700,.2),ivory)
line('Touchlines',[(-52.5,-34,.006),(52.5,-34,.006),(52.5,34,.006),(-52.5,34,.006)],.024,ink,True)
line('Halfway line',[(0,-34,.008),(0,34,.008)],.022)
for sign in [-1,1]:
 x=sign*52.5
 line('Penalty area',[(x,-20.16,.008),(x-sign*16.5,-20.16,.008),(x-sign*16.5,20.16,.008),(x,20.16,.008)],.022)
 line('Goal area',[(x,-9.16,.008),(x-sign*5.5,-9.16,.008),(x-sign*5.5,9.16,.008),(x,9.16,.008)],.022)
 line('Goal frame',[(x,-3.66,.025),(x,-3.66,2.44),(x,3.66,2.44),(x,3.66,.025)],.056,chrome)
 for y in [-3.66,3.66]:line('Goal support',[(x,y,2.44),(x+sign*2,y,2.15),(x+sign*2,y,.03),(x,y,.03)],.034,chrome)
 for yy in np.linspace(-3.66,3.66,38):line('Net vertical',[(x,yy,2.44),(x+sign*2,yy,2.15),(x+sign*2,yy,.03)],.006,netmat)
 for zz in np.linspace(.08,2.15,13):line('Net horizontal',[(x+sign*2,-3.66,zz),(x+sign*2,3.66,zz)],.006,netmat)
 for y in [-3.66,3.66]:
  for zz in np.linspace(.1,2.15,12):line('Net side',[(x,y,zz),(x+sign*2,y,zz)],.006,netmat)
 line('Penalty mark',[(x-sign*11-.10,0,.009),(x-sign*11+.10,0,.009)],.035)
 angles=np.linspace(-.925,.925,60) if sign==1 else np.linspace(math.pi-.925,math.pi+.925,60)
 line('Penalty arc',[(x-sign*11-sign*9.15*math.cos(t if sign==1 else t-math.pi),9.15*math.sin(t),.01) for t in angles],.022)
line('Centre circle',[(9.15*math.cos(a),9.15*math.sin(a),.009) for a in np.linspace(0,math.tau,160)],.022,ink,True)
# Tiny coordinate ticks, never a UI box.
for x in np.arange(30,54,2):
 line('Metric tick',[(x,-22,.009),(x,-22.20,.009)],.012)
 label('Metric label',str(int(x)),(x,-22.6,.011),.19,ink,align='CENTER')
label('Field title','A T L A S   /   M O T I O N',(29,-16,.012),1.1,ink)
label('Field subtitle','33     J. QUINONES       /       MONOCULAR STUDY',(29,-17.2,.012),.3,ink)
# Licensed MHR v1.0.1; original 127-joint rig and skin weights.
def state_matrix(s):
 q=Quaternion((float(s[6]),float(s[3]),float(s[4]),float(s[5])));M=(A@q.to_matrix()@A.transposed()).to_4x4()
 for i in range(3):
  for j in range(3):M[i][j]*=float(s[7])
 M.translation=A@(Vector(s[:3])/100);return M
arm=bpy.data.armatures.new('MHR127 / licensed anatomical rig');rig=bpy.data.objects.new('Quinones / MHR127 rig',arm);bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for i,s in enumerate(rest):
 M=state_matrix(s);b=arm.edit_bones.new(f'MHR_{i:03d}');b.head=M.translation;b.tail=b.head+M.to_3x3().col[1].normalized()*.065;b.align_roll(M.to_3x3().col[2]);b.use_connect=False
 if 0<=int(parents[i])<i:b.parent=arm.edit_bones[f'MHR_{int(parents[i]):03d}']
bpy.ops.object.mode_set(mode='OBJECT');rig.select_set(False);rig.show_in_front=True
coords=(np.asarray(A)@D['rest_vertices'].T).T/100
mesh=bpy.data.meshes.new('MHR original anatomical surface');mesh.from_pydata(coords.tolist(),[],D['mesh.faces'].tolist());mesh.update();body=bpy.data.objects.new('Quinones / neutral humanoid surface',mesh);bpy.context.collection.objects.link(body)
# Continuous rest-coordinate shader boundaries avoid material-edge tessellation.
a=mesh.attributes.new('rest_position','FLOAT_VECTOR','POINT');a.data.foreach_set('vector',coords.astype(np.float32).ravel())
uniform=mat('Atlas / continuous technical uniform',(.6,.6,.6),.67)
nt=uniform.node_tree;N=nt.nodes;L=nt.links;p=N.get('Principled BSDF');p.inputs['Specular IOR Level'].default_value=.23
attr=N.new('ShaderNodeAttribute');attr.attribute_name='rest_position';xyz=N.new('ShaderNodeSeparateXYZ');L.new(attr.outputs['Vector'],xyz.inputs[0])
def mathnode(op,a,b=None):
 n=N.new('ShaderNodeMath');n.operation=op
 for ix,v in enumerate([a,b]):
  if v is None:continue
  if isinstance(v,(int,float)):n.inputs[ix].default_value=v
  else:L.new(v,n.inputs[ix])
 return n.outputs[0]
x,y,z=[xyz.outputs[k] for k in ['X','Y','Z']]
shirt=mathnode('MULTIPLY',mathnode('MULTIPLY',mathnode('GREATER_THAN',z,1.0),mathnode('LESS_THAN',z,1.46)),mathnode('LESS_THAN',mathnode('ABSOLUTE',x),.38))
shorts=mathnode('MULTIPLY',mathnode('GREATER_THAN',z,.65),mathnode('LESS_THAN',z,1.01))
socks=mathnode('LESS_THAN',z,.38)
hair=mathnode('GREATER_THAN',z,99)
carbon=mathnode('MAXIMUM',mathnode('MAXIMUM',shorts,socks),hair)
blackall=mathnode('MAXIMUM',carbon,shirt)
redhalf=mathnode('MULTIPLY',shirt,mathnode('LESS_THAN',x,0.))
mix=N.new('ShaderNodeMixRGB');mix.blend_type='MIX';mix.inputs[1].default_value=(.63,.66,.67,1);mix.inputs[2].default_value=(.004,.005,.006,1);L.new(blackall,mix.inputs[0])
mix2=N.new('ShaderNodeMixRGB');mix2.inputs[2].default_value=(.42,.002,.008,1);L.new(mix.outputs[0],mix2.inputs[1]);L.new(redhalf,mix2.inputs[0]);L.new(mix2.outputs[0],p.inputs['Base Color'])
mesh.materials.append(uniform)
for poly in mesh.polygons:poly.use_smooth=True
weights=[[] for _ in range(len(coords))]
for vi,si,w in zip(D['linear_blend_skinning.vert_indices_flattened'],D['linear_blend_skinning.skin_indices_flattened'],D['linear_blend_skinning.skin_weights_flattened']):weights[int(vi)].append((int(si),float(w)))
groups=[body.vertex_groups.new(name=f'MHR_{i:03d}') for i in range(127)]
for vi,items in enumerate(weights):
 for j,w in items:groups[j].add([vi],w,'REPLACE')
mod=body.modifiers.new('Original MHR linear skinning','ARMATURE');mod.object=rig;mod.use_deform_preserve_volume=False
sub=body.modifiers.new('Anatomical surface refinement','SUBSURF');sub.levels=1;sub.render_levels=1
# Typography is skinned to original torso weights; no borrowed identity likeness.
tree=kdtree.KDTree(len(coords))
for i,v in enumerate(coords):tree.insert(v,i)
tree.balance()
for text,loc,size in [('33',(.075,-.164,1.245),.10),('A',(-.085,-.176,1.326),.06)]:
 ob=label('Uniform / '+text,text,loc,size,white,(math.pi/2,0,0),'CENTER');bpy.context.view_layer.objects.active=ob;ob.select_set(True);bpy.ops.object.convert(target='MESH');ob=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);ob.select_set(False)
 for j in range(127):ob.vertex_groups.new(name=f'MHR_{j:03d}')
 for v in ob.data.vertices:
  _,idx,_=tree.find(v.co)
  for j,w in weights[idx]:ob.vertex_groups[j].add([v.index],w,'REPLACE')
 md=ob.modifiers.new('Follow licensed torso skin','ARMATURE');md.object=rig
 ob['atlas_decal']=True
# Skeleton markers are real positions, driven by pose during the motion bake.
major=[1,2,3,4,18,19,20,34,35,36,37,39,40,41,75,76,77,110,113]
for i in major:
 bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,radius=.018,location=state_matrix(rest[i]).translation);o=bpy.context.object;o.name=f'Joint / {i:03d}';o.data.materials.append(vector)
 for p in o.data.polygons:p.use_smooth=True
# Rig/model remain centred for GLB; all staging uses a common world offset later.
rig['rig_license']='Momentum Human Rig v1.0.1 / Apache-2.0';rig['motion_provenance']='Awaiting extracted SAM 3D Body animation';body['identity']='Generic licensed human, not a scanned likeness'
# Football: ceramic green, great-circle inlays, three orbit-view readable seams.
bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=.11,location=(.6,-.2,.11));ball=bpy.context.object;ball.name='Ball / estimated monocular track';ball.data.materials.append(acid)
for p in ball.data.polygons:p.use_smooth=True
for axis in range(3):
 pts=[]
 for aa in np.linspace(0,math.tau,80):
  p=[.111*math.cos(aa),.111*math.sin(aa),0];p[axis],p[2]=p[2],p[axis];pts.append(p)
 o=line('Ball inlay',pts,.0016,black,True);o.parent=ball;o.location=(0,0,0)
# Key light fills the clinical room; long penumbras define anatomy.
world=bpy.data.worlds.new('White cyclorama');scene.world=world;world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.86,.88,.87,1);world.node_tree.nodes['Background'].inputs[1].default_value=.25
for name,loc,power,size in [('Key softbox',(34,-11,13),1700,8),('Rim softbox',(43,5,10),2200,7),('Fill softbox',(29,4,7),1100,10)]:
 bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.name=name;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.rotation_euler=(Vector((38,0,1))-o.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(32,-7,4));cam=bpy.context.object;cam.name='Drone / cinematic orbit';cam.data.lens=48;cam.data.sensor_width=36;cam.data.clip_end=1500;cam.data.clip_start=.01;cam.rotation_euler=(Vector((38,0,1))-cam.location).to_track_quat('-Z','Y').to_euler();scene.camera=cam
# A temporary clean world placement. The real motion bake replaces these values.
rig.location=(38,0,.015);body.location=rig.location
for o in bpy.data.objects:
 if o.name.startswith('Joint /') or o.get('atlas_decal'):o.location+=rig.location
ball.location+=rig.location
scene.render.engine='CYCLES';scene.cycles.samples=48;scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.025;scene.cycles.use_denoising=True;scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=3;scene.cycles.glossy_bounces=3;scene.render.resolution_x=1920;scene.render.resolution_y=1080;scene.render.resolution_percentage=100;scene.render.fps=24;scene.frame_start=1;scene.frame_end=288;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.view_settings.view_transform='AgX';scene.render.use_persistent_data=True
scene['dataset']='Julian Quinones / Atlas vs Pachuca / 2022-05-26';scene['calibration_note']='Pitch is regulation geometry; monocular body depth and stage placement are inferred, not measured';scene['stage_version']='atlas-clinical-v1'
# Store drawn typography as geometry; do not redistribute a system font binary.
for ob in list(bpy.data.objects):
 if ob.type=='FONT':
  bpy.ops.object.select_all(action='DESELECT');ob.select_set(True);bpy.context.view_layer.objects.active=ob;bpy.ops.object.convert(target='MESH');ob.select_set(False)
for font in list(bpy.data.fonts):
 if font.users==0 and font.name!='Bfont':bpy.data.fonts.remove(font)
bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'assets'/'atlas-stage.blend'),compress=True)
print('STAGE_READY',len(bpy.data.objects),'objects',len(coords),'vertices',len(rest),'bones',flush=True)
