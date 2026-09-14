import bpy,numpy as np,math,json,argparse,sys,hashlib
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--motion',required=True);ap.add_argument('--rig-only',action='store_true');ap.add_argument('--glb-name',default='humanoid.glb');ap.add_argument('--allow-sparse-probe',action='store_true');args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);R=Path(args.root);Z=dict(np.load(args.motion));D=dict(np.load(R/'assets/rig.npz'))
available_segments=[]
if 'source_times' in Z:
 all_times=np.asarray(Z['source_times']);cuts=np.r_[0,np.where(np.diff(all_times)>3/float(Z.get('fps',29.97003))+.00001)[0]+1,len(all_times)]
 available_segments=[{'source_start':float(all_times[a]),'source_end':float(all_times[b-1]),'poses':int(b-a)} for a,b in zip(cuts[:-1],cuts[1:])]
 if not args.allow_sparse_probe:
  ci=int(np.argmin(abs(all_times-88.822067)));a,b=next((int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:]) if a<=ci<b)
  if b-a<2:raise RuntimeError('No continuous observed pose interval around strike')
  n0=len(all_times)
  for key,value in list(Z.items()):
   if isinstance(value,np.ndarray) and value.ndim>0 and len(value)==n0:Z[key]=value[a:b]
bpy.ops.wm.open_mainfile(filepath=str(R/'assets/atlas-stage.blend'))
s=bpy.context.scene;rig=bpy.data.objects['Quinones / MHR127 rig'];body=bpy.data.objects['Quinones / neutral humanoid surface'];cam=s.camera
A=Matrix(((1,0,0),(0,0,-1),(0,1,0)));NATIVE=np.array(A);parents=D['skeleton.joint_parents'];states=Z['state'];frames=Z.get('frame_indices',Z.get('source_frames',np.arange(len(states))));fps=float(Z.get('fps',29.97003));clip_source_start=float(Z.get('source_start',80.8));absolute_times=np.array(Z.get('source_times',clip_source_start+(frames-frames[0])/fps));source_start=float(absolute_times[0]);times=absolute_times-source_start;duration=float(times[-1])
if states.ndim==4:states=states[:,0]
# Shape vertices retain the original licensed weights and inverse bind skeleton.
if 'identity_rest' in Z:
 coords=(NATIVE@Z['identity_rest'].T).T/100
 body.data.vertices.foreach_set('co',coords.astype(np.float32).ravel());body.data.update()
else:coords=np.array([v.co[:] for v in body.data.vertices])
# Root relative reconstruction: remove horizontal camera-space pelvis translation.
# Ground alignment is inferred independently, never a measured pitch homography.
P=(NATIVE@states[:,:,:3].transpose(0,2,1)).transpose(0,2,1)/100
shift=np.zeros((len(P),3));shift[:,:2]=-P[:,1,:2]
footmin=P[:,[4,7,8,20,23,24],2].min(axis=1);shift[:,2]=.018-footmin
P+=shift[:,None,:]
major=[1,2,3,4,18,19,20,34,35,36,37,39,40,41,75,76,77,110,113]
# One marker for each significant anatomical centre, with 127 positions in JSON.
for o in bpy.data.objects:
 if o.name.startswith('Joint /'):o.animation_data_clear()

def bag(ob,name):
 ob.animation_data_clear();ob.animation_data_create();act=bpy.data.actions.new(name);slot=act.slots.new(id_type='OBJECT',name=ob.name);ob.animation_data.action=act;ob.animation_data.action_slot=slot;layer=act.layers.new('Motion');strip=layer.strips.new(type='KEYFRAME');return strip.channelbags.new(slot)
def curve(b,path,ix,v):
 a=np.asarray(v,np.float32);const=np.ptp(a)<1e-7;f=b.fcurves.new(data_path=path,index=ix);tt=np.array([1],np.float32) if const else np.arange(1,len(a)+1,dtype=np.float32);vv=a[:1] if const else a;f.keyframe_points.add(len(tt));f.keyframe_points.foreach_set('co',np.column_stack((tt,vv)).ravel());f.keyframe_points.foreach_set('interpolation',np.ones(len(tt),np.int32));f.update()
def curveall(ob,name,props):
 b=bag(ob,name)
 for path,ar in props:
  ar=np.asarray(ar)
  if ar.ndim==1:curve(b,path,0,ar)
  else:
   for j in range(ar.shape[1]):curve(b,path,j,ar[:,j])
def sample(t):
 ii=min(np.searchsorted(times,t,side='right')-1,len(times)-2);ii=max(0,ii);w=np.clip((t-times[ii])/(times[ii+1]-times[ii]),0,1);data=states[ii]*(1-w)+states[ii+1]*w
 for j in range(127):
  qa=Quaternion((states[ii,j,6],*states[ii,j,3:6]));qb=Quaternion((states[ii+1,j,6],*states[ii+1,j,3:6]));q=qa.slerp(qb,float(w));data[j,3:7]=[q.x,q.y,q.z,q.w]
 sh=shift[ii]*(1-w)+shift[ii+1]*w;pos=(NATIVE@data[:,:3].T).T/100+sh
 return data,sh,pos

def bake(ts,world=(0,0,0),name='SAM3D / source replay time'):
 vals=np.zeros((127,len(ts),10),np.float32);positions=[]
 for fi,t in enumerate(ts):
  data,sh,pos=sample(t);target=[]
  for z in data:
   q=Quaternion((float(z[6]),float(z[3]),float(z[4]),float(z[5])));M=(A@q.to_matrix()@A.transposed()).to_4x4()
   for a in range(3):
    for b in range(3):M[a][b]*=float(z[7])
   M.translation=A@(Vector(z[:3])/100)+Vector(sh);target.append(M)
  for j in range(127):
   bone=rig.data.bones[f'MHR_{j:03d}'];pa=int(parents[j]);kw={}
   if 0<=pa<j:kw={'parent_matrix':target[pa],'parent_matrix_local':rig.data.bones[f'MHR_{pa:03d}'].matrix_local}
   basis=bone.convert_local_to_pose(target[j],bone.matrix_local,invert=True,**kw);loc,q,sc=basis.decompose()
   if fi and np.dot(vals[j,fi-1,3:7],q)<0:q.negate()
   vals[j,fi,:3]=loc;vals[j,fi,3:7]=q;vals[j,fi,7:]=sc
  positions.append(pos+world)
 b=bag(rig,name)
 for j in range(127):
  rig.pose.bones[f'MHR_{j:03d}'].rotation_mode='QUATERNION';path=f'pose.bones["MHR_{j:03d}"]'
  for prop,off,n in [('location',0,3),('rotation_quaternion',3,4),('scale',7,3)]:
   for k in range(n):curve(b,path+'.'+prop,k,vals[j,:,off+k])
 rig.location=world;body.location=world
 for o in bpy.data.objects:
  if o.get('atlas_decal'):o.location=world
 return np.array(positions)
# Original replay-rate rig export, with vertex colours for glTF interoperability.
exportfps=fps;t_real=np.arange(round(duration*exportfps)+1)/exportfps;pos_real=bake(t_real);s.render.fps=30;s.render.fps_base=30/exportfps;s.frame_start=1;s.frame_end=len(t_real);s.frame_set(1)
semantics={'pelvis':1,'left_hip':2,'left_knee':3,'left_ankle':4,'left_toe':8,'right_hip':18,'right_knee':19,'right_ankle':20,'right_toe':24,'spine':35,'thorax':37,'right_shoulder':39,'right_elbow':40,'right_wrist':41,'left_shoulder':75,'left_elbow':76,'left_wrist':77,'neck':110,'head':113}
named_report={'schema':'atlas-rig-joints-v1','source_start':source_start,'timebase':'broadcast replay seconds, not physical event time','axes':'glTF Y up; coordinates=(Blender x, Blender z, -Blender y)','units':'metres of the generic inferred MHR body','manual_ik_applied':bool(Z.get('manual_ik_applied',False)),'sample_fps':exportfps,'frames':[{'time':round(float(t),6),'sourceTime':round(source_start+float(t),6),'joints3dNamed':{name:[round(float(p[j,0]),6),round(float(p[j,2]),6),round(float(-p[j,1]),6)] for name,j in semantics.items()}} for t,p in zip(t_real,pos_real)]}
(R/'deliverable'/('rig-joints-before.json' if 'before' in args.glb_name else 'rig-joints.json')).write_text(json.dumps(named_report,separators=(',',':')))
mat=body.data.materials[0];nt=mat.node_tree;shader=nt.nodes.get('Principled BSDF');savedlinks=[(l.from_socket,l.to_socket) for l in list(nt.links) if l.to_socket==shader.inputs['Base Color']]
for a,b in savedlinks:
 for l in list(nt.links):
  if l.from_socket==a and l.to_socket==b:nt.links.remove(l)
colors=[]
for x,y,z in coords:
 shirt=1.<z<1.46 and abs(x)<.38;dark=.65<z<1.01 or z<.38 or shirt;c=(.42,.002,.008,1) if shirt and x<0 else ((.004,.005,.006,1) if dark else (.63,.66,.67,1));colors.append(c)
ca=body.data.color_attributes.new(name='AtlasUniform',type='FLOAT_COLOR',domain='POINT');ca.data.foreach_set('color',np.array(colors,np.float32).ravel());vc=nt.nodes.new('ShaderNodeVertexColor');vc.layer_name='AtlasUniform';nt.links.new(vc.outputs['Color'],shader.inputs['Base Color'])
sub=body.modifiers.get('Anatomical surface refinement');sub.show_viewport=False;sub.show_render=False
bpy.ops.object.select_all(action='DESELECT')
for o in [rig,body]+[o for o in bpy.data.objects if o.get('atlas_decal')]:o.select_set(True)
bpy.context.view_layer.objects.active=rig
bpy.ops.export_scene.gltf(filepath=str(R/'deliverable'/args.glb_name),export_format='GLB',use_selection=True,export_animations=True,export_frame_range=True,export_force_sampling=True,export_skins=True,export_yup=True,export_materials='EXPORT',export_apply=False)
if args.rig_only:
 print('RIG_ONLY_READY',args.glb_name,flush=True);sys.exit(0)
# Restore precise Cycles shader and physical surface refinement.
nt.nodes.remove(vc)
for a,b in savedlinks:nt.links.new(a,b)
sub.show_viewport=True;sub.show_render=True
# Export static regulation field independently; no player/material overlay.
bpy.ops.object.select_all(action='DESELECT')
for o in bpy.data.objects:
 if o.type in ['MESH','CURVE','FONT'] and o!=body and o.name!='Infinite white room' and not o.name.startswith(('Joint /','Ball','Uniform /')):o.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(R/'deliverable/pitch.glb'),export_format='GLB',use_selection=True,export_animations=False,export_yup=True,export_apply=True)
# Interactive analytical data uses Blender axes; app can map (x,z,-y) to Three.
report={'schema':'atlas-motion-1','source_start':source_start,'source_end':source_start+duration,'source_fps':fps,'timebase':'broadcast replay seconds; not physical event time','duration':duration,'rig':'MHR v1.0.1 / 127 bones / Apache-2.0','root_mode':'pelvis-centred; floor contact inferred per frame','axes':'Blender Z up, X horizontal image axis, Y inferred depth','joints':[f'MHR_{i:03d}' for i in range(127)],'parents':parents.tolist(),'sample_fps':exportfps,'frames':[{'t':round(float(t),5),'joints':np.round(p,5).tolist()} for t,p in zip(t_real,pos_real)]}
(R/'deliverable/rig-motion.json').write_text(json.dumps(report,separators=(',',':')))
# Cinematic time warp: reproduce all source motion, then slow the striking window.
playfps=24;movie_duration=duration+3.;ntotal=round(movie_duration*playfps);tt=np.arange(ntotal)/playfps;strike=min(88.822067-source_start,duration-.5);window=.34;lo=strike-window/2;hi=strike+window/2
source_t=[]
for t in tt:
 if t<lo:st=t
 elif t<lo+window+3:st=lo+(t-lo)*window/(window+3)
 else:st=t-3
 source_t.append(min(st,duration))
source_t=np.array(source_t);center=np.array([38.,0.,0.]);pos=bake(source_t,tuple(center),'SAM3D / cinematic slow motion at strike');s.render.fps=playfps;s.render.fps_base=1;s.frame_end=ntotal
# A separate overlay view layer keeps projected 3D skeletons readable through the surface.
main=bpy.data.collections.new('Clinical stage / Cycles beauty');s.collection.children.link(main);over=bpy.data.collections.new('Analytical vectors / composited overlay');s.collection.children.link(over)
for o in list(bpy.data.objects):
 if o.type in ['CAMERA','LIGHT']:continue
 dest=over if o.name.startswith('Joint /') else main
 for c in list(o.users_collection):c.objects.unlink(o)
 dest.objects.link(o)
vmat=bpy.data.materials.new('Analytical red / emission');vmat.use_nodes=True;n=vmat.node_tree.nodes;n.clear();out=n.new('ShaderNodeOutputMaterial');em=n.new('ShaderNodeEmission');em.inputs[0].default_value=(.53,.003,.012,1);em.inputs[1].default_value=.9;vmat.node_tree.links.new(em.outputs[0],out.inputs[0])
for j in major:
 ob=bpy.data.objects[f'Joint / {j:03d}'];ob.data.materials.clear();ob.data.materials.append(vmat);curveall(ob,'Joint centre / '+str(j),[('location',pos[:,j])])
# Mesh cylinders and cones are animated from the actual derivatives in replay time.
def rod(name,radius,depth=1):
 bpy.ops.mesh.primitive_cylinder_add(vertices=10,radius=radius,depth=depth);o=bpy.context.object;o.name=name;o.data.materials.append(vmat)
 for c in list(o.users_collection):c.objects.unlink(o)
 over.objects.link(o);return o
def animate_segment(ob,ends1,ends2,radius=.007):
 loc=(ends1+ends2)/2;q=[];sc=[]
 for a,b in zip(ends1,ends2):
  v=Vector(b-a);qq=v.to_track_quat('Z','Y')
  if q and np.dot(q[-1],qq)<0:qq.negate()
  q.append(qq[:]);sc.append((1,1,max(v.length,.001)))
 ob.rotation_mode='QUATERNION';curveall(ob,ob.name,[('location',loc),('rotation_quaternion',q),('scale',sc)])
links=[(1,2),(2,3),(3,4),(1,18),(18,19),(19,20),(1,34),(34,35),(35,36),(36,37),(37,39),(39,40),(40,41),(37,75),(75,76),(76,77),(37,110),(110,113)]
for a,b in links:animate_segment(rod(f'Anatomical bone / {a:03d}-{b:03d}',.006),pos[:,a],pos[:,b])
# Relative-body velocity excludes any arbitrary stage translation.
vel=np.gradient(P,times,axis=0)
for j in major:
 v=np.stack([np.interp(source_t,times,vel[:,j,k]) for k in range(3)],1);norm=np.linalg.norm(v,axis=1);vv=v/np.maximum(norm[:,None],1e-6)*np.clip(norm*.11,.025,.6)[:,None];end=pos[:,j]+vv
 animate_segment(rod(f'Joint velocity / {j:03d}',.005),pos[:,j],end)
 bpy.ops.mesh.primitive_cone_add(vertices=12,radius1=.020,radius2=0,depth=.055);ob=bpy.context.object;ob.name=f'Velocity tip / {j:03d}';ob.data.materials.append(vmat)
 for c in list(ob.users_collection):c.objects.unlink(ob)
 over.objects.link(ob);q=[Vector(x).to_track_quat('Z','Y')[:] for x in vv];ob.rotation_mode='QUATERNION';curveall(ob,ob.name,[('location',end),('rotation_quaternion',q)])
# Ankle locus is reconstructed body motion, not a claimed ball flight model.
for j in [4,20]:
 cu=bpy.data.curves.new(f'Ankle motion locus / {j}','CURVE');cu.dimensions='3D';cu.bevel_depth=.0035;cu.bevel_resolution=2;sp=cu.splines.new('POLY');loc=P[:,j]+center;sp.points.add(len(loc)-1)
 for q,p in zip(sp.points,loc):q.co=(*p,1)
 ob=bpy.data.objects.new(f'Ankle motion locus / {j}',cu);over.objects.link(ob);cu.materials.append(vmat)
# Ball's ground-plane track is supplied separately from manually annotated pixels.
ball=bpy.data.objects['Ball / estimated monocular track'];ball.animation_data_clear();ball.location=(38.7,0,.11)
if 'ball_local' in Z:
 b=np.array(Z['ball_local']);bp=np.stack([np.interp(source_t,times,b[:,k]) for k in range(3)],1)+center;v=np.array(Z.get('ball_visible',np.ones(len(times))));vv=np.interp(source_t,times,v)>.999;curveall(ball,'Ball / observed screen-plane lift',[('location',bp),('hide_render',~vv)])
 for o in bpy.data.objects:
  if o.parent==ball:curveall(o,o.name+' / observed visibility',[('hide_render',~vv)])
else:
 ball.hide_render=True
 for o in bpy.data.objects:
  if o.parent==ball:o.hide_render=True
# Continuous drone orbit; slow window gives a readable near-180 degree reveal.
caml=[];camq=[];camera_checks=[]
heights=(pos[:,126,2]+np.minimum(pos[:,8,2],pos[:,24,2]))/2
heights=np.convolve(np.pad(heights,(8,8),mode='edge'),np.ones(17)/17,mode='valid')
for i,(t,st) in enumerate(zip(tt,source_t)):
 progress=i/max(ntotal-1,1);angle=math.radians(-133+250*progress);radius=6.10+.20*math.sin(progress*math.pi*2);target=center+np.array([0,0,heights[i]]);offset=np.array([radius*math.cos(angle),radius*math.sin(angle),1.65+.55*math.sin(progress*math.pi)])
 # All anatomical centres plus a conservative surface/boot margin must fit.
 pts=pos[i,1:];padding=np.array([[0,0,.11],[0,0,-.11],[.11,0,0],[-.11,0,0],[0,.11,0],[0,-.11,0]]);pts=(pts[:,None,:]+padding[None,:,:]).reshape(-1,3)
 for trial in range(8):
  cl=target+offset;forward=(target-cl)/np.linalg.norm(target-cl);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward);delta=pts-cl;depth=delta@forward;u=(delta@right)/depth/(36/(2*49));v=(delta@up)/depth/(36/(2*49)*9/16);factor=max(np.max(abs(u))/.86,np.max(abs(v))/.82)
  if factor<=1:break
  offset*=factor*1.01
 camera_checks.append([float(np.max(abs(u))),float(np.max(abs(v)))]);caml.append(cl);q=(Vector(target)-Vector(cl)).to_track_quat('-Z','Y')
 if camq and np.dot(camq[-1],q)<0:q.negate()
 camq.append(q[:])
(R/'deliverable/camera-framing-audit.json').write_text(json.dumps({'frames_checked':len(camera_checks),'maximum_horizontal_normalized':max(q[0] for q in camera_checks),'maximum_vertical_normalized':max(q[1] for q in camera_checks),'clip_boundary':1,'scope':'All anatomical joint centres with11cm surface margin. Full-render contact sheets are reviewed separately.'},indent=2))
cam.rotation_mode='QUATERNION';cam.data.lens=49;curveall(cam,'Drone / 250 degree orbit',[('location',caml),('rotation_quaternion',camq)])
s.render.film_transparent=True
beauty=s.view_layers[0];beauty.name='Beauty';beauty.layer_collection.children[over.name].exclude=True
ov=s.view_layers.new('Vectors');ov.layer_collection.children[main.name].exclude=True
s.use_nodes=True;nt=bpy.data.node_groups.new('Atlas / analytical compositing','CompositorNodeTree');s.compositing_node_group=nt;nt.interface.new_socket(name='Image',in_out='OUTPUT',socket_type='NodeSocketColor');a=nt.nodes.new('CompositorNodeRLayers');a.layer='Beauty';b=nt.nodes.new('CompositorNodeRLayers');b.layer='Vectors';mix=nt.nodes.new('CompositorNodeAlphaOver');mix.inputs['Factor'].default_value=.76;nt.links.new(a.outputs['Image'],mix.inputs['Background']);nt.links.new(b.outputs['Image'],mix.inputs['Foreground']);out=nt.nodes.new('NodeGroupOutput');nt.links.new(mix.outputs[0],out.inputs['Image'])
# Verify the retargeting implementation against imported MHR joint centres.
errors=[]
for fi in sorted(set([0,len(source_t)//2,len(source_t)-1])):
 s.frame_set(fi+1);bpy.context.view_layer.update()
 for j in range(127):
  q=rig.matrix_world@rig.pose.bones[f'MHR_{j:03d}'].matrix.translation;errors.append(float(np.linalg.norm(np.array(q)-pos[fi,j])))
(R/'deliverable/retarget-audit.json').write_text(json.dumps({'frames_checked':3,'joints_checked':len(errors),'max_joint_error_m':max(errors),'mean_joint_error_m':float(np.mean(errors)),'scope':'Implementation equivalence to imported MHR joint centres, not independent real-world 3D accuracy.'},indent=2))
if max(errors)>.002:raise RuntimeError('Retarget joint error exceeds2mm: '+str(max(errors)))
s['animation_provenance']='SAM 3D Body fitted to actual broadcast frames; inferred monocular depth. Relative body joint vectors are derivatives per replay second.';s['ball_note']='Observed image positions only; no calibrated 3D ball flight';s['cinematic_timewarp']='Complete replay motion; 3s extension at strike for drone orbit';rig['motion_provenance']=s['animation_provenance'];s.frame_set(1);bpy.context.preferences.filepaths.save_version=0;bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(R/'deliverable/Atlas_Quinones_Motion.blend'),compress=True)
manifest={'source_valid_segments':available_segments,'animationStart':source_start-clip_source_start,'animationEnd':source_start+duration-clip_source_start,'clipSourceStart':clip_source_start,'humanoid':{'path':'humanoid.glb','duration':(len(t_real)-1)/exportfps,'source_start':source_start,'source_end':source_start+duration,'sourceStartFrame':int(round(source_start*fps)),'sourceEndFrame':int(round((source_start+duration)*fps)),'center':[0,0,.9],'axes':'glTF Y up; Blender Z up converted on export','skin_joints':127,'base_vertices':18439},'pitch':{'path':'pitch.glb','length_m':105,'width_m':68,'goal_width_m':7.32,'goal_height_m':2.44},'cinematic':{'path':'atlas-cinematic.mp4','fps':24,'frames':ntotal,'duration':ntotal/24,'source_time_per_frame':np.round(source_start+source_t,4).tolist(),'slowdown_source_window':[source_start+lo,source_start+hi],'engine':'Blender Cycles ray tracing with composited analytical vectors','body_root':'relative / no calibrated field translation'},'limitations':['Broadcast replay time is not physical time.','Single-camera depth, contacts and occluded joints are inferred.','Stage coordinates and ball depth are not calibrated.'],'license':'MHR-LICENSE.txt'}
(R/'deliverable/blender-manifest.json').write_text(json.dumps(manifest,indent=2));print('ATLAS_MOTION_READY',len(t_real),ntotal,flush=True)
