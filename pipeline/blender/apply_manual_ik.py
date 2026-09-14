import numpy as np,json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--targets',required=True);p.add_argument('--output',required=True);p.add_argument('--mapping');a=p.parse_args();Z=dict(np.load(a.input));targets=json.load(open(a.targets));states=Z['state'].astype(np.float64).copy();original=states.copy();frames=Z.get('frame_indices',Z.get('source_frames'));source_times=Z.get('source_times',float(Z.get('source_start',80.8))+(frames-frames[0])/float(Z.get('fps',29.97003)));cam=Z.get('cam',Z.get('pred_cam_t'));focal=Z.get('focal',Z.get('focal_length'));uv=Z.get('keypoints2d',Z.get('uv70'));parents=np.load(Path(__file__).resolve().parent/'assets/rig.npz')['skeleton.joint_parents']
if cam is None or focal is None or uv is None:raise RuntimeError('Need native cam, focal and original uv70 to preserve projection/depth')
focal=np.asarray(focal)
if focal.size==1:focal=np.repeat(focal.item(),len(states))
if frames.max()<2000:
 # Exact source-frame identity is taken from the stored source timestamps.
 source_frames=np.rint(source_times*30000/1001).astype(int)
else:source_frames=frames.astype(int)
def descend(j):
 ans={j}
 for k in range(j+1,127):
  if int(parents[k]) in ans:ans.add(k)
 return ans
def qbetween(a,b):
 aa=a/np.linalg.norm(a);bb=b/np.linalg.norm(b);q=np.r_[np.cross(aa,bb),1+np.dot(aa,bb)]
 if np.linalg.norm(q)<1e-8:raise RuntimeError('Antiparallel IK rotation')
 return q/np.linalg.norm(q)
def qmul(a,b):return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-np.dot(a[:3],b[:3])]
def qmatrix(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def solve(st,s,e,t,target):
 old=st.copy();S,E,T=old[[s,e,t],:3];L1=np.linalg.norm(E-S);L2=np.linalg.norm(T-E);v=target-S;d=np.linalg.norm(v);reach=np.clip(d,abs(L1-L2)+1e-5,L1+L2-1e-5);target=S+v/d*reach;axis=(target-S)/reach;bend=(E-S)-axis*np.dot(E-S,axis)
 if np.linalg.norm(bend)<1e-6:bend=np.cross(axis,[0,0,1])
 bend/=np.linalg.norm(bend);c=(L1*L1-L2*L2+reach*reach)/(2*reach);EE=S+axis*c+bend*np.sqrt(max(L1*L1-c*c,0));qa=qbetween(E-S,EE-S);qb=qbetween(T-E,target-EE)
 for ids,q,src,dst in [(descend(s)-descend(e),qa,S,S),(descend(e),qb,E,EE)]:
  M=qmatrix(q)
  for j in ids:st[j,:3]=M@(old[j,:3]-src)+dst;st[j,3:7]=qmul(q,old[j,3:7])
 return {'length_error_cm':max(abs(np.linalg.norm(st[e,:3]-st[s,:3])-L1),abs(np.linalg.norm(st[t,:3]-st[e,:3])-L2)),'target_clamp_cm':float(abs(d-reach))}
projector=None
if a.mapping:
 from mhr_project import MHRProjector
 rigdata=dict(np.load(Path(__file__).resolve().parent/'assets/rig.npz'));projector=MHRProjector(rigdata,Z.get('identity_rest',rigdata['rest_vertices']),a.mapping)
rows=targets.get('frames',targets.get('samples',targets.get('corrections',[])));audit=[]
# Target schema: frames[{source_frame, source_time, right_wrist:{x,y,visible}, right_ankle:{x,y,visible}}].
for row in rows:
 f=int(row.get('source_frame',row.get('sourceFrame',0)));matches=np.where(source_frames==f)[0]
 if len(matches)==0:continue
 i=int(matches[0])
 for name,k,chain in [('right_wrist',41,(39,40,41)),('right_ankle',14,(18,19,20)),('left_ankle',13,(2,3,4))]:
  point=row.get(name,row.get('joints',row.get('targets',{})).get(name))
  if point is None or point.get('usable_for_ik') is False:continue
  if isinstance(point,(list,tuple)):point={'x':point[0],'y':point[1]}
  if point.get('x') is None or point.get('y') is None or point.get('visible') is False:continue
  raw_inference=uv[i,k];raw=projector(states[i],cam[i],focal[i])[k] if projector is not None else raw_inference;desired=np.array([point['x'],point['y']],float);delta=desired-raw;native=states[i,chain[-1],:3].copy();depth=cam[i,2]-native[2]/100.;target=native+np.array([delta[0],-delta[1],0])*depth/focal[i]*100.;result=solve(states[i],*chain,target);
  if projector is not None:
   for refine in range(3):
    residual=desired-projector(states[i],cam[i],focal[i])[k]
    if np.linalg.norm(residual)<.08:break
    target2=states[i,chain[-1],:3]+np.array([residual[0],-residual[1],0])*depth/focal[i]*100.;solve(states[i],*chain,target2)
  actual_delta=(states[i,chain[-1],:3]-native)*[1,-1,-1]/100.;reproj=raw+focal[i]*actual_delta[:2]/depth
  audit.append({'source_frame':f,'source_time':float(source_times[i]),'joint':name,'original_pixel':raw.tolist(),'raw_inference_pixel':raw_inference.tolist(),'manual_pixel':desired.tolist(),'applied_delta_px':delta.tolist(),'endpoint_depth_change_cm':float(states[i,chain[-1],2]-native[2]),'estimated_reprojection_residual_px':float(np.linalg.norm(reproj-desired)),**result})
# The manually measured window is preserved at every annotated frame. Add only a
# three-frame transition outside each boundary to avoid a pop; these are inferred.
changed=np.where(np.any(abs(states-original)>.000001,axis=(1,2)))[0]
if len(changed):
 for edge,direction in [(changed.min(),-1),(changed.max(),1)]:
  for n in range(1,4):
   j=edge+direction*n
   if j<0 or j>=len(states):continue
   w=(4-n)/4.;delta=states[edge,:,:3]-original[edge,:,:3];states[j,:,:3]=original[j,:,:3]+delta*w
   # Re-solve endpoints for exact bone lengths instead of linearly moving joints.
   temp=original[j].copy()
   for chain in [(39,40,41),(18,19,20),(2,3,4)]:solve(temp,*chain,original[j,chain[-1],:3]+delta[chain[-1]]*w)
   states[j]=temp
Z['state_before_manual_ik']=original.astype(np.float32);Z['state']=states.astype(np.float32);Z['manual_ik_applied']=np.array(True);Z['manual_ik_note']=np.array('Manual2D targets lifted at original endpoint camera depth;2segment IK preserves lengths.3frame inferred transition at outer edges. Depth not independently observed.');np.savez_compressed(a.output,**Z)
report={'method':'Frame-specific manually annotated image targets; native camera-depth-preserving lift and exact two-segment IK','input':str(a.input),'targets':str(a.targets),'output':str(a.output),'rows_applied':len(audit),'changed_frames':int(np.any(abs(states-original)>1e-6,axis=(1,2)).sum()),'max_bone_length_error_cm':max((r['length_error_cm'] for r in audit),default=0),'max_reprojection_residual_px':max((r['estimated_reprojection_residual_px'] for r in audit),default=0),'max_depth_change_cm':max((abs(r['endpoint_depth_change_cm']) for r in audit),default=0),'records':audit,'limits':['Pixel targets are visually estimated, with stated annotation tolerances.','Depth stays at the model estimate; no new3D measurement.','Residual checks endpoint displacement relative to original landmark projection; anatomical landmark offsets may differ from rig pivots.','The three-frame edge transition is inferred, not separately hand-annotated.']}
Path(a.output).with_suffix('.audit.json').write_text(json.dumps(report,indent=2));print('MANUAL_IK',len(audit),'joints',report['changed_frames'],'frames',report['max_reprojection_residual_px'],'px',report['max_bone_length_error_cm'],'cm')
