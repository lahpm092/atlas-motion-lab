"""Preserve estimated MHR states for exact skinning and viewer output.
Fixed identity shape and scale reduce estimator shape drift; pose remains raw.
Unseen frames remain absent. Clip-time != real-play elapsed time.
"""
import os,json,sys
from pathlib import Path
import numpy as np,torch
R=Path(os.environ.get('ATLAS_PROCESSING',str(Path(__file__).resolve().parent)));torch.set_num_threads(3)
mp=Path(os.environ.get('MHR_MODEL',str(Path(__file__).resolve().parents[1]/'models/sam-3d-body/assets/mhr_model.pt')))
files=sorted((R/'raw3d').glob('*.npz'));data=[dict(np.load(f)) for f in files];good=[d for d in data if bool(d['valid'])]
if not good:raise RuntimeError('No valid inferred pose')
shape=np.median(np.array([d['shape_params'] for d in good]),0).astype('float32');params=np.array([d['mhr_model_params'] for d in good]).astype('float32')
print('raw params',params.shape,'shape',shape.shape,flush=True)
params=params.reshape(len(good),204);shape=shape.reshape(45);params[:,136:]=np.median(params[:,136:],0)
m=torch.jit.load(str(mp),map_location='cpu');m.eval()
with torch.no_grad():
 state_parts=[]
 for a in range(0,len(good),16):
  n=min(16,len(good)-a);_,sp=m(torch.tensor(shape).reshape(1,45).expand(n,-1),torch.tensor(params[a:a+n]),torch.zeros(n,72));state_parts.append(sp)
 states=torch.cat(state_parts)
 identity=m.character_torch.blend_shape(torch.tensor(shape).reshape(1,45)).numpy()[0]
state=states.numpy();np.save(R/'quinones_identity_rest.npy',identity)
state_path=R/'motion_mhr.npz';np.savez_compressed(state_path,state=state,shape=shape,params=params,source_frames=np.array([int(d['frame']) for d in good]),source_times=np.array([float(d['source_time']) for d in good]),cam=np.array([d['pred_cam_t'] for d in good]),focal=np.array([d['focal_length'] for d in good]),joints70=np.array([d['pred_keypoints_3d'][:70] for d in good]),keypoints2d=np.array([d['pred_keypoints_2d'][:70] for d in good]))
names=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel','neck']
indices=[0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14,15,16,17,18,19,20,69]
bones=[['neck','nose'],['neck','left_shoulder'],['neck','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle'],['left_ankle','left_big_toe'],['right_ankle','right_big_toe']]
frames=[]
for d in data:
 f=int(d['frame']);valid=bool(d['valid']);row={'frame':f,'index':f,'sourceTime':float(d['source_time']),'time':float(d['source_time'])-80.8,'valid':valid,'joints2d':{},'joints3d':{},'ball':None,'mask':f'masks/{f:05}_mask.png'}
 if valid:
  k2=d['pred_keypoints_2d'];k3=d['pred_keypoints_3d'];hip=(k3[9]+k3[10])/2
  row['bbox']=d['mask_box'].tolist();row['joints2d']={n:{'x':round(float(k2[j,0]),3),'y':round(float(k2[j,1]),3),'confidence':None} for n,j in zip(names,indices)}
  row['joints3d']={n:[round(float(k3[j,0]-hip[0]),6),round(float(k3[j,2]-hip[2]),6),round(float(-k3[j,1]+hip[1]),6)] for n,j in zip(names,indices)}
  row['jointsCamera3d']={n:[round(float(v),6) for v in k3[j]] for n,j in zip(names,indices)}
  a,b,c=k3[9],k3[11],k3[13];v=a-b;w=c-b;angle=float(np.degrees(np.arccos(np.clip(np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w)),-1,1))))
  row['leftKneeAngle']=round(angle,3)
 frames.append(row)
meta={'source':'https://www.youtube.com/watch?v=y4dl3bDOxYI','player':'Julián Quiñones','jersey':33,'match':'Atlas 2–0 Pachuca, Final ida Clausura 2022','sourceStart':80.8,'sourceEnd':92.2,'sourceFps':29.97002997002997,'fps':29.97002997002997,'dimensions':[1280,720],'pose':'SAM 3D Body monocular MHR-70','segmentation':'SAM 3.1, independent frame inference','analysisTemporalGrid':'Source integer frame indices; every source frame, no synthetic detections','joints3dSpace':'Body-centered camera axes rotated to Blender: x screen-right, y camera depth, z up. Meters predicted by model, not calibrated on field.','rigStateSpace':'MHR native centimeters Y-up; state=[x,y,z,qx,qy,qz,qw,scale]','fixedIdentityShape':True,'fixedSkeletonScale':True,'ball':None,'manualCorrections':[],'limitations':['Broadcast replay is slow motion: no real-world speed inference.','Depth, contacts and occluded joints are monocular estimates.','No calibrated field homography or measured ball launch elevation.','Confidence fields null because this output does not provide calibrated joint confidence.']}
for name in ['track-raw.json','track.json']:(R/name).write_text(json.dumps({'metadata':meta,'bones':bones,'jointNames':names,'frames':frames},ensure_ascii=False,separators=(',',':')))
(R/'motion.json').write_text(json.dumps({'fps':meta['fps'],'metadata':meta,'frames':[{**f,'joints':f['joints3d']} for f in frames if f['valid']]},ensure_ascii=False,separators=(',',':')))
print('EXPORTED',len(frames),'valid',len(good),state.shape,flush=True)
