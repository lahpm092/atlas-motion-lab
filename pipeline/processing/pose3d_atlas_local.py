"""SAM3D Body masks-to-MHR inference. Consumes real SAM3.1 silhouettes.
Raw predictions remain auditable; no kinematic motion is invented.
"""
import os,sys,json,time,contextlib,io,argparse
from pathlib import Path
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1';os.environ['MOMENTUM_ENABLED']='0'
import numpy as np,torch,cv2
R=Path(str(Path(__file__).resolve().parent));sys.path.insert(0,os.environ.get('SAM3D_SOURCE',str(Path(__file__).resolve().parents[1]/'models/sam-3d-body-src')))
from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
import sam_3d_body.sam_3d_body_estimator as em
ap=argparse.ArgumentParser();ap.add_argument('--stream',action='store_true');ap.add_argument('--probe',action='store_true');args=ap.parse_args()
torch.set_num_threads(2)
_orig_load=torch.load
def mapped_load(path,*a,**kw):
 if isinstance(path,(str,Path)) and str(path).endswith('model.ckpt'):kw['mmap']=True
 return _orig_load(path,*a,**kw)
torch.load=mapped_load
model,cfg=load_sam_3d_body(os.environ.get('SAM3D_CHECKPOINT',str(Path(__file__).resolve().parents[1]/'models/sam-3d-body/model.ckpt')),device='mps',mhr_path=os.environ.get('MHR_MODEL',str(Path(__file__).resolve().parents[1]/'models/sam-3d-body/assets/mhr_model.pt')))
orig=em.recursive_to;em.recursive_to=lambda x,d:orig(x,'mps' if d=='cuda' else d);est=SAM3DBodyEstimator(model,cfg)
(R/'raw3d').mkdir(exist_ok=True);(R/'overlays').mkdir(exist_ok=True)
np.savez_compressed(R/'model_mappings.npz',faces=est.faces,keypoint_mapping=model.head_pose.keypoint_mapping.cpu().numpy(),joint_rotation=model.head_pose.joint_rotation.cpu().numpy(),scale_mean=model.head_pose.scale_mean.cpu().numpy(),scale_comps=model.head_pose.scale_comps.cpu().numpy())
cap=cv2.VideoCapture(str(R.parent/'source/atlas-quinones-pachuca-2022-source.mp4'));t0=time.time();last=time.time()
bones=[(69,0),(69,5),(69,6),(5,7),(7,62),(6,8),(8,41),(5,9),(6,10),(9,10),(9,11),(11,13),(10,12),(12,14),(13,15),(14,18)]
while True:
 pending=[x for x in sorted((R/'masks-v3').glob('*.npz')) if not (R/'raw3d'/x.name).exists()]
 if args.probe:pending=[p for p in pending if int(p.stem) in [2655,2662,2665]]
 for mf in pending:
  z=dict(np.load(mf));f=int(z['frame']);out=R/'raw3d'/mf.name
  cap.set(1,f);ok,im=cap.read()
  if not ok:raise RuntimeError(f)
  rec={'frame':f,'source_time':float(z['source_time']),'valid':False}
  if int(z['n_atlas']):
   w,h=z['source_wh'];mask=np.unpackbits(z['atlas0_mask'])[:w*h].reshape(h,w).astype(bool);box=z['atlas0_box'].astype(float);pad=.04*max(box[2]-box[0],box[3]-box[1]);box+=np.array([-pad,-pad,pad,pad]);box=np.clip(box,[0,0,0,0],[w,h,w,h])
   try:
    with contextlib.redirect_stdout(io.StringIO()):res=est.process_one_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),bboxes=box.astype(np.float32).reshape(1,4),masks=mask.reshape(1,h,w),use_mask=True,inference_type='body')
    if res:
     rec['valid']=True
     for k,v in res[0].items():
      if k in ['shape_params','mhr_model_params','pred_cam_t','focal_length','pred_keypoints_3d','pred_keypoints_2d']:rec[k]=v
     k2=res[0]['pred_keypoints_2d'];rec['mask_box']=box
     view=im.copy();view[mask]=(view[mask]*.68+np.array([70,255,100])*.32).astype('uint8')
     for a,b in bones:cv2.line(view,tuple(k2[a].astype(int)),tuple(k2[b].astype(int)),(215,255,40),2,cv2.LINE_AA)
     for i in set(sum(([a,b] for a,b in bones),[])):cv2.circle(view,tuple(k2[i].astype(int)),4,(25,25,25),-1,cv2.LINE_AA);cv2.circle(view,tuple(k2[i].astype(int)),2,(255,255,255),-1,cv2.LINE_AA)
     cv2.putText(view,f'SAM 3.1 / SAM 3D BODY   f{f}   t={f/29.97002997:.3f}s',(30,690),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
     cv2.imwrite(str(R/'overlays'/f'{f:05}.jpg'),view,[cv2.IMWRITE_JPEG_QUALITY,85])
   except Exception as e:rec['error']=str(e);print('POSE_ERROR',f,str(e),flush=True)
  np.savez_compressed(out,**rec);last=time.time();print('POSE',f,rec['valid'],round(time.time()-t0,1),flush=True)
  del rec
 if (R/'sam31-manifest.json').exists() and not pending:break
 if not args.stream:break
 if time.time()-last>1800:raise RuntimeError('No SAM output for 30 min')
 time.sleep(3)
print('POSE_DONE',flush=True)
(R/'pose-complete.json').write_text(json.dumps({'algorithm':'Meta SAM 3D Body, mask-conditioned MHR monocular body reconstruction','device':'Mac mini M4 MPS; memory-mapped CPU checkpoint staging','frames':len(list((R/'raw3d').glob('*.npz'))),'source_space':'camera coordinates; meters, x right y down z away','limitations':['no multiview triangulation','occluded joints inferred','replay clock is not event clock','no calibrated physical ball trajectory']}))
