"""Actual SAM3.1 concept inference at every selected source frame.
No silhouette is synthesized. Color/geometry identify the on-field Atlas kit.
Source frames stay explicit: the initial pass samples 10Hz; dense pass --stride 1.
"""
import argparse,importlib.util,sys,os,json,time,hashlib
from pathlib import Path
import numpy as np,cv2
R=Path(os.environ.get('ATLAS_PROCESSING',str(Path(__file__).resolve().parent)));R.mkdir(exist_ok=True)
ROOT=Path(os.environ.get('ATLAS_PIPELINE',str(Path(__file__).resolve().parent)));sys.path.insert(0,str(ROOT))
os.environ.setdefault('FIGHTLAB_SAM31',os.environ.get('FIGHTLAB_SAM31',str(Path(__file__).resolve().parents[1]/'models/sam3.1-4bit')))
spec=importlib.util.spec_from_file_location('masklib',ROOT/'03_masks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=argparse.ArgumentParser();a.add_argument('--stride',type=int,default=1);a.add_argument('--start',type=float,default=2422*1001/30000);a.add_argument('--end',type=float,default=2763*1001/30000);a.add_argument('--resolution',type=int,default=384);args=a.parse_args()
source=R.parent/'source/atlas-quinones-pachuca-2022-source.mp4'; cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);wh=(int(cap.get(3)),int(cap.get(4)))
out=R/'masks-v3';out.mkdir(exist_ok=True);sam=m.sam31_video(args.resolution);t0=time.time();record=[]
for fi in range(round(args.start*fps),round(args.end*fps)+1,args.stride):
 fn=out/f'{fi:05}.npz'
 if fn.exists():continue
 cap.set(1,fi);ok,fr=cap.read()
 if not ok:raise RuntimeError(fi)
 rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB); features=m.frame_features(sam,rgb)
 res=sam['detect'](sam['predictor'],features,['person'],wh,.12)
 hsv=cv2.cvtColor(fr,cv2.COLOR_BGR2HSV);red=((hsv[:,:,0]<11)|(hsv[:,:,0]>164))&(hsv[:,:,1]>100)&(hsv[:,:,2]>60)&(fr[:,:,2].astype(float)>1.8*fr[:,:,1].astype(float)+10)&(fr[:,:,2].astype(float)>1.3*fr[:,:,0]+10);blue=(hsv[:,:,0]>95)&(hsv[:,:,0]<135)&(hsv[:,:,1]>70)&(hsv[:,:,2]>100)
 candidates=[]
 for q,mask in enumerate(res.masks):
  mask=np.asarray(mask,bool)
  if mask.shape!=(wh[1],wh[0]):mask=cv2.resize(mask.astype('uint8'),wh,interpolation=cv2.INTER_NEAREST).astype(bool)
  mask=m.largest_component(m.clip_to_detection(mask,np.asarray(res.boxes[q])))
  box=m.mask_box(mask); bw,bh=box[2]-box[0],box[3]-box[1];area=int(mask.sum());rr=float((mask&red).sum())/max(1,area)
  br=float((mask&blue).sum())/max(1,area)
  if bh>180 and box[3]>450 and rr>.06 and rr>1.3*br and area>4000 and bh/bw>.9:
   candidates.append((area*rr,mask,box,float(res.scores[q]),rr))
 candidates.sort(key=lambda x:x[0],reverse=True)
 data={'frame':fi,'source_time':fi/fps,'source_fps':fps,'source_wh':wh,'n_atlas':len(candidates)}
 for j,c in enumerate(candidates):
  _,mask,box,score,rr=c;data[f'atlas{j}_mask']=np.packbits(mask.ravel());data[f'atlas{j}_box']=box;data[f'atlas{j}_score']=score;data[f'atlas{j}_red_fraction']=rr
 np.savez_compressed(fn,**data)
 print(json.dumps({'frame':fi,'t':round(fi/fps,2),'atlas':len(candidates),'boxes':[c[2].tolist() for c in candidates],'seconds':round(time.time()-t0,1)}),flush=True)
 del features,res;sam['mx'].clear_cache()
manifest={'algorithm':'SAM 3.1 MLX concept detector, independent inference per analysed frame','checkpoint':'local 4-bit affine conversion of mlx-community/sam3.1-bf16','prompt':'person','source':str(source),'source_fps':fps,'source_dimensions':wh,'source_start':args.start,'source_end':args.end,'stride':args.stride,'sample_hz':fps/args.stride,'identity_rule':'foreground full-body silhouette with Atlas red kit; reviewed by frame, uncertain occlusions may be absent','mask_representation':'packbits row-major [H,W], npz','no_temporal_propagation':True,'official_model':'https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md'}
(R/'sam31-manifest-v3.json').write_text(json.dumps(manifest,indent=2));print('DONE',flush=True)
