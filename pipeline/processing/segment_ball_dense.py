import sys,importlib.util,os,json,time
from pathlib import Path
import cv2,numpy as np
R=Path(str(Path(__file__).resolve().parent));P=Path(str(Path(__file__).resolve().parent));sys.path.insert(0,str(P));os.environ['FIGHTLAB_SAM31']=os.environ.get('FIGHTLAB_SAM31',str(Path(__file__).resolve().parents[1]/'models/sam3.1-4bit'));sp=importlib.util.spec_from_file_location('masklib',P/'03_masks.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
data=json.loads((R.parent/'source/ball-track-manual.json').read_text());manual=[s for s in data['samples'] if s.get('x') is not None]; manual_by={v['sourceFrame']:v for v in manual};grid=np.array([v['sourceFrame'] for v in manual]);samples=[]
for f in list(range(2637,2671))+[2679]:
 if f in manual_by:samples.append({**manual_by[f],'roiInterpolated':False})
 else:samples.append({'sourceFrame':f,'x':int(round(np.interp(f,grid,[v['x'] for v in manual]))),'y':int(round(np.interp(f,grid,[v['y'] for v in manual]))),'r':float(np.interp(f,grid,[v['r'] for v in manual])),'roiInterpolated':True})
sam=m.sam31_video(384);c=cv2.VideoCapture(str(R.parent/'source/atlas-quinones-pachuca-2022-source.mp4'));out=R/'deliverable/ball-masks';out.mkdir(exist_ok=True);records=[]
for s in samples:
 f=s['sourceFrame'];cx,cy,r=s['x'],s['y'],s['r'];c.set(1,f);ok,fr=c.read();h,w=fr.shape[:2];pad=max(100,int(r*3));xa=max(0,cx-pad);ya=max(0,cy-pad);xb=min(w,cx+pad);yb=min(h,cy+pad);rgb=cv2.cvtColor(fr[ya:yb,xa:xb],cv2.COLOR_BGR2RGB);H,W=rgb.shape[:2];feat=m.frame_features(sam,rgb);res=sam['detect'](sam['predictor'],feat,['soccer ball'],(W,H),.08);cands=[]
 for q,mask in enumerate(res.masks):
  mask=np.asarray(mask,bool)
  if mask.shape!=(H,W):mask=cv2.resize(mask.astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  mask=m.largest_component(mask);yy,xx=np.nonzero(mask)
  if not len(xx):continue
  dist=np.hypot(xx.mean()+xa-cx,yy.mean()+ya-cy);ratio=len(xx)/(np.pi*r*r)
  if dist<r*1.5 and .25<ratio<2.7:cands.append((dist/r+abs(np.log(ratio))*.2,mask,float(res.scores[q]),ratio))
 rec={'sourceFrame':f,'sourceTime':f*1001/30000,'clipTime':(f-2422)*1001/30000,'clipFrame':f-2422,'roiGuideCenter':[cx,cy],'roiGuideRadius':r,'roiGuideInterpolated':s['roiInterpolated'],'manualCenter':None if s['roiInterpolated'] else [cx,cy],'manualRadius':None if s['roiInterpolated'] else r,'crop':[xa,ya,xb,yb],'algorithm':'SAM3.1 MLX independent concept segmentation','prompt':'soccer ball','accepted':bool(cands)}
 if cands:
  _,mask,score,ratio=sorted(cands,key=lambda z:z[0])[0];full=np.zeros((h,w),'uint8');full[ya:yb,xa:xb]=mask.astype('uint8')*255;name=f'{f:05}_ball.png';cv2.imwrite(str(out/name),full,[cv2.IMWRITE_PNG_COMPRESSION,9]);yy,xx=np.nonzero(full);rec.update({'mask':f'ball-masks/{name}','score':score,'area':int(len(xx)),'centroid':[float(xx.mean()),float(yy.mean())],'areaToManualCircleRatio':ratio,'quality':'model mask accepted by manual-center distance and area sanity checks'})
 else:rec['quality']='no compatible SAM mask; center stays manual only'
 records.append(rec);print(f,rec['accepted'],flush=True);del feat,res;sam['mx'].clear_cache()
manifest={'schema':'atlas-ball-sam31-v1','source':data['sourceUrl'],'frame_selection':'Every source frame2637..2670(34independent inferences), plus2679partialedge; no inference oroverlay2671..2678whenhidden','model':'SAM3.1 MLX4-bit,384px input crop','actual_mask_model_executed':True,'manual_support':'Original13manual centers guide cropping. ROIcenters/radii are linearly interpolated solelyforpromptcropping onintermediateframes; theyare notpublishedasobservations. Everymask comesfromindependentSAM3.1 inference; no drawn circles orinterpolatedmasks.','samples':records,'accepted':sum(r['accepted'] for r in records),'attempted':len(records)};(R/'ball-sam31.json').write_text(json.dumps(manifest,indent=2));print('BALL_DONE',manifest['accepted'],len(records))
