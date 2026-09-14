import json,struct,numpy as np,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent/'deliverable';receipts=[]
for name in ['humanoid.glb','humanoid-before.glb']:
 p=R/name;raw=p.read_bytes();n=struct.unpack_from('<I',raw,12)[0];j=json.loads(raw[20:20+n]);offset=20+n;bn,btype=struct.unpack_from('<I4s',raw,offset);binary=bytearray(raw[offset+8:offset+8+bn]);ids=sorted({s['input'] for a in j['animations'] for s in a['samplers']});arrays=[]
 for ix in ids:
  a=j['accessors'][ix];v=j['bufferViews'][a['bufferView']];off=v.get('byteOffset',0)+a.get('byteOffset',0);arr=np.frombuffer(binary,dtype='<f4',count=a['count'],offset=off);arrays.append((ix,arr))
 longest=max(arrays,key=lambda x:len(x[1]))[1];first=float(longest[0]);step=float(np.median(np.diff(longest)))
 if abs(first)<1e-6 and abs(step-1001/30000)<1e-6:continue
 for ix,arr in arrays:
  fi=np.rint((arr-first)/step);arr[:]=fi*(1001/30000);j['accessors'][ix]['min']=[float(arr.min())];j['accessors'][ix]['max']=[float(arr.max())]
 js=json.dumps(j,separators=(',',':'),ensure_ascii=False).encode();js+=b' '*((-len(js))%4);binary+=b'\0'*((-len(binary))%4);data=struct.pack('<4sII',b'glTF',2,12+8+len(js)+8+len(binary))+struct.pack('<I4s',len(js),b'JSON')+js+struct.pack('<I4s',len(binary),b'BIN\0')+binary;p.write_bytes(data);receipts.append({'file':name,'sha256':hashlib.sha256(data).hexdigest(),'inputs_normalized':len(ids),'first':float(longest[0]),'last':float(longest[-1]),'expected_last':210*1001/30000,'fps':30000/1001,'reason':'Exporter frame1 offset and inverted fps_base corrected; only animation input timestamps changed.'})
(R/'glb-time-audit.json').write_text(json.dumps(receipts,indent=2));m=json.load(open(R/'blender-manifest.json'));m['humanoid']['duration']=210*1001/30000;m['humanoid']['time_normalization']='Sourceframe2500->0; frame2710->7.007s. Accessors normalized after export; posevalues unchanged.';(R/'blender-manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(receipts,indent=2))
