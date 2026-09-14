"""Independent contact, playback, causality and ballistic audits."""
from pathlib import Path
import json,bisect
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent;out=ROOT/'output'
d=json.loads((out/'simulation.json').read_text());m=mujoco.MjModel.from_xml_path(str(out/'model.xml'));state=mujoco.MjData(m)
fs=d['frames'];times=np.array([f['t'] for f in fs]);ball=next(i for i,g in enumerate(d['geoms']) if g['name']=='ball');boot=next(i for i,g in enumerate(d['geoms']) if g['name']=='left_boot')
maxerr=0;min_gap=100
for f in fs:
    state.qpos[:]=f['qpos'];state.qvel[:]=f['qvel'];mujoco.mj_forward(m,state)
    maxerr=max(maxerr,float(np.max(np.abs(state.geom_xpos-np.array(f['p'])))))
    assert np.isfinite(state.qacc).all()
def rotation(a,b,u):
    qa=np.zeros(4);qb=np.zeros(4);mujoco.mju_mat2Quat(qa,np.array(a));mujoco.mju_mat2Quat(qb,np.array(b));dot=np.dot(qa,qb)
    if dot<0:qb=-qb;dot=-dot
    if dot>.9995:q=qa+(qb-qa)*u;q/=np.linalg.norm(q)
    else:theta=np.arccos(np.clip(dot,-1,1));q=(np.sin((1-u)*theta)*qa+np.sin(u*theta)*qb)/np.sin(theta)
    mat=np.zeros(9);mujoco.mju_quat2Mat(mat,q);return mat
for t in np.linspace(.55,.8,2501):
    ix=max(0,min(len(fs)-2,bisect.bisect_right(times,t)-1));a,b=fs[ix:ix+2];u=(t-a['t'])/(b['t']-a['t'])
    for i in[ball,boot]:state.geom_xpos[i]=np.array(a['p'][i])*(1-u)+np.array(b['p'][i])*u;state.geom_xmat[i]=rotation(a['r'][i],b['r'][i],u)
    gap=mujoco.mj_geomDistance(m,state,boot,ball,10,np.zeros(6));min_gap=min(min_gap,gap)
# Free flight must follow gravity; fit the same coordinates shown in the viewer.
si=d['sites'].index('ball_center');flight=[f for f in fs if .70<f['t']<.95]
ts=np.array([f['t'] for f in flight]);zs=np.array([f['j'][si][2] for f in flight]);coef=np.polyfit(ts,zs,2);gravity_fit=float(2*coef[0]);residual=float(np.max(np.abs(np.polyval(coef,ts)-zs)))
# Remove human-ball contact while preserving ground contact and the same control.
# Excluding only the boot allows the shin to hit the ball during follow-through.
human_bodies=['pelvis']+[f'{side}_{limb}' for side in ['left','right'] for limb in ['foot','shin','thigh','upperarm','forearm']]
exclusions='<contact>'+''.join(f'<exclude body1="{body}" body2="ball"/>' for body in human_bodies)+'</contact>'
xml=(out/'model.xml').read_text().replace('<equality>',exclusions+'<equality>')
cm=mujoco.MjModel.from_xml_string(xml);cd=mujoco.MjData(cm);initial=fs[0];cd.qpos[:]=initial['qpos'];cd.qvel[:]=initial['qvel'];mujoco.mj_forward(cm,cd)
from simulate import target
from types import SimpleNamespace
args=SimpleNamespace(**d['metadata']['parameters'])
for step in range(round(4/cm.opt.timestep)):cd.ctrl[:]=target(cd.time,args);mujoco.mj_step(cm,cd)
mujoco.mj_forward(cm,cd);ballbody=mujoco.mj_name2id(cm,mujoco.mjtObj.mjOBJ_BODY,'ball');no_contact_displacement=float(np.linalg.norm(cd.xpos[ballbody][:2]-np.array(fs[0]['j'][si][:2])))
r={'status':'pass','poseGeometryMaxErrorM':maxerr,'interpolatedBootBallMinimumGapM':float(min_gap),'playbackSamplesAudited':2501,'fittedFreeFlightGravityMS2':gravity_fit,'freeFlightMaxResidualM':residual,'ballHorizontalDisplacementWithoutHumanContactM':no_contact_displacement,'goal':d['metadata']['goal'],'zeroBallActuators':all(cm.actuator_trnid[i,0]!=cm.body_jntadr[ballbody] for i in range(cm.nu))}
assert maxerr<.000002
assert min_gap>=0
assert abs(gravity_fit+9.81)<.005 and residual<.000002
assert no_contact_displacement<.00001
assert r['goal']['inside'] and r['zeroBallActuators']
r['minimumPelvisHeightM']=min(f['j'][0][2] for f in fs)
assert r['minimumPelvisHeightM']>.85, 'Body lost balance after the strike'
(out/'audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
