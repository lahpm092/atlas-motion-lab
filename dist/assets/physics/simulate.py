"""Atlas contact reconstruction. SI units, articulated body and an unforced ball.

The support boot is welded to the ground (planted-foot boundary condition).
The kicking leg is torque-controlled; neither the boot nor ball is teleported.
This is a constrained biomechanical experiment, not recovered athlete dynamics.
"""
from pathlib import Path
import argparse, json, time, platform
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
DT=.0001

def build_model(ball_x=.48, ball_z=.112, hip_end=-.80, knee_end=.08):
    limbs=[]; actuators=[]
    for side, y in [('left',.115),('right',-.115)]:
        limbs.append(f'''<body name="{side}_thigh" pos="0 {y} 0">
          <joint name="{side}_hip" axis="0 1 0" range="-100 70"/>
          <geom name="{side}_thigh_geom" type="capsule" fromto="0 0 -.06 0 0 -.37" size=".068" mass="8"/>
          <site name="{side}_hip" pos="0 0 0"/>
          <body name="{side}_shin" pos="0 0 -.43">
            <joint name="{side}_knee" axis="0 1 0" range="0 140"/>
            <geom name="{side}_shin_geom" type="capsule" fromto="0 0 -.045 0 0 -.38" size=".047" mass="3.5"/>
            <site name="{side}_knee"/>
            <body name="{side}_foot" pos="0 0 -.43">
              <joint name="{side}_ankle" axis="0 1 0" range="-50 50"/>
              <geom name="{side}_boot" type="ellipsoid" pos=".065 0 -.015" size=".145 .058 .052" mass="1" rgba=".12 .12 .12 1" friction=".8 .01 .001"/>
              <site name="{side}_ankle"/><site name="{side}_toe" pos=".2 0 -.015"/>
            </body>
          </body>
        </body>''')
        limbs.append(f'''<body name="{side}_upperarm" pos="0 {y/abs(y)*.22} .39">
          <joint name="{side}_shoulder" axis="0 1 0" range="-100 100"/>
          <geom name="{side}_upperarm_geom" type="capsule" fromto="0 0 0 0 {y/abs(y)*.06} -.25" size=".047" mass="2"/>
          <site name="{side}_shoulder"/>
          <body name="{side}_forearm" pos="0 {y/abs(y)*.06} -.28">
            <joint name="{side}_elbow" axis="0 1 0" range="-140 0"/>
            <geom name="{side}_forearm_geom" type="capsule" fromto="0 0 0 0 0 -.22" size=".034" mass="1.2"/>
            <site name="{side}_elbow"/><site name="{side}_wrist" pos="0 0 -.25"/>
            <geom name="{side}_hand" type="ellipsoid" pos="0 0 -.265" size=".025 .043 .065" mass=".4"/>
          </body>
        </body>''')
        for name, kp,kv,force in [('hip',1400,65,500),('knee',1100,45,350),('ankle',350,20,150),('shoulder',200,16,100),('elbow',150,12,70)]:
            if side=='right' and name in ['hip','knee','ankle']:
                kp,kv,force={'hip':(6000,250,600),'knee':(5000,170,500),'ankle':(3500,180,250)}[name]
            actuators.append(f'<position name="{side}_{name}" joint="{side}_{name}" kp="{kp}" kv="{kv}" forcerange="-{force} {force}"/>')
    xml=f'''<mujoco model="Atlas planted-support kick">
      <compiler angle="degree" autolimits="true"/>
      <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast" iterations="80" tolerance="1e-10" cone="elliptic"/>
      <size njmax="3000" nconmax="300"/>
      <default><joint damping=".3" armature=".03" limited="true"/>
        <geom condim="6" friction=".75 .006 .0002" margin=".001" solref=".0004 1" solimp=".995 .9999 .0002" rgba=".83 .84 .82 1"/>
        <site size=".01" rgba=".8 .05 .03 1"/>
      </default>
      <worldbody>
        <light pos="0 -3 6" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="60 40 .1" rgba=".95 .95 .93 1" friction=".8 .002 .00005"/>
        <body name="pelvis" pos="0 0 .927"><freejoint name="root"/>
          <geom name="pelvis_geom" type="capsule" fromto="0 -.09 .015 0 .09 .015" size=".095" mass="10"/>
          <site name="pelvis"/>
          <geom name="abdomen" type="ellipsoid" pos="0 0 .16" size=".11 .14 .18" mass="10"/>
          <geom name="chest" type="ellipsoid" pos="0 0 .32" size=".125 .195 .16" mass="13"/>
          <geom name="neck" type="capsule" fromto="0 0 .44 0 0 .5" size=".045" mass=".8"/>
          <geom name="head" type="ellipsoid" pos=".008 0 .61" size=".085 .078 .11" mass="4.5"/>
          <site name="neck" pos="0 0 .47"/><site name="head" pos="0 0 .61"/>
          {''.join(limbs)}
        </body>
        <body name="ball" pos="{ball_x} .115 {ball_z}"><freejoint name="ball_free"/>
          <geom name="ball" type="sphere" size=".11" mass=".43" rgba=".72 .08 .045 1" solref=".0004 .8" friction=".55 .001 .00002"/>
          <site name="ball_center"/>
        </body>
        <geom name="post_left" type="capsule" fromto="12 -3.72 .06 12 -3.72 2.50" size=".06"/>
        <geom name="post_right" type="capsule" fromto="12 3.72 .06 12 3.72 2.50" size=".06"/>
        <geom name="crossbar" type="capsule" fromto="12 -3.72 2.50 12 3.72 2.50" size=".06"/>
        <geom name="net_back" type="box" pos="14 0 1.22" size=".015 3.66 1.22" rgba=".8 .8 .8 .15" solref=".0004 1"/>
      </worldbody>
      <equality><weld name="planted_support" body1="right_foot" solref=".003 1"/></equality>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml),xml

def smooth(a,b,t):
    t=np.clip(t,0,1); t=t*t*t*(10+t*(-15+6*t)); return a+(b-a)*t

def target(t,args):
    strike=smooth(0,1,(t-.5)/args.swing)
    recovery=smooth(0,1,(t-(.5+args.swing+.1))/.55)
    hip=(1-recovery)*(.4+(args.hip-.4)*strike)+recovery*(-.2)
    knee=(1-recovery)*(.85+(args.knee-.85)*strike)+recovery*.25
    # Plantar flexion stabilizes the instep. The foot still obeys joint dynamics.
    ankle=-.20*(1-strike)+args.ankle*strike
    return np.array([hip,knee,ankle,-.3+.8*strike,-.7,
                     0,.015,-.015,.35-.65*strike,-.8])

def run(args):
    m,xml=build_model(args.ball_x,args.ball_z,args.hip,args.knee)
    d=mujoco.MjData(m)
    q=target(0,args)
    for i in range(m.nu):
        jid=m.actuator_trnid[i,0]; d.qpos[m.jnt_qposadr[jid]]=q[i]
    d.ctrl[:]=q
    mujoco.mj_forward(m,d)
    ball=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ball')
    boot=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'left_boot')
    ballgeom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ball')
    sites=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) for i in range(m.nsite)]
    geoms=[{'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i),'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'body':int(m.geom_bodyid[i]),'rgba':m.geom_rgba[i].tolist()} for i in range(m.ngeom)]
    frames=[]; contacts=[]; maxpen=0; maxpair=None; maxboot=0; goal=None; peak=0; first=None; last=None; launch=None
    tick=time.perf_counter(); previous_ball=d.xpos[ball].copy(); min_ground=100
    for step in range(round(args.seconds/DT)):
        d.ctrl[:]=target(d.time,args)
        mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        t=float(d.time)
        pos=d.xpos[ball].copy(); speed=float(np.linalg.norm(d.qvel[m.jnt_dofadr[m.body_jntadr[ball]]:m.jnt_dofadr[m.body_jntadr[ball]]+3]));peak=max(peak,speed)
        contactforce=np.zeros(6); frame_contacts=[]
        for ci in range(d.ncon):
            c=d.contact[ci]
            if -c.dist>maxpen:
                maxpen=-c.dist;maxpair=[geoms[c.geom1]['name'],geoms[c.geom2]['name'],t]
            if boot in [c.geom1,c.geom2] and ballgeom in [c.geom1,c.geom2]:
                first=t if first is None else first; last=t;maxboot=max(maxboot,-c.dist)
                mujoco.mj_contactForce(m,d,ci,contactforce)
                contacts.append({'t':round(t,6),'force':round(float(contactforce[0]),5),'penetration':round(max(0.,-float(c.dist)),7),'position':c.pos.tolist()})
            if ballgeom in [c.geom1,c.geom2]: frame_contacts.append([int(c.geom1),int(c.geom2),float(c.dist)])
        if first is not None and t>last+DT*.5 and launch is None:
            adr=m.jnt_dofadr[m.body_jntadr[ball]];v=d.qvel[adr:adr+3].copy()
            launch={'t':t,'v':v.tolist(),'speed':float(np.linalg.norm(v)),'elevation':float(np.degrees(np.arctan2(v[2],np.linalg.norm(v[:2]))))}
        if previous_ball[0]<12<=pos[0] and goal is None:
            alpha=(12-previous_ball[0])/(pos[0]-previous_ball[0]); cross=previous_ball+(pos-previous_ball)*alpha
            goal={'t':t-DT+alpha*DT,'center':cross.tolist(),'inside':bool(abs(cross[1])+.11<3.66 and .11<cross[2]<2.44-.11)}
        previous_ball=pos.copy()
        min_ground=min(min_ground,float(pos[2]-.11))
        if step%int(round(1/(120*DT)))==0 or .61<=t<=.70:
            # mj_step positions refer to the beginning of the integration interval.
            # Recompute positions and velocities at the exported qpos/qvel time.
            sitevel=[]
            for si in range(m.nsite):
                v=np.zeros(6); mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,si,v,0);sitevel.append(v[3:].tolist())
            frames.append({'t':round(t,6),'p':np.round(d.geom_xpos,6).tolist(),'r':np.round(d.geom_xmat,7).tolist(),'j':np.round(d.site_xpos,6).tolist(),'v':np.round(sitevel,6).tolist(),'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist()})
    times=np.array([f['t'] for f in frames]); velocities=np.array([f['v'] for f in frames]); accelerations=np.gradient(velocities,times,axis=0,edge_order=2)
    for f,a in zip(frames,accelerations):f['a']=np.round(a,5).tolist()
    report={'engine':'MuJoCo','version':mujoco.__version__,'backend':'CPU','timestep':DT,'gravity':[0,0,-9.81],'ballMass':.43,'ballRadius':.11,'support':'Right boot welded to the ground; articulated dynamic body, torque-limited position servos.','ballActuation':'None. Initial ball velocity is zero. Motion comes only from gravity and contacts.','firstBootContact':first,'lastBootContact':last,'bootImpulse':sum(c['force']*DT for c in contacts),'maxBootPenetration':maxboot,'maxContactPenetration':maxpen,'minBallGroundClearance':min_ground,'peakBallSpeed':peak,'goal':goal,'elapsed':time.perf_counter()-tick,'parameters':vars(args),'realism':'Constrained reconstruction of a left-footed kick, not measured forces or physical timing of Quinones.','frameCount':len(frames),'sampleInterval':float(times[1]-times[0])}
    report['maxPenetrationPair']=maxpair;report['launch']=launch
    report['pelvisHeightRange']=[min(f['j'][0][2] for f in frames),max(f['j'][0][2] for f in frames)]
    print(json.dumps(report,indent=2),flush=True)
    if args.export:
        out=ROOT/'output';out.mkdir(exist_ok=True)
        (out/'model.xml').write_text(xml,encoding='utf-8')
        (out/'simulation.json').write_text(json.dumps({'metadata':report,'geoms':geoms,'sites':sites,'frames':frames,'contacts':contacts},separators=(',',':')),encoding='utf-8')
        (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ball-x',type=float,default=.48);p.add_argument('--ball-z',type=float,default=.112);p.add_argument('--hip',type=float,default=-.80);p.add_argument('--knee',type=float,default=.08);p.add_argument('--ankle',type=float,default=-.1);p.add_argument('--swing',type=float,default=.22);p.add_argument('--seconds',type=float,default=4);p.add_argument('--export',action='store_true');run(p.parse_args())
