import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/controls/OrbitControls.js';
import {jointAngle} from './geometry.js';

const $=id=>document.getElementById(id), V=a=>new THREE.Vector3(...a);
const length=a=>Math.hypot(...a);
const lerp=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
const fmt=(v,n=1)=>Number.isFinite(v)?v.toFixed(n):'—';

export function sampleSimulation(data,time){
 const fs=data.frames;let lo=0,hi=fs.length-1;
 while(lo+1<hi){const m=(lo+hi)>>1;if(fs[m].t<=time)lo=m;else hi=m;}
 if(time>=fs.at(-1).t)lo=hi=fs.length-1;
 const a=fs[lo],b=fs[hi],u=a===b?0:THREE.MathUtils.clamp((time-a.t)/(b.t-a.t),0,1);
 return {a,b,u,index:lo,t:time,j:a.j.map((v,i)=>lerp(v,b.j[i],u)),v:a.v.map((v,i)=>lerp(v,b.v[i],u)),acc:a.a.map((v,i)=>lerp(v,b.a[i],u))};
}

function rotation(a){return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(a[0],a[1],a[2],0,a[3],a[4],a[5],0,a[6],a[7],a[8],0,0,0,0,1));}
function geomGeometry(g){
 const s=g.size;
 if(g.type===2)return new THREE.SphereGeometry(s[0],48,32);
 if(g.type===3){const geo=new THREE.CapsuleGeometry(s[0],2*s[1],12,32);geo.rotateX(Math.PI/2);return geo;}
 if(g.type===4){const geo=new THREE.SphereGeometry(1,48,32);geo.scale(...s);return geo;}
 if(g.type===6)return new THREE.BoxGeometry(2*s[0],2*s[1],2*s[2]);
 return null;
}

export async function initPhysicsViewer(){
 const container=$('physics-viewer');let data;
 try{const r=await fetch('/assets/physics/simulation.json');if(!r.ok)throw Error('Simulation unavailable');data=await r.json();if(!data.frames?.length||!data.contacts?.length)throw Error('Incomplete simulation');}
 catch(e){$('physics-status').textContent='NO SE PUDO CARGAR LA SIMULACIÓN. RECARGA LA PÁGINA.';console.error(e);return;}
 const scene=new THREE.Scene();scene.background=new THREE.Color('#ffffff');scene.fog=new THREE.Fog('#ffffff',28,90);
 const camera=new THREE.PerspectiveCamera(34,1,.015,160);camera.up.set(0,0,1);
 let renderer;try{renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'low-power'});}catch{$('physics-status').textContent='LA VISTA 3D REQUIERE WEBGL. LA PELÍCULA ESTÁ DISPONIBLE ABAJO.';return;}
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));renderer.setClearColor('#ffffff');renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;container.prepend(renderer.domElement);
 renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','Simulación del remate. Arrastra para girar y usa la rueda para acercar. Flechas para girar, signos más y menos para acercar.');
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.6;controls.maxDistance=80;controls.maxPolarAngle=Math.PI*.49;controls.target.set(.3,0,.8);
 const ambient=new THREE.HemisphereLight(0xffffff,0xaaa9a1,2.4);ambient.position.set(0,0,10);scene.add(ambient);
 const key=new THREE.DirectionalLight(0xffffff,3.2);key.position.set(-1,-4,9);key.castShadow=true;key.shadow.mapSize.set(2048,2048);Object.assign(key.shadow.camera,{left:-7,right:16,top:9,bottom:-9,near:.1,far:30});key.shadow.bias=-.00015;key.shadow.normalBias=.002;scene.add(key);key.target.position.set(4,0,0);scene.add(key.target);
 const rim=new THREE.DirectionalLight(0xffffff,1);rim.position.set(2,6,4);scene.add(rim);
 const ground=new THREE.Mesh(new THREE.PlaneGeometry(160,120),new THREE.MeshStandardMaterial({color:'#f9f9f6',roughness:.95}));ground.receiveShadow=true;scene.add(ground);
 const pale=new THREE.MeshPhysicalMaterial({color:'#89938b',roughness:.24,metalness:.13,clearcoat:.3,clearcoatRoughness:.3});
 const ink=new THREE.MeshPhysicalMaterial({color:'#272a29',roughness:.36,metalness:.1});
 const ballMat=new THREE.MeshPhysicalMaterial({color:'#c12b1e',roughness:.34,clearcoat:.3});
 const staticMat=new THREE.MeshStandardMaterial({color:'#a9ada9',roughness:.5});
 const meshes=data.geoms.map(g=>{if(g.type===0)return null;const geo=geomGeometry(g);if(!geo)return null;const isBall=g.name==='ball',isBoot=g.name.endsWith('_boot');const material=isBall?ballMat:isBoot?ink:g.body===0?staticMat:pale;const o=new THREE.Mesh(geo,material);o.name=g.name;o.castShadow=true;o.receiveShadow=true;if(g.name==='net_back'){o.material=new THREE.MeshStandardMaterial({color:'#a8aea7',transparent:true,opacity:.045,depthWrite:false});o.castShadow=false;}scene.add(o);return o;});
 const ballIndex=data.geoms.findIndex(g=>g.name==='ball');const seams=new THREE.Group();for(let i=0;i<3;i++){const s=new THREE.Mesh(new THREE.TorusGeometry(.11015,.0007,4,64),ink);if(i===1)s.rotation.x=Math.PI/2;if(i===2)s.rotation.y=Math.PI/2;seams.add(s);}meshes[ballIndex].add(seams);
 // Markings have no collision role; visible body surfaces use the exact solver geoms.
 const pitchLines=[];function seg(a,b){pitchLines.push(...a,...b);}
 const z=.002;for(const y of[-34,34])seg([-93,y,z],[12,y,z]);for(const x of[-93,12])seg([x,-34,z],[x,34,z]);
 for(const [depth,width] of[[16.5,20.16],[5.5,9.16]]){seg([12-depth,-width,z],[12,-width,z]);seg([12-depth,width,z],[12,width,z]);seg([12-depth,-width,z],[12-depth,width,z]);}
 const lines=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(pitchLines,3)),new THREE.LineBasicMaterial({color:'#b7bcb5'}));scene.add(lines);
 const netPoints=[];for(let y=-3.6;y<=3.601;y+=.18)netPoints.push(13.98,y,.02,13.98,y,2.4);for(let h=.06;h<=2.4;h+=.18)netPoints.push(13.98,-3.66,h,13.98,3.66,h);scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(netPoints,3)),new THREE.LineBasicMaterial({color:'#b9bdb8',transparent:true,opacity:.55})));
 const sites=Object.fromEntries(data.sites.map((n,i)=>[n,i]));
 const jointNames=data.sites.filter(n=>!['head','neck','left_toe','right_toe'].includes(n));
 const vectorGroups={velocity:new THREE.Group(),acceleration:new THREE.Group()};Object.values(vectorGroups).forEach(g=>scene.add(g));
 const arrows={};for(const [kind,group]of Object.entries(vectorGroups)){arrows[kind]=jointNames.map(()=>{const a=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),.3,kind==='velocity'?0xc8241a:0x3050a0,.07,.025);group.add(a);return a;});}
 const angles=[['Rodilla I','left_hip','left_knee','left_ankle'],['Cadera I','left_shoulder','left_hip','left_knee'],['Codo I','left_shoulder','left_elbow','left_wrist'],['Tobillo I','left_knee','left_ankle','left_toe']];
 const angleGroup=new THREE.Group();scene.add(angleGroup);const angleObjects=angles.map(()=>{const l=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#c8241a',depthTest:false}));l.renderOrder=4;angleGroup.add(l);const label=document.createElement('span');label.className='physics-label';$('physics-labels').appendChild(label);return{line:l,label,at:null};});
 const jointDots=new THREE.Group();scene.add(jointDots);for(const n of jointNames){const dot=new THREE.Mesh(new THREE.SphereGeometry(.017,12,8),new THREE.MeshBasicMaterial({color:0xc8241a,depthTest:false}));dot.renderOrder=5;jointDots.add(dot);}
 const trajectory=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#c8241a',transparent:true,opacity:.48}));scene.add(trajectory);
 const footTrail=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#252725',transparent:true,opacity:.5}));scene.add(footTrail);
 let time=0,playing=false,last=0,visible=true,mode='strike',userCamera=false,snapshot=null;
 const end=data.frames.at(-1).t,contact=data.metadata.firstBootContact;$('physics-timeline').max=end;$('physics-timeline').step=data.metadata.timestep;
 function setView(v){mode=v;userCamera=false;const presets={strike:{target:[.2,0,.9],camera:[3.0,-4.5,2.35]},goal:{target:[6,0,.75],camera:[4.7,-14.5,5.1]},top:{target:[6.5,0,0],camera:[6.5,-.01,16]},drone:{target:[.3,0,.8],camera:[2.8,-4,2.7]}};const p=presets[v];controls.target.fromArray(p.target);camera.position.fromArray(p.camera);if(v==='goal'){camera.position.sub(controls.target).multiplyScalar(Math.max(1.25,2.35/camera.aspect)).add(controls.target);}if(v==='top'){camera.position.sub(controls.target).multiplyScalar(Math.max(1,1.8/camera.aspect)).add(controls.target);}controls.update();document.querySelectorAll('[data-physics-view]').forEach(b=>{const yes=b.dataset.physicsView===v;b.classList.toggle('selected',yes);b.setAttribute('aria-pressed',String(yes));});}
 function setPlaying(value){playing=value;$('physics-play').textContent=playing?'Ⅱ PAUSAR TIRO':'▶ REPRODUCIR TIRO';$('physics-play').setAttribute('aria-pressed',String(playing));}
 function update(t){
  time=THREE.MathUtils.clamp(t,0,end);snapshot=sampleSimulation(data,time);const {a,b,u,j,v,acc,index}=snapshot;
  meshes.forEach((mesh,i)=>{if(!mesh)return;mesh.position.fromArray(lerp(a.p[i],b.p[i],u));mesh.quaternion.copy(rotation(a.r[i])).slerp(rotation(b.r[i]),u);});
  const showAngles=$('physics-angles').checked,showV=$('physics-velocity').checked,showA=$('physics-acceleration').checked;angleGroup.visible=showAngles;jointDots.visible=showAngles||showV||showA;
  jointDots.children.forEach((o,i)=>o.position.fromArray(j[sites[jointNames[i]]]));
  for(const [kind,vectors,scale,show]of [['velocity',v,.075,showV],['acceleration',acc,.0015,showA]]){vectorGroups[kind].visible=show;arrows[kind].forEach((arrow,i)=>{const si=sites[jointNames[i]],vector=V(vectors[si]),mag=vector.length();arrow.visible=show&&mag>.01;if(arrow.visible){arrow.position.fromArray(j[si]);arrow.setDirection(vector.normalize());const len=Math.min(1.8,mag*scale);arrow.setLength(Math.max(.015,len),Math.min(.09,len*.25),Math.min(.035,len*.1));}});}
  angles.forEach(([name,an,bn,cn],i)=>{const obj=angleObjects[i];obj.label.hidden=!showAngles;if(!showAngles)return;const A=V(j[sites[an]]),B=V(j[sites[bn]]),C=V(j[sites[cn]]),theta=jointAngle(A.toArray(),B.toArray(),C.toArray());obj.label.textContent=`${name} ${fmt(theta)}°`;obj.at=B;const va=A.sub(B).normalize(),vc=C.sub(B).normalize(),points=[],axis=new THREE.Vector3().crossVectors(va,vc);if(axis.lengthSq()<1e-10){axis.crossVectors(va,Math.abs(va.z)<.9?new THREE.Vector3(0,0,1):new THREE.Vector3(1,0,0));}axis.normalize();for(let k=0;k<=24;k++){points.push(va.clone().applyAxisAngle(axis,theta*Math.PI/180*k/24).multiplyScalar(.13).add(B));}obj.line.geometry.dispose();obj.line.geometry=new THREE.BufferGeometry().setFromPoints(points);});
  $('physics-knee').textContent=fmt(jointAngle(j[sites.left_hip],j[sites.left_knee],j[sites.left_ankle]))+'°';$('physics-foot-speed').innerHTML=fmt(length(v[sites.left_toe]),2)+' <small>m/s</small>';$('physics-foot-acc').innerHTML=fmt(length(acc[sites.left_toe]),1)+' <small>m/s²</small>';$('physics-ball-speed').innerHTML=fmt(length(v[sites.ball_center]),2)+' <small>m/s</small>';$('physics-time').textContent=time.toFixed(3)+' s';$('physics-timeline').value=time;
  const past=data.frames.slice(0,index+1);for(const [line,si,start]of [[trajectory,sites.ball_center,contact],[footTrail,sites.left_toe,Math.max(0,time-.5)]]){const points=past.filter(f=>f.t>=start).map(f=>V(f.j[si]));points.push(V(j[si]));line.geometry.dispose();line.geometry=new THREE.BufferGeometry().setFromPoints(points);line.visible=points.length>1&&(line===trajectory||showV||showA);}
  $('physics-legend').textContent=(showV?'Rojo: velocidad, 1 m/s = 7.5 cm. ':'')+(showA?'Azul: aceleración, 100 m/s² = 15 cm. ':'')+((showV||showA)?'Longitud visual máxima: 1.8 m. Las cifras mantienen su valor completo.':'Activa los ángulos y vectores para explorar el remate. El control inferior usa segundos de la simulación.');
 }
 function placeLabels(){angleObjects.forEach(o=>{if(o.label.hidden||!o.at)return;const p=o.at.clone().project(camera);o.label.style.left=((p.x+1)*.5*container.clientWidth)+'px';o.label.style.top=((-p.y+1)*.5*container.clientHeight)+'px';o.label.style.display=p.z>1||p.z<-1?'none':'';});}
 document.querySelectorAll('[data-physics-view]').forEach(b=>b.onclick=()=>{setView(b.dataset.physicsView);userCamera=b.dataset.physicsView!=='drone';});$('physics-play').onclick=()=>{if(time>=end-.001)update(0);setPlaying(!playing);};$('physics-contact').onclick=()=>{setPlaying(false);update(contact);setView('strike');};$('physics-timeline').oninput=e=>{setPlaying(false);update(Number(e.target.value));};for(const id of['physics-angles','physics-velocity','physics-acceleration'])$(id).onchange=()=>update(time);
 controls.addEventListener('start',()=>userCamera=true);$('cinematic')?.addEventListener('play',()=>{setPlaying(false);$('source')?.pause();});
 renderer.domElement.addEventListener('keydown',e=>{const keys=['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'];if(!keys.includes(e.key))return;e.preventDefault();userCamera=true;const off=camera.position.clone().sub(controls.target);if(e.key==='+'||e.key==='=')off.multiplyScalar(.88);else if(e.key==='-')off.multiplyScalar(1.12);else{const axis=e.key.includes('Left')||e.key.includes('Right')?new THREE.Vector3(0,0,1):new THREE.Vector3().crossVectors(off,new THREE.Vector3(0,0,1)).normalize();off.applyAxisAngle(axis,['ArrowLeft','ArrowUp'].includes(e.key)?.1:-.1);}camera.position.copy(controls.target).add(off);controls.update();});
 const observer=new IntersectionObserver(([e])=>{visible=e.isIntersecting;document.body.classList.toggle('physics-active',visible);if(visible)$('source')?.pause();else setPlaying(false);},{threshold:.2});observer.observe(container);const resize=new ResizeObserver(()=>{renderer.setSize(container.clientWidth,container.clientHeight,false);camera.aspect=container.clientWidth/container.clientHeight;camera.updateProjectionMatrix();});resize.observe(container);
 const m=data.metadata,launch=m.launch;$('physics-contact-summary').textContent=`El contacto transfiere ${fmt(m.bootImpulse,2)} N·s. El balón sale a ${fmt(launch?.speed,2)} m/s con ${fmt(launch?.elevation,1)}° de elevación. ${m.goal?.inside?'El tiro cruza dentro de la portería.':'El tiro no cruza dentro de la portería.'}`;
 $('physics-verification').textContent=`Integración cada ${(m.timestep*1000).toFixed(2)} ms. Interpenetración máxima registrada: ${(m.maxContactPenetration*1000).toFixed(3)} mm. ${data.frames.length} muestras exportadas. Motor ${m.engine} ${m.version}. La geometría visible del cuerpo coincide con sus superficies de colisión.`;
 drawForceChart(data);setView('strike');update(0);container.classList.add('loaded');
 function render(now){requestAnimationFrame(render);const dt=Math.min((now-last)/1000,.05);last=now;if(document.hidden||!visible)return;if(playing){update(time+dt*Number($('physics-speed').value));if(time>=end)setPlaying(false);}if(mode==='drone'&&!userCamera&&!matchMedia('(prefers-reduced-motion: reduce)').matches){const off=camera.position.clone().sub(controls.target);off.applyAxisAngle(new THREE.Vector3(0,0,1),dt*.22);camera.position.copy(controls.target).add(off);}if(playing&&!userCamera&&['strike','drone'].includes(mode)&&time>contact+.12){const x=THREE.MathUtils.smoothstep(time,contact+.12,contact+.9);controls.target.lerp(new THREE.Vector3(6,0,.75),Math.min(1,dt*3*x));const goalTarget=new THREE.Vector3(6,0,.75),goalPosition=new THREE.Vector3(4.7,-14.5,5.1).sub(goalTarget).multiplyScalar(Math.max(1.25,2.35/camera.aspect)).add(goalTarget);camera.position.lerp(goalPosition,Math.min(1,dt*2.5*x));}controls.update();placeLabels();renderer.render(scene,camera);}requestAnimationFrame(render);
 renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();setPlaying(false);container.classList.remove('loaded');$('physics-status').textContent='VISTA 3D PAUSADA POR EL DISPOSITIVO. RECARGA O ABRE LA PELÍCULA.';});
 window.addEventListener('pagehide',()=>{observer.disconnect();resize.disconnect();controls.dispose();renderer.dispose();scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose?.();});},{once:true});
}

function drawForceChart(data){const c=$('physics-force-chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,cs=data.contacts,peak=Math.max(...cs.map(c=>c.force)),start=cs[0].t,end=cs.at(-1).t,span=Math.max(data.metadata.timestep,end-start),px=t=>65+(t-start)/span*(w-100),py=f=>h-45-f/peak*(h-85);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.font='15px "Courier New",monospace';ctx.fillStyle='#555';ctx.fillText('FUERZA NORMAL / N',65,22);for(const f of[0,peak/2,peak]){ctx.strokeStyle='#deded9';ctx.beginPath();ctx.moveTo(65,py(f));ctx.lineTo(w-25,py(f));ctx.stroke();ctx.fillText(f.toFixed(0),3,py(f)+5);}ctx.beginPath();cs.forEach((c,i)=>i?ctx.lineTo(px(c.t),py(c.force)):ctx.moveTo(px(c.t),py(c.force)));ctx.strokeStyle='#c8241a';ctx.lineWidth=2;ctx.stroke();ctx.fillText(start.toFixed(4)+' s',65,h-15);ctx.fillText(end.toFixed(4)+' s',w-120,h-15);}

if(typeof document!=='undefined'&&$('physics-viewer')){const o=new IntersectionObserver(([e])=>{if(e.isIntersecting){o.disconnect();initPhysicsViewer();}},{rootMargin:'600px'});o.observe($('physics-viewer'));}
