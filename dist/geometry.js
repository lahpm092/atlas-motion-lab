export function jointAngle(a,b,c){
 if(!a||!b||!c)return null;
 const pa=Array.isArray(a)?a:[a.x,a.y],pb=Array.isArray(b)?b:[b.x,b.y],pc=Array.isArray(c)?c:[c.x,c.y];
 if(pa.length!==pb.length||pb.length!==pc.length||![...pa,...pb,...pc].every(Number.isFinite))return null;
 const u=pa.map((v,i)=>v-pb[i]),v=pc.map((w,i)=>w-pb[i]),norm=Math.hypot(...u)*Math.hypot(...v);
 if(norm<1e-8)return null;
 return Math.acos(Math.max(-1,Math.min(1,u.reduce((n,w,i)=>n+w*v[i],0)/norm)))*180/Math.PI;
}
export function projectedBallAngle(a,b){if(a?.x==null||b?.x==null||a?.y==null||b?.y==null)return null;const dx=b.x-a.x,dy=b.y-a.y;if(Math.hypot(dx,dy)<1e-8)return null;return Math.atan2(-dy,Math.abs(dx))*180/Math.PI;}
