import numpy as np
class MHRProjector:
 def __init__(self,rig,identity,mapping):
  self.K=np.load(mapping)['keypoint_mapping'][:70];self.vi=rig['linear_blend_skinning.vert_indices_flattened'];self.si=rig['linear_blend_skinning.skin_indices_flattened'];self.w=rig['linear_blend_skinning.skin_weights_flattened'];ib=rig['linear_blend_skinning.inverse_bind_pose'][self.si];self.local=self.qrot(ib[:,3:7],identity[self.vi]*ib[:,7,None])+ib[:,:3]
 @staticmethod
 def qrot(q,v):
  t=2*np.cross(q[:,:3],v);return v+q[:,3,None]*t+np.cross(q[:,:3],t)
 def __call__(self,st,cam,focal):
  b=st[self.si];p=self.qrot(b[:,3:7],self.local*b[:,7,None])+b[:,:3];v=np.zeros((18439,3),np.float32);np.add.at(v,self.vi,p*self.w[:,None]);kp=(self.K[:,:18439]@v+self.K[:,18439:]@st[:,:3])*[.01,-.01,-.01]+cam;return focal*kp[:,:2]/kp[:,2,None]+[640,360]
