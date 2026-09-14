from pathlib import Path
import sys,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parent
import imageio_ffmpeg
ff=imageio_ffmpeg.get_ffmpeg_exe()
frames=ROOT/'blender/frames'
assert len(list(frames.glob('frame_*.png')))==240
out=ROOT/'blender/atlas-cinematic.mp4'
subprocess.run([ff,'-y','-v','error','-framerate','24','-i',str(frames/'frame_%04d.png'),'-frames:v','240','-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
subprocess.run([ff,'-v','error','-i',str(out),'-f','null','-'],check=True)
r={'engine':'Blender Cycles','device':'Mac Studio / Metal','samples':32,'frames':240,'fps':24,'dimensions':[1920,1080],'duration':10,'decodePass':True,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'physicsTiming':'film-timing.json'}
(ROOT/'blender/render-validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
