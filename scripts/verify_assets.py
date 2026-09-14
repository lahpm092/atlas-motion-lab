"""Verify the evidence and deliverable, without browser automation."""
from pathlib import Path
import json, math, re, struct, subprocess, hashlib
from html.parser import HTMLParser
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
A = DIST / 'assets'
FPS = 30000 / 1001

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = []; self.refs = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if 'id' in d: self.ids.append(d['id'])
        for key in ['src', 'href', 'poster', 'data-film']:
            if d.get(key, '').startswith('/'): self.refs.append(d[key])

html = Links(); html.feed((DIST / 'index.html').read_text())
assert len(html.ids) == len(set(html.ids)), 'Duplicate element IDs'
for ref in html.refs: assert (DIST / ref.lstrip('/')).is_file(), ref
js = (DIST / 'app.js').read_text() + (DIST / 'physics-viewer.js').read_text()
assert set(re.findall(r"\$\('([^']+)'\)", js)) <= set(html.ids)
d = json.loads((A / 'analysis.json').read_text())
frames = d['frames']
assert len(frames) == 342 and sum(bool(f['valid']) for f in frames) == 217
for i, f in enumerate(frames):
    assert f['clipFrame'] == i and f['frame'] == i + 2422
    assert abs(f['time'] - i / FPS) < 1e-8
    if not f['valid']: assert not f['mask'] and not f['joints2d']
    if f['mask']:
        with Image.open(DIST / f['mask'].lstrip('/')) as img:
            assert img.size == (1280, 720)
            hist = img.convert('L').histogram(); assert not any(hist[1:255]) and hist[255] > 0
    for p in f['joints2d'].values():
        assert math.isfinite(p['x']) and math.isfinite(p['y'])
assert frames[240]['frame'] == 2662 and abs(frames[240]['time'] - 8.008) < 1e-8
assert frames[240]['joints2d']['right_wrist']['x'] == 545
assert frames[240]['raw2d']['right_wrist']['x'] > 600
review = d['manualReview']
assert len(review['frames']) == 16
assert abs(review['frames'][7]['clip_time'] - 8.008) < 1e-8
ball = d['ballSegmentation']['samples']
assert len(ball) == 35
assert {b['sourceFrame'] for b in ball} == set(range(2637, 2671)) | {2679}
for b in ball:
    assert b['accepted'] and abs(b['clipTime'] - (b['sourceFrame'] - 2422) / FPS) < 1e-8
    with Image.open(A / b['mask']) as img:
        assert img.size == (1280, 720)
        hist = img.convert('L').histogram(); assert not any(hist[1:255]) and hist[255] > 0
for filename in ['humanoid.glb', 'humanoid-before.glb']:
    b = (A / filename).read_bytes()
    magic, version, total = struct.unpack_from('<III', b)
    assert magic == 0x46546c67 and version == 2 and total == len(b)
    n, kind = struct.unpack_from('<II', b, 12)
    g = json.loads(b[20:20+n])
    assert len(g['skins'][0]['joints']) == 127 and len(g['animations']) == 1
    accessors = [g['accessors'][s['input']] for s in g['animations'][0]['samplers']]
    assert min(x['min'][0] for x in accessors) == 0
    assert abs(max(x['max'][0] for x in accessors) - 7.007) < 1e-5
assert abs(d['metadata']['animationStart'] - 2.6026) < 1e-8
for name, count, fps, size in [
    ('source.mp4',342,FPS,(1280,720)),
    ('segmentation.mp4',342,FPS,(1280,720)),
    ('pose.mp4',342,FPS,(1280,720)),
    ('manual-review.mp4',168,24,(1280,760)),
    ('atlas-cinematic.mp4',240,24,(1920,1080)),
    ('retargeting.mp4',240,24,(1920,1080)),
]:
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(A/name)]))['streams'][0]
    assert int(probe['nb_frames']) == count, (name, probe['nb_frames'])
    num, den = map(int, probe['avg_frame_rate'].split('/'))
    assert abs(num/den - fps) < .0001 and (probe['width'],probe['height']) == size
    assert probe['codec_name'] == 'h264' and probe['pix_fmt'] == 'yuv420p'
print('PASS: frame alignment, observed/absent data, binary masks, corrected points, GLB timelines, media and local assets.')

physics = json.loads((A / 'physics/simulation.json').read_text())
audit = json.loads((A / 'physics/audit.json').read_text())
gpu = json.loads((A / 'physics/gpu-compatibility.json').read_text())
assert audit['status'] == 'pass' and audit['zeroBallActuators']
assert audit['interpolatedBootBallMinimumGapM'] >= 0
assert audit['ballHorizontalDisplacementWithoutHumanContactM'] < 1e-5
assert abs(audit['fittedFreeFlightGravityMS2'] + 9.81) < .005
assert gpu['finite'] and gpu['device'] == 'cuda:0' and gpu['steps'] == 1
assert physics['metadata']['goal']['inside']
assert physics['metadata']['maxContactPenetration'] == 0
assert physics['metadata']['timestep'] == .0001
assert len(physics['frames']) == 1371
previous = -1
for f in physics['frames']:
    assert f['t'] > previous
    previous = f['t']
    for key in ['p', 'r', 'j', 'v', 'a']:
        assert all(math.isfinite(x) for row in f[key] for x in row)
    assert len(f['p']) == len(physics['geoms'])
    assert len(f['j']) == len(f['v']) == len(f['a']) == len(physics['sites'])
assert 'PERFORMANCE LAB' in (DIST / 'index.html').read_text()
print('PASS: physical data, contact causality, free flight, interpolated collision gap, GPU compatibility and control targets.')

assert audit['minimumPelvisHeightM'] > .85, 'Body loses balance'
assert hashlib.sha256((A / 'physics/model.xml').read_bytes()).hexdigest() == gpu['modelSHA256']
film = json.loads((A / 'physics/render-validation.json').read_text())
assert film['decodePass'] and film['frames'] == 240 and film['samples'] == 32
assert hashlib.sha256((A / 'atlas-cinematic.mp4').read_bytes()).hexdigest() == film['sha256']
assert json.loads((A / 'blender-manifest.json').read_text())['cinematic']['sha256'] == film['sha256']
print('PASS: final stable body, exact CUDA model and final Cycles movie provenance.')
