# -*- coding: utf-8 -*-
"""
FACIAL INHERITANCE PREDICTOR - Web App
=======================================
Flask web app: upload parent photos, generate predicted child face.
"""

import os
import uuid
import json
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'facepredictor'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED = {'jpg', 'jpeg', 'png', 'webp'}

# ── Face processing using YuNet ONNX + proper blending ────

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet.onnx")
_detector  = None

def get_detector():
    global _detector
    if _detector is None:
        if os.path.exists(MODEL_PATH):
            _detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320,320))
        else:
            # fallback to haar cascade
            _detector = "haar"
    return _detector


def detect_face_region(img):
    """Returns (x1,y1,x2,y2) of largest face, or None."""
    h, w = img.shape[:2]
    det = get_detector()

    if det == "haar":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        return x, y, x+fw, y+fh

    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda x: x[2]*x[3])
    return int(f[0]), int(f[1]), int(f[0]+f[2]), int(f[1]+f[3])


def crop_face(img, padding=0.4, size=512):
    """Crop and align face from image."""
    region = detect_face_region(img)
    h, w   = img.shape[:2]
    if region:
        x1, y1, x2, y2 = region
        fw, fh = x2-x1, y2-y1
        pad = int(max(fw, fh) * padding)
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(w, x2+pad); y2 = min(h, y2+pad)
        crop = img[y1:y2, x1:x2]
    else:
        # No face - use centre crop
        s = min(h, w)
        y0 = (h-s)//2; x0 = (w-s)//2
        crop = img[y0:y0+s, x0:x0+s]
    return cv2.resize(crop, (size, size))


def laplacian_blend(img1, img2, alpha=0.5, levels=5):
    """
    Laplacian pyramid blending - much better than simple addWeighted.
    Blends low frequencies (skin tone, shape) and high frequencies
    (texture, features) separately for a natural result.
    """
    def build_gaussian(img, lvls):
        gp = [img.astype(np.float32)]
        for _ in range(lvls):
            gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def build_laplacian(gp):
        lp = []
        for i in range(len(gp)-1):
            up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    gp1 = build_gaussian(img1, levels)
    gp2 = build_gaussian(img2, levels)
    lp1 = build_laplacian(gp1)
    lp2 = build_laplacian(gp2)

    blended_lp = [l1*alpha + l2*(1-alpha) for l1, l2 in zip(lp1, lp2)]

    result = blended_lp[-1]
    for i in range(len(blended_lp)-2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_lp[i].shape[1],
                                            blended_lp[i].shape[0]))
        result += blended_lp[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def extract_embedding(image_path):
    """Extract face features using color + texture descriptors."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not read image"
        face = crop_face(img, size=128)

        # Multi-scale color histograms in LAB space (perceptually uniform)
        lab  = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([lab],[ch],None,[64],[0,256]).flatten()
            hists.append(h / (h.sum()+1e-7))

        # LBP-like texture features
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag  = np.sqrt(gx**2 + gy**2)
        ang  = np.arctan2(gy, gx)
        hog, _ = np.histogram(ang.flatten(), bins=32, weights=mag.flatten())
        hog = hog / (hog.sum()+1e-7)

        embedding = np.concatenate(hists + [hog])
        return embedding, None
    except Exception as e:
        return None, str(e)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def analyze_face_traits(image_path):
    """Detect face and return basic info."""
    try:
        img    = cv2.imread(image_path)
        region = detect_face_region(img)
        if region:
            x1,y1,x2,y2 = region
            fw, fh = x2-x1, y2-y1
            # Rough age hint from face size relative to image
            ih, iw = img.shape[:2]
            ratio  = (fw*fh) / (iw*ih)
            return {"age": "?", "gender": "?",
                    "race": f"face detected ({fw}x{fh}px)"}
        return {"age": "?", "gender": "?", "race": "no face detected"}
    except Exception:
        return {"age": "?", "gender": "?", "race": "?"}


def blend_embeddings(pairs):
    total   = sum(w for _, w in pairs)
    blended = np.zeros(pairs[0][0].shape, dtype=np.float64)
    for emb, w in pairs:
        blended += emb * (w / total)
    return blended


def generate_child_face(blended_embedding, parent_images, output_path,
                        weights=None, age_target=10):
    """
    Generate child face using Laplacian pyramid blending of aligned faces.
    Supports multiple parents/relatives with individual weights.
    """
    aligned = []
    wts     = weights if weights else [1.0] * len(parent_images)

    for path in parent_images:
        img = cv2.imread(path)
        if img is None:
            continue
        aligned.append(crop_face(img, size=512))

    if len(aligned) < 2:
        return False, "Could not process images"

    # Normalise weights
    total = sum(wts[:len(aligned)])
    wts   = [w/total for w in wts[:len(aligned)]]

    # Blend first two with Laplacian pyramid
    result = laplacian_blend(aligned[0], aligned[1], alpha=wts[0])

    # Blend in any additional family members
    for i in range(2, len(aligned)):
        result = laplacian_blend(result, aligned[i],
                                 alpha=1.0-wts[i])

    # Slight skin smoothing for child-like appearance
    result = cv2.bilateralFilter(result, 9, 75, 75)

    cv2.imwrite(output_path, result)
    return True, "Generated with Laplacian pyramid blend"


def apply_age_effect(image_path, output_path, target_age):
    """
    Apply simple age effect to a face image.
    Young ages: smoother, slightly rounder
    Older ages: add subtle texture
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    if target_age <= 5:
        # Baby/toddler: very smooth, slightly enlarged eyes area
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        # Slightly brighten
        img = cv2.convertScaleAbs(img, alpha=1.05, beta=10)
    elif target_age <= 12:
        # Child: smooth skin
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        img = cv2.convertScaleAbs(img, alpha=1.02, beta=5)
    elif target_age <= 20:
        # Teen: normal
        pass
    elif target_age <= 40:
        # Adult: slight sharpening
        kernel = np.array([[0,-0.3,0],[-0.3,2.2,-0.3],[0,-0.3,0]])
        img = cv2.filter2D(img, -1, kernel)
    else:
        # Older: add subtle texture + slight desaturation
        kernel = np.array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]])
        img = cv2.filter2D(img, -1, kernel)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= 0.85
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_path, img)
    return True


# ── HTML ───────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Child Face Predictor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0d0d0d;color:#f0f0f0;font-family:'Segoe UI',sans-serif;min-height:100vh;}
h1{text-align:center;padding:24px;font-size:1.6em;letter-spacing:2px;
   background:linear-gradient(135deg,#1a1a2e,#16213e);border-bottom:1px solid #333;}
h1 span{color:#e94560;}
.container{max-width:1100px;margin:0 auto;padding:20px;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;}
.card{background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:20px;}
.card h3{font-size:0.85em;letter-spacing:2px;color:#888;margin-bottom:14px;}
.upload-zone{border:2px dashed #444;border-radius:8px;padding:30px;text-align:center;
  cursor:pointer;transition:border-color 0.2s;}
.upload-zone:hover,.upload-zone.drag{border-color:#e94560;}
.upload-zone img{max-width:100%;max-height:200px;border-radius:6px;margin-top:10px;display:none;}
.upload-zone p{color:#666;font-size:0.85em;}
.upload-zone .icon{font-size:2.5em;margin-bottom:8px;}
input[type=file]{display:none;}
.trait-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.7em;
  margin:3px;background:#2a2a2a;border:1px solid #444;}
.slider-row{display:flex;align-items:center;gap:10px;margin:8px 0;}
.slider-row label{font-size:0.75em;color:#888;min-width:80px;}
input[type=range]{flex:1;accent-color:#e94560;}
.weight-val{font-size:0.8em;color:#e94560;min-width:35px;}
.btn{width:100%;padding:14px;border:none;border-radius:8px;font-size:1em;
  cursor:pointer;letter-spacing:1px;font-weight:600;transition:opacity 0.2s;}
.btn-primary{background:linear-gradient(135deg,#e94560,#c23152);color:#fff;}
.btn-primary:hover{opacity:0.9;}
.btn-primary:disabled{opacity:0.4;cursor:not-allowed;}
.ages{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;}
.age-btn{padding:6px 14px;border:1px solid #444;border-radius:20px;background:#222;
  color:#aaa;cursor:pointer;font-size:0.8em;transition:all 0.2s;}
.age-btn.active,.age-btn:hover{background:#e94560;border-color:#e94560;color:#fff;}
.results{display:none;}
.result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:16px;}
.result-card{background:#1a1a1a;border:1px solid #333;border-radius:10px;overflow:hidden;text-align:center;}
.result-card img{width:100%;aspect-ratio:1;object-fit:cover;}
.result-card .age-label{padding:8px;font-size:0.8em;color:#aaa;}
.progress{display:none;text-align:center;padding:30px;}
.spinner{width:50px;height:50px;border:4px solid #333;border-top-color:#e94560;
  border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 16px;}
@keyframes spin{to{transform:rotate(360deg);}}
.status-msg{color:#888;font-size:0.85em;}
.family-section{margin-top:16px;}
.add-btn{padding:8px 16px;border:1px dashed #555;border-radius:6px;background:none;
  color:#888;cursor:pointer;font-size:0.8em;width:100%;}
.add-btn:hover{border-color:#e94560;color:#e94560;}
.extra-member{background:#222;border-radius:8px;padding:12px;margin-top:8px;position:relative;}
.remove-btn{position:absolute;top:8px;right:8px;background:none;border:none;
  color:#666;cursor:pointer;font-size:1.1em;}
.remove-btn:hover{color:#e94560;}
.similarity-bar{height:6px;background:#333;border-radius:3px;margin-top:6px;}
.similarity-fill{height:100%;background:linear-gradient(90deg,#e94560,#ff8c69);border-radius:3px;}
</style>
</head>
<body>
<h1>Child Face <span>Predictor</span></h1>
<div class="container">

  <div class="grid">
    <!-- Parent 1 -->
    <div class="card">
      <h3>PARENT 1 (YOU)</h3>
      <div class="upload-zone" id="zone1" onclick="document.getElementById('file1').click()"
           ondragover="event.preventDefault();this.classList.add('drag')"
           ondragleave="this.classList.remove('drag')"
           ondrop="handleDrop(event,'file1')">
        <div class="icon">👤</div>
        <p>Click or drag photo here</p>
        <img id="preview1">
      </div>
      <input type="file" id="file1" accept="image/*" onchange="uploadPhoto(this,'1')">
      <div id="traits1" style="margin-top:10px;"></div>
      <div class="slider-row" style="margin-top:12px;">
        <label>Contribution</label>
        <input type="range" id="weight1" min="10" max="90" value="50"
               oninput="document.getElementById('wval1').textContent=this.value+'%'">
        <span class="weight-val" id="wval1">50%</span>
      </div>
    </div>

    <!-- Parent 2 -->
    <div class="card">
      <h3>PARENT 2 (PARTNER)</h3>
      <div class="upload-zone" id="zone2" onclick="document.getElementById('file2').click()"
           ondragover="event.preventDefault();this.classList.add('drag')"
           ondragleave="this.classList.remove('drag')"
           ondrop="handleDrop(event,'file2')">
        <div class="icon">👤</div>
        <p>Click or drag photo here</p>
        <img id="preview2">
      </div>
      <input type="file" id="file2" accept="image/*" onchange="uploadPhoto(this,'2')">
      <div id="traits2" style="margin-top:10px;"></div>
      <div class="slider-row" style="margin-top:12px;">
        <label>Contribution</label>
        <input type="range" id="weight2" min="10" max="90" value="50"
               oninput="document.getElementById('wval2').textContent=this.value+'%'">
        <span class="weight-val" id="wval2">50%</span>
      </div>
    </div>
  </div>

  <!-- Extra family members -->
  <div class="card family-section">
    <h3>OPTIONAL: ADD FAMILY MEMBERS (improves accuracy)</h3>
    <p style="font-size:0.75em;color:#666;margin-bottom:10px;">
      Add grandparents, siblings, aunts/uncles to refine the prediction
    </p>
    <div id="extra-members"></div>
    <button class="add-btn" onclick="addFamilyMember()">+ Add family member</button>
  </div>

  <!-- Age selection -->
  <div class="card" style="margin-top:16px;">
    <h3>PREDICT AT AGES</h3>
    <div class="ages">
      <div class="age-btn active" onclick="toggleAge(this,1)">1 yr</div>
      <div class="age-btn active" onclick="toggleAge(this,5)">5 yrs</div>
      <div class="age-btn active" onclick="toggleAge(this,10)">10 yrs</div>
      <div class="age-btn active" onclick="toggleAge(this,18)">18 yrs</div>
      <div class="age-btn" onclick="toggleAge(this,25)">25 yrs</div>
      <div class="age-btn" onclick="toggleAge(this,40)">40 yrs</div>
    </div>
    <button class="btn btn-primary" id="predict-btn" onclick="predict()" disabled>
      PREDICT CHILD FACE
    </button>
  </div>

  <!-- Progress -->
  <div class="progress" id="progress">
    <div class="spinner"></div>
    <div class="status-msg" id="status-msg">Analysing faces...</div>
  </div>

  <!-- Results -->
  <div class="results card" id="results" style="margin-top:16px;">
    <h3>PREDICTED CHILD</h3>
    <div class="result-grid" id="result-grid"></div>
  </div>

</div>

<script>
const uploads = {};  // slot -> filename
const extras  = [];  // [{slot, filename, weight, label}]
let extraCount = 0;
const selectedAges = new Set([1,5,10,18]);

function toggleAge(el, age) {
  el.classList.toggle('active');
  if (selectedAges.has(age)) selectedAges.delete(age);
  else selectedAges.add(age);
}

function handleDrop(e, inputId) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) {
    const inp = document.getElementById(inputId);
    const dt  = new DataTransfer();
    dt.items.add(file);
    inp.files = dt.files;
    inp.dispatchEvent(new Event('change'));
  }
}

function uploadPhoto(input, slot) {
  const file = input.files[0];
  if (!file) return;
  const preview = document.getElementById('preview' + slot);
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';

  const fd = new FormData();
  fd.append('photo', file);
  fd.append('slot', slot);

  fetch('/upload', {method:'POST', body:fd})
    .then(r=>r.json()).then(d=>{
      if (d.error) { alert(d.error); return; }
      uploads[slot] = d.filename;
      showTraits(slot, d.traits);
      checkReady();
    }).catch(e=>alert('Upload failed: '+e));
}

function showTraits(slot, traits) {
  const el = document.getElementById('traits' + slot);
  if (!traits) return;
  el.innerHTML = `
    <span class="trait-badge">Age ~${traits.age}</span>
    <span class="trait-badge">${traits.gender}</span>
    <span class="trait-badge">${traits.race}</span>`;
}

function checkReady() {
  const ready = uploads['1'] && uploads['2'];
  document.getElementById('predict-btn').disabled = !ready;
}

function addFamilyMember() {
  extraCount++;
  const id = 'extra' + extraCount;
  const div = document.createElement('div');
  div.className = 'extra-member';
  div.id = id;
  div.innerHTML = `
    <button class="remove-btn" onclick="removeExtra('${id}')">✕</button>
    <div style="display:flex;gap:10px;align-items:flex-start;">
      <div class="upload-zone" style="width:100px;flex-shrink:0;padding:10px;"
           onclick="document.getElementById('efile${extraCount}').click()">
        <div style="font-size:1.5em;">👤</div>
        <img id="epreview${extraCount}" style="max-width:80px;display:none;">
      </div>
      <div style="flex:1;">
        <input type="text" id="elabel${extraCount}" placeholder="e.g. Grandmother"
          style="width:100%;background:#333;border:1px solid #444;border-radius:6px;
                 padding:6px 10px;color:#fff;font-size:0.8em;margin-bottom:8px;">
        <div class="slider-row">
          <label>Weight</label>
          <input type="range" id="eweight${extraCount}" min="5" max="40" value="10"
                 oninput="document.getElementById('ewval${extraCount}').textContent=this.value+'%'">
          <span class="weight-val" id="ewval${extraCount}">10%</span>
        </div>
      </div>
    </div>
    <input type="file" id="efile${extraCount}" accept="image/*"
           onchange="uploadExtra(this,${extraCount})">`;
  document.getElementById('extra-members').appendChild(div);
}

function removeExtra(id) {
  document.getElementById(id).remove();
}

function uploadExtra(input, n) {
  const file = input.files[0];
  if (!file) return;
  const prev = document.getElementById('epreview'+n);
  prev.src = URL.createObjectURL(file);
  prev.style.display = 'block';
  const fd = new FormData();
  fd.append('photo', file);
  fd.append('slot', 'extra'+n);
  fetch('/upload', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
    if (!d.error) {
      extras.push({slot:'extra'+n, filename:d.filename,
                   labelId:'elabel'+n, weightId:'eweight'+n});
    }
  });
}

function predict() {
  if (!uploads['1'] || !uploads['2']) return;

  document.getElementById('progress').style.display = 'block';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('predict-btn').disabled   = true;

  const members = [
    {filename: uploads['1'], weight: parseInt(document.getElementById('weight1').value), label:'Parent 1'},
    {filename: uploads['2'], weight: parseInt(document.getElementById('weight2').value), label:'Parent 2'},
  ];

  extras.forEach(e => {
    const fn = uploads[e.slot] || e.filename;
    if (!fn) return;
    members.push({
      filename: fn,
      weight:   parseInt(document.getElementById(e.weightId).value),
      label:    document.getElementById(e.labelId).value || 'Family'
    });
  });

  const ages = [...selectedAges].sort((a,b)=>a-b);

  setStatus('Extracting facial features...');
  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({members, ages})
  }).then(r=>r.json()).then(d=>{
    document.getElementById('progress').style.display = 'none';
    document.getElementById('predict-btn').disabled   = false;
    if (d.error) { alert(d.error); return; }
    showResults(d.results);
  }).catch(e=>{
    document.getElementById('progress').style.display = 'none';
    document.getElementById('predict-btn').disabled   = false;
    alert('Prediction failed: '+e);
  });
}

function setStatus(msg) {
  document.getElementById('status-msg').textContent = msg;
}

function showResults(results) {
  const grid = document.getElementById('result-grid');
  grid.innerHTML = '';
  results.forEach(r => {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <img src="/output/${r.filename}?t=${Date.now()}" alt="Age ${r.age}">
      <div class="age-label">Age ${r.age}</div>`;
    grid.appendChild(card);
  });
  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['photo']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path     = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    traits = analyze_face_traits(path)
    return jsonify({"filename": filename, "traits": traits})


@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    members = data.get('members', [])
    ages    = data.get('ages', [5, 10, 18])

    if len(members) < 2:
        return jsonify({"error": "Need at least 2 photos"}), 400

    # Extract embeddings
    pairs = []
    parent_paths = []
    for m in members:
        path = os.path.join(UPLOAD_DIR, m['filename'])
        emb, err = extract_embedding(path)
        if emb is None:
            return jsonify({"error": f"No face detected: {err}"}), 400
        pairs.append((emb, m['weight']))
        parent_paths.append(path)

    # Blend embeddings
    blended = blend_embeddings(pairs)

    # Generate base child face
    parent_weights = [m['weight'] for m in members]
    base_output = os.path.join(OUTPUT_DIR, f"child_base_{uuid.uuid4().hex}.jpg")
    ok, msg = generate_child_face(blended, parent_paths, base_output,
                                  weights=parent_weights)
    if not ok:
        return jsonify({"error": f"Generation failed: {msg}"}), 500

    # Apply age effects
    results = []
    for age in ages:
        age_filename = f"child_age{age}_{uuid.uuid4().hex}.jpg"
        age_path     = os.path.join(OUTPUT_DIR, age_filename)
        apply_age_effect(base_output, age_path, age)
        results.append({"age": age, "filename": age_filename})

    return jsonify({"results": results})


@app.route('/output/<filename>')
def serve_output(filename):
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return '', 404


if __name__ == '__main__':
    import webbrowser, threading, time
    print("=" * 50)
    print("  CHILD FACE PREDICTOR")
    print("=" * 50)
    print("  Loading models...")

    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5001")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=False)
